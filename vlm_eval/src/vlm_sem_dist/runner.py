#!/usr/bin/env python3
"""Evaluate image pairs with the blinded SR/SQ/SC protocol."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from vlm_sem_dist import PROTOCOL_ID
from vlm_sem_dist.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_PATH, DEFAULT_MODEL_REVISION
from vlm_sem_dist.images import load_image
from vlm_sem_dist.manifest import load_manifest
from vlm_sem_dist.parsing import (
    parse_candidate,
    parse_consistency,
    parse_reference,
)
from vlm_sem_dist.rubric import (
    PROMPT_SHA256,
    build_candidate_messages,
    build_consistency_messages,
    build_reference_messages,
    candidate_consistency_view,
    reference_consistency_view,
)
from vlm_sem_dist.schemas import (
    CandidateAssessmentResult,
    CandidateUnitAssessment,
    ConsistencyAssessmentResult,
    ErrorRecord,
    EvaluationRecord,
    ModelInfo,
    ReferenceAssessmentResult,
    ReferenceUnitInventory,
    SemanticConsistencyAssessment,
)
from vlm_sem_dist.scoring import derive_semantic_scores, score_consistency

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    """Replace a checkpoint/output atomically so interruption cannot leave truncated JSON."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


@dataclass
class LoadedSample:
    pair: Any
    reference_key: str
    reference_info: Any
    reconstruction_info: Any
    reconstruction_image: Any

    def close(self) -> None:
        self.reconstruction_image.close()


@dataclass
class ParsedGeneration:
    judgment: Any
    raw_output: str
    attempts: int
    latency_seconds: float
    input_tokens: int
    output_tokens: int


class BatchRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
        except ImportError as exc:
            raise RuntimeError(
                "Use the isolated vLLM environment from scripts/setup_env.sh"
            ) from exc

        self.args = args
        self.torch_version = torch.__version__
        self.transformers_version = transformers.__version__
        self.vllm_version = distribution_version("vllm")
        self.SamplingParams = SamplingParams
        self.StructuredOutputsParams = StructuredOutputsParams
        self.stage_seconds = {"reference": 0.0, "candidate": 0.0, "consistency": 0.0}
        self.retry_count = {"reference": 0, "candidate": 0, "consistency": 0}

        started = time.perf_counter()
        self.llm = LLM(
            model=args.model,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            max_num_seqs=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1, "video": 0, "audio": 0},
            mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
            seed=0,
            enforce_eager=args.enforce_eager,
            disable_log_stats=True,
            structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True},
        )
        self.engine_init_seconds = time.perf_counter() - started
        self.processor = AutoProcessor.from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

    def _sampling(self, schema: type[BaseModel] | dict[str, Any], max_tokens: int):
        json_schema = schema if isinstance(schema, dict) else schema.model_json_schema()
        return self.SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            structured_outputs=self.StructuredOutputsParams(json=json_schema),
        )

    def _model_input(self, messages: list[dict[str, Any]], images: list[Any]):
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        value: dict[str, Any] = {"prompt": prompt}
        if images:
            value["multi_modal_data"] = {"image": images[0] if len(images) == 1 else images}
        return value

    @staticmethod
    def _tokens(output: Any) -> tuple[int, int]:
        return len(output.prompt_token_ids or []), len(output.outputs[0].token_ids or [])

    def generate_stage(
        self,
        *,
        stage: str,
        builders: list[Callable[[str | None], tuple[list[dict[str, Any]], list[Any]]]],
        schema: type[SchemaT],
        parser: Callable[[str], SchemaT],
        max_tokens: int,
        contextual_validator: Callable[[int, SchemaT], None] | None = None,
        retry_json_schema: Callable[[int], dict[str, Any]] | None = None,
    ) -> list[ParsedGeneration | Exception]:
        if not builders:
            return []
        inputs = [self._model_input(*builder(None)) for builder in builders]
        started = time.perf_counter()
        outputs = self.llm.generate(inputs, self._sampling(schema, max_tokens), use_tqdm=False)
        elapsed = time.perf_counter() - started
        self.stage_seconds[stage] += elapsed
        amortized = elapsed / len(outputs) if outputs else 0.0
        results: list[ParsedGeneration | Exception] = []
        for index, (builder, output) in enumerate(zip(builders, outputs)):
            raw = output.outputs[0].text
            inp, out = self._tokens(output)
            try:
                judgment = parser(raw)
                if contextual_validator:
                    contextual_validator(index, judgment)
                results.append(ParsedGeneration(judgment, raw, 1, amortized, inp, out))
                continue
            except Exception as first_error:
                error: Exception = first_error

            # Retry only the failed item, explicitly naming the schema/context failure.
            self.retry_count[stage] += 1
            messages, images = builder(str(error))
            retry_started = time.perf_counter()
            retry_output = self.llm.generate(
                [self._model_input(messages, images)],
                self._sampling(
                    retry_json_schema(index) if retry_json_schema else schema,
                    max(max_tokens, self.args.retry_max_tokens),
                ),
                use_tqdm=False,
            )[0]
            retry_elapsed = time.perf_counter() - retry_started
            self.stage_seconds[stage] += retry_elapsed
            retry_raw = retry_output.outputs[0].text
            retry_inp, retry_out = self._tokens(retry_output)
            try:
                judgment = parser(retry_raw)
                if contextual_validator:
                    contextual_validator(index, judgment)
                results.append(
                    ParsedGeneration(
                        judgment,
                        retry_raw,
                        2,
                        amortized + retry_elapsed,
                        retry_inp,
                        out + retry_out,
                    )
                )
            except Exception as retry_error:
                results.append(
                    ValueError(
                        f"{stage} failed after retry: {type(retry_error).__name__}: {retry_error}"
                    )
                )
        return results

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_path=self.args.model,
            model_id=self.args.model_id,
            revision=self.args.model_revision,
            backend="vllm",
            backend_version=self.vllm_version,
            configured_batch_size=self.args.batch_size,
            transformers_version=self.transformers_version,
            torch_version=self.torch_version,
            prompt_sha256=PROMPT_SHA256,
            dtype=self.args.dtype,
            min_pixels=self.args.min_pixels,
            max_pixels=self.args.max_pixels,
            reference_max_tokens=self.args.reference_max_tokens,
            candidate_max_tokens=self.args.candidate_max_tokens,
            consistency_max_tokens=self.args.consistency_max_tokens,
        )


def consistency_retry_schema(
    reference: ReferenceUnitInventory, candidate: CandidateUnitAssessment
) -> dict[str, Any]:
    """Constrain retry IDs to the supplied inventories without changing the prompt."""
    schema = SemanticConsistencyAssessment.model_json_schema()
    match_schema = schema["$defs"]["DirectionalUnitMatch"]

    def direction(source_ids: list[str], target_ids: list[str]) -> dict[str, Any]:
        item = copy.deepcopy(match_schema)
        if source_ids:
            item["properties"]["source_unit_id"] = {
                "type": "string",
                "enum": source_ids,
            }
        targets = item["properties"]["matched_target_unit_ids"]
        if target_ids:
            targets["items"] = {"type": "string", "enum": target_ids}
        else:
            targets["maxItems"] = 0
        return {
            "type": "array",
            "minItems": len(source_ids),
            "maxItems": len(source_ids),
            "items": item,
        }

    reference_ids = [unit.unit_id for unit in reference.semantic_units]
    candidate_ids = [unit.unit_id for unit in candidate.semantic_units]
    schema["properties"]["reference_to_candidate"] = direction(reference_ids, candidate_ids)
    schema["properties"]["candidate_to_reference"] = direction(candidate_ids, reference_ids)
    return schema


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--reference-max-tokens", type=int, default=2400)
    parser.add_argument("--candidate-max-tokens", type=int, default=3000)
    parser.add_argument("--consistency-max-tokens", type=int, default=3600)
    parser.add_argument("--retry-max-tokens", type=int, default=4000)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--reference-cache",
        type=Path,
        help="Optional JSON cache. Existing compatible inventories are reused; missing ones are added.",
    )
    parser.add_argument("--no-raw-output", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.max_model_len <= args.consistency_max_tokens + 1000:
        raise SystemExit("--max-model-len is too small for consistency output plus inventory input")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0,1]")
    pairs = load_manifest(args.manifest, check_files=True)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise SystemExit("manifest contains no pairs")

    wall_started = time.perf_counter()
    runner = BatchRunner(args)
    reference_images: dict[str, Any] = {}
    reference_infos: dict[str, Any] = {}
    loaded: list[LoadedSample] = []
    early_errors: dict[str, Exception] = {}
    try:
        for pair in pairs:
            try:
                ref_image, ref_info = load_image(pair.reference)
                key = ref_info.sha256
                if key not in reference_images:
                    reference_images[key] = ref_image
                    reference_infos[key] = ref_info
                else:
                    ref_image.close()
                reconstruction_image, reconstruction_info = load_image(pair.reconstruction)
                loaded.append(
                    LoadedSample(
                        pair=pair,
                        reference_key=key,
                        reference_info=ref_info,
                        reconstruction_info=reconstruction_info,
                        reconstruction_image=reconstruction_image,
                    )
                )
            except Exception as exc:
                early_errors[pair.id] = exc

        reference_keys = list(reference_images)
        references: dict[str, ParsedGeneration] = {}
        reference_failures: dict[str, Exception] = {}
        cached_reference_count = 0
        if args.reference_cache and args.reference_cache.is_file():
            cached = json.loads(args.reference_cache.read_text(encoding="utf-8"))
            if cached.get("protocol") != PROTOCOL_ID:
                raise ValueError("reference cache protocol does not match this evaluator")
            if cached.get("prompt_sha256") != PROMPT_SHA256:
                raise ValueError("reference cache prompt_sha256 does not match current prompts")
            if cached.get("model_revision") != args.model_revision:
                raise ValueError("reference cache model_revision does not match this run")
            for key, value in cached.get("references", {}).items():
                references[key] = ParsedGeneration(
                    judgment=ReferenceUnitInventory.model_validate(value["judgment"]),
                    raw_output=value.get("raw_output", ""),
                    attempts=int(value.get("attempts", 1)),
                    latency_seconds=float(value.get("latency_seconds", 0.0)),
                    input_tokens=int(value.get("input_tokens", 0)),
                    output_tokens=int(value.get("output_tokens", 0)),
                )
            cached_reference_count = sum(key in references for key in reference_keys)

        missing_reference_keys = [key for key in reference_keys if key not in references]
        if missing_reference_keys:
            reference_builders = [
                (
                    lambda note, key=key: build_reference_messages(
                        reference_images[key], repair_note=note
                    )
                )
                for key in missing_reference_keys
            ]
            reference_generated = runner.generate_stage(
                stage="reference",
                builders=reference_builders,
                schema=ReferenceUnitInventory,
                parser=parse_reference,
                max_tokens=args.reference_max_tokens,
            )
            for key, generated in zip(missing_reference_keys, reference_generated):
                if isinstance(generated, Exception):
                    reference_failures[key] = generated
                else:
                    references[key] = generated

        if args.reference_cache:
            args.reference_cache.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                args.reference_cache,
                json.dumps(
                    {
                        "protocol": PROTOCOL_ID,
                        "prompt_sha256": PROMPT_SHA256,
                        "model_id": args.model_id,
                        "model_revision": args.model_revision,
                        "references": {
                            key: {
                                "judgment": value.judgment.model_dump(mode="json"),
                                "raw_output": value.raw_output,
                                "attempts": value.attempts,
                                "latency_seconds": value.latency_seconds,
                                "input_tokens": value.input_tokens,
                                "output_tokens": value.output_tokens,
                            }
                            for key, value in references.items()
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
            )

        candidate_items = [item for item in loaded if item.reference_key in references]
        candidate_builders = [
            (
                lambda note, item=item: build_candidate_messages(
                    item.reconstruction_image, repair_note=note
                )
            )
            for item in candidate_items
        ]
        candidate_generated = runner.generate_stage(
            stage="candidate",
            builders=candidate_builders,
            schema=CandidateUnitAssessment,
            parser=parse_candidate,
            max_tokens=args.candidate_max_tokens,
        )
        candidates: dict[str, ParsedGeneration] = {}
        failures: dict[str, Exception] = {}
        for item, generated in zip(candidate_items, candidate_generated):
            if isinstance(generated, Exception):
                failures[item.pair.id] = generated
            else:
                candidates[item.pair.id] = generated

        consistency_items = [item for item in candidate_items if item.pair.id in candidates]
        consistency_builders = []
        for item in consistency_items:
            reference_view = reference_consistency_view(
                references[item.reference_key].judgment.model_dump(mode="json")
            )
            candidate_view = candidate_consistency_view(
                candidates[item.pair.id].judgment.model_dump(mode="json")
            )
            consistency_builders.append(
                lambda note, rv=reference_view, cv=candidate_view: build_consistency_messages(
                    rv, cv, repair_note=note
                )
            )

        def validate_consistency(index: int, judgment: SemanticConsistencyAssessment) -> None:
            item = consistency_items[index]
            score_consistency(
                candidates[item.pair.id].judgment,
                references[item.reference_key].judgment,
                judgment,
            )

        consistency_retry_schemas = [
            consistency_retry_schema(
                references[item.reference_key].judgment,
                candidates[item.pair.id].judgment,
            )
            for item in consistency_items
        ]
        consistency_generated = runner.generate_stage(
            stage="consistency",
            builders=consistency_builders,
            schema=SemanticConsistencyAssessment,
            parser=parse_consistency,
            max_tokens=args.consistency_max_tokens,
            contextual_validator=validate_consistency,
            retry_json_schema=lambda index: consistency_retry_schemas[index],
        )
        consistencies: dict[str, ParsedGeneration] = {}
        for item, generated in zip(consistency_items, consistency_generated):
            if isinstance(generated, Exception):
                failures[item.pair.id] = generated
            else:
                consistencies[item.pair.id] = generated

        info = runner.model_info()
        records: list[dict[str, Any]] = []
        loaded_by_id = {item.pair.id: item for item in loaded}
        for pair in pairs:
            if pair.id in early_errors:
                exc = early_errors[pair.id]
            elif pair.id not in loaded_by_id:
                exc = ValueError("sample did not load")
            else:
                item = loaded_by_id[pair.id]
                if item.reference_key in reference_failures:
                    exc = reference_failures[item.reference_key]
                elif pair.id in failures:
                    exc = failures[pair.id]
                elif pair.id not in consistencies:
                    exc = ValueError("sample has no complete consistency judgment")
                else:
                    reference_result = references[item.reference_key]
                    candidate_result = candidates[pair.id]
                    consistency_result = consistencies[pair.id]
                    record = EvaluationRecord(
                        protocol=PROTOCOL_ID,
                        sample_id=pair.id,
                        reference=item.reference_info.model_dump(),
                        reconstruction=item.reconstruction_info.model_dump(),
                        metadata=pair.metadata,
                        model=info,
                        reference_assessment=ReferenceAssessmentResult(
                            judgment=reference_result.judgment,
                            raw_output=(
                                None if args.no_raw_output else reference_result.raw_output
                            ),
                            parse_attempts=reference_result.attempts,
                            latency_seconds=round(reference_result.latency_seconds, 4),
                            input_tokens=reference_result.input_tokens,
                            output_tokens=reference_result.output_tokens,
                            cache_key=item.reference_key,
                        ),
                        candidate_assessment=CandidateAssessmentResult(
                            judgment=candidate_result.judgment,
                            raw_output=(
                                None if args.no_raw_output else candidate_result.raw_output
                            ),
                            parse_attempts=candidate_result.attempts,
                            latency_seconds=round(candidate_result.latency_seconds, 4),
                            input_tokens=candidate_result.input_tokens,
                            output_tokens=candidate_result.output_tokens,
                        ),
                        consistency_assessment=ConsistencyAssessmentResult(
                            judgment=consistency_result.judgment,
                            raw_output=(
                                None if args.no_raw_output else consistency_result.raw_output
                            ),
                            parse_attempts=consistency_result.attempts,
                            latency_seconds=round(consistency_result.latency_seconds, 4),
                            input_tokens=consistency_result.input_tokens,
                            output_tokens=consistency_result.output_tokens,
                        ),
                        semantic_scores=derive_semantic_scores(
                            candidate_result.judgment,
                            reference_result.judgment,
                            consistency_result.judgment,
                        ),
                        created_at=utc_now(),
                    )
                    records.append(record.model_dump(mode="json"))
                    continue
            error = ErrorRecord(
                protocol=PROTOCOL_ID,
                sample_id=pair.id,
                reference_path=pair.reference,
                reconstruction_path=pair.reconstruction,
                error_type=type(exc).__name__,
                error_message=str(exc),
                created_at=utc_now(),
            )
            records.append(error.model_dump(mode="json"))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            args.output,
            "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in records
            ),
        )
        wall_seconds = time.perf_counter() - wall_started
        sidecar = args.output.with_suffix(args.output.suffix + ".run.json")
        atomic_write_text(
            sidecar,
            json.dumps(
                {
                    "protocol": PROTOCOL_ID,
                    "prompt_sha256": PROMPT_SHA256,
                    "model": args.model,
                    "vllm_version": runner.vllm_version,
                    "python": platform.python_version(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "pairs": len(pairs),
                    "successful": sum(record["status"] == "ok" for record in records),
                    "errors": sum(record["status"] == "error" for record in records),
                    "unique_references": len(reference_keys),
                    "cached_references_used": cached_reference_count,
                    "references_generated": len(missing_reference_keys),
                    "reference_cache": str(args.reference_cache) if args.reference_cache else None,
                    "engine_init_seconds": runner.engine_init_seconds,
                    "stage_generation_seconds": runner.stage_seconds,
                    "stage_retries": runner.retry_count,
                    "wall_seconds": wall_seconds,
                    "settings": vars(args)
                    | {"manifest": str(args.manifest), "output": str(args.output)},
                },
                indent=2,
                default=str,
            )
            + "\n",
        )
        print(
            f"wrote={len(records)} ok={sum(r['status'] == 'ok' for r in records)} "
            f"errors={sum(r['status'] == 'error' for r in records)} wall={wall_seconds:.2f}s "
            f"output={args.output}",
            file=sys.stderr,
        )
        return 0 if all(record["status"] == "ok" for record in records) else 1
    finally:
        for item in loaded:
            item.close()
        for image in reference_images.values():
            image.close()


if __name__ == "__main__":
    raise SystemExit(main())
