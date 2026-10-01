from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import ValidationError

from .schemas import PairInput

_REFERENCE_KEYS = ("reference", "gt", "ground_truth", "ground_truth_path", "reference_path")
_RECONSTRUCTION_KEYS = ("reconstruction", "recon", "candidate", "reconstruction_path")
_ID_KEYS = ("id", "sample_id", "name")


class ManifestError(ValueError):
    pass


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _resolve_image_path(value: Any, base_dir: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"Expected a nonempty image path, got {value!r}")
    expanded = Path(os.path.expandvars(value)).expanduser()
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return str(expanded.resolve())


def _normalize_record(record: dict[str, Any], base_dir: Path, line: int) -> PairInput:
    reference = _first(record, _REFERENCE_KEYS)
    reconstruction = _first(record, _RECONSTRUCTION_KEYS)
    sample_id = _first(record, _ID_KEYS)
    if sample_id is None:
        if reconstruction:
            sample_id = Path(str(reconstruction)).stem
        else:
            sample_id = f"sample-{line:06d}"

    metadata = record.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata) if metadata.strip() else {}
        except json.JSONDecodeError:
            metadata = {"value": metadata}
    if not isinstance(metadata, dict):
        metadata = {"value": metadata}

    recognized = set(_REFERENCE_KEYS + _RECONSTRUCTION_KEYS + _ID_KEYS + ("metadata",))
    extras = {key: value for key, value in record.items() if key not in recognized}
    metadata = {**extras, **metadata}
    try:
        return PairInput(
            id=str(sample_id),
            reference=_resolve_image_path(reference, base_dir),
            reconstruction=_resolve_image_path(reconstruction, base_dir),
            metadata=metadata,
        )
    except (ValidationError, ManifestError) as exc:
        raise ManifestError(f"Invalid manifest record {line}: {exc}") from exc


def _raw_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(f"Invalid JSON on line {line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ManifestError(f"Manifest line {line_number} is not an object")
                yield line_number, record
    elif suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Invalid JSON manifest: {exc}") from exc
        if isinstance(data, dict) and isinstance(data.get("pairs"), list):
            data = data["pairs"]
        if not isinstance(data, list):
            raise ManifestError("A .json manifest must be a list or an object with a pairs list")
        for index, record in enumerate(data, start=1):
            if not isinstance(record, dict):
                raise ManifestError(f"Manifest item {index} is not an object")
            yield index, record
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, record in enumerate(csv.DictReader(handle), start=2):
                yield line_number, dict(record)
    else:
        raise ManifestError("Manifest extension must be .jsonl, .ndjson, .json, or .csv")


def load_manifest(path_like: str | Path, *, check_files: bool = False) -> list[PairInput]:
    path = Path(path_like).expanduser().resolve()
    if not path.is_file():
        raise ManifestError(f"Manifest does not exist: {path}")
    pairs: list[PairInput] = []
    seen: set[str] = set()
    for line_number, raw in _raw_records(path):
        pair = _normalize_record(raw, path.parent, line_number)
        if pair.id in seen:
            raise ManifestError(f"Duplicate sample id {pair.id!r} at record {line_number}")
        seen.add(pair.id)
        if check_files:
            for label, image_path in (
                ("reference", pair.reference),
                ("reconstruction", pair.reconstruction),
            ):
                if not Path(image_path).is_file():
                    raise ManifestError(
                        f"Missing {label} image for sample {pair.id!r}: {image_path}"
                    )
        pairs.append(pair)
    if not pairs:
        raise ManifestError("Manifest contains no pairs")
    return pairs


def read_jsonl(path_like: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ManifestError(f"Expected object in {path} line {line_number}")
            yield value


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def select_shard(
    pairs: Iterable[PairInput], *, shard_index: int, num_shards: int
) -> list[PairInput]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ManifestError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    return [pair for index, pair in enumerate(pairs) if index % num_shards == shard_index]


def merge_result_files(inputs: list[Path], output: Path) -> int:
    if not inputs:
        raise ManifestError("At least one input result file is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    seen: set[str] = set()
    count = 0
    with temporary.open("w", encoding="utf-8") as destination:
        for source in inputs:
            for record in read_jsonl(source):
                sample_id = record.get("sample_id")
                if not isinstance(sample_id, str):
                    raise ManifestError(f"Record in {source} has no string sample_id")
                if sample_id in seen:
                    raise ManifestError(f"Duplicate result id during merge: {sample_id!r}")
                seen.add(sample_id)
                append_record(destination, record)
                count += 1
    temporary.replace(output)
    return count
