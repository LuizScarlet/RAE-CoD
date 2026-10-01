"""Parse schema-constrained VLM judgments."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .schemas import (
    CandidateUnitAssessment,
    ReferenceUnitInventory,
    SemanticConsistencyAssessment,
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
ModelT = TypeVar("ModelT", bound=BaseModel)


class JudgeParseError(ValueError):
    """Raised when a model response contains no valid protocol judgment."""


def _candidate_objects(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    stripped = text.strip()
    try:
        direct = json.loads(stripped)
        if isinstance(direct, dict):
            candidates.append(direct)
    except json.JSONDecodeError:
        pass
    for fenced in _CODE_FENCE.findall(stripped):
        try:
            value = json.loads(fenced.strip())
            if isinstance(value, dict):
                candidates.append(value)
        except json.JSONDecodeError:
            pass
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in candidates:
        signature = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if signature not in seen:
            seen.add(signature)
            unique.append(value)
    return unique


def _parse_response(raw_output: str, schema: type[ModelT], label: str) -> ModelT:
    if not raw_output or not raw_output.strip():
        raise JudgeParseError(f"Model returned an empty {label} response")
    cleaned = _THINK_BLOCK.sub("", raw_output).strip()
    candidates = _candidate_objects(cleaned)
    if not candidates:
        raise JudgeParseError(f"No valid JSON object was found in the {label} response")
    failures: list[str] = []
    for candidate in candidates:
        try:
            return schema.model_validate(candidate)
        except ValidationError as exc:
            failures.append(str(exc).replace("\n", " ")[:700])
    joined = " | ".join(failures[:3])
    raise JudgeParseError(f"JSON object(s) failed the {label} schema: {joined}")


def parse_candidate(raw_output: str) -> CandidateUnitAssessment:
    return _parse_response(raw_output, CandidateUnitAssessment, "candidate-unit judgment")


def parse_reference(raw_output: str) -> ReferenceUnitInventory:
    return _parse_response(raw_output, ReferenceUnitInventory, "reference inventory")


def parse_consistency(raw_output: str) -> SemanticConsistencyAssessment:
    return _parse_response(raw_output, SemanticConsistencyAssessment, "consistency judgment")


__all__ = [
    "JudgeParseError",
    "parse_candidate",
    "parse_reference",
    "parse_consistency",
]
