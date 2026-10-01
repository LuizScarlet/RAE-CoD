"""Validated records for the SR/SQ/SC semantic-unit protocol."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PairInput(BaseModel):
    """One reference/reconstruction pair loaded from a manifest."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    reconstruction: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageInfo(BaseModel):
    """Decoded-image metadata stored with each result."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mode: str


SemanticUnitType = Literal[
    "global_scene",
    "primary_entity",
    "secondary_entity",
    "attribute",
    "action_state",
    "relation_layout",
    "text_symbol",
    "context",
]
ImportanceLabel = Literal["critical", "major", "moderate", "supporting", "minor"]
MatchType = Literal[
    "exact",
    "equivalent",
    "partial",
    "weak",
    "absent",
    "unrelated",
    "contradicted",
]


def _clean_strings(values: Any, limit: int) -> Any:
    if not isinstance(values, list):
        return values
    cleaned: list[Any] = []
    for value in values:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        cleaned.append(value)
    return cleaned[:limit]


class UnitQualityDimensions(BaseModel):
    """Intrinsic visual quality of one semantic unit, independent of the GT."""

    model_config = ConfigDict(extra="forbid")

    semantic_clarity: int = Field(ge=0, le=10)
    structural_integrity: int = Field(ge=0, le=10)
    appearance_naturalness: int = Field(ge=0, le=10)
    artifact_non_dominance: int = Field(ge=0, le=10)


class CandidateSemanticUnit(BaseModel):
    """One nonredundant semantic hypothesis extracted from the reconstruction alone."""

    model_config = ConfigDict(extra="forbid")

    unit_id: str = Field(pattern=r"^C[1-9][0-9]*$")
    unit_type: SemanticUnitType
    description: str = Field(min_length=1, max_length=500)
    visible_evidence: str = Field(min_length=1, max_length=800)
    importance: int = Field(ge=1, le=5)
    importance_label: ImportanceLabel
    identity_confidence: int = Field(ge=0, le=100)
    quality_dimensions: UnitQualityDimensions
    quality_defects: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("quality_defects", mode="before")
    @classmethod
    def clean_quality_defects(cls, values: Any) -> Any:
        return _clean_strings(values, 6)

    @model_validator(mode="after")
    def importance_label_matches_value(self) -> "CandidateSemanticUnit":
        expected = {5: "critical", 4: "major", 3: "moderate", 2: "supporting", 1: "minor"}
        if self.importance_label != expected[self.importance]:
            raise ValueError(
                f"importance={self.importance} requires label={expected[self.importance]}"
            )
        return self


class CandidateUnitAssessment(BaseModel):
    """Detailed GT-blind reconstruction description and semantic-unit assessment."""

    model_config = ConfigDict(extra="forbid")

    detailed_description: str = Field(min_length=1, max_length=4000)
    global_interpretation: str = Field(min_length=1, max_length=1000)
    semantic_units: list[CandidateSemanticUnit] = Field(default_factory=list, max_length=12)
    uncertainties: list[str] = Field(default_factory=list, max_length=10)
    overall_quality_observations: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("uncertainties", "overall_quality_observations", mode="before")
    @classmethod
    def clean_evidence(cls, values: Any) -> Any:
        return _clean_strings(values, 10)

    @model_validator(mode="after")
    def unique_ordered_unit_ids(self) -> "CandidateUnitAssessment":
        ids = [unit.unit_id for unit in self.semantic_units]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate semantic unit IDs must be unique")
        expected = [f"C{index}" for index in range(1, len(ids) + 1)]
        if ids != expected:
            raise ValueError(f"candidate unit IDs must be consecutive in order: {expected}")
        return self


class ReferenceSemanticUnit(BaseModel):
    """One candidate-independent semantic unit extracted from the GT alone."""

    model_config = ConfigDict(extra="forbid")

    unit_id: str = Field(pattern=r"^R[1-9][0-9]*$")
    unit_type: SemanticUnitType
    description: str = Field(min_length=1, max_length=500)
    visible_evidence: str = Field(min_length=1, max_length=800)
    importance: int = Field(ge=1, le=5)
    importance_label: ImportanceLabel

    @model_validator(mode="after")
    def importance_label_matches_value(self) -> "ReferenceSemanticUnit":
        expected = {5: "critical", 4: "major", 3: "moderate", 2: "supporting", 1: "minor"}
        if self.importance_label != expected[self.importance]:
            raise ValueError(
                f"importance={self.importance} requires label={expected[self.importance]}"
            )
        return self


class ReferenceUnitInventory(BaseModel):
    """Frozen GT description reused for every reconstruction of that source."""

    model_config = ConfigDict(extra="forbid")

    detailed_description: str = Field(min_length=1, max_length=4000)
    global_interpretation: str = Field(min_length=1, max_length=1000)
    semantic_units: list[ReferenceSemanticUnit] = Field(default_factory=list, max_length=12)
    ambiguities: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("ambiguities", mode="before")
    @classmethod
    def clean_ambiguities(cls, values: Any) -> Any:
        return _clean_strings(values, 8)

    @model_validator(mode="after")
    def unique_ordered_unit_ids(self) -> "ReferenceUnitInventory":
        ids = [unit.unit_id for unit in self.semantic_units]
        if len(ids) != len(set(ids)):
            raise ValueError("reference semantic unit IDs must be unique")
        expected = [f"R{index}" for index in range(1, len(ids) + 1)]
        if ids != expected:
            raise ValueError(f"reference unit IDs must be consecutive in order: {expected}")
        return self


class DirectionalUnitMatch(BaseModel):
    """Similarity from one source-side unit to zero or more target-side units."""

    model_config = ConfigDict(extra="forbid")

    source_unit_id: str = Field(min_length=2, max_length=8)
    matched_target_unit_ids: list[str] = Field(default_factory=list, max_length=4)
    match_type: MatchType
    semantic_similarity: int = Field(ge=0, le=100)
    rationale: str = Field(min_length=1, max_length=800)

    @field_validator("matched_target_unit_ids", mode="before")
    @classmethod
    def clean_target_ids(cls, values: Any) -> Any:
        return _clean_strings(values, 4)

    @model_validator(mode="after")
    def validate_target_ids(self) -> "DirectionalUnitMatch":
        if len(self.matched_target_unit_ids) != len(set(self.matched_target_unit_ids)):
            raise ValueError("matched target unit IDs must be unique")
        return self


class SemanticConsistencyAssessment(BaseModel):
    """Text-only comparison between frozen GT and blind reconstruction inventories."""

    model_config = ConfigDict(extra="forbid")

    global_similarity: int = Field(ge=0, le=100)
    global_rationale: str = Field(min_length=1, max_length=1200)
    reference_to_candidate: list[DirectionalUnitMatch] = Field(default_factory=list, max_length=12)
    candidate_to_reference: list[DirectionalUnitMatch] = Field(default_factory=list, max_length=12)
    missing_reference_units: list[str] = Field(default_factory=list, max_length=12)
    unsupported_candidate_units: list[str] = Field(default_factory=list, max_length=12)
    comparison_summary: str = Field(min_length=1, max_length=1600)

    @field_validator("missing_reference_units", "unsupported_candidate_units", mode="before")
    @classmethod
    def clean_unit_lists(cls, values: Any) -> Any:
        return _clean_strings(values, 12)


class CandidateUnitScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_id: str
    importance: int = Field(ge=1, le=5)
    identity_confidence: float = Field(ge=0, le=100)
    intrinsic_quality: float = Field(ge=0, le=100)
    sr_contribution: float = Field(ge=0, le=100)
    sq_contribution: float = Field(ge=0, le=100)


class SemanticScores(BaseModel):
    """Independent scores; no progressive ordering constraint is imposed."""

    model_config = ConfigDict(extra="forbid")

    SR: float = Field(ge=0, le=100)
    SQ: float = Field(ge=0, le=100)
    SC: float = Field(ge=0, le=100)
    SC_global: float = Field(ge=0, le=100)
    SC_unit_recall: float = Field(ge=0, le=100)
    SC_unit_precision: float = Field(ge=0, le=100)
    SC_unit_f1: float = Field(ge=0, le=100)
    candidate_unit_count: int = Field(ge=0)
    reference_unit_count: int = Field(ge=0)
    candidate_total_importance: int = Field(ge=0)
    reference_total_importance: int = Field(ge=0)
    unmatched_nonzero_similarity_count: int = Field(ge=0)
    candidate_units: list[CandidateUnitScore] = Field(default_factory=list, max_length=12)


class ReferenceAssessmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgment: ReferenceUnitInventory
    raw_output: str | None = None
    parse_attempts: int = Field(default=1, ge=1)
    latency_seconds: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_key: str


class CandidateAssessmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgment: CandidateUnitAssessment
    raw_output: str | None = None
    parse_attempts: int = Field(default=1, ge=1)
    latency_seconds: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ConsistencyAssessmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgment: SemanticConsistencyAssessment
    raw_output: str | None = None
    parse_attempts: int = Field(default=1, ge=1)
    latency_seconds: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str
    model_id: str = "Qwen/Qwen3.5-9B"
    revision: str | None = None
    backend: str = "vllm"
    backend_version: str | None = None
    generation_mode: Literal["offline_batch"] = "offline_batch"
    structured_output: bool = True
    configured_batch_size: int = Field(ge=1)
    transformers_version: str | None = None
    torch_version: str | None = None
    prompt_sha256: str
    dtype: str
    min_pixels: int = Field(gt=0)
    max_pixels: int = Field(gt=0)
    reference_max_tokens: int = Field(ge=128)
    candidate_max_tokens: int = Field(ge=128)
    consistency_max_tokens: int = Field(ge=128)
    logical_calls_per_pair: Literal[3] = 3
    reference_inventory_cached: Literal[True] = True


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    protocol: str
    status: Literal["ok"] = "ok"
    sample_id: str
    reference: ImageInfo
    reconstruction: ImageInfo
    metadata: dict[str, Any] = Field(default_factory=dict)
    model: ModelInfo
    reference_assessment: ReferenceAssessmentResult
    candidate_assessment: CandidateAssessmentResult
    consistency_assessment: ConsistencyAssessmentResult
    semantic_scores: SemanticScores
    created_at: str


class ErrorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    protocol: str
    status: Literal["error"] = "error"
    sample_id: str
    reference_path: str
    reconstruction_path: str
    error_type: str
    error_message: str
    created_at: str
