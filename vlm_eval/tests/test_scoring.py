from __future__ import annotations

import json

import pytest

from vlm_sem_dist.parsing import parse_candidate
from vlm_sem_dist.rubric import candidate_consistency_view
from vlm_sem_dist.schemas import (
    CandidateUnitAssessment,
    ReferenceUnitInventory,
    SemanticConsistencyAssessment,
)
from vlm_sem_dist.scoring import derive_semantic_scores


def quality(value: int) -> dict:
    return {
        "semantic_clarity": value,
        "structural_integrity": value,
        "appearance_naturalness": value,
        "artifact_non_dominance": value,
    }


def candidate() -> CandidateUnitAssessment:
    return CandidateUnitAssessment.model_validate(
        {
            "detailed_description": "A dog beside a small house.",
            "global_interpretation": "An outdoor dog and house scene.",
            "semantic_units": [
                {
                    "unit_id": "C1",
                    "unit_type": "primary_entity",
                    "description": "a dog",
                    "visible_evidence": "four-legged animal outline",
                    "importance": 5,
                    "importance_label": "critical",
                    "identity_confidence": 80,
                    "quality_dimensions": quality(8),
                    "quality_defects": [],
                },
                {
                    "unit_id": "C2",
                    "unit_type": "secondary_entity",
                    "description": "a small house",
                    "visible_evidence": "roof-like shape",
                    "importance": 1,
                    "importance_label": "minor",
                    "identity_confidence": 20,
                    "quality_dimensions": quality(4),
                    "quality_defects": ["blur"],
                },
            ],
            "uncertainties": ["the second object may not be a house"],
            "overall_quality_observations": ["severe blur"],
        }
    )


def reference() -> ReferenceUnitInventory:
    return ReferenceUnitInventory.model_validate(
        {
            "detailed_description": "A dog beside a house.",
            "global_interpretation": "An outdoor dog and house scene.",
            "semantic_units": [
                {
                    "unit_id": "R1",
                    "unit_type": "primary_entity",
                    "description": "a dog",
                    "visible_evidence": "dog body",
                    "importance": 5,
                    "importance_label": "critical",
                },
                {
                    "unit_id": "R2",
                    "unit_type": "secondary_entity",
                    "description": "a house",
                    "visible_evidence": "building facade",
                    "importance": 3,
                    "importance_label": "moderate",
                },
            ],
            "ambiguities": [],
        }
    )


def consistency() -> SemanticConsistencyAssessment:
    return SemanticConsistencyAssessment.model_validate(
        {
            "global_similarity": 75,
            "global_rationale": "Same broad scene with uncertainty.",
            "reference_to_candidate": [
                {
                    "source_unit_id": "R1",
                    "matched_target_unit_ids": ["C1"],
                    "match_type": "exact",
                    "semantic_similarity": 100,
                    "rationale": "dog matches dog",
                },
                {
                    "source_unit_id": "R2",
                    "matched_target_unit_ids": ["C2"],
                    "match_type": "partial",
                    "semantic_similarity": 50,
                    "rationale": "house is only tentative",
                },
            ],
            "candidate_to_reference": [
                {
                    "source_unit_id": "C1",
                    "matched_target_unit_ids": ["R1"],
                    "match_type": "exact",
                    "semantic_similarity": 100,
                    "rationale": "dog matches dog",
                },
                {
                    "source_unit_id": "C2",
                    "matched_target_unit_ids": [],
                    "match_type": "unrelated",
                    "semantic_similarity": 0,
                    "rationale": "not reliably source-supported",
                },
            ],
            "missing_reference_units": [],
            "unsupported_candidate_units": ["C2"],
            "comparison_summary": "Strong dog match; house is uncertain.",
        }
    )


def test_sr_and_sq_are_independent_weighted_unit_means():
    scores = derive_semantic_scores(candidate(), reference(), consistency())
    assert scores.SR == 70.0
    assert scores.SQ == pytest.approx(73.333, abs=0.001)
    assert scores.SQ > scores.SR  # SR and SQ are intentionally independent.
    assert scores.candidate_units[0].intrinsic_quality == 80.0
    assert scores.candidate_units[1].intrinsic_quality == 40.0


def test_sc_uses_global_and_bidirectional_unit_f1():
    scores = derive_semantic_scores(candidate(), reference(), consistency())
    assert scores.SC_global == 75.0
    assert scores.SC_unit_recall == 81.25
    assert scores.SC_unit_precision == pytest.approx(83.333, abs=0.001)
    assert scores.SC_unit_f1 == pytest.approx(82.278, abs=0.001)
    assert scores.SC == pytest.approx(79.367, abs=0.001)


def test_missing_directional_unit_is_rejected():
    value = consistency().model_dump()
    value["reference_to_candidate"] = value["reference_to_candidate"][:1]
    with pytest.raises(ValueError, match="cover every source unit once"):
        derive_semantic_scores(
            candidate(), reference(), SemanticConsistencyAssessment.model_validate(value)
        )


def test_empty_candidate_has_zero_sr_sq_and_unit_consistency():
    cand = CandidateUnitAssessment.model_validate(
        {
            "detailed_description": "No recognizable semantic content.",
            "global_interpretation": "No stable interpretation.",
            "semantic_units": [],
            "uncertainties": [],
            "overall_quality_observations": ["artifact field"],
        }
    )
    comp = SemanticConsistencyAssessment.model_validate(
        {
            "global_similarity": 0,
            "global_rationale": "No source meaning is reconstructed.",
            "reference_to_candidate": [
                {
                    "source_unit_id": unit.unit_id,
                    "matched_target_unit_ids": [],
                    "match_type": "absent",
                    "semantic_similarity": 0,
                    "rationale": "absent",
                }
                for unit in reference().semantic_units
            ],
            "candidate_to_reference": [],
            "missing_reference_units": ["R1", "R2"],
            "unsupported_candidate_units": [],
            "comparison_summary": "Nothing recognized.",
        }
    )
    scores = derive_semantic_scores(cand, reference(), comp)
    assert scores.SR == scores.SQ == scores.SC == 0


def test_sc_projection_excludes_sr_and_sq_fields():
    view = candidate_consistency_view(candidate().model_dump(mode="json"))
    rendered = json.dumps(view)
    assert "identity_confidence" not in rendered
    assert "quality_dimensions" not in rendered
    assert "importance" in rendered


def test_candidate_parser_and_consecutive_ids():
    parsed = parse_candidate(candidate().model_dump_json())
    assert parsed.semantic_units[0].unit_id == "C1"
    payload = candidate().model_dump(mode="json")
    payload["semantic_units"][1]["unit_id"] = "C3"
    with pytest.raises(ValueError, match="consecutive"):
        CandidateUnitAssessment.model_validate(payload)


def test_importance_label_must_match_numeric_value():
    payload = candidate().model_dump(mode="json")
    payload["semantic_units"][0]["importance_label"] = "minor"
    with pytest.raises(ValueError, match="requires label"):
        CandidateUnitAssessment.model_validate(payload)


def test_unmatched_nonzero_similarity_is_zeroed_and_audited():
    value = consistency().model_dump(mode="json")
    value["candidate_to_reference"][1]["semantic_similarity"] = 25
    judgment = SemanticConsistencyAssessment.model_validate(value)
    scores = derive_semantic_scores(candidate(), reference(), judgment)
    assert scores.SC_unit_precision == pytest.approx(83.333, abs=0.001)
    assert scores.unmatched_nonzero_similarity_count == 1


def test_consistency_retry_schema_uses_inventory_ids():
    from vlm_sem_dist.runner import consistency_retry_schema

    schema = consistency_retry_schema(reference(), candidate())
    ref_direction = schema["properties"]["reference_to_candidate"]
    cand_direction = schema["properties"]["candidate_to_reference"]
    assert ref_direction["minItems"] == ref_direction["maxItems"] == 2
    assert cand_direction["minItems"] == cand_direction["maxItems"] == 2
    assert ref_direction["items"]["properties"]["source_unit_id"]["enum"] == ["R1", "R2"]
    assert ref_direction["items"]["properties"]["matched_target_unit_ids"]["items"]["enum"] == [
        "C1",
        "C2",
    ]
