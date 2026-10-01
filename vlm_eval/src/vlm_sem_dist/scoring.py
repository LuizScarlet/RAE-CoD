"""Deterministic SR, SQ, and SC scoring."""

from __future__ import annotations

from .schemas import (
    CandidateUnitAssessment,
    CandidateUnitScore,
    DirectionalUnitMatch,
    ReferenceUnitInventory,
    SemanticConsistencyAssessment,
    SemanticScores,
    UnitQualityDimensions,
)

QUALITY_WEIGHTS = {
    "semantic_clarity": 0.30,
    "structural_integrity": 0.30,
    "appearance_naturalness": 0.20,
    "artifact_non_dominance": 0.20,
}
SC_GLOBAL_WEIGHT = 0.40
SC_UNIT_WEIGHT = 0.60


def intrinsic_quality(dimensions: UnitQualityDimensions) -> float:
    """Map four 0-10 intrinsic-quality dimensions to 0-100."""

    values = dimensions.model_dump()
    score = 10.0 * sum(QUALITY_WEIGHTS[name] * values[name] for name in QUALITY_WEIGHTS)
    return round(max(0.0, min(100.0, score)), 3)


def score_candidate_units(
    candidate: CandidateUnitAssessment,
) -> tuple[float, float, list[CandidateUnitScore], int]:
    """Compute decoupled SR and SQ as importance-weighted unit means."""

    total_importance = sum(unit.importance for unit in candidate.semantic_units)
    if total_importance == 0:
        return 0.0, 0.0, [], 0

    sr_numerator = 0.0
    sq_numerator = 0.0
    unit_scores: list[CandidateUnitScore] = []
    for unit in candidate.semantic_units:
        quality = intrinsic_quality(unit.quality_dimensions)
        sr_contribution = unit.importance * unit.identity_confidence / total_importance
        sq_contribution = unit.importance * quality / total_importance
        sr_numerator += unit.importance * unit.identity_confidence
        sq_numerator += unit.importance * quality
        unit_scores.append(
            CandidateUnitScore(
                unit_id=unit.unit_id,
                importance=unit.importance,
                identity_confidence=unit.identity_confidence,
                intrinsic_quality=quality,
                sr_contribution=round(sr_contribution, 3),
                sq_contribution=round(sq_contribution, 3),
            )
        )
    return (
        round(sr_numerator / total_importance, 3),
        round(sq_numerator / total_importance, 3),
        unit_scores,
        total_importance,
    )


def _validated_directional_score(
    matches: list[DirectionalUnitMatch],
    source_units: list,
    target_ids: set[str],
    *,
    direction: str,
) -> tuple[float, int]:
    source_ids = [unit.unit_id for unit in source_units]
    emitted_ids = [match.source_unit_id for match in matches]
    if len(emitted_ids) != len(set(emitted_ids)):
        raise ValueError(f"{direction} contains duplicate source_unit_id values")
    if set(emitted_ids) != set(source_ids):
        missing = sorted(set(source_ids) - set(emitted_ids))
        extra = sorted(set(emitted_ids) - set(source_ids))
        raise ValueError(
            f"{direction} must cover every source unit once; missing={missing}, extra={extra}"
        )

    by_id = {match.source_unit_id: match for match in matches}
    numerator = 0.0
    total_importance = 0
    for unit in source_units:
        match = by_id[unit.unit_id]
        unknown_targets = sorted(set(match.matched_target_unit_ids) - target_ids)
        if unknown_targets:
            valid_targets = sorted(target_ids)
            raise ValueError(
                f"{direction} has unknown target IDs: {unknown_targets}; "
                f"the only valid target IDs are {valid_targets}. Replace every unknown ID "
                "with the intended valid ID, or use an empty target list and similarity 0 "
                "when there is no supported match."
            )
        # An empty target list means no explicit semantic correspondence. Some models still
        # emit a small nonzero number for vague global resemblance; it is deterministically
        # zeroed rather than treated as a unit match. The raw judgment remains stored.
        effective_similarity = match.semantic_similarity if match.matched_target_unit_ids else 0
        numerator += unit.importance * effective_similarity
        total_importance += unit.importance
    if total_importance == 0:
        return 0.0, 0
    return round(numerator / total_importance, 3), total_importance


def score_consistency(
    candidate: CandidateUnitAssessment,
    reference: ReferenceUnitInventory,
    consistency: SemanticConsistencyAssessment,
) -> tuple[float, float, float, float, int]:
    """Compute global similarity, bidirectional unit similarity, and final independent SC.

    Reference-to-candidate coverage is semantic recall. Candidate-to-reference support is
    semantic precision and penalizes invented/substituted candidate content. Their harmonic
    mean is combined with the independently judged global similarity.
    """

    reference_units = reference.semantic_units
    candidate_units = candidate.semantic_units
    reference_ids = {unit.unit_id for unit in reference_units}
    candidate_ids = {unit.unit_id for unit in candidate_units}

    if not reference_units and not candidate_units:
        recall = precision = unit_f1 = 100.0
        reference_importance = 0
    elif not reference_units:
        if consistency.reference_to_candidate:
            raise ValueError("reference_to_candidate must be empty when the reference has no units")
        recall = 100.0
        precision, _ = _validated_directional_score(
            consistency.candidate_to_reference,
            candidate_units,
            reference_ids,
            direction="candidate_to_reference",
        )
        unit_f1 = 0.0 if precision == 0 else 2.0 * recall * precision / (recall + precision)
        reference_importance = 0
    elif not candidate_units:
        recall, reference_importance = _validated_directional_score(
            consistency.reference_to_candidate,
            reference_units,
            candidate_ids,
            direction="reference_to_candidate",
        )
        if consistency.candidate_to_reference:
            raise ValueError("candidate_to_reference must be empty when the candidate has no units")
        precision = 0.0
        unit_f1 = 0.0
    else:
        recall, reference_importance = _validated_directional_score(
            consistency.reference_to_candidate,
            reference_units,
            candidate_ids,
            direction="reference_to_candidate",
        )
        precision, _ = _validated_directional_score(
            consistency.candidate_to_reference,
            candidate_units,
            reference_ids,
            direction="candidate_to_reference",
        )
        unit_f1 = (
            0.0 if recall + precision == 0 else 2.0 * recall * precision / (recall + precision)
        )

    global_similarity = float(consistency.global_similarity)
    sc = SC_GLOBAL_WEIGHT * global_similarity + SC_UNIT_WEIGHT * unit_f1
    return (
        round(sc, 3),
        round(global_similarity, 3),
        round(recall, 3),
        round(precision, 3),
        reference_importance,
    )


def derive_semantic_scores(
    candidate: CandidateUnitAssessment,
    reference: ReferenceUnitInventory,
    consistency: SemanticConsistencyAssessment,
) -> SemanticScores:
    sr, sq, unit_scores, candidate_importance = score_candidate_units(candidate)
    sc, global_similarity, recall, precision, reference_importance = score_consistency(
        candidate, reference, consistency
    )
    unit_f1 = 0.0 if recall + precision == 0 else 2.0 * recall * precision / (recall + precision)
    if not reference.semantic_units and not candidate.semantic_units:
        unit_f1 = 100.0
    return SemanticScores(
        SR=sr,
        SQ=sq,
        SC=sc,
        SC_global=global_similarity,
        SC_unit_recall=recall,
        SC_unit_precision=precision,
        SC_unit_f1=round(unit_f1, 3),
        candidate_unit_count=len(candidate.semantic_units),
        reference_unit_count=len(reference.semantic_units),
        candidate_total_importance=candidate_importance,
        reference_total_importance=reference_importance,
        unmatched_nonzero_similarity_count=sum(
            not match.matched_target_unit_ids and match.semantic_similarity > 0
            for match in consistency.reference_to_candidate + consistency.candidate_to_reference
        ),
        candidate_units=unit_scores,
    )
