"""Frozen prompts for the SR/SQ/SC semantic-unit protocol."""

from __future__ import annotations

import hashlib
import json
from typing import Any


CANDIDATE_SYSTEM_PROMPT = r"""
You are a strict human-style semantic inspector. You see ONLY one anonymous reconstructed
image. You do not know its ground truth, codec, bitrate, filename, or expected result.

First write a detailed description of what is visibly present. Separate direct evidence from
uncertainty. Do not turn compression blobs into a confident story. Then decompose the
apparent semantic content into 0-12 nonredundant semantic units. A unit may be the global
scene/event, a central entity, a secondary entity, a meaningful action/state, an
identity-defining attribute, a relation/layout, meaningful text/symbol, or important context.
Do not make separate duplicate units for the same meaning. Semantic units must describe
meaning, not degradation: never create a unit for blur, grain, compression artifacts,
lighting, camera angle, generic color/texture, or "image quality". Put those observations
only in quality fields. Color, material, or text may be a unit only when it changes identity
or communicative meaning. If no semantic content is human-recognizable, return an empty
semantic_units list.

For every unit assign importance from 1 to 5 according to its centrality to the apparent
communicative meaning—not its size, confidence, quality, or similarity to an unknown source:
5 critical, 4 major, 3 moderate, 2 supporting, 1 minor. Do not lower importance merely
because a central object is damaged or uncertain.

IDENTITY CONFIDENCE (integer 0-100)
Estimate how confidently an ordinary unprimed human could defend the exact stated semantic
identity from visible evidence. This is not confidence that some fluent caption can be
invented. Check parts, boundaries, geometry, scale, spatial relations, and alternative
interpretations.
95-100 unmistakable exact identity; 80-94 clear with minor ambiguity; 60-79 probable;
40-59 only broad category or materially uncertain identity; 20-39 weak reading; 1-19
speculative/pareidolic cue; 0 no defensible identity. Severe artifacts, contradictory
geometry, or multiple equally plausible readings must lower identity confidence.

INTRINSIC QUALITY FOR EACH UNIT (four integers 0-10)
Judge the visible quality of that unit independently of its identity confidence and of any
unknown GT:
- semantic_clarity: legibility of meaningful features;
- structural_integrity: coherent, non-melted, non-duplicated parts and boundaries;
- appearance_naturalness: plausible surfaces, shapes, colors, and transitions for the domain;
- artifact_non_dominance: 10 means artifacts do not dominate the unit; 0 means the unit is
  essentially an artifact field.
Use 9-10 excellent, 7-8 good, 4-6 damaged but usable, 1-3 severely broken, 0 no valid visual
quality. SQ quality is not photographic beauty. A blurry but intact unit may retain identity
while receiving lower clarity.

Use unit IDs C1, C2, ... consecutively in decreasing semantic importance. Return strict JSON
only. The detailed description must not include knowledge unavailable from this image.
""".strip()

REFERENCE_SYSTEM_PROMPT = r"""
You are building a candidate-independent semantic inventory for one ground-truth image.
Describe only clearly visible, human-relevant meaning. Then extract 0-12 nonredundant units
using the same ontology: global scene/event, primary or secondary entity, meaningful
attribute, action/state, relation/layout, meaningful text/symbol, or context.

Assign importance by semantic centrality to the source image: 5 critical, 4 major,
3 moderate, 2 supporting, 1 minor. Do not use reconstruction knowledge, compression
expectations, or likely failure modes. Avoid duplicate units and incidental microtexture.
Do not create units for photographic quality, blur, grain, lighting, camera angle, or
generic color/texture; color, material, and text count only when semantically identifying.
Use IDs R1, R2, ... consecutively in decreasing importance. Return strict JSON only.
""".strip()

CONSISTENCY_SYSTEM_PROMPT = r"""
You are a semantic matcher. You receive two frozen text inventories created independently:
REFERENCE was extracted from a GT image; CANDIDATE was extracted GT-blind from a
reconstruction. You see no images in this call. Compare meanings, not wording, visual style,
pixel fidelity, or intrinsic image quality. Do not use the candidate's identity confidence
or quality values when assigning semantic similarity.

Give one global_similarity integer 0-100 for agreement of the overall scene/event/purpose and
central identities: 95-100 same meaning; 80-94 same central meaning with minor differences;
60-79 substantial agreement with a meaningful change; 40-59 broad partial agreement;
20-39 weak overlap with major substitution; 1-19 incidental overlap; 0 unrelated or
contradictory global meaning.

Then compare units in BOTH directions:
1. reference_to_candidate: one entry for every R unit, exactly once. This measures whether
   source semantics were retained (recall).
2. candidate_to_reference: one entry for every C unit, exactly once. This measures whether
   reconstruction semantics are supported by the source (precision/hallucination control).

A source unit may match zero, one, or several target IDs when granularity differs. Use
semantic_similarity 0-100: 95-100 exact; 80-94 clearly equivalent; 60-79 same core category
with attribute/detail change; 40-59 partial but important overlap; 20-39 weak broad overlap;
1-19 incidental relation; 0 absent, unrelated, or contradicted. With no target match, emit an
empty matched_target_unit_ids list and similarity 0. Do not omit difficult units. Use only
IDs present in the supplied inventories. Return strict JSON only.
""".strip()

PROMPT_SHA256 = hashlib.sha256(
    (
        CANDIDATE_SYSTEM_PROMPT
        + "\n---REFERENCE---\n"
        + REFERENCE_SYSTEM_PROMPT
        + "\n---CONSISTENCY---\n"
        + CONSISTENCY_SYSTEM_PROMPT
    ).encode("utf-8")
).hexdigest()


def build_candidate_messages(candidate_image: Any, *, repair_note: str | None = None):
    instruction = "Describe this anonymous reconstruction and return the strict unit JSON."
    if repair_note:
        instruction += f" Correct the prior invalid output: {repair_note[:700]}"
    return [
        {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "ANONYMOUS RECONSTRUCTION:"},
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
    ], [candidate_image]


def build_reference_messages(reference_image: Any, *, repair_note: str | None = None):
    instruction = "Create the frozen GT semantic inventory and return strict JSON."
    if repair_note:
        instruction += f" Correct the prior invalid output: {repair_note[:700]}"
    return [
        {"role": "system", "content": REFERENCE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "GROUND-TRUTH IMAGE:"},
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
    ], [reference_image]


def build_consistency_messages(
    reference_inventory: dict[str, Any],
    candidate_inventory: dict[str, Any],
    *,
    repair_note: str | None = None,
):
    reference_json = json.dumps(reference_inventory, ensure_ascii=False, separators=(",", ":"))
    candidate_json = json.dumps(candidate_inventory, ensure_ascii=False, separators=(",", ":"))
    instruction = "Return the strict bidirectional semantic-comparison JSON."
    if repair_note:
        instruction += f" Correct the prior invalid output: {repair_note[:700]}"
    return [
        {"role": "system", "content": CONSISTENCY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "FROZEN REFERENCE INVENTORY:\n"
                + reference_json
                + "\n\nFROZEN GT-BLIND CANDIDATE INVENTORY:\n"
                + candidate_json
                + "\n\n"
                + instruction
            ),
        },
    ], []


def reference_consistency_view(inventory: dict[str, Any]) -> dict[str, Any]:
    """Keep only semantic content and importance for the independent SC call."""

    return {
        "detailed_description": inventory["detailed_description"],
        "global_interpretation": inventory["global_interpretation"],
        "semantic_units": [
            {
                "unit_id": unit["unit_id"],
                "unit_type": unit["unit_type"],
                "description": unit["description"],
                "visible_evidence": unit["visible_evidence"],
                "importance": unit["importance"],
            }
            for unit in inventory["semantic_units"]
        ],
    }


def candidate_consistency_view(inventory: dict[str, Any]) -> dict[str, Any]:
    """Remove SR confidence and SQ quality before SC to keep the scores decoupled."""

    return {
        "detailed_description": inventory["detailed_description"],
        "global_interpretation": inventory["global_interpretation"],
        "semantic_units": [
            {
                "unit_id": unit["unit_id"],
                "unit_type": unit["unit_type"],
                "description": unit["description"],
                "visible_evidence": unit["visible_evidence"],
                "importance": unit["importance"],
            }
            for unit in inventory["semantic_units"]
        ],
    }
