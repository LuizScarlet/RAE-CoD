#!/usr/bin/env python3
"""Apply per-source JPEG--identity normalization to raw SR and SQ scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def clip100(value: float) -> float:
    return 100.0 * min(max(value, 0.0), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Raw method results JSONL.")
    parser.add_argument(
        "--anchors", type=Path, required=True, help="Evaluated anchor results JSONL."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guard", type=float, default=10.0)
    parser.add_argument("--fallback-sr", type=float, default=30.434)
    parser.add_argument("--fallback-sq", type=float, default=37.015)
    args = parser.parse_args()

    anchors: dict[str, dict[str, dict]] = {}
    for record in read_jsonl(args.anchors):
        if record.get("status") != "ok":
            continue
        metadata = record.get("metadata", {})
        source_path = str(Path(metadata["source_path"]).resolve())
        kind = metadata["anchor"]
        anchors.setdefault(source_path, {})[kind] = record["semantic_scores"]

    output_records = []
    for record in read_jsonl(args.results):
        if record.get("status") != "ok":
            output_records.append(record)
            continue
        source_path = str(Path(record["reference"]["path"]).resolve())
        source_anchors = anchors.get(source_path, {})
        if set(source_anchors) != {"identity", "jpeg"}:
            raise KeyError(f"Missing identity/JPEG anchors for {source_path}")
        raw = record["semantic_scores"]
        final = {"SC": float(raw["SC"])}
        details = {}
        for metric, fallback in (("SR", args.fallback_sr), ("SQ", args.fallback_sq)):
            upper = float(source_anchors["identity"][metric])
            lower = float(source_anchors["jpeg"][metric])
            headroom = upper - lower
            used_fallback = headroom <= args.guard
            denominator = fallback if used_fallback else headroom
            final[metric] = clip100((float(raw[metric]) - lower) / denominator)
            details[metric] = {
                "identity": upper,
                "jpeg": lower,
                "headroom": headroom,
                "denominator": denominator,
                "fallback": used_fallback,
            }
        record["semantic_scores_raw"] = {
            "SR": float(raw["SR"]),
            "SQ": float(raw["SQ"]),
            "SC": float(raw["SC"]),
        }
        record["semantic_scores"] = final
        record["ji10_normalization"] = details
        output_records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"records={len(output_records)} output={args.output}")


if __name__ == "__main__":
    main()
