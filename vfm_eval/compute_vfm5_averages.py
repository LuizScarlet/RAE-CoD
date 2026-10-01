#!/usr/bin/env python3
"""Aggregate five-VFM JSON reports into RelMSE^5, COS^5, and FDr^5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_KEYS = ("inception", "dino", "siglip", "clip", "convnext")


def load_models(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    missing = set(MODEL_KEYS) - set(models)
    if missing:
        raise ValueError(f"{path} is missing VFM results: {sorted(missing)}")
    return models


def aggregate(models: dict, baseline: dict) -> dict[str, float]:
    return {
        "rel_mse5": sum(float(models[key]["rel_mse"]) for key in MODEL_KEYS) / 5,
        "cos5": sum(float(models[key]["cosine"]) for key in MODEL_KEYS) / 5,
        "fdr5": sum(
            float(models[key]["fd"]) / float(baseline[key]["fd"]) for key in MODEL_KEYS
        )
        / 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports", type=Path, nargs="+", help="Per-rate VFM JSON reports."
    )
    parser.add_argument(
        "--fd-baseline",
        type=Path,
        required=True,
        help="JSON report supplying the per-VFM FD normalization denominators.",
    )
    parser.add_argument(
        "--bpp", type=float, nargs="*", help="Optional bpp for each report."
    )
    parser.add_argument("--output", type=Path, help="Output JSON; stdout if omitted.")
    args = parser.parse_args()
    if args.bpp is not None and len(args.bpp) not in (0, len(args.reports)):
        raise SystemExit("--bpp must contain one value per report")

    baseline = load_models(args.fd_baseline)
    rows = []
    for index, report in enumerate(args.reports):
        row = {"report": str(report), **aggregate(load_models(report), baseline)}
        if args.bpp:
            row["bpp"] = args.bpp[index]
        rows.append(row)
    rendered = json.dumps({"points": rows}, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
