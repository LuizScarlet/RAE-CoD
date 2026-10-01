#!/usr/bin/env python3
"""Create clean-identity and JPEG-Q1 anchor pairs for an evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def source_key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Method-pair JSONL manifest.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, help="Defaults below --output-dir.")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    jpeg_dir = output_dir / "jpeg_q1"
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.anchor_manifest or output_dir / "anchors.jsonl"

    manifest_input = args.manifest.expanduser().resolve()
    references: dict[str, Path] = {}
    for record in read_jsonl(manifest_input):
        reference = Path(os.path.expandvars(record["reference"])).expanduser()
        if not reference.is_absolute():
            reference = manifest_input.parent / reference
        reference = reference.resolve()
        if not reference.is_file():
            raise FileNotFoundError(reference)
        references[str(reference)] = reference

    anchors = []
    for reference in sorted(references.values(), key=str):
        key = source_key(reference)
        jpeg_path = jpeg_dir / f"{key}.jpg"
        with Image.open(reference) as image:
            image.convert("RGB").save(
                jpeg_path,
                format="JPEG",
                quality=1,
                subsampling=2,
                optimize=True,
                progressive=False,
            )
        common = {"source_key": key, "source_path": str(reference)}
        anchors.extend(
            [
                {
                    "id": f"anchor/{key}/identity",
                    "reference": str(reference),
                    "reconstruction": str(reference),
                    "metadata": {**common, "anchor": "identity"},
                },
                {
                    "id": f"anchor/{key}/jpeg-q1",
                    "reference": str(reference),
                    "reconstruction": str(jpeg_path),
                    "metadata": {**common, "anchor": "jpeg"},
                },
            ]
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in anchors:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"sources={len(references)} anchors={len(anchors)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
