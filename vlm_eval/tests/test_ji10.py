from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_anchor_preparation_resolves_paths_from_manifest(tmp_path):
    manifest_dir = tmp_path / "input"
    image_dir = manifest_dir / "images"
    image_dir.mkdir(parents=True)
    source = image_dir / "source.png"
    Image.new("RGB", (16, 16), "blue").save(source)
    manifest = manifest_dir / "pairs.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": "sample",
                "reference": "images/source.png",
                "reconstruction": "images/source.png",
            }
        ],
    )

    output_dir = tmp_path / "anchors"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_ji10_anchors.py"),
            str(manifest),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records = [json.loads(line) for line in (output_dir / "anchors.jsonl").read_text().splitlines()]
    assert [record["metadata"]["anchor"] for record in records] == ["identity", "jpeg"]
    assert all(record["reference"] == str(source.resolve()) for record in records)
    assert Path(records[1]["reconstruction"]).is_file()


def test_normalizer_preserves_raw_scores_and_sc(tmp_path):
    source = str((tmp_path / "source.png").resolve())
    raw_path = tmp_path / "raw.jsonl"
    anchor_path = tmp_path / "anchors.jsonl"
    output_path = tmp_path / "final.jsonl"
    write_jsonl(
        raw_path,
        [
            {
                "status": "ok",
                "reference": {"path": source},
                "semantic_scores": {"SR": 60, "SQ": 55, "SC": 40},
            }
        ],
    )
    write_jsonl(
        anchor_path,
        [
            {
                "status": "ok",
                "metadata": {"source_path": source, "anchor": "identity"},
                "semantic_scores": {"SR": 90, "SQ": 85, "SC": 100},
            },
            {
                "status": "ok",
                "metadata": {"source_path": source, "anchor": "jpeg"},
                "semantic_scores": {"SR": 30, "SQ": 25, "SC": 50},
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/normalize_ji10.py"),
            str(raw_path),
            "--anchors",
            str(anchor_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(output_path.read_text())
    assert record["semantic_scores"] == {"SC": 40.0, "SR": 50.0, "SQ": 50.0}
    assert record["semantic_scores_raw"] == {"SR": 60.0, "SQ": 55.0, "SC": 40.0}
    assert record["ji10_normalization"]["SR"]["fallback"] is False
