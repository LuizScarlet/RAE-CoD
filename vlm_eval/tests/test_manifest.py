from __future__ import annotations

import json

import pytest
from PIL import Image

from vlm_sem_dist.manifest import (
    ManifestError,
    load_manifest,
    merge_result_files,
    select_shard,
)


def test_relative_manifest_paths_and_sharding(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8), "red").save(images / "a.png")
    Image.new("RGB", (8, 8), "blue").save(images / "b.png")
    manifest = tmp_path / "pairs.jsonl"
    rows = [
        {"id": "a", "gt": "images/a.png", "recon": "images/b.png", "rate": 0.01},
        {"id": "b", "reference": "images/b.png", "reconstruction": "images/a.png"},
    ]
    manifest.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    pairs = load_manifest(manifest, check_files=True)
    assert pairs[0].reference == str((images / "a.png").resolve())
    assert pairs[0].metadata["rate"] == 0.01
    assert [pair.id for pair in select_shard(pairs, shard_index=1, num_shards=2)] == ["b"]


def test_duplicate_ids_rejected(tmp_path):
    manifest = tmp_path / "pairs.jsonl"
    row = {"id": "same", "reference": "a.png", "reconstruction": "b.png"}
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
    with pytest.raises(ManifestError, match="Duplicate"):
        load_manifest(manifest)


def test_merge_accepts_empty_shard(tmp_path):
    empty = tmp_path / "empty.jsonl"
    full = tmp_path / "full.jsonl"
    output = tmp_path / "merged.jsonl"
    empty.write_text("", encoding="utf-8")
    full.write_text('{"sample_id":"a","status":"error"}\n', encoding="utf-8")
    assert merge_result_files([empty, full], output) == 1
    assert '"sample_id":"a"' in output.read_text(encoding="utf-8")
