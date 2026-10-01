# VLM semantic evaluation (SR / SQ / SC)

This directory is a standalone VLM evaluator for RAE-CoD. It requires no
model, training, or inference module from the parent repository: copy this
folder anywhere, install it from this directory, and follow the commands below.

The evaluator uses a blinded Qwen3.5-9B judge to separate three properties of a
reconstructed image:

- **Semantic Recognizability (SR):** how confidently an unprimed observer can
  identify the visible content;
- **Semantic Quality (SQ):** whether that content is coherent, structurally
  intact, natural, and not dominated by artifacts;
- **Semantic Consistency (SC):** how faithfully the reconstruction retains the
  source meaning without unsupported or substituted content.

SR and SQ are judged without showing the source. SC is obtained by comparing
independently generated source and reconstruction inventories. The complete
frozen protocol and formulas are in [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

## Protocol overview

Each source/reconstruction pair uses three logically separate calls:

1. **Source inventory:** inspect only the source and extract a global
   interpretation and up to 12 importance-weighted semantic units. The result
   is cached and reused for all reconstructions of that source.
2. **Reconstruction inventory:** inspect only the anonymous reconstruction.
   For each visible semantic unit, estimate identity confidence and intrinsic
   quality. These values produce raw SR and SQ.
3. **Consistency matching:** compare only the two text inventories in both
   directions. This produces source recall, reconstruction precision, and SC;
   the images and the SR/SQ ratings are withheld from this call.

Codec name, bitrate, filename, conventional metrics, and results from other
methods are never included in a model prompt.

## Installation

The validated backend is Qwen3.5-9B with vLLM and should use an environment
separate from model training. From this directory:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
bash scripts/download_model.sh /path/to/Qwen3.5-9B
```

The setup script installs this local package and its runtime dependencies. If
your CUDA platform requires a specific vLLM wheel, create the environment with
`python -m venv .venv`, install the appropriate wheel, and then run
`pip install -e .`.

The frozen model revision is
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Keep the model revision,
Transformers/vLLM versions, and evaluator settings fixed within a comparison.

## Pair manifest

Provide one JSON object per line:

```json
{"id":"method/rate/000001","reference":"/path/to/ground_truth/000001.png","reconstruction":"/path/to/reconstructions/000001.png","metadata":{"method":"method","bpp":0.001}}
{"id":"method/rate/000002","reference":"/path/to/ground_truth/000002.png","reconstruction":"/path/to/reconstructions/000002.png","metadata":{"method":"method","bpp":0.001}}
```

IDs must be unique. Image paths may be absolute or relative to the manifest.
Use the same source path across methods and rates so that the reference cache is
shared. [`examples/manifest.jsonl`](examples/manifest.jsonl) is a small runnable
schema example.

## 1. Compute raw SR, SQ, and SC

```bash
CUDA_VISIBLE_DEVICES=0 vlm-sem-dist pairs.jsonl \
  --model /path/to/Qwen3.5-9B \
  --output outputs/raw.jsonl \
  --reference-cache outputs/reference_cache.json \
  --batch-size 32 \
  --min-pixels 262144 --max-pixels 262144
```

`python -m vlm_sem_dist` is equivalent to `vlm-sem-dist`. The default settings
use temperature 0, seed 0, greedy decoding, disabled thinking, and
schema-constrained JSON. The runner also writes `raw.jsonl.run.json`, which
records the model, prompt hash, runtime settings, retry counts, and timing.

Each successful record contains the three structured assessments, raw
`semantic_scores.SR`, `semantic_scores.SQ`, and `semantic_scores.SC`, detailed SC
components, image hashes, and model/runtime metadata. Use
`--reference-cache` for dataset-scale comparisons; a cache is accepted only
when its protocol, prompt hash, and model revision match the current run.

## 2. Evaluate JPEG--identity anchors

Final SR and SQ use a source-specific JPEG--identity normalization. First create
one identity control and one baseline JPEG quality-1 (4:2:0) control per source:

```bash
python scripts/prepare_ji10_anchors.py pairs.jsonl \
  --output-dir outputs/ji10
```

Evaluate the generated controls with exactly the same model and settings:

```bash
CUDA_VISIBLE_DEVICES=0 vlm-sem-dist outputs/ji10/anchors.jsonl \
  --model /path/to/Qwen3.5-9B \
  --output outputs/ji10/anchor_results.jsonl \
  --reference-cache outputs/reference_cache.json \
  --batch-size 32 \
  --min-pixels 262144 --max-pixels 262144
```

Only the controls' raw SR and SQ values are used as normalization anchors.

## 3. Produce final scores

```bash
python scripts/normalize_ji10.py outputs/raw.jsonl \
  --anchors outputs/ji10/anchor_results.jsonl \
  --output outputs/final_scores.jsonl
```

For each source and `M` in `{SR, SQ}`, the normalizer computes

```text
headroom    = identity_M - jpeg_M
denominator = headroom, if headroom > 10; otherwise fallback_M
M_final     = 100 * clip((M_raw - jpeg_M) / denominator, 0, 1)
```

The frozen fallback headrooms are `30.434` for SR and `37.015` for SQ. SC is
already source matched and is copied without normalization. The output keeps the
original values in `semantic_scores_raw`, stores final SR/SQ/SC in
`semantic_scores`, and records every anchor and denominator under
`ji10_normalization`.

## Reproducibility and checks

- Compare methods on exactly the same source set and resize/crop policy.
- Do not mix model revisions, prompt hashes, image budgets, or runtime backends.
- Normalize each image before averaging a dataset and inspect clipping rates
  when diagnosing saturation.
- An empty reconstruction inventory deterministically receives SR=SQ=0. The SC
  implementation validates that both directional match lists cover every
  source-side unit exactly once.
- Raw model text is retained by default for auditing; use `--no-raw-output` only
  when storage is constrained.

Run the local unit tests without the parent repository:

```bash
pip install -e '.[dev]'
pytest -q
vlm-sem-dist --help
```
