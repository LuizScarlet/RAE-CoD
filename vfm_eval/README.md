# VFM evaluation

This directory is a standalone five-encoder feature evaluator for RAE-CoD. It
has its own dependencies and requires no model, training, or inference module
from the parent repository. It compares ground-truth and
reconstructed images in heterogeneous frozen vision spaces:

| Key | Encoder | Feature |
|---|---|---|
| `inception` | Inception-v3 | pooled 2048-D feature |
| `dino` | DINOv2 ViT-L/14 | class token |
| `siglip` | SigLIP2 ViT-SO400M/16 | class token |
| `clip` | OpenAI CLIP ViT-L/14 | class token |
| `convnext` | ConvNeXt-v2-B | globally pooled feature |

## Metrics

For source feature `f` and reconstruction feature `f_hat`, the evaluator reports:

- **MSE:** mean squared feature error;
- **RelMSE:** MSE divided by the source feature energy for each image, then
  averaged (lower is better);
- **COS:** cosine similarity (higher is better);
- **FD:** Frechet distance between source and reconstruction feature
  distributions (lower is better).

The five-encoder aggregate is

```text
RelMSE^5 = mean_m RelMSE_m
COS^5    = mean_m COS_m
FDr^5    = mean_m (FD_m / FD_m^baseline)
```

The FD baseline must be selected once and kept fixed across all methods and
operating points.

## Installation

From this directory:

```bash
bash setup_env.sh
source .venv/bin/activate
python download_weights.py
```

For manual setup, create any Python environment and run
`pip install -r requirements.txt`. All evaluator imports are local to this
folder; no parent-repository `PYTHONPATH` is needed.

`timm` downloads DINOv2, SigLIP2, CLIP, and ConvNeXt-v2 weights. Inception-v3
uses the TensorFlow-compatible torch-fidelity checkpoint. Weight files are
cached through the normal PyTorch/timm cache directories.

## Input layout

Ground truth and reconstruction directories are paired by **identical
filenames**. Supported extensions are PNG, JPEG, BMP, TIFF, and WebP.

```text
ground_truth/
  000001.png
  000002.png
reconstructions/
  000001.png
  000002.png
```

Unmatched reconstruction files are ignored; missing reconstructions reduce the
number of pairs and should be treated as an evaluation error by the caller.

## Run all five encoders

```bash
python compute_semantic_distance.py \
  --gt-dir /path/to/ground_truth \
  --recon-dir /path/to/reconstructions \
  --output outputs/method_rate.json \
  --batch-size 64 --num-workers 8 --device cuda
```

A `.json` output is machine readable. A non-JSON suffix writes a compact text
report. Add `--print-per-image` to include paired MSE/RelMSE/COS records.
Individual encoders can be disabled, for example `--no-inception` or
`--no-clip`, but all five must be enabled when computing the five-encoder aggregate.

## Multiple stochastic seeds

Pass several reconstruction directories after one `--recon-dir`:

```bash
python compute_semantic_distance.py \
  --gt-dir /path/to/ground_truth \
  --recon-dir /path/to/seed0 /path/to/seed1 /path/to/seed2 \
  --output outputs/method_rate_multiseed.json
```

Paired metrics are averaged over seeds. Reconstruction sufficient statistics
are pooled before FD is computed; source statistics are counted once.

For multiple operating points, the helper runs one report per directory:

```bash
bash batch_eval.sh /path/to/ground_truth outputs/vfm \
  /path/to/rate12 /path/to/rate16 /path/to/rate24
```

## Fixed AEIC-ME FD anchor

`FDr^5` uses the highest-rate AEIC-ME point on MSCOCO-30K as one fixed
normalization anchor. Its average bitrate is `0.0073` bpp.
The required per-encoder denominators are provided in
[`aeic_me_mscoco30k_anchor.json`](aeic_me_mscoco30k_anchor.json):

| Evaluator key | Encoder | FD denominator |
|---|---|---:|
| `inception` | Inception-v3 | 17.8907 |
| `dino` | DINOv2 | 556.9173 |
| `siglip` | SigLIP2 | 36.2722 |
| `clip` | CLIP | 302.0514 |
| `convnext` | ConvNeXt-v2 | 875.5074 |

Use this file unchanged for every method, bitrate, and stochastic seed. The
anchor is fixed rather than bitrate matched or interpolated.

## Five-VFM aggregation

Evaluate every method/rate point as JSON and then run:

```bash
python compute_vfm5_averages.py \
  outputs/rate12.json outputs/rate16.json outputs/rate24.json \
  --fd-baseline aeic_me_mscoco30k_anchor.json \
  --bpp 0.008 0.004 0.002 \
  --output outputs/aggregate.json
```

Each output point contains `rel_mse5`, `cos5`, and `fdr5`. The AEIC-ME anchor
itself has `fdr5 = 1`. Keep this exact baseline fixed across all methods,
operating points, and stochastic seeds.

## Reproducibility notes

- Evaluate every method with the same image set, preprocessing, encoder weights,
  and software stack.
- FD is a dataset statistic and is unreliable on very small sample sets, like Kodak.
- The script enables TF32 and bfloat16 autocast for throughput. Record hardware
  and package versions for strict audits.
