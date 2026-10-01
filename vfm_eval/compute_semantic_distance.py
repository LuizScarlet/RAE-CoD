#!/usr/bin/env python3
"""Evaluate reconstructed images in five frozen vision-model spaces.

For each encoder, the script reports feature MSE, relative feature MSE,
cosine similarity, and distributional Frechet distance. Images are paired by
filename. Multiple reconstruction directories are interpreted as independent
sampling seeds: paired metrics are averaged and reconstruction statistics are
pooled for Frechet distance.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable fast GPU math (following FD-Loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# =============================================================================
# Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    "inception": {
        "name": "inception",
        "target_size": 299,
        "feat_dim": 2048,
        "pool_type": "cls",
    },
    "dino": {
        "name": "vit_large_patch14_dinov2.lvd142m",
        "target_size": 256,
        "feat_dim": 1024,
        "pool_type": "cls",
    },
    "siglip": {
        "name": "vit_so400m_patch16_siglip_256.v2_webli",
        "target_size": 224,
        "feat_dim": 1152,
        "pool_type": "cls",
    },
    "clip": {
        "name": "vit_large_patch14_clip_224.openai",
        "target_size": 256,
        "feat_dim": 768,
        "pool_type": "cls",
    },
    "convnext": {
        "name": "convnext",
        "target_size": 224,
        "feat_dim": 1024,
        "pool_type": "cls",
    },
}


# =============================================================================
# Model Loading
# =============================================================================


def load_model(model_key, device="cuda"):
    """Load a VFM feature extractor."""
    from repr_models import TimmReprModel

    cfg = MODEL_CONFIGS[model_key]
    name = cfg["name"]

    if name == "inception":
        from inception import load_inception

        model = load_inception(device=device, normalize=False)
        return model, cfg
    elif name == "convnext":
        model = TimmReprModel(
            "convnextv2_base.fcmae_ft_in22k_in1k", device=device, target_size=224
        )
        cfg["feat_dim"] = model.feat_dim
        return model, cfg
    else:
        model = TimmReprModel(name, device=device, target_size=cfg["target_size"])
        cfg["feat_dim"] = model.feat_dim
        return model, cfg


# =============================================================================
# Dataset and DataLoader
# =============================================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def get_image_pairs(gt_dir, recon_dir):
    """Get paired image paths (matching filenames)."""
    gt_dir = Path(gt_dir)
    recon_dir = Path(recon_dir)

    gt_files = {}
    for f in sorted(gt_dir.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            gt_files[f.name] = f

    pairs = []
    for name, gt_path in sorted(gt_files.items()):
        recon_path = recon_dir / name
        if recon_path.exists():
            pairs.append((gt_path, recon_path, name))
        else:
            print(f"  Warning: no reconstruction found for {name}, skipping.")

    return pairs


class PairedImageDataset(Dataset):
    """Dataset that loads GT and reconstruction image pairs as [0,1] tensors."""

    def __init__(self, pairs):
        """pairs: list of (gt_path, recon_path, name)"""
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_path, recon_path, _ = self.pairs[idx]
        gt_img = Image.open(gt_path).convert("RGB")
        recon_img = Image.open(recon_path).convert("RGB")
        # Convert to [0, 1] float tensor (C, H, W)
        gt_tensor = torch.from_numpy(np.array(gt_img)).permute(2, 0, 1).float() / 255.0
        recon_tensor = (
            torch.from_numpy(np.array(recon_img)).permute(2, 0, 1).float() / 255.0
        )
        return gt_tensor, recon_tensor, idx


# =============================================================================
# Feature Extraction (Batched)
# =============================================================================


@torch.inference_mode()
def extract_features_batch(model, model_key, images):
    """Extract features from a batch of images.

    Args:
        model: the VFM
        model_key: one of the MODEL_CONFIGS keys
        images: (B, 3, H, W) tensor in [0, 1]

    Returns:
        features tensor of shape (B, feat_dim)
    """
    cfg = MODEL_CONFIGS[model_key]

    if model_key == "inception":
        # Inception does its own preprocessing (no autocast — uses custom TF resize)
        pool, _ = model(images)
        return pool
    else:
        # Use bf16 autocast for timm models (following FD-Loss)
        with torch.autocast(
            device_type=images.device.type,
            enabled=images.is_cuda,
            dtype=torch.bfloat16,
        ):
            cls_token, mean_token = model(images)
        if cfg["pool_type"] == "avg" and mean_token is not None:
            return mean_token.float()
        return cls_token.float()


# =============================================================================
# Metrics
# =============================================================================


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    return float(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    )


# =============================================================================
# Main Evaluation
# =============================================================================


@torch.inference_mode()
def evaluate_model(model_key, pairs, device="cuda", batch_size=64, num_workers=8):
    """Run full evaluation for one VFM with batched inference."""
    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {model_key.upper()}")
    print(f"  Model: {MODEL_CONFIGS[model_key]['name']}")
    print(f"{'=' * 60}")

    # Load model
    model, cfg = load_model(model_key, device=device)
    feat_dim = cfg["feat_dim"]

    if not pairs:
        print("  No image pairs found!")
        return None

    num_images = len(pairs)
    print(
        f"  Found {num_images} image pairs, batch_size={batch_size}, workers={num_workers}"
    )

    # Create DataLoader for parallel I/O
    dataset = PairedImageDataset(pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Online accumulators for FD (O(1) memory, following FD-Loss evaluator.py)
    gt_feat_sum = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    gt_feat_outer = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
    recon_feat_sum = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    recon_feat_outer = torch.zeros(
        feat_dim, feat_dim, dtype=torch.float64, device=device
    )

    # Per-image metrics storage
    all_mse = torch.empty(num_images)
    all_rel_mse = torch.empty(num_images)
    all_cosine = torch.empty(num_images)

    t0 = time.perf_counter()
    processed = 0

    for gt_batch, recon_batch, indices in dataloader:
        gt_batch = gt_batch.to(device, non_blocking=True)
        recon_batch = recon_batch.to(device, non_blocking=True)
        B = gt_batch.shape[0]

        # Batched feature extraction
        gt_feats = extract_features_batch(model, model_key, gt_batch)
        recon_feats = extract_features_batch(model, model_key, recon_batch)

        # Per-image MSE and cosine (vectorized over batch)
        mse_batch = (gt_feats - recon_feats).pow(2).mean(dim=1)  # (B,)
        normalizer = gt_feats.pow(2).mean(dim=1).clamp_min(float(1e-12))
        rel_mse_batch = mse_batch / normalizer

        cosine_batch = F.cosine_similarity(gt_feats, recon_feats, dim=1)  # (B,)

        all_mse[indices] = mse_batch.cpu()
        all_rel_mse[indices] = rel_mse_batch.cpu()
        all_cosine[indices] = cosine_batch.cpu()

        # Online accumulation for FD (sufficient statistics)
        gt_feats64 = gt_feats.double()
        gt_feat_sum.add_(gt_feats64.sum(0))
        gt_feat_outer.addmm_(gt_feats64.T, gt_feats64)

        recon_feats64 = recon_feats.double()
        recon_feat_sum.add_(recon_feats64.sum(0))
        recon_feat_outer.addmm_(recon_feats64.T, recon_feats64)

        processed += B
        if processed % (batch_size * 10) == 0 or processed == num_images:
            elapsed = time.perf_counter() - t0
            print(
                f"  Processed {processed}/{num_images} pairs "
                f"({elapsed:.1f}s, {processed / elapsed:.1f} img/s)"
            )

    # Compute FD from sufficient statistics
    N = num_images
    gt_s = gt_feat_sum.cpu().numpy()
    gt_S = gt_feat_outer.cpu().numpy()
    mu_gt = (gt_s / N).astype(np.float64)
    sigma_gt = ((gt_S - np.outer(gt_s, gt_s) / N) / (N - 1)).astype(np.float64)

    recon_s = recon_feat_sum.cpu().numpy()
    recon_S = recon_feat_outer.cpu().numpy()
    mu_recon = (recon_s / N).astype(np.float64)
    sigma_recon = ((recon_S - np.outer(recon_s, recon_s) / N) / (N - 1)).astype(
        np.float64
    )

    fd = compute_frechet_distance(mu_gt, sigma_gt, mu_recon, sigma_recon)

    # Averages
    avg_mse = all_mse.mean().item()
    avg_rel_mse = all_rel_mse.mean().item()
    avg_cosine = all_cosine.mean().item()

    elapsed = time.perf_counter() - t0
    print(f"  Avg MSE: {avg_mse:.6f}")
    print(f"  Avg RelMSE: {avg_rel_mse:.6f}")
    print(f"  Avg Cosine: {avg_cosine:.6f}")
    print(f"  FD: {fd:.4f}")
    print(f"  Total time: {elapsed:.1f}s ({num_images / elapsed:.1f} img/s)")

    # Build per-image results
    per_image_results = []
    for i, (_, _, name) in enumerate(pairs):
        per_image_results.append(
            {
                "name": name,
                "mse": all_mse[i].item(),
                "rel_mse": all_rel_mse[i].item(),
                "cosine": all_cosine[i].item(),
            }
        )

    return {
        "model_key": model_key,
        "model_name": cfg["name"],
        "per_image": per_image_results,
        "avg_mse": avg_mse,
        "avg_rel_mse": avg_rel_mse,
        "avg_cosine": avg_cosine,
        "fd": fd,
        "num_images": num_images,
        # Sufficient statistics for multi-seed FD merging
        "gt_feat_sum": gt_feat_sum.cpu(),
        "gt_feat_outer": gt_feat_outer.cpu(),
        "recon_feat_sum": recon_feat_sum.cpu(),
        "recon_feat_outer": recon_feat_outer.cpu(),
    }


def average_results(seed_results):
    """Average evaluation results across multiple seeds.

    MSE, RelMSE, and cosine: averaged directly across seeds.
    FD: computed from pooled sufficient statistics (all reconstructions treated
        as one distribution, GT stats use the first seed since GT is the same).
    """
    num_seeds = len(seed_results)
    ref = seed_results[0]

    # Average per-image MSE, RelMSE, and cosine
    avg_mse = np.mean([r["avg_mse"] for r in seed_results])
    avg_rel_mse = np.mean([r["avg_rel_mse"] for r in seed_results])
    avg_cosine = np.mean([r["avg_cosine"] for r in seed_results])

    # FD: pool recon sufficient statistics across all seeds
    # GT is the same across seeds, so just use the first seed's GT stats
    gt_feat_sum = ref["gt_feat_sum"]
    gt_feat_outer = ref["gt_feat_outer"]
    N_gt = ref["num_images"]

    # Recon: accumulate across all seeds (treating all seed reconstructions as separate samples)
    recon_feat_sum = torch.zeros_like(ref["recon_feat_sum"])
    recon_feat_outer = torch.zeros_like(ref["recon_feat_outer"])
    N_recon = 0
    for r in seed_results:
        recon_feat_sum.add_(r["recon_feat_sum"])
        recon_feat_outer.add_(r["recon_feat_outer"])
        N_recon += r["num_images"]

    # Compute mu and sigma from sufficient statistics
    gt_s = gt_feat_sum.numpy()
    gt_S = gt_feat_outer.numpy()
    mu_gt = (gt_s / N_gt).astype(np.float64)
    sigma_gt = ((gt_S - np.outer(gt_s, gt_s) / N_gt) / (N_gt - 1)).astype(np.float64)

    recon_s = recon_feat_sum.numpy()
    recon_S = recon_feat_outer.numpy()
    mu_recon = (recon_s / N_recon).astype(np.float64)
    sigma_recon = (
        (recon_S - np.outer(recon_s, recon_s) / N_recon) / (N_recon - 1)
    ).astype(np.float64)

    fd = compute_frechet_distance(mu_gt, sigma_gt, mu_recon, sigma_recon)

    print(
        f"\n  --- Averaged over {num_seeds} seeds (FD pooled from {N_recon} recon samples) ---"
    )
    print(f"  Avg MSE: {avg_mse:.6f}")
    print(f"  Avg RelMSE: {avg_rel_mse:.6f}")
    print(f"  Avg Cosine: {avg_cosine:.6f}")
    print(f"  FD: {fd:.4f}")

    # Average per-image results (assumes same image names in same order)
    per_image = []
    for i in range(ref["num_images"]):
        per_image.append(
            {
                "name": ref["per_image"][i]["name"],
                "mse": np.mean([r["per_image"][i]["mse"] for r in seed_results]),
                "rel_mse": np.mean(
                    [r["per_image"][i]["rel_mse"] for r in seed_results]
                ),
                "cosine": np.mean([r["per_image"][i]["cosine"] for r in seed_results]),
            }
        )

    return {
        "model_key": ref["model_key"],
        "model_name": ref["model_name"],
        "per_image": per_image,
        "avg_mse": avg_mse,
        "avg_rel_mse": avg_rel_mse,
        "avg_cosine": avg_cosine,
        "fd": fd,
        "num_images": ref["num_images"],
        "num_seeds": num_seeds,
    }


def _result_payload(results, include_per_image=False):
    models = {}
    for result in results:
        if result is None:
            continue
        item = {
            "model_name": result["model_name"],
            "num_images": result["num_images"],
            "num_seeds": result.get("num_seeds", 1),
            "mse": float(result["avg_mse"]),
            "rel_mse": float(result["avg_rel_mse"]),
            "cosine": float(result["avg_cosine"]),
            "fd": float(result["fd"]),
        }
        if include_per_image:
            item["per_image"] = result["per_image"]
        models[result["model_key"]] = item
    return {"models": models}


def write_results(results, output_path, print_per_image=False):
    """Write a machine-readable JSON file or a compact text report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        payload = _result_payload(results, include_per_image=print_per_image)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nResults written to: {output_path}")
        return

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Semantic Distance Evaluation Results\n" + "=" * 80 + "\n\n")
        for result in results:
            if result is None:
                continue
            handle.write(
                f"{result['model_key'].upper()} ({result['model_name']})\n"
                f"  image pairs: {result['num_images']}\n"
                f"  seeds: {result.get('num_seeds', 1)}\n"
                f"  MSE: {result['avg_mse']:.6f}\n"
                f"  RelMSE: {result['avg_rel_mse']:.6f}\n"
                f"  cosine: {result['avg_cosine']:.6f}\n"
                f"  FD: {result['fd']:.4f}\n\n"
            )
            if print_per_image:
                for item in result["per_image"]:
                    handle.write(
                        f"  {item['name']} mse={item['mse']:.6f} "
                        f"rel_mse={item['rel_mse']:.6f} cosine={item['cosine']:.6f}\n"
                    )
                handle.write("\n")
    print(f"\nResults written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute semantic distance metrics between GT and reconstructions"
    )
    parser.add_argument(
        "--gt-dir",
        dest="gt_dir",
        type=str,
        required=True,
        help="Directory with ground-truth images",
    )
    parser.add_argument(
        "--recon-dir",
        dest="recon_dir",
        type=str,
        nargs="+",
        required=True,
        help="Directory(ies) with reconstruction images. If multiple, results are averaged across seeds.",
    )
    parser.add_argument(
        "--output", type=str, default="results.json", help="Output .json or text report"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=8,
        help="DataLoader workers (default: 8)",
    )
    parser.add_argument(
        "--print-per-image",
        dest="print_per_image",
        action="store_true",
        default=False,
        help="Print per-image results in output file (default: False)",
    )

    # Per-model boolean flags (all True by default)
    parser.add_argument(
        "--inception",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate Inception V3 (default: True)",
    )
    parser.add_argument(
        "--dino",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate DINOv2 ViT-L/14 (default: True)",
    )
    parser.add_argument(
        "--siglip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate SigLIP ViT-SO400M/16 (default: True)",
    )
    parser.add_argument(
        "--clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate CLIP ViT-L/14 (default: True)",
    )
    parser.add_argument(
        "--convnext",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate ConvNeXtV2-B (default: True)",
    )

    args = parser.parse_args()

    # Collect enabled models
    all_model_keys = ["inception", "dino", "siglip", "clip", "convnext"]
    enabled_models = [k for k in all_model_keys if getattr(args, k)]

    if not enabled_models:
        print("Error: No models enabled. Enable at least one model.")
        sys.exit(1)

    # Validate directories
    if not os.path.isdir(args.gt_dir):
        print(f"Error: GT directory does not exist: {args.gt_dir}")
        sys.exit(1)
    for recon_dir in args.recon_dir:
        if not os.path.isdir(recon_dir):
            print(f"Error: Reconstruction directory does not exist: {recon_dir}")
            sys.exit(1)

    num_seeds = len(args.recon_dir)
    print(f"GT directory:      {args.gt_dir}")
    print(f"Recon directories: {args.recon_dir} ({num_seeds} seed(s))")
    print(f"Models:            {enabled_models}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Num workers:       {args.num_workers}")
    print(f"Print per-image:   {args.print_per_image}")
    print(f"Device:            {args.device}")

    # Run evaluation for each model, averaging across seeds
    all_results = []
    for model_key in enabled_models:
        seed_results = []
        for seed_idx, recon_dir in enumerate(args.recon_dir):
            if num_seeds > 1:
                print(f"\n  [Seed {seed_idx + 1}/{num_seeds}] recon_dir={recon_dir}")
            pairs = get_image_pairs(args.gt_dir, recon_dir)
            if not pairs:
                print(f"Error: No matching image pairs found in {recon_dir}!")
                sys.exit(1)
            result = evaluate_model(
                model_key,
                pairs,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            seed_results.append(result)

        # Average across seeds
        if num_seeds == 1:
            all_results.append(seed_results[0])
        else:
            all_results.append(average_results(seed_results))

        # Free GPU memory between models
        torch.cuda.empty_cache()

    # Write results
    write_results(all_results, args.output, print_per_image=args.print_per_image)


if __name__ == "__main__":
    main()
