
#!/usr/bin/env python3
"""Standalone inference for released RAE-CoD checkpoints.

The script restores the EMA denoiser, frozen RAEv2 decoder, DINOv3 encoder,
and 100-step shifted Euler sampler. It writes reconstructed PNG files and
reports the estimated main-latent, hyper-latent, and total rates per image.
"""

import argparse
import os

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

# Match the matrix-multiplication precision used during training.
torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------

def build_denoiser(resolution=256, finetune_mode="full",
                   pretrained_dit_path=None, use_aux_head=True,
                   use_y_embedder_x=False, vq_only=False):
    """Instantiate the RAE-CoD entropy-model denoiser.

    Args:
        resolution: Image resolution.
        finetune_mode: "lora" or "full". For inference we use "full" so that
            checkpoint loading can either inject the saved LoRA modules or
            merge them into a clean model.
        pretrained_dit_path: Path to RAEv2 DiT checkpoint. None means no
            pretrained loading at construction time (we load from ckpt instead).
        vq_only: Build the dedicated fixed 16-bit codec without a main-latent
            entropy model.
    """
    from src.models.rae_cod import (
        RAECoD,
    )
    cls = RAECoD

    input_size = resolution // 16  # 16 for 256, 32 for 512
    return cls(
        pred="x",
        input_size=input_size,
        in_channels=1024,
        patch_size=[1, 1],
        hidden_size=[1440, 2048],
        depth=[28, 2],
        num_heads=[20, 16],
        mlp_ratio=4.0,
        num_t_tokens=4,
        cond_dim=1152,
        pretrained_dit_path=pretrained_dit_path,
        lora_rank=32,
        lora_alpha=32,
        finetune_mode=finetune_mode,
        base_model_depth=8,
        up2x=False,
        use_aux_head=use_aux_head,
        use_y_embedder_x=use_y_embedder_x,
        vq_only=vq_only,
    )


def build_vae(
    resolution=256,
    decoder_path=None,
    stats_path=None,
):
    """Instantiate the base RAEDecoder.

    The official decoder initializes the architecture; when validation loads a
    jointly-trained checkpoint, its ``ema_vae`` weights replace these values.
    """
    from src.rae_decoder import RAEDecoder

    if decoder_path is None or stats_path is None:
        raise ValueError("decoder_path and stats_path are required")
    input_size = resolution // 16
    num_patches = input_size * input_size
    return RAEDecoder(
        decoder_config=dict(
            decoder_hidden_size=1152,
            decoder_num_hidden_layers=28,
            decoder_num_attention_heads=16,
            decoder_intermediate_size=4096,
        ),
        pretrained_decoder_path=decoder_path,
        normalization_stat_path=stats_path,
        encoder_hidden_size=1024,
        decoder_patch_size=16,
        image_size=resolution,
        num_patches=num_patches,
    )


def build_sampler(num_steps=100, ig_scale=1.78, ig_t_min=0.10, ig_t_max=1.0,
                  cfg_scale=1.0, cfg_t_min=0.0, cfg_t_max=1.0,
                  timeshift=8.0, t_eps=0.05, hyper_only=False):
    """Instantiate RAECoDSampler with default config."""
    from src.sampling import RAECoDSampler

    return RAECoDSampler(
        num_steps=num_steps,
        time_dist_shift=timeshift,
        t_eps=t_eps,
        prediction="x",
        ig_scale=ig_scale,
        ig_t_min=ig_t_min,
        ig_t_max=ig_t_max,
        cfg_scale=cfg_scale,
        cfg_t_min=cfg_t_min,
        cfg_t_max=cfg_t_max,
        hyper_only=hyper_only,
    )


def build_vfm_encoder(ckpt_dir=None, repo_dir=None):
    """Instantiate frozen DINOv3 multi-layer encoder."""
    from src.models.dinov3 import DINOv3MultiLayer

    if ckpt_dir is None or repo_dir is None:
        raise ValueError("ckpt_dir and repo_dir are required")
    return DINOv3MultiLayer(
        ckpt_dir=ckpt_dir,
        repo_dir=repo_dir,
        model_name="dinov3_vitl16",
        layer_indices=[11, 13, 15, 17, 19, 21, 23],
    )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_sharded_ckpt(ckpt_path):
    """Load a Lightning checkpoint, handling both single-file and sharded formats.

    Sharded checkpoints (DDP split) are directories containing separate .pt files
    for each checkpoint component (state_dict, global_step, epoch, etc.).
    Single-file checkpoints are standard .ckpt files loaded directly.

    Returns:
        dict with keys: "state_dict", "global_step", "epoch".
    """
    if os.path.isdir(ckpt_path):
        # Sharded checkpoint directory
        sd_path = os.path.join(ckpt_path, "checkpoint-state_dict.pt")
        gs_path = os.path.join(ckpt_path, "checkpoint-global_step.pt")
        ep_path = os.path.join(ckpt_path, "checkpoint-epoch.pt")

        sd_file = torch.load(
            sd_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        state_dict = sd_file.get("state_dict", sd_file)

        global_step = -1
        if os.path.exists(gs_path):
            gs_file = torch.load(gs_path, map_location="cpu", weights_only=False)
            global_step = gs_file.get("global_step", -1) if isinstance(gs_file, dict) else gs_file

        epoch = -1
        if os.path.exists(ep_path):
            ep_file = torch.load(ep_path, map_location="cpu", weights_only=False)
            epoch = ep_file.get("epoch", -1) if isinstance(ep_file, dict) else ep_file

        return {"state_dict": state_dict, "global_step": global_step, "epoch": epoch}
    else:
        # Single-file checkpoint
        return torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )


def detect_checkpoint_codec_mode(ckpt_path):
    """Return ``standard`` or ``vq`` from the checkpoint's codec keys."""
    checkpoint = _load_sharded_ckpt(ckpt_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    codec_keys = [
        key
        for key in state_dict
        if key.startswith(("denoiser.y_embedder.", "ema_denoiser.y_embedder."))
    ]
    if not codec_keys:
        raise RuntimeError(f"Checkpoint has no RAE-CoD codec keys: {ckpt_path}")
    has_main_context = any(
        ".y_in." in key or ".y_cm." in key or ".y_out." in key
        for key in codec_keys
    )
    has_vq = any(".entropy_bottleneck.embedding.weight" in key for key in codec_keys)
    if not has_vq:
        raise RuntimeError(f"Checkpoint has no VQ codebook: {ckpt_path}")
    return "standard" if has_main_context else "vq"


def _extract_prefixed_state(state_dict, prefix):
    """Extract a component state dict and remove its Lightning prefix."""
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _load_component_state(
    module,
    state_dict,
    prefix,
    component_name,
    preserve_lora=False,
    lora_injector=None,
):
    """Load one checkpoint component, preserving or merging LoRA adapters.

    Returns ``False`` when the checkpoint has no keys for ``prefix``; otherwise
    loads all shape-compatible tensors and returns ``True``.
    """
    from src.models.lora import merge_lora_state_dict

    model_state = _extract_prefixed_state(state_dict, prefix)
    if not model_state:
        return False

    has_lora = any(key.endswith(".lora_A.weight") for key in model_state)
    if has_lora:
        if preserve_lora:
            target_has_lora = any(
                key.endswith(".lora_A.weight") for key in module.state_dict()
            )
            if not target_has_lora:
                if lora_injector is None:
                    raise RuntimeError(
                        f"Cannot preserve LoRA for {component_name}: no LoRA "
                        "injector was provided."
                    )
                lora_injector(module)
                target_has_lora = any(
                    key.endswith(".lora_A.weight") for key in module.state_dict()
                )
            if not target_has_lora:
                raise RuntimeError(
                    f"Failed to construct LoRA layers for {component_name}."
                )
            print(
                f"  Detected LoRA {component_name} checkpoint — preserving "
                "LoRA modules to match training validation."
            )
        else:
            print(f"  Detected LoRA {component_name} checkpoint — merging LoRA weights...")
            model_state = merge_lora_state_dict(
                model_state, lora_alpha=32, lora_rank=32,
            )
            print(f"  Merged. {component_name} state dict now has {len(model_state)} keys.")

    own_state = module.state_dict()
    compatible_state = {}
    skipped = 0
    for key, value in model_state.items():
        if key in own_state and own_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped += 1
            if key in own_state:
                print(
                    f"  Shape mismatch, skipped {component_name} key: {key} "
                    f"(ckpt={value.shape} vs model={own_state[key].shape})"
                )
            else:
                print(f"  Missing in {component_name} model, skipped: {key}")

    if not compatible_state:
        raise RuntimeError(
            f"Found keys with prefix '{prefix}', but none are compatible with "
            f"the constructed {component_name} model."
        )

    missing = sorted(set(own_state) - set(compatible_state))
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            f"Checkpoint is incomplete for the constructed {component_name}: "
            f"{len(missing)} model tensors are missing ({preview}). Check that "
            "the standard/vq codec architecture matches the checkpoint."
        )

    # Architecture coverage is checked above; strict=False only avoids a
    # redundant full-state copy while loading the compatible tensor mapping.
    module.load_state_dict(compatible_state, strict=False)
    print(
        f"  Loaded {len(compatible_state)} {component_name} tensors from "
        f"'{prefix}', skipped {skipped}"
    )
    return True


def load_checkpoint(
    denoiser,
    ckpt_path,
    use_ema=True,
    vae=None,
    use_ema_vae=True,
    preserve_lora=False,
):
    """Load denoiser and, when supplied, RAEv2 decoder checkpoint weights.

    Handles both LoRA and full fine-tuning checkpoints. LoRA weights can either
    remain as separate residual modules (matching training validation) or be
    merged into plain inference modules.

    Supports both single-file .ckpt and sharded (directory) checkpoints
    produced by Lightning DDP.

    Args:
        denoiser: The model to load weights into.
        ckpt_path: Path to the .ckpt file or sharded checkpoint directory.
        use_ema: If True, load ema_denoiser weights; otherwise load denoiser weights.
        vae: Optional RAEDecoder to restore from the same checkpoint.
        use_ema_vae: If True, prefer ema_vae weights; otherwise prefer vae weights.
        preserve_lora: Keep LoRA modules separate instead of merging them.

    Returns:
        (global_step, epoch) from the checkpoint metadata.
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = _load_sharded_ckpt(ckpt_path)

    state_dict = ckpt.get("state_dict", ckpt)
    prefix = "ema_denoiser." if use_ema else "denoiser."

    def inject_denoiser_lora(module):
        from src.models.lora import apply_lora_to_dit
        module._lora_params = apply_lora_to_dit(
            module.dit, rank=32, alpha=32,
        )

    if not _load_component_state(
        denoiser,
        state_dict,
        prefix,
        "denoiser",
        preserve_lora=preserve_lora,
        lora_injector=inject_denoiser_lora,
    ):
        if not use_ema and _extract_prefixed_state(state_dict, "ema_denoiser."):
            raise RuntimeError(
                "This is an EMA-only inference checkpoint; remove --use-raw-model."
            )
        raise RuntimeError(
            f"No keys with prefix '{prefix}' found in checkpoint. "
            f"Available prefixes: {sorted(set(k.split('.')[0] for k in state_dict))}"
        )

    if vae is not None:
        def inject_decoder_lora(module):
            from src.models.lora import apply_lora_to_decoder
            apply_lora_to_decoder(module.decoder, rank=32, alpha=32)

        vae_prefix = "ema_vae." if use_ema_vae else "vae."
        loaded_vae = _load_component_state(
            vae,
            state_dict,
            vae_prefix,
            "RAEv2 decoder",
            preserve_lora=preserve_lora,
            lora_injector=inject_decoder_lora,
        )

        # Some older fine-tuning checkpoints may contain only one decoder copy.
        # Prefer the requested copy, but use the other trained copy rather than
        # silently falling all the way back to the official decoder.
        if not loaded_vae:
            fallback_prefix = "vae." if use_ema_vae else "ema_vae."
            if _extract_prefixed_state(state_dict, fallback_prefix):
                print(
                    f"  WARNING: No '{vae_prefix}' weights found; falling back "
                    f"to trained decoder weights under '{fallback_prefix}'."
                )
                loaded_vae = _load_component_state(
                    vae,
                    state_dict,
                    fallback_prefix,
                    "RAEv2 decoder",
                    preserve_lora=preserve_lora,
                    lora_injector=inject_decoder_lora,
                )

        if not loaded_vae:
            print(
                f"  WARNING: Checkpoint has no '{vae_prefix}' or alternate "
                "decoder weights. Keeping the decoder initialized from "
                "--decoder_path (expected for checkpoints trained with a frozen "
                "decoder)."
            )

    global_step = ckpt.get("global_step", -1)
    epoch = ckpt.get("epoch", -1)
    print(f"  Checkpoint step={global_step}, epoch={epoch}")
    return global_step, epoch


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path, resolution=None):
    """Load an image as (3, H, W) tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    if resolution is not None:
        img = img.resize((resolution, resolution), Image.LANCZOS)
    return to_tensor(img)


def make_noise_generator(seed):
    """Create one persistent CPU generator, or use PyTorch's global RNG."""
    if seed is None:
        return None
    return torch.Generator().manual_seed(seed)


def sample_latent_noise(batch_size, latent_shape, generator=None):
    """Draw independent latent noise while advancing the supplied generator."""
    return torch.randn(
        (batch_size, *latent_shape),
        generator=generator,
        dtype=torch.float32,
    )


def fp2uint8(x):
    """Convert [-1, 1] float tensor to [0, 255] uint8."""
    return torch.clip_((x + 1) * 127.5 + 0.5, 0, 255).to(torch.uint8)


def save_image(tensor, path):
    """Save a (3, H, W) uint8 tensor as an image file."""
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(args):
    device = torch.device(args.device)

    # ---- Detect checkpoint architecture and build models ----
    codec_mode = detect_checkpoint_codec_mode(args.ckpt)
    if args.hyper_only and codec_mode != "vq":
        raise ValueError(
            "--hyper-only requires the dedicated 16-bit VQ checkpoint; the "
            "provided checkpoint contains the standard main-latent codec."
        )
    vq_only = codec_mode == "vq"
    print(f"Building models (codec_mode={codec_mode})...")
    denoiser = build_denoiser(
        resolution=args.resolution,
        finetune_mode="full",  # always build without LoRA for inference
        pretrained_dit_path=None,  # don't load pretrained DiT; we load from ckpt
        vq_only=vq_only,
    )
    vae = build_vae(
        resolution=args.resolution,
        decoder_path=args.decoder_path,
        stats_path=args.stats_path,
    )
    sampler = build_sampler(
        num_steps=args.num_steps,
        ig_scale=args.ig_scale,
        ig_t_min=args.ig_t_min,
        ig_t_max=args.ig_t_max,
        cfg_scale=args.cfg_scale,
        cfg_t_min=args.cfg_t_min,
        cfg_t_max=args.cfg_t_max,
        timeshift=args.timeshift,
        t_eps=args.t_eps,
        hyper_only=vq_only,
    )
    vfm_encoder = build_vfm_encoder(
        ckpt_dir=args.dinov3_ckpt_dir,
        repo_dir=args.dinov3_repo_dir,
    )

    # ---- Load checkpoint ----
    load_checkpoint(
        denoiser,
        args.ckpt,
        use_ema=not args.use_raw_model,
        vae=vae,
        # Training validation always decodes with ema_vae, independently of
        # whether the raw or EMA denoiser is selected.
        use_ema_vae=True,
        preserve_lora=not getattr(args, "merge_lora", False),
    )

    # ---- Move to device & eval mode ----
    denoiser.requires_grad_(False)
    vae.requires_grad_(False)
    denoiser = denoiser.to(device).to(torch.float32).eval()
    vae = vae.to(device).to(torch.float32).eval()
    sampler = sampler.to(device).eval()
    vfm_encoder = vfm_encoder.to(device).eval()

    # ---- Collect eval images ----
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    # ImageTestDataset uses os.listdir() without sorting. Preserve that order so
    # batch composition also matches single-rank training validation.
    image_names = [
        f for f in os.listdir(args.image_dir)
        if os.path.splitext(f)[1].lower() in supported_ext
    ]
    if args.max_images > 0:
        image_names = image_names[:args.max_images]
    print(f"Found {len(image_names)} images in {args.image_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.save_comparison:
        os.makedirs(os.path.join(args.output_dir, "comparison"), exist_ok=True)

    # ---- Latent shape ----
    latent_h = args.resolution // 16
    latent_shape = (1024, latent_h, latent_h)

    # ---- Run validation ----
    total_bpp, total_y_bpp, total_z_bpp = 0.0, 0.0, 0.0
    num_processed = 0
    per_image_bpp, per_image_y_bpp, per_image_z_bpp = {}, {}, {}  # name -> bpp
    batch_size = args.batch_size
    noise_generator = make_noise_generator(args.noise_seed)

    for batch_start in range(0, len(image_names), batch_size):
        batch_names = image_names[batch_start:batch_start + batch_size]
        B = len(batch_names)

        # Load and stack images
        images = []
        for name in batch_names:
            resize_to = args.resolution if getattr(args, "resize_images", False) else None
            img = load_image(
                os.path.join(args.image_dir, name),
                resolution=resize_to,
            )
            expected_hw = (args.resolution, args.resolution)
            if img.shape[-2:] != expected_hw:
                raise ValueError(
                    f"{name} has spatial size {tuple(img.shape[-2:])}, expected "
                    f"{expected_hw}. Training validation does not resize images; "
                    "pass --resize_images only if intentional."
                )
            images.append(img)
        y = torch.stack(images, dim=0).to(device)  # (B, 3, H, W) in [0, 1]

        xT = sample_latent_noise(B, latent_shape, noise_generator).to(device)

        # VFM features from raw images
        vfm_features = vfm_encoder(y)

        # assert y.shape[0] == 1
        assert y.shape[1] == 3

        # Diffusion sampling (sampler runs codec internally)
        samples, batch_bpp, batch_y_bpp, batch_z_bpp = sampler(
            denoiser, xT, y, vfm_features=vfm_features,
        )

        # Decode latent predictions → pixels via RAEv2 decoder
        pixels = vae.decode(samples)  # (B, 3, H, W) in [-1, 1]
        outputs = fp2uint8(pixels)    # (B, 3, H, W) uint8

        # Per-image BPP is normally a batch vector; retain scalar compatibility.
        if batch_bpp.dim() == 0:
            bpp_per_sample, y_bpp_per_sample, z_bpp_per_sample = batch_bpp.item(), batch_y_bpp.item(), batch_z_bpp.item()
            bpp_values = [bpp_per_sample] * B
            y_bpp_values = [y_bpp_per_sample] * B
            z_bpp_values = [z_bpp_per_sample] * B
        else:
            bpp_values = batch_bpp.tolist()
            y_bpp_values = batch_y_bpp.tolist()
            z_bpp_values = batch_z_bpp.tolist()

        # Save images and log per-image BPP
        for i, name in enumerate(batch_names):
            bpp_i, y_bpp_i, z_bpp_i = bpp_values[i], y_bpp_values[i], z_bpp_values[i]
            per_image_bpp[name] = bpp_i
            per_image_y_bpp[name] = y_bpp_i
            per_image_z_bpp[name] = z_bpp_i
            total_bpp += bpp_i
            total_y_bpp += y_bpp_i
            total_z_bpp += z_bpp_i

            stem = os.path.splitext(name)[0]
            out_path = os.path.join(args.output_dir, f"{stem}.png")
            save_image(outputs[i], out_path)

            if args.save_comparison:
                orig_uint8 = (y[i].clamp(0, 1) * 255).to(torch.uint8)
                comp = torch.cat([orig_uint8, outputs[i]], dim=2)  # concat width
                comp_path = os.path.join(args.output_dir, "comparison", f"{stem}.png")
                save_image(comp, comp_path)

        num_processed += B
        print(f"  [{num_processed}/{len(image_names)}] batch_avg_bpp={sum(bpp_values)/B:.8f}")

    # Print per-image BPP summary
    print(f"\n{'='*50}")
    print(f"{'Image':<40} {'BPP':>8} {'Y_BPP':>8} {'Z_BPP':>8}")
    print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    for name, bpp in per_image_bpp.items():
        y_bpp = per_image_y_bpp[name]
        z_bpp = per_image_z_bpp[name]
        print(f"{name:<40} {bpp:>8.8f} {y_bpp:>8.8f} {z_bpp:>8.8f}")
    print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*8}")

    avg_bpp = total_bpp / max(num_processed, 1)
    avg_y_bpp = total_y_bpp / max(num_processed, 1)
    avg_z_bpp = total_z_bpp / max(num_processed, 1)
    print(f"{'Average':<40} {avg_bpp:>8.8f} {avg_y_bpp:>8.8f} {avg_z_bpp:>8.8f}")
    print(f"{'='*50}")
    print(f"\nDone. Processed {num_processed} images.")
    print(f"Outputs saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone inference for RAE-CoD checkpoints",
    )

    # Required
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a Lightning .ckpt file or sharded checkpoint directory")
    parser.add_argument("--image-dir", dest="image_dir", type=str, required=True,
                        help="Directory of 256x256 input images")
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True,
                        help="Directory for reconstructed PNG files")

    # Model / resolution
    parser.add_argument("--resolution", type=int, default=256, choices=[256],
                        help="Released models operate at 256x256")
    parser.add_argument("--use-raw-model", dest="use_raw_model", action="store_true",
                        help="Use raw denoiser weights instead of EMA weights")
    parser.add_argument(
        "--merge-lora", action=argparse.BooleanOptionalAction, default=True,
        help="Merge LoRA weights for deployment (default: true)",
    )

    # Sampling — RAECoDSampler parameters
    parser.add_argument("--num-steps", dest="num_steps", type=int, default=100,
                        help="ODE sampling steps")
    parser.add_argument("--ig-scale", dest="ig_scale", type=float, default=1.78,
                        help="Internal Guidance scale (1.0 = disabled)")
    parser.add_argument("--ig-t-min", dest="ig_t_min", type=float, default=0.10,
                        help="IG active interval lower bound")
    parser.add_argument("--ig-t-max", dest="ig_t_max", type=float, default=1.0,
                        help="IG active interval upper bound")
    parser.add_argument("--cfg-scale", dest="cfg_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (1.0 = disabled)")
    parser.add_argument("--cfg-t-min", dest="cfg_t_min", type=float, default=0.0,
                        help="CFG active interval lower bound")
    parser.add_argument("--cfg-t-max", dest="cfg_t_max", type=float, default=1.0,
                        help="CFG active interval upper bound")
    parser.add_argument("--timeshift", type=float, default=8.0,
                        help="Time distribution shift")
    parser.add_argument("--t-eps", dest="t_eps", type=float, default=0.05,
                        help="Minimum timestep epsilon for velocity conversion")
    parser.add_argument(
        "--noise-seed",
        dest="noise_seed",
        type=int,
        default=None,
        help=(
            "Optional latent-noise seed. By default noise is unseeded; when set, "
            "one generator is reused and advanced across all images."
        ),
    )
    parser.add_argument(
        "--hyper-only",
        action="store_true",
        help=(
            "Require the dedicated VQ-only 16-bit checkpoint. Codec structure "
            "is auto-detected even when this flag is omitted."
        ),
    )

    # Paths
    parser.add_argument("--decoder-path", dest="decoder_path", type=str, required=True,
                        help="Path to base RAEv2 decoder weights (overridden by "
                             "ema_vae weights when present in --ckpt)")
    parser.add_argument("--stats-path", dest="stats_path", type=str, required=True,
                        help="Path to normalization stats")
    parser.add_argument("--dinov3-ckpt-dir", dest="dinov3_ckpt_dir", type=str, required=True,
                        help="Directory containing DINOv3 checkpoint")
    parser.add_argument("--dinov3-repo-dir", dest="dinov3_repo_dir", type=str, required=True,
                        help="Local DINOv3 repo for torch.hub (must match training)")

    # Output
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1,
                        help="Local validation batch size; matching the training "
                             "rank's batch composition minimizes BF16 kernel differences")
    parser.add_argument("--max-images", dest="max_images", type=int, default=0,
                        help="Max images to process (0 = all)")
    parser.add_argument("--save-comparison", dest="save_comparison", action="store_true",
                        help="Save side-by-side original|reconstruction")
    parser.add_argument("--resize-images", dest="resize_images", action="store_true",
                        help="Resize inputs to --resolution. Disabled by default "
                             "because training validation reads images unchanged")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
