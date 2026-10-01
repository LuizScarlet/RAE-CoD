"""DeCo denoiser wrapping a pretrained RAEv2 DiTwDDTHeadIG with LoRA or full fine-tuning.

Loads a pretrained RAEv2 DiT checkpoint (including base_final_layer for IG).
Supports three fine-tuning modes via ``finetune_mode``:
  - ``"lora"`` (default): freezes base weights and injects trainable LoRA adapters
    into attention, feed-forward, adaptive-normalization, interface/projector,
    and output-head projections.
  - ``"full"``: unfreezes all DiT parameters for full fine-tuning.
  - ``"frozen"``: freezes all DiT parameters; no LoRA injected. Useful when only
    the codec is being fine-tuned.

The codec is trained from scratch in all modes (except when codec is also frozen).

Uses RAEv2's native time convention (t=0→clean, t=1→noise) with Internal
Guidance (IG) support. The model forward returns (x_full, x_base) always.
"""
import logging
import torch
import torch.nn as nn
from types import SimpleNamespace

from src.models.codec import RAECodec
from src.models.ddt import DiTwDDTHeadIG, RoPE
from src.models.lora import apply_lora_to_dit


class RAECoDBase(nn.Module):
    """RAE-CoD base containing the learned compression codec."""

    def __init__(self, use_aux_head=False, vq_only=False, *args, **kwargs):
        super().__init__()
        self.y_embedder = RAECodec(
            use_aux_head=use_aux_head,
            vq_only=vq_only,
        )

logger = logging.getLogger(__name__)


def _init_deco(self, input_size, in_channels, patch_size, hidden_size, depth,
               num_heads, mlp_ratio, num_t_tokens, cond_dim, pretrained_dit_path,
               lora_rank, lora_alpha, finetune_mode, base_model_depth,
               use_y_embedder_x=False):
    """Initialize the codec-conditioned RAEv2 DDT."""
    if patch_size is None:
        patch_size = [1, 1]
    if hidden_size is None:
        hidden_size = [1440, 2048]
    if depth is None:
        depth = [28, 2]
    if num_heads is None:
        num_heads = [20, 16]

    assert finetune_mode in ("lora", "full", "frozen"), f"Unknown finetune_mode: {finetune_mode}"
    self.finetune_mode = finetune_mode

    # Number of codec condition tokens = spatial patches from codec output
    num_codec_patches = (input_size // patch_size[0]) ** 2

    # Instantiate the DiTwDDTHeadIG (with IG, matching RAEv2 pretraining)
    cond_arch = SimpleNamespace(
        num_t_tokens=num_t_tokens,
        num_c_tokens=num_codec_patches,
    )
    self.dit = DiTwDDTHeadIG(
        base_model_depth=base_model_depth,
        input_size=input_size,
        in_channels=in_channels,
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        cond_arch=cond_arch,
        condition_type="label",  # placeholder; ctx_embedder is replaced below
        use_cfg_conds=False,
    )

    # Replace label/text ConditionEmbedder with identity (or linear projection)
    # so that codec condition tokens flow through _build_sequence unchanged.
    enc_hidden_size = hidden_size[0]
    if cond_dim != enc_hidden_size:
        self.dit.ctx_embedder = nn.Linear(cond_dim, enc_hidden_size)
    else:
        self.dit.ctx_embedder = nn.Identity()

    # Codec condition injection into decoder via x_embedder residual (zero-init)
    if use_y_embedder_x:
        dec_hidden_size = hidden_size[1]
        self.dit.y_embedder_x = nn.Linear(cond_dim, dec_hidden_size)
        nn.init.zeros_(self.dit.y_embedder_x.weight)
        nn.init.zeros_(self.dit.y_embedder_x.bias)

    # Store head dims for dynamic RoPE recreation at different resolutions
    self._enc_head_dim = hidden_size[0] // num_heads[0]
    self._dec_head_dim = hidden_size[1] // num_heads[1]
    self._num_t_tokens = num_t_tokens

    # Load pretrained RAEv2 DiT checkpoint and optionally inject LoRA
    self._lora_params = []
    if pretrained_dit_path is not None:
        _load_pretrained_dit(self, path=pretrained_dit_path)
        if finetune_mode == "lora":
            self._lora_params = apply_lora_to_dit(self.dit, rank=lora_rank, alpha=lora_alpha)
            logger.info(
                "LoRA injected: %d trainable LoRA parameters (rank=%d, alpha=%d)",
                sum(p.numel() for p in self._lora_params), lora_rank, lora_alpha,
            )
        elif finetune_mode == "full":
            logger.info("Full DiT fine-tuning mode: all %d DiT parameters are trainable",
                        sum(p.numel() for p in self.dit.parameters()))
        else:
            logger.info("Frozen DiT mode: no LoRA injected, all DiT parameters frozen")


def _load_pretrained_dit(self, path):
    """Load pretrained RAEv2 DiT weights.

    When finetune_mode="lora": freezes all DiT (including base_final_layer),
        then unfreezes only ctx_embedder (codec→DiT projection).
        LoRA adapters are injected separately after this call.
    When finetune_mode="full": unfreezes all DiT parameters.
    """
    logger.info("Loading pretrained DiT from %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Use EMA weights if available
    if "ema" in ckpt:
        sd = ckpt["ema"]
        logger.info("Using EMA weights from checkpoint")
    elif "model" in ckpt:
        sd = ckpt["model"]
        logger.info("Using model weights from checkpoint")
    else:
        sd = ckpt
        logger.info("Using raw checkpoint dict as state dict")

    # Filter out keys that differ between pretrained (label-conditioned) and our (codec-conditioned) model:
    # - ctx_embedder.*: replaced with linear projection from codec dim
    # - enc_rope.*: RoPE buffer size differs (num_cond_tokens changed)
    # - y_embedder_x.*: new zero-init layer, not in pretrained checkpoint
    # - repa_projector.*: unused REPA interface from RAEv2 (if present in checkpoint)
    skip_prefixes = ("ctx_embedder.", "enc_rope.", "y_embedder_x.", "repa_projector.")
    filtered_sd = {}
    skipped_keys = []
    for k, v in sd.items():
        if any(k.startswith(p) for p in skip_prefixes):
            skipped_keys.append(k)
        else:
            filtered_sd[k] = v

    if skipped_keys:
        logger.info("Skipped %d keys from checkpoint: %s", len(skipped_keys), skipped_keys)

    # Load into self.dit with strict=False (ctx_embedder/enc_rope are new/resized)
    missing, unexpected = self.dit.load_state_dict(filtered_sd, strict=False)
    if missing:
        logger.info("Missing keys in DiT (expected for ctx_embedder/enc_rope): %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in DiT checkpoint: %s", unexpected)

    if self.finetune_mode == "full":
        # Full fine-tuning: all DiT parameters are trainable
        for p in self.dit.parameters():
            p.requires_grad = True
        logger.info("Full fine-tuning: unfroze all %d DiT parameters",
                    sum(p.numel() for p in self.dit.parameters()))
    elif self.finetune_mode == "frozen":
        # Frozen: all DiT parameters are frozen, no LoRA
        for p in self.dit.parameters():
            p.requires_grad = False
        logger.info("Frozen DiT: all %d DiT parameters are frozen (no LoRA)",
                    sum(p.numel() for p in self.dit.parameters()))
    else:
        # LoRA mode: freeze ALL DiT parameters (including base_final_layer)
        for p in self.dit.parameters():
            p.requires_grad = False
        logger.info("Froze all DiT base parameters (including base_final_layer)")

        # Unfreeze ctx_embedder (codec→DiT interface, trained from scratch)
        for p in self.dit.ctx_embedder.parameters():
            p.requires_grad = True
        logger.info("Unfroze ctx_embedder (%d params)",
                     sum(p.numel() for p in self.dit.ctx_embedder.parameters()))

        if hasattr(self.dit, 'y_embedder_x'):
            for p in self.dit.y_embedder_x.parameters():
                p.requires_grad = True
            logger.info("Unfroze y_embedder_x (%d params)",
                         sum(p.numel() for p in self.dit.y_embedder_x.parameters()))


def _update_resolution(self, H, W):
    """Re-create DiT's RoPE and patch counts when input resolution changes."""
    s_ps = self.dit.s_patch_size
    x_ps = self.dit.x_patch_size
    new_s_patches = (H // s_ps) * (W // s_ps)
    new_x_patches = (H // x_ps) * (W // x_ps)

    if new_s_patches == self.dit.s_embedder.num_patches:
        return  # resolution unchanged, nothing to do

    # Codec condition tokens scale with vision patches (same spatial grid)
    new_cond_tokens = self._num_t_tokens + new_s_patches

    device = self.dit.enc_rope.freqs_cos.device
    self.dit.enc_rope = RoPE(self._enc_head_dim, new_s_patches, new_cond_tokens).to(device)
    self.dit.dec_rope = RoPE(self._dec_head_dim, new_x_patches).to(device)
    self.dit.s_embedder.num_patches = new_s_patches
    self.dit.x_embedder.num_patches = new_x_patches
    self.dit.num_cond_tokens = new_cond_tokens


class RAECoD(RAECoDBase):
    """Codec-conditioned denoiser with pretrained DiTwDDTHeadIG backbone.

    Architecture:
        self.y_embedder  — RAECodec (trained from scratch)
        self.dit         — DiTwDDTHeadIG (pretrained, LoRA or full fine-tuning)

    Fine-tuning modes (``finetune_mode``):
        "lora":   Freeze the base DiT and adapt attention, FFN, adaptive-
                  normalization, interface, and output projections (default).
        "full":   Unfreeze all DiT parameters for full fine-tuning.
        "frozen": Freeze all DiT parameters; no LoRA. Useful when only the codec
                  is being fine-tuned.

    Uses RAEv2 time convention natively (t=0→clean, t=1→noise).
    Forward returns (x_full, x_base) tuple always (for Internal Guidance).
    """

    def __init__(
        self,
        input_size=16,
        in_channels=1024,
        patch_size=None,
        hidden_size=None,
        depth=None,
        num_heads=None,
        mlp_ratio=4.0,
        num_t_tokens=4,
        cond_dim=1152,
        pretrained_dit_path=None,
        lora_rank=32,
        lora_alpha=32,
        finetune_mode="lora",
        base_model_depth=8,
        use_y_embedder_x=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)  # creates self.y_embedder (codec)
        _init_deco(self, input_size, in_channels, patch_size, hidden_size, depth,
                   num_heads, mlp_ratio, num_t_tokens, cond_dim, pretrained_dit_path,
                   lora_rank, lora_alpha, finetune_mode, base_model_depth,
                   use_y_embedder_x)

    @property
    def lora_params(self):
        return self._lora_params

    @property
    def blocks(self):
        return self.dit.blocks

    def _update_resolution(self, H, W):
        _update_resolution(self, H, W)

    def forward(self, x, t, **condition_kwargs):
        """Forward pass using RAEv2 time convention natively.

        Args:
            x: Noisy latent (B, in_channels, H, W).
            t: Timestep (B,) in RAEv2 convention (t=0→clean, t=1→noise).
            **condition_kwargs: Must contain 'context' key with codec condition tokens.

        Returns:
            (x_full, x_base): Tuple of full and base (IG early exit) predictions.
        """
        _, _, H, W = x.shape
        self._update_resolution(H, W)
        return self.dit(x, t, **condition_kwargs)

