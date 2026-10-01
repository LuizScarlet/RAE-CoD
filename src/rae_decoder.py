"""RAEv2 decoder as a RAE-CoD VAE interface (BaseAE subclass).

Self-contained: embeds GeneralDecoder, ViTMAEConfig, and support classes
from RAEv2 so there is no runtime dependency on the RAEv2 codebase.

decode() denormalizes DINOv3 features, applies GeneralDecoder, and
unpatchifies the result into pixels.
"""

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from src.models.autoencoder.base import BaseAE


# ============================================================================
# Vendored ViTMAE components from RAEv2 (no transformers dependency)
# ============================================================================

ACT2FN = {
    "gelu": nn.functional.gelu,
    "relu": nn.functional.relu,
    "silu": nn.functional.silu,
    "tanh": torch.tanh,
}


class ViTMAEConfig:
    """Minimal ViT-MAE config matching RAEv2's decoder requirements."""

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=256,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=1152,
        decoder_num_hidden_layers=28,
        decoder_intermediate_size=4096,
        chunk_size_feed_forward=0,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.chunk_size_feed_forward = chunk_size_feed_forward


# ---------- positional embeddings ----------

def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    if add_cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


# ---------- transformer blocks ----------

class _SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        probs = nn.functional.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size()[:-2] + (self.all_head_size,))
        return (ctx, probs) if output_attentions else (ctx,)


class _SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        return self.dropout(self.dense(hidden_states))


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = _SelfAttention(config)
        self.output = _SelfOutput(config)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]


class _Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.intermediate_act_fn(self.dense(hidden_states))


class _Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        return self.dropout(self.dense(hidden_states)) + input_tensor


class _Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = _Attention(config)
        self.intermediate = _Intermediate(config)
        self.output = _Output(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_attn_out = self.attention(self.layernorm_before(hidden_states), head_mask, output_attentions)
        attention_output = self_attn_out[0]
        hidden_states = attention_output + hidden_states
        layer_output = self.intermediate(self.layernorm_after(hidden_states))
        layer_output = self.output(layer_output, hidden_states)
        return (layer_output,) + self_attn_out[1:]


# ---------- GeneralDecoder ----------

class GeneralDecoder(nn.Module):
    """Vendored from RAEv2 src/stage1/decoders/decoder.py."""

    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [_Layer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )
        self.gradient_checkpointing = False
        self.config = config
        self.num_patches = num_patches
        self._initialize_weights(num_patches)
        self.decoder_config = decoder_config
        self._set_trainable_cls_token()

    def _set_trainable_cls_token(self, tensor=None):
        tensor = torch.zeros(1, 1, self.decoder_config.hidden_size) if tensor is None else tensor
        self.trainable_cls_token = nn.Parameter(tensor)

    def _initialize_weights(self, num_patches):
        pos_embed = _get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def interpolate_latent(self, x):
        b, token_count, c = x.shape
        if token_count == self.num_patches:
            return x
        h = w = int(token_count**0.5)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        target_size = (int(self.num_patches**0.5), int(self.num_patches**0.5))
        x = nn.functional.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous().view(b, self.num_patches, c)

    def unpatchify(self, patchified_pixel_values, original_image_size=None):
        patch_size = self.config.patch_size
        num_channels = self.config.num_channels
        if original_image_size is None:
            original_image_size = (self.config.image_size, self.config.image_size)
        oh, ow = original_image_size
        nph, npw = oh // patch_size, ow // patch_size
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, nph, npw, patch_size, patch_size, num_channels
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        return patchified_pixel_values.reshape(batch_size, num_channels, nph * patch_size, npw * patch_size)

    def forward(self, hidden_states, drop_cls_token=False):
        x = self.decoder_embed(hidden_states)
        if drop_cls_token:
            x_ = x[:, 1:, :]
            x_ = self.interpolate_latent(x_)
        else:
            x_ = self.interpolate_latent(x)
        cls_token = self.trainable_cls_token.expand(x_.shape[0], -1, -1)
        x = torch.cat([cls_token, x_], dim=1)
        hidden_states = x + self.decoder_pos_embed

        for layer_module in self.decoder_layers:
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer_module.__call__, hidden_states, None, False, use_reentrant=False
                )
            else:
                layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]

        hidden_states = self.decoder_norm(hidden_states)
        logits = self.decoder_pred(hidden_states)
        logits = logits[:, 1:, :]  # remove cls token
        return logits


# ============================================================================
# RAEDecoder — cod-compatible BaseAE wrapper
# ============================================================================

class RAEDecoder(BaseAE):
    """RAEv2 decoder as RAE-CoD VAE interface.

    decode() denormalizes DINOv3 latent features, runs GeneralDecoder, and
    unpatchifies the result into pixels in [-1, 1].

    Args:
        decoder_config: Dict of ViTMAEConfig kwargs for the decoder architecture.
        pretrained_decoder_path: Path to decoder.pt checkpoint.
        normalization_stat_path: Path to stats.pt with mean/var tensors.
        encoder_hidden_size: Hidden size of the upstream encoder (DINOv3 = 1024).
        decoder_patch_size: Patch size used by the decoder (16 for RAEv2).
        image_size: Target output image size (256).
        num_patches: Number of spatial patches (256 = 16x16).
        eps: Epsilon for normalization stability.
    """

    def __init__(
        self,
        decoder_config: dict = None,
        pretrained_decoder_path: str = "",
        normalization_stat_path: str = "",
        encoder_hidden_size: int = 1024,
        decoder_patch_size: int = 16,
        image_size: int = 256,
        num_patches: int = 256,
        eps: float = 1e-5,
        finetune_mode: str = "frozen",
    ):
        super().__init__()
        self.eps = eps
        self.encoder_hidden_size = encoder_hidden_size

        # Build config
        if decoder_config is None:
            decoder_config = {}
        config = ViTMAEConfig(
            hidden_size=encoder_hidden_size,
            patch_size=decoder_patch_size,
            image_size=image_size,
            **decoder_config,
        )

        # Build decoder
        self.decoder = GeneralDecoder(config, num_patches=num_patches)

        # Load pretrained weights
        if pretrained_decoder_path:
            state_dict = torch.load(pretrained_decoder_path, map_location="cpu", weights_only=False)
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if keys.missing_keys:
                print(f"[RAEDecoder] Missing keys: {keys.missing_keys}")
            if keys.unexpected_keys:
                print(f"[RAEDecoder] Unexpected keys: {keys.unexpected_keys}")
            print(f"[RAEDecoder] Loaded decoder from {pretrained_decoder_path}")

        # Load normalization stats
        self.do_normalization = False
        if normalization_stat_path:
            stats = torch.load(normalization_stat_path, map_location="cpu", weights_only=False)
            mean = stats.get("mean", None)
            var = stats.get("var", None)
            if mean is not None:
                self.register_buffer("latent_mean", mean)
            if var is not None:
                self.register_buffer("latent_var", var)
            self.do_normalization = True
            print(f"[RAEDecoder] Loaded normalization stats from {normalization_stat_path}")

        # Apply finetune_mode: frozen (default), lora, or full
        self.finetune_mode = finetune_mode
        if finetune_mode == "frozen":
            self.requires_grad_(False)
        elif finetune_mode == "lora":
            self.requires_grad_(False)
            from src.models.lora import apply_lora_to_decoder
            apply_lora_to_decoder(self.decoder)
        # "full": leave all parameters trainable

    def _impl_decode(self, z):
        """Denormalize + decode latents to pixels.

        Args:
            z: (B, 1024, H, W) normalized latent features.

        Returns:
            (B, 3, image_size, image_size) pixel values in [-1, 1] range.
        """
        # Denormalize
        if self.do_normalization:
            mean = self.latent_mean.to(z.device)
            var = self.latent_var.to(z.device)
            z = z * torch.sqrt(var + self.eps) + mean

        # Reshape to tokens: (B, C, H, W) → (B, N, C)
        b, c, h, w = z.shape
        z = z.view(b, c, h * w).transpose(1, 2)  # (B, N, C)

        # Decoder forward + unpatchify
        logits = self.decoder(z, drop_cls_token=False)  # (B, N, patch_size^2 * 3)
        pixels = self.decoder.unpatchify(logits)  # (B, 3, image_size, image_size)

        # RAEv2 decoder outputs [0, 1] range; RAE-CoD expects [-1, 1] for fp2uint8.
        pixels = pixels * 2 - 1
        return pixels
