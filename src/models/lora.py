"""LoRA (Low-Rank Adaptation) for DiTwDDTHeadIG and GeneralDecoder.

Wraps existing nn.Linear layers with low-rank A/B matrices. Only the A and B
matrices are trainable; the original weight is frozen.

LoRA is applied to the DiT:
  - All attention Q/K/V/proj layers in encoder and decoder blocks
  - All FFN w1/w2/w3 layers in encoder and decoder blocks
  - Decoder adaptive-normalization projections
  - The encoder-to-decoder interface projection
  - final_layer.linear (full model output head)
  - base_final_layer.linear (IG early exit output head)

LoRA is applied to the GeneralDecoder:
  - All attention query/key/value/output.dense layers in decoder_layers
  - All FFN intermediate.dense/output.dense layers in decoder_layers
  - decoder_embed (input embedding projection)
  - decoder_pred (output prediction head)
"""
import math
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank adapter.

    Forward: out = original(x) + B(A(x)) * (alpha / rank)
    Only A and B are trainable; the original linear's parameters are frozen.
    """

    def __init__(self, original: nn.Linear, rank: int = 32, alpha: int = 32):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # A: down-projection, B: up-projection (initialized so adapter starts at zero)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize A with Kaiming uniform, B with zeros → adapter output is zero at init
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features


def _wrap_linear(module, attr_name, rank, alpha):
    """Replace a nn.Linear attribute with a LoRALinear wrapper. Returns LoRA params."""
    original = getattr(module, attr_name)
    if not isinstance(original, nn.Linear):
        return []
    lora_layer = LoRALinear(original, rank=rank, alpha=alpha)
    setattr(module, attr_name, lora_layer)
    return list(lora_layer.lora_A.parameters()) + list(lora_layer.lora_B.parameters())


def _wrap_sequential_linear(seq, idx, rank, alpha):
    """Replace a nn.Linear at seq[idx] inside an nn.Sequential with LoRALinear. Returns LoRA params."""
    original = seq[idx]
    if not isinstance(original, nn.Linear):
        return []
    lora_layer = LoRALinear(original, rank=rank, alpha=alpha)
    seq[idx] = lora_layer
    return list(lora_layer.lora_A.parameters()) + list(lora_layer.lora_B.parameters())


def apply_lora_to_dit(dit, rank=32, alpha=32):
    """Inject LoRA adapters into attention, FFN, and output head layers.

    Targets:
      - blocks[*].attn.{q, k, v, proj}   — attention projections
      - blocks[*].mlp.{w1, w2, w3}        — SwiGLU FFN projections
      - blocks[*].adaln_modulation[1]       — adaptive-normalization projection
      - s_projector                         — encoder-to-decoder interface
      - final_layer.linear                  — full model output head
      - base_final_layer.linear            — IG early exit output head

    Args:
        dit: DiTwDDTHeadIG instance.
        rank: LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        List of trainable LoRA parameters (for optimizer param groups).
    """
    lora_params = []

    # Attention and FFN in all encoder + decoder blocks
    for block in dit.blocks:
        for attr_name in ("q", "k", "v", "proj"):
            lora_params.extend(_wrap_linear(block.attn, attr_name, rank, alpha))
        for attr_name in ("w1", "w2", "w3"):
            lora_params.extend(_wrap_linear(block.mlp, attr_name, rank, alpha))
        # Decoder block adaln_modulation: Sequential(SiLU(), Linear(h, 6*h))
        if hasattr(block, "adaln_modulation"):
            lora_params.extend(_wrap_sequential_linear(block.adaln_modulation, 1, rank, alpha))

    # Encoder→decoder projection
    if hasattr(dit, "s_projector") and isinstance(dit.s_projector, nn.Linear):
        lora_params.extend(_wrap_linear(dit, "s_projector", rank, alpha))

    # Output heads
    lora_params.extend(_wrap_linear(dit.final_layer, "linear", rank, alpha))
    if hasattr(dit, "base_final_layer"):
        lora_params.extend(_wrap_linear(dit.base_final_layer, "linear", rank, alpha))

    return lora_params


def apply_lora_to_decoder(decoder, rank=32, alpha=32):
    """Inject LoRA adapters into GeneralDecoder attention and FFN layers.

    Targets per decoder_layers block:
      - attention.attention.{query, key, value}  — self-attention QKV projections
      - attention.output.dense                   — attention output projection
      - intermediate.dense                        — FFN up-projection
      - output.dense                              — FFN down-projection
    Plus:
      - decoder_embed  — input embedding projection
      - decoder_pred   — output prediction head

    Args:
        decoder: GeneralDecoder instance.
        rank: LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        List of trainable LoRA parameters.
    """
    lora_params = []

    for layer in decoder.decoder_layers:
        for attr in ("query", "key", "value"):
            lora_params.extend(_wrap_linear(layer.attention.attention, attr, rank, alpha))
        lora_params.extend(_wrap_linear(layer.attention.output, "dense", rank, alpha))
        lora_params.extend(_wrap_linear(layer.intermediate, "dense", rank, alpha))
        lora_params.extend(_wrap_linear(layer.output, "dense", rank, alpha))

    lora_params.extend(_wrap_linear(decoder, "decoder_embed", rank, alpha))
    lora_params.extend(_wrap_linear(decoder, "decoder_pred", rank, alpha))

    return lora_params


def merge_lora_state_dict(state_dict, lora_alpha=32, lora_rank=32):
    """Merge LoRA weights in a state dict, converting LoRA keys to plain Linear keys.

    Detects LoRA key patterns (*.lora_A.weight, *.lora_B.weight, *.original.weight),
    computes merged weights, and returns a clean state dict with plain nn.Linear keys.

    This enables loading a LoRA checkpoint into a model without LoRA layers
    (e.g., for full fine-tuning after a LoRA stage).

    Args:
        state_dict: State dict potentially containing LoRA keys.
        lora_alpha: LoRA scaling alpha (must match the value used during LoRA training).
        lora_rank: LoRA rank (must match the value used during LoRA training).

    Returns:
        New state dict with LoRA weights merged and LoRA keys removed.
        If no LoRA keys are found, returns the original state dict unchanged.
    """
    # Find all LoRA key groups: prefix is everything before ".lora_A.weight"
    lora_prefixes = set()
    for k in state_dict:
        if k.endswith(".lora_A.weight"):
            lora_prefixes.add(k[: -len(".lora_A.weight")])

    if not lora_prefixes:
        return state_dict

    scaling = lora_alpha / lora_rank
    merged = {}
    lora_keys = set()

    for prefix in lora_prefixes:
        lora_a_key = f"{prefix}.lora_A.weight"
        lora_b_key = f"{prefix}.lora_B.weight"
        orig_weight_key = f"{prefix}.original.weight"
        orig_bias_key = f"{prefix}.original.bias"

        lora_keys.update({lora_a_key, lora_b_key, orig_weight_key, orig_bias_key})

        # Merge: W = W_original + B @ A * scaling
        w = state_dict[orig_weight_key].clone()
        w.add_((state_dict[lora_b_key] @ state_dict[lora_a_key]) * scaling)
        merged[f"{prefix}.weight"] = w

        if orig_bias_key in state_dict:
            merged[f"{prefix}.bias"] = state_dict[orig_bias_key]

    # Copy all non-LoRA keys, remap remaining *.original.* keys
    out = {}
    for k, v in state_dict.items():
        if k in lora_keys:
            continue
        # Remap any leftover .original. references (shouldn't happen, but be safe)
        if ".original." in k:
            new_k = k.replace(".original.", ".")
            out[new_k] = v
        else:
            out[k] = v
    out.update(merged)
    return out
