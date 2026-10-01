"""Frozen feature extractors for semantic distance computation.

Supports DINOv2, CLIP, SigLIP2, ConvNeXtV2, and InceptionV3.
Ported from FD-Loss (https://github.com/Jiawei-Yang/FD-Loss).
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("sem_dist")


def _preprocess(x, mean, std, target_size=None):
    """[0,1] float -> resize to target_size -> ImageNet normalize."""
    if target_size is not None and (
        x.shape[-2] != target_size or x.shape[-1] != target_size
    ):
        x = F.interpolate(
            x,
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    return (x - mean) / std


class TimmReprModel(torch.nn.Module):
    """Wraps a timm model as a frozen feature extractor.

    Handles preprocessing: [0, 1] -> resize -> ImageNet normalize.
    Returns ``(cls_token, mean_token)``.
    """

    def __init__(self, model_name: str, device="cuda", target_size: int | None = None):
        super().__init__()
        import timm
        from timm.data import resolve_data_config

        logger.info(f"[TimmReprModel] Loading model: {model_name}")
        kwargs = dict(pretrained=True, num_classes=0)
        try:
            self.model = timm.create_model(
                model_name, dynamic_img_size=True, dynamic_img_pad=True, **kwargs
            )
        except TypeError:
            self.model = timm.create_model(model_name, **kwargs)
        self.model.to(device).eval().requires_grad_(False)
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        self.has_attn_pool = (
            hasattr(self.model, "attn_pool") and self.model.attn_pool is not None
        )
        self.feat_dim = self.model.num_features

        data_cfg = resolve_data_config(self.model.pretrained_cfg)
        native_size = data_cfg["input_size"][-1]
        if target_size is not None and target_size != native_size:
            self.target_size = target_size
            logger.info(
                f"[TimmReprModel] Overriding target_size: {native_size} -> {target_size}"
            )
        else:
            self.target_size = native_size

        mean = torch.tensor(data_cfg["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(data_cfg["std"], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        interpolation = data_cfg.get("interpolation", "bicubic")
        logger.info(
            f"[TimmReprModel] {model_name}: feat_dim={self.feat_dim}, "
            f"target_size={self.target_size}, interpolation={interpolation}, "
            f"mean={data_cfg['mean']}, std={data_cfg['std']}"
        )

    def forward(self, x: torch.Tensor):
        x = _preprocess(x, self.mean, self.std, self.target_size)
        feats = self.model.forward_features(x)
        # CNN models return (B, C, H, W); pool spatially
        if feats.ndim == 4:
            cls_token = feats.mean(dim=[2, 3])
            return cls_token, None
        # ViT models return (B, N, C)
        patch_tokens = feats[:, self.num_prefix_tokens :]
        mean_token = patch_tokens.mean(1)
        if self.num_prefix_tokens > 0:
            cls_token = feats[:, 0]
        elif self.has_attn_pool:
            pool = getattr(self.model, "pool", None) or getattr(
                self.model, "_pool", None
            )
            cls_token = pool(feats)
        else:
            cls_token = mean_token
        return cls_token, mean_token
