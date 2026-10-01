import os
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import contextmanager
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


DINOV3_HUB_REF = "facebookresearch/dinov3:94a96ac83c2446f15f9bdcfae23cad3c6a9d4988"

SHA_CHECKSUM = {
    "dinov3_vits16": "08c60483",
    "dinov3_vits16plus": "4057cbaa",
    "dinov3_vitb16": "73cec8be",
    "dinov3_vitl16": "8aa4cbdd",
    "dinov3_vith16plus": "7c1da9a5",
    "dinov3_vit7b16": "a955f4ea",
}

DEFAULT_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [5, 11, 17, 23],
    "dinov3_vith16plus": [8, 16, 24, 31],
}


@contextmanager
def _rank0_first():
    """Gate torch.hub download so only rank 0 fetches, others wait."""
    initialized = dist.is_initialized()
    rank = dist.get_rank() if initialized else 0
    if initialized and rank != 0:
        dist.barrier()
    try:
        yield
    finally:
        if initialized and rank == 0:
            dist.barrier()


class DINOv3MultiLayer(nn.Module):
    """Frozen DINOv3 multi-layer feature encoder for VFM supervision.

    Loads a DINOv3 model, extracts intermediate layer features, averages them,
    and adds a broadcast mean of the final layer (matching RAEv2's
    DINOv3MultiLayerSimpleAddEncoder).

    Args:
        ckpt_dir: Directory containing DINOv3 checkpoint files.
        repo_dir: Local clone of the dinov3 repo (for torch.hub.load source="local").
                  If empty/None, falls back to GitHub hub reference.
        model_name: DINOv3 model variant (e.g. "dinov3_vitl16").
        layer_indices: Which intermediate layers to extract and average.
    """

    def __init__(
        self,
        ckpt_dir: str = "",
        repo_dir: str = "",
        model_name: str = "dinov3_vitl16",
        layer_indices: list = None,
    ):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.repo_dir = repo_dir
        self.model_name = model_name
        self.layer_indices = layer_indices or DEFAULT_LAYERS.get(model_name, [5, 11, 17, 23])

        self.normalize = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self._load_model()

    def _load_model(self):
        sha = SHA_CHECKSUM[self.model_name]
        weights = os.path.join(self.ckpt_dir, f"{self.model_name}_pretrain_lvd1689m-{sha}.pth")

        with _rank0_first():
            if self.repo_dir and os.path.isfile(os.path.join(self.repo_dir, "hubconf.py")):
                model = torch.hub.load(
                    self.repo_dir, self.model_name,
                    source="local", trust_repo=True, skip_validation=True,
                    weights=weights,
                )
            else:
                model = torch.hub.load(
                    DINOV3_HUB_REF, self.model_name,
                    source="github", trust_repo=True, skip_validation=True,
                    weights=weights,
                )

        # Strip norm affine (matches RAEv2 default)
        embed_dim = model.embed_dim
        model.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

        model = model.to(torch.bfloat16)
        model.eval()
        model.requires_grad_(False)
        self.model = model
        self.embed_dim = embed_dim  # 1024 for vitl16

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, x):
        """Extract multi-layer DINOv3 features.

        Args:
            x: (B, 3, H, W) images in [0, 1] range.

        Returns:
            patch_tokens: (B, num_patches, embed_dim) where
                num_patches = (H/16) * (W/16).
        """
        x = self.normalize(x)

        outputs = self.model.get_intermediate_layers(
            x, n=self.layer_indices, reshape=False,
            return_class_token=False, norm=True,
        )

        # Average across selected layers
        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        # Add broadcast mean of final layer (global pooled signal)
        final_mean = outputs[-1].mean(dim=1, keepdim=True)
        patch_tokens = patch_tokens + final_mean

        return patch_tokens
