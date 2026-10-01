#!/usr/bin/env python3
"""Download pretrained weights for all VFMs used in semantic distance computation.

- Inception V3: downloads from torch-fidelity GitHub release
- DINOv2, SigLIP2, CLIP, ConvNeXtV2: downloaded automatically by timm on first use

This script pre-downloads all weights so evaluation runs offline.
"""

import torch
import timm


INCEPTION_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/"
    "v0.2.0/weights-inception-2015-12-05-6726825d.pth"
)

TIMM_MODELS = [
    "vit_large_patch14_dinov2.lvd142m",
    "vit_so400m_patch16_siglip_256.v2_webli",
    "vit_large_patch14_clip_224.openai",
    "convnextv2_base.fcmae_ft_in22k_in1k",
]


def download_inception():
    """Download Inception V3 weights."""
    print("Downloading Inception V3 weights...")
    torch.hub.load_state_dict_from_url(INCEPTION_URL, progress=True)
    print("  Done.")


def download_timm_model(model_name):
    """Pre-download a timm model."""
    print(f"Downloading {model_name}...")
    try:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
    except TypeError:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
    del model
    print("  Done.")


def main():
    print("=" * 60)
    print("Downloading pretrained weights for semantic distance VFMs")
    print("=" * 60)
    print()

    download_inception()
    print()

    for name in TIMM_MODELS:
        download_timm_model(name)
        print()

    print("=" * 60)
    print("All weights downloaded successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
