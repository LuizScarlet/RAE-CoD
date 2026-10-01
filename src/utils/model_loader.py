import os
import torch

import logging
logger = logging.getLogger(__name__)

def load_pretrained_state_dict(pretrained_ckpt_path):  
    # for splitted saved ddp weights
    if os.path.exists(os.path.join(pretrained_ckpt_path, "checkpoint-state_dict.pt")):
        pretrained_ckpt_path = os.path.join(pretrained_ckpt_path, "checkpoint-state_dict.pt")
    # for deepspeed
    elif os.path.isdir(pretrained_ckpt_path):
        pretrained_ckpt_path = os.path.join(pretrained_ckpt_path, "checkpoint", "mp_rank_00_model_states.pt")

    state = torch.load(
        pretrained_ckpt_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if "module" in state:
        pretrained_dict = state["module"]
    elif "state_dict" in state:
        pretrained_dict = state["state_dict"]
    else:
        pretrained_dict = state
    return pretrained_dict


def load_pretrained_model(
    model,
    pretrained_ckpt_path,
    pretrained_ema,
    strict=False,
    log=True,
):
    if log:
        print(
            f"Loading pretrained weights from {pretrained_ckpt_path} "
            f"(ema={pretrained_ema}, strict={strict})"
        )

    pretrained_dict = load_pretrained_state_dict(pretrained_ckpt_path)
    model_dict = model.state_dict()
    matched_dict = {}
    used_source_keys = set()
    shape_mismatches = []

    for target_key, target_value in model_dict.items():
        candidates = [target_key]
        if pretrained_ema:
            if target_key.startswith("denoiser."):
                candidates = [f"ema_{target_key}", target_key]
            elif target_key.startswith("ema_denoiser."):
                candidates = [target_key, target_key.removeprefix("ema_")]

        source_key = next((key for key in candidates if key in pretrained_dict), None)
        if source_key is None:
            continue
        source_value = pretrained_dict[source_key]
        if target_value.shape != source_value.shape:
            shape_mismatches.append(
                (source_key, f"shape mismatch: {source_value.shape} vs {target_value.shape}")
            )
            continue
        matched_dict[target_key] = source_value
        used_source_keys.add(source_key)

    unused_source_keys = sorted(set(pretrained_dict) - used_source_keys)
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=strict)

    if log:
        print(f"Loaded {len(matched_dict)} parameters.")
        print(
            f"Skipped {len(unused_source_keys)} unused source parameters and "
            f"{len(shape_mismatches)} shape mismatches."
        )
        for item in shape_mismatches:
            print(item)

    return model
