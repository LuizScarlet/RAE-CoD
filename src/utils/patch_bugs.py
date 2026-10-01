"""Runtime settings shared by training and inference."""
import os

import torch

torch.set_float32_matmul_precision("medium")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
