#!/usr/bin/env python3
"""Training entry for standard-rate and 16-bit RAE-CoD models.

Stage I adapts the pretrained RAEv2 DDT with LoRA while progressively increasing
its rate penalty. Stage II starts from a corresponding Stage-I checkpoint,
merges LoRA into the backbone, and jointly fine-tunes the codec and full DDT at
a fixed target rate. The optional VQ stage starts from the rate-48 Stage-II
checkpoint, removes the main-latent entropy model, and fine-tunes the resulting
fixed 16-bit codec without a rate loss.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml

from src.utils.config_utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATE_CONFIG = PROJECT_ROOT / "src/configs/train_256.yaml"
STAGE_COMPLETE_FLAG = ".stage_complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    source = data.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-root", type=Path, help="Root for paths in --train-metadata.")
    source.add_argument("--train-lmdb", type=str, help="LMDB path or shard glob.")
    data.add_argument("--train-metadata", type=Path, help="One relative image path per line.")
    data.add_argument("--train-size", type=int, help="Number of samples in --train-lmdb.")
    data.add_argument("--eval-root", type=Path, required=True, help="Directory of 256x256 validation images.")
    data.add_argument("--num-workers", type=int, default=16)

    weights = parser.add_argument_group("pretrained weights")
    weights.add_argument("--dinov3-ckpt-dir", type=Path, required=True)
    weights.add_argument("--dinov3-repo-dir", type=Path, required=True)
    weights.add_argument("--pretrained-dit", type=Path, required=True, help="Pretrained RAEv2 DDT checkpoint.")
    weights.add_argument("--rae-decoder", type=Path, required=True)
    weights.add_argument("--rae-stats", type=Path, required=True)

    run = parser.add_argument_group("run")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--experiment-name", default="rae_cod")
    run.add_argument(
        "--run-stage",
        choices=("all", "stage1", "stage2", "vq"),
        default="all",
    )
    run.add_argument("--stage2-checkpoint", type=Path, help="Required for a standalone Stage-II run.")
    run.add_argument(
        "--vq-source-checkpoint",
        type=Path,
        help="Rate-48 Stage-II checkpoint required for --run-stage=vq.",
    )
    run.add_argument("--stage2-rate", type=float, choices=(12, 16, 24, 32, 48), default=48)
    run.add_argument("--stage1-max-steps", type=int, default=180_000)
    run.add_argument("--stage2-max-steps", type=int, default=100_000)
    run.add_argument("--vq-max-steps", type=int, default=100_000)
    run.add_argument("--stage1-lr", type=float, default=1.0e-4)
    run.add_argument("--stage2-lr", type=float, default=1.0e-5)
    run.add_argument("--vq-lr", type=float, default=1.0e-5)
    run.add_argument("--target-batch-size", type=int, default=128)
    run.add_argument("--stage1-grad-accum", type=int, default=1)
    run.add_argument("--stage2-grad-accum", type=int, default=2)
    run.add_argument("--vq-grad-accum", type=int, default=2)
    run.add_argument("--gpus", type=int, help="Defaults to all visible CUDA devices.")
    run.add_argument("--overwrite-completed", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="Write configs and print commands only.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.train_root is not None and args.train_metadata is None:
        raise SystemExit("--train-metadata is required with --train-root")
    if args.train_lmdb is not None and args.train_size is None:
        raise SystemExit("--train-size is required with --train-lmdb")
    if args.run_stage == "stage2" and args.stage2_checkpoint is None:
        raise SystemExit("--stage2-checkpoint is required when --run-stage=stage2")
    if args.run_stage == "vq" and args.vq_source_checkpoint is None:
        raise SystemExit("--vq-source-checkpoint is required when --run-stage=vq")
    for checkpoint_name in ("stage2_checkpoint", "vq_source_checkpoint"):
        checkpoint = getattr(args, checkpoint_name)
        if checkpoint is not None and not checkpoint.exists():
            option = checkpoint_name.replace("_", "-")
            raise SystemExit(f"--{option} does not exist: {checkpoint}")
    for path_name in (
        "eval_root", "dinov3_ckpt_dir", "dinov3_repo_dir", "pretrained_dit",
        "rae_decoder", "rae_stats",
    ):
        path = getattr(args, path_name)
        if not path.exists():
            raise SystemExit(f"--{path_name.replace('_', '-')} does not exist: {path}")
    if args.train_root is not None:
        for path_name in ("train_root", "train_metadata"):
            path = getattr(args, path_name)
            if not path.exists():
                raise SystemExit(f"--{path_name.replace('_', '-')} does not exist: {path}")
    if args.gpus is not None and args.gpus < 1:
        raise SystemExit("--gpus must be positive")
    if min(args.stage1_lr, args.stage2_lr, args.vq_lr) <= 0:
        raise SystemExit("learning rates must be positive")
    if min(args.stage1_grad_accum, args.stage2_grad_accum, args.vq_grad_accum) < 1:
        raise SystemExit("gradient accumulation must be positive")


def effective_batch_size(target: int, gpus: int, accumulation: int) -> tuple[int, int]:
    denominator = gpus * accumulation
    per_gpu = max(target // denominator, 1)
    actual = per_gpu * denominator
    if actual != target:
        print(
            f"Warning: effective batch size is {actual}, not {target} "
            f"({per_gpu} per GPU x {gpus} GPUs x {accumulation} accumulation)."
        )
    return per_gpu, actual


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step[=-](\d+)", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(stage_dir: Path) -> Path | None:
    checkpoints = list(stage_dir.rglob("*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: (checkpoint_step(path), path.stat().st_mtime))


def next_log_path(stage_dir: Path) -> Path:
    index = 0
    while True:
        path = stage_dir / f"train_{index}.log"
        if not path.exists():
            return path
        index += 1


def build_overrides(
    args: argparse.Namespace,
    *,
    stage_dir: Path,
    batch_size: int,
    num_devices: int,
    finetune_mode: str,
    max_steps: int,
    checkpoint: Path | None,
    fixed_rate: float | None,
    learning_rate: float,
    vq_only: bool = False,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "torch_hub_dir": str(Path.home() / ".cache/torch/hub"),
        "trainer.default_root_dir": str(stage_dir),
        "trainer.accumulate_grad_batches": 1,  # accumulation is handled by the CLI below
        "trainer.devices": num_devices,
        "trainer.max_steps": max_steps,
        "model.denoiser.init_args.pretrained_dit_path": str(args.pretrained_dit.resolve()),
        "model.denoiser.init_args.finetune_mode": finetune_mode,
        "model.vae.init_args.pretrained_decoder_path": str(args.rae_decoder.resolve()),
        "model.optimizer.init_args.lr": learning_rate,
        "model.vae.init_args.normalization_stat_path": str(args.rae_stats.resolve()),
        "model.diffusion_trainer.init_args.encoder.init_args.ckpt_dir": str(args.dinov3_ckpt_dir.resolve()),
        "model.diffusion_trainer.init_args.encoder.init_args.repo_dir": str(args.dinov3_repo_dir.resolve()),
        "model.diffusion_trainer.init_args.normalization_stat_path": str(args.rae_stats.resolve()),
        "data.eval_dataset.init_args.root": str(args.eval_root.resolve()),
        "data.pred_dataset.init_args.root": str(args.eval_root.resolve()),
        "data.train_batch_size": batch_size,
        "data.train_num_workers": args.num_workers,
        "tags.exp": stage_dir.name,
    }
    if args.train_lmdb:
        overrides["data.train_dataset"] = {
            "class_path": "src.data.dataset.lmdb_image.LMDBImageDataset",
            "init_args": {
                "lmdb_path": args.train_lmdb,
                "length": args.train_size,
                "resolution": 256,
            },
        }
    else:
        overrides["data.train_dataset"] = {
            "class_path": "src.data.dataset.image.ImageDataset",
            "init_args": {
                "root": str(args.train_root.resolve()),
                "metadata": str(args.train_metadata.resolve()),
                "resolution": 256,
            },
        }
    if checkpoint is not None:
        overrides["pretrained_ckpt_path"] = str(checkpoint.resolve())
    if vq_only:
        overrides["model.denoiser.init_args.vq_only"] = True
        overrides["model.diffusion_sampler.init_args.hyper_only"] = True
        # Inference-minimal releases contain only EMA denoiser weights. Map
        # those weights into both the trainable and EMA copies for VQ adaptation.
        overrides["pretrained_ema"] = True
        overrides["model.diffusion_trainer.init_args.lambda_rate"] = 0.0
        overrides["model.diffusion_trainer.init_args.lambda_rate_schedule"] = None
    elif fixed_rate is not None:
        overrides["model.diffusion_trainer.init_args.lambda_rate"] = fixed_rate
        overrides["model.diffusion_trainer.init_args.lambda_rate_schedule"] = None
    return overrides


def write_stage_config(overrides: dict[str, Any], path: Path) -> None:
    with TEMPLATE_CONFIG.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = load_config(config, overrides)

    # Preserve the release checkpoint cadence for full runs, while ensuring a
    # short debug run still emits a checkpoint that a following stage can load.
    max_steps = config["trainer"]["max_steps"]
    for callback in config["trainer"].get("callbacks", []):
        if callback.get("class_path", "").endswith(".ModelCheckpoint"):
            init_args = callback.setdefault("init_args", {})
            interval = init_args.get("every_n_train_steps")
            if interval is not None:
                init_args["every_n_train_steps"] = min(interval, max_steps)

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def run_and_tee(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_stage(
    args: argparse.Namespace,
    *,
    name: str,
    finetune_mode: str,
    max_steps: int,
    accumulation: int,
    gpus: int,
    checkpoint: Path | None = None,
    fixed_rate: float | None = None,
    learning_rate: float = 1.0e-4,
    vq_only: bool = False,
) -> Path:
    stage_dir = args.output_dir.resolve() / args.experiment_name / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    complete = stage_dir / STAGE_COMPLETE_FLAG
    if complete.exists() and not args.overwrite_completed:
        print(f"Skipping completed stage: {stage_dir}")
        return stage_dir

    batch_size, actual_batch = effective_batch_size(args.target_batch_size, gpus, accumulation)
    config_path = stage_dir / "config.yaml"
    overrides = build_overrides(
        args,
        stage_dir=stage_dir,
        batch_size=batch_size,
        num_devices=gpus,
        finetune_mode=finetune_mode,
        max_steps=max_steps,
        checkpoint=checkpoint,
        fixed_rate=fixed_rate,
        learning_rate=learning_rate,
        vq_only=vq_only,
    )
    # Lightning performs gradient accumulation; keep it in the resolved config.
    overrides["trainer.accumulate_grad_batches"] = accumulation
    write_stage_config(overrides, config_path)

    command = [
        sys.executable,
        "-m",
        "src.train_cli",
        "fit",
        "-c",
        str(config_path),
    ]
    print("\n" + "=" * 72)
    print(f"Stage: {name}")
    print(f"Output: {stage_dir}")
    rate_description = (
        "fixed 16 bits (no rate loss)"
        if vq_only
        else (fixed_rate or "progressive")
    )
    print(
        f"Fine-tuning: {finetune_mode}; rate: {rate_description}; "
        f"learning rate: {learning_rate:.1e}"
    )
    print(f"GPUs: {gpus}; per-GPU batch: {batch_size}; accumulation: {accumulation}")
    print(f"Effective batch: {actual_batch}; max steps: {max_steps}")
    if checkpoint:
        print(f"Initialization checkpoint: {checkpoint}")
    print("Command:", " ".join(command))
    print("=" * 72)
    if args.dry_run:
        return stage_dir

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # Lightning/fsspec may stage a complete checkpoint in the system temporary
    # directory before committing it. Point a short /tmp alias at the output
    # filesystem: this avoids both a small root disk and the Unix-socket path
    # limit used by multiprocessing DataLoader workers.
    temp_alias: Path | None = None
    if not env.get("TMPDIR"):
        checkpoint_tmp_dir = stage_dir / ".checkpoint_tmp"
        checkpoint_tmp_dir.mkdir(exist_ok=True)
        temp_alias = Path(tempfile.gettempdir()) / f"rae_cod_{os.getpid()}"
        temp_alias.unlink(missing_ok=True)
        temp_alias.symlink_to(checkpoint_tmp_dir, target_is_directory=True)
        env["TMPDIR"] = str(temp_alias)
    try:
        run_and_tee(command, next_log_path(stage_dir), env)
    finally:
        if temp_alias is not None:
            temp_alias.unlink(missing_ok=True)
    complete.write_text("complete\n", encoding="utf-8")
    return stage_dir


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    validate_args(args)
    gpus = args.gpus if args.gpus is not None else torch.cuda.device_count()
    if gpus < 1:
        raise SystemExit("No CUDA devices detected. Set CUDA_VISIBLE_DEVICES or pass --gpus.")

    stage1_dir: Path | None = None
    if args.run_stage in ("all", "stage1"):
        stage1_dir = run_stage(
            args,
            name="stage1_lora_progressive",
            finetune_mode="lora",
            max_steps=args.stage1_max_steps,
            accumulation=args.stage1_grad_accum,
            gpus=gpus,
            learning_rate=args.stage1_lr,
        )

    if args.run_stage in ("all", "stage2"):
        checkpoint = args.stage2_checkpoint
        if checkpoint is None and stage1_dir is not None:
            checkpoint = latest_checkpoint(stage1_dir)
        if checkpoint is None and args.dry_run and stage1_dir is not None:
            checkpoint = stage1_dir / "STAGE1_CHECKPOINT.ckpt"
        if checkpoint is None:
            raise SystemExit(
                "No Stage-I checkpoint found. Pass --stage2-checkpoint explicitly."
            )
        run_stage(
            args,
            name=f"stage2_full_rate{int(args.stage2_rate)}",
            finetune_mode="full",
            max_steps=args.stage2_max_steps,
            accumulation=args.stage2_grad_accum,
            gpus=gpus,
            checkpoint=checkpoint,
            fixed_rate=args.stage2_rate,
            learning_rate=args.stage2_lr,
        )

    if args.run_stage == "vq":
        run_stage(
            args,
            name="vq16_from_rate48",
            finetune_mode="full",
            max_steps=args.vq_max_steps,
            accumulation=args.vq_grad_accum,
            gpus=gpus,
            checkpoint=args.vq_source_checkpoint,
            fixed_rate=None,
            learning_rate=args.vq_lr,
            vq_only=True,
        )


if __name__ == "__main__":
    main()
