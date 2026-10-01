import os
import torch
import time
from typing import Any
import re

import src.utils.patch_bugs  # noqa: F401  (import applies runtime settings)
from lightning import Trainer, LightningModule
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback



class ReWriteRootSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        stamp = time.strftime('%y%m%d%H%M')
        file_path = os.path.join(trainer.default_root_dir, f"config-{stage}-{stamp}.yaml")
        self.parser.save(
            self.config, file_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
        )


class BaseReWriteRootDirCli(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        class TagsClass:
            def __init__(self, exp:str):
                ...
        parser.add_class_arguments(TagsClass, nested_key="tags")

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument("--torch_hub_dir", type=str, default=None, help=("torch hub dir"),)
        parser.add_argument("--huggingface_cache_dir", type=str, default=None, help=("huggingface hub dir"),)
        parser.add_argument("--pretrained_ckpt_path", type=str, default=None, help="Path to pretrained checkpoint")
        parser.add_argument("--pretrained_ema", type=bool, default=False, help="Load ema weights")

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        config_trainer = self._get(self.config_init, "trainer", default={})
        default_root_dir = config_trainer.get("default_root_dir", None)

        if default_root_dir is None:
            default_root_dir = os.path.join(os.getcwd(), "workdirs")

        dirname = ""
        for v, k in self._get(self.config, "tags", default={}).items():
            dirname += f"{v}_{k}"
        default_root_dir = os.path.join(default_root_dir, dirname)

        self.resume_path = self._get(self.config_init, "ckpt_path", default=None)
        if not self.resume_path and os.path.exists(default_root_dir):
            ckpts = [ckpt for ckpt in os.listdir(default_root_dir) if ckpt.endswith('.ckpt')]
            if len(ckpts) > 0:
                def extract_step(f):
                    m = re.search(r"step=(\d+)", f)
                    return int(m.group(1)) if m else -1
                latest_ckpt = max(ckpts, key=extract_step)
                self.resume_path = os.path.join(default_root_dir, latest_ckpt)

                # for splitted saved ddp weights
                if os.path.exists(os.path.join(self.resume_path, "ddp_split.txt")):
                    merged_ckpt_path = os.path.join(self.resume_path, "checkpoint_merged.ckpt")
                    if not os.path.exists(merged_ckpt_path):
                        checkpoint = {}
                        for fname in os.listdir(self.resume_path):
                            if fname.startswith("checkpoint-") and fname.endswith(".pt"):
                                data = torch.load(os.path.join(self.resume_path, fname), map_location="cpu")
                                checkpoint.update(data)
                        torch.save(checkpoint, merged_ckpt_path)
                    self.resume_path = merged_ckpt_path

                print(f"[Resume] checkpoint: {latest_ckpt}")

        config_trainer.default_root_dir = default_root_dir
        trainer = super().instantiate_trainer(**kwargs)
        if trainer.is_global_zero:
            os.makedirs(default_root_dir, exist_ok=True)
        if self.resume_path:
            trainer.ckpt_path = self.resume_path
        return trainer

    def _load_pretrained(self):
        raise NotImplementedError

    def instantiate_classes(self) -> None:
        torch_hub_dir = self._get(self.config, "torch_hub_dir")
        huggingface_cache_dir = self._get(self.config, "huggingface_cache_dir")
        if huggingface_cache_dir is not None:
            os.environ["HUGGINGFACE_HUB_CACHE"] = huggingface_cache_dir
        if torch_hub_dir is not None:
            os.environ["TORCH_HOME"] = torch_hub_dir
            torch.hub.set_dir(torch_hub_dir)
        super().instantiate_classes()

        if getattr(self, "resume_path", None) and (os.path.isfile(self.resume_path) or os.path.isfile(os.path.join(self.resume_path, "latest"))):
            print("Resume detected; skipping pretrained_ckpt_path")
            return
    
        self._load_pretrained()
