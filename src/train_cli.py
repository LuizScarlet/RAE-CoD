import src.utils.patch_bugs  # noqa: F401  (import applies runtime patches)
from src.utils.model_loader import load_pretrained_model, load_pretrained_state_dict
from src.lightning_data import DataModule
from src.lightning_module import RAECoDLightningModule
from src.lightning_utils import ReWriteRootSaveConfigCallback, BaseReWriteRootDirCli
from src.models.lora import merge_lora_state_dict

import logging
logger = logging.getLogger("lightning.pytorch")


class ReWriteRootDirCli(BaseReWriteRootDirCli):

    def _load_pretrained(self):
        pretrained_ckpt_path = self._get(self.config, "pretrained_ckpt_path")
        if pretrained_ckpt_path is not None:
            pretrained_ema = self._get(self.config, "pretrained_ema")

            # When loading a LoRA checkpoint into a full-finetune model,
            # merge LoRA weights in the state dict before loading.
            # Read finetune_mode from the already-instantiated model.
            finetune_mode = getattr(self.model.denoiser, "finetune_mode", "lora")
            if finetune_mode == "full":
                sd = load_pretrained_state_dict(pretrained_ckpt_path)
                has_lora_keys = any(k.endswith(".lora_A.weight") for k in sd)
                if has_lora_keys:
                    lora_rank = self._get(self.config, "model.denoiser.init_args.lora_rank") or 32
                    lora_alpha = self._get(self.config, "model.denoiser.init_args.lora_alpha") or 32
                    logger.info("Merging LoRA weights from checkpoint (rank=%d, alpha=%d) for full fine-tuning",
                                lora_rank, lora_alpha)
                    sd = merge_lora_state_dict(sd, lora_alpha=lora_alpha, lora_rank=lora_rank)
                    # Load merged state dict directly
                    model_dict = self.model.state_dict()
                    matched, skipped = 0, []
                    for k, v in sd.items():
                        if k in model_dict and model_dict[k].shape == v.shape:
                            model_dict[k] = v
                            matched += 1
                        else:
                            skipped.append(k)
                    self.model.load_state_dict(model_dict, strict=False)
                    logger.info("Loaded %d params from LoRA-merged checkpoint (%d skipped)", matched, len(skipped))
                    return

            self.model = load_pretrained_model(self.model, pretrained_ckpt_path, pretrained_ema, strict=False, log=True)


if __name__ == "__main__":

    ReWriteRootDirCli(RAECoDLightningModule, DataModule,
                            auto_configure_optimizers=False,
                            save_config_callback=ReWriteRootSaveConfigCallback,
                            save_config_kwargs={"overwrite": True})
