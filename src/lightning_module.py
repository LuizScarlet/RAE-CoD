import copy
import torch
import logging

from src.lightning_model import LightningModel
from src.models.autoencoder.base import fp2uint8
from src.utils.no_grad import no_grad
from src.utils.copy import copy_params

logger = logging.getLogger(__name__)


class RAECoDLightningModule(LightningModel):
    """LightningModel for latent-space diffusion with pretrained DiT + LoRA.

    Extends LightningModel to support:
    - Flexible DiT fine-tuning modes: frozen / lora / full
    - Flexible decoder fine-tuning modes: frozen / lora / full
    - EMA for both denoiser and vae (tracked by the shared ema_tracker callback)
    - torch.compile on both denoiser and vae
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create EMA copy of vae alongside the existing ema_denoiser
        self.ema_vae = copy.deepcopy(self.vae)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Strip optimizer state when the trainable parameter set has changed."""
        if "optimizer_states" in checkpoint:
            # Compare number of param groups: current model vs checkpoint
            try:
                saved_groups = checkpoint["optimizer_states"][0]["param_groups"]
                current_params = sum(
                    1 for _ in filter(lambda p: p.requires_grad, self.parameters())
                )
                saved_params = sum(len(g["params"]) for g in saved_groups)
                if current_params != saved_params:
                    logger.warning(
                        "Optimizer param count mismatch (saved=%d, current=%d). "
                        "Discarding optimizer state — optimizer will reinitialize.",
                        saved_params, current_params,
                    )
                    checkpoint["optimizer_states"] = []
                    # Also clear lr_schedulers since they reference the old optimizer
                    checkpoint["lr_schedulers"] = []
            except (KeyError, IndexError):
                pass

    @property
    def vfm_encoder(self):
        return self.diffusion_trainer.encoder

    def _extract_vfm_features(self, raw_images):
        """Extract VFM features from raw images using frozen DINOv3."""
        with torch.no_grad():
            return self.vfm_encoder(raw_images)

    def configure_model(self) -> None:
        """Set up model: copy params to EMA copies, freeze non-trainable components."""
        self.trainer.strategy.barrier()

        # Initialize EMA copies from current weights only on fresh start.
        # On checkpoint resume, Lightning restores ema_denoiser/ema_vae weights
        # AFTER configure_model runs, so we must not overwrite them here.
        if self.trainer.ckpt_path is None:
            copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)
            copy_params(src_model=self.vae, dst_model=self.ema_vae)

        # Freeze non-trainable components
        if getattr(self.vae, "finetune_mode", "frozen") == "frozen":
            no_grad(self.vae)
        no_grad(self.ema_denoiser)
        no_grad(self.ema_vae)  # ema_vae is always frozen; updated via EMA step

        # torch.compile the denoiser — LoRA wrappers are pure tensor ops (no graph breaks),
        # and the codec's forward is already decorated with @torch.compiler.disable.
        # Only compile the training denoiser and vae; ema copies are used in validation
        # where a PyTorch inductor bug in quantization pattern matching hits LoRA scaling.
        self.denoiser.compile()
        self.vae.compile()

        # Log all trainable parameters (name, shape, numel)
        trainable = [(name, p) for name, p in self.named_parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for _, p in trainable)
        print("=" * 80)
        print(f"[TrainableParams] {len(trainable)} tensors, "
              f"{total_trainable} total elements ({total_trainable / 1e6:.2f}M)")
        for name, p in trainable:
            print(f"[TrainableParams]   {name}  {list(p.shape)}  numel={p.numel()}")
        print("=" * 80)

    def on_train_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self.ema_tracker.setup_models(net=self.denoiser, ema_net=self.ema_denoiser)
        # Also track vae EMA when decoder is being fine-tuned
        if getattr(self.vae, "finetune_mode", "frozen") != "frozen":
            self.ema_vae.to(torch.float32)
            self.ema_tracker.add_models(net=self.vae, ema_net=self.ema_vae)

    def on_validation_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self.ema_vae.to(torch.float32)

    def on_predict_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self.ema_vae.to(torch.float32)

    def training_step(self, batch, batch_idx):
        _, raw_images, _ = batch

        # The trainer extracts VFM features internally (single DINOv3 forward)
        # and computes the normalized latent z for flow matching.
        loss = self.diffusion_trainer(
            self.denoiser, raw_images, global_step=self.global_step
        )
        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)
        return loss["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        xT, y, _ = batch

        vfm_features = self._extract_vfm_features(y)

        net = self.denoiser if self.eval_original_model else self.ema_denoiser
        # RAECoDSampler runs codec internally; pass raw images as condition
        samples, batch_bpp, batch_y_bpp, batch_z_bpp = self.diffusion_sampler(
            net, xT, y, vfm_features=vfm_features,
        )

        batch_size = y.size(0)
        # Decode latent predictions → pixels via EMA decoder
        samples = self.ema_vae.decode(samples)

        self.log("val_bpp", batch_bpp.mean(), prog_bar=True,
                 on_step=False, on_epoch=True,
                 batch_size=batch_size, sync_dist=True)
        self.log("val_y_bpp", batch_y_bpp.mean(), prog_bar=True,
                 on_step=False, on_epoch=True,
                 batch_size=batch_size, sync_dist=True)
        self.log("val_z_bpp", batch_z_bpp.mean(), prog_bar=True,
                 on_step=False, on_epoch=True,
                 batch_size=batch_size, sync_dist=True)

        return fp2uint8(samples)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        xT, y, _ = batch

        vfm_features = self._extract_vfm_features(y)

        net = self.denoiser if self.eval_original_model else self.ema_denoiser
        # RAECoDSampler runs codec internally; pass raw images as condition
        samples, _, _, _ = self.diffusion_sampler(
            net, xT, y, vfm_features=vfm_features,
        )

        # Decode latent predictions → pixels via EMA decoder
        samples = self.ema_vae.decode(samples)
        return fp2uint8(samples)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Save denoiser, ema_denoiser, diffusion_trainer (always), and
        vae + ema_vae when the decoder is being fine-tuned."""
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.denoiser.state_dict(
            destination=destination, prefix=prefix + "denoiser.", keep_vars=keep_vars)
        self.ema_denoiser.state_dict(
            destination=destination, prefix=prefix + "ema_denoiser.", keep_vars=keep_vars)
        self.diffusion_trainer.state_dict(
            destination=destination, prefix=prefix + "diffusion_trainer.", keep_vars=keep_vars)
        # Save vae weights only when they are being trained (not frozen)
        if getattr(self.vae, "finetune_mode", "frozen") != "frozen":
            self.vae.state_dict(
                destination=destination, prefix=prefix + "vae.", keep_vars=keep_vars)
            self.ema_vae.state_dict(
                destination=destination, prefix=prefix + "ema_vae.", keep_vars=keep_vars)
        return destination
