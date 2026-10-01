from typing import Any, Dict

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

class SimpleEMA(Callback):
    def __init__(self,
                 decay: float = 0.9999,
                 every_n_steps: int = 1,
                 ):
        super().__init__()
        self.decay = decay
        self.every_n_steps = every_n_steps
        self._stream = torch.cuda.Stream()
        self.previous_step = 0

    def setup_models(self, net: nn.Module, ema_net: nn.Module):
        self.net_params = list(net.parameters())
        self.ema_params = list(ema_net.parameters())

    def add_models(self, net: nn.Module, ema_net: nn.Module):
        """Add another source/EMA model pair to the same EMA update.

        ``RAECoDLightningModule`` uses this to track the jointly trained
        RAEv2 decoder in addition to the denoiser. Parameter order is preserved
        because ``ema_net`` is a deepcopy of ``net``.
        """
        if not hasattr(self, "net_params") or not hasattr(self, "ema_params"):
            raise RuntimeError("Call setup_models() before add_models().")

        net_params = list(net.parameters())
        ema_params = list(ema_net.parameters())
        if len(net_params) != len(ema_params):
            raise ValueError(
                "EMA model parameter count does not match source model: "
                f"{len(ema_params)} != {len(net_params)}"
            )
        for index, (param, ema_param) in enumerate(zip(net_params, ema_params)):
            if param.shape != ema_param.shape:
                raise ValueError(
                    f"EMA parameter shape mismatch at index {index}: "
                    f"{tuple(ema_param.shape)} != {tuple(param.shape)}"
                )

        self.net_params.extend(net_params)
        self.ema_params.extend(ema_params)

    def ema_step(self):
        @torch.no_grad()
        def ema_update(ema_model_tuple, current_model_tuple, decay):
            torch._foreach_mul_(ema_model_tuple, decay)
            torch._foreach_add_(
                ema_model_tuple, current_model_tuple, alpha=(1.0 - decay),
            )

        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            ema_update(self.ema_params, self.net_params, self.decay)
        assert self.ema_params[0].dtype == torch.float32

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if trainer.global_step == self.previous_step:
            return
        self.previous_step = trainer.global_step
        if trainer.global_step % self.every_n_steps == 0:
            self.ema_step()


    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.every_n_steps = state_dict["every_n_steps"]
