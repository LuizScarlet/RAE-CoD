import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.no_grad import no_grad


def _expand_t(t, x):
    """Expand scalar t to match spatial dims of x."""
    return t.view(t.size(0), *([1] * (len(x.size()) - 1)))


class RAECoDTrainer(nn.Module):
    """RAEv2-aligned training loop for latent-space diffusion with Internal Guidance.

    Uses RAEv2 time convention (t=0→clean, t=1→noise) with:
    - DiTwDDTHeadIG dual-output (full + base) with IG loss
    - CFG dropout on codec condition tokens
    - Flow matching loss converted from x-pred to velocity

    Loss terms (matching RAEv2 pretraining):
    - fm_loss: flow matching MSE on full model output
    - loss_base: flow matching MSE on base (IG early exit) output
    - sq_loss: standard-codec rate loss (disabled for the fixed 16-bit codec)
    - aux_cos_loss: codec auxiliary cosine loss (masked for CFG-dropped samples)
    """

    def __init__(
        self,
        lambda_rate: float = 1.0,
        lambda_rate_schedule: list = None,
        lambda_aux_cos: float = 0.0,
        t_eps: float = 0.05,
        base_model_coeff: float = 1.0,
        cfg_dropout_prob: float = 0.1,
        prediction: str = "x",
        lognorm_t=True,
        timeshift=8.0,
        encoder: nn.Module = None,
        normalization_stat_path: str = "",
        eps: float = 1e-5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_rate = lambda_rate
        self.lambda_rate_schedule = lambda_rate_schedule  # list of [step, value] pairs
        self.lambda_aux_cos = lambda_aux_cos
        self.t_eps = t_eps
        self.base_model_coeff = base_model_coeff
        self.cfg_dropout_prob = cfg_dropout_prob
        self.prediction = prediction
        self.lognorm_t = lognorm_t
        self.timeshift = timeshift
        self.eps = eps

        # Frozen VFM encoder (DINOv3)
        self.encoder = encoder
        no_grad(self.encoder)

        # Load normalization stats for encoding raw features → latent
        self.do_normalization = False
        if normalization_stat_path:
            stats = torch.load(normalization_stat_path, map_location="cpu", weights_only=False)
            mean = stats.get("mean", None)
            var = stats.get("var", None)
            if mean is not None:
                self.register_buffer("latent_mean", mean)
            if var is not None:
                self.register_buffer("latent_var", var)
            self.do_normalization = True

    def normalize_latent(self, z):
        """Normalize DINOv3 features: (z - mean) / sqrt(var + eps)."""
        if self.do_normalization:
            mean = self.latent_mean.to(z.device)
            var = self.latent_var.to(z.device)
            z = (z - mean) / torch.sqrt(var + self.eps)
        return z

    def _get_lambda_rate(self, global_step):
        """Resolve lambda_rate based on schedule and current step.

        If lambda_rate_schedule is set, finds the value for the current step.
        Schedule is a list of [step, value] pairs sorted by step.
        The last pair whose step <= global_step determines the value.
        Falls back to self.lambda_rate if no schedule or before first entry.
        """
        if not self.lambda_rate_schedule:
            return self.lambda_rate
        current_val = self.lambda_rate  # default before first scheduled step
        for step, val in self.lambda_rate_schedule:
            if global_step >= step:
                current_val = val
            else:
                break
        return current_val

    def _convert_model_pred(self, output, xt, t):
        """Convert model output to velocity prediction (RAEv2 convention)."""
        if self.prediction == "velocity":
            return output
        elif self.prediction == "x":
            t_safe = _expand_t(t, xt).clamp_min(self.t_eps)
            return (xt - output) / t_safe
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction}")

    def _compute_loss(self, output, vt, xt, t):
        """Compute per-element MSE loss, converting x-pred to velocity first."""
        output = self._convert_model_pred(output, xt, t)
        return (output - vt) ** 2

    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def __call__(self, net, raw_images, global_step=0):
        batch_size = raw_images.shape[0]

        # === Shared frozen VFM forward (computed ONCE) ===
        with torch.no_grad():
            vfm_features = self.encoder(raw_images)  # (B, num_patches, 1024)

        # === Convert VFM features to normalized latent (diffusion target) ===
        B, T, D = vfm_features.shape
        H_vfm = int(math.sqrt(T))
        W_vfm = T // H_vfm
        vfm_spatial = vfm_features.transpose(1, 2).reshape(B, D, H_vfm, W_vfm)  # (B, 1024, 16, 16)
        z = self.normalize_latent(vfm_spatial)  # normalized latent = diffusion target (x1 in RAEv2)

        # === Run codec on raw images → condition tokens + codec losses ===
        cond, codec_res = net.y_embedder(raw_images, vfm_features=vfm_features)
        # (B, cond_dim, h, w) → (B, h*w, cond_dim) token sequence
        cond_tokens = cond.flatten(2, 3).transpose(1, 2)

        # === CFG dropout on condition tokens (replace with zeros) ===
        cfg_mask = torch.rand(batch_size, device=cond_tokens.device) < self.cfg_dropout_prob  # (B,) bool
        null_cond_tokens = torch.zeros_like(cond_tokens)
        cond_tokens = torch.where(
            cfg_mask.view(-1, 1, 1),
            null_cond_tokens,
            cond_tokens,
        )

        # === Flow matching time sampling (RAEv2 convention: t=0→clean, t=1→noise) ===
        if self.lognorm_t:
            base_t = torch.randn(batch_size, device=z.device, dtype=torch.float32).sigmoid()
        else:
            base_t = torch.rand(batch_size, device=z.device, dtype=torch.float32)
        # Apply time shift: t' = shift * t / (1 + (shift - 1) * t)
        t = self.timeshift * base_t / (1 + (self.timeshift - 1) * base_t)
        t = t.to(z.dtype)

        noise = torch.randn_like(z)

        # RAEv2 convention: xt = (1-t)*x1 + t*x0 where x1=clean, x0=noise
        t_expanded = _expand_t(t, z)
        xt = (1 - t_expanded) * z + t_expanded * noise
        # Target velocity: vt = (xt - x1) / t.clamp_min(t_eps)
        vt = (xt - z) / t_expanded.clamp_min(self.t_eps)

        # === Model forward (IG dual-output: full + base) ===
        x_full, x_base = net(xt, t, context=cond_tokens)

        # === Flow matching losses (x-pred → velocity conversion, matching RAEv2 compute_loss) ===
        fm_loss = self._compute_loss(x_full, vt, xt, t).mean()
        loss_base = self._compute_loss(x_base, vt, xt, t).mean()

        # === Auxiliary alignment loss ===
        # Mask for unconditional samples: (B,) float with 0 for uncond, 1 for cond
        cond_float_mask = (~cfg_mask).float()  # (B,)

        # Auxiliary cosine loss: codec output → VFM feature space
        aux_cos_loss_mean = torch.tensor(0.0, device=z.device)
        if self.lambda_aux_cos > 0:
            aux_pred = codec_res["aux_pred"]  # (B, num_patches, vfm_dim)
            aux_cos_loss = 1 - F.cosine_similarity(aux_pred, vfm_features.detach(), dim=-1)  # (B, num_patches)
            aux_cos_loss = aux_cos_loss * cond_float_mask[:, None]
            aux_cos_loss_mean = aux_cos_loss.mean()

        # === Rate loss (mask unconditional) ===
        raw_sq_loss = codec_res["sq_loss"]
        sq_mask = cond_float_mask.view(-1, *([1] * (raw_sq_loss.ndim - 1)))
        sq_loss = raw_sq_loss * sq_mask

        raw_vq_loss = codec_res["vq_loss"]
        vq_mask = cond_float_mask.view(-1, *([1] * (raw_vq_loss.ndim - 1)))
        vq_loss = raw_vq_loss * vq_mask

        sq_loss_mean = sq_loss.mean()
        vq_loss_mean = vq_loss.mean()

        # The dedicated 16-bit codec has a fixed VQ payload and no learned
        # rate term. Standard checkpoints retain the scheduled rate objective.
        if getattr(net.y_embedder, "vq_only", False):
            current_lambda_rate = 0.0
        else:
            current_lambda_rate = self._get_lambda_rate(global_step)

        # === Total loss (matches RAEv2 transport.training_losses + codec losses) ===
        loss = (
            fm_loss
            + self.base_model_coeff * loss_base
            + current_lambda_rate * sq_loss_mean
            + 0.25 * vq_loss_mean
            + self.lambda_aux_cos * aux_cos_loss_mean
        )

        return dict(
            fm_loss=fm_loss,
            loss_base=loss_base,
            sq_loss=sq_loss_mean,
            vq_loss=vq_loss_mean,
            aux_cos_loss=aux_cos_loss_mean,
            lambda_rate=torch.tensor(current_lambda_rate),
            loss=loss,
        )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """No extra state to save."""
        if destination is None:
            destination = {}
        return destination
