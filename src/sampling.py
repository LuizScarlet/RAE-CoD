"""RAEv2-aligned ODE sampler with Internal Guidance (IG) for latent-space diffusion.

Uses RAEv2 time convention (t=0→clean, t=1→noise) with Euler ODE integration
from t=1→0. Supports IG, CFG, and combined IG+CFG guidance modes.

The sampler runs the codec internally (on the first step) to produce condition
tokens, then integrates the ODE using the guidance-wrapped model forward.
"""
import torch



def _expand_t(t, x):
    """Expand scalar t to match spatial dims of x."""
    return t.view(t.size(0), *([1] * (len(x.size()) - 1)))


class RAECoDSampler(torch.nn.Module):
    """ODE sampler with Internal Guidance for latent-space diffusion.

    Integrates the probability flow ODE from t=1 (noise) to t=0 (clean)
    using Euler steps, with guidance applied via IG and/or CFG.

    The model is expected to be a RAECoD whose forward
    returns (x_full, x_base) from DiTwDDTHeadIG.
    """

    def __init__(
        self,
        num_steps=50,
        time_dist_shift=8.0,
        t_eps=0.05,
        prediction="x",
        ig_scale=1.5,
        ig_t_min=0.0,
        ig_t_max=1.0,
        cfg_scale=1.0,
        cfg_t_min=0.0,
        cfg_t_max=1.0,
        hyper_only=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.time_dist_shift = time_dist_shift
        self.t_eps = t_eps
        self.prediction = prediction
        self.ig_scale = ig_scale
        self.ig_t_min = ig_t_min
        self.ig_t_max = ig_t_max
        self.cfg_scale = cfg_scale
        self.cfg_t_min = cfg_t_min
        self.cfg_t_max = cfg_t_max
        self.hyper_only = hyper_only

    def _convert_model_pred(self, output, xt, t):
        """Convert model output to velocity (drift) prediction."""
        if self.prediction == "velocity":
            return output
        elif self.prediction == "x":
            t_safe = _expand_t(t, xt).clamp_min(self.t_eps)
            return (xt - output) / t_safe
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction}")

    def _apply_ig(self, x_full, x_base, t):
        """Apply Internal Guidance: base + ig_scale * (full - base)."""
        in_interval = (t >= self.ig_t_min) & (t <= self.ig_t_max)
        in_interval = in_interval.view(-1, *([1] * (x_full.ndim - 1)))
        ig_out = torch.where(
            in_interval,
            x_base + self.ig_scale * (x_full - x_base),
            x_full,
        )
        return ig_out

    def _model_forward_ig(self, net, x, t, cond_tokens):
        """Forward with IG only (no CFG doubling)."""
        x_full, x_base = net(x, t, context=cond_tokens)
        ig_out = self._apply_ig(x_full, x_base, t)
        return self._convert_model_pred(ig_out, x, t)

    def _model_forward_cfg(self, net, x, t, cond_tokens, null_cond_tokens):
        """Forward with CFG (doubled batch, no IG)."""
        x_doubled = torch.cat([x, x], dim=0)
        t_doubled = t.repeat(2)
        ctx_doubled = torch.cat([cond_tokens, null_cond_tokens], dim=0)
        model_out = net(x_doubled, t_doubled, context=ctx_doubled)
        # IG model returns (full, base) — use full only for CFG
        if isinstance(model_out, tuple):
            model_out = model_out[0]
        cond_out, uncond_out = model_out.chunk(2, dim=0)
        in_interval = (t >= self.cfg_t_min) & (t <= self.cfg_t_max)
        in_interval = in_interval.view(-1, *([1] * (cond_out.ndim - 1)))
        guided = torch.where(
            in_interval,
            uncond_out + self.cfg_scale * (cond_out - uncond_out),
            cond_out,
        )
        return self._convert_model_pred(guided, x, t)

    def _model_forward_ig_and_cfg(self, net, x, t, cond_tokens, null_cond_tokens):
        """Forward with combined IG + CFG (doubled batch, IG on both branches)."""
        x_doubled = torch.cat([x, x], dim=0)
        t_doubled = t.repeat(2)
        ctx_doubled = torch.cat([cond_tokens, null_cond_tokens], dim=0)
        x_full, x_base = net(x_doubled, t_doubled, context=ctx_doubled)

        full_c, full_u = x_full.chunk(2, dim=0)
        base_c, base_u = x_base.chunk(2, dim=0)

        # Apply IG to both cond and uncond branches
        ig_cond = self._apply_ig(full_c, base_c, t)
        ig_uncond = self._apply_ig(full_u, base_u, t)

        # Apply CFG
        in_interval = (t >= self.cfg_t_min) & (t <= self.cfg_t_max)
        in_interval = in_interval.view(-1, *([1] * (ig_cond.ndim - 1)))
        guided = torch.where(
            in_interval,
            ig_uncond + self.cfg_scale * (ig_cond - ig_uncond),
            ig_cond,
        )
        return self._convert_model_pred(guided, x, t)

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, net, noise, condition, vfm_features=None):
        """Run ODE sampling from noise to clean latent.

        Args:
            net: RAECoD model.
            noise: Initial noise (B, C, H, W) in latent space.
            condition: Raw images (B, 3, H_img, W_img) for codec.
            vfm_features: DINOv3 features (B, T, D).

        Returns:
            (samples, bpp): Final clean latent and bits-per-pixel from codec.
        """
        # Run codec to get condition tokens
        cond, codec_res = net.y_embedder.inference(
            condition,
            vfm_features=vfm_features,
            hyper_only=self.hyper_only,
        )
        cond_tokens = cond.flatten(2, 3).transpose(1, 2)  # (B, h*w, cond_dim)
        null_cond_tokens = torch.zeros_like(cond_tokens)
        bpp, y_bpp, z_bpp = codec_res["sq_loss"], codec_res["y_bpp"], codec_res["z_bpp"]

        # Build time grid: t from 1.0 → 0.0, with time shift
        t_grid = torch.linspace(1.0, 0.0, self.num_steps + 1, device=noise.device)
        shift = self.time_dist_shift
        t_grid = shift * t_grid / (1 + (shift - 1) * t_grid)

        # Determine guidance mode
        use_ig = self.ig_scale != 1.0
        use_cfg = self.cfg_scale != 1.0

        B = noise.shape[0]
        x = noise

        for i in range(self.num_steps):
            h = t_grid[i] - t_grid[i + 1]
            t_batch = torch.full((B,), t_grid[i].item(), device=x.device)

            if use_ig and use_cfg:
                drift = self._model_forward_ig_and_cfg(net, x, t_batch, cond_tokens, null_cond_tokens)
            elif use_ig:
                drift = self._model_forward_ig(net, x, t_batch, cond_tokens)
            elif use_cfg:
                drift = self._model_forward_cfg(net, x, t_batch, cond_tokens, null_cond_tokens)
            else:
                # No guidance — use full model output directly
                model_out = net(x, t_batch, context=cond_tokens)
                if isinstance(model_out, tuple):
                    model_out = model_out[0]
                drift = self._convert_model_pred(model_out, x, t_batch)

            x = x - h * drift

        return x, bpp, y_bpp, z_bpp
