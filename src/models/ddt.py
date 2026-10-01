"""RAEv2's official DiT implementation (verbatim copy).

Adapted from the RAEv2 stage-2 DDT implementation.

Provides: RoPE, RMSNorm, NormAttention, SwiGLUFFN, GaussianFourierEmbedding,
ConditionEmbedder, DDTEncoderBlock, DDTDecoderBlock, DDTFinalLayer,
DiTwDDTHead, DiTwDDTHeadIG.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed


def rotate_half(x):
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class RoPE(nn.Module):
    """2D Rotary Position Embeddings for vision tokens, zero PE for condition tokens."""

    def __init__(self, dim, vis_len, cond_len=0, theta=10000.0):
        super().__init__()
        d, T = dim // 2, int(vis_len ** 0.5)
        vis_freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
        vis_base_angles = torch.outer(torch.arange(T).float(), vis_freqs)
        vis_angles = torch.cat([
            vis_base_angles[:, None].expand(-1, T, -1),
            vis_base_angles[None, :].expand(T, -1, -1),
        ], dim=-1).reshape(vis_len, d)
        cond_angles = torch.zeros(cond_len, dim // 2)
        angles = torch.cat([vis_angles, cond_angles], dim=0).repeat_interleave(2, dim=-1)
        self.register_buffer("freqs_cos", angles.cos())
        self.register_buffer("freqs_sin", angles.sin())

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class NormAttention(nn.Module):
    """Multi-head attention with QK normalization and RoPE."""

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope, attn_mask=None):
        B, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = rope(q), rope(k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        return self.proj(out)


class GaussianFourierEmbedding(nn.Module):
    """Timestep embedder using Gaussian Fourier features + learnable tokens."""

    def __init__(self, hidden_size, n_tokens=4, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.learnable_tokens = nn.Parameter(torch.normal(0, 1 / hidden_size ** 0.5, (n_tokens, hidden_size)))

    def forward(self, t, return_base_embed=False):
        t = t[:, None] * self.W[None, :] * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed)
        if return_base_embed:
            t_embed = t_embed.unsqueeze(1)
            return t_embed, self.learnable_tokens + t_embed
        else:
            return self.learnable_tokens + t_embed.unsqueeze(1)


class DDTEncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * hidden_size * mlp_ratio))

    def forward(self, x, rope, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DDTDecoderBlock(DDTEncoderBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__(hidden_size, num_heads, mlp_ratio)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c, rope, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope, attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaln_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


# =========================================================================
# ConditionEmbedder & DiTwDDTHead — verbatim from RAEv2/src/stage2/models/
# =========================================================================

class ConditionEmbedder(nn.Module):
    def __init__(self, hidden_size, num_classes=1000, context_dim=768, condition_type="label", n_tokens=8,
                 latent_in_channels=768, latent_patch_size=1, n_action_tokens=4):
        super().__init__()
        self.condition_type = condition_type
        self.hidden_size = hidden_size

        if condition_type == "label":
            self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
            self.learnable_tokens = nn.Parameter(torch.normal(0, 1 / hidden_size**0.5, (n_tokens, hidden_size)))
        elif condition_type == "text":
            self.norm = RMSNorm(context_dim)
            self.proj = nn.Linear(context_dim, hidden_size)
        elif condition_type == "nwm":
            self.context_patch_embed = nn.Conv2d(
                latent_in_channels, hidden_size,
                kernel_size=latent_patch_size, stride=latent_patch_size,
            )
            self.n_action_tokens = n_action_tokens
            self.action_proj = nn.Linear(3, hidden_size)
            self.action_tokens = nn.Parameter(
                torch.normal(0, 1 / hidden_size**0.5, (n_action_tokens, hidden_size))
            )
            self.time_emb = GaussianFourierEmbedding(hidden_size, n_tokens=1)
        else:
            raise ValueError(f"Unknown condition_type: {condition_type}")

    def forward(self, y) -> torch.Tensor:
        if self.condition_type == "nwm":
            ctx = y["context_latents"]
            B, K = ctx.shape[:2]
            patches = self.context_patch_embed(ctx.flatten(0, 1))
            patches = patches.flatten(2).transpose(1, 2)
            patches = patches.reshape(B, K * patches.shape[1], -1)
            act_tokens = self.action_tokens + self.action_proj(y["action"]).unsqueeze(1)
            time_token = self.time_emb(y["rel_time"].squeeze(-1))
            return torch.cat([patches, act_tokens, time_token], dim=1)
        if self.condition_type == "label":
            return self.learnable_tokens + self.embedding_table(y).unsqueeze(1)
        else:
            return self.proj(self.norm(y))


class DiTwDDTHead(nn.Module):
    def __init__(
        self,
        input_size=16,
        in_channels=768,
        patch_size=[1, 1],
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
        mlp_ratio=4.0,
        num_classes=1000,
        condition_type="label",
        context_dim=768,
        cond_arch=None,
        use_cfg_conds=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.enc_hidden_size, dec_hidden_size = hidden_size
        self.num_enc_blocks, self.num_dec_blocks = depth
        self.s_patch_size, self.x_patch_size = patch_size
        enc_num_heads, dec_num_heads = num_heads

        self.use_cfg_conds = use_cfg_conds

        self.s_embedder = PatchEmbed(input_size, self.s_patch_size, in_channels, self.enc_hidden_size)
        self.x_embedder = PatchEmbed(input_size, self.x_patch_size, in_channels, dec_hidden_size)
        self.s_projector = nn.Linear(self.enc_hidden_size, dec_hidden_size) if self.enc_hidden_size != dec_hidden_size else nn.Identity()

        self.num_cond_tokens = cond_arch.num_t_tokens + cond_arch.num_c_tokens
        self.t_embedder = GaussianFourierEmbedding(self.enc_hidden_size, cond_arch.num_t_tokens)
        self.ctx_embedder = ConditionEmbedder(
            self.enc_hidden_size, num_classes, context_dim, condition_type, cond_arch.num_c_tokens,
            latent_in_channels=in_channels,
            latent_patch_size=self.s_patch_size,
            n_action_tokens=getattr(cond_arch, "n_action_tokens", 4),
        )
        if self.use_cfg_conds:
            self.num_cond_tokens += cond_arch.num_cfg_omega_tokens
            self.cfg_w_embedder = GaussianFourierEmbedding(self.enc_hidden_size, cond_arch.num_cfg_omega_tokens)

        self.blocks = []
        for _ in range(self.num_enc_blocks):
            self.blocks.append(DDTEncoderBlock(self.enc_hidden_size, enc_num_heads, mlp_ratio))
        for _ in range(self.num_dec_blocks):
            self.blocks.append(DDTDecoderBlock(dec_hidden_size, dec_num_heads, mlp_ratio))
        self.blocks = nn.ModuleList(self.blocks)

        self.final_layer = DDTFinalLayer(dec_hidden_size, self.x_patch_size, in_channels)
        self.enc_rope = RoPE(self.enc_hidden_size // enc_num_heads, self.s_embedder.num_patches, self.num_cond_tokens)
        self.dec_rope = RoPE(dec_hidden_size // dec_num_heads, self.x_embedder.num_patches)

        self.initialize_weights()

    def initialize_weights(self):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        if hasattr(self.ctx_embedder, "mlp"):
            nn.init.normal_(self.ctx_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.ctx_embedder.mlp[2].weight, std=0.02)
        if hasattr(self.ctx_embedder, "embedding_table"):
            nn.init.normal_(self.ctx_embedder.embedding_table.weight, std=0.02)

        for block in self.blocks:
            if hasattr(block, "adaln_modulation"):
                nn.init.constant_(block.adaln_modulation[-1].weight, 0)
                nn.init.constant_(block.adaln_modulation[-1].bias, 0)

        t_embedders = ["t_embedder", "cfg_w_embedder"]
        for t_embedder in t_embedders:
            if hasattr(self, t_embedder):
                nn.init.normal_(getattr(self, t_embedder).mlp[0].weight, std=0.02)
                nn.init.normal_(getattr(self, t_embedder).mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """[N, T, patch_size**2 * C] -> [N, C, H, W]"""
        h, c = int(x.shape[1] ** 0.5), self.in_channels
        x = x.reshape(x.shape[0], h, h, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h*p, h*p)
        return x

    def _build_sequence(self, x, t, condition_kwargs):
        seq = []
        seq.append(self.s_embedder(x))
        t_emb_base, t_emb = self.t_embedder(t, return_base_embed=True)
        seq.append(t_emb)
        if self.use_cfg_conds:
            seq.append(self.cfg_w_embedder(condition_kwargs["omega"]))
        seq.append(self.ctx_embedder(condition_kwargs["context"]))
        seq = torch.cat(seq, dim=1)
        return seq, t_emb_base

    def _build_attn_mask(self, seq, condition_kwargs):
        attn_mask = torch.ones((seq.shape[0], seq.shape[1]), device=seq.device)
        cond_mask = condition_kwargs.get("attn_mask")
        if cond_mask is not None:
            attn_mask[:, -cond_mask.shape[1]:] = cond_mask
        attn_mask = (1.0 - attn_mask[:, None, None, :]) * torch.finfo(seq.dtype).min
        return attn_mask

    def forward(self, x, t, **condition_kwargs):
        seq, t_emb_base = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(seq, condition_kwargs)
        for i in range(self.num_enc_blocks):
            seq = self.blocks[i](seq, self.enc_rope, attn_mask)
        seq = self.s_projector(F.silu(t_emb_base + seq[:, :self.s_embedder.num_patches, :]))

        x = self.x_embedder(x)
        for i in range(self.num_dec_blocks):
            x = self.blocks[self.num_enc_blocks + i](x, seq, self.dec_rope)

        x = self.final_layer(x, seq)
        x = self.unpatchify(x, self.x_patch_size)

        return x


class DiTwDDTHeadIG(DiTwDDTHead):
    """DiTwDDTHead with Internal Guidance (IG) — verbatim from RAEv2."""

    def __init__(self, base_model_depth=8, **kwargs):
        super().__init__(**kwargs)
        self.base_model_depth = base_model_depth

        self.base_final_layer = DDTFinalLayer(self.enc_hidden_size, self.s_patch_size, self.in_channels)
        nn.init.constant_(self.base_final_layer.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.base_final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.base_final_layer.linear.weight, 0)
        nn.init.constant_(self.base_final_layer.linear.bias, 0)

    def forward(self, x, t, **condition_kwargs):
        x_base = None
        seq, t_emb_base = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(seq, condition_kwargs)
        for i in range(self.num_enc_blocks):
            seq = self.blocks[i](seq, self.enc_rope, attn_mask)
            if (i + 1) == self.base_model_depth:
                x_base = seq[:, :self.s_embedder.num_patches, :]
        seq = self.s_projector(F.silu(t_emb_base + seq[:, :self.s_embedder.num_patches, :]))

        x = self.x_embedder(x)
        if hasattr(self, 'y_embedder_x') and 'context' in condition_kwargs:
            x = x + self.y_embedder_x(condition_kwargs['context'])
        for i in range(self.num_dec_blocks):
            x = self.blocks[self.num_enc_blocks + i](x, seq, self.dec_rope)

        x = self.final_layer(x, seq)
        x = self.unpatchify(x, self.x_patch_size)

        x_base = F.silu(t_emb_base + x_base)
        x_base = self.base_final_layer(x_base, x_base)
        x_base = self.unpatchify(x_base, self.s_patch_size)

        return x, x_base
