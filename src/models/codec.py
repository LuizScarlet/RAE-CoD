import math
import torch
import torch.nn as nn
from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste as ste_round
from torch import Tensor
from .vq import VectorQuantizer


class InceptionDWConv2d(nn.Module):
    def __init__(self, split_indexes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        
        self.dwconv_hw = nn.Conv2d(split_indexes[1], split_indexes[1], square_kernel_size, padding=square_kernel_size//2, groups=split_indexes[1])
        self.dwconv_w = nn.Conv2d(split_indexes[2], split_indexes[2], kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=split_indexes[2])
        self.dwconv_h = nn.Conv2d(split_indexes[3], split_indexes[3], kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=split_indexes[3])
        self.split_indexes = split_indexes
        
    def forward(self, x):
        id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)    


class PartialDWConv2d(nn.Module):
    def __init__(self, split_indexes, square_kernel_size=5):
        super().__init__()
        
        self.dwconv_hw = nn.Conv2d(split_indexes[1], split_indexes[1], square_kernel_size, padding=square_kernel_size//2, groups=split_indexes[1])
        self.split_indexes = split_indexes
        
    def forward(self, x):
        id, x_hw = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((id, self.dwconv_hw(x_hw)), dim=1)
    

class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2, use_inception=True):
        super().__init__()
        self.dwconv = InceptionDWConv2d((dim - (dim // 8) * 3, dim // 8, dim // 8, dim // 8)) \
            if use_inception else PartialDWConv2d((dim - (dim // 4), dim // 4))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Conv2d(dim, mlp_ratio * dim * 2, 1)
        self.fc2 = nn.Conv2d(mlp_ratio * dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x1, x2 = self.fc1(x).chunk(2, 1)
        x = self.fc2(self.act(x1) * x2)
        return x + shortcut


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)
    


    
    
class Adapter(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, (in_ch + out_ch) // 2, 1),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, (in_ch + out_ch) // 2, 5, padding=2, groups=(in_ch + out_ch) // 2),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, out_ch, 1),
        )
        self.branch2 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)
    

class AnalysisTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            Downsample(3, 64),
            StarBlock(64),
            StarBlock(64),
            Downsample(64, 128),
            StarBlock(128),
            StarBlock(128),
            Downsample(128, 192),
            StarBlock(192),
            StarBlock(192),
            Downsample(192, 256),
        )

    def forward(self, x):
        x = self.analysis_transform(x)
        return x
    

class LatentFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            Adapter(1280, 256),
            StarBlock(256),
            StarBlock(256),
            Downsample(256, 256),
            StarBlock(256),
            StarBlock(256),
        )

    def forward(self, x):
        x = self.block(x)
        return x
    

class HyperAnalysis(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reduction = nn.Sequential(
            Downsample(256, 256),
            StarBlock(256, use_inception=False),
            StarBlock(256, use_inception=False),
            Downsample(256, 256),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class HyperSynthesis(nn.Module):
    def __init__(self):
        super().__init__()
        self.increase = nn.Sequential(
            Upsample(256, 256),
            StarBlock(256, use_inception=False),
            StarBlock(256, use_inception=False),
            Upsample(256, 256),
        )

    def forward(self, x):
        x = self.increase(x)
        return x
    

class SynthesisTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            Adapter(512, 512),
            StarBlock(512),
            StarBlock(512),
            StarBlock(512),
            StarBlock(512),
            StarBlock(512),
            StarBlock(512),
            Upsample(512, 1152),
        )

    def forward(self, x):
        x = self.synthesis_transform(x)
        return x
    

class SpatialContext(nn.Module):
    def __init__(self, in_ch, use_inception=False):
        super().__init__()
        self.block = nn.Sequential(
            StarBlock(in_ch, use_inception=use_inception),
            StarBlock(in_ch, use_inception=use_inception),
            StarBlock(in_ch, use_inception=use_inception),
            StarBlock(in_ch, use_inception=use_inception),
        )

    def forward(self, x):
        context = self.block(x)
        return context



    

class RateModule(nn.Module):
    def __init__(self):
        super().__init__()

    def _calc_bits_per_batch(self, likelihoods: Tensor) -> Tensor:
        batch_size = likelihoods.shape[0]
        likelihoods = likelihoods.reshape(batch_size, -1)
        return likelihoods.log().sum(1) / -math.log(2)

    def forward(self, num_pixels, y_likelihoods, detail_return=False):
        y_bpp = self._calc_bits_per_batch(y_likelihoods) / num_pixels
        total_bpp = y_bpp

        if detail_return:
            return total_bpp, y_bpp
        
        return total_bpp
    

class RAECodec(nn.Module):
    """RAE-CoD codec with standard entropy coding or a fixed 16-bit bottleneck.

    The standard codec transmits the quantized main latent ``y`` together with
    four 4-bit hyper-latent indices. The ``vq_only`` codec removes the main-
    latent entropy model entirely and conditions diffusion only on the decoded
    hyper latent. This matches the separately fine-tuned 16-bit checkpoint.
    """

    VQ_BITS = 4

    def __init__(
        self,
        vfm_dim: int = 1024,
        use_aux_head: bool = False,
        vq_only: bool = False,
    ):
        super().__init__()

        M = 256
        self.vfm_dim = vfm_dim
        self.vq_only = vq_only
        self.encoder = AnalysisTransform()
        self.vfm_fuse = LatentFusion()
        self.decoder = SynthesisTransform()
        self.h_a = HyperAnalysis()
        self.h_s = HyperSynthesis()
        self.entropy_bottleneck = VectorQuantizer(2**self.VQ_BITS, M, 0.25)
        self.entropy_bottleneck.forward = torch.compiler.disable(
            self.entropy_bottleneck.forward
        )

        # The standard-rate model entropy-codes y. These modules are omitted
        # from the 16-bit model and therefore do not appear in its checkpoint.
        self.masks = {}
        if not self.vq_only:
            context_dim = M * 2
            self.y_in = nn.ModuleList(Adapter(M * 2, context_dim) for _ in range(4))
            self.y_cm = SpatialContext(context_dim)
            self.y_out = nn.ModuleList(Adapter(context_dim, M * 2) for _ in range(4))
            self.gaussian_conditional = GaussianConditional(None)
            self.gaussian_conditional.forward = torch.compiler.disable(
                self.gaussian_conditional.forward
            )
            self.rate = RateModule()

        # Auxiliary semantic-alignment head used only during training.
        self.aux_head = None
        if use_aux_head:
            self.aux_head = nn.Sequential(
                nn.Linear(1152, 1024),
                nn.SiLU(),
                nn.Linear(1024, vfm_dim),
            )

    def _encode(self, x, vfm_features):
        batch_size = x.shape[0]
        y = self.encoder(x)
        if vfm_features is not None:
            token_count = vfm_features.shape[1]
            spatial_size = int(math.sqrt(token_count))
            vfm_spatial = vfm_features.transpose(1, 2).reshape(
                batch_size,
                self.vfm_dim,
                spatial_size,
                spatial_size,
            )
            y = self.vfm_fuse(torch.cat([y, vfm_spatial], dim=1))
        return y

    def _fixed_bpp(self, z, num_pixels):
        bits_per_image = z.shape[-2] * z.shape[-1] * self.VQ_BITS
        return torch.full(
            (z.shape[0],),
            bits_per_image / num_pixels,
            device=z.device,
            dtype=z.dtype,
        )

    @staticmethod
    def _vq_condition(y, hyperprior):
        # Preserve the pretrained 512-channel synthesis-transform interface:
        # the absent main latent occupies the first half and the decoded hyper
        # prior occupies the second half.
        return torch.cat([torch.zeros_like(y), hyperprior], dim=1)

    def _standard_condition(self, y, hyperprior, *, training):
        batch_size, channels, height, width = y.shape
        masks = self.get_mask_four_parts(
            batch_size,
            channels,
            height,
            width,
            device=y.device,
        )
        y_hat = torch.zeros_like(y)
        y_means = torch.zeros_like(y)
        y_scales = torch.zeros_like(y)

        for index, mask in enumerate(masks):
            context = torch.cat([y_hat, hyperprior], dim=1)
            context = self.y_in[index](context)
            context = self.y_cm(context)
            means, scales = self.y_out[index](context).chunk(2, 1)
            means = means * mask
            scales = scales * mask
            y_part = y * mask
            if training:
                y_hat_part = ste_round(y_part - means) + means
            else:
                y_hat_part = torch.round(y_part - means) + means
            y_hat = y_hat + y_hat_part
            y_means = y_means + means
            y_scales = y_scales + scales

        return torch.cat([y_hat, hyperprior], dim=1), y_means, y_scales

    def _add_auxiliary_prediction(self, x_hat, result):
        if self.aux_head is not None:
            tokens = x_hat.flatten(2).transpose(1, 2)
            result["aux_pred"] = self.aux_head(tokens)

    def forward(self, x, vfm_features=None, num_pixels=None):
        if num_pixels is None:
            num_pixels = x.shape[-2] * x.shape[-1]

        y = self._encode(x, vfm_features)
        z = self.h_a(y)
        z_hat, vq_loss, _ = self.entropy_bottleneck(z)
        hyperprior = self.h_s(z_hat)

        if self.vq_only:
            condition = self._vq_condition(y, hyperprior)
            bpp = self._fixed_bpp(z, num_pixels)
        else:
            condition, y_means, y_scales = self._standard_condition(
                y,
                hyperprior,
                training=True,
            )
            _, y_likelihoods = self.gaussian_conditional(y, y_scales, y_means)
            bpp = self.rate(num_pixels=num_pixels, y_likelihoods=y_likelihoods)

        x_hat = self.decoder(condition)
        result = {"sq_loss": bpp, "vq_loss": vq_loss}
        self._add_auxiliary_prediction(x_hat, result)
        return x_hat, result

    def inference(self, x, vfm_features=None, num_pixels=None, hyper_only=False):
        if hyper_only and not self.vq_only:
            raise ValueError(
                "The 16-bit endpoint requires a vq_only codec and its dedicated "
                "checkpoint; it cannot be emulated by dropping y from a standard codec."
            )
        if num_pixels is None:
            num_pixels = x.shape[-2] * x.shape[-1]

        y = self._encode(x, vfm_features)
        z = self.h_a(y)
        z_hat, _, _ = self.entropy_bottleneck(z)
        hyperprior = self.h_s(z_hat)
        z_bpp = self._fixed_bpp(z, num_pixels)

        if self.vq_only:
            condition = self._vq_condition(y, hyperprior)
            y_bpp = torch.zeros_like(z_bpp)
        else:
            condition, y_means, y_scales = self._standard_condition(
                y,
                hyperprior,
                training=False,
            )
            _, y_likelihoods = self.gaussian_conditional(
                y,
                y_scales,
                y_means,
                training=False,
            )
            _, y_bpp = self.rate(
                num_pixels=num_pixels,
                y_likelihoods=y_likelihoods,
                detail_return=True,
            )

        x_hat = self.decoder(condition)
        result = {
            "sq_loss": y_bpp + z_bpp,
            "y_bpp": y_bpp,
            "z_bpp": z_bpp,
        }
        return x_hat, result

    def get_mask_four_parts(self, batch, channel, height, width, device='cuda'):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        if curr_mask_str not in self.masks:
            micro_m0 = torch.tensor(((1., 0), (0, 0)), device=device)
            m0 = micro_m0.repeat((height + 1) // 2, (width + 1) // 2)
            m0 = m0[:height, :width]
            m0 = torch.unsqueeze(m0, 0)
            m0 = torch.unsqueeze(m0, 0)

            micro_m1 = torch.tensor(((0, 1.), (0, 0)), device=device)
            m1 = micro_m1.repeat((height + 1) // 2, (width + 1) // 2)
            m1 = m1[:height, :width]
            m1 = torch.unsqueeze(m1, 0)
            m1 = torch.unsqueeze(m1, 0)

            micro_m2 = torch.tensor(((0, 0), (1., 0)), device=device)
            m2 = micro_m2.repeat((height + 1) // 2, (width + 1) // 2)
            m2 = m2[:height, :width]
            m2 = torch.unsqueeze(m2, 0)
            m2 = torch.unsqueeze(m2, 0)

            micro_m3 = torch.tensor(((0, 0), (0, 1.)), device=device)
            m3 = micro_m3.repeat((height + 1) // 2, (width + 1) // 2)
            m3 = m3[:height, :width]
            m3 = torch.unsqueeze(m3, 0)
            m3 = torch.unsqueeze(m3, 0)

            m = torch.ones((batch, channel // 4, height, width), device=device)
            mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
            mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
            mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
            mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)
            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]
