import torch


class BaseAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def decode(self, x):
        return self._impl_decode(x).to(torch.bfloat16)

    def _impl_decode(self, x):
        raise NotImplementedError


def fp2uint8(x):
    return torch.clip_((x + 1) * 127.5 + 0.5, 0, 255).to(torch.uint8)
