"""Bridge modules for Scene decoder."""

from ...typing import FeatBridgeInp, FeatBridgeOut, ImageEncoderOut
from ..train import WithTrainStrat, adapt_train_strat
from dataclasses import dataclass
from torch import Tensor, nn
from torch.nn import functional as nnf
import torch


@dataclass
class FeatBridgeBuilder:
    """Feature bridge for lifting features from patch to pixel."""

    geo: "FeatBridgeConfig"
    sem: "FeatBridgeConfig"

    dim_o: int = None

    def __call__(self, base_geo: nn.Module, base_sem: nn.Module, dim_o: int) -> "FeatBridge":
        """Build the module."""

        self.dim_o = dim_o * int(self.geo["enabled"] + self.sem["enabled"])

        return FeatBridge(base_geo=base_geo, base_sem=base_sem, config=self)


class FeatBridge(nn.Module):
    """Feature bridge for lifting features from patch to pixel."""

    def __init__(self, base_geo: nn.Module, base_sem: nn.Module, config: FeatBridgeBuilder) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        if config.geo["enabled"]:
            self.base_geo = base_geo
            adapt_train_strat(base_geo, config.geo["train_strat"])
        if config.sem["enabled"]:
            self.base_sem = base_sem
            adapt_train_strat(base_sem, config.sem["train_strat"])

    def forward(self, inp: FeatBridgeInp) -> FeatBridgeOut:
        x = inp.out_image

        y: list[Tensor] = []
        if self.config.geo["enabled"]:
            # [(B, S, 1 + R_g + H_P_g * W_P_g, C_g); L] -> (B, S, C_o, H, W)
            y_geo = self.base_geo(x.token_geo, *x.frame_resol, x.patch_index_geo)
            y.append(y_geo)
        if self.config.sem["enabled"]:
            # [(B, S, 1 + R_s + H_P_s * W_P_s, C_s); 1] -> (B, S, C_o, H, W)
            y_sem = self.base_sem(inp.image, x)
            y.append(y_sem)
        # (B, S, C_g + C_s, H, W)
        y = torch.cat(y, dim=-3)

        return FeatBridgeOut(feat=y)


class FeatBridgeConfig(WithTrainStrat):
    enabled: bool


class TokenSemProject(nn.Module):
    """
    Semi-orthogonal token readout lifted patch-to-pixel by image-guided filter upsampling.

    Shape:
        [(B, S, 3, H, W), (B, S, H_P_s * W_P_s, C_s)] -> (B, S, C_o, H, W)
    """

    def __init__(self, dim_i: int, dim_o: int, patch: int) -> None:
        super().__init__()

        # Initialize sub-modules in execution order
        self.patch = patch
        self.box = nn.AvgPool2d(5, stride=1, padding=2, count_include_pad=False)
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]), persistent=False)
        self.register_buffer("wght", torch.empty(dim_i, dim_o), persistent=True)

        # Initialize parameters
        nn.init.orthogonal_(self.wght)

    def forward(self, image: Tensor, out_image: ImageEncoderOut) -> Tensor:
        frame_resol = out_image.frame_resol
        # (B, S, H_P_s * W_P_s, C_s)
        # -> (B * S, H_P_s * W_P_s, C_s) -> (B * S, C_o, H_P_s, W_P_s)
        p = (
            out_image.token_sem[-1][..., out_image.patch_index_sem :, :]
            .flatten(0, -3)
            .matmul(self.wght)
            .mT.unflatten(-1, tuple(v // self.patch for v in frame_resol))
        )
        with torch.autocast(p.device.type, enabled=False):
            p = p.float().contiguous()
            # (B, S, 3, H, W) in [0, 1] -> (B * S, 1, H, W) in [0, 1]
            g = image.flatten(0, -4).mul(self.luma[:, None, None]).sum(dim=-3, keepdim=True)
            g_lo = nnf.interpolate(g, p.shape[-2:], mode="area")
            g_avg: Tensor = self.box(g_lo)
            p_avg: Tensor = self.box(p)
            cov: Tensor = self.box(g_lo.mul(p)) - g_avg.mul(p_avg)
            g_var: Tensor = self.box(g_lo.square()) - g_avg.square()
            gain = cov.div(g_var.clamp(min=1e-4))
            bias = p_avg.sub(g_avg.mul(gain))
            # (B * S, C_o, H_P_s, W_P_s) -> (B * S, C_o, H, W)
            gain = nnf.interpolate(gain, frame_resol, align_corners=False, mode="bilinear")
            bias = nnf.interpolate(bias, frame_resol, align_corners=False, mode="bilinear")
            # (B * S, C_o, H, W) -> (B, S, C_o, H, W)
            y = g.mul(gain).add(bias).unflatten(0, image.shape[0:-3])

        return y
