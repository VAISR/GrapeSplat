"""Pipeline module for GrapeSplat: Camera and ray decoder."""

from ...data import transform as trf
from ..typing import CamRayDecoderInp, CamRayDecoderOut
from .train import TrainStrat, adapt_train_strat
from dataclasses import dataclass
from torch import Tensor, nn
import torch


@dataclass
class CamRayDecoderBuilder:
    """Camera and ray decoder module."""

    train_strat: TrainStrat

    def __call__(self, base: nn.Module) -> "CamRayDecoder":
        """Build the module."""

        return CamRayDecoder(base=base, config=self)


class CamRayDecoder(nn.Module):
    """Camera and ray decoder module."""

    def __init__(self, base: nn.Module, config: CamRayDecoderBuilder) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        self.base = base

        adapt_train_strat(base, config.train_strat)

    def forward(self, inp: CamRayDecoderInp) -> CamRayDecoderOut:
        x = inp.out_image
        # (..., 3 + 4 + 2) in world + wxyz + [h, w]
        y: Tensor = self.base(x.token_geo, *x.frame_resol)[..., [0, 1, 2, 6, 3, 4, 5, 7, 8]]
        with torch.autocast(y.device.type, enabled=False):
            extr, intr = trf.get_cam_from_enc(x.frame_resol, y)
            ray_d, ray_o = trf.get_ray_unprojected(x.frame_resol, extr, intr)
        return CamRayDecoderOut(
            cam_enc=y,
            cam_extr=extr,
            cam_intr=intr,
            ray_d=ray_d,
            ray_o=ray_o,
        )
