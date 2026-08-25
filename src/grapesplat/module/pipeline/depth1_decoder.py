"""Pipeline module for GrapeSplat: Depth 1 decoder."""

from ..typing import Depth1DecoderInp, Depth1DecoderOut
from .train import TrainStrat, adapt_train_strat
from dataclasses import dataclass
from torch import Tensor, nn
import torch


@dataclass
class Depth1DecoderBuilder:
    """Depth 1 decoder module."""

    train_strat: TrainStrat

    def __call__(self, base: nn.Module) -> "Depth1Decoder":
        """Build the module."""

        return Depth1Decoder(base=base, config=self)


class Depth1Decoder(nn.Module):
    """Depth 1 decoder module."""

    def __init__(self, base: nn.Module, config: Depth1DecoderBuilder) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        self.base = base

        adapt_train_strat(base, config.train_strat)

    def forward(self, inp: Depth1DecoderInp) -> Depth1DecoderOut:
        x = inp.out_image
        y: tuple[Tensor, Tensor] = self.base(x.token_geo, *x.frame_resol, x.patch_index_geo)
        # [(B, S, H, W, 1); 2] -> [(B, S, 1, H, W); 2]
        y = tuple(v.movedim(-1, -3).contiguous() for v in y)
        eps = torch.finfo(y[0].dtype).eps
        return Depth1DecoderOut(
            depth=y[0].clamp(eps, 1.0 / eps),
            depth_conf=y[1].clamp(1.0, 1.0 / eps),
        )
