"""Spatial UNet modules."""

from ....typing import UNetInp, UNetOut
from .spconv import (
    SpConvRefBlockBuilder,
    UNetSpConvDecBlockBuilder,
    UNetSpConvEncBlockBuilder,
)
from dataclasses import asdict, dataclass
from torch import nn
from torchsparse import SparseTensor


@dataclass
class UNetBuilder:
    """Spatial UNet for feature refinement."""

    dim_h_base: int
    dim_h_mult: list[float]
    spconv: "UNetSpConvStrategy"

    dim_i: int = None

    def __call__(self, dim_i: int) -> "UNet":
        """Build the module."""

        self.dim_i = dim_i
        return UNet(config=self)

    @property
    def dim_o(self) -> int:
        return int(self.dim_h_base * self.dim_h_mult[0])


class UNet(nn.Module):
    """Spatial UNet for feature refinement."""

    def __init__(self, config: UNetBuilder) -> None:
        super().__init__()

        dims = [config.dim_i] + [int(config.dim_h_base * m) for m in config.dim_h_mult]

        self.config = config

        # Initialize sub-modules in execution order
        self.enc_conv = nn.ModuleList(
            [
                UNetSpConvEncBlockBuilder(
                    **asdict(config.spconv),
                    dim_h_base=dims[i + 1],
                    dim_i=dims[i],
                    dim_o=dims[i + 1],
                )()
                for i in range(0, len(dims) - 2, 1)
            ]
        )
        self.btnk_conv = SpConvRefBlockBuilder(
            **asdict(config.spconv),
            dim_h_base=dims[-1],
            dim_i=dims[-2],
            dim_o=dims[-1],
        )()
        self.dec_conv = nn.ModuleList(
            [
                UNetSpConvDecBlockBuilder(
                    **asdict(config.spconv),
                    dim_h_base=dims[i],
                    dim_i=dims[i + 1],
                    dim_o=dims[i],
                )()
                for i in range(len(dims) - 2, 0, -1)
            ]
        )

    def forward(self, inp: UNetInp) -> UNetOut:
        # NOTE: SparseTensor is saved by checkpoint, so kernel map caches persist across recompute.
        x = inp.grid
        skip: list[SparseTensor] = []

        # Encoder
        for enc_conv in self.enc_conv:
            x, s = enc_conv(x)
            skip.append(s)
        # Bottleneck
        x = self.btnk_conv(x)
        # Decoder
        for dec_conv in self.dec_conv:
            x = dec_conv(x, skip.pop())

        return UNetOut(grid=x)


@dataclass
class UNetSpConvStrategy:
    """Spatial UNet convolution strategy."""

    depth: int
    dim_h_mult: float
    kernel: int
