"""Multilayer perceptron modules for Scene decoder."""

from dataclasses import dataclass
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
import torch


@dataclass
class MLPBlockBuilder:
    """Multilayer perceptron block."""

    dim_h_base: int
    dim_h_mult: float
    dim_o: int

    dim_i: int = None

    def __call__(self, dim_i: int) -> "MLPBlock":
        """Build the module."""

        self.dim_i = dim_i

        return MLPBlock(config=self)

    @property
    def dim_h(self) -> int:
        return int(self.dim_h_base * self.dim_h_mult)


class MLPBlock(nn.Module):
    """
    Multilayer perceptron block.

    Shape:
        (..., C_i) -> (..., C_o)
    """

    def __init__(self, config: MLPBlockBuilder) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        self.proj = nn.Linear(config.dim_i, config.dim_o, bias=True)
        self.base = nn.Sequential(
            nn.LayerNorm(config.dim_i),
            nn.Linear(config.dim_i, config.dim_h_base, bias=True),
            nn.GELU(),
            nn.Linear(config.dim_h_base, config.dim_h, bias=True),
            nn.GELU(),
            nn.Linear(config.dim_h, config.dim_o, bias=True),
        )

    def init(self, bias: float | list[float], std: float) -> None:
        """Initialize parameters."""

        with torch.no_grad():
            self.base[-1].bias[...] = self.base[-1].bias.new_tensor(bias)
        nn.init.normal_(self.base[-1].weight, std=std * 0.5**0.5)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, std=std * 0.5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        # (..., C_i) -> (..., C_o)
        s = self.proj(x)
        # (..., C_i) -> (..., C_o)
        x = checkpoint(self.base, x, use_reentrant=False)
        # Skip connection
        x = x + s
        return x


@dataclass
class MLPBuilder:
    """Multilayer perceptron for attribute prediction."""

    act: nn.Module
    block: MLPBlockBuilder
    init_bias: float | list[float]
    label: str

    dim_i: int = None

    def __call__(self, dim_i: int) -> "MLP":
        """Build the module."""

        self.dim_i = dim_i

        return MLP(config=self)


class MLP(nn.Module):
    """
    Multilayer perceptron for attribute prediction.

    Shape:
        (..., C_i) -> (..., C_o)
    """

    def __init__(self, config: MLPBuilder) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        self.block = config.block(dim_i=config.dim_i)
        self.act = config.act

        # Initialize parameters
        self.block.init(bias=config.init_bias, std=0.2)

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(x.device.type, enabled=False):
            return self.forward_unwrapped(x.float())

    def forward_unwrapped(self, x: Tensor) -> Tensor:
        # (..., C_i) -> (..., C_o)
        x = self.block(x)
        # (..., C_o) -> (..., ...)
        x = self.act(x)
        return x
