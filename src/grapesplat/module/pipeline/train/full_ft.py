"""Full Fine-Tuning training strategy."""

from .strat import StrategyBuilder
from .util import debug_train_strat, get_param_size
from dataclasses import dataclass
from torch import nn


@dataclass
class FullFTBuilder(StrategyBuilder):
    """Full Fine-Tuning adapter."""

    grad: bool

    def __call__(self, base: nn.Module) -> "FullFT":
        if isinstance(base, FullFT):
            return base
        if self.target:
            raise ValueError(f"Unsupported target: expected [], got {self.target}")
        return FullFT(base=base, config=self)

    @property
    def name(self) -> str:
        return "full_tune" if self.grad else "full_lock"


class FullFT(nn.Module):
    """Full Fine-Tuning adapter."""

    def __init__(self, base: nn.Module, config: FullFTBuilder) -> None:
        super().__init__()

        self.base = base
        self.config = config

        base.requires_grad_(config.grad)
        debug_train_strat(config.name, get_param_size(base, config.grad), type(base).__name__)

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)
