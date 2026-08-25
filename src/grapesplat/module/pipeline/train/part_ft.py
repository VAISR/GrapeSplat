"""Partial Fine-Tuning training strategy."""

from .strat import StrategyBuilder
from .util import debug_train_strat, get_param_size
from dataclasses import dataclass
from torch import nn
import re


@dataclass
class PartFTBuilder(StrategyBuilder):
    """Partial Fine-Tuning adapter."""

    grad: bool

    def __call__(self, base: nn.Module) -> "PartFT":
        if isinstance(base, PartFT):
            return base
        return PartFT(base=base, config=self)

    @property
    def name(self) -> str:
        return "part_tune" if self.grad else "part_lock"


class PartFT(nn.Module):
    """Partial Fine-Tuning adapter."""

    def __init__(self, base: nn.Module, config: PartFTBuilder) -> None:
        super().__init__()

        self.base = base
        self.config = config

        target = [re.compile(t) for t in config.target]
        for path, param in base.named_parameters():
            if any(t.fullmatch(path) for t in target):
                param.requires_grad_(config.grad)
                debug_train_strat(config.name, get_param_size(param, config.grad), path)

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)
