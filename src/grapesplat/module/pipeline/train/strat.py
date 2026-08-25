from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import nn


@dataclass
class StrategyBuilder(ABC):
    target: list[str]

    @abstractmethod
    def __call__(self, base: nn.Module) -> nn.Module:
        """Build the adapted module."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name."""
