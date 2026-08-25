"""Dataset modules for GrapeSplat pipeline data."""

from .filter import FilterBuilder
from .transform import TransformBuilder
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class DatasetBuilder(ABC):
    """Dataset module for GrapeSplat."""

    filter: FilterBuilder
    root_path: str
    weight_test: int
    weight_train: int

    stage: str = None
    trans: TransformBuilder = None

    @abstractmethod
    def __call__(self, stage: str, trans: TransformBuilder) -> Dataset:
        """Build the module."""

        self.stage = stage
        self.trans = trans
        return Dataset()
