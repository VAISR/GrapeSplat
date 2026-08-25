"""Filter modules for GrapeSplat pipeline data."""

from .. import util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class FilterBuilder(ABC):
    """Filter module for GrapeSplat."""

    seq_len_min: int
    split_path: Path
    split_size_test: int
    split_size_train: int
    split_size_val: int

    root_path: Path = None

    @abstractmethod
    def __call__(self, root_path: str) -> Callable[[], Any]:
        """Build the module."""

        self.root_path = Path(root_path)
        self.split_path = Path(self.split_path)
        return lambda: None

    @property
    def meta(self) -> dict[str]:
        """Get filter metadata."""

        return dict(
            seed=util.get_seed(),
            seq_len_min=self.seq_len_min,
            split_size_test=self.split_size_test,
            split_size_train=self.split_size_train,
            split_size_val=self.split_size_val,
        )
