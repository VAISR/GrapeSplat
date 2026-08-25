"""Dataset modules for DTU pipeline data."""

from ..dataset import DatasetBuilder, TransformBuilder
from ..typing import PipelineData
from .filter import DTUFilterBuilder
from .transform import DTUTransformBuilder
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class DTUDatasetBuilder(DatasetBuilder):
    """Dataset module for DTU."""

    filter: DTUFilterBuilder

    trans: DTUTransformBuilder = None

    def __call__(self, stage: str, trans: TransformBuilder) -> "DTUDataset":
        super().__call__(stage, trans)

        self.trans = DTUTransformBuilder(**vars(trans))
        return DTUDataset(config=self)


class DTUDataset(Dataset):
    """Dataset module for DTU."""

    def __init__(self, config: DTUDatasetBuilder):
        self.config = config
        self.data = config.filter(config.root_path)()[config.stage]
        self.trans = config.trans(config.root_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PipelineData:
        seq_name, seq_anno = self.data[index]
        return self.trans(seq_name, seq_anno)
