"""Dataset modules for ScanNetpp pipeline data."""

from ..dataset import DatasetBuilder, TransformBuilder
from ..typing import PipelineData
from .filter import ScanNetppFilterBuilder
from .transform import ScanNetppTransformBuilder
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class ScanNetppDatasetBuilder(DatasetBuilder):
    """Dataset module for ScanNetpp."""

    filter: ScanNetppFilterBuilder

    trans: ScanNetppTransformBuilder = None

    def __call__(self, stage: str, trans: TransformBuilder) -> "ScanNetppDataset":
        super().__call__(stage, trans)

        self.trans = ScanNetppTransformBuilder(**vars(trans))
        return ScanNetppDataset(config=self)


class ScanNetppDataset(Dataset):
    """Dataset module for ScanNetpp."""

    def __init__(self, config: ScanNetppDatasetBuilder):
        self.config = config
        self.data = config.filter(config.root_path)()[config.stage]
        self.trans = config.trans(config.root_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PipelineData:
        seq_name, seq_anno = self.data[index]
        return self.trans(seq_name, seq_anno)
