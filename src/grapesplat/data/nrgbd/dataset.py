"""Dataset modules for NRGBD pipeline data."""

from ..dataset import DatasetBuilder, TransformBuilder
from ..typing import PipelineData
from .filter import NRGBDFilterBuilder
from .transform import NRGBDTransformBuilder
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class NRGBDDatasetBuilder(DatasetBuilder):
    """Dataset module for NRGBD."""

    filter: NRGBDFilterBuilder

    trans: NRGBDTransformBuilder = None

    def __call__(self, stage: str, trans: TransformBuilder) -> "NRGBDDataset":
        super().__call__(stage, trans)

        self.trans = NRGBDTransformBuilder(**vars(trans))
        return NRGBDDataset(config=self)


class NRGBDDataset(Dataset):
    """Dataset module for NRGBD."""

    def __init__(self, config: NRGBDDatasetBuilder):
        self.config = config
        self.data = config.filter(config.root_path)()[config.stage]
        self.trans = config.trans(config.root_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PipelineData:
        seq_name, seq_anno = self.data[index]
        return self.trans(seq_name, seq_anno)
