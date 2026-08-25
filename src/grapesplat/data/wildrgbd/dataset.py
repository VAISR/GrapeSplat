"""Dataset modules for WildRGBD pipeline data."""

from ..dataset import DatasetBuilder, TransformBuilder
from ..typing import PipelineData
from .filter import WildRGBDFilterBuilder
from .transform import WildRGBDTransformBuilder
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class WildRGBDDatasetBuilder(DatasetBuilder):
    """Dataset module for WildRGBD."""

    filter: WildRGBDFilterBuilder

    trans: WildRGBDTransformBuilder = None

    def __call__(self, stage: str, trans: TransformBuilder) -> "WildRGBDDataset":
        super().__call__(stage, trans)

        self.trans = WildRGBDTransformBuilder(**vars(trans))
        return WildRGBDDataset(config=self)


class WildRGBDDataset(Dataset):
    """Dataset module for WildRGBD."""

    def __init__(self, config: WildRGBDDatasetBuilder):
        self.config = config
        self.data = config.filter(config.root_path)()[config.stage]
        self.trans = config.trans(config.root_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PipelineData:
        seq_name, seq_anno = self.data[index]
        return self.trans(seq_name, seq_anno)
