"""Dataset modules for Hypersim pipeline data."""

from ..dataset import DatasetBuilder, TransformBuilder
from ..typing import PipelineData
from .filter import HypersimFilterBuilder
from .transform import HypersimTransformBuilder
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class HypersimDatasetBuilder(DatasetBuilder):
    """Dataset module for Hypersim."""

    filter: HypersimFilterBuilder

    trans: HypersimTransformBuilder = None

    def __call__(self, stage: str, trans: TransformBuilder) -> "HypersimDataset":
        super().__call__(stage, trans)

        self.trans = HypersimTransformBuilder(**vars(trans))
        return HypersimDataset(config=self)


class HypersimDataset(Dataset):
    """Dataset module for Hypersim."""

    def __init__(self, config: HypersimDatasetBuilder):
        self.config = config
        self.data = config.filter(config.root_path)()[config.stage]
        self.trans = config.trans(config.root_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PipelineData:
        seq_name, seq_anno = self.data[index]
        return self.trans(seq_name, seq_anno)
