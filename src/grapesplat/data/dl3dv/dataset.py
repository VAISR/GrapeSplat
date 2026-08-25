"""Dataset modules for DL3DV pipeline data."""

from ..dataset import DatasetBuilder, TransformBuilder
from ..typing import PipelineData
from .filter import DL3DVFilterBuilder
from .transform import DL3DVTransformBuilder
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class DL3DVDatasetBuilder(DatasetBuilder):
    """Dataset module for DL3DV."""

    filter: DL3DVFilterBuilder

    trans: DL3DVTransformBuilder = None

    def __call__(self, stage: str, trans: TransformBuilder) -> "DL3DVDataset":
        super().__call__(stage, trans)

        self.trans = DL3DVTransformBuilder(**vars(trans))
        return DL3DVDataset(config=self)


class DL3DVDataset(Dataset):
    """Dataset module for DL3DV."""

    def __init__(self, config: DL3DVDatasetBuilder):
        self.config = config
        self.data = config.filter(config.root_path)()[config.stage]
        self.trans = config.trans(config.root_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PipelineData:
        seq_name, seq_anno = self.data[index]
        return self.trans(seq_name, seq_anno)
