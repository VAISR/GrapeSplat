"""Dataset modules for 7Scenes pipeline data."""

from ..dataset import DatasetBuilder, TransformBuilder
from ..typing import PipelineData
from .filter import SevenScenesFilterBuilder
from .transform import SevenScenesTransformBuilder
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class SevenScenesDatasetBuilder(DatasetBuilder):
    """Dataset module for SevenScenes."""

    filter: SevenScenesFilterBuilder

    trans: SevenScenesTransformBuilder = None

    def __call__(self, stage: str, trans: TransformBuilder) -> "SevenScenesDataset":
        super().__call__(stage, trans)

        self.trans = SevenScenesTransformBuilder(**vars(trans))
        return SevenScenesDataset(config=self)


class SevenScenesDataset(Dataset):
    """Dataset module for SevenScenes."""

    def __init__(self, config: SevenScenesDatasetBuilder):
        self.config = config
        self.data = config.filter(config.root_path)()[config.stage]
        self.trans = config.trans(config.root_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PipelineData:
        seq_name, seq_anno = self.data[index]
        return self.trans(seq_name, seq_anno)
