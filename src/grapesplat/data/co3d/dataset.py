"""Dataset modules for CO3D pipeline data."""

from ..dataset import DatasetBuilder, TransformBuilder
from ..typing import PipelineData
from .filter import CO3DFilterBuilder
from .transform import CO3DTransformBuilder
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class CO3DDatasetBuilder(DatasetBuilder):
    """Dataset module for CO3D."""

    filter: CO3DFilterBuilder

    trans: CO3DTransformBuilder = None

    def __call__(self, stage: str, trans: TransformBuilder) -> "CO3DDataset":
        super().__call__(stage, trans)

        self.trans = CO3DTransformBuilder(**vars(trans))
        return CO3DDataset(config=self)


class CO3DDataset(Dataset):
    """Dataset module for CO3D."""

    def __init__(self, config: CO3DDatasetBuilder):
        self.config = config
        self.data = config.filter(config.root_path)()[config.stage]
        self.trans = config.trans(config.root_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PipelineData:
        seq_name, seq_anno = self.data[index]
        return self.trans(seq_name, seq_anno)
