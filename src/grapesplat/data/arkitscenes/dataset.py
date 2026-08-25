"""Dataset modules for ARKitScenes pipeline data."""

from ..dataset import DatasetBuilder, TransformBuilder
from ..typing import PipelineData
from .filter import ARKitScenesFilterBuilder
from .transform import ARKitScenesTransformBuilder
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class ARKitScenesDatasetBuilder(DatasetBuilder):
    """Dataset module for ARKitScenes."""

    filter: ARKitScenesFilterBuilder

    trans: ARKitScenesTransformBuilder = None

    def __call__(self, stage: str, trans: TransformBuilder) -> "ARKitScenesDataset":
        super().__call__(stage, trans)

        self.trans = ARKitScenesTransformBuilder(**vars(trans))
        return ARKitScenesDataset(config=self)


class ARKitScenesDataset(Dataset):
    """Dataset module for ARKitScenes."""

    def __init__(self, config: ARKitScenesDatasetBuilder):
        self.config = config
        self.data = config.filter(config.root_path)()[config.stage]
        self.trans = config.trans(config.root_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PipelineData:
        seq_name, seq_anno = self.data[index]
        return self.trans(seq_name, seq_anno)
