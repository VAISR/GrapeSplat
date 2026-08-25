"""Dataset modules for TartanAirV2 pipeline data."""

from ..dataset import DatasetBuilder, TransformBuilder
from ..typing import PipelineData
from .filter import TartanAirV2FilterBuilder
from .transform import TartanAirV2TransformBuilder
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class TartanAirV2DatasetBuilder(DatasetBuilder):
    """Dataset module for TartanAirV2/TartanGround."""

    filter: TartanAirV2FilterBuilder

    trans: TartanAirV2TransformBuilder = None

    def __call__(self, stage: str, trans: TransformBuilder) -> "TartanAirV2Dataset":
        super().__call__(stage, trans)

        self.trans = TartanAirV2TransformBuilder(**vars(trans))
        return TartanAirV2Dataset(config=self)


class TartanAirV2Dataset(Dataset):
    """Dataset module for TartanAirV2/TartanGround."""

    def __init__(self, config: TartanAirV2DatasetBuilder):
        self.config = config
        self.data = config.filter(config.root_path)()[config.stage]
        self.trans = config.trans(config.root_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PipelineData:
        seq_name, seq_anno = self.data[index]
        return self.trans(seq_name, seq_anno)
