"""Data layer typing definitions for GrapeSplat."""

from torch import Tensor
from typing import NamedTuple


ARKitScenesDatasetAnno = list[tuple[str, "ARKitScenesSequenceAnno"]]

ARKitScenesFrameAnno = dict[str, list[float] | str]

ARKitScenesSequenceAnno = dict[str, str | list[ARKitScenesFrameAnno]]

CO3DDatasetAnno = list[tuple[str, "CO3DSequenceAnno"]]

CO3DFrameAnno = dict[str, dict[str]]

CO3DSequenceAnno = list[CO3DFrameAnno]

DL3DVDatasetAnno = list[tuple[str, "DL3DVSequenceAnno"]]

DL3DVFrameAnno = dict[str, list[float] | str]

DL3DVSequenceAnno = dict[str, list[float] | list[DL3DVFrameAnno]]

DTUDatasetAnno = list[tuple[str, "DTUSequenceAnno"]]

DTUFrameAnno = dict[str, int | list[float]]

DTUSequenceAnno = list[DTUFrameAnno]

ETH3DDatasetAnno = list[tuple[str, "ETH3DSequenceAnno"]]

ETH3DFrameAnno = dict[str, list[float] | str]

ETH3DSequenceAnno = list[ETH3DFrameAnno]

HypersimDatasetAnno = list[tuple[str, "HypersimSequenceAnno"]]

HypersimFrameAnno = dict[str, list[float] | str]

HypersimSequenceAnno = dict[str, float | list[float] | list[HypersimFrameAnno]]

NRGBDDatasetAnno = list[tuple[str, "NRGBDSequenceAnno"]]

NRGBDFrameAnno = dict[str, list[float] | str]

NRGBDSequenceAnno = dict[str, float | list[NRGBDFrameAnno]]

ScanNetppDatasetAnno = list[tuple[str, "ScanNetppSequenceAnno"]]

ScanNetppFrameAnno = dict[str, list[list[float]] | str]

ScanNetppSequenceAnno = dict[str, float | int | list[ScanNetppFrameAnno]]

SevenScenesDatasetAnno = list[tuple[str, "SevenScenesSequenceAnno"]]

SevenScenesFrameAnno = dict[str, list[float] | str]

SevenScenesSequenceAnno = list[SevenScenesFrameAnno]

TartanAirV2DatasetAnno = list[tuple[str, "TartanAirV2SequenceAnno"]]

TartanAirV2FrameAnno = dict[str, list[float] | str]

TartanAirV2SequenceAnno = dict[str, dict[str, int] | list[TartanAirV2FrameAnno]]

WildRGBDDatasetAnno = list[tuple[str, "WildRGBDSequenceAnno"]]

WildRGBDFrameAnno = dict[str, list[float] | str]

WildRGBDSequenceAnno = dict[str, list[float] | list[WildRGBDFrameAnno]]


class PipelineData(NamedTuple):
    """Batched sequence data for pipeline supervision."""

    cam_extr: Tensor
    """(B, S, 3, 4) in W2C"""
    cam_intr: Tensor
    """(B, S, 3, 3) in PC"""
    depth: Tensor
    """(B, S, 1, H, W) in [0, inf)"""
    image: Tensor
    """(B, S, 3, H, W) in [0, 1]"""
    mask_depth: Tensor
    """(B, S, 1, H, W)"""
    mask_image: Tensor
    """(B, S, 3, H, W)"""
    mask_rec: Tensor
    """(S,) where sum = S_rec"""
    mask_ren: Tensor
    """(S,) where sum = S_ren"""
    name: tuple[str, ...]
    """(B,)"""
    point: Tensor
    """(B, S, 3, H, W) in world"""

    def get_batch_size(self) -> int:
        return self.image.shape[-5]

    def get_frame_count(self) -> int:
        return self.get_batch_size() * self.get_seq_size()

    def get_seq_size(self) -> int:
        return self.mask_rec.numel()


class PipelineFrame(NamedTuple):
    """Frame data for pipeline supervision."""

    cam_extr: Tensor
    """(3, 4) in W2C"""
    cam_intr: Tensor
    """(3, 3) in PC"""
    depth: Tensor
    """(1, H, W) in meters"""
    image: Tensor
    """(3, H, W) in [0, 1]"""
    mask_depth: Tensor
    """(1, H, W)"""
    mask_image: Tensor
    """(3, H, W)"""
