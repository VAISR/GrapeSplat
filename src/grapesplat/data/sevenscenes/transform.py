"""Transform modules for 7Scenes pipeline data."""

from .. import transform as trf
from ..transform import TransformBuilder
from ..typing import PipelineData, PipelineFrame, SevenScenesSequenceAnno
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor
from torchvision.transforms.v2 import functional as tvf
import torch


@dataclass
class SevenScenesTransformBuilder(TransformBuilder):
    """Transform module for SevenScenes sequence annotation."""

    def __call__(self, root_path: str) -> "SevenScenesTransform":
        super().__call__(root_path)
        return SevenScenesTransform(config=self)


class SevenScenesTransform:
    """Transform module for SevenScenes sequence annotation."""

    def __init__(self, config: SevenScenesTransformBuilder):
        self.config = config

    def __call__(self, seq_name: str, seq_anno: SevenScenesSequenceAnno) -> PipelineData:
        index = trf.sample_stratified_temporal_bin(
            bin_count=self.config.seq_size,
            frame_count=len(seq_anno),
            random=self.config.random,
        )
        seq_dir = self.config.root_path / seq_name

        frames: list[PipelineFrame] = []
        for i in index:
            frame_anno = seq_anno[i]
            file_stem = frame_anno["file_stem"]
            depth_path = seq_dir / f"{file_stem}.depth.png"
            image_path = seq_dir / f"{file_stem}.color.png"

            depth = load_depth(depth_path)
            mask_depth = None
            depth = trf.clamp_depth(depth, self.config.depth_bound_pr, mask_depth)
            extr = load_cam_extr(frame_anno["extr"])
            image = trf.load_image(image_path)
            intr = load_cam_intr()
            mask_depth = trf.get_mask_for_depth(depth)
            mask_image = image.isfinite()
            frame = PipelineFrame(
                cam_extr=extr,
                cam_intr=intr,
                depth=depth,
                image=image,
                mask_depth=mask_depth,
                mask_image=mask_image,
            )
            frame = trf.resize_frame_on_shorter(frame, self.config.frame_resol)
            frame = trf.crop_frame(frame, self.config.frame_resol)
            frame = trf.cast_frame_dtype(frame)
            frames.append(frame)

        data = trf.collate_frames(frames, f"sevenscenes/{seq_name}")
        data = data._replace(
            mask_rec=trf.get_mask_for_first_elem_and_subset(
                set_size=self.config.seq_size,
                subset_size=self.config.seq_size_rec,
            ),
            point=trf.unproject(
                depth=data.depth,
                extr=data.cam_extr,
                intr=data.cam_intr,
            ),
        )
        data = trf.normalize_depth_extr_point(data)
        return data


def load_cam_extr(extr: list[float]) -> Tensor:
    """
    Load camera extrinsic from OpenCV C2W to OpenCV W2C.

    Shape:
        (16,) -> (3, 4)
    """

    # (16,) -> (4, 4) -> (3, 4)
    extr_c2w = torch.tensor(extr, dtype=torch.float32).reshape(4, 4)[0:3]
    # C2W to W2C transformation
    # (3, 4) -> [(3, 3), (3, 1)] -> (3, 4)
    extr_w2c = torch.cat(trf.get_cam_extr_inv(extr_c2w), dim=-1)
    return extr_w2c


def load_cam_intr() -> Tensor:
    """
    Load camera intrinsic in OpenCV PC.

    Shape:
        (3, 3)
    """

    fx, fy, cx, cy = 585.0, 585.0, 320.0, 240.0
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def load_depth(path: Path) -> Tensor:
    """
    Load depth from U16 PNG in millimeters.

    Shape:
        (1, H, W) in meters
    """

    depth = tvf.to_dtype(tvf.to_image(Image.open(path)), torch.float32)
    return depth.where(depth.lt(torch.iinfo(torch.uint16).max), 0.0).div(1000.0)
