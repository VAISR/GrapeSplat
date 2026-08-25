"""Transform modules for ScanNetpp pipeline data."""

from .. import transform as trf
from ..transform import TransformBuilder
from ..typing import PipelineData, PipelineFrame, ScanNetppSequenceAnno
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor
from torchvision.transforms.v2 import functional as tvf
import torch


@dataclass
class ScanNetppTransformBuilder(TransformBuilder):
    """Transform module for ScanNetpp sequence annotation."""

    def __call__(self, root_path: str) -> "ScanNetppTransform":
        super().__call__(root_path)
        return ScanNetppTransform(config=self)


class ScanNetppTransform:
    """Transform module for ScanNetpp sequence annotation."""

    def __init__(self, config: ScanNetppTransformBuilder):
        self.config = config

    def __call__(self, seq_name: str, seq_anno: ScanNetppSequenceAnno) -> PipelineData:
        index = trf.sample_stratified_temporal_bin(
            bin_count=self.config.seq_size,
            frame_count=len(seq_anno["frame"]),
            random=self.config.random,
        )
        seq_dir = self.config.root_path / "data" / seq_name / "dslr"

        frames: list[PipelineFrame] = []
        for i in index:
            frame_anno = seq_anno["frame"][i]
            depth_path = seq_dir / "render_undistorted_depth" / frame_anno["depth_path"]
            image_path = seq_dir / "resized_undistorted_images" / frame_anno["file_path"]
            mask_image_path = seq_dir / "resized_undistorted_masks" / frame_anno["mask_path"]

            depth = load_depth(depth_path)
            mask_depth = None
            depth = trf.clamp_depth(depth, self.config.depth_bound_pr, mask_depth)
            extr = load_cam_extr(frame_anno["transform_matrix"])
            image = trf.load_image(image_path)
            intr = load_cam_intr(seq_anno)
            mask_depth = trf.get_mask_for_depth(depth)
            mask_image = trf.load_mask(mask_image_path).expand(3, -1, -1)
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

        data = trf.collate_frames(frames, f"scannetpp/{seq_name}")
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


def load_cam_extr(transform_matrix: list[list[float]]) -> Tensor:
    """
    Load camera extrinsic from Nerfstudio OpenGL C2W to OpenCV W2C.

    Shape:
        (4, 4) -> (3, 4)
    """

    flip = torch.tensor([[1.0, -1.0, -1.0, 1.0]])
    # (4, 4) -> (3, 4) * (1, 4) -> (3, 4)
    extr_c2w = torch.tensor(transform_matrix, dtype=torch.float32)[0:3] * flip
    # C2W to W2C transformation
    # (3, 4) -> [(3, 3), (3, 1)] -> (3, 4)
    extr_w2c = torch.cat(trf.get_cam_extr_inv(extr_c2w), dim=-1)
    return extr_w2c


def load_cam_intr(seq_anno: dict[str, float]) -> Tensor:
    """
    Load camera intrinsic in Nerfstudio OpenCV PC.

    Shape:
        (3, 3)
    """

    s = seq_anno
    cx, cy, fx, fy = s["cx"], s["cy"], s["fl_x"], s["fl_y"]
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def load_depth(path: Path) -> Tensor:
    """
    Load depth from U16 PNG in millimeters.

    Shape:
        (1, H, W) in meters
    """

    return tvf.to_dtype(tvf.to_image(Image.open(path)), torch.float32).div(1000.0)
