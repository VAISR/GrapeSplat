"""Transform modules for ARKitScenes pipeline data."""

from .. import transform as trf
from ..transform import TransformBuilder
from ..typing import ARKitScenesSequenceAnno, PipelineData, PipelineFrame
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor
from torchvision.transforms.v2 import functional as tvf
import torch


@dataclass
class ARKitScenesTransformBuilder(TransformBuilder):
    """Transform module for ARKitScenes sequence annotation."""

    def __call__(self, root_path: str) -> "ARKitScenesTransform":
        super().__call__(root_path)
        return ARKitScenesTransform(config=self)


class ARKitScenesTransform:
    """Transform module for ARKitScenes sequence annotation."""

    def __init__(self, config: ARKitScenesTransformBuilder):
        self.config = config

    def __call__(self, seq_name: str, seq_anno: ARKitScenesSequenceAnno) -> PipelineData:
        # NOTE: Skip calibration frames in the first bin
        index = trf.sample_stratified_temporal_bin(
            bin_count=self.config.seq_size + 1,
            frame_count=len(seq_anno["frame"]),
            random=self.config.random,
        )[1:]
        seq_dir = Path(self.config.root_path / "raw" / seq_anno["parent_name"] / seq_name)

        frames: list[PipelineFrame] = []
        for i in index:
            frame_anno = seq_anno["frame"][i]
            file_stem = frame_anno["file_stem"]
            depth_path = seq_dir / "lowres_depth" / f"{file_stem}.png"
            image_path = seq_dir / "vga_wide" / f"{file_stem}.png"
            mask_depth_path = seq_dir / "confidence" / f"{file_stem}.png"

            depth = load_depth(depth_path)
            mask_depth = trf.load_mask(mask_depth_path)
            depth = trf.clamp_depth(depth, self.config.depth_bound_pr, mask_depth)
            extr = load_cam_extr(frame_anno["extr"])
            image = trf.load_image(image_path)
            intr = load_cam_intr(frame_anno["intr"])
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
            frame = orient_frame(frame, seq_anno["sky_direction"])
            frame = trf.resize_frame_on_shorter(frame, self.config.frame_resol)
            frame = trf.crop_frame(frame, self.config.frame_resol)
            frame = trf.cast_frame_dtype(frame)
            frames.append(frame)

        data = trf.collate_frames(frames, f"arkitscenes/{seq_name}")
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
    Load camera extrinsic from ARKit W2C to OpenCV W2C.

    Shape:
        (6,) -> (3, 4)
    """

    dtype = torch.float32
    eps = torch.finfo(dtype).eps
    axisangle, translation = torch.tensor(extr, dtype=dtype).split(3, dim=-1)

    # Rodrigues' rotation formula
    # (3,) -> (3, 3)
    angle = axisangle.norm()
    ax, ay, az = axisangle.div(angle.clamp(min=eps)).unbind(dim=-1)
    cross = torch.tensor(
        [
            [0.0, -az, ay],
            [az, 0.0, -ax],
            [-ay, ax, 0.0],
        ]
    )
    rotation = torch.eye(3) + angle.sin() * cross + (1.0 - angle.cos()) * cross @ cross

    # [(3, 3), (3, 1)] -> (3, 4)
    return torch.cat([rotation, translation[..., None]], dim=-1)


def load_cam_intr(intr: list[float]) -> Tensor:
    """
    Load camera intrinsic in OpenCV PC.

    Shape:
        (6,) -> (3, 3)
    """

    fx, fy, cx, cy = intr[2:6]
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def load_depth(path: Path) -> Tensor:
    """
    Load depth from U16 PNG in millimeters.

    Shape:
        (1, H, W) in meters
    """

    return tvf.to_dtype(tvf.to_image(Image.open(path)), torch.float32).div(1000.0)


def orient_frame(frame: PipelineFrame, skydir: str) -> PipelineFrame:
    """Orient frame to upright based on sky direction."""

    if skydir == "Up":
        return frame
    if skydir == "Down":
        return trf.flip_frame_horizontally(trf.flip_frame_vertically(frame))
    if skydir == "Left":
        return trf.flip_frame_horizontally(trf.transpose_frame(frame))
    if skydir == "Right":
        return trf.flip_frame_vertically(trf.transpose_frame(frame))
    raise ValueError(f"Unknown sky direction: {skydir}")
