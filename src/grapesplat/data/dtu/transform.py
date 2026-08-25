"""Transform modules for DTU pipeline data."""

from .. import transform as trf
from ..transform import TransformBuilder
from ..typing import DTUSequenceAnno, PipelineData, PipelineFrame
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor
import numpy
import torch


@dataclass
class DTUTransformBuilder(TransformBuilder):
    """Transform module for DTU sequence annotation."""

    def __call__(self, root_path: str) -> "DTUTransform":
        super().__call__(root_path)
        return DTUTransform(config=self)


class DTUTransform:
    """Transform module for DTU sequence annotation."""

    def __init__(self, config: DTUTransformBuilder):
        self.config = config

    def __call__(self, seq_name: str, seq_anno: DTUSequenceAnno) -> PipelineData:
        index = trf.sample_stratified_temporal_bin(
            bin_count=self.config.seq_size,
            frame_count=len(seq_anno),
            random=self.config.random,
        )
        seq_dir = self.config.root_path / seq_name

        frames: list[PipelineFrame] = []
        for i in index:
            frame_anno = seq_anno[i]
            file_id = frame_anno["file_id"]
            depth_path = seq_dir / "depths" / f"depth_map_{file_id:04d}.pfm"
            image_path = seq_dir / "images" / f"rect_{file_id + 1:03d}_3_r5000.png"

            depth = load_depth(depth_path)
            mask_depth = None
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
            frame = trf.resize_frame_on_shorter(frame, self.config.frame_resol)
            frame = trf.crop_frame(frame, self.config.frame_resol)
            frame = trf.cast_frame_dtype(frame)
            frames.append(frame)

        data = trf.collate_frames(frames, f"dtu/{seq_name}")
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
    Load camera extrinsic in OpenCV W2C.

    Shape:
        (16,) -> (3, 4)
    """

    # (16,) -> (4, 4)
    matr = torch.tensor(extr, dtype=torch.float32).reshape(4, 4)
    # (4, 4) -> (3, 3 + 1)
    rotation, translation = matr[0:3].split([3, 1], dim=-1)
    # NOTE: Rescale translation from millimeters to meters per codebase convention.
    return torch.cat([rotation, translation.div(1000.0)], dim=-1)


def load_cam_intr(intr: list[float]) -> Tensor:
    """
    Load camera intrinsic in OpenCV PC.

    Shape:
        (9,) -> (3, 3)
    """

    return torch.tensor(intr, dtype=torch.float32).reshape(3, 3)


def load_depth(path: Path) -> Tensor:
    """
    Load depth from PFM in millimeters.

    Shape:
        (1, H, W) in meters
    """

    with open(path, "rb") as f:
        f.readline()
        w, h = [int(v) for v in f.readline().split()]
        scale = float(f.readline())
        endian = "<" if scale < 0 else ">"
        data = numpy.frombuffer(f.read(), dtype=numpy.dtype(endian + "f4")).reshape(h, w)
    return torch.from_numpy(data.copy()).flip(0).unsqueeze(0).div(1000.0)
