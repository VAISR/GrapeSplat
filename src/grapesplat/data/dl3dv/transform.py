"""Transform modules for DL3DV pipeline data."""

from .. import transform as trf
from ..transform import TransformBuilder
from ..typing import DL3DVSequenceAnno, PipelineData, PipelineFrame
from dataclasses import dataclass
from torch import Tensor
import torch


@dataclass
class DL3DVTransformBuilder(TransformBuilder):
    """Transform module for DL3DV sequence annotation."""

    def __call__(self, root_path: str) -> "DL3DVTransform":
        super().__call__(root_path)
        return DL3DVTransform(config=self)


class DL3DVTransform:
    """Transform module for DL3DV sequence annotation."""

    def __init__(self, config: DL3DVTransformBuilder):
        self.config = config

    def __call__(self, seq_name: str, seq_anno: DL3DVSequenceAnno) -> PipelineData:
        index = trf.sample_stratified_temporal_bin(
            bin_count=self.config.seq_size,
            frame_count=len(seq_anno["frame"]),
            random=self.config.random,
        )
        seq_dir = self.config.root_path / seq_name / "gaussian_splat"
        intr = load_cam_intr(seq_anno["intr"])

        frames: list[PipelineFrame] = []
        for i in index:
            frame_anno = seq_anno["frame"][i]
            image_path = seq_dir / "images" / frame_anno["file_name"]

            image = trf.load_image(image_path)
            depth = torch.ones((1,) + image.shape[-2:])
            mask_depth = trf.get_mask_for_depth(depth)
            extr = load_cam_extr(frame_anno["extr"])
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

        data = trf.collate_frames(frames, f"dl3dv/{seq_name}")
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
        return data


def load_cam_extr(extr: list[float]) -> Tensor:
    """
    Load camera extrinsic from COLMAP W2C to OpenCV W2C.

    Shape:
        (7,) in [qx, qy, qz, qw, tx, ty, tz] -> (3, 4)
    """

    rotation = trf.get_matr_from_quat(torch.tensor(extr[3:4] + extr[0:3], dtype=torch.float32))
    translation = torch.tensor(extr[4:7], dtype=torch.float32)
    # [(3, 3), (3, 1)] -> (3, 4)
    return torch.cat([rotation, translation[..., None]], dim=-1)


def load_cam_intr(intr: list[float]) -> Tensor:
    """
    Load camera intrinsic from COLMAP PINHOLE to PC.

    Shape:
        (4,) in [fx, fy, cx, cy] -> (3, 3)
    """

    fx, fy, cx, cy = intr
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
