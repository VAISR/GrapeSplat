"""Transform modules for CO3D pipeline data."""

from .. import transform as trf
from ..transform import TransformBuilder
from ..typing import CO3DSequenceAnno, PipelineData, PipelineFrame
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor
import torch


@dataclass
class CO3DTransformBuilder(TransformBuilder):
    """Transform module for CO3D sequence annotation."""

    def __call__(self, root_path: str) -> "CO3DTransform":
        super().__call__(root_path)
        return CO3DTransform(config=self)


class CO3DTransform:
    """Transform module for CO3D sequence annotation."""

    def __init__(self, config: CO3DTransformBuilder):
        self.config = config

    def __call__(self, seq_name: str, seq_anno: CO3DSequenceAnno) -> PipelineData:
        index = trf.sample_stratified_temporal_bin(
            bin_count=self.config.seq_size,
            frame_count=len(seq_anno),
            random=self.config.random,
        )
        root = self.config.root_path

        frames: list[PipelineFrame] = []
        for i in index:
            frame_anno = seq_anno[i]
            mask_depth_path = root / frame_anno["depth"]["mask_path"]
            depth_path = root / frame_anno["depth"]["path"]
            image_path = root / frame_anno["image"]["path"]
            cam = frame_anno["viewpoint"]

            depth = load_depth(depth_path)
            mask_depth = trf.load_mask(mask_depth_path)
            depth = trf.clamp_depth(depth, self.config.depth_bound_pr, mask_depth)
            extr = load_cam_extr(cam)
            image = trf.load_image(image_path)
            intr = load_cam_intr(cam, image.shape[-2:])
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

        data = trf.collate_frames(frames, f"co3d/{seq_name}")
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


def load_cam_extr(cam: dict[str, list[float]]) -> Tensor:
    """
    Load camera extrinsic from PyTorch3D W2C to OpenCV W2C.

    Shape:
        [(3, 3), (3,)] -> (3, 4)
    """

    flip = torch.tensor([-1.0, -1.0, 1.0])[..., None]
    rotation = torch.tensor(cam["R"], dtype=torch.float32)
    translation = torch.tensor(cam["T"], dtype=torch.float32)
    # [(3, 3), (3, 1)] * (3, 1) -> (3, 4)
    return torch.cat([rotation.mT.contiguous(), translation[..., None]], dim=-1).mul(flip)


def load_cam_intr(cam: dict[str, float], frame_resol: tuple[int, int]) -> Tensor:
    """
    Load camera intrinsic from PyTorch3D Isotropic NDC to OpenCV PC.

    Shape:
        (3, 3)
    """

    iy, ix = frame_resol
    s = min(ix, iy)
    (fx, fy), (cx, cy) = cam["focal_length"], cam["principal_point"]
    fx, fy = fx * s / 2.0, fy * s / 2.0
    cx, cy = (1.0 - cx) * ix / 2.0, (1.0 - cy) * iy / 2.0
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def load_depth(path: Path) -> Tensor:
    """
    Load depth from FP16 PNG in meters.

    Shape:
        (1, H, W) in meters
    """

    depth = Image.open(path)
    return (
        torch.frombuffer(bytearray(depth.tobytes()), dtype=torch.float16)
        .float()
        .reshape(1, depth.height, depth.width)
    )
