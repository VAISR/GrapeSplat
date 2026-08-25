"""Transform modules for Hypersim pipeline data."""

from .. import transform as trf
from ..transform import TransformBuilder
from ..typing import PipelineData, PipelineFrame, HypersimSequenceAnno
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor
import h5py
import torch


@dataclass
class HypersimTransformBuilder(TransformBuilder):
    """Transform module for Hypersim sequence annotation."""

    def __call__(self, root_path: str) -> "HypersimTransform":
        super().__call__(root_path)
        return HypersimTransform(config=self)


class HypersimTransform:
    """Transform module for Hypersim sequence annotation."""

    def __init__(self, config: HypersimTransformBuilder):
        self.config = config

    def __call__(self, seq_name: str, seq_anno: HypersimSequenceAnno) -> PipelineData:
        index = trf.sample_stratified_temporal_bin(
            bin_count=self.config.seq_size,
            frame_count=len(seq_anno["frame"]),
            random=self.config.random,
        )
        scene_name, camera_name = seq_name.split("/")
        seq_dir = self.config.root_path / "evermotion_dataset" / "scenes" / scene_name

        frames: list[PipelineFrame] = []
        for i in index:
            frame_anno = seq_anno["frame"][i]
            frame_index = frame_anno["frame_index"]
            depth_path = (
                seq_dir
                / "images"
                / f"scene_{camera_name}_geometry_hdf5"
                / f"frame.{frame_index}.depth_meters.hdf5"
            )
            image_path = (
                seq_dir
                / "images"
                / f"scene_{camera_name}_final_hdf5"
                / f"frame.{frame_index}.color.hdf5"
            )

            image = load_image(image_path)
            intr = load_cam_intr(seq_anno, image.shape[-2:])
            depth = load_depth(depth_path, intr)
            mask_depth = depth.isfinite()
            depth = trf.clamp_depth(depth, self.config.depth_bound_pr, mask_depth)
            extr = load_cam_extr(frame_anno)
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

        data = trf.collate_frames(frames, f"hypersim/{seq_name}")
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


def load_cam_extr(frame_anno: dict[str, list[float]]) -> Tensor:
    """
    Load camera extrinsic from OpenGL C2W to OpenCV W2C.

    Shape:
        [(9,), (3,)] -> (3, 4)
    """

    flip = torch.tensor([[1.0, -1.0, -1.0, 1.0]])
    rotation = torch.tensor(frame_anno["orientation"], dtype=torch.float32).reshape(3, 3)
    translation = torch.tensor(frame_anno["position"], dtype=torch.float32)
    # OpenGL C2W -> OpenCV C2W
    # [(3, 3), (3, 1)] -> (3, 4) * (1, 4) -> (3, 4)
    extr_c2w = torch.cat([rotation, translation[..., None]], dim=-1) * flip
    # C2W to W2C transformation
    # (3, 4) -> [(3, 3), (3, 1)] -> (3, 4)
    extr_w2c = torch.cat(trf.get_cam_extr_inv(extr_c2w), dim=-1)
    return extr_w2c


def load_cam_intr(seq_anno: dict[str, list[float]], frame_resol: tuple[int, int]) -> Tensor:
    """
    Load camera intrinsic from Hypersim UV to OpenCV PC.

    Shape:
        (3, 3)
    """

    m00, m11, m02, m12 = seq_anno["M_cam_from_uv"]
    H, W = frame_resol
    fx, fy = W / (2 * m00), H / (2 * m11)
    cx, cy = (W - 1) / 2 - m02 * fx, (H - 1) / 2 + m12 * fy
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def load_depth(path: Path, intr: Tensor) -> Tensor:
    """
    Load depth from HDF5 Euclidean meters to planar z-depth.

    Shape:
        [(1, H, W), (3, 3)] -> (1, H, W) in meters
    """

    # (1, H, W)
    with h5py.File(path, "r") as f:
        depth = torch.from_numpy(f["dataset"][...][None].astype("float32"))

    # (3, 3) -> [(1, 1); 4]
    fx = intr[0:1, 0:1]
    fy = intr[1:2, 1:2]
    cx = intr[0:1, 2:3]
    cy = intr[1:2, 2:3]

    # [(H, W); 2]
    u, v = torch.meshgrid(
        depth.new_tensor(range(depth.shape[-1])),
        depth.new_tensor(range(depth.shape[-2])),
        indexing="xy",
    )

    # (H, W, 3)
    ray = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1)
    # (1, H, W) / (H, W) -> (1, H, W)
    return depth / ray.norm(dim=-1)


def load_image(path: Path) -> Tensor:
    """
    Load HDR image from HDF5 with CCIR601 tone mapping.

    Shape:
        (H, W, 3) -> (3, H, W) in uint8
    """

    # (H, W, 3)
    with h5py.File(path, "r") as f:
        color = torch.from_numpy(f["dataset"][...].astype("float32"))

    # CCIR601 YIQ brightness
    # (H, W, 3) @ (3,) -> (H, W)
    brightness = color @ torch.tensor([0.3, 0.59, 0.11])
    # (H, W) -> ()
    scale = brightness.quantile(0.9).clamp(min=1e-4).reciprocal().mul(0.8**2.2)
    # Tone mapping
    image = color.mul(scale).clamp(min=0.0).pow(1.0 / 2.2)
    # Rescale and convert to uint8
    image = image.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    # (H, W, 3) -> (3, H, W)
    return image.movedim(-1, -3).contiguous()
