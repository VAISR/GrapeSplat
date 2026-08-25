"""Transform modules for TartanAirV2 pipeline data."""

from .. import transform as trf
from ..transform import TransformBuilder
from ..typing import PipelineData, PipelineFrame, TartanAirV2SequenceAnno
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor
from torchvision.transforms.v2 import functional as tvf
import torch


@dataclass
class TartanAirV2TransformBuilder(TransformBuilder):
    """Transform module for TartanAirV2/TartanGround sequence annotation."""

    def __call__(self, root_path: str) -> "TartanAirV2Transform":
        super().__call__(root_path)
        return TartanAirV2Transform(config=self)


class TartanAirV2Transform:
    """Transform module for TartanAirV2/TartanGround sequence annotation."""

    def __init__(self, config: TartanAirV2TransformBuilder):
        self.config = config

    def __call__(self, seq_name: str, seq_anno: TartanAirV2SequenceAnno) -> PipelineData:
        index = trf.sample_stratified_temporal_bin(
            bin_count=self.config.seq_size,
            frame_count=len(seq_anno["frame"]),
            random=self.config.random,
        )
        label_sky = seq_anno["seg_label_map"].get("sky")
        seq_dir = self.config.root_path / seq_name

        frames: list[PipelineFrame] = []
        for i in index:
            frame_anno = seq_anno["frame"][i]
            file_id = frame_anno["file_id"]
            depth_path = seq_dir / "depth_lcam_front" / f"{file_id}_lcam_front_depth.png"
            image_path = seq_dir / "image_lcam_front" / f"{file_id}_lcam_front.png"
            seg_path = seq_dir / "seg_lcam_front" / f"{file_id}_lcam_front_seg.png"

            depth = load_depth(depth_path)
            mask_depth = None if label_sky is None else ~load_mask_sky(seg_path, label_sky)
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

        data = trf.collate_frames(frames, f"tartanairv2/{seq_name}")
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
    Load camera extrinsic from NED C2W to OpenCV W2C.

    Shape:
        (7,) in [tx, ty, tz, qx, qy, qz, qw] -> (3, 4)
    """

    translation = torch.tensor(extr[0:3], dtype=torch.float32)
    quaternion = torch.tensor(extr[6:7] + extr[3:6], dtype=torch.float32)

    # NED to OpenCV camera axis convention
    # (4,) in wxyz -> (3, 3) @ (3, 3) -> (3, 3)
    ned_r_cam = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotation = trf.get_matr_from_quat(quaternion) @ ned_r_cam

    # C2W to W2C transformation
    # [(3, 3), (3, 1)] -> (3, 4) -> [(3, 3), (3, 1)] -> (3, 4)
    extr_c2w = torch.cat([rotation, translation[..., None]], dim=-1)
    extr_w2c = torch.cat(trf.get_cam_extr_inv(extr_c2w), dim=-1)
    return extr_w2c


def load_cam_intr() -> Tensor:
    """
    Load camera intrinsic in OpenCV PC.

    Shape:
        (3, 3)
    """

    fx, fy, cx, cy = (320.0,) * 4
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def load_depth(path: Path) -> Tensor:
    """
    Load depth from F32 PNG in meters.

    Shape:
        (1, H, W) in meters
    """

    depth = Image.open(path)
    pack = depth.split()
    buffer = bytearray(Image.merge("RGBA", pack[-2::-1] + pack[-1:]).tobytes())
    depth = torch.frombuffer(buffer, dtype=torch.float32).reshape(1, depth.height, depth.width)
    return depth


def load_mask_sky(seg_path: Path, label_sky: int) -> Tensor:
    """
    Load sky mask from U8 PNG.

    Shape:
        (1, H, W) in bool
    """

    return tvf.to_dtype(tvf.to_image(Image.open(seg_path)), torch.uint8).eq(label_sky)
