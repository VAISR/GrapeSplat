"""Transform modules for ETH3D pipeline data."""

from .. import transform as trf
from ..transform import TransformBuilder
from ..typing import ETH3DSequenceAnno, PipelineData, PipelineFrame
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor
from torch.nn import functional as nnf
from torchvision.transforms.v2 import functional as tvf
import numpy
import pycolmap
import torch


@dataclass
class ETH3DTransformBuilder(TransformBuilder):
    """Transform module for ETH3D sequence annotation."""

    def __call__(self, root_path: str) -> "ETH3DTransform":
        super().__call__(root_path)
        return ETH3DTransform(config=self)


class ETH3DTransform:
    """Transform module for ETH3D sequence annotation."""

    def __init__(self, config: ETH3DTransformBuilder):
        self.config = config

    def __call__(self, seq_name: str, seq_anno: ETH3DSequenceAnno) -> PipelineData:
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
            depth_path = seq_dir / "ground_truth_depth" / "dslr_images" / f"{file_stem}.JPG"
            image_path = seq_dir / "images" / "dslr_images" / f"{file_stem}.JPG"
            mask_path = seq_dir / "masks_for_images" / "dslr_images" / f"{file_stem}.png"

            image = trf.load_image(image_path)
            depth = load_depth(depth_path, image.shape[-2:])
            mask_depth = load_mask(mask_path)
            depth = trf.clamp_depth(depth, self.config.depth_bound_pr, mask_depth)
            extr = load_cam_extr(frame_anno["extr"])
            intr, grid = load_cam_intr_and_undist_grid(frame_anno["params"], image.shape[-2:])
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
            frame = undist_frame(frame, grid)
            frame = trf.resize_frame_on_shorter(frame, self.config.frame_resol)
            frame = trf.crop_frame(frame, self.config.frame_resol)
            frame = trf.cast_frame_dtype(frame)
            frames.append(frame)

        data = trf.collate_frames(frames, f"eth3d/{seq_name}")
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
    Load camera extrinsic from COLMAP W2C to OpenCV W2C.

    Shape:
        (7,) -> (3, 4)
    """

    rotation = trf.get_matr_from_quat(torch.tensor(extr[0:4], dtype=torch.float32))
    translation = torch.tensor(extr[4:7], dtype=torch.float32)
    # [(3, 3), (3, 1)] -> (3, 4)
    return torch.cat([rotation, translation[..., None]], dim=-1)


def load_cam_intr_and_undist_grid(
    params: list[float],
    frame_resol_src: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load camera intrinsic and undistortion grid from COLMAP THIN_PRISM_FISHEYE to PC.

    Shape:
        [(12,), [H_i, W_i]] -> [(3, 3), (H_o, W_o, 2)]
    """

    cam_src = pycolmap.Camera(
        model="THIN_PRISM_FISHEYE",
        height=frame_resol_src[0],
        width=frame_resol_src[1],
        params=list(params),
    )
    opt = pycolmap.UndistortCameraOptions()
    opt.blank_pixels = 0.0
    cam_tar = pycolmap.undistort_camera(opt, cam_src)

    # (3, 3)
    cam_intr = torch.tensor(
        [
            [cam_tar.focal_length_x, 0.0, cam_tar.principal_point_x],
            [0.0, cam_tar.focal_length_y, cam_tar.principal_point_y],
            [0.0, 0.0, 1.0],
        ]
    )
    # [H, W]
    frame_resol_tar = (cam_tar.height, cam_tar.width)
    # (H, W, 2)
    pix_tar = torch.stack(
        torch.meshgrid(
            torch.tensor(range(frame_resol_tar[1])),
            torch.tensor(range(frame_resol_tar[0])),
            indexing="xy",
        ),
        dim=-1,
    ).add(0.5)
    # (H * W, 2) -> (H * W, 3)
    pix_unp = torch.from_numpy(
        numpy.asarray(
            cam_tar.cam_from_img(
                pix_tar.flatten(0, 1).double().numpy(),
            )
        )
    )
    # (H * W, 3) -> (H * W, 2) -> (H, W, 2)
    pix_src = (
        torch.from_numpy(
            numpy.asarray(cam_src.img_from_cam(nnf.pad(pix_unp, (0, 1), "constant", 1.0).numpy()))
        )
        .unflatten(0, frame_resol_tar)
        .float()
        .sub(0.5)
    )
    # (H, W, 2) / (2,) -> (H, W, 2)
    undist_grid = pix_src.div(pix_src.new_tensor(frame_resol_src[::-1]).mul(0.5).sub(0.5)).sub(1.0)
    return cam_intr, undist_grid


def load_depth(path: Path, depth_size: tuple[int, int]) -> Tensor:
    """
    Load depth from FP32 buffer in meters.

    Shape:
        (1, H, W) in meters
    """

    depth = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float32)
    depth = depth.reshape(1, *depth_size)
    return depth.nan_to_num(0.0, 0.0, 0.0)


def load_mask(path: Path) -> Tensor | None:
    """
    Load mask from U8 PNG.

    Shape:
        (1, H, W)? in bool
    """

    try:
        return tvf.to_image(Image.open(path)).eq(0)
    except FileNotFoundError:
        return


def undist_frame(frame: PipelineFrame, undist_grid: Tensor) -> PipelineFrame:
    """
    Undistort frame.

    Shape:
        [[(C, H_i, W_i); n], (H_o, W_o, 2)] -> [(C, H_o, W_o); n]
    """

    arg = dict(
        align_corners=True,
        grid=undist_grid[None],
    )
    frame = frame._replace(
        depth=nnf.grid_sample(frame.depth.float()[None], mode="nearest", **arg).float()[0],
        image=nnf.grid_sample(frame.image.float()[None], mode="bilinear", **arg).byte()[0],
        mask_depth=nnf.grid_sample(frame.mask_depth.float()[None], mode="nearest", **arg).bool()[0],
        mask_image=nnf.grid_sample(frame.mask_image.float()[None], mode="nearest", **arg).bool()[0],
    )
    return frame
