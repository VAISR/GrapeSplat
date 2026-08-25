"""Transform modules for GrapeSplat pipeline data."""

from .typing import PipelineData, PipelineFrame
from PIL import Image
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from pytorch3d.ops import corresponding_points_alignment as umeyama
from torch import Tensor
from torch.nn import functional as nnf
from torch.utils.data import default_collate
from torchvision.transforms.v2 import functional as tvf, InterpolationMode
from typing import Callable
import torch


@dataclass
class TransformBuilder(ABC):
    """Transform module for GrapeSplat."""

    depth_bound_pr: tuple[float, float]
    frame_resol: int
    seq_size: int
    seq_size_rec: int

    random: bool = None
    root_path: Path = None

    def __call__(self, root_path: str) -> Callable[..., PipelineData]:
        """Build the module."""

        self.root_path = Path(root_path)
        return lambda *_: None


def align_cam_extr(
    extr_src: Tensor, extr_tar: Tensor, mask_src: Tensor, mask_tar: Tensor
) -> Tensor:
    """
    Align camera extrinsic from source to target coordinate.

    Shape:
        [(B, S, 3, 4) in W2C, (B, S_t, 3, 4) in W2C,
         (S,) where sum = S_s, (S,) where sum = S_t]
        -> (B, S_s, 3, 4) in W2C
    """

    with torch.autocast(extr_src.device.type, enabled=False):
        # [(B, S, 3, 4); 2] -> [(B, S_t, 3); 2] -> [(B, 3, 3), (B, 3), (B,)]
        p_src = get_cam_extr_inv(extr_src[:, mask_tar])[1].squeeze(-1)
        p_tar = get_cam_extr_inv(extr_tar)[1].squeeze(-1)
        r, t, s = umeyama(p_tar, p_src, estimate_scale=True)
        # [(B, S_s, 3, 3), (B, S_s, 3, 1)]
        r_src, t_src = extr_src[:, mask_src].split([3, 1], dim=-1)
        # (B, S_s, 3, 3) @ (B, 1, 3, 3) * (B, 1, 1, 1) -> (B, S_s, 3, 3)
        r_new = r_src.matmul(r.mT[:, None]).mul(s[:, None, None, None])
        # (B, S_s, 3, 3) @ (B, 1, 3, 1) + (B, S_s, 3, 1) -> (B, S_s, 3, 1)
        t_new = r_src.matmul(t[..., None, :, None]).add(t_src)
        # (B, S_s, 3, 3 + 1)
        return torch.cat([r_new, t_new], dim=-1)


def cast_frame_dtype(frame: PipelineFrame) -> PipelineFrame:
    """Cast frame data types to desired ones."""

    return frame._replace(image=cast_image_dtype(frame.image))


def cast_image_dtype(image: Tensor) -> Tensor:
    """Cast image data type to desired one."""

    return tvf.to_dtype(image, dtype=torch.float32, scale=True)


def clamp_depth(
    depth: Tensor,
    bound_pr: tuple[float, float],
    mask_depth: Tensor | None,
) -> Tensor:
    """Zero out depth artifacts by mask and outliers by percentile."""

    N, PR_MAX = depth.numel(), 2**24

    if mask_depth is not None:
        depth = depth.where(mask_depth.expand_as(depth), 0.0)

    # TODO(eval-only): depth_flat = depth[depth.ne(0.0)]
    depth_flat = depth.flatten()
    if N > PR_MAX:
        depth_flat = depth_flat[torch.randperm(N)[:PR_MAX]]

    bound = tuple(depth_flat.quantile(p) for p in bound_pr)
    depth = depth.where((bound[0] <= depth) & (depth <= bound[1]), 0.0)
    return depth


def collate_frames(frames: list[PipelineFrame], seq_name: str) -> PipelineData:
    """Collate list of frames into sequence data."""

    return PipelineData(
        cam_extr=torch.stack([f.cam_extr for f in frames], dim=0),
        cam_intr=torch.stack([f.cam_intr for f in frames], dim=0),
        depth=torch.stack([f.depth for f in frames], dim=0),
        image=torch.stack([f.image for f in frames], dim=0),
        mask_depth=torch.stack([f.mask_depth for f in frames], dim=0),
        mask_image=torch.stack([f.mask_image for f in frames], dim=0),
        mask_rec=torch.empty((0,)),
        mask_ren=torch.empty((0,)),
        name=seq_name,
        point=torch.empty((0,) * 5),
    )


def collate_seqs(seqs: list[PipelineData]) -> PipelineData:
    """Collate list of sequences into batch data."""

    data: PipelineData = default_collate(seqs)
    mask_rec = seqs[0].mask_rec
    return data._replace(mask_rec=mask_rec, mask_ren=~mask_rec)


def crop_frame(frame: PipelineFrame, frame_resol: int) -> PipelineFrame:
    """Crop frame at principal point."""

    frame_resol = int(frame_resol)
    frame_resol_src = torch.tensor(frame.image.shape[-2:])
    # [iy, ix]
    frame_resol_tar = torch.tensor([frame_resol, frame_resol])
    # [cy, cx] - [iy/2, ix/2] -> s
    start = (frame.cam_intr[0:2, 2].flip(0).int() - frame_resol_tar // 2).clamp(
        min=torch.tensor([0, 0]),
        max=frame_resol_src - frame_resol_tar,
    )
    # [cy, cx] + [iy/2, ix/2] -> e
    end = start + frame_resol_tar
    # [..., sy:ey, sx:ex]
    index = (slice(None),) + tuple(slice(*r) for r in zip(start, end))
    # [cx, cy] - [sx, sy] -> [ix/2, iy/2]
    intr = frame.cam_intr.clone()
    intr[:2, 2] -= start.flip(0)

    return PipelineFrame(
        cam_extr=frame.cam_extr,
        cam_intr=intr,
        depth=frame.depth[index],
        image=frame.image[index],
        mask_depth=frame.mask_depth[index],
        mask_image=frame.mask_image[index],
    )


def flip_frame(frame: PipelineFrame, axis: int) -> PipelineFrame:
    """Flip frame along the given spatial axis."""

    # (C, H, W)
    dim = axis - 2
    depth = frame.depth.flip(dim)
    image = frame.image.flip(dim)
    mask_depth = frame.mask_depth.flip(dim)
    mask_image = frame.mask_image.flip(dim)

    # axis=0 -> -Y -> Y
    # axis=1 -> -X -> X
    extr = frame.cam_extr.clone()
    extr[1 - axis] = -extr[1 - axis]

    # axis=0 -> H - 1 - cy -> cy
    # axis=1 -> W - 1 - cx -> cx
    intr = frame.cam_intr.clone()
    intr[1 - axis, 2] = frame.image.shape[dim] - 1 - intr[1 - axis, 2]

    return PipelineFrame(
        cam_extr=extr,
        cam_intr=intr,
        depth=depth,
        image=image,
        mask_depth=mask_depth,
        mask_image=mask_image,
    )


def flip_frame_horizontally(frame: PipelineFrame) -> PipelineFrame:
    """Flip frame along the horizontal axis."""

    return flip_frame(frame, axis=1)


def flip_frame_vertically(frame: PipelineFrame) -> PipelineFrame:
    """Flip frame along the vertical axis."""

    return flip_frame(frame, axis=0)


def get_cam_extr_in_first_frame_canon(extr: Tensor) -> Tensor:
    """
    Get camera extrinsic in first-frame-canonical coordinate.

    Shape:
        (..., S, 3, 4) -> (..., S, 3, 4)
    """

    # Inverse first camera extrinsic
    # (..., S, 3, 4) -> (..., 1, 3, 4)
    extr_inv_0 = torch.cat(get_cam_extr_inv(extr[..., 0:1, :, :]), dim=-1)
    # (..., 1, 3 + 1, 4)
    extr_inv_0 = torch.cat(
        [
            extr_inv_0,
            extr_inv_0.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(
                extr_inv_0.shape[0:-2] + (1, 4),
            ),
        ],
        dim=-2,
    )

    # Transform to first-frame-canonical coordinate
    # (..., S, 3, 4) @ (..., 1, 4, 4) -> (..., S, 3, 4)
    return extr @ extr_inv_0


def get_cam_extr_inv(extr: Tensor) -> tuple[Tensor, Tensor]:
    """
    Get camera extrinsic inverse.

    Formula:
        R^-1 = R^t
        T^-1 = -R^t @ T

    Shape:
        (..., 3, 4) -> [(..., 3, 3), (..., 3, 1)]
    """

    r, t = extr.split([3, 1], dim=-1)
    r_inv = r.mT.contiguous()
    t_inv = r_inv.matmul(t).neg()
    return r_inv, t_inv


def get_cam_from_enc(frame_resol: tuple[int, int], cam_enc: Tensor) -> tuple[Tensor, Tensor]:
    """
    Get camera extrinsic and intrinsic from camera encoding.

    Shape:
        [[H, W], (..., 3 + 4 + 2) in world + wxyz + [h, w]] -> [(..., 3, 4), (..., 3, 3)]
    """

    FOV_BOUND = (0.2, 2.8)

    t, q, fov = cam_enc.split([3, 4, 2], dim=-1)
    dim = t.shape[0:-1]

    # (..., 4) in wxyz -> (..., 3, 3)
    r = get_matr_from_quat(q)
    # (..., 3, 3 + 1) -> (..., 3, 4)
    extr = torch.cat([r, t[..., None]], dim=-1)
    # (..., 2) in [h, w] -> in [w, h]
    pp = fov.new_tensor(frame_resol).mul(0.5).flip(-1).expand_as(fov)
    fl = pp.div(fov.clamp(*FOV_BOUND).mul(0.5).tan().flip(-1))
    # (..., 2, 2 + 1) -> (..., 2 + 1, 3) -> (..., 3, 3)
    intr = torch.cat(
        [
            torch.cat([fl.diag_embed(), pp[..., None]], dim=-1),
            fov.new_tensor([0.0, 0.0, 1.0]).expand(dim + (1, 3)),
        ],
        dim=-2,
    )
    return extr, intr


def get_depth_unprojected(depth: Tensor, ray_d: Tensor, ray_o: Tensor) -> Tensor:
    """
    Get depth unprojected to point in world coordinate.

    Formula:
        P = D * Y + C

    Shape:
        [(B, S, 1, H, W), (B, S, 3, H, W), (B, S, 3, H, W)] -> (B, S, 3, H, W)
    """

    point = depth.mul(ray_d).add(ray_o)
    return point


def get_enc_from_cam(
    frame_resol: tuple[int, int],
    cam_extr: Tensor,
    cam_intr: Tensor,
) -> Tensor:
    """
    Get camera encoding from camera extrinsic and intrinsic.

    Shape:
        [[H, W], (..., 3, 4), (..., 3, 3)] -> [(..., 3 + 4 + 2) in world + wxyz + [h, w]]
    """

    # (..., 3, 4) -> (..., 2) in [h, w]
    fov = (
        cam_intr.new_tensor(frame_resol)
        .mul(0.5)
        .div(cam_intr.diagonal(dim1=-2, dim2=-1)[..., 0:2].flip(-1))
        .atan()
        .div(0.5)
    )
    # (..., 3, 4) -> [(..., 3, 3), (..., 3, 1)]
    r, t = cam_extr.split([3, 1], dim=-1)
    # (..., 3, 3) -> (..., 4) in wxyz
    q = get_quat_from_matr(r)
    # (..., 3, 1) -> (..., 3) in world
    t = t.squeeze(-1)

    return torch.cat([t, q, fov], dim=-1)


def get_mask_for_depth(depth: Tensor) -> Tensor:
    """Get valid mask for depth."""

    return depth > torch.finfo(depth.dtype).eps


def get_mask_for_first_elem_and_subset(set_size: int, subset_size: int) -> Tensor:
    """Get mask for first element and roughly evenly spaced subset."""

    index = [i * set_size // subset_size for i in range(subset_size)]
    mask = torch.zeros(set_size, dtype=torch.bool)
    mask[index] = True
    return mask


def get_matr_from_quat(quat: Tensor) -> Tensor:
    """
    Get rotation matrix from quaternion.

    Shape:
        (..., 4) in wxyz -> (..., 3, 3)
    """

    eps = torch.finfo(quat.dtype).eps
    # (..., 4) -> [(...); 4]
    w, x, y, z = torch.unbind(quat, dim=-1)
    # (...)
    s = quat.new_tensor(2.0) / quat.square().sum(dim=-1).clamp(min=eps)
    # [(...); 9] -> (..., 3, 3) in col
    return torch.stack(
        (
            1 - (y * y + z * z) * s,
            s * (x * y - z * w),
            s * (x * z + y * w),
            s * (x * y + z * w),
            1 - (x * x + z * z) * s,
            s * (y * z - x * w),
            s * (x * z - y * w),
            s * (y * z + x * w),
            1 - (x * x + y * y) * s,
        ),
        dim=-1,
    ).unflatten(-1, (3, 3))


def get_quat_from_matr(matr: Tensor) -> Tensor:
    """
    Get quaternion from rotation matrix via soft branch selection.

    Shape:
        (..., 3, 3) -> (..., 4) in wxyz
    """

    eps = torch.finfo(matr.dtype).eps
    # [(...); 9]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matr.flatten(-2, -1), dim=-1)
    # [(...); 4] -> (..., 4) in wxyz
    q_sqr = (
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    ).clamp(min=eps)
    q_abs = q_sqr.sqrt()
    # [[(...); 4]; 4] -> (..., 4, 4)
    q_cand = torch.stack(
        [
            torch.stack([q_sqr[..., 0], (m21 - m12), (m02 - m20), (m10 - m01)], dim=-1),
            torch.stack([(m21 - m12), q_sqr[..., 1], (m10 + m01), (m02 + m20)], dim=-1),
            torch.stack([(m02 - m20), (m10 + m01), q_sqr[..., 2], (m12 + m21)], dim=-1),
            torch.stack([(m10 - m01), (m20 + m02), (m21 + m12), q_sqr[..., 3]], dim=-1),
        ],
        dim=-2,
    )
    # Soft branch selection
    # (..., 4) -> (..., 4, 1)
    q_abs2 = q_abs.mul(2.0)[..., None]
    # (..., 4, 4) / (..., 4, 1) -> (..., 4, 4)
    q_cand = q_cand.div(q_abs2)
    q_cand = q_cand.where(q_cand[..., 0:1].ge(0.0), -q_cand)
    # (..., 4, 1) * (..., 4, 4) -> (..., 4)
    q_out = q_abs2.mul(20.0).softmax(dim=-2).mul(q_cand).sum(dim=-2)
    q_out = nnf.normalize(q_out, dim=-1, eps=eps)
    return q_out


def get_ray_unprojected(
    frame_resol: tuple[int, int],
    extr: Tensor,
    intr: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Get camera unprojected to ray in world coordinate.

    Formula:
        Y = R^t @ K^-1 @ [u, v, 1]^t
        C = -R^t @ T

    Shape:
        [[H, W], (..., 3, 4) in W2C, (..., 3, 3)] -> [(..., 3, H, W) in world; 2]
    """

    H, W = frame_resol
    eps = torch.finfo(intr.dtype).eps

    # Create pixel grid
    # [(H, W); 2]
    u, v = torch.meshgrid(
        intr.new_tensor(range(W)),
        intr.new_tensor(range(H)),
        indexing="xy",
    )

    # (..., 3, 3) -> [(..., 1, 1); 4]
    fx = intr[..., 0:1, 0:1]
    fy = intr[..., 1:2, 1:2]
    cx = intr[..., 0:1, 2:3]
    cy = intr[..., 1:2, 2:3]

    # Ray direction in camera coordinate
    # [(..., H, W); 3] -> (..., H, W, 3) -> (..., H, W, 3, 1)
    ray_x = u.sub(cx).div(fx.clamp(min=eps))
    ray_y = v.sub(cy).div(fy.clamp(min=eps))
    ray_z = torch.ones_like(ray_x)
    y_cam = torch.stack([ray_x, ray_y, ray_z], dim=-1)[..., None]

    # C2W transformation
    # (..., 1, 1, 3, 4) -> [(..., 1, 1, 3, 3), (..., 1, 1, 3, 1)]
    r_inv, t_inv = get_cam_extr_inv(extr[..., None, None, :, :])
    # (..., 1, 1, 3, 3) @ (..., H, W, 3, 1) -> (..., H, W, 3, 1)
    # [(..., H, W, 3, 1), (..., 1, 1, 3, 1)] -> [(..., 3, H, W), (..., 3, 1, 1)]
    y_wld, c_wld = tuple(
        v.squeeze(-1).movedim(-1, -3).contiguous() for v in (r_inv.matmul(y_cam), t_inv)
    )
    # (..., 3, 1, 1) -> (..., 3, H, W)
    c_wld = c_wld.expand_as(y_wld).contiguous()
    return y_wld, c_wld


def load_image(path: Path) -> Tensor:
    """
    Load RGB image.

    Shape:
        (3, H, W) in uint8
    """

    return tvf.to_dtype(tvf.to_image(Image.open(path).convert("RGB")), torch.uint8)


def load_mask(path: Path) -> Tensor:
    """
    Load mask.

    Shape:
        (1, H, W) in bool
    """

    return tvf.to_dtype(tvf.to_image(Image.open(path)), torch.bool)


def normalize_depth_extr_point(data: PipelineData) -> PipelineData:
    """Normalize depth, camera extrinsic (translation), and point by average point distance."""

    dtype = data.depth.dtype
    eps = torch.finfo(dtype).eps

    # Camera in first-frame canonical coordinate
    # (..., S, 3, 4)
    extr = get_cam_extr_in_first_frame_canon(data.cam_extr)

    # Flatten
    # (..., S, 1, H, W) -> (..., N)
    depth = data.depth.flatten(-4, -1)
    # (..., S, 1, H, W) -> (..., N)
    mask = data.mask_depth.flatten(-4, -1)
    # (..., S, 3, 1) -> (..., N)
    extr_t = extr[..., 3:4].flatten(-3, -1)
    # (..., S, 3, H, W) -> (..., S, 3, H * W)
    point = data.point.flatten(-2, -1)

    # Point in first-frame canonical coordinate
    # (..., 1, 3, 3) @ (..., S, 3, H * W) + (..., 1, 3, 1) -> (..., S, 3, H * W)
    extr_r_0, extr_t_0 = data.cam_extr[..., 0:1, :, :].split([3, 1], dim=-1)
    point = extr_r_0 @ point + extr_t_0
    # Point distance
    # (..., S, 3, H * W) -> (..., S, H * W) -> (..., N)
    dist = point.norm(dim=-2).nan_to_num().clamp(eps, 1.0 / eps).flatten(-2, -1)
    # Flatten
    # (..., S, 3, H * W) -> (..., N)
    point = point.flatten(-3, -1)

    # Rescale by average distance
    # (..., N) * (..., N) / (..., N) -> (..., 1)
    scale = dist.mul(mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=eps)
    scale = scale.clamp(eps, 1.0 / eps)[..., None]
    # (..., N) / (..., 1) -> (..., N)
    depth = depth / scale
    extr_t = extr_t / scale
    point = point / scale

    # Unflatten
    # (..., N) -> (..., S, 1, H, W)
    depth = depth.unflatten(-1, data.depth.shape[-4:])
    # (..., N) -> (..., S, 3, 1)
    extr_t = extr_t.unflatten(-1, data.cam_extr.shape[-3:-1] + (1,))
    extr[..., 3:4] = extr_t
    # (..., N) -> (..., S, 3, H, W)
    point = point.unflatten(-1, data.point.shape[-4:])

    return data._replace(cam_extr=extr, depth=depth, point=point)


def resize_frame_on_shorter(frame: PipelineFrame, frame_resol: int) -> PipelineFrame:
    """Resize frame on shorter edge while maintaining aspect ratio."""

    # NOTE: Margin for off-center principal point
    margin = 4
    frame_resol = int(frame_resol + margin)

    depth = tvf.resize(frame.depth, frame_resol, InterpolationMode.NEAREST)
    image = tvf.resize(frame.image, frame_resol, InterpolationMode.BICUBIC)
    mask_depth = tvf.resize(frame.mask_depth, frame_resol, InterpolationMode.NEAREST)
    mask_image = tvf.resize(frame.mask_image, frame_resol, InterpolationMode.NEAREST)

    scale = min(image.shape[-2:]) / min(frame.image.shape[-2:])
    intr = frame.cam_intr.clone()
    intr[:2, 2] += 0.5
    intr[:2, :] *= scale
    intr[:2, 2] -= 0.5

    return PipelineFrame(
        cam_extr=frame.cam_extr,
        cam_intr=intr,
        depth=depth,
        image=image,
        mask_depth=mask_depth,
        mask_image=mask_image,
    )


def sample_stratified_temporal_bin(
    bin_count: int,
    frame_count: int,
    random: bool,
) -> Tensor:
    """
    Sample one frame per temporal bin via stratified sampling.

    Shape:
        (bin_count,)
    """

    if frame_count < bin_count:
        raise ValueError(f"Not enough frames: expected {bin_count}, got {frame_count}")

    origin = torch.arange(bin_count) * frame_count // bin_count
    if not random:
        return origin

    step = frame_count // bin_count
    pad = int(bool(frame_count % bin_count))
    shift = torch.randint(0, step + pad, (bin_count,))
    return origin + shift


def transpose_frame(frame: PipelineFrame) -> PipelineFrame:
    """Transpose frame across spatial dimensions."""

    # (C, H, W) -> (C, W, H)
    depth = frame.depth.mT.contiguous()
    image = frame.image.mT.contiguous()
    mask_depth = frame.mask_depth.mT.contiguous()
    mask_image = frame.mask_image.mT.contiguous()

    # X <-> Y
    extr = frame.cam_extr.clone()
    extr[0:2] = frame.cam_extr[0:2].flip(0)

    # [fx, cx] <-> [fy, cy]
    intr = frame.cam_intr.clone()
    intr[0:2] = frame.cam_intr[0:2].flip(0)
    intr[:, 0:2] = intr[:, 0:2].flip(1)

    return PipelineFrame(
        cam_extr=extr,
        cam_intr=intr,
        depth=depth,
        image=image,
        mask_depth=mask_depth,
        mask_image=mask_image,
    )


def unproject(depth: Tensor, extr: Tensor, intr: Tensor) -> Tensor:
    """
    Unproject camera and depth to point in world coordinate.

    Shape:
        [(..., 1, H, W), (..., 3, 4) in W2C, (..., 3, 3)] -> (..., 3, H, W)
    """

    ray_d, ray_o = get_ray_unprojected(frame_resol=depth.shape[-2:], extr=extr, intr=intr)
    ray_d = ray_d.reshape(depth.shape[:-3] + ray_d.shape[-3:])
    ray_o = ray_o.reshape(depth.shape[:-3] + ray_o.shape[-3:])
    return get_depth_unprojected(depth=depth, ray_d=ray_d, ray_o=ray_o)
