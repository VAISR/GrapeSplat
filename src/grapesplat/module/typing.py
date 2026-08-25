"""Module layer typing definitions for GrapeSplat."""

from ..data import PipelineData
from PIL.Image import Image
from torch import Tensor
from torchsparse import SparseTensor
from typing import NamedTuple, TypedDict

__all__ = ["PipelineData"]


class CamRayDecoderInp(NamedTuple):
    """Input to Camera and ray decoder pipeline module."""

    out_image: "ImageEncoderOut"
    """Output from Image encoder pipeline module."""


class CamRayDecoderOut(NamedTuple):
    """Output from Camera and ray decoder pipeline module."""

    cam_enc: Tensor
    """(B, S, 3 + 4 + 2) in world + wxyz + [h, w]"""
    cam_extr: Tensor
    """(B, S, 3, 4) in W2C"""
    cam_intr: Tensor
    """(B, S, 3, 3) in PC"""
    ray_d: Tensor
    """(B, S, 3, H, W) in world"""
    ray_o: Tensor
    """(B, S, 3, H, W) in world"""


class CompositeLossOut(TypedDict):
    """Composite loss for pipeline objectives."""

    loss: Tensor
    """The weighted objective."""
    loss_cam: Tensor
    """Camera objective."""
    loss_depth1: Tensor
    """Depth 1 objective."""
    loss_depth2: Tensor
    """Depth 2 objective."""
    loss_image_perc: Tensor
    """Image perceptual objective."""
    loss_image_regr: Tensor
    """Image regression objective."""
    loss_point: Tensor
    """Point objective."""


class CompositeMetricOut(TypedDict, total=False):
    """Composite metric for evaluation."""

    depth1_absrel: Tensor
    """Depth 1 Absolute Relative Error."""
    depth1_delta1: Tensor
    """Depth 1 Threshold Accuracy."""
    depth2_absrel: Tensor
    """Depth 2 Absolute Relative Error."""
    depth2_delta1: Tensor
    """Depth 2 Threshold Accuracy."""
    image_lpips: Tensor
    """Image Learned Perceptual Image Patch Similarity."""
    image_psnr: Tensor
    """Image Peak Signal-to-Noise Ratio."""
    image_ssim: Tensor
    """Image Structural Similarity Index Measure."""
    point_cd: Tensor
    """Point Chamfer Distance."""
    point_nc: Tensor
    """Point Normal Consistency."""
    pose_rel_auc3: Tensor
    """Area under the curve for relative pose angular accuracy with 3 degree threshold."""
    pose_rel_auc30: Tensor
    """Area under the curve for relative pose angular accuracy with 30 degree threshold."""
    scene_numel: Tensor
    """Number of primitives in the scene."""


class Depth1DecoderInp(NamedTuple):
    """Input to Depth 1 decoder pipeline module."""

    out_image: "ImageEncoderOut"
    """Output from Image encoder pipeline module."""


class Depth1DecoderOut(NamedTuple):
    """Output from Depth 1 decoder pipeline module."""

    depth: Tensor
    """(B, S, 1, H, W) in exp"""
    depth_conf: Tensor
    """(B, S, 1, H, W) in exp+1"""


class Depth2ImageDecoderInp(NamedTuple):
    """Input to Depth 2 and image decoder pipeline module."""

    cam_extr: Tensor
    """(B, S, 3, 4) in W2C"""
    cam_intr: Tensor
    """(B, S, 3, 3) in PC"""
    frame_resol: tuple[int, int]
    """[H, W]"""
    scene: "SceneDecoderOut"
    """Scene decoder output (B)."""


class Depth2ImageDecoderOut(NamedTuple):
    """Output from Depth 2 and image decoder pipeline module."""

    depth: Tensor
    """(B, S, 1, H, W) in [0, inf)"""
    image: Tensor
    """(B, S, 3, H, W) in [0, 1]"""


class FeatBridgeInp(NamedTuple):
    """Input to Feature bridge pipeline module."""

    image: Tensor
    """(B, S, 3, H, W) in [0, 1]"""
    out_image: "ImageEncoderOut"
    """Output from Image encoder pipeline module."""


class FeatBridgeOut(NamedTuple):
    """Output from Feature bridge pipeline module."""

    feat: Tensor
    """(B, S, C_o, H, W)"""


class GaussHeadInp(NamedTuple):
    """Input to Gaussian head."""

    grid: SparseTensor
    """(V_i, C_i)"""


class GaussHeadOut(NamedTuple):
    """Output from Gaussian head."""

    batch_index: Tensor
    """(V_o,)"""
    grid_resol: int
    """g"""
    color: Tensor
    """(V_o, K, 3) in SH"""
    opacity: Tensor
    """(V_o, 1) in [0, 1]"""
    pos_grid: Tensor
    """(V_o, 3) in [0, g - 1]"""
    pos_local: Tensor
    """(V_o, 3) in [0, 1]"""
    rotation: Tensor
    """(V_o, 3, 3) in world"""
    scale_local: Tensor
    """(V_o, 3) in local"""


class ImageEncoderInp(NamedTuple):
    """Input to Image encoder pipeline module."""

    image: Tensor
    """(B, S, 3, H, W) in [0, 1]"""


class ImageEncoderOut(NamedTuple):
    """Output from Image encoder pipeline module."""

    frame_resol: tuple[int, int]
    """[H, W]"""
    patch_index_geo: int
    """1 + R_g"""
    patch_index_sem: int
    """1 + R_s"""
    token_geo: list[Tensor]
    """[(B, S, 1 + R_g + H_P_g * W_P_g, C_g); L]"""
    token_sem: list[Tensor]
    """[(B, S, 1 + R_s + H_P_s * W_P_s, C_s); 1]"""


class PipelineInp(NamedTuple):
    """Input to pipeline orchestrator."""

    cam_extr: Tensor
    """(B, S, 3, 4) in W2C"""
    cam_intr: Tensor
    """(B, S, 3, 3)"""
    image: Tensor
    """(B, S_rec, 3, H, W) in [0, 1]"""
    mask_rec: Tensor
    """(S,) where sum = S_rec"""
    mask_ren: Tensor
    """(S,) where sum = S_ren"""


class PipelineOut(NamedTuple):
    """Output from Pipeline orchestrator."""

    batch_size: int
    """B"""
    cam_enc: Tensor
    """(B, S_rec, 9)"""
    cam_extr: Tensor
    """(B, S_rec, 3, 4) in W2C"""
    cam_intr: Tensor
    """(B, S_rec, 3, 3)"""
    depth1: Tensor
    """(B, S_rec, 1, H, W) in exp"""
    depth1_conf: Tensor
    """(B, S_rec, 1, H, W) in exp+1"""
    depth2_rec: Tensor
    """(B, S_rec, 1, H, W) in [0, inf)"""
    depth2_ren: Tensor
    """(B, S_ren, 1, H, W) in [0, inf)"""
    frame_resol: tuple[int, int]
    """[H, W]"""
    image_rec: Tensor
    """(B, S_rec, 3, H, W) in [0, 1]"""
    image_ren: Tensor
    """(B, S_ren, 3, H, W) in [0, 1]"""
    point: Tensor
    """(B, S_rec, 3, H, W) in world"""
    ray_d: Tensor
    """(B, S_rec, 3, H, W) in world"""
    ray_o: Tensor
    """(B, S_rec, 3, H, W) in world"""
    scene: "SceneDecoderOut"
    """Scene decoder output (B)."""


class PipelineVis(TypedDict, total=True):
    """Visualization output from Pipeline orchestrator."""

    data_cam: list[Image]
    """[(H, W, 3); B]"""
    data_depth: list[Image]
    """[(H, W, 3); B * S_ren]"""
    data_image: list[Image]
    """[(H, W, 3); B * S_ren]"""
    data_point: list[Image]
    """[(H, W, 3); B * S_rec]"""
    data_ray_d: list[Image]
    """[(H, W, 3); B * S_rec]"""
    pred_cam: list[Image]
    """[(H, W, 3); B]"""
    pred_depth1: list[Image]
    """[(H, W, 3); B * S_rec]"""
    pred_depth1_conf: list[Image]
    """[(H, W, 3); B * S_rec]"""
    pred_depth2_rec: list[Image]
    """[(H, W, 3); B * S_rec]"""
    pred_depth2_ren: list[Image]
    """[(H, W, 3); B * S_ren]"""
    pred_image_rec: list[Image]
    """[(H, W, 3); B * S_rec]"""
    pred_image_ren: list[Image]
    """[(H, W, 3); B * S_ren]"""
    pred_point: list[Image]
    """[(H, W, 3); B * S_rec]"""
    pred_ray_d: list[Image]
    """[(H, W, 3); B * S_rec]"""
    pred_scene_rec: list[Image]
    """[(H, W, 3); B * S_rec]"""


class SceneDecoderInp(NamedTuple):
    """Input to Scene decoder pipeline module."""

    image: Tensor
    """(B, S, 3, H, W) in [0, 1]"""
    out_depth1: Depth1DecoderOut
    """Output from Depth 1 decoder pipeline module."""
    out_image: ImageEncoderOut
    """Output from Image encoder pipeline module."""
    out_ray: CamRayDecoderOut
    """Output from Ray decoder pipeline module."""
    point: Tensor
    """(B, S, 3, H, W) in world"""


class SceneDecoderOut(NamedTuple):
    """Output from Scene decoder pipeline module."""

    batch_index: Tensor
    """(V,)"""
    batch_size: int
    """B"""
    color: Tensor
    """(V, K, 3) in SH"""
    covariance: Tensor
    """(V, 3, 3) in world"""
    opacity: Tensor
    """(V, 1) in [0, 1] | (V, K, 1) in SH"""
    position: Tensor
    """(V, 3) in world"""


class SceneOptimizerInp(NamedTuple):
    """Input to Scene optimizer pipeline module."""

    image: Tensor
    """(B, S_rec, 3, H, W) in [0, 1]"""
    inp_depth2_image: Depth2ImageDecoderInp
    """Input to Renderer pipeline module."""


class UNetInp(NamedTuple):
    """Input to UNet module."""

    grid: SparseTensor
    """(V, C_i)"""


class UNetOut(NamedTuple):
    """Output from UNet module."""

    grid: SparseTensor
    """(V, C_o)"""


class VoxelizeDeInp(NamedTuple):
    """Input to De-voxelization module."""

    batch_index: Tensor
    """(V,)"""
    centroid: Tensor
    """(B, 3)"""
    grid_resol: int
    """g"""
    pos_grid: Tensor
    """(V, 3) in [0, g - 1]"""
    pos_local: Tensor
    """(V, 3) in [0, 1]"""
    rotation: Tensor
    """(V, 3, 3)"""
    scale: Tensor
    """(B, 1)"""
    scale_local: Tensor
    """(V, 3)"""


class VoxelizeDeOut(NamedTuple):
    """Output from De-voxelization module."""

    cov_world: Tensor
    """(V, 3, 3)"""
    pos_world: Tensor
    """(V, 3)"""


class VoxelizeInp(NamedTuple):
    """Input to Voxelization module."""

    inp_scene: SceneDecoderInp
    """Input to Scene decoder pipeline module."""


class VoxelizeOut(NamedTuple):
    """Output from Voxelization module."""

    batch_size: int
    """B"""
    centroid: Tensor
    """(B, 3)"""
    grid: SparseTensor
    """(V, C_o)"""
    scale: Tensor
    """(B, 1)"""
