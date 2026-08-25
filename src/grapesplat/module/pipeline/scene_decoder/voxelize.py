"""Voxelization modules for Scene decoder."""

from ... import typing as Tp
from .bridge import FeatBridgeBuilder
from .code import SPEBuilder
from .mlp import MLPBlockBuilder
from dataclasses import dataclass
from torch import Tensor, nn
from torch.nn import functional as nnf
from torchsparse import SparseTensor
from torchsparse.nn import functional as spnnf
import torch


@dataclass
class VoxelizeBuilder:
    """Per-axis Extent-normalized Assembly of Cue-aware Hypotenuse-companded Voxelization (PEACH-Vox)."""

    dim_h_base: int
    dim_h_mult: float
    feat_bridge: FeatBridgeBuilder
    grid_resol: int
    peach: bool
    spe_depth_log: SPEBuilder
    spe_image: SPEBuilder
    spe_pos_local: SPEBuilder
    spe_pos_world: SPEBuilder
    spe_ray: SPEBuilder
    voxel_count_train_max: int

    dim_i: int = None

    def __call__(self, base_geo: nn.Module, base_sem: nn.Module, dim_i: int) -> "Voxelize":
        """Build the module."""

        self.dim_i = dim_i

        if self.spe_pos_world.band is None:
            self.spe_pos_world.band = (self.grid_resol - 1).bit_length()

        return Voxelize(base_geo=base_geo, base_sem=base_sem, config=self)

    @property
    def dim_h(self) -> int:
        return self.dim_h_base

    @property
    def dim_o(self) -> int:
        return self.dim_h

    @property
    def dim_w(self) -> int:
        return (
            (1 + 1) * self.spe_depth_log.dim_x
            + 3 * self.spe_image.dim_x
            + (3 + 3) * self.spe_ray.dim_x
        )

    @property
    def dim_x(self) -> int:
        return 3 * self.spe_pos_local.dim_x + 3 * self.spe_pos_world.dim_x + 2


class Voxelize(nn.Module):
    """Per-axis Extent-normalized Assembly of Cue-aware Hypotenuse-companded Voxelization (PEACH-Vox)."""

    def __init__(self, base_geo: nn.Module, base_sem: nn.Module, config: VoxelizeBuilder) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        self.feat_bridge = config.feat_bridge(
            base_geo=base_geo,
            base_sem=base_sem,
            dim_o=config.dim_i,
        )
        config.dim_i = config.feat_bridge.dim_o
        self.fuse = nn.ModuleList(
            [
                MLPBlockBuilder(
                    dim_h_base=config.dim_h_base,
                    dim_h_mult=config.dim_h_mult,
                    dim_o=config.dim_h,
                )(dim_i=config.dim_i + config.dim_w),
                MLPBlockBuilder(
                    dim_h_base=config.dim_h_base,
                    dim_h_mult=config.dim_h_mult,
                    dim_o=config.dim_o,
                )(dim_i=config.dim_h + config.dim_x),
            ]
        )
        self.spe_depth_log = config.spe_depth_log()
        self.spe_image = config.spe_image()
        self.spe_pos_local = config.spe_pos_local()
        self.spe_pos_world = config.spe_pos_world()
        self.spe_ray = config.spe_ray()

    def forward(self, inp: Tp.VoxelizeInp) -> Tp.VoxelizeOut:
        with torch.autocast(inp.inp_scene.point.device.type, enabled=False):
            return self.forward_unwrapped(inp)

    def forward_unwrapped(self, inp: Tp.VoxelizeInp) -> Tp.VoxelizeOut:
        x = inp.inp_scene
        B, S, _, H, W = x.point.shape
        N = S * H * W
        G = self.config.grid_resol
        AFFINE_BOUND_PR = 0.98
        FUSE1_CHUNK_SIZE = 2**18
        RAVEL_I32_MAX = 2**31 - 1
        dtype = x.point.dtype
        eps = torch.finfo(dtype).eps

        # Flatten
        # (B, S, 3, H, W) -> (B, S, H, W, 3) -> (B, N, 3)
        pos_wld = x.point.movedim(-3, -1).contiguous().flatten(-4, -2)

        # Move world centroid to origin
        # (B, N, 3) -> (B, 3)
        centroid = pos_wld.mean(dim=-2)
        # (B, N, 3) - (B, 1, 3) -> (B, N, 3)
        pos_c = pos_wld.sub(centroid[:, None])
        # Calculate point distance to centroid
        # (B, N, 1)
        dst_c = pos_c.norm(dim=-1, keepdim=True).nan_to_num().clamp(eps, 1.0 / eps)

        if self.config.peach:
            # Compressing
            # (B, N, 1) -> (B, 1, 1)
            scale = dst_c.mean(dim=-2, keepdim=True)
            # (B, N, 3) / (B, N, 3) -> (B, N, 3) in (0, 1)
            pos_nuc = pos_c.div(scale.hypot(pos_c)).mul(0.5).add(0.5)
        else:
            # Normalizing
            # (B, N, 1) -> (B, 1, 1)
            scale = dst_c.kthvalue(k=int(N * AFFINE_BOUND_PR), dim=-2, keepdim=True).values
            # (B, N, 3) / (B, 1, 1) -> (B, N, 3) in [-1, 1] + {ext}
            pos_ndc = pos_c.div(scale)
            # (B, N, 3) in [0, 1]
            pos_nuc = pos_ndc.mul(0.5).add(0.5)

        # (B, 1, 1) -> (B, 1)
        scale = scale.squeeze(-1)

        # Position in local / voxel / world coordinate
        # (B, N, 3) -> (M, 3) in [0, 1] / [0, g - 1] / [0, 1]
        pos_world = pos_nuc.flatten(-3, -2).clamp(0.0, 1.0)
        pos_fused = pos_world.mul(G)
        pos_local = pos_fused.frac()
        pos_voxel = pos_fused.int().clamp(0, G - 1)

        # Batch index
        # (B * N,) -> (M,)
        batch_index = pos_voxel.new_tensor(range(B)).repeat_interleave(N)
        # Batch-indexed position in voxel coord
        # (M, 1 + 3) in [b, x, y, z]
        coord = torch.cat([batch_index[..., None], pos_voxel], dim=-1)
        spatial_range = (B,) + (G,) * 3

        # Ravel coordinate
        # (M, 4) -> (M,)
        c = coord.to(dtype=torch.int64 if B * G**3 > RAVEL_I32_MAX else torch.int32)
        h = ((c[..., 0] * G + c[..., 1]) * G + c[..., 2]) * G + c[..., 3]
        # Quantize coordinate
        # [(V,), (M,), (V,)]
        t: tuple[Tensor, Tensor, Tensor] = h.unique(
            return_counts=True,
            return_inverse=True,
            sorted=True,
        )
        h_uni, point_index, count = t
        count, point_index = count.int(), point_index.int()
        V = h_uni.shape[0]
        # Unravel coordinate
        # (V,) -> [(V,), (V, 4)]
        v_uni = count.new_empty(V, 4)
        v_uni[..., 3], h_uni = h_uni % G, h_uni // G
        v_uni[..., 2], h_uni = h_uni % G, h_uni // G
        v_uni[..., 1], h_uni = h_uni % G, h_uni // G
        v_uni[..., 0] = h_uni
        batch_index, pos_grid = h_uni, v_uni

        # Voxel aggregation for coordinate
        # (M, 3)[(V,)] -> (V, 3)
        pos_local_agg = spnnf.spvoxelize(pos_local, point_index, count)
        pos_world_agg = spnnf.spvoxelize(pos_world, point_index, count)

        # Early feature fusion
        # (V, C_h)
        feat_agg = x.point.new_zeros(V, self.config.dim_h)
        for i in range(S):
            s = slice(i, i + 1)
            # (B, 1, H, W, C_h) -> (M / S, C_h)
            feat = self.fuse0_per_frame(x, s).flatten(0, -2)
            # Voxel aggregation for feature
            # (M / S,)
            point_index_per_frame = point_index.reshape(B, S, -1)[:, s].reshape(-1)
            # (M / S, C_h)[(M / S,)] -> (V / S, C_h)
            feat_agg.add_(spnnf.spvoxelize(feat, point_index_per_frame, count))

        # Release image encoder tokens in place
        x.out_image.token_geo.clear()
        x.out_image.token_sem.clear()

        # Density conditioning
        # (V,) -> (V, 1)
        mag = count.to(dtype).sqrt()[..., None]
        feat_agg = feat_agg.mul(mag)

        # Extent conditioning
        # (B, 1)[(V,)] -> (V, 1)
        ext = scale[batch_index]

        # Latter feature fusion
        # (V, C_o)
        feat_grid = feat_agg.new_empty(V, self.config.dim_o)
        for i in range(0, V, FUSE1_CHUNK_SIZE):
            s = slice(i, i + FUSE1_CHUNK_SIZE)
            feat_grid[s] = self.fuse1_per_chunk(ext, feat_agg, mag, pos_local_agg, pos_world_agg, s)

        # Training-time voxel pruning
        feat_grid, pos_grid = self.prune(feat_grid, pos_grid)

        return Tp.VoxelizeOut(
            batch_size=B,
            centroid=centroid,
            grid=SparseTensor(
                coords=pos_grid,
                feats=feat_grid,
                spatial_range=spatial_range,
                stride=1,
            ),
            scale=scale,
        )

    def fuse0_per_frame(self, x: Tp.SceneDecoderInp, s: slice) -> Tensor:
        """
        Fuse per-point cues per frame.

        Shape:
            (B, S, C_i + C_w, H, W) -> (B, 1, H, W, C_h)
        """

        eps = torch.finfo(x.point.dtype).eps

        with torch.autocast(x.point.device.type, dtype=torch.bfloat16):
            # (B, 1, C_i, H, W)
            out_bridge: Tp.FeatBridgeOut = self.feat_bridge(
                Tp.FeatBridgeInp(
                    image=x.image[:, s],
                    out_image=x.out_image._replace(
                        token_geo=[t if t is None else t[:, s] for t in x.out_image.token_geo],
                        token_sem=[t[:, s] for t in x.out_image.token_sem],
                    ),
                )
            )
        # (B, 1, 1 + 1, H, W)
        depth_log = (
            torch.cat(
                [
                    x.out_depth1.depth_conf[:, s].detach(),
                    x.out_depth1.depth[:, s],
                ],
                dim=-3,
            )
            .clamp(min=eps)
            .log()
        )
        # (B, 1, 3 + 3, H, W)
        ray = torch.cat(
            [
                nnf.normalize(x.out_ray.ray_d[:, s], dim=-3, eps=eps),
                x.out_ray.ray_o[:, s],
            ],
            dim=-3,
        )
        # (B, 1, C_i + C_w, H, W) -> (B, 1, H, W, C_h)
        feat = (
            torch.cat(
                [
                    out_bridge.feat,
                    self.spe_depth_log(depth_log, dim=-3),
                    self.spe_image(x.image[:, s], dim=-3),
                    self.spe_ray(ray, dim=-3),
                ],
                dim=-3,
            )
            .movedim(-3, -1)
            .contiguous()
        )
        return self.fuse[0](feat)

    def fuse1_per_chunk(
        self,
        ext: Tensor,
        feat_agg: Tensor,
        mag: Tensor,
        pos_local_agg: Tensor,
        pos_world_agg: Tensor,
        s: slice,
    ) -> Tensor:
        """
        Fuse per-voxel features per chunk.

        Shape:
            (V, C_h + C_x) -> (V_s, C_o)
        """

        # Positional encoding
        # (V_s, 3) -> (V_s, 3 * (1 + 2L))
        feat_pos_local = self.spe_pos_local(pos_local_agg[s])
        feat_pos_world = self.spe_pos_world(pos_world_agg[s])
        # (V_s, C_h + C_x)
        feat_grid = torch.cat(
            [feat_agg[s], feat_pos_local, feat_pos_world, ext[s], mag[s]],
            dim=-1,
        )
        return self.fuse[1](feat_grid)

    def inverse(self, inp: Tp.VoxelizeDeInp) -> Tp.VoxelizeDeOut:
        """Per-scene de-voxelization from local to world coordinate."""

        with torch.autocast(inp.pos_local.device.type, enabled=False):
            return self.inverse_unwrapped(inp)

    def inverse_unwrapped(self, inp: Tp.VoxelizeDeInp) -> Tp.VoxelizeDeOut:
        G = inp.grid_resol
        eps_sqrt = torch.finfo(inp.pos_local.dtype).eps ** 0.5

        # (B, 3)[(V,)] -> (V, 3)
        centroid = inp.centroid[inp.batch_index]
        # (B, 1)[(V,)] -> (V, 1)
        scale = inp.scale[inp.batch_index]

        # (V, 3) in [0, g] -> in [0, 1]
        pos_nuc = inp.pos_grid.add(inp.pos_local).div(G)
        scl_nuc = inp.scale_local.div(G)
        # (V, 3) in [-1, 1]
        pos_ndc = pos_nuc.sub(0.5).div(0.5).clamp(-1.0 + eps_sqrt, 1.0 - eps_sqrt)
        # (V, 3)
        scl_ndc = scl_nuc.div(0.5)
        # (V, 3, 3) * (V, 1, 3) -> (V, 3, 3)
        shp_ndc = inp.rotation.mul(scl_ndc[..., None, :])

        if self.config.peach:
            # Expanding
            # (V, 3) -> (V, 3)
            stretch = pos_ndc.new_ones(()).sub(pos_ndc.square()).rsqrt()
            # (V, 3) * (V, 1) * (V, 3) -> (V, 3)
            pos_c = pos_ndc.mul(scale).mul(stretch)
            # (V, 1) * (V, 3) -> (V, 3)
            dilation = scale.mul(stretch.pow(3))
            # (V, 3, 3) * (V, 3, 1) -> (V, 3, 3)
            shp_wld = shp_ndc.mul(dilation[..., None])
            # (V, 3) + (V, 3) -> (V, 3)
            pos_wld = pos_c.add(centroid)
        else:
            # Denormalizing
            # (V, 3, 3) * (V, 1, 1) -> (V, 3, 3)
            shp_wld = shp_ndc.mul(scale[..., None])
            # (V, 3) * (V, 1) + (V, 3) -> (V, 3)
            pos_wld = pos_ndc.mul(scale).add(centroid)

        # (V, 3, 3) @ (V, 3, 3) -> (V, 3, 3)
        cov_wld = shp_wld.matmul(shp_wld.mT.contiguous())
        return Tp.VoxelizeDeOut(cov_world=cov_wld, pos_world=pos_wld)

    def prune(self, feat_grid: Tensor, pos_grid: Tensor) -> tuple[Tensor, Tensor]:
        """
        Voxel pruning for saving resources during training.

        Shape:
            [(V_i, C), (V_i, 4)] -> [(V_o, C), (V_o, 4)]
        """

        V, V_MAX = feat_grid.shape[0], self.config.voxel_count_train_max
        if self.training and V > V_MAX:
            # (V_o,)
            sample_index = torch.randperm(V, device=feat_grid.device)[:V_MAX].sort().values
            # (V_i, C)[(V_o,)] -> (V_o, C)
            feat_grid = feat_grid[sample_index]
            pos_grid = pos_grid[sample_index]
        return feat_grid, pos_grid

    def extra_repr(self) -> str:
        return f"G={self.config.grid_resol}, peach={self.config.peach}"
