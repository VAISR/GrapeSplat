"""Pipeline module for GrapeSplat: Scene decoder."""

from ... import typing as Tp
from ..train import TrainStrat, adapt_train_strat
from .head import GaussHeadBuilder
from .unet import UNetBuilder
from .voxelize import VoxelizeBuilder
from dataclasses import dataclass
from torch import nn


@dataclass
class SceneDecoderBuilder:
    """Scene decoder module for 3D Gaussian reconstruction."""

    gausshead: GaussHeadBuilder
    train_strat: TrainStrat
    unet: UNetBuilder
    voxelize: VoxelizeBuilder

    dim_i: int = None

    def __call__(self, base_geo: nn.Module, base_sem: nn.Module, dim_i: int) -> "SceneDecoder":
        """Build the module."""

        self.dim_i = dim_i

        return SceneDecoder(base_geo=base_geo, base_sem=base_sem, config=self)


class SceneDecoder(nn.Module):
    """Scene decoder module for 3D Gaussian reconstruction."""

    def __init__(
        self,
        base_geo: nn.Module,
        base_sem: nn.Module,
        config: SceneDecoderBuilder,
    ) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        self.voxelize = config.voxelize(
            base_geo=base_geo,
            base_sem=base_sem,
            dim_i=config.dim_i,
        )
        self.unet = config.unet(
            dim_i=config.voxelize.dim_o,
        )
        self.gausshead = config.gausshead(
            dim_i=config.unet.dim_o,
        )

        adapt_train_strat(self, config.train_strat)

    def forward(self, inp: Tp.SceneDecoderInp) -> Tp.SceneDecoderOut:
        # Voxelization
        out_vox: Tp.VoxelizeOut = self.voxelize(
            Tp.VoxelizeInp(
                inp_scene=inp,
            )
        )
        # Spatial refinement
        out_unet: Tp.UNetOut = self.unet(
            Tp.UNetInp(
                grid=out_vox.grid,
            )
        )
        # Gaussian prediction
        out_gs: Tp.GaussHeadOut = self.gausshead(
            Tp.GaussHeadInp(
                grid=out_unet.grid,
            )
        )
        # De-voxelization
        out_devox: Tp.VoxelizeDeOut = self.voxelize.inverse(
            Tp.VoxelizeDeInp(
                batch_index=out_gs.batch_index,
                centroid=out_vox.centroid,
                grid_resol=out_gs.grid_resol,
                pos_grid=out_gs.pos_grid,
                pos_local=out_gs.pos_local,
                rotation=out_gs.rotation,
                scale=out_vox.scale,
                scale_local=out_gs.scale_local,
            )
        )

        return Tp.SceneDecoderOut(
            batch_index=out_gs.batch_index,
            batch_size=out_vox.batch_size,
            color=out_gs.color,
            covariance=out_devox.cov_world,
            opacity=out_gs.opacity,
            position=out_devox.pos_world,
        )
