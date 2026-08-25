"""Pipeline module for GrapeSplat: Depth 2 and image decoder."""

from ..typing import Depth2ImageDecoderInp, Depth2ImageDecoderOut
from .train import TrainStrat, adapt_train_strat
from dataclasses import dataclass
from torch import Tensor, nn
from typing import Literal
import gsplat
import torch


@dataclass
class Depth2ImageDecoderBuilder:
    """Depth 2 and image decoder module to render scene."""

    antialiased: bool
    bgcolor: tuple[float, float, float] | None
    depth_mode: Literal["D", "ED"]
    far: float
    near: float
    train_strat: TrainStrat

    def __call__(self) -> "Depth2ImageDecoder":
        """Build the module."""

        return Depth2ImageDecoder(config=self)

    @property
    def raster_mode(self) -> str:
        return "antialiased" if self.antialiased else "classic"


class Depth2ImageDecoder(nn.Module):
    """Depth 2 and image decoder module to render scene."""

    def __init__(self, config: Depth2ImageDecoderBuilder) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        self.register_buffer(
            "bgcolor",
            None if config.bgcolor is None else torch.tensor(config.bgcolor),
            persistent=False,
        )

        adapt_train_strat(self, config.train_strat)

    def forward(self, inp: Depth2ImageDecoderInp) -> Depth2ImageDecoderOut:
        B, S = inp.cam_extr.shape[0:-2]
        H, W = inp.frame_resol

        # Scene primitives
        s = inp.scene
        eps = torch.finfo(s.opacity.dtype).eps

        # Spherical harmonics degree
        sh_degree_color = int(s.color.shape[-2] ** 0.5) - 1
        sh_degree_opacity = None if s.opacity.ndim == 2 else int(s.opacity.shape[-2] ** 0.5) - 1
        is_opacity_sh = sh_degree_opacity is not None

        # 3DGS rasterization per scene
        alphas: list[Tensor] = []
        frames: list[Tensor] = []
        for i in range(B):
            # Collect per-scene Gaussian primitives
            # (V_i, ...)
            index = s.batch_index == i
            cl_i = s.color[index]
            cv_i = s.covariance[index]
            op_i = s.opacity[index]
            po_i = s.position[index]

            # (S, 3, 3) in PC
            intr = inp.cam_intr[i]
            # (S, 3 + 1, 4) in W2C
            extr = inp.cam_extr[i]
            extr = torch.cat(
                [extr, extr.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(S, 1, -1)],
                dim=-2,
            )

            if is_opacity_sh:
                # Per-view opacity
                # (S, 3) in world
                po_cam = extr.inverse()[..., 0:3, 3]
                # (V_i, 3) - (S, 1, 3) -> (S, V_i, 3) in world
                dir_i = po_i.sub(po_cam[..., None, :])
                # (V_i, K_op, 1) -> (S, V_i, K_op, 3) -> (S, V_i, 3) in SH
                op_i = gsplat.spherical_harmonics(
                    sh_degree_opacity,
                    dir_i,
                    op_i[None].expand(S, -1, -1, 3),
                )
                # (S, V_i, 3) -> (S, V_i) in [0, 1]
                op_i = op_i[..., 0].sigmoid()
            else:
                # Per-point opacity
                # (V_i, 1) -> (V_i,) in [0, 1]
                op_i = op_i.squeeze(-1)

            # 3D Gaussian rasterization
            if S > 0 and index.sum().is_nonzero():
                alpha = []
                frame = []
                for j in range(S):
                    s = slice(j, j + 1)
                    # [(1, H, W, 3 + 1), (1, H, W, 1)]
                    frame_j, alpha_j, _ = gsplat.rasterization(
                        Ks=intr[s],
                        backgrounds=None,
                        colors=cl_i,
                        covars=cv_i,
                        far_plane=self.config.far,
                        height=H,
                        means=po_i,
                        near_plane=self.config.near,
                        opacities=op_i[s] if is_opacity_sh else op_i,
                        packed=True,
                        quats=None,
                        rasterize_mode=self.config.raster_mode,
                        render_mode=f"RGB+{self.config.depth_mode}",
                        sh_degree=sh_degree_color,
                        scales=None,
                        viewmats=extr[s],
                        width=W,
                    )
                    alpha.append(alpha_j)
                    frame.append(frame_j)
                # [(S, H, W, 3 + 1), (S, H, W, 1)]
                alpha = torch.cat(alpha, dim=0)
                frame = torch.cat(frame, dim=0)
            else:
                dummy = cl_i.sum() + cv_i.sum() + op_i.sum() + po_i.sum()
                alpha = dummy.expand(S, H, W, 1)
                frame = dummy.expand(S, H, W, 3 + 1)
            alphas.append(alpha)
            frames.append(frame)

        # Reshape frame
        # (B, S, H, W, 1 + 1 + 3) -> (B, S, 1 + 1 + 3, H, W)
        alpha = torch.stack(alphas, dim=0).movedim(-1, -3).contiguous()
        frame = torch.stack(frames, dim=0).movedim(-1, -3).contiguous()
        image, depth = frame.split([3, 1], dim=-3)

        # Alpha-blend the background color
        if self.bgcolor is not None:
            bgcolor = self.bgcolor[None, None, ..., None, None]
            image = image + alpha.new_ones(()).sub(alpha).mul(bgcolor)

        # Clamp frame
        depth = depth.clamp(eps, 1.0 / eps)
        image = image.clamp(0.0, 1.0)

        return Depth2ImageDecoderOut(depth=depth, image=image)

    def extra_repr(self) -> str:
        config = self.config
        return f"depth_mode={config.depth_mode}, raster_mode={config.raster_mode}"
