"""Pipeline orchestrator for StructSplat."""

from ....data import transform as trf
from ... import typing as Tp
from ..pipeline import Pipeline, PipelineBaseConfig, PipelineBuilder
from dataclasses import dataclass
from structsplat.model.gaussian_wrapper import GaussianWrapper
from torch import Tensor
import torch


@dataclass
class PipelineStructSplatBuilder(PipelineBuilder):
    """Pipeline orchestrator for StructSplat."""

    base: PipelineBaseConfig

    def __call__(self) -> "PipelineStructSplat":
        """Build the module."""

        return PipelineStructSplat(config=self)


class PipelineStructSplat(Pipeline):
    """Pipeline orchestrator for StructSplat."""

    def __init__(self, config: PipelineStructSplatBuilder) -> None:
        super().__init__()

        base = GaussianWrapper.from_pretrained(config.base["ckpt"])

        self.config = config

        # Initialize sub-modules in execution order
        self.base = base
        self.depth2_image_decoder = config.depth2_image_decoder()
        self.scene_optimizer = config.scene_optimizer(base=self.depth2_image_decoder)

    def reconstruct(self, image: Tensor) -> Tp.PipelineOut:
        B, S, _, H, W = image.shape
        frame_resol = H, W

        out_enc: dict[str, Tensor] = self.base(image)
        # (B, S, 3 + 4 + 2)
        cam_enc: Tensor = out_enc["camera"]
        # (B, S, 4, 4) -> (B, S, 3, 4) in W2C
        cam_extr: Tensor = out_enc["camera_full"]["extrinsic"][..., 0:3, :].contiguous()
        # (B, S, 3, 3) in PC
        cam_intr: Tensor = out_enc["camera_full"]["intrinsic"]
        # (B, S, 1, H, W) in [0, inf)
        depth1: Tensor = out_enc["depth"].unsqueeze(-3)
        # (B, S, 1, H, W)
        depth1_conf = torch.ones_like(depth1)

        with torch.autocast(image.device.type, enabled=False):
            # [(B, S, 3, H, W) in world; 2]
            ray_d, ray_o = trf.get_ray_unprojected(
                frame_resol=frame_resol,
                extr=cam_extr,
                intr=cam_intr,
            )
            # (B, S, 3, H, W) in world
            point = trf.get_depth_unprojected(depth=depth1, ray_d=ray_d, ray_o=ray_o)

            # (B, N, 3)
            position = out_enc["coordinate"]
            # S * G * H * W -> N
            N = position.shape[1]
            # (B * N,) -> (M,)
            batch_index = image.new_tensor(range(B), dtype=torch.int).repeat_interleave(N)
            # (B, N, 3) in RGB -> (B, N, 1, 3) in SH
            color = out_enc["color"].sub(0.5).mul(2.0 * torch.pi**0.5)[..., None, :]
            # (B, N, 4) in wxyz -> (B, N, 3, 3) * (B, N, 1, 3) -> (B, N, 3, 3)
            shape = trf.get_matr_from_quat(out_enc["rotation"]).mul(out_enc["scale"][..., None, :])
            # (B, N, 3, 3) @ (B, N, 3, 3) -> (B, N, 3, 3) in world
            covariance = shape.matmul(shape.mT.contiguous())
            # (B, N, 1)
            opacity = out_enc["opacity"][..., None]
            # (B, N, ...) -> (M, ...)
            out_scene = Tp.SceneDecoderOut(
                batch_index=batch_index,
                batch_size=B,
                color=color.flatten(0, 1),
                covariance=covariance.flatten(0, 1),
                opacity=opacity.flatten(0, 1),
                position=position.flatten(0, 1),
            )
            inp_depth2_image = Tp.Depth2ImageDecoderInp(
                cam_extr=cam_extr.detach(),
                cam_intr=cam_intr.detach(),
                frame_resol=frame_resol,
                scene=out_scene,
            )
            out_scene = self.scene_optimizer(
                Tp.SceneOptimizerInp(
                    image=image,
                    inp_depth2_image=inp_depth2_image,
                )
            )
            # Render rec view unless testing
            out_depth2_image_rec: Tp.Depth2ImageDecoderOut = (
                Tp.Depth2ImageDecoderOut(
                    depth=image.new_empty((0,) * 5),
                    image=image.new_empty((0,) * 5),
                )
                if self.config.testing
                else self.depth2_image_decoder(inp_depth2_image._replace(scene=out_scene))
            )

        return Tp.PipelineOut(
            batch_size=B,
            cam_enc=cam_enc,
            cam_extr=cam_extr,
            cam_intr=cam_intr,
            depth1=depth1,
            depth1_conf=depth1_conf,
            depth2_rec=out_depth2_image_rec.depth,
            depth2_ren=image.new_empty((0,) * 5),
            frame_resol=frame_resol,
            image_rec=out_depth2_image_rec.image,
            image_ren=image.new_empty((0,) * 5),
            point=point,
            ray_d=ray_d,
            ray_o=ray_o,
            scene=out_scene,
        )
