"""Pipeline orchestrator for DA3."""

from ....data import transform as trf
from ... import typing as Tp
from ..pipeline import Pipeline, PipelineBaseConfig, PipelineBuilder
from depth_anything_3.api import DepthAnything3, InputProcessor
from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.model.gs_adapter import Gaussians
from dataclasses import dataclass
from torch import Tensor
import torch


@dataclass
class PipelineDA3Builder(PipelineBuilder):
    """Pipeline orchestrator for DA3."""

    base: PipelineBaseConfig

    def __call__(self) -> "PipelineDA3":
        """Build the module."""

        return PipelineDA3(config=self)


class PipelineDA3(Pipeline):
    """Pipeline orchestrator for DA3."""

    def __init__(self, config: PipelineDA3Builder) -> None:
        super().__init__()

        base: DepthAnything3Net = DepthAnything3.from_pretrained(config.base["ckpt"]).model
        del base.cam_enc

        self.config = config

        # Initialize sub-modules in execution order
        self.base = base
        self.depth2_image_decoder = config.depth2_image_decoder()
        self.scene_optimizer = config.scene_optimizer(base=self.depth2_image_decoder)

    def reconstruct(self, image: Tensor) -> Tp.PipelineOut:
        B, S, _, H, W = image.shape
        frame_resol = H, W

        out_enc: dict[str, Tensor] = self.base(
            InputProcessor.NORMALIZE(image),
            infer_gs=True,
            use_ray_pose=True,
        )
        # (B, S, 1, H, W) in exp
        depth1 = out_enc["depth"].unsqueeze(-3)
        # (B, S, 1, H, W) in exp+1
        depth1_conf = out_enc["depth_conf"].unsqueeze(-3)
        # (B, S, 3, 4) in W2C
        cam_extr = out_enc["extrinsics"]
        # (B, S, 3, 3) in PC
        cam_intr = out_enc["intrinsics"]

        with torch.autocast(image.device.type, enabled=False):
            # (B, S, 3 + 4 + 2)
            cam_enc = trf.get_enc_from_cam(
                frame_resol=frame_resol,
                cam_extr=cam_extr,
                cam_intr=cam_intr,
            )
            # [(B, S, 3, H, W) in world; 2]
            ray_d, ray_o = trf.get_ray_unprojected(
                frame_resol=frame_resol,
                extr=cam_extr,
                intr=cam_intr,
            )
            # (B, S, 3, H, W) in world
            point = trf.get_depth_unprojected(depth=depth1, ray_d=ray_d, ray_o=ray_o)

            # (B, N, ...)
            gs: Gaussians = out_enc["gaussians"]
            # S * H * W -> N
            N = gs.means.shape[1]
            # (B * N,) -> (M,)
            batch_index = image.new_tensor(range(B), dtype=torch.int).repeat_interleave(N)
            # (B, N, 4) in wxyz -> (B, N, 3, 3) * (B, N, 1, 3) -> (B, N, 3, 3)
            shape = trf.get_matr_from_quat(gs.rotations).mul(gs.scales[..., None, :])
            # (B, N, 3, 3) @ (B, N, 3, 3) -> (B, N, 3, 3) in world
            covariance = shape.matmul(shape.mT.contiguous())
            # (B, N, 3, K) -> (B, N, K, 3)
            color = gs.harmonics.mT.contiguous()
            # (B, N, 1)
            opacity = gs.opacities[..., None]
            # (B, N, 3)
            position = gs.means
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
