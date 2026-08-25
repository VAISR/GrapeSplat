"""Pipeline orchestrator for AnySplat."""

from ....data import transform as trf
from ... import typing as Tp
from ..pipeline import Pipeline, PipelineBaseConfig, PipelineBuilder
from anysplat.model.encoder.encoder import EncoderOutput
from anysplat.model.model import AnySplat
from dataclasses import dataclass
from torch import Tensor
import torch


@dataclass
class PipelineAnySplatBuilder(PipelineBuilder):
    """Pipeline orchestrator for AnySplat."""

    base: PipelineBaseConfig

    def __call__(self) -> "PipelineAnySplat":
        """Build the module."""

        return PipelineAnySplat(config=self)


class PipelineAnySplat(Pipeline):
    """Pipeline orchestrator for AnySplat."""

    def __init__(self, config: PipelineAnySplatBuilder) -> None:
        super().__init__()

        base = AnySplat.from_pretrained(config.base["ckpt"])

        self.config = config

        # Initialize sub-modules in execution order
        self.base = base
        self.depth2_image_decoder = config.depth2_image_decoder()
        self.scene_optimizer = config.scene_optimizer(base=self.depth2_image_decoder)

    def reconstruct(self, image: Tensor) -> Tp.PipelineOut:
        B, S, _, H, W = image.shape
        frame_resol = H, W

        out_enc: EncoderOutput = self.base.encoder(image)
        # (B, S, 3 + 4 + 2)
        cam_enc: Tensor = out_enc.pred_pose_enc_list[-1]
        # (B, S, 3, 4) in W2C
        cam_extr: Tensor = out_enc.pred_context_pose["extrinsic"]
        cam_extr = cam_extr.inverse()[..., 0:3, :].contiguous()
        # (B, S, 3, 3) in PC
        cam_intr: Tensor = out_enc.pred_context_pose["intrinsic"]
        cam_intr = cam_intr.mul(cam_intr.new_tensor(frame_resol[::-1] + (1.0,))[..., None])
        # (B, S, 1, H, W) in exp
        depth1: Tensor = out_enc.depth_dict["depth"]
        depth1 = depth1.movedim(-1, -3).contiguous()
        # (B, S, 1, H, W) in exp+1
        depth1_conf: Tensor = out_enc.depth_dict["depth_conf"]
        depth1_conf = depth1_conf.unsqueeze(-3)

        with torch.autocast(image.device.type, enabled=False):
            # [(B, S, 3, H, W) in world; 2]
            ray_d, ray_o = trf.get_ray_unprojected(
                frame_resol=frame_resol,
                extr=cam_extr,
                intr=cam_intr,
            )
            # (B, S, 3, H, W) in world
            point = trf.get_depth_unprojected(depth=depth1, ray_d=ray_d, ray_o=ray_o)

            # [V_i; B]
            count: list[int] = out_enc.infos["valid_count"]
            # (B, V_i, ...)
            gs = out_enc.gaussians
            batch: list[torch.Tensor] = []
            color: list[torch.Tensor] = []
            covariance: list[torch.Tensor] = []
            opacity: list[torch.Tensor] = []
            position: list[torch.Tensor] = []
            for i in range(B):
                V = count[i]
                # (V_i,)
                batch.append(gs.means.new_full((V,), i, dtype=torch.int))
                # (V_i, 3, K) -> (V_i, K, 3)
                color.append(gs.harmonics[i, 0:V].mT.contiguous())
                # (V_i, 3, 3)
                covariance.append(gs.covariances[i, 0:V])
                # (V_i, 1)
                opacity.append(gs.opacities[i, 0:V, None])
                # (V_i, 3)
                position.append(gs.means[i, 0:V])
            out_scene = Tp.SceneDecoderOut(
                batch_index=torch.cat(batch, dim=0),
                batch_size=B,
                color=torch.cat(color, dim=0),
                covariance=torch.cat(covariance, dim=0),
                opacity=torch.cat(opacity, dim=0),
                position=torch.cat(position, dim=0),
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
