"""Pipeline orchestrator for 2Xplat."""

from ....data import transform as trf
from ... import typing as Tp
from ..pipeline import Pipeline, PipelineBaseConfig, PipelineBuilder
from dataclasses import dataclass
from hydra import compose, initialize
from pathlib import Path
from torch import Tensor
from torch.nn import functional as nnf
from twoxplat.model.gaussians import GaussianField
from twoxplat.model.model import TwoExpertModel
from twoxplat.utils import camera_utils as cam_util
import torch


@dataclass
class PipelineTwoXplatBuilder(PipelineBuilder):
    """Pipeline orchestrator for 2Xplat."""

    base: PipelineBaseConfig
    frame_resol: int

    def __call__(self) -> "PipelineTwoXplat":
        """Build the module."""

        return PipelineTwoXplat(config=self)


class PipelineTwoXplat(Pipeline):
    """Pipeline orchestrator for 2Xplat."""

    def __init__(self, config: PipelineTwoXplatBuilder) -> None:
        super().__init__()

        ckpt = Path(config.base["ckpt"])
        with initialize(
            config_path="../../../../../gitmodules/twoxplat/twoxplat/configs", version_base="1.3"
        ):
            config_base = compose(
                config_name="inference",
                overrides=[
                    f"data.resize_h={config.frame_resol}",
                    f"data.resize_w={config.frame_resol}",
                    "model.da_model_weights_path=null",
                    "model.mvp_weights_path=null",
                ],
            )

        state: dict[str, Tensor] = torch.load(ckpt, mmap=True)["ema"]
        config_base.model.patch_size = int(
            (state["image_tokenizer.1.weight"].shape[-1] / config_base.model.in_channels) ** 0.5
        )

        base = TwoExpertModel(config_base)
        base.load_state_dict(
            {k: state[k] for k in state if not k.startswith("loss_computer.")},
            strict=True,
        )

        self.config = config

        # Initialize sub-modules in execution order
        self.base = base
        self.depth2_image_decoder = config.depth2_image_decoder()
        self.scene_optimizer = config.scene_optimizer(base=self.depth2_image_decoder)

    def reconstruct(self, image: Tensor) -> Tp.PipelineOut:
        B, S, _, H, W = image.shape
        frame_resol = H, W

        # NOTE: Only time travelers use target images as model input
        # [(B, S, 2 + 2), (B, S, 4, 4) in C2W]
        cam_intr_enc, cam_extr_c2w, _, _ = self.base.geometry_expert._run_da3_and_normalize(
            image, image[:, 0:0]
        )
        # (B, S, 3, 3) in PC
        cam_intr = cam_util.fxfycxcy_to_K(cam_intr_enc)
        # [(B, S, 3 + 3 + 3 + 3, H, W), (B, S, 4, 4) in W2C]
        raymap, _, cam_extr_w2c = self.base._build_raymap_input(
            dict(image=image), cam_intr_enc, cam_extr_c2w
        )
        out_gs: GaussianField = self.base.appearance_expert.predict_gaussians(
            raymap, cam_extr_w2c, cam_intr, cam_intr_enc, cam_extr_c2w, t_c2w=None
        )

        with torch.autocast(image.device.type, enabled=False):
            # (B, S, 3, 4) in W2C
            cam_extr = cam_extr_w2c[..., 0:3, :].contiguous()
            # (B, S, 3 + 4 + 2)
            cam_enc = trf.get_enc_from_cam(
                frame_resol=frame_resol,
                cam_extr=cam_extr,
                cam_intr=cam_intr,
            )

            # (B, N, 3) in world
            position = out_gs.xyz
            # S * H * W -> N
            N = position.shape[-2]
            # NOTE: Use position z in camera coordinate as depth
            # (B, S, 1, 3) @ (B, S, 3, H * W) + (B, S, 1) -> (B, S, H * W)
            depth1 = (
                cam_extr[..., 2:3, 0:3]
                .matmul(position.unflatten(1, (S, N // S)).mT.contiguous())
                .squeeze(-2)
                .add(cam_extr[..., 2:3, 3])
            )
            # (B, S, 1, H, W)
            depth1 = depth1.unflatten(-1, (1,) + frame_resol)
            # (B, S, 1, H, W)
            depth1_conf = torch.ones_like(depth1)

            # [(B, S, 3, H, W) in world; 2]
            ray_d, ray_o = trf.get_ray_unprojected(
                frame_resol=frame_resol,
                extr=cam_extr,
                intr=cam_intr,
            )
            # (B, S, 3, H, W) in world
            point = trf.get_depth_unprojected(depth=depth1, ray_d=ray_d, ray_o=ray_o)

            # (B * N,) -> (M,)
            batch_index = image.new_tensor(range(B), dtype=torch.int).repeat_interleave(N)
            # (B, N, K, 3) in SH
            color = out_gs.feature
            # (B, N, 4) in wxyz -> (B, N, 3, 3) * (B, N, 1, 3) -> (B, N, 3, 3)
            shape = trf.get_matr_from_quat(nnf.normalize(out_gs.rotation, dim=-1)).mul(
                out_gs.scale.exp()[..., None, :]
            )
            # (B, N, 3, 3) @ (B, N, 3, 3) -> (B, N, 3, 3) in world
            covariance = shape.matmul(shape.mT.contiguous())
            # (B, N, K_op, 1) in SH
            opacity = out_gs.opacity
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
