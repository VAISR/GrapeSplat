"""Pipeline orchestrator for GrapeSplat."""

from ...data import transform as trf
from .. import typing as Tp
from .cam_ray_decoder import CamRayDecoderBuilder
from .depth1_decoder import Depth1DecoderBuilder
from .depth2_image_decoder import Depth2ImageDecoderBuilder
from .image_encoder import ImageEncoderBuilder
from .scene_decoder import SceneDecoderBuilder, TokenSemProject
from .scene_optimizer import SceneOptimizerBuilder
from abc import ABC, abstractmethod
from anysplat.model.model import AnySplat
from dataclasses import dataclass
from dvlt.model.dvlt.model import DVLTModel
from dvlt.model.dvlt.shim import DVLT
from hydra import compose, initialize
from hydra.utils import call
from pathlib import Path
from torch import Tensor, nn
from torch.nn import functional as nnf
from torchvision.transforms.v2 import Normalize
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from typing import TypedDict
from vggt.models.vggt import VGGT
import copy
import huggingface_hub as hf_hub
import torch


@dataclass
class PipelineBuilder(ABC):
    """Pipeline orchestrator."""

    depth2_image_decoder: Depth2ImageDecoderBuilder
    scene_optimizer: SceneOptimizerBuilder
    testing: bool

    @abstractmethod
    def __call__(self) -> "Pipeline":
        """Build the module."""


class Pipeline(ABC, nn.Module):
    """Pipeline orchestrator."""

    @abstractmethod
    def reconstruct(self, image: Tensor) -> Tp.PipelineOut:
        """
        Reconstruct static scene from multi-view images.

        Shape:
            (B, S_rec, 3, H, W)
        """
        # NOTE: Uncalibrated method cannot use camera intrinsics as model input

    def render(self, inp: Tp.PipelineInp, out_rec: Tp.PipelineOut) -> Tp.PipelineOut:
        """
        Render static scene from multi-view cameras.

        Shape:
            (B, S_ren, C, H, W)
        """

        out_ren: Tp.Depth2ImageDecoderOut = self.depth2_image_decoder(
            Tp.Depth2ImageDecoderInp(
                # NOTE: Only cheaters use pose-TTO in NVS evaluation
                cam_extr=trf.align_cam_extr(
                    extr_src=inp.cam_extr,
                    extr_tar=out_rec.cam_extr,
                    mask_src=inp.mask_ren,
                    mask_tar=inp.mask_rec,
                ),
                cam_intr=inp.cam_intr[:, inp.mask_ren],
                frame_resol=out_rec.frame_resol,
                scene=out_rec.scene,
            )
        )
        return out_rec._replace(
            depth2_ren=out_ren.depth,
            image_ren=out_ren.image,
        )


class PipelineBaseConfig(TypedDict):
    ckpt: str
    type: str


@dataclass
class PipelineGrapeSplatBuilder(PipelineBuilder):
    """Pipeline orchestrator for GrapeSplat."""

    base_geo: PipelineBaseConfig
    base_sem: PipelineBaseConfig
    cam_ray_decoder: CamRayDecoderBuilder
    depth1_decoder: Depth1DecoderBuilder
    image_encoder: ImageEncoderBuilder
    scene_decoder: SceneDecoderBuilder

    def __call__(self) -> "PipelineGrapeSplat":
        """Build the module."""

        return PipelineGrapeSplat(config=self)


class PipelineGrapeSplat(Pipeline):
    """Pipeline orchestrator for GrapeSplat."""

    def __init__(self, config: PipelineGrapeSplatBuilder) -> None:
        super().__init__()

        # Extract sub-module bases from pipeline geo base
        if config.testing:
            config.base_geo["ckpt"] = None
        if config.base_geo["type"] in ("anysplat", "dvlt", "vggt"):
            if config.base_geo["type"] == "anysplat":
                base_geo = VGGT()
                if config.base_geo["ckpt"] is not None:
                    base_outer = AnySplat.from_pretrained(config.base_geo["ckpt"]).encoder
                    del base_outer.gaussian_param_head
                    base_geo.load_state_dict(base_outer.state_dict())
            elif config.base_geo["type"] == "dvlt":
                base_geo_inner = (
                    DVLTModel(drop_path=0.0)
                    if config.base_geo["ckpt"] is None
                    else DVLTModel.from_pretrained(config.base_geo["ckpt"], drop_path=0.0)
                )
                base_geo = DVLT(base_geo_inner)
            elif config.base_geo["type"] == "vggt":
                base_geo = (
                    VGGT()
                    if config.base_geo["ckpt"] is None
                    else VGGT.from_pretrained(config.base_geo["ckpt"])
                )

            base_geo_cam = base_geo.camera_head
            base_geo_depth = base_geo.depth_head
            base_geo_image = base_geo.aggregator
            base_geo_feat = copy.deepcopy(base_geo.depth_head)
            base_geo_feat.feature_only = True
            if config.base_geo["type"] in ("anysplat", "vggt"):
                feat_dim_o = base_geo_feat.scratch.output_conv1.out_channels
                del base_geo_feat.scratch.output_conv2
            elif config.base_geo["type"] == "dvlt":
                feat_dim_o = base_geo_feat.output_block[0].in_channels
                del base_geo_feat.output_block
        else:
            raise ValueError(f"Unsupported pipeline geo base type: {config.base_geo['type']}")

        # Extract sub-module bases from pipeline sem base
        if config.base_sem["type"] in ("dinov3",):
            if config.base_sem["type"] == "dinov3":
                base_sem_image = (
                    DINOv3ViTModelShim(
                        (
                            DINOv3ViTConfig()
                            if config.base_sem["ckpt"] is None
                            else DINOv3ViTConfig.from_pretrained(config.base_sem["ckpt"])
                        )
                    )
                    if config.base_sem["ckpt"] is None or config.testing
                    else DINOv3ViTModelShim.from_pretrained(config.base_sem["ckpt"])
                )
                feat_dim_s = base_sem_image.config.hidden_size

            base_sem_feat = TokenSemProject(
                dim_i=feat_dim_s,
                dim_o=feat_dim_o,
                patch=base_sem_image.config.patch_size,
            )
        else:
            raise ValueError(f"Unsupported pipeline sem base type: {config.base_sem['type']}")

        self.config = config

        # Initialize sub-modules in execution order
        self.image_encoder = config.image_encoder(
            base_geo=base_geo_image,
            base_sem=base_sem_image,
        )
        self.depth1_decoder = config.depth1_decoder(
            base=base_geo_depth,
        )
        self.cam_ray_decoder = config.cam_ray_decoder(
            base=base_geo_cam,
        )
        self.scene_decoder = config.scene_decoder(
            base_geo=base_geo_feat,
            base_sem=base_sem_feat,
            dim_i=feat_dim_o,
        )
        self.depth2_image_decoder = config.depth2_image_decoder()
        self.scene_optimizer = config.scene_optimizer(
            base=self.depth2_image_decoder,
        )

    def reconstruct(self, image: Tensor) -> Tp.PipelineOut:
        out_image: Tp.ImageEncoderOut = self.image_encoder(
            Tp.ImageEncoderInp(
                image=image,
            )
        )
        out_depth1: Tp.Depth1DecoderOut = self.depth1_decoder(
            Tp.Depth1DecoderInp(
                out_image=out_image,
            )
        )
        out_cam_ray: Tp.CamRayDecoderOut = self.cam_ray_decoder(
            Tp.CamRayDecoderInp(
                out_image=out_image,
            )
        )
        point = trf.get_depth_unprojected(
            depth=out_depth1.depth,
            ray_d=out_cam_ray.ray_d,
            ray_o=out_cam_ray.ray_o,
        )
        out_scene: Tp.SceneDecoderOut = self.scene_decoder(
            Tp.SceneDecoderInp(
                image=image,
                out_depth1=out_depth1,
                out_image=out_image,
                out_ray=out_cam_ray,
                point=point.detach(),
            )
        )
        inp_depth2_image = Tp.Depth2ImageDecoderInp(
            cam_extr=out_cam_ray.cam_extr.detach(),
            cam_intr=out_cam_ray.cam_intr.detach(),
            frame_resol=out_image.frame_resol,
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
            batch_size=out_scene.batch_size,
            cam_enc=out_cam_ray.cam_enc,
            cam_extr=out_cam_ray.cam_extr,
            cam_intr=out_cam_ray.cam_intr,
            depth1=out_depth1.depth,
            depth1_conf=out_depth1.depth_conf,
            depth2_rec=out_depth2_image_rec.depth,
            depth2_ren=image.new_empty((0,) * 5),
            frame_resol=out_image.frame_resol,
            image_rec=out_depth2_image_rec.image,
            image_ren=image.new_empty((0,) * 5),
            point=point,
            ray_d=out_cam_ray.ray_d,
            ray_o=out_cam_ray.ray_o,
            scene=out_scene,
        )

    @classmethod
    def from_pretrained(
        cls,
        dir_or_repo: str,
        config_name: str = "default",
        overrides: list[str] = [],
    ) -> "PipelineGrapeSplat":
        """Initialize from checkpoint directory or Hugging Face repository."""

        ckpt_path = Path(dir_or_repo) / f"{config_name}.ckpt"
        if not ckpt_path.exists():
            ckpt_path = (
                Path(
                    hf_hub.snapshot_download(
                        dir_or_repo,
                        allow_patterns=[ckpt_path.name],
                    )
                )
                / ckpt_path.name
            )

        with initialize(
            config_path="../../../../config/grapesplat/module/pipeline",
            version_base="1.3",
        ):
            config = compose(config_name=config_name, overrides=overrides)

        state: dict[str] = torch.load(ckpt_path)["state_dict"]
        module: nn.Module = call(config, _convert_="all")()
        module.load_state_dict(
            {k.removeprefix("pipeline."): v for k, v in state.items() if k.startswith("pipeline.")},
            strict=True,
        )

        return module


class DINOv3ViTModelShim(DINOv3ViTModel):
    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__(config)

        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: Tensor, *_, **__) -> tuple[list[Tensor], int]:
        P = self.config.patch_size
        # (B, S, 3, H, W) -> (B * S, 3, H_P * P, W_P * P) -> (B, S, C, H_P * W_P)
        y = nnf.interpolate(
            self.normalize(x.flatten(0, -4)),
            tuple(v // P * P for v in x.shape[-2:]),
            align_corners=False,
            antialias=True,
            mode="bilinear",
        )
        y = super().forward(y).last_hidden_state.unflatten(0, x.shape[0:-3])
        return [y], 1 + self.config.num_register_tokens
