"""Pipeline module for GrapeSplat: Scene optimizer."""

from ..optimizer import CompositeOptimizerBuilder
from ..scheduler import CompositeSchedulerBuilder
from ..typing import (
    Depth2ImageDecoderInp,
    Depth2ImageDecoderOut,
    SceneDecoderOut,
    SceneOptimizerInp,
)
from .depth2_image_decoder import Depth2ImageDecoder
from dataclasses import dataclass
from torch import Tensor, nn
import torch


@dataclass
class SceneOptimizerBuilder:
    """Scene optimizer module for test-time optimization on reconstruction view."""

    enabled: bool
    loss: dict[str, float | str]
    optimizer: CompositeOptimizerBuilder
    scheduler: CompositeSchedulerBuilder
    step: int

    def __call__(self, base: Depth2ImageDecoder) -> "SceneOptimizer":
        """Build the module."""

        return SceneOptimizer(base=base, config=self)


class SceneOptimizer(nn.Module):
    """Scene optimizer module for test-time optimization on reconstruction view."""

    def __init__(self, base: Depth2ImageDecoder, config: SceneOptimizerBuilder) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        self.base = base

    def forward(self, inp: SceneOptimizerInp) -> SceneDecoderOut:
        out_scene = inp.inp_depth2_image.scene
        if not self.config.enabled or self.training:
            return out_scene

        device = inp.image.device
        eps = torch.finfo(inp.image.dtype).eps
        is_opacity_sh = out_scene.opacity.ndim == 3

        # NOTE: Tensor entering the graph must be cloned out of inference mode
        with (
            torch.autocast(device.type, enabled=False),
            torch.enable_grad(),
            torch.inference_mode(False),
        ):
            # (B, S_rec, 3, H, W) -> (B * S_rec, 3, H, W)
            data_image = inp.image.clone().flatten(0, -4)
            inp: Depth2ImageDecoderInp = inp.inp_depth2_image._replace(
                cam_extr=inp.inp_depth2_image.cam_extr.clone(),
                cam_intr=inp.inp_depth2_image.cam_intr.clone(),
            )

            # Parameterize scene primitive to optimize
            param = nn.Module()
            param.color = nn.Parameter(out_scene.color.clone())
            param.opacity = nn.Parameter(
                out_scene.opacity.clone()
                if is_opacity_sh
                else out_scene.opacity.clamp(eps, 1.0 - eps).logit()
            )
            param.position = nn.Parameter(out_scene.position.clone())
            param.shape = nn.Parameter(
                torch.linalg.cholesky(
                    out_scene.covariance.add(torch.eye(3, device=device).mul(eps))
                )
            )

            # Initialize loss, optimizer, and scheduler in execution order
            loss_fns = {"l1": nn.L1Loss, "l2": nn.MSELoss}
            loss_fn: nn.Module = loss_fns[self.config.loss["type"]](reduction="mean")
            optimizer = self.config.optimizer(param)
            scheduler = self.config.scheduler(iters=self.config.step, optimizer=optimizer)
            weight = float(self.config.loss["weight"])

            for _ in range(self.config.step):
                # Render scene primitive at camera
                out_image: Depth2ImageDecoderOut = self.base(
                    inp._replace(
                        scene=inp.scene._replace(
                            color=param.color,
                            covariance=param.shape.matmul(param.shape.mT.contiguous()),
                            opacity=param.opacity if is_opacity_sh else param.opacity.sigmoid(),
                            position=param.position,
                        )
                    )
                )
                # (B, S_rec, 3, H, W) -> (B * S_rec, 3, H, W)
                pred_image = out_image.image.flatten(0, -4)

                # Image loss
                loss: Tensor = loss_fn(pred_image, data_image)
                if loss.isnan().item():
                    loss = pred_image.nan_to_num().mul(0.0).sum()
                loss = loss * weight

                # Update scene primitive
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Detach scene primitive from graph
            color = param.color.detach()
            opacity = param.opacity.detach()
            position = param.position.detach()
            shape = param.shape.detach()

            return out_scene._replace(
                color=color,
                covariance=shape.matmul(shape.mT.contiguous()),
                opacity=opacity if is_opacity_sh else opacity.sigmoid(),
                position=position,
            )

    def extra_repr(self) -> str:
        config = self.config
        return f"enabled={config.enabled}, step={config.step}"
