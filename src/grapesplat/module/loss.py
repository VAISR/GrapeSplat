"""Composite loss module for GrapeSplat."""

from ..data import transform as trf
from .metric import LPIPS
from .typing import CompositeLossOut, PipelineData, PipelineOut
from dataclasses import dataclass
from torch import Tensor, autograd as ad, nn
from torch.nn import functional as nnf
from typing import Literal, Self
import torch

__all__ = ["LPIPS"]


@dataclass
class CompositeLossBuilder:
    """Composite loss for pipeline objectives."""

    cam: dict[str, float | str]
    depth1: dict[str, float | str, float | str]
    depth2: dict[str, float | str]
    image: dict[str, dict[str, float | str]]
    point: dict[str, float | str]

    def __call__(self) -> "CompositeLoss":
        """Build the loss."""

        return CompositeLoss(config=self)


class CompositeLoss(nn.Module):
    """Composite loss for pipeline objectives."""

    def __init__(self, config: CompositeLossBuilder) -> None:
        super().__init__()

        fns = {
            "l1": nn.L1Loss,
            "l1_grad": L1GradLoss,
            "l1_grad_nll": L1GradNllLoss,
            "l2": nn.MSELoss,
            "lpips": LPIPS(reduction="none"),
        }

        self.config = config

        # Initialize sub-modules in execution order
        self.cam: nn.Module = fns[config.cam["type"]](reduction="none")
        self.depth1: nn.Module = fns[config.depth1["type"]](reduction="none")
        self.depth2: nn.Module = fns[config.depth2["type"]](reduction="none")
        self.image_perc: nn.Module = fns[config.image["perc"]["type"]]
        self.image_regr: nn.Module = fns[config.image["regr"]["type"]](reduction="none")
        self.point: nn.Module = fns[config.point["type"]](reduction="none")

        self.train(False)

    def forward(self, pred: PipelineOut, data: PipelineData) -> CompositeLossOut:
        with torch.autocast(data.depth.device.type, enabled=False):
            return self.forward_unwrapped(pred, data)

    def forward_unwrapped(self, pred: PipelineOut, data: PipelineData) -> CompositeLossOut:
        # Skip if data is invalid
        is_invalid = data.mask_depth.sum().item() < 1000 or data.mask_image.sum().item() < 1000

        # Camera encoding
        # (B * S_rec, 3 + 4 + 2)
        data_cam_enc = trf.get_enc_from_cam(
            frame_resol=pred.frame_resol,
            cam_extr=data.cam_extr,
            cam_intr=data.cam_intr,
        )[:, data.mask_rec].flatten(0, 1)
        pred_cam_enc = pred.cam_enc.flatten(0, 1)
        # Align camera encoding quaternion sign
        data_cam_enc[..., 3:7] = data_cam_enc[..., 3:7].where(
            data_cam_enc[..., 3:7].mul(pred_cam_enc[..., 3:7]).sum(dim=-1, keepdim=True).ge(0.0),
            -data_cam_enc[..., 3:7],
        )
        # Depth 1 and confidence
        # (B * S_rec, 1, H, W)
        data_mask_depth1 = data.mask_depth[:, data.mask_rec].flatten(0, 1)
        data_depth1 = data.depth[:, data.mask_rec].flatten(0, 1)
        pred_depth1 = pred.depth1.flatten(0, 1)
        pred_depth1_conf = pred.depth1_conf.flatten(0, 1)
        # Depth 2
        # (B * S_rec, 1, H, W)
        data_mask_depth2 = data.mask_depth[:, data.mask_rec].flatten(0, 1)
        data_depth2 = data.depth[:, data.mask_rec].flatten(0, 1)
        pred_depth2 = pred.depth2_rec.flatten(0, 1)
        # Image
        # (B * S_rec, 3, H, W)
        data_mask_image = data.mask_image[:, data.mask_rec].flatten(0, 1)
        data_image = data.image[:, data.mask_rec].flatten(0, 1)
        pred_image = pred.image_rec.flatten(0, 1)
        # Point and confidence
        # (B * S_rec, 3, H, W)
        data_mask_point = data.mask_depth.expand_as(data.point)[:, data.mask_rec].flatten(0, 1)
        data_point = data.point[:, data.mask_rec].flatten(0, 1)
        pred_point = pred.point.flatten(0, 1)
        pred_point_conf = pred.depth1_conf.flatten(0, 1)

        # Loss weights
        weight_cam = float(self.config.cam["weight"])
        weight_depth1 = float(self.config.depth1["weight"])
        weight_depth2 = float(self.config.depth2["weight"])
        weight_image_perc = float(self.config.image["perc"]["weight"])
        weight_image_regr = float(self.config.image["regr"]["weight"])
        weight_point = float(self.config.point["weight"])

        # Inherited reconstruction losses
        loss = CompositeLossOut()

        # Camera loss
        if is_invalid or weight_cam == 0.0:
            loss_cam = pred_cam_enc.nan_to_num().mul(0.0).sum()
        else:
            loss_cam: Tensor = self.cam(pred_cam_enc, data_cam_enc)
            loss_cam = loss_cam.mean()
            if loss_cam.isnan().item():
                loss_cam = pred_cam_enc.nan_to_num().mul(0.0).sum()
        loss["loss_cam"] = loss_cam
        loss_cam = loss_cam * weight_cam

        # Depth 1 loss
        if is_invalid or weight_depth1 == 0.0:
            loss_depth1 = pred_depth1.nan_to_num().mul(0.0).sum()
        else:
            loss_depth1: Tensor = self.depth1(pred_depth1, data_depth1, pred_depth1_conf)
            loss_depth1 = loss_depth1[data_mask_depth1]
            loss_depth1 = loss_depth1.clamp(max=100.0)
            loss_depth1 = loss_depth1.mean()
            if loss_depth1.isnan().item():
                loss_depth1 = pred_depth1.nan_to_num().mul(0.0).sum()
        loss["loss_depth1"] = loss_depth1
        loss_depth1 = loss_depth1 * weight_depth1

        # Depth 2 (rec) loss
        if is_invalid or weight_depth2 == 0.0:
            loss_depth2 = pred_depth2.nan_to_num().mul(0.0).sum()
        else:
            loss_depth2: Tensor = self.depth2(pred_depth2, data_depth2)
            loss_depth2 = loss_depth2[data_mask_depth2]
            loss_depth2 = loss_depth2.clamp(max=100.0)
            loss_depth2 = loss_depth2.mean()
            if loss_depth2.isnan().item():
                loss_depth2 = pred_depth2.nan_to_num().mul(0.0).sum()
        loss["loss_depth2"] = loss_depth2
        loss_depth2 = loss_depth2 * weight_depth2

        # Image (rec) perceptual loss
        if is_invalid or weight_image_perc == 0.0:
            loss_image_perc = pred_image.nan_to_num().mul(0.0).sum()
        else:
            loss_image_perc: Tensor = self.image_perc(pred_image, data_image)
            loss_image_perc = loss_image_perc[data_mask_image]
            loss_image_perc = loss_image_perc.mean()
            if loss_image_perc.isnan().item():
                loss_image_perc = pred_image.nan_to_num().mul(0.0).sum()
        loss["loss_image_perc"] = loss_image_perc
        loss_image_perc = loss_image_perc * weight_image_perc

        # Image (rec) regression loss
        if is_invalid or weight_image_regr == 0.0:
            loss_image_regr = pred_image.nan_to_num().mul(0.0).sum()
        else:
            loss_image_regr: Tensor = self.image_regr(pred_image, data_image)
            loss_image_regr = loss_image_regr[data_mask_image]
            loss_image_regr = loss_image_regr.mean()
            if loss_image_regr.isnan().item():
                loss_image_regr = pred_image.nan_to_num().mul(0.0).sum()
        loss["loss_image_regr"] = loss_image_regr
        loss_image_regr = loss_image_regr * weight_image_regr

        # Point loss
        if is_invalid or weight_point == 0.0:
            loss_point = pred_point.nan_to_num().mul(0.0).sum()
        else:
            loss_point: Tensor = self.point(pred_point, data_point, pred_point_conf)
            loss_point = loss_point[data_mask_point]
            loss_point = loss_point.clamp(max=100.0)
            loss_point = loss_point.mean()
            if loss_point.isnan().item():
                loss_point = pred_point.nan_to_num().mul(0.0).sum()
        loss["loss_point"] = loss_point
        loss_point = loss_point * weight_point

        # Aggregate weighted loss
        loss_agg = (
            loss_cam + loss_depth1 + loss_depth2 + loss_image_perc + loss_image_regr + loss_point
        )
        # Log weighted loss
        loss["loss"] = loss_agg

        return loss

        # TODO: Debug gradient statistics
        def fn_get_grad_stat(
            loss_agg: Tensor,
            loss: Tensor,
            pred: list[Tensor] | Tensor,
        ) -> tuple[Tensor, Tensor]:
            eps = torch.finfo(loss_agg.dtype).eps
            grad_loss = torch.stack(ad.grad(loss, pred, retain_graph=True))
            grad_loss_agg = torch.stack(ad.grad(loss_agg, pred, retain_graph=True))
            grad_mag_loss = grad_loss.norm().clamp(min=eps)
            grad_mag_loss_agg = grad_loss_agg.norm().clamp(min=eps)
            grad_deg_loss = (
                (grad_loss_agg.mul(grad_loss).sum().div(grad_mag_loss_agg).div(grad_mag_loss))
                .clamp(-1.0, 1.0)
                .arccos()
                .mul(180.0 / torch.pi)
            )
            return grad_deg_loss, grad_mag_loss

        if loss_cam.requires_grad:
            loss["grad_deg_loss_cam"], loss["grad_mag_loss_cam"] = fn_get_grad_stat(
                loss_agg, loss_cam, pred_cam_enc
            )
        if loss_depth1.requires_grad:
            loss["grad_deg_loss_depth1"], loss["grad_mag_loss_depth1"] = fn_get_grad_stat(
                loss_agg, loss_depth1, pred_depth1
            )
        if loss_depth2.requires_grad:
            loss["grad_deg_loss_depth2"], loss["grad_mag_loss_depth2"] = fn_get_grad_stat(
                loss_agg, loss_depth2, pred_depth2
            )
        if loss_image_perc.requires_grad:
            loss["grad_deg_loss_image_perc"], loss["grad_mag_loss_image_perc"] = fn_get_grad_stat(
                loss_agg, loss_image_perc, pred_image
            )
        if loss_image_regr.requires_grad:
            loss["grad_deg_loss_image_regr"], loss["grad_mag_loss_image_regr"] = fn_get_grad_stat(
                loss_agg, loss_image_regr, pred_image
            )
        if loss_point.requires_grad:
            loss["grad_deg_loss_point"], loss["grad_mag_loss_point"] = fn_get_grad_stat(
                loss_agg, loss_point, pred_point
            )

        return loss

    def train(self, mode: bool = True) -> Self:
        mode = False
        self.requires_grad_(mode)
        return super().train(mode)


class L1GradLoss(nn.Module):
    """
    L1-norm and gradient loss.

    Formula:
        l = |p - d| + |∇(p - d)|

    Shape:
        [(B, C, H, W), (B, C, H, W), (B, 1, H, W)] -> () || (B, C, H, W)
    """

    def __init__(self, reduction: Literal["mean", "none", "sum"]) -> None:
        super().__init__()

        self.reduction = reduction

    def grad(self, x: Tensor) -> Tensor:
        """
        Gradient magnitude with finite difference.

        Shape:
            (B, C, H, W) -> (B, C, 2, H, W)
        """

        return torch.stack(
            [
                nnf.avg_pool2d(
                    nnf.pad(x, (1, 1, 0, 0), mode="replicate").diff(dim=-1),
                    kernel_size=(1, 2),
                    stride=1,
                ),
                nnf.avg_pool2d(
                    nnf.pad(x, (0, 0, 1, 1), mode="replicate").diff(dim=-2),
                    kernel_size=(2, 1),
                    stride=1,
                ),
            ],
            dim=-3,
        )

    def forward(self, pred: Tensor, data: Tensor) -> Tensor:
        diff = pred.sub(data)
        # (B, C, H, W)
        regr = diff.abs()
        # (B, C, 2, H, W) -> (B, C, H, W)
        grad = self.grad(diff).abs().sum(dim=-3)
        value = regr.add(grad)
        if self.reduction == "mean":
            return value.mean()
        elif self.reduction == "sum":
            return value.sum()
        else:
            return value


class L1GradNllLoss(nn.Module):
    """
    L1-norm, gradient, and negative log-likelihood loss.

    Formula:
        l = c * (|p - d| + |∇(p - d)|) - m * log(c)

    Shape:
        [(B, C, H, W), (B, C, H, W), (B, 1, H, W) in (0, inf)] -> () || (B, C, H, W)
    """

    def __init__(self, reduction: Literal["mean", "none", "sum"], mult: float = 0.2) -> None:
        super().__init__()

        self.base = L1GradLoss(reduction="none")
        self.mult = mult
        self.reduction = reduction

    def forward(self, pred: Tensor, data: Tensor, conf: Tensor) -> Tensor:
        value: Tensor = self.base(pred, data)
        value = value.mul(conf).sub(conf.log().mul(self.mult))
        if self.reduction == "mean":
            return value.mean()
        elif self.reduction == "sum":
            return value.sum()
        else:
            return value
