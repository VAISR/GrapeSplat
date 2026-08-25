"""Metric modules for GrapeSplat."""

from ..data import transform as trf
from .typing import CompositeMetricOut, PipelineData, PipelineOut
from dataclasses import dataclass
from fused_ssim import FusedSSIMMap
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import (
    estimate_pointcloud_normals as estimate_normals,
    corresponding_points_alignment as umeyama,
)
from torch import Tensor, nn
from typing import Literal, Self, TypedDict
import lpips
import torch


@dataclass
class CompositeMetricBuilder:
    """Composite metric for evaluation."""

    depth1_absrel: "MetricConfig"
    depth1_delta1: "MetricConfig"
    depth2_rec_absrel: "MetricConfig"
    depth2_rec_delta1: "MetricConfig"
    depth2_ren_absrel: "MetricConfig"
    depth2_ren_delta1: "MetricConfig"
    image_rec_lpips: "MetricConfig"
    image_rec_psnr: "MetricConfig"
    image_rec_ssim: "MetricConfig"
    image_ren_lpips: "MetricConfig"
    image_ren_psnr: "MetricConfig"
    image_ren_ssim: "MetricConfig"
    point_cd_nc: "MetricConfig"
    pose_rel_auc3: "MetricConfig"
    pose_rel_auc30: "MetricConfig"

    def __call__(self) -> "CompositeMetric":
        """Build the metric."""

        return CompositeMetric(config=self)


class CompositeMetric(nn.Module):
    """Composite metric for evaluation."""

    def __init__(self, config: CompositeMetricBuilder) -> None:
        super().__init__()

        lpips = LPIPS(reduction="none")

        self.config = config

        # Initialize sub-modules in execution order
        self.depth1_absrel = AbsRel(reduction="none")
        self.depth1_delta1 = Delta1(reduction="none")
        self.depth2_rec_absrel = AbsRel(reduction="none")
        self.depth2_rec_delta1 = Delta1(reduction="none")
        self.depth2_ren_absrel = AbsRel(reduction="none")
        self.depth2_ren_delta1 = Delta1(reduction="none")
        self.image_rec_lpips = lpips
        self.image_rec_psnr = PSNR()
        self.image_rec_ssim = SSIM(reduction="none")
        self.image_ren_lpips = lpips
        self.image_ren_psnr = PSNR()
        self.image_ren_ssim = SSIM(reduction="none")
        self.point_cd_nc = CDNC(count_max=100000)
        self.pose_rel_auc3 = PoseRelAUC(angle_max_deg=3)
        self.pose_rel_auc30 = PoseRelAUC(angle_max_deg=30)

        self.train(False)

    def forward(self, pred: PipelineOut, data: PipelineData) -> CompositeMetricOut:
        with torch.autocast(data.depth.device.type, enabled=False):
            return self.forward_unwrapped(pred, data)

    def forward_unwrapped(self, pred: PipelineOut, data: PipelineData) -> CompositeMetricOut:
        out = CompositeMetricOut()

        # Skip if data is invalid
        is_invalid = data.mask_depth.sum().item() < 1000 or data.mask_image.sum().item() < 1000
        if is_invalid:
            return out

        # Camera extrinsic
        # (B, S_rec, 3, 4)
        data_cam_extr = data.cam_extr[:, data.mask_rec]
        pred_cam_extr = pred.cam_extr
        # Depth 1
        # (B, S_rec, 1, H, W)
        data_mask_depth1 = data.mask_depth[:, data.mask_rec]
        data_depth1 = data.depth[:, data.mask_rec]
        pred_depth1 = pred.depth1
        # Depth 2 (rec)
        # (B, S_rec, 1, H, W)
        data_mask_depth2_rec = data.mask_depth[:, data.mask_rec]
        data_depth2_rec = data.depth[:, data.mask_rec]
        pred_depth2_rec = pred.depth2_rec
        # Depth 2 (ren)
        # (B, S_ren, 1, H, W)
        data_mask_depth2_ren = data.mask_depth[:, data.mask_ren]
        data_depth2_ren = data.depth[:, data.mask_ren]
        pred_depth2_ren = pred.depth2_ren
        # Image (rec)
        # (B * S_rec, 3, H, W)
        data_mask_image_rec = data.mask_image[:, data.mask_rec].flatten(0, 1)
        data_image_rec = data.image[:, data.mask_rec].flatten(0, 1)
        pred_image_rec = pred.image_rec.flatten(0, 1)
        # Image (ren)
        # (B * S_ren, 3, H, W)
        data_mask_image_ren = data.mask_image[:, data.mask_ren].flatten(0, 1)
        data_image_ren = data.image[:, data.mask_ren].flatten(0, 1)
        pred_image_ren = pred.image_ren.flatten(0, 1)
        # Point
        # (B, S_rec * H * W * 1)
        data_mask_point = data.mask_depth[:, data.mask_rec].flatten(1, -1)
        # (B, S_rec * H * W, 3)
        data_point = data.point[:, data.mask_rec].movedim(-3, -1).contiguous().flatten(1, -2)
        pred_point = pred.point.movedim(-3, -1).contiguous().flatten(1, -2)
        # Scene primitive count
        # M
        pred_scene_numel = pred.scene.batch_index.bincount(minlength=pred.batch_size).float().mean()

        # Align valid depth from pred to data scale
        if pred_depth1.numel() > 0 and (
            self.config.depth1_absrel["enabled"] or self.config.depth1_delta1["enabled"]
        ):
            # (B, S_rec, 1, H, W)
            pred_depth1 = torch.cat(
                [
                    d[m].div(p[m]).median().mul(p)
                    for d, m, p in zip(data_depth1, data_mask_depth1, pred_depth1)
                ],
                dim=0,
            )
        if pred_depth2_rec.numel() > 0 and (
            self.config.depth2_rec_absrel["enabled"] or self.config.depth2_rec_delta1["enabled"]
        ):
            # (B, S_rec, 1, H, W)
            pred_depth2_rec = torch.cat(
                [
                    d[m].div(p[m]).median().mul(p)
                    for d, m, p in zip(data_depth2_rec, data_mask_depth2_rec, pred_depth2_rec)
                ],
                dim=0,
            )
        if pred_depth2_ren.numel() > 0 and (
            self.config.depth2_ren_absrel["enabled"] or self.config.depth2_ren_delta1["enabled"]
        ):
            # (B, S_ren, 1, H, W)
            data_mask_depth2_ren = data_mask_depth2_ren & trf.get_mask_for_depth(pred_depth2_ren)
            pred_depth2_ren = torch.cat(
                [
                    d[m].div(p[m]).median().mul(p)
                    for d, m, p in zip(data_depth2_ren, data_mask_depth2_ren, pred_depth2_ren)
                ],
                dim=0,
            )
        # Align valid point from pred to data coordinate
        if self.config.point_cd_nc["enabled"] and pred_point.numel() > 0:
            # [(B, N, 3); (B, N, 3), (B, N)] -> [(B, 3, 3), (B, 3), (B,)]
            r, t, s = umeyama(pred_point, data_point, data_mask_point.float(), estimate_scale=True)
            # (B, N, 3) @ (B, 3, 3) * (B, 1, 1) + (B, 1, 3)
            # -> (B, S_rec * H * W, 3)
            pred_point = pred_point.matmul(r).mul(s[:, None, None]).add(t[:, None])

        if self.config.depth1_absrel["enabled"] and pred_depth1.numel() > 0:
            depth1_absrel: Tensor = self.depth1_absrel(pred_depth1, data_depth1)
            depth1_absrel = depth1_absrel[data_mask_depth1]
            depth1_absrel = depth1_absrel.mean()
            if not depth1_absrel.isnan().item():
                out["depth1_absrel"] = depth1_absrel
        if self.config.depth1_delta1["enabled"] and pred_depth1.numel() > 0:
            depth1_delta1: Tensor = self.depth1_delta1(pred_depth1, data_depth1)
            depth1_delta1 = depth1_delta1[data_mask_depth1]
            depth1_delta1 = depth1_delta1.mean()
            if not depth1_delta1.isnan().item():
                out["depth1_delta1"] = depth1_delta1

        if self.config.depth2_rec_absrel["enabled"] and pred_depth2_rec.numel() > 0:
            depth2_rec_absrel: Tensor = self.depth2_rec_absrel(pred_depth2_rec, data_depth2_rec)
            depth2_rec_absrel = depth2_rec_absrel[data_mask_depth2_rec]
            depth2_rec_absrel = depth2_rec_absrel.mean()
            if not depth2_rec_absrel.isnan().item():
                out["depth2_rec_absrel"] = depth2_rec_absrel
        if self.config.depth2_rec_delta1["enabled"] and pred_depth2_rec.numel() > 0:
            depth2_rec_delta1: Tensor = self.depth2_rec_delta1(pred_depth2_rec, data_depth2_rec)
            depth2_rec_delta1 = depth2_rec_delta1[data_mask_depth2_rec]
            depth2_rec_delta1 = depth2_rec_delta1.mean()
            if not depth2_rec_delta1.isnan().item():
                out["depth2_rec_delta1"] = depth2_rec_delta1

        if self.config.depth2_ren_absrel["enabled"] and pred_depth2_ren.numel() > 0:
            depth2_ren_absrel: Tensor = self.depth2_ren_absrel(pred_depth2_ren, data_depth2_ren)
            depth2_ren_absrel = depth2_ren_absrel[data_mask_depth2_ren]
            depth2_ren_absrel = depth2_ren_absrel.mean()
            if not depth2_ren_absrel.isnan().item():
                out["depth2_ren_absrel"] = depth2_ren_absrel
        if self.config.depth2_ren_delta1["enabled"] and pred_depth2_ren.numel() > 0:
            depth2_ren_delta1: Tensor = self.depth2_ren_delta1(pred_depth2_ren, data_depth2_ren)
            depth2_ren_delta1 = depth2_ren_delta1[data_mask_depth2_ren]
            depth2_ren_delta1 = depth2_ren_delta1.mean()
            if not depth2_ren_delta1.isnan().item():
                out["depth2_ren_delta1"] = depth2_ren_delta1

        if self.config.image_rec_lpips["enabled"] and pred_image_rec.numel() > 0:
            image_rec_lpips: Tensor = self.image_rec_lpips(pred_image_rec, data_image_rec)
            image_rec_lpips = image_rec_lpips[data_mask_image_rec]
            image_rec_lpips = image_rec_lpips.mean()
            if not image_rec_lpips.isnan().item():
                out["image_rec_lpips"] = image_rec_lpips
        if self.config.image_rec_psnr["enabled"] and pred_image_rec.numel() > 0:
            image_rec_psnr: Tensor = self.image_rec_psnr(
                pred_image_rec, data_image_rec, data_mask_image_rec
            )
            if not image_rec_psnr.isnan().item():
                out["image_rec_psnr"] = image_rec_psnr
        if self.config.image_rec_ssim["enabled"] and pred_image_rec.numel() > 0:
            image_rec_ssim: Tensor = self.image_rec_ssim(pred_image_rec, data_image_rec)
            image_rec_ssim = image_rec_ssim[data_mask_image_rec]
            image_rec_ssim = image_rec_ssim.mean()
            if not image_rec_ssim.isnan().item():
                out["image_rec_ssim"] = image_rec_ssim

        if self.config.image_ren_lpips["enabled"] and pred_image_ren.numel() > 0:
            image_ren_lpips: Tensor = self.image_ren_lpips(pred_image_ren, data_image_ren)
            image_ren_lpips = image_ren_lpips[data_mask_image_ren]
            image_ren_lpips = image_ren_lpips.mean()
            if not image_ren_lpips.isnan().item():
                out["image_ren_lpips"] = image_ren_lpips
        if self.config.image_ren_psnr["enabled"] and pred_image_ren.numel() > 0:
            image_ren_psnr: Tensor = self.image_ren_psnr(
                pred_image_ren, data_image_ren, data_mask_image_ren
            )
            if not image_ren_psnr.isnan().item():
                out["image_ren_psnr"] = image_ren_psnr
        if self.config.image_ren_ssim["enabled"] and pred_image_ren.numel() > 0:
            image_ren_ssim: Tensor = self.image_ren_ssim(pred_image_ren, data_image_ren)
            image_ren_ssim = image_ren_ssim[data_mask_image_ren]
            image_ren_ssim = image_ren_ssim.mean()
            if not image_ren_ssim.isnan().item():
                out["image_ren_ssim"] = image_ren_ssim

        if self.config.point_cd_nc["enabled"] and pred_point.numel() > 0:
            x: tuple[Tensor, Tensor] = self.point_cd_nc(pred_point, data_point, data_mask_point)
            point_cd, point_nc = x
            if not point_cd.isnan().item():
                out["point_cd"] = point_cd
            if not point_nc.isnan().item():
                out["point_nc"] = point_nc

        if self.config.pose_rel_auc3["enabled"] and pred_cam_extr.numel() > 0:
            out["pose_rel_auc3"] = self.pose_rel_auc3(pred_cam_extr, data_cam_extr)
        if self.config.pose_rel_auc30["enabled"] and pred_cam_extr.numel() > 0:
            out["pose_rel_auc30"] = self.pose_rel_auc30(pred_cam_extr, data_cam_extr)

        if not pred_scene_numel.isnan().item():
            out["scene_numel"] = pred_scene_numel

        return out

    def train(self, mode: bool = True) -> Self:
        mode = False
        self.requires_grad_(mode)
        return super().train(mode)


class AbsRel(nn.Module):
    """Absolute relative error. (lower is better)"""

    def __init__(self, reduction: Literal["mean", "none", "sum"]) -> None:
        super().__init__()

        self.reduction = reduction

    def forward(self, pred: Tensor, data: Tensor) -> Tensor:
        eps = torch.finfo(pred.dtype).eps
        value = pred.sub(data).abs() / data.clamp(min=eps)
        if self.reduction == "mean":
            return value.mean()
        if self.reduction == "sum":
            return value.sum()
        return value


class CDNC(nn.Module):
    """
    Chamfer distance. (lower is better)
    Normal consistency. (higher is better)

    Shape:
        [(B, S * H * W, 3), (B, S * H * W, 3), (B, S * H * W * 1)] -> [() as CD, () as NC]
    """

    def __init__(self, count_max: int) -> None:
        super().__init__()

        self.count_max = count_max

    def forward(self, pred: Tensor, data: Tensor, mask: Tensor) -> Tensor:
        B = mask.shape[0]
        # (B, N) -> (B,) -> L
        L = mask.sum(dim=-1).amin().clamp(max=self.count_max).item()
        p: list[Tensor] = []
        d: list[Tensor] = []
        for i in range(B):
            # (N,) -> (M,)
            index = mask[i].nonzero().squeeze(-1)
            # (M,)[(L,)] -> (L,)
            index = index[torch.randperm(index.numel(), device=index.device)[:L]]
            p.append(pred[i][index]), d.append(data[i][index])
        # [(B, L, 3); 2]
        p, d = torch.stack(p, dim=0), torch.stack(d, dim=0)
        p_n = estimate_normals(p, disambiguate_directions=False)
        d_n = estimate_normals(d, disambiguate_directions=False)
        # [(B, L, 3); 2] -> [(); 2]
        loss_cd, loss_nc = chamfer_distance(x=p, y=d, x_normals=p_n, y_normals=d_n)
        return loss_cd.div(2.0), loss_nc.new_ones(()).sub(loss_nc.div(2.0))


class Delta1(nn.Module):
    """Threshold accuracy. (higher is better)"""

    def __init__(self, reduction: Literal["mean", "none", "sum"]) -> None:
        super().__init__()

        self.reduction = reduction
        self.register_buffer("threshold", torch.tensor(1.25), persistent=False)

    def forward(self, pred: Tensor, data: Tensor) -> Tensor:
        dtype = pred.dtype
        eps = torch.finfo(dtype).eps
        ratio = (pred / data.clamp(min=eps)).maximum(data / pred.clamp(min=eps))
        value = (ratio < self.threshold).to(dtype)
        if self.reduction == "mean":
            return value.mean()
        if self.reduction == "sum":
            return value.sum()
        return value


class LPIPS(nn.Module):
    """
    Learned perceptual image patch similarity using VGG16 features. (lower is better)

    Shape:
        [(B, C, H, W) in [0, 1], (B, C, H, W) in [0, 1]] -> () || (B, C, H, W)
    """

    def __init__(self, reduction: Literal["mean", "none", "sum"]) -> None:
        super().__init__()

        self.base = lpips.LPIPS(net="vgg", spatial=reduction == "none", verbose=False)
        self.base.requires_grad_(False)
        self.reduction = reduction

    def forward(self, pred: Tensor, data: Tensor) -> Tensor:
        value: Tensor = torch.cat(
            [self.base(p[None], d[None], normalize=True) for p, d in zip(pred, data)],
            dim=0,
        )
        if self.reduction == "mean":
            return value.mean()
        elif self.reduction == "sum":
            return value.sum()
        else:
            return value.expand_as(pred)


class PoseRelAUC(nn.Module):
    """
    Area under the curve for relative pose angular accuracy. (higher is better)

    Shape:
        [(B, S, 3, 4), (B, S, 3, 4)] -> ()
    """

    def __init__(self, angle_max_deg: int) -> None:
        super().__init__()

        self.register_buffer(
            "angle_bin_deg",
            torch.arange(1, angle_max_deg + 1),
            persistent=False,
        )

    def forward(self, pred: Tensor, data: Tensor) -> Tensor:
        S = pred.shape[1]
        device, dtype = pred.device, pred.dtype
        eps = torch.finfo(dtype).eps

        # Extract pairs
        # (B, S, 3, 4)[(P,)] -> (B, P, 3, 4)
        i, j = torch.triu_indices(S, S, 1, device=device)
        data_i, data_j = data[:, i], data[:, j]
        pred_i, pred_j = pred[:, i], pred[:, j]

        # Relative pose via inverse
        # (B, P, 3, 3 + 1)
        r_data_i, t_data_i = data_i.split([3, 1], dim=-1)
        r_pred_i, t_pred_i = pred_i.split([3, 1], dim=-1)
        r_data_inv_j, t_data_inv_j = trf.get_cam_extr_inv(data_j)
        r_pred_inv_j, t_pred_inv_j = trf.get_cam_extr_inv(pred_j)
        r_data = r_data_i @ r_data_inv_j
        r_pred = r_pred_i @ r_pred_inv_j
        t_data = r_data_i @ t_data_inv_j + t_data_i
        t_pred = r_pred_i @ t_pred_inv_j + t_pred_i

        # Relative rotation angle error (RRA)
        # (B, P, 3, 3) -> (B, P, 4) -> (B, P, 1)
        q_data = trf.get_quat_from_matr(r_data)
        q_pred = trf.get_quat_from_matr(r_pred)
        r_angle = (
            ((q_pred * q_data).sum(dim=-1).square() * 2.0 - 1.0).clamp(-1.0, 1.0).acos()[..., None]
        )

        # Relative translation angle error (RTA)
        # (B, P, 3, 1) -> (B, P, 1)
        t_dot = (t_data * t_pred).sum(dim=-2)
        t_norm_prod = (t_data.norm(dim=-2) * t_pred.norm(dim=-2)).clamp(min=eps)
        t_angle: Tensor = (t_dot / t_norm_prod).abs().clamp(max=1.0).acos()

        # Max angle error in degrees
        # (B, P, 1)
        angle_deg = r_angle.maximum(t_angle).rad2deg()

        # AUC via cumulative accuracy
        # (B, P, 1) < (N,) -> (B, P, N) -> ()
        auc = (angle_deg < self.angle_bin_deg).to(dtype).mean()

        return auc


class PSNR(nn.Module):
    """Peak signal-to-noise ratio. (higher is better)"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, data: Tensor, mask: Tensor) -> Tensor:
        eps = torch.finfo(pred.dtype).eps

        mask = mask.expand_as(pred).flatten()
        value = pred.sub(data).square()
        value = value.flatten()
        value = value.mul(mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=eps)
        value = value.mean().clamp(min=eps).log10().mul(-10.0)
        return value


class SSIM(nn.Module):
    """
    Structural similarity index measure metric. (higher is better)

    Shape:
        [(B, C, H, W) in [0, 1], (B, C, H, W) in [0, 1]] -> () || (B, C, H, W)
    """

    def __init__(self, reduction: Literal["mean", "none", "sum"]) -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, data: Tensor) -> Tensor:
        value: Tensor = FusedSSIMMap.apply(0.01**2, 0.03**2, pred, data, "same", False)
        if self.reduction == "mean":
            return value.mean()
        elif self.reduction == "sum":
            return value.sum()
        else:
            return value


class MetricConfig(TypedDict):
    enabled: bool
