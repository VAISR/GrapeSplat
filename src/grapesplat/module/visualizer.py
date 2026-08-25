"""Visualizer modules for GrapeSplat."""

from ..data import transform as trf
from .pipeline import Depth2ImageDecoder, Depth2ImageDecoderBuilder
from . import typing as Tp
from dataclasses import dataclass
from PIL import Image
from matplotlib import colormaps, pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as nnf
import torch


@dataclass
class CompositeVisualizerBuilder:
    """Composite visualizer for pipeline prediction."""

    step_size: int

    base: Depth2ImageDecoderBuilder = None

    def __call__(self, base: Depth2ImageDecoderBuilder) -> "CompositeVisualizer":
        """Build the module."""

        self.base = base
        return CompositeVisualizer(config=self)


class CompositeVisualizer(nn.Module):
    """Composite visualizer for pipeline prediction."""

    def __init__(self, config: CompositeVisualizerBuilder) -> None:
        super().__init__()

        self.config = config
        self.base = config.base()

    def forward(self, pred: Tp.PipelineOut, data: Tp.PipelineData) -> Tp.PipelineVis:
        with torch.autocast(data.depth.device.type, enabled=False):
            return self.forward_unwrapped(pred, data)

    def forward_unwrapped(self, pred: Tp.PipelineOut, data: Tp.PipelineData) -> Tp.PipelineVis:
        # Skip if data is invalid
        is_invalid = data.mask_depth.sum().item() < 1000 or data.mask_image.sum().item() < 1000
        if is_invalid:
            return Tp.PipelineVis()

        # Flatten
        # (B, S, ...) -> (B * S, ...)
        data_cam_extr = data.cam_extr[:, data.mask_rec]
        data_cam_intr = data.cam_intr[:, data.mask_rec]
        data_depth = data.depth.flatten(0, 1)
        data_image = data.image.flatten(0, 1)
        data_mask_depth = data.mask_depth.flatten(0, 1)
        data_mask_depth1 = data.mask_depth[:, data.mask_rec].flatten(0, 1)
        data_mask_depth2_rec = data.mask_depth[:, data.mask_rec].flatten(0, 1)
        data_mask_depth2_ren = data.mask_depth[:, data.mask_ren]
        data_mask_depth2_ren = data_mask_depth2_ren.flatten(0, 1)
        data_mask_image = data.mask_image.flatten(0, 1)
        data_mask_image_rec = data.mask_image[:, data.mask_rec].flatten(0, 1)
        data_mask_image_ren = data.mask_image[:, data.mask_ren].flatten(0, 1)
        data_mask_point = data.mask_depth[:, data.mask_rec].flatten(0, 1)
        data_mask_ray = data.mask_depth[:, data.mask_rec].flatten(0, 1)
        data_point = data.point[:, data.mask_rec].flatten(0, 1)
        data_ray_d = trf.get_ray_unprojected(
            pred.frame_resol,
            data_cam_extr,
            data_cam_intr,
        )[0].flatten(0, 1)

        pred_cam_extr = pred.cam_extr
        pred_cam_intr = pred.cam_intr
        pred_depth1 = pred.depth1.flatten(0, 1)
        pred_depth1_conf = pred.depth1_conf.flatten(0, 1)
        pred_depth2_rec = pred.depth2_rec.flatten(0, 1)
        pred_depth2_ren = pred.depth2_ren.flatten(0, 1)
        pred_image_rec = pred.image_rec.flatten(0, 1)
        pred_image_ren = pred.image_ren.flatten(0, 1)
        pred_point = pred.point.flatten(0, 1)
        pred_ray_d = pred.ray_d.flatten(0, 1)

        out = Tp.PipelineVis(
            data_cam=[
                visualize_cam(pred.frame_resol, data_cam_extr[b], data_cam_intr[b])
                for b in range(pred.batch_size)
            ],
            data_depth=[
                visualize_heatmap(d.where(m, 0.0)) for d, m in zip(data_depth, data_mask_depth)
            ],
            data_image=[
                visualize_image(i.where(m, 0.0)) for i, m in zip(data_image, data_mask_image)
            ],
            data_point=[
                visualize_point(p.where(m, 0.0)) for p, m in zip(data_point, data_mask_point)
            ],
            data_ray_d=[visualize_ray(r.where(m, 0.0)) for r, m in zip(data_ray_d, data_mask_ray)],
            pred_cam=[
                visualize_cam(pred.frame_resol, pred_cam_extr[b], pred_cam_intr[b])
                for b in range(pred.batch_size)
            ],
            pred_depth1=[
                visualize_heatmap(d.where(m, 0.0)) for d, m in zip(pred_depth1, data_mask_depth1)
            ],
            pred_depth1_conf=[
                visualize_heatmap(c.where(m, 0.0))
                for c, m in zip(pred_depth1_conf, data_mask_depth1)
            ],
            pred_depth2_rec=[
                visualize_heatmap(d.where(m, 0.0))
                for d, m in zip(pred_depth2_rec, data_mask_depth2_rec)
            ],
            pred_depth2_ren=[
                visualize_heatmap(d.where(m, 0.0))
                for d, m in zip(pred_depth2_ren, data_mask_depth2_ren)
            ],
            pred_image_rec=[
                visualize_image(i.where(m, 0.0))
                for i, m in zip(pred_image_rec, data_mask_image_rec)
            ],
            pred_image_ren=[
                visualize_image(i.where(m, 0.0))
                for i, m in zip(pred_image_ren, data_mask_image_ren)
            ],
            pred_point=[
                visualize_point(p.where(m, 0.0)) for p, m in zip(pred_point, data_mask_point)
            ],
            pred_ray_d=[visualize_ray(r.where(m, 0.0)) for r, m in zip(pred_ray_d, data_mask_ray)],
            pred_scene_rec=[
                visualize_image(i.where(m, 0.0))
                for i, m in zip(
                    visualize_scene(
                        Tp.Depth2ImageDecoderInp(
                            cam_extr=pred_cam_extr,
                            cam_intr=pred_cam_intr,
                            frame_resol=pred.frame_resol,
                            scene=pred.scene,
                        ),
                        base=self.base,
                    ),
                    data_mask_image_rec,
                )
            ],
        )
        return out


def visualize_cam(
    frame_resol: tuple[int, int],
    extr: Tensor,
    intr: Tensor,
) -> Image.Image:
    """
    Draw cameras on canvas.

    Shape:
        [(S, 3, 4) in W2C, (S, 3, 3), (H, W)] -> (H, W, 3)
    """

    (H, W), S = frame_resol, extr.shape[-3]
    eps = torch.finfo(extr.dtype).eps

    # Camera position and direction in XZ plane
    # (S, 3, 4) -> [(S, 3, 3), (S, 3, 1)] -> [[(S,), (S,)], [(S,), (S,)]]
    r_inv, t_inv = trf.get_cam_extr_inv(extr)
    dir_x, dir_z = nnf.normalize(r_inv[..., 2, [0, 2]], dim=-1, eps=eps).unbind(dim=-1)
    pos_x, pos_z = t_inv[..., [0, 2], 0].unbind(dim=-1)

    # Horizontal FOV half-angle for XZ plane
    # [(S,); 2]
    fov_half = (W / 2 / intr[..., 0, 0]).atan()
    fov_cos, fov_sin = fov_half.cos(), fov_half.sin()

    # Frustum edge directions scaled by position spread
    # [(S,); 4]
    frus_len = torch.stack([pos_x, pos_z], dim=0).std()
    frus_l_x = (dir_x * fov_cos - dir_z * fov_sin) * frus_len
    frus_l_z = (dir_x * fov_sin + dir_z * fov_cos) * frus_len
    frus_r_x = (dir_x * fov_cos + dir_z * fov_sin) * frus_len
    frus_r_z = (-dir_x * fov_sin + dir_z * fov_cos) * frus_len

    frus_l_x, frus_l_z = frus_l_x.cpu().numpy(), frus_l_z.cpu().numpy()
    frus_r_x, frus_r_z = frus_r_x.cpu().numpy(), frus_r_z.cpu().numpy()
    pos_x, pos_z = pos_x.cpu().numpy(), pos_z.cpu().numpy()

    # Draw XZ top-down view
    dpi = 200
    fg, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    fg.set_dpi(dpi)
    fg.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.scatter(pos_x, pos_z, c="#4ECDC4", s=50)
    for i in range(S):
        ax.plot(
            [pos_x[i], pos_x[i] + frus_l_x[i]],
            [pos_z[i], pos_z[i] + frus_l_z[i]],
            c="#FF6B6B",
        )
        ax.plot(
            [pos_x[i], pos_x[i] + frus_r_x[i]],
            [pos_z[i], pos_z[i] + frus_r_z[i]],
            c="#FF6B6B",
        )
    ax.set_aspect("equal")
    ax.axis("off")
    fg.tight_layout(pad=0)

    # Render to image
    fg.canvas.draw()
    image = Image.frombuffer(
        "RGBA",
        size=fg.canvas.get_width_height(),
        data=fg.canvas.buffer_rgba(),
    ).convert("RGB")
    plt.close(fg)

    return image


def visualize_heatmap(scalar: Tensor) -> Image.Image:
    """
    Colorize 2D scalar map with turbo colormap.

    Shape:
        (1, H, W) -> (H, W, 3)
    """

    eps = torch.finfo(scalar.dtype).eps

    # Normalize
    # (1, H, W) -> (H, W) in [0, 1]
    scalar = scalar.squeeze(0)
    scalar_min = scalar.min()
    scalar = (scalar - scalar_min) / (scalar.max() - scalar_min).clamp(min=eps)

    # Apply colormap
    # (H, W) -> (H, W, 3) in uint8
    scalar = scalar.cpu().numpy()
    scalar = (colormaps["turbo"](scalar)[..., 0:3] * 255.0).astype("uint8")
    return Image.fromarray(scalar)


def visualize_image(image: Tensor) -> Image.Image:
    """
    Convert RGB tensor to PIL image.

    Shape:
        (3, H, W) in [0, 1] -> (H, W, 3)
    """

    # Rescale and convert to uint8
    image = image.movedim(-3, -1).contiguous().clamp(0.0, 1.0).mul(255.0)
    image = image.to(torch.uint8).cpu().numpy()
    return Image.fromarray(image)


def visualize_point(point: Tensor) -> Image.Image:
    """
    Colorize 3D point map with normal direction.

    Shape:
        (3, H, W) -> (H, W, 3)
    """
    eps = torch.finfo(point.dtype).eps

    # Partial derivative along orthogonal directions
    # (3, H, W) -> [(3, H - 1, W - 1); 2]
    dh = point[..., 1:, :-1] - point[..., :-1, :-1]
    dw = point[..., :-1, 1:] - point[..., :-1, :-1]

    # Cross product and normalize
    # [(3, H - 1, W - 1); 2] -> (3, H - 1, W - 1)
    normal = nnf.normalize(torch.cross(dh, dw, dim=-3), dim=-3, eps=eps)

    # Pad by replicating border
    # (3, H - 1, W - 1) -> (3, H, W)
    normal = nnf.pad(normal, (0, 1, 0, 1), mode="replicate")

    # Rescale to unit range
    normal = normal.mul(0.5).add(0.5)
    return visualize_image(normal)


def visualize_ray(ray: Tensor) -> Image.Image:
    """
    Visualize ray direction.

    Shape:
        (3, H, W) -> (H, W, 3)
    """

    eps = torch.finfo(ray.dtype).eps
    # Normalize and rescale to unit range
    ray = nnf.normalize(ray, dim=-3, eps=eps).mul(0.5).add(0.5)

    return visualize_image(ray)


def visualize_scene(
    inp: Tp.Depth2ImageDecoderInp,
    base: Depth2ImageDecoder,
) -> Tensor:
    """
    Render scene points from all viewpoints.

    Shape:
        (B * S, 3, H, W)
    """

    s = inp.scene
    inp = inp._replace(
        scene=s._replace(
            color=s.color.mul(0.2),
            covariance=s.covariance.mul(0.1),
            opacity=s.opacity.new_full(s.batch_index.shape + (1,), 0.2),
        )
    )
    out_ren: Tp.Depth2ImageDecoderOut = base(inp)
    # (B * S, 3, H, W)
    image = out_ren.image.flatten(0, 1)
    return image
