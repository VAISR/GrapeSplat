"""Pipeline module for GrapeSplat: Image encoder."""

from ..typing import ImageEncoderInp, ImageEncoderOut
from .train import WithTrainStrat, adapt_train_strat
from dataclasses import dataclass
from torch import Tensor, nn
import torch


@dataclass
class ImageEncoderBuilder:
    """Image encoder module to patchify image."""

    geo: "ImageEncoderConfig"
    sem: "ImageEncoderConfig"

    def __call__(self, base_geo: nn.Module, base_sem: nn.Module) -> "ImageEncoder":
        """Build the module."""

        return ImageEncoder(base_geo=base_geo, base_sem=base_sem, config=self)


class ImageEncoder(nn.Module):
    """Image encoder module to patchify image."""

    def __init__(
        self,
        base_geo: nn.Module,
        base_sem: nn.Module,
        config: ImageEncoderBuilder,
    ) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        if config.geo["enabled"]:
            self.base_geo = base_geo
            adapt_train_strat(self.base_geo, config.geo["train_strat"])
        if config.sem["enabled"]:
            self.base_sem = base_sem
            adapt_train_strat(self.base_sem, config.sem["train_strat"])
            for m in self.base_sem.modules():
                is_frozen = not any(p.requires_grad for p in m.parameters())
                is_safe = isinstance(m, (nn.Conv2d, nn.Linear))
                if is_frozen and is_safe:
                    m.to(config.sem["dtype"])

    def forward(self, inp: ImageEncoderInp) -> ImageEncoderOut:
        x = inp.image
        y_geo: tuple[list[Tensor], int] = (
            self.base_geo(x) if self.config.geo["enabled"] else ([], 0)
        )
        y_sem: tuple[list[Tensor], int] = (
            self.base_sem(x) if self.config.sem["enabled"] else ([], 0)
        )

        return ImageEncoderOut(
            frame_resol=x.shape[-2:],
            patch_index_geo=y_geo[1],
            patch_index_sem=y_sem[1],
            token_geo=y_geo[0],
            token_sem=y_sem[0],
        )


class ImageEncoderConfig(WithTrainStrat):
    dtype: torch.dtype
    enabled: bool
