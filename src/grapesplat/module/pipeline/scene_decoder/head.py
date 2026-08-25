"""Prediction head modules for Scene decoder."""

from ... import typing as Tp
from .mlp import MLP, MLPBuilder
from dataclasses import dataclass
from torch import Tensor, nn
import torch


@dataclass
class GaussHeadBuilder:
    """Gaussian head for clustered prediction."""

    count: int
    group: list[MLPBuilder]

    dim_i: int = None

    def __call__(self, dim_i: int) -> "GaussHead":
        """Build the module."""

        self.dim_i = dim_i
        return GaussHead(config=self)


class GaussHead(nn.Module):
    """
    Gaussian head for clustered prediction.

    Shape:
        (V_i, C_i) -> (V_o, C_o)
    """

    def __init__(self, config: GaussHeadBuilder) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        self.group: dict[str, list[MLP]] = nn.ModuleDict(
            {
                m.label: nn.ModuleList(m(dim_i=config.dim_i) for _ in range(config.count))
                for m in config.group
            }
        )
        self.register_buffer("gene", torch.empty(config.count, config.dim_i), persistent=True)

        # Initialize parameters
        nn.init.normal_(self.gene, std=0.5 / config.count**0.5)

    def forward(self, inp: Tp.GaussHeadInp) -> Tp.GaussHeadOut:
        x = inp.grid
        CHUNK_SIZE = 2**18
        # [g, (V_i, 4), (V_i, C_i)]
        G, c, f = x.spatial_range[-1], x.coords, x.feats
        L, V = self.config.count, f.shape[0]

        # Clustered prediction per chunk
        # [(V_i, C_i); L] -> (V_o, C_o)
        y: dict[str, Tensor] = {}
        for i in range(L):
            for j in range(0, V, CHUNK_SIZE):
                # (V_s, C_i)
                e = self.gene[i] + f[j : j + CHUNK_SIZE]
                for label, heads in self.group.items():
                    o = heads[i](e)
                    k = V * i + j
                    if label not in y:
                        y[label] = o.new_empty((L * V,) + o.shape[1:])
                    y[label][k : k + o.shape[0]] = o

        # [(V_i, 4); L] -> (V_o, 1 + 3)
        b, p = c.repeat(L, 1).split([1, 3], dim=-1)

        return Tp.GaussHeadOut(
            **y,
            batch_index=b.squeeze(-1).contiguous(),
            grid_resol=G,
            pos_grid=p,
        )
