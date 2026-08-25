"""Spatial convolution modules for UNet."""

from dataclasses import dataclass, replace
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchsparse import SparseTensor, nn as spnn
import torchsparse as tchsp


class SpSequentialChunked(nn.Sequential):
    """
    Spatial point-wise module container per chunk.

    Shape:
        (V, C_i) -> (V, C_o)
    """

    def __init__(self, *module: nn.Module) -> None:
        super().__init__(*module)

        for m in self:
            if isinstance(m, spnn.Conv3d):
                if m.kernel_volume > 1:
                    raise ValueError(f"Unsupported kernel size: expected 1, got {m.kernel_size}")
                if any(s != 1 for s in m.stride):
                    raise ValueError(f"Unsupported stride: expected 1, got {m.stride}")
                if m.transposed:
                    raise TypeError("Unsupported kernel type: expected transposed=False")

    def forward(self, x: SparseTensor) -> SparseTensor:
        CHUNK_SIZE = 2**18
        V = x.feats.shape[0]

        f = None
        for i in range(0, V, CHUNK_SIZE):
            s = slice(i, i + CHUNK_SIZE)
            y = SparseTensor(
                coords=x.coords[s],
                feats=x.feats[s],
                spatial_range=x.spatial_range,
                stride=x.stride,
            )
            for module in self:
                y = module(y)
            if f is None:
                f = y.feats.new_empty(V, y.feats.shape[-1])
            f[s] = y.feats

        y = SparseTensor(
            coords=x.coords,
            feats=f,
            spatial_range=x.spatial_range,
            stride=x.stride,
        )
        y._caches = x._caches
        return y


@dataclass
class SpConvBlockBuilder:
    """Spatial convolution block."""

    dim_h_base: int
    dim_h_mult: float
    dim_i: int
    dim_o: int
    kernel: int

    def __call__(self) -> "SpConvBlock":
        """Build the module."""

        return SpConvBlock(config=self)

    @property
    def dim_h(self) -> int:
        return int(self.dim_h_base * self.dim_h_mult)


class SpConvBlock(nn.Module):
    """
    Spatial convolution block.

    Shape:
        (V, C_i) -> (V, C_o)
    """

    def __init__(self, config: SpConvBlockBuilder) -> None:
        super().__init__()

        arg_proj = dict(
            bias=True,
            kernel_size=1,
            stride=1,
        )
        self.config = config

        # Initialize sub-modules in execution order
        self.proj = SpSequentialChunked(
            spnn.Conv3d(config.dim_i, config.dim_o, **arg_proj),
        )
        self.fuse_i = SpSequentialChunked(
            spnn.LayerNorm(config.dim_i),
            spnn.Conv3d(config.dim_i, config.dim_h_base, **arg_proj),
        )
        self.conv = spnn.Conv3d(
            config.dim_h_base,
            config.dim_h_base,
            bias=True,
            kernel_size=config.kernel,
            stride=1,
        )
        self.fuse_o = SpSequentialChunked(
            spnn.Conv3d(config.dim_h_base, config.dim_h, **arg_proj),
            spnn.GELU(),
            spnn.Conv3d(config.dim_h, config.dim_o, **arg_proj),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        # (V, C_i) -> (V, C_o)
        s = self.proj(x)
        # (V, C_i) -> (V, C_o)
        x = checkpoint(self.forward_inner, x, use_reentrant=False)
        # Skip connection
        x = x + s
        return x

    def forward_inner(self, x: SparseTensor) -> SparseTensor:
        # (V, C_i) -> (V, C_h) -> (V, C_o)
        return self.fuse_o(self.conv(self.fuse_i(x)))


@dataclass
class SpConvRefBlockBuilder(SpConvBlockBuilder):
    """Spatial convolution refinement block."""

    depth: int

    def __call__(self) -> "SpConvRefBlock":
        """Build the module."""

        return SpConvRefBlock(config=self)


class SpConvRefBlock(nn.Module):
    """
    Spatial convolution refinement block.

    Shape:
        (V, C_i) -> (V, C_o)
    """

    def __init__(self, config: SpConvRefBlockBuilder) -> None:
        super().__init__()

        dims = [config.dim_i] + [config.dim_o] * config.depth
        self.config = config

        # Initialize sub-modules in execution order
        self.block = nn.Sequential(
            *(
                SpConvBlockBuilder.__call__(replace(config, dim_i=dims[i], dim_o=dims[i + 1]))
                for i in range(len(dims) - 1)
            )
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        # (V, C_i) -> (V, C_o)
        return self.block(x)


@dataclass
class UNetSpConvEncBlockBuilder(SpConvRefBlockBuilder):
    """UNet spatial convolution encoder block."""

    def __call__(self) -> "UNetSpConvEncBlock":
        """Build the module."""

        return UNetSpConvEncBlock(config=self)


class UNetSpConvEncBlock(nn.Module):
    """
    UNet spatial convolution encoder block.

    Shape:
        (V_i, C_i) -> [(V_o, C_o), (V_i, C_o)]
    """

    def __init__(self, config: UNetSpConvEncBlockBuilder) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        self.ref = SpConvRefBlockBuilder.__call__(config)
        self.down = nn.Sequential(
            spnn.LayerNorm(config.dim_o),
            spnn.Conv3d(
                config.dim_o,
                config.dim_o,
                bias=True,
                kernel_size=2,
                stride=2,
            ),
        )

    def forward(self, x: SparseTensor) -> tuple[SparseTensor, SparseTensor]:
        # (V_i, C_i) -> (V_i, C_o)
        s = self.ref(x)
        # (V_i, C_o) -> (V_o, C_o)
        x = checkpoint(self.down, s, use_reentrant=False)
        return x, s


@dataclass
class UNetSpConvDecBlockBuilder(SpConvRefBlockBuilder):
    """UNet spatial convolution decoder block."""

    def __call__(self) -> "UNetSpConvDecBlock":
        """Build the module."""

        return UNetSpConvDecBlock(config=self)


class UNetSpConvDecBlock(nn.Module):
    """
    UNet spatial convolution decoder block.

    Shape:
        [(V_i, C_i), (V_o, C_o)] -> (V_o, C_o)
    """

    def __init__(self, config: UNetSpConvDecBlockBuilder) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        self.up = nn.Sequential(
            spnn.LayerNorm(config.dim_i),
            spnn.Conv3d(
                config.dim_i,
                config.dim_o,
                bias=True,
                kernel_size=2,
                stride=2,
                transposed=True,
            ),
        )
        self.ref = SpConvRefBlockBuilder.__call__(replace(config, dim_i=config.dim_o * 2))

    def forward(self, x: SparseTensor, s: SparseTensor) -> SparseTensor:
        # (V_i, C_i) -> (V_o, C_o)
        x = checkpoint(self.up, x, use_reentrant=False)
        # Skip connection
        # [(V_o, C_o), (V_o, C_o)] -> (V_o, C_o * 2)
        x = tchsp.cat([x, s])
        # (V_o, C_o * 2) -> (V_o, C_o)
        x = self.ref(x)
        return x
