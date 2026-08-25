"""Coding modules for Scene decoder."""

from dataclasses import dataclass
from torch import Tensor, nn
from torch.nn import functional as nnf
import torch


class ExpTanh(nn.Module):
    """
    Exponential of hyperbolic tangent.

    Formula:
        y = exp(b * tanh(x))
    """

    def __init__(self, bound_log: float) -> None:
        super().__init__()

        self.register_buffer("b", torch.tensor(bound_log), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh().mul(self.b).exp()


class FFN(nn.Module):
    """
    Feed-forward network.

    Shape:
        (..., C_i) -> (..., C_i)
    """

    def __init__(self, dim_h: int, dim_i: int) -> None:
        super().__init__()

        # Initialize sub-modules in execution order
        self.fc1 = nn.Linear(dim_i, dim_h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim_h, dim_i)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GSO(nn.Module):
    """
    Gram-Schmidt orthogonalization.

    Formula:
        b_0 = Normalize(a_0)
        b_1 = Normalize(a_1 - b_0 * Dot(b_0, a_1))
        r = [b_0, b_1, Cross(b_0, b_1)]^t

    Shape:
        (..., 6) -> (..., 3, 3)
    """

    def __init__(self) -> None:
        super().__init__()

        self.normalize = NormalizeL2(eps=torch.finfo(torch.float32).eps ** 0.5)

    def forward(self, x: Tensor) -> Tensor:
        # (..., 6) -> [(..., 3); 2]
        a = x.split(3, dim=-1)
        # [(..., 3); 2] -> [(..., 3); 3]
        b: list[Tensor] = []
        b += [self.normalize(a[0])]
        b += [self.normalize(a[1] - b[0] * b[0].mul(a[1]).sum(dim=-1, keepdim=True))]
        b += [b[0].cross(b[1], dim=-1)]
        # [(..., 3); 3] -> (..., 3, 3)
        return torch.stack(b, dim=-1)


class LayerScale(nn.Module):
    """Layer scale."""

    def __init__(self, dim: int, init_value: float) -> None:
        super().__init__()

        # Initialize sub-modules in execution order
        self.gamma = nn.Parameter(torch.full((dim,), float(init_value)))

    def forward(self, x: Tensor) -> Tensor:
        return self.gamma.mul(x)


class NormalizeL2(nn.Module):
    """L2 normalization."""

    def __init__(self, eps: float) -> None:
        super().__init__()

        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return nnf.normalize(x, dim=-1, eps=self.eps)


class SHAtt(nn.Module):
    """
    Spherical harmonics band-wise attenuation.

    Formula:
        y_k = a^k x_k

    Shape:
        (..., K * 3) -> (..., K, 3)
    """

    def __init__(self, decay: float) -> None:
        super().__init__()

        DEGREE_MAX = 5

        mask = torch.empty((DEGREE_MAX + 1) ** 2, 1)
        for k in range(DEGREE_MAX + 1):
            mask[k**2 : (k + 1) ** 2] = decay**k

        # Initialize sub-modules in execution order
        # ((5 + 1)^2, 1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        K = x.shape[-1] // 3
        # (..., K * 3) -> (..., K, 3)
        x = x.unflatten(-1, (K, 3))
        # (..., K, 3) * (K, 1) -> (..., K, 3)
        x = x * self.mask[0:K]
        return x


@dataclass
class SPEBuilder:
    """Sinusoidal positional encoding in NeRF style."""

    band: int

    def __call__(self) -> "SPE":
        """Build the module."""

        return SPE(config=self)

    @property
    def dim_x(self) -> int:
        return 1 + 2 * self.band


class SPE(nn.Module):
    """
    Sinusoidal positional encoding in NeRF style.

    Formula:
        y_c = [sin(2^0 π p_c), cos(2^0 π p_c), ..., sin(2^(L-1) π p_c), cos(2^(L-1) π p_c)]
        y = Cat[p, Cat_c[y_c]]

    Shape:
        (..., C, ...) -> (..., C * (1 + 2L), ...)
    """

    def __init__(self, config: SPEBuilder) -> None:
        super().__init__()

        self.config = config

        # Initialize sub-modules in execution order
        # (L,)
        self.register_buffer(
            "freq",
            torch.pi * 2.0 ** torch.arange(config.band),
            persistent=False,
        )

    def forward(self, p: Tensor, dim: int = -1) -> Tensor:
        if dim < 0:
            dim += p.ndim

        freq = self.freq[(None,) * (dim + 1) + (...,) + (None,) * (p.ndim - dim - 1)]
        # (..., C, 1, ...) * (..., L, ...) -> (..., C, L, ...)
        t = p.unsqueeze(dim + 1).mul(freq)
        # (..., C, L, 2, ...) -> (..., C * L * 2, ...) -> (..., C * (1 + 2L), ...)
        e = torch.stack([t.sin(), t.cos()], dim=dim + 2).flatten(dim, dim + 2)
        return torch.cat([p, e], dim=dim)

    def extra_repr(self):
        return f"band={self.config.band}"
