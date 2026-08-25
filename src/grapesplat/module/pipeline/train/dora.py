"""DoRA training strategy."""

from .strat import StrategyBuilder
from dataclasses import dataclass
from torch import Tensor, nn
from torch.nn import functional as nnf
import torch


@dataclass
class DoRABuilder(StrategyBuilder):
    """Weight-Decomposed Low-Rank Adaptation (DoRA)."""

    alpha: int
    dropout: float
    rank: int

    def __call__(self, base: nn.Linear) -> "DoRA":
        if isinstance(base, DoRA):
            return base
        if not isinstance(base, nn.Linear):
            raise TypeError(
                f"Mismatched base type: expected torch.nn.Linear, got {type(base).__name__}"
            )
        return DoRA(base=base, config=self)

    @property
    def name(self) -> str:
        return "dora"


class DoRA(nn.Module):
    """
    Weight-Decomposed Low-Rank Adaptation (DoRA).

    Formula:
        V = W_0 + (B @ A) * (a / R)
        W_1 = m * V / RowNorm(V)
        y = x @ W_1^T + b

    Shape:
        (..., C_i) -> (..., C_o)
    """

    def __init__(self, base: nn.Linear, config: DoRABuilder) -> None:
        super().__init__()

        self.config = config
        self.in_features = base.in_features
        self.out_features = base.out_features

        # Initialize sub-modules in execution order
        self.register_buffer(
            "scale",
            torch.tensor(config.alpha / config.rank),
            persistent=False,
        )
        # (R, C_i)
        self.lora_a = nn.Parameter(torch.empty(config.rank, base.in_features))
        # (C_o, R)
        self.lora_b = nn.Parameter(torch.empty(base.out_features, config.rank))
        # (C_o, 1)
        self.mag = nn.Parameter(base.weight.detach().norm(dim=-1, keepdim=True))
        self.dropout = nn.Dropout(p=config.dropout)
        # (C_o, C_i)
        self.weight = nn.Parameter(base.weight.detach().clone())
        # (C_o,)
        if base.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(base.bias.detach().clone())

        # Freeze parameters
        if base.bias is not None:
            self.bias.requires_grad_(False)
        self.weight.requires_grad_(False)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        eps = torch.finfo(dtype).eps
        x = x.to(self.weight.dtype)

        # Weight direction
        # (C_o, C_i)
        v = self.weight + (self.lora_b @ self.lora_a) * self.scale
        # NOTE: The original implementation of DoRA detached the norm intentionally
        v = v / v.detach().norm(dim=-1, keepdim=True).nan_to_num().clamp(eps, 1.0 / eps)

        # Weight magnification
        # (C_o, C_i)
        w = self.mag * v

        # Linear projection with adapted weight
        # (..., C_i) @ (C_i, C_o) + (C_o,) -> (..., C_o)
        x = nnf.linear(self.dropout(x), w, self.bias)
        x = x.to(dtype)
        return x

    def extra_repr(self) -> str:
        return (
            f"{self.in_features}, {self.out_features}, bias={self.bias is not None}, "
            f"rank={self.config.rank}, alpha={self.config.alpha}"
        )
