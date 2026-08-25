"""Optimizer modules for GrapeSplat."""

from dataclasses import dataclass
from torch import nn, optim
from torch.optim import Optimizer
from typing import Literal, TypedDict


@dataclass
class CompositeOptimizerBuilder:
    """Composite optimizer for multiple parameter groups."""

    group: list["ParamGroup"]
    type: Literal["adam_w"]

    def __call__(self, module: nn.Module) -> Optimizer:
        """Build the optimizer."""

        return {"adam_w": optim.AdamW}[self.type](
            [
                {
                    **g,
                    "params": [
                        param
                        for name, param in module.named_parameters()
                        if (name + ".").startswith(g["params"] + ".")
                    ]
                    if g["params"]
                    else [],
                }
                for g in self.group
            ],
        )


class ParamGroup(TypedDict):
    lr: float
    params: str | None
    weight_decay: float
