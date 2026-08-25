"""Pipeline training utilities."""

from ....util import logger
from torch import nn


def debug_train_strat(strat_name: str, param_size: int, module_path: str) -> None:
    param_size = f"{round(param_size / 1e6, 3):.03f}M"
    logger.debug(f"{strat_name:<20} : {param_size:>10} : '{module_path}'")


def get_param_size(
    target: nn.Module | nn.Parameter,
    requires_grad: bool | None = None,
) -> int:
    if isinstance(target, nn.Parameter):
        return target.numel()
    if isinstance(target, nn.Module):
        return sum(
            p.numel()
            for p in target.parameters()
            if requires_grad is None or p.requires_grad == requires_grad
        )
    return 0
