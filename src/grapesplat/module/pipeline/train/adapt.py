"""Pipeline training strategy orchestration."""

from .full_ft import FullFTBuilder
from .part_ft import PartFTBuilder
from .strat import StrategyBuilder
from .util import debug_train_strat, get_param_size, logger
from torch import nn
from typing import Iterable, TypedDict
import copy
import re

TrainStrat = list[StrategyBuilder]


class WithTrainStrat(TypedDict):
    train_strat: TrainStrat


def adapt_train_strat(module: nn.Module, train_strat: TrainStrat) -> None:
    """Adapt training strategies to module."""

    if not isinstance(module, nn.Module):
        raise TypeError(f"Mismatched module type: expected torch.nn.Module, got {type(module)}")
    train_strat = copy.deepcopy(train_strat)

    # Freeze parameters
    FullFTBuilder([], grad=False)(module)

    strat_complex: list[tuple[int, str, StrategyBuilder]] = []
    strat_global = []
    for strat in train_strat:
        # Skip simple strategies
        if isinstance(strat, (FullFTBuilder, PartFTBuilder)):
            strat_global.append(strat)
            continue

        # Collect complex strategies to adapt
        target = [re.compile(t) for t in strat.target]
        for module_path, base_module in list(module.named_modules()):
            if any(t.fullmatch(module_path) for t in target):
                strat_complex.append((module_path.count("."), module_path, strat))

    # Adapt complex strategies on deeper modules first
    strat_complex.sort(key=lambda x: -x[0])
    for _, module_path, strat in strat_complex:
        # Make adapted module
        base_module = module.get_submodule(module_path)
        source = strat(base_module)

        # Inject adapted module if applicable
        target, child = module, module_path
        if "." in module_path:
            parent, child = module_path.rsplit(".", 1)
            target = module.get_submodule(parent)
        if not isinstance(strat, (PartFTBuilder,)):
            setattr(target, child, source)
        debug_train_strat(strat.name, get_param_size(source), module_path)

    # Adapt global strategy last
    for strat in strat_global:
        strat(module)

    # Show final stats
    class_name = type(module).__name__
    logger.debug(f"{' ' * 80}")
    debug_train_strat("LOCK", get_param_size(module, False), class_name)
    debug_train_strat("TUNE", get_param_size(module, True), class_name)
    debug_train_strat("TOTAL", get_param_size(module, None), class_name)
    logger.debug(f"{'-' * 80}")


def match_train_strat(
    source: TrainStrat,
    target: type[StrategyBuilder],
) -> Iterable[tuple[int, StrategyBuilder]]:
    return (s for s in enumerate(source) if isinstance(s[1], target))
