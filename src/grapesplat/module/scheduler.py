"""Scheduler modules for GrapeSplat."""

from bisect import bisect_right
from dataclasses import dataclass
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import math


@dataclass
class CompositeSchedulerBuilder:
    """Composite scheduler for adaptive learning rates."""

    lr_decay_min: float
    lr_warmup_iters_pr: float

    iters: int = None

    def __call__(
        self,
        iters: int,
        optimizer: Optimizer,
    ) -> "CompositeScheduler":
        """Build the scheduler."""

        self.iters = iters

        return CompositeScheduler(config=self, optimizer=optimizer)


class CompositeScheduler:
    def __init__(
        self,
        config: CompositeSchedulerBuilder,
        optimizer: Optimizer,
    ) -> None:
        lr_warmup_iters = int(config.iters * config.lr_warmup_iters_pr)
        lr_decay_iters = config.iters - lr_warmup_iters

        lr_decay = CosineAnnealingLR(optimizer, T_max=lr_decay_iters, eta_min=config.lr_decay_min)
        if lr_warmup_iters > 0:
            lr_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=lr_warmup_iters)
            lr_scheduler = SequentialLR(optimizer, [lr_warmup, lr_decay], [lr_warmup_iters])
        else:
            lr_scheduler = lr_decay

        self.config = config

        # Initialize sub-modules in execution order
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer

    def load_state_dict(self, state_dict: dict[str]) -> None:
        """Restore progress onto the freshly built schedule."""

        iters_done = state_dict["lr_scheduler"]["last_epoch"]
        lr_scheduler = self.lr_scheduler
        lr_scheduler.last_epoch = iters_done
        if isinstance(lr_scheduler, SequentialLR):
            index = bisect_right(lr_scheduler._milestones, iters_done)
            iters_done -= ([0] + lr_scheduler._milestones)[index]
            lr_scheduler = lr_scheduler._schedulers[index]
        lr_scheduler._update_lr(iters_done)

    def state_dict(self) -> dict[str]:
        return dict(
            lr_scheduler=self.lr_scheduler.state_dict(),
        )

    def step(self) -> None:
        self.lr_scheduler.step()


class CosineAnnealingWt:
    """
    Cosine annealing schedule for weight attribute.

    Formula:
        w_t = w_init * (w_start_pr + (w_start_pr - w_end_pr) * (cos(π * t / t_max) * 0.5 - 0.5)
    """

    def __init__(
        self,
        obj: dict[str],
        t_max: int,
        w_end_pr: float,
        w_start_pr: float,
        step_last: int = -1,
    ) -> None:
        self.obj = obj
        self.step_last = step_last
        self.t_max = t_max
        self.w_init = float(obj["weight"])
        self.w_end_pr = w_end_pr
        self.w_start_pr = w_start_pr
        self.step()

    @property
    def w(self) -> float:
        return self.w_init * (
            self.w_start_pr
            + (self.w_start_pr - self.w_end_pr)
            * (math.cos(math.pi * self.step_last / self.t_max) * 0.5 - 0.5)
        )

    def load_state_dict(self, state_dict: dict[str]) -> None:
        self.__dict__.update(state_dict)
        self.obj["weight"] = self.w_last

    def state_dict(self) -> dict[str]:
        return {key: value for key, value in self.__dict__.items() if key != "obj"}

    def step(self) -> None:
        self.step_last += 1
        w = self.w
        self.w_last = w
        self.obj["weight"] = w


class ParallelWt:
    """Parallel scheduler for multiple weight attributes."""

    def __init__(self, schedulers: list[CosineAnnealingWt]) -> None:
        self.schedulers = schedulers

    def load_state_dict(self, state_dict: list[dict[str]]) -> None:
        for scheduler, sd in zip(self.schedulers, state_dict):
            scheduler.load_state_dict(sd)

    def state_dict(self) -> list[dict[str]]:
        return [scheduler.state_dict() for scheduler in self.schedulers]

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()
