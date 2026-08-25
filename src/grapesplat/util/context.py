"""Context management utility for GrapeSplat."""

from contextlib import AbstractContextManager
import time
import torch


class InferenceProfiler(AbstractContextManager):
    """Profile latency and peak memory during inference."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self.metric: dict[str, float] = {}

    def __enter__(self) -> "InferenceProfiler":
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        self.pc = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        torch.cuda.synchronize(self.device)
        sec = time.perf_counter() - self.pc
        gib = torch.cuda.max_memory_allocated(self.device) / 2**30
        self.metric = dict(compute_gib=gib, compute_sec=sec)
        del self.pc
