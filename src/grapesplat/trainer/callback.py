"""Trainer callbacks for GrapeSplat."""

from ..module import PipelineData
from lightning.pytorch import callbacks as plcb
from torch import distributed as dist
import gc
import torch


class CacheCleanup(plcb.Callback):
    """Callback for in-memory cache cleanup."""

    def on_test_epoch_end(self, *_, **__) -> None:
        self.run()

    def on_test_epoch_start(self, *_, **__) -> None:
        self.run()

    def on_train_epoch_end(self, *_, **__) -> None:
        self.run()

    def on_train_epoch_start(self, *_, **__) -> None:
        self.run()

    def on_validation_epoch_end(self, *_, **__) -> None:
        self.run()

    def on_validation_epoch_start(self, *_, **__) -> None:
        self.run()

    def run(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


class OnKeyboardInterruptCleanup(plcb.Callback):
    """Callback for on-keyboard-interrupt cleanup."""

    def on_exception(self, *args, **kwargs) -> None:
        err = kwargs.get("exception") or args[-1]
        if isinstance(err, KeyboardInterrupt) and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


class ThroughputMonitor(plcb.ThroughputMonitor):
    """Callback for throughput monitoring."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            batch_size_fn=PipelineData.get_batch_size,
            length_fn=PipelineData.get_seq_size,
            **kwargs,
        )
