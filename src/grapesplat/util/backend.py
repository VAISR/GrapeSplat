"""Backend utility for GrapeSplat."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from omegaconf import DictConfig
import torch


def get_worker_count() -> int:
    """Get CPUs per GPU for data processing."""

    MAX = 8

    ncpu = os.cpu_count() or 1
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
    return min(max((ncpu + ngpu - 1) // ngpu - 1, 1), MAX)


def is_launched() -> bool:
    """Check if launched in distributed environment."""

    return any(v in os.environ for v in ("SLURM_NTASKS", "TORCHELASTIC_RUN_ID"))


def setup_backend(config: DictConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", config.node_ifname)
    os.environ.setdefault("NCCL_SOCKET_IFNAME", config.node_ifname)
    os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "4")
    os.environ.setdefault("NCCL_SOCKET_NTHREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", str(get_worker_count()))
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
