"""Logging utility for GrapeSplat."""

import lightning  # noqa: F401
import logging
import pytorch_lightning  # noqa: F401
import transformers
import torch
import warnings

logger = logging.getLogger("grapesplat")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)s : %(levelname)s : %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.WARNING)


def mute_other_logger() -> None:
    """Mute other loggers to reduce verbosity."""

    logging.getLogger("lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    torch.set_printoptions(edgeitems=2, linewidth=120, precision=4, sci_mode=False)
    transformers.logging.disable_progress_bar()

    warnings.filterwarnings(
        "ignore",
        message=r"FLOPs not found",
        module=r"lightning\.fabric\.utilities\.throughput",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`isinstance\(treespec, LeafSpec\)` is deprecated",
        module=r"lightning\.pytorch\.utilities\._pytree",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Checkpoint directory .+ exists and is not empty",
        module=r"lightning\.pytorch\.callbacks\.model_checkpoint",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Precision .+ is not supported",
        module=r"lightning\.pytorch\.utilities\.model_summary",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The epoch parameter in `scheduler\.step\(\)` was not necessary",
        module=r"torch\.optim\.lr_scheduler",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The argument 'device' of Tensor\.(pin_memory|is_pinned)\(\) is deprecated",
        module=r"torch\.utils\.data\._utils\.pin_memory",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"This DataLoader will create .+ worker processes in total",
        module=r"torchdata\.stateful_dataloader\.stateful_dataloader",
    )
