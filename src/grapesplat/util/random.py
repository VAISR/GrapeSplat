"""Random utility for GrapeSplat."""

from lightning import pytorch as pl
import os


def get_seed() -> int:
    """Get current universal seed value."""
    return int(os.environ.get("PL_GLOBAL_SEED", "0"))


def set_seed(value: int) -> None:
    """Set current universal seed value."""
    pl.seed_everything(int(value), workers=True, verbose=False)
