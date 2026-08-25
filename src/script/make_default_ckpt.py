"""
Make default.ckpt for GrapeSplat.

Usage:
    uv run make_default_ckpt
"""

from grapesplat import util
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call
from lightning import pytorch as pl
from subprocess import run
import sys


def main() -> None:
    cmd, opts = sys.argv[0], sys.argv[1:]
    with initialize(config_path="../../config/grapesplat", version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "module.optimizer.group.0.lr=0.0",
                "module.optimizer.group.1.lr=0.0",
                "module.optimizer.group.2.lr=0.0",
                "module.optimizer.group.3.lr=0.0",
                "trainer.devices=1",
                "trainer.num_nodes=1",
                "+trainer.max_steps=1",
            ],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(config)

    util.setup_backend(config)

    if util.is_launched():
        util.mute_other_logger()
        util.set_seed(config.seed)

        datamodule: pl.LightningDataModule = call(config.data, _convert_="all")()
        module: pl.LightningModule = call(config.module, _convert_="all")()
        trainer: pl.Trainer = call(config.trainer)

        trainer.fit(datamodule=datamodule, model=module)
        trainer.save_checkpoint(".model/default.ckpt")

    else:
        devices = int(config.trainer.devices)
        nodes = int(config.trainer.num_nodes)
        run(
            [
                "srun",
                "--pty",
                f"--gpus-per-node={devices}",
                f"--nodes={nodes}",
                f"--ntasks-per-node={devices}",
                sys.executable,
                cmd,
            ]
            + opts,
            check=True,
        )


if __name__ == "__main__":
    main()
