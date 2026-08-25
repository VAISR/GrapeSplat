"""
Run experiment for GrapeSplat.

Usage:
    uv run run_once test
    uv run run_once test ckpt=last
    uv run run_once test ckpt={path}
    uv run run_once test data=eth3d module/pipeline=ref_anysplat
    uv run run_once test data=nrgbd module/pipeline=our_voxel_affine +view@data=c8n16

    uv run run_once train
    uv run run_once train trainer.devices=1
    uv run run_once train trainer.num_nodes=2 node_ifname=p2p
    uv run run_once train module/pipeline=our_voxel_peach ckpt=last
"""

from grapesplat import util
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call
from lightning import pytorch as pl
from subprocess import run
import sys


def main() -> None:
    cmd, name, opts = sys.argv[0], sys.argv[1], sys.argv[2:]
    with initialize(config_path="../../config/grapesplat", version_base="1.3"):
        config = compose(config_name=name, overrides=opts, return_hydra_config=True)
        HydraConfig.instance().set_config(config)

    util.setup_backend(config)

    if util.is_launched():
        util.mute_other_logger()
        util.set_seed(config.seed)

        datamodule: pl.LightningDataModule = call(config.data, _convert_="all")()
        module: pl.LightningModule = call(config.module, _convert_="all")()
        trainer: pl.Trainer = call(config.trainer)

        args_trainer = dict(
            ckpt_path=config.ckpt,
            datamodule=datamodule,
            model=module,
        )
        if name.startswith("test"):
            trainer.test(**args_trainer)
        elif name.startswith("train"):
            trainer.fit(**args_trainer)
        else:
            raise ValueError(f"Unknown command: {name}")
    else:
        devices = int(config.trainer.devices)
        nodes = int(config.trainer.num_nodes)
        # TODO: Add support to torchrun
        run(
            [
                "srun",
                "--pty",
                f"--gpus-per-node={devices}",
                f"--nodes={nodes}",
                f"--ntasks-per-node={devices}",
                sys.executable,
                cmd,
                name,
            ]
            + opts,
            check=True,
        )


if __name__ == "__main__":
    main()
