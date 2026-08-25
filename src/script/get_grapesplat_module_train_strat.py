"""Inspect pipeline training strategies for GrapeSplat."""

from hydra import compose, initialize
from hydra.utils import call
from grapesplat.module import GrapeSplatLightningModule
from grapesplat.module.pipeline.train import debug_train_strat, get_param_size
from grapesplat.util import logger
from pprint import pprint
import logging
import torch


NAMES = [
    "our_voxel_affine",
    "our_voxel_peach",
    "our_gauss_clustered",
    "our_sem_encoded",
    "ref_anysplat",
    "ref_da3",
    "ref_splatweaver",
    "ref_structsplat",
    "ref_twoxplat",
]


def main() -> None:
    logger.setLevel(logging.DEBUG)
    for name in NAMES:
        overrides = [f"ckpt=.model/{name}.ckpt"] if name.startswith("our_") else []
        with initialize(
            config_path="../../config/grapesplat",
            version_base="1.3",
        ):
            config = compose(
                config_name="test",
                overrides=overrides + [f"module/pipeline={name}"],
            )
            builder = call(config.module, _convert_="all")

        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        module: GrapeSplatLightningModule = builder()
        module.configure_model()
        module = module.to(device)
        mem_gpu_init_mb = f"{torch.cuda.max_memory_allocated(device) / 1e6:.1f}MB"

        module = module.pipeline
        class_name = type(module).__name__
        debug_train_strat("LOCK", get_param_size(module, False), class_name)
        debug_train_strat("TUNE", get_param_size(module, True), class_name)
        debug_train_strat("TOTAL", get_param_size(module, None), class_name)
        logger.debug(f"{'MEM_GPU_INIT':<20} : {mem_gpu_init_mb:>10} : ")
        logger.debug(f"{'-' * 80}")
        pprint(module)


if __name__ == "__main__":
    main()
