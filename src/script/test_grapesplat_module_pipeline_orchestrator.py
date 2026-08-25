"""Integration test for GrapeSplat module layer: Pipeline orchestrator."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dataclasses import dataclass
from datetime import timedelta
from grapesplat import util
from grapesplat.data import GrapeSplatDataModule
from grapesplat.module import GrapeSplatLightningModule, PipelineData
from hydra import compose, initialize
from hydra.utils import call
from lightning.pytorch import utilities as pl_util
from math import prod
from time import time
from torch import distributed as dist
from typing import Self
import gc
import pytest
import sys
import torch

N = 40
device = "cuda:3"


@pytest.fixture(autouse=True)
def cleanup():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


@pytest.fixture
def datamodule() -> GrapeSplatDataModule:
    with initialize(
        config_path="../../config/grapesplat/data",
        version_base="1.3",
    ):
        return call(
            compose(
                config_name="default",
                overrides=[
                    "loader.num_workers=0",
                    "loader.persistent_workers=null",
                    "loader.prefetch_factor=null",
                    "transform.seq_size=16",
                ],
            ),
            _convert_="all",
        )()


@pytest.fixture(scope="session", autouse=True)
def init():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=30))
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="session", autouse=True)
def seed():
    util.set_seed(902902902)


class TestGrapeSplatLightningModule:
    @dataclass
    class TrainerMock:
        accumulate_grad_batches: int = 4
        estimated_stepping_batches: int = 1000

    @pytest.fixture
    def module_default(self) -> GrapeSplatLightningModule:
        with initialize(
            config_path="../../config/grapesplat/module",
            version_base="1.3",
        ):
            module: GrapeSplatLightningModule = call(
                compose(config_name="gauss_clustered"),
                _convert_="all",
            )()
        module.configure_model()
        module = module.to(device)
        return module

    @pytest.fixture
    def module_enabled(self) -> GrapeSplatLightningModule:
        with initialize(
            config_path="../../config/grapesplat/module",
            version_base="1.3",
        ):
            module: GrapeSplatLightningModule = call(
                compose(config_name="sem_encoded"),
                _convert_="all",
            )()
        module.configure_model()
        module = module.to(device)
        return module

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def test_training_step_default(
        self,
        datamodule: GrapeSplatDataModule,
        module_default: GrapeSplatLightningModule,
    ) -> None:
        module_default.train(True)
        module_default.trainer = self.TrainerMock()
        optimizer = module_default.configure_optimizers()["optimizer"]
        module_default.trainer = None

        datamodule.setup(stage="fit")
        loader = iter(datamodule.train_dataloader())
        data: PipelineData = pl_util.move_data_to_device(
            next(loader),
            device=device,
        )
        optimizer.zero_grad()
        loss = module_default.training_step(data, batch_index=0)
        loss.backward()
        optimizer.step()
        del loss

        print("\n# Training Step (Default)")
        with CudaStats(frame_count=N * prod(data.image.shape[0:2])):
            for i in range(N):
                data: PipelineData = pl_util.move_data_to_device(
                    next(loader),
                    device=device,
                )
                optimizer.zero_grad()
                loss = module_default.training_step(data, batch_index=i + 1)
                loss.backward()
                optimizer.step()

                assert loss.ndim == 0
                assert loss.dtype == torch.float32
                assert not loss.isinf(), "Loss is Inf"
                assert not loss.isnan(), "Loss is NaN"

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def test_training_step_enabled(
        self,
        datamodule: GrapeSplatDataModule,
        module_enabled: GrapeSplatLightningModule,
    ) -> None:
        module_enabled.train(True)
        module_enabled.trainer = self.TrainerMock()
        optimizer = module_enabled.configure_optimizers()["optimizer"]
        module_enabled.trainer = None

        datamodule.setup(stage="fit")
        loader = iter(datamodule.train_dataloader())
        data: PipelineData = pl_util.move_data_to_device(
            next(loader),
            device=device,
        )
        optimizer.zero_grad()
        loss = module_enabled.training_step(data, batch_index=0)
        loss.backward()
        optimizer.step()
        del loss

        print("\n# Training Step (Enabled)")
        with CudaStats(frame_count=N * prod(data.image.shape[0:2])):
            for i in range(N):
                data: PipelineData = pl_util.move_data_to_device(
                    next(loader),
                    device=device,
                )
                optimizer.zero_grad()
                loss = module_enabled.training_step(data, batch_index=i + 1)
                loss.backward()
                optimizer.step()

                assert loss.ndim == 0
                assert loss.dtype == torch.float32
                assert not loss.isinf(), "Loss is Inf"
                assert not loss.isnan(), "Loss is NaN"

    @torch.autocast("cuda", dtype=torch.bfloat16)
    @torch.inference_mode()
    def test_validation_step_default(
        self,
        datamodule: GrapeSplatDataModule,
        module_default: GrapeSplatLightningModule,
    ) -> None:
        module_default.train(False)

        datamodule.setup(stage="validate")
        loader = iter(datamodule.val_dataloader())
        data: PipelineData = pl_util.move_data_to_device(
            next(loader),
            device=device,
        )
        module_default.validation_step(data, batch_index=0)

        print("\n# Validation Step (Default)")
        with CudaStats(frame_count=N * prod(data.image.shape[0:2])):
            for i in range(N):
                data: PipelineData = pl_util.move_data_to_device(
                    next(loader),
                    device=device,
                )
                loss = module_default.validation_step(data, batch_index=i + 1)

                assert loss is None

    @torch.autocast("cuda", dtype=torch.bfloat16)
    @torch.inference_mode()
    def test_validation_step_enabled(
        self,
        datamodule: GrapeSplatDataModule,
        module_enabled: GrapeSplatLightningModule,
    ) -> None:
        module_enabled.train(False)

        datamodule.setup(stage="validate")
        loader = iter(datamodule.val_dataloader())
        data: PipelineData = pl_util.move_data_to_device(
            next(loader),
            device=device,
        )
        module_enabled.validation_step(data, batch_index=0)

        print("\n# Validation Step (Enabled)")
        with CudaStats(frame_count=N * prod(data.image.shape[0:2])):
            for i in range(N):
                data: PipelineData = pl_util.move_data_to_device(
                    next(loader),
                    device=device,
                )
                loss = module_enabled.validation_step(data, batch_index=i + 1)

                assert loss is None


class CudaStats:
    """CUDA memory and timing statistics."""

    def __init__(self, frame_count: int) -> None:
        self.frame_count = int(frame_count)

    def __enter__(self) -> Self:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        self.time = time()
        return self

    def __exit__(self, *_) -> None:
        torch.cuda.synchronize(device)
        f = self.frame_count
        t = time() - self.time
        m = torch.cuda.max_memory_allocated(device)
        print(f"Memory (GiB): {m / 2**30:.3f}")
        print(f"Time   (sec): {t:.3f}")
        print(f"Speed  (fps): {f / t:.3f}")


def main() -> None:
    sys.exit(pytest.main([__file__, "-sv"] + sys.argv[1:]))


if __name__ == "__main__":
    main()
