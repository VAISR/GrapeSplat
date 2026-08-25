"""Integration test for GrapeSplat data layer: data module."""

from grapesplat import util
from grapesplat.data import GrapeSplatDataModule, PipelineData
from hydra import compose, initialize
from hydra.utils import call
from torch import distributed as dist
import os
import pytest
import sys
import torch


@pytest.fixture
def datamodule() -> GrapeSplatDataModule:
    with initialize(
        config_path="../../config/grapesplat/data",
        version_base="1.3",
    ):
        return call(
            compose(
                config_name="dl3dv",
                overrides=[
                    "loader.num_workers=0",
                    "loader.persistent_workers=null",
                    "loader.prefetch_factor=null",
                ],
            ),
            _convert_="all",
        )()


@pytest.fixture(scope="session", autouse=True)
def distributed():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="session", autouse=True)
def seed():
    util.set_seed(902902902)


def assert_data(data: PipelineData, datamodule: GrapeSplatDataModule) -> None:
    B = datamodule.config.loader["batch_size"]
    S = datamodule.config.transform.seq_size
    H = datamodule.config.transform.frame_resol
    W = datamodule.config.transform.frame_resol

    assert data.get_batch_size() == B
    assert data.get_seq_size() == S

    assert data.depth.shape == (B, S, 1, H, W)
    assert data.image.shape == (B, S, 3, H, W)
    assert data.mask_depth.shape == (B, S, 1, H, W)
    assert data.mask_image.shape == (B, S, 3, H, W)
    assert len(data.name) == B
    assert data.point.shape == (B, S, 3, H, W)
    assert data.pose_extr.shape == (B, S, 3, 4)
    assert data.pose_intr.shape == (B, S, 3, 3)

    assert data.depth.dtype == torch.float32
    assert data.image.dtype == torch.float32
    assert data.mask_depth.dtype == torch.bool
    assert data.mask_image.dtype == torch.bool
    assert all(isinstance(name, str) for name in data.name)
    assert data.point.dtype == torch.float32
    assert data.pose_extr.dtype == torch.float32
    assert data.pose_intr.dtype == torch.float32

    if data.mask_depth.any():
        assert data.depth[data.mask_depth].min() > 0.0
    assert data.image.min() >= 0.0
    assert data.image.max() <= 1.0


class TestGrapeSplatDataModule:
    def test_setup_fit(self, datamodule: GrapeSplatDataModule) -> None:
        datamodule.setup(stage="fit")
        assert datamodule.dataset_train is not None
        assert datamodule.dataset_val is not None
        assert len(datamodule.dataset_train) == sum(
            d.filter.split_size_train * d.weight_train for d in datamodule.config.dataset
        )
        assert len(datamodule.dataset_val) == sum(
            d.filter.split_size_val for d in datamodule.config.dataset
        )

    def test_setup_test(self, datamodule: GrapeSplatDataModule) -> None:
        datamodule.setup(stage="test")
        assert datamodule.dataset_test is not None
        assert len(datamodule.dataset_test) == sum(
            d.filter.split_size_test * d.weight_test for d in datamodule.config.dataset
        )

    def test_train_data(self, datamodule: GrapeSplatDataModule) -> None:
        datamodule.setup(stage="fit")
        data = next(iter(datamodule.train_dataloader()))
        assert_data(data, datamodule)

    def test_val_data(self, datamodule: GrapeSplatDataModule) -> None:
        datamodule.setup(stage="fit")
        for data in datamodule.val_dataloader():
            assert_data(data, datamodule)

    def test_test_data(self, datamodule: GrapeSplatDataModule) -> None:
        datamodule.setup(stage="test")
        for data in datamodule.test_dataloader():
            assert_data(data, datamodule)


def main() -> None:
    sys.exit(pytest.main([__file__, "-sv"] + sys.argv[1:]))


if __name__ == "__main__":
    main()
