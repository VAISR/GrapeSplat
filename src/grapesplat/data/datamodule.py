"""GrapeSplat LightningDataModule for pipeline orchestration."""

from .dataset import DatasetBuilder
from .transform import TransformBuilder, collate_seqs
from dataclasses import dataclass, replace
from grapesplat import util
from lightning import pytorch as pl
from torch.utils.data import ConcatDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


@dataclass
class GrapeSplatDataModuleBuilder:
    """GrapeSplat LightningDataModule for pipeline orchestration."""

    dataset: list[DatasetBuilder]
    loader: dict[str]
    shared: bool
    transform: TransformBuilder

    def __call__(self) -> "GrapeSplatDataModule":
        """Build the module."""

        return GrapeSplatDataModule(config=self)


class GrapeSplatDataModule(pl.LightningDataModule):
    """GrapeSplat LightningDataModule for pipeline orchestration."""

    def __init__(self, config: GrapeSplatDataModuleBuilder) -> None:
        super().__init__()

        self.config = config
        self.dataset_test: ConcatDataset | None = None
        self.dataset_train: ConcatDataset | None = None
        self.dataset_val: ConcatDataset | None = None
        self.prepare_data_per_node = not config.shared

    def prepare_data(self) -> None:
        ConcatDataset(
            [
                dataset(stage="val", trans=replace(self.config.transform, random=False))
                for dataset in self.config.dataset
            ]
        )

    def setup(self, stage: str) -> None:
        stages = ("fit", "test", "validate")

        if stage not in stages:
            raise TypeError(f"Unsupported datamodule stage: expected one of {stages}, got {stage}")

        if stage == stages[0]:
            self.dataset_train = ConcatDataset(
                [
                    dataset(stage="train", trans=replace(self.config.transform, random=True))
                    for dataset in self.config.dataset
                    for _ in range(dataset.weight_train)
                ]
            )
        if stage == stages[1]:
            self.dataset_test = ConcatDataset(
                [
                    dataset(stage="test", trans=replace(self.config.transform, random=False))
                    for dataset in self.config.dataset
                    for _ in range(dataset.weight_test)
                ]
            )
        if stage in stages[::2]:
            self.dataset_val = ConcatDataset(
                [
                    dataset(stage="val", trans=replace(self.config.transform, random=False))
                    for dataset in self.config.dataset
                ]
            )

    def test_dataloader(self) -> StatefulDataLoader:
        args = self.config.loader.copy()
        args.update(
            collate_fn=collate_seqs,
            num_workers=util.get_worker_count(),
            sampler=StatefulDistributedSampler(
                self.dataset_test,
                drop_last=self.config.loader["drop_last"],
                seed=util.get_seed(),
                shuffle=False,
            ),
        )
        return StatefulDataLoader(self.dataset_test, **args)

    def train_dataloader(self) -> StatefulDataLoader:
        args = self.config.loader.copy()
        args.update(
            collate_fn=collate_seqs,
            num_workers=util.get_worker_count(),
            sampler=StatefulDistributedSampler(
                self.dataset_train,
                drop_last=self.config.loader["drop_last"],
                seed=util.get_seed(),
                shuffle=True,
            ),
        )
        return StatefulDataLoader(self.dataset_train, **args)

    def val_dataloader(self) -> StatefulDataLoader:
        args = self.config.loader.copy()
        args.update(
            collate_fn=collate_seqs,
            num_workers=util.get_worker_count(),
            sampler=StatefulDistributedSampler(
                self.dataset_val,
                drop_last=self.config.loader["drop_last"],
                seed=util.get_seed(),
                shuffle=False,
            ),
        )
        return StatefulDataLoader(self.dataset_val, **args)
