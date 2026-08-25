"""GrapeSplat LightningModule for pipeline orchestration."""

from ..util import InferenceProfiler, logger
from . import typing as Tp
from .loss import CompositeLossBuilder
from .metric import CompositeMetricBuilder
from .optimizer import CompositeOptimizerBuilder
from .pipeline import PipelineBuilder
from .scheduler import CompositeSchedulerBuilder
from .visualizer import CompositeVisualizerBuilder
from dataclasses import dataclass
from lightning import pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRSchedulerConfig
from torch import Tensor
import torch
import wandb


@dataclass
class GrapeSplatLightningModuleBuilder:
    """GrapeSplat LightningModule for pipeline orchestration."""

    loss: CompositeLossBuilder
    metric: CompositeMetricBuilder
    optimizer: CompositeOptimizerBuilder
    pipeline: PipelineBuilder
    scheduler: CompositeSchedulerBuilder
    visualizer: CompositeVisualizerBuilder

    def __call__(self) -> "GrapeSplatLightningModule":
        """Build the module."""

        return GrapeSplatLightningModule(config=self)


class GrapeSplatLightningModule(pl.LightningModule):
    """GrapeSplat LightningModule for pipeline orchestration."""

    def __init__(self, config: GrapeSplatLightningModuleBuilder) -> None:
        super().__init__()

        self.config = config
        self.flops_per_batch = None

    def configure_model(self) -> None:
        self.loss = self.config.loss()
        self.metric = self.config.metric()
        self.pipeline = self.config.pipeline()
        self.visualizer = self.config.visualizer(
            base=self.config.pipeline.depth2_image_decoder,
        )
        self.train()

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = self.config.optimizer(self.pipeline)
        scheduler = self.config.scheduler(
            iters=self.trainer.estimated_stepping_batches,
            optimizer=optimizer,
        )
        return OptimizerLRSchedulerConfig(
            lr_scheduler={"interval": "step", "scheduler": scheduler},
            optimizer=optimizer,
        )

    def get_inp_from_data(self, data: Tp.PipelineData) -> Tp.PipelineInp:
        return Tp.PipelineInp(
            cam_extr=data.cam_extr,
            cam_intr=data.cam_intr,
            image=data.image,
            mask_rec=data.mask_rec,
            mask_ren=data.mask_ren,
        )

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, *_, **__) -> None:
        scheduler.step()

    def on_test_epoch_start(self) -> None:
        with self.trainer.strategy.precision_plugin.test_step_context():
            data: Tp.PipelineData = self.transfer_batch_to_device(
                batch=next(iter(self.trainer.test_dataloaders)),
                device=self.device,
                dataloader_idx=0,
            )
            self.predict_step(data.image[:, data.mask_rec])
        torch.cuda.synchronize(self.device)

    def predict_step(self, image: Tensor, *_, **__) -> Tp.PipelineOut:
        # (B, S, 3, H, W)
        return self.pipeline.reconstruct(image)

    def test_step(self, data: Tp.PipelineData, batch_index: int, *_, **__) -> None:
        with InferenceProfiler(device=self.device) as prof:
            pred = self.predict_step(data.image[:, data.mask_rec])
        pred = self.pipeline.render(self.get_inp_from_data(data), pred)
        metric: Tp.CompositeMetricOut = self.metric(pred, data)

        if not metric or self._trainer is None:
            return

        metric |= prof.metric
        self.log_dict(
            {f"test/metric_{k}": v for k, v in sorted(metric.items())},
            batch_size=data.get_batch_size(),
            sync_dist=True,
        )

        is_vis_step = (
            self.trainer.is_global_zero
            and self.config.visualizer.step_size > 0
            and batch_index % self.config.visualizer.step_size == 0
        )
        if is_vis_step and isinstance(self.logger, WandbLogger):
            self.logger.log_metrics(
                {"test/vis_seq_name": wandb.Html("<br>".join(data.name))}
                | {
                    f"test/vis_{name}": [wandb.Image(i) for i in image]
                    for name, image in self.visualizer(pred, data).items()
                    if image
                },
                step=self.global_step + batch_index,
            )

    def training_step(self, data: Tp.PipelineData, batch_index: int, *_, **__) -> Tensor:
        pred = self.predict_step(data.image[:, data.mask_rec])
        loss: Tp.CompositeLossOut = self.loss(pred, data)

        if not loss["loss"].isfinite().item():
            logger.error(
                f"Detected non-finite loss at b={batch_index} and n={data.name}: {loss['loss'].item()}"
            )
            loss["loss"] = loss["loss"].nan_to_num(0.0, 0.0, 0.0)

        if self._trainer is None:
            return loss["loss"]

        self.log_dict(
            {f"train/{k}": v for k, v in sorted(loss.items())},
            batch_size=data.get_batch_size(),
        )

        return loss["loss"]

    def validation_step(self, data: Tp.PipelineData, batch_index: int, *_, **__) -> None:
        pred = self.predict_step(data.image[:, data.mask_rec])
        metric: Tp.CompositeMetricOut = self.metric(pred, data)

        if not metric or self._trainer is None:
            return

        self.log_dict(
            {f"validate/metric_{k}": v for k, v in sorted(metric.items())},
            batch_size=data.get_batch_size(),
            sync_dist=True,
        )

        is_vis_step = (
            self.trainer.is_global_zero
            and self.config.visualizer.step_size > 0
            and batch_index % self.config.visualizer.step_size == 0
        )
        if is_vis_step and isinstance(self.logger, WandbLogger):
            self.logger.log_metrics(
                {"validate/vis_seq_name": wandb.Html("<br>".join(data.name))}
                | {
                    f"validate/vis_{name}": [wandb.Image(i) for i in image]
                    for name, image in self.visualizer(pred, data).items()
                    if image
                },
                step=self.global_step + batch_index,
            )
