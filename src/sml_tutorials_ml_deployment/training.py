from pathlib import Path
from typing import Any, Callable

import pytorch_lightning as pl
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import Dice, JaccardIndex
from typing_extensions import override

from .utils import add_prefix_to_dict_keys


def make_segmentation_metrics_collection(num_classes: int) -> MetricCollection:
    task = "multiclass" if num_classes > 2 else "binary"
    multiclass = num_classes > 2
    return MetricCollection(
        {
            "jaccard_micro": JaccardIndex(task=task, num_classes=num_classes, average="micro"),
            "dice_micro": Dice(multiclass=multiclass, num_classes=num_classes, average="micro"),
            "jaccard_macro": JaccardIndex(task=task, num_classes=num_classes, average="macro"),
            "dice_macro": Dice(multiclass=multiclass, num_classes=num_classes, average="macro"),
        }
    )


class SegmentationTask(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: Callable[[torch.Tensor], torch.Tensor] | None = None,
        metrics: MetricCollection | None = None,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.loss = loss if loss is not None else torch.nn.CrossEntropyLoss()
        self.metrics = metrics if metrics is not None else MetricCollection({})
        self.lr = lr

    @override
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @override
    def training_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        pred = self.model(batch["image"])
        loss = self.loss(pred, batch["mask"])
        self.log("train_loss", loss)
        return loss

    @override
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        pred = self.model(batch["image"])
        loss = self.loss(pred, batch["mask"])
        metrics = {"loss": loss} | self.metrics(pred, batch["mask"])
        self.log_dict(add_prefix_to_dict_keys(metrics, "val_"))
        return pred

    @override
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        pred = self.model(batch["image"])
        loss = self.loss(pred, batch["mask"])
        metrics = {"loss": loss} | self.metrics(pred, batch["mask"])
        self.log_dict(add_prefix_to_dict_keys(metrics, "test_"))
        return pred

    @staticmethod
    def load_model_state_dict_from_checkpoint(ckpt_path: Path) -> dict:
        checkpoint = torch.load(ckpt_path)
        print(checkpoint["state_dict"])
        return checkpoint["state_dict"]
