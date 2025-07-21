from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanSquaredError, R2Score
from torchmetrics.classification import BinaryJaccardIndex

from src.models.components.pocknet import PockNet


class PockNetBindingSiteModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        input_dim: int = 48,
        output_dim: int = 1,
        n_steps: int = 3,
        n_d: int = 8,
        n_a: int = 8,
        n_shared: int = 2,
        n_independent: int = 2,
        gamma: float = 1.3,
        epsilon: float = 1e-15,
        dropout: float = 0.0,
        use_iou_metric: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.model = None

        # metrics
        self.train_mse = MeanSquaredError()
        self.val_mse   = MeanSquaredError()
        self.test_mse  = MeanSquaredError()
        self.train_r2  = R2Score()
        self.val_r2    = R2Score()
        self.test_r2   = R2Score()

        if use_iou_metric:
            self.train_iou = BinaryJaccardIndex()
            self.val_iou   = BinaryJaccardIndex()
            self.test_iou  = BinaryJaccardIndex()

    def setup(self, stage: str) -> None:
        if self.model is None:
            # infer input_dim
            if hasattr(self.trainer.datamodule, 'dims') and self.trainer.datamodule.dims is not None:
                self.hparams.input_dim = self.trainer.datamodule.dims
            self.model = PockNet(
                input_dim      = self.hparams.input_dim,
                output_dim     = self.hparams.output_dim,
                n_steps        = self.hparams.n_steps,
                n_d            = self.hparams.n_d,
                n_a            = self.hparams.n_a,
                n_shared       = self.hparams.n_shared,
                n_independent  = self.hparams.n_independent,
                gamma          = self.hparams.gamma,
                epsilon        = self.hparams.epsilon,
                dropout        = self.hparams.dropout,
            ).to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # return raw logits and attention
        logits, attention_weights = self.model(x)
        return logits.squeeze(-1), attention_weights

    def model_step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits, _ = self.forward(x)
        logits = logits.view(-1)
        y_float = y.float().view(-1)

        # use BCE with logits
        loss = F.binary_cross_entropy_with_logits(logits, y_float)

        # for metrics and logging, convert to probabilities
        preds = torch.sigmoid(logits)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_mse(preds, targets)
        self.train_r2(preds, targets)
        if hasattr(self, 'train_iou'):
            bpred = (preds > 0.5).float()
            btrue = (targets > 0.5).float()
            self.train_iou(bpred, btrue)
            self.log("train/iou", self.train_iou, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mse", self.train_mse, on_epoch=True, sync_dist=True)
        self.log("train/r2", self.train_r2, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_mse(preds, targets)
        self.val_r2(preds, targets)
        if hasattr(self, 'val_iou'):
            bpred = (preds > 0.5).float()
            btrue = (targets > 0.5).float()
            self.val_iou(bpred, btrue)
            self.log("val/iou", self.val_iou, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mse", self.val_mse, on_epoch=True, sync_dist=True)
        self.log("val/r2", self.val_r2, on_epoch=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_mse(preds, targets)
        self.test_r2(preds, targets)
        if hasattr(self, 'test_iou'):
            bpred = (preds > 0.5).float()
            btrue = (targets > 0.5).float()
            self.test_iou(bpred, btrue)
            self.log("test/iou", self.test_iou, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)
        self.log("test/mse", self.test_mse, on_epoch=True, sync_dist=True)
        self.log("test/r2", self.test_r2, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor":   "val/loss",
                    "interval":  "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
