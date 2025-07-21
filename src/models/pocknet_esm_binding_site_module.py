from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanSquaredError, R2Score, Accuracy
from torchmetrics.classification import BinaryJaccardIndex

from src.models.components.pocknet_esm import PockNetESM


class PockNetESMBindingSiteModule(LightningModule):
    """PockNet model enhanced with ESM-2 embeddings for protein binding site prediction.
    
    This module extends the base PockNet architecture to incorporate ESM-2 protein embeddings
    along with the traditional tabular features for improved binding site prediction performance.
    """
    
    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tabular_dim: int = 42,  # Original tabular features
        esm_dim: int = 2560,   # ESM-2 embedding dimensions
        output_dim: int = 1,
        n_steps: int = 3,
        n_d: int = 8,
        n_a: int = 8,
        n_shared: int = 2,
        n_independent: int = 2,
        gamma: float = 1.3,
        epsilon: float = 1e-15,
        dropout: float = 0.0,
        esm_projection_dim: int = 128,  # Dimension to project ESM embeddings to
        fusion_strategy: str = "concatenate",  # Options: "concatenate", "attention", "gated"
        use_iou_metric: bool = True,
    ):
        """Initialize PockNetESMBindingSiteModule.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            tabular_dim: Dimension of tabular features
            esm_dim: Dimension of ESM embeddings
            output_dim: Output dimension (1 for binary classification)
            n_steps: Number of TabNet decision steps
            n_d: Dimension of decision features
            n_a: Dimension of attention features
            n_shared: Number of shared layers in feature transformer
            n_independent: Number of independent layers per step
            gamma: Coefficient for attention regularization
            epsilon: Small constant for numerical stability
            dropout: Dropout rate
            esm_projection_dim: Dimension to project ESM embeddings to
            fusion_strategy: How to combine tabular and ESM features
            use_iou_metric: Whether to compute IoU metric
        """
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.model = None

        # metrics
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
        
        # Accuracy metrics for binary classification
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        if use_iou_metric:
            self.train_iou = BinaryJaccardIndex()
            self.val_iou = BinaryJaccardIndex()
            self.test_iou = BinaryJaccardIndex()

    def setup(self, stage: str) -> None:
        """Setup model with proper input dimensions when data is available."""
        if self.model is None:
            # Calculate total input dimension based on fusion strategy
            if self.hparams.fusion_strategy == "concatenate":
                total_input_dim = self.hparams.tabular_dim + self.hparams.esm_projection_dim
            else:
                # For attention and gated fusion, we'll use the projected ESM dimension
                total_input_dim = max(self.hparams.tabular_dim, self.hparams.esm_projection_dim)
            
            self.model = PockNetESM(
                tabular_dim=self.hparams.tabular_dim,
                esm_dim=self.hparams.esm_dim,
                output_dim=self.hparams.output_dim,
                n_steps=self.hparams.n_steps,
                n_d=self.hparams.n_d,
                n_a=self.hparams.n_a,
                n_shared=self.hparams.n_shared,
                n_independent=self.hparams.n_independent,
                gamma=self.hparams.gamma,
                epsilon=self.hparams.epsilon,
                dropout=self.hparams.dropout,
                esm_projection_dim=self.hparams.esm_projection_dim,
                fusion_strategy=self.hparams.fusion_strategy,
            ).to(self.device)

    def forward(self, tabular_features: torch.Tensor, esm_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the ESM-augmented PockNet model.
        
        Args:
            tabular_features: Tabular input features [batch_size, tabular_dim]
            esm_features: ESM embeddings [batch_size, esm_dim]
            
        Returns:
            logits: Raw predictions [batch_size, output_dim]
            attention_weights: Attention weights for interpretability
        """
        # Forward through ESM-augmented PockNet
        logits, attention_weights = self.model(tabular_features, esm_features)
        return logits.squeeze(-1), attention_weights

    def model_step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step (forward pass + loss calculation)."""
        tabular_x, esm_x, y = batch
        logits, _ = self.forward(tabular_x, esm_x)
        logits = logits.view(-1)
        y_float = y.float().view(-1)

        # Use BCE with logits for binary classification
        loss = F.binary_cross_entropy_with_logits(logits, y_float)

        # Convert to probabilities for metrics
        preds = torch.sigmoid(logits)
        return loss, preds, y_float

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, preds, targets = self.model_step(batch)
        
        # Update metrics
        self.train_mse(preds, targets)
        self.train_r2(preds, targets)
        
        # Update accuracy metric (convert continuous predictions to binary)
        binary_preds = (preds > 0.5).float()
        binary_targets = (targets > 0.5).float()
        self.train_acc(binary_preds, binary_targets)
        
        if hasattr(self, 'train_iou'):
            bpred = (preds > 0.5).float()
            btrue = (targets > 0.5).float()
            self.train_iou(bpred, btrue)
            self.log("train/iou", self.train_iou, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log metrics
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mse", self.train_mse, on_epoch=True, sync_dist=True)
        self.log("train/r2", self.train_r2, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Validation step."""
        loss, preds, targets = self.model_step(batch)
        
        # Update metrics
        self.val_mse(preds, targets)
        self.val_r2(preds, targets)
        
        # Update accuracy metric (convert continuous predictions to binary)
        binary_preds = (preds > 0.5).float()
        binary_targets = (targets > 0.5).float()
        self.val_acc(binary_preds, binary_targets)
        
        if hasattr(self, 'val_iou'):
            bpred = (preds > 0.5).float()
            btrue = (targets > 0.5).float()
            self.val_iou(bpred, btrue)
            self.log("val/iou", self.val_iou, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log metrics
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mse", self.val_mse, on_epoch=True, sync_dist=True)
        self.log("val/r2", self.val_r2, on_epoch=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Test step."""
        loss, preds, targets = self.model_step(batch)
        
        # Update metrics
        self.test_mse(preds, targets)
        self.test_r2(preds, targets)
        
        # Update accuracy metric (convert continuous predictions to binary)
        binary_preds = (preds > 0.5).float()
        binary_targets = (targets > 0.5).float()
        self.test_acc(binary_preds, binary_targets)
        
        if hasattr(self, 'test_iou'):
            bpred = (preds > 0.5).float()
            btrue = (targets > 0.5).float()
            self.test_iou(bpred, btrue)
            self.log("test/iou", self.test_iou, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log metrics
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)
        self.log("test/acc", self.test_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/mse", self.test_mse, on_epoch=True, sync_dist=True)
        self.log("test/r2", self.test_r2, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = self.hparams.optimizer(self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
