from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from hopular.loss import ClusteringLoss
from hopular.model import HOPField
from lightning import LightningModule
from torchmetrics import MeanSquaredError, R2Score
from torchmetrics.classification import BinaryJaccardIndex  # IoU metric


class HopularBindingSiteModule(LightningModule):
    """LightningModule for Hopular binding site prediction.
    Uses Hopular with higher-order polynomials to predict binding sites.
    Includes IoU metric for comparison.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        hidden_dim: int = 128,
        hopfield_dim: int = 64,
        heads: int = 4,
        scaling: float = None,
        dropout: float = 0.1,
        layers: int = 3,
        pattern_dim: int = 32,
        update_steps: int = 1,
        polynomial_degree: int = 2,
        use_iou_metric: bool = True,
    ):
        """Initialize a Hopular module for binding site prediction.
        Args:
            optimizer: The optimizer to use for training
            scheduler: The learning rate scheduler
            hidden_dim: Width of the hidden representation
            hopfield_dim: Width of the Hopfield space
            heads: Number of attention heads
            scaling: Scaling factor (default: 1/sqrt(d))
            dropout: Dropout probability
            layers: Number of Hopular layers
            pattern_dim: Dimension for stored patterns
            update_steps: Number of Hopfield update steps
            polynomial_degree: Degree of the polynomial for higher-order patterns
            use_iou_metric: Whether to use IoU as an additional metric
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=True)
        # Get feature dimensions from first batch in setup method
        self.input_dim = None

        # Hopular model will be initialized in setup when we know the input dimensions
        self.model = None

        # Initialize clustering loss for Hopular
        self.clustering_loss = ClusteringLoss()

        # Initialize metrics
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

        # Initialize IoU metric if requested
        if self.hparams.use_iou_metric:
            # Using PyTorch's BinaryJaccardIndex (IoU) implementation
            self.train_iou = BinaryJaccardIndex()
            self.val_iou = BinaryJaccardIndex()
            self.test_iou = BinaryJaccardIndex()

    def setup(self, stage: str) -> None:
        """Setup model with proper input dimensions when data is available."""
        if stage == "fit" and self.model is None:
            # Get a batch from the train dataloader to infer input shape
            batch = next(iter(self.trainer.datamodule.train_dataloader()))
            x, _ = batch
            self.input_dim = x.shape[1]

            # Initialize Hopular model now that we know the input dimensions
            self.model = HOPField(
                input_dim=self.input_dim,
                output_dim=1,  # Binary prediction
                hidden_dim=self.hparams.hidden_dim,
                hopfield_dim=self.hparams.hopfield_dim,
                heads=self.hparams.heads,
                scaling=self.hparams.scaling,
                dropout=self.hparams.dropout,
                layers=self.hparams.layers,
                pattern_dim=self.hparams.pattern_dim,
                update_steps=self.hparams.update_steps,
                polynomial_degree=self.hparams.polynomial_degree,
            )

            # Move model to the appropriate device (CPU/GPU)
            self.model = self.model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the model."""
        x = x.to(self.device)
        outputs = self.model(x)

        # Apply sigmoid to get probabilities
        preds = torch.sigmoid(outputs.squeeze(-1))
        return preds  # Output probabilities

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        x, y = batch
        preds = self.forward(x)

        # Primary loss - MSE
        mse_loss = F.mse_loss(preds, y)

        # Clustering loss for Hopular
        cl_loss = self.clustering_loss(self.model.hopfields[-1])

        # Calculate IoU loss if enabled
        iou_loss = 0.0
        if self.hparams.use_iou_metric:
            # Convert to binary predictions for IoU
            binary_preds = (preds > 0.5).float()
            binary_targets = (y > 0.5).float()

            # Update IoU metric
            self.train_iou(binary_preds, binary_targets)
            self.log("train/iou", self.train_iou, on_step=True, on_epoch=True, prog_bar=True)

            # Use 1 - IoU as a loss component
            iou_loss = 1 - self.train_iou.compute()
            self.log("train/iou_loss", iou_loss, on_step=True, on_epoch=True)

        # Combine losses with weights - 0.5 for MSE, 0.4 for IoU, 0.1 for clustering loss
        loss = 0.5 * mse_loss + 0.4 * iou_loss + 0.1 * cl_loss

        # Log other metrics
        self.train_mse(preds, y)
        self.train_r2(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True)
        self.log("train/r2", self.train_r2, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        x, y = batch
        preds = self.forward(x)

        # Calculate loss
        loss = F.mse_loss(preds, y)

        # Update metrics
        self.val_mse(preds, y)
        self.val_r2(preds, y)

        # Calculate IoU if enabled
        if self.hparams.use_iou_metric:
            binary_preds = (preds > 0.5).float()
            binary_targets = (y > 0.5).float()

            # Update IoU metric
            self.val_iou(binary_preds, binary_targets)
            self.log("val/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        x, y = batch
        preds = self.forward(x)

        # Calculate loss
        loss = F.mse_loss(preds, y)

        # Update metrics
        self.test_mse(preds, y)
        self.test_r2(preds, y)

        # Calculate IoU if enabled
        if self.hparams.use_iou_metric:
            binary_preds = (preds > 0.5).float()
            binary_targets = (y > 0.5).float()

            # Update IoU metric
            self.test_iou(binary_preds, binary_targets)
            self.log("test/iou", self.test_iou, on_step=False, on_epoch=True, prog_bar=True)

        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True)
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = self.hparams.optimizer(params=self.parameters())

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
