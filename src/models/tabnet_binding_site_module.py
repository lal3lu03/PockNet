from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanSquaredError, R2Score
from torchmetrics.classification import BinaryJaccardIndex  # IoU metric
from pytorch_tabnet.tab_network import TabNetNoEmbeddings


class TabNetBindingSiteModule(LightningModule):
    """LightningModule for TabNet binding site prediction.

    Uses TabNet to predict binding sites and includes IoU metric for comparison.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        cat_idxs: List[int] = None,
        cat_dims: List[int] = None,
        use_iou_metric: bool = True,
    ):
        """Initialize a TabNet module for binding site prediction.

        Args:
            optimizer: The optimizer to use for training
            scheduler: The learning rate scheduler
            n_d: Width of the decision prediction layer
            n_a: Width of the attention embedding for each step
            n_steps: Number of steps in the architecture
            gamma: Coefficient for feature reusage in the masks
            cat_idxs: List of categorical feature indices
            cat_dims: List of dimensions for each categorical feature
            use_iou_metric: Whether to use IoU as an additional metric
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=True)

        # Get feature dimensions from first batch in setup method
        self.input_dim = None
        
        # TabNet model will be initialized in setup when we know the input dimensions
        self.model = None
        
        # Initialize metrics
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
        
        # Initialize IoU metric if requested (using PyTorch's implementation)
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
            
            # Initialize TabNet model now that we know the input dimensions
            self.model = TabNetNoEmbeddings(
                input_dim=self.input_dim,
                output_dim=1,  # Binding site prediction (single output)
                n_d=self.hparams.n_d,
                n_a=self.hparams.n_a,
                n_steps=self.hparams.n_steps,
                gamma=self.hparams.gamma,
            )
            
            # Move model to the appropriate device (CPU/GPU)
            self.model = self.model.to(self.device)
            
            # Move model's internal buffers and parameters to the same device
            for name, param in self.model.named_parameters():
                param.data = param.data.to(self.device)
            for name, buffer in self.model.named_buffers():
                buffer.data = buffer.data.to(self.device)
                
            # Force initialize and move internal matrices
            if hasattr(self.model.encoder, "init_matrices"):
                self.model.encoder.init_matrices()
            if hasattr(self.model.encoder, "group_attention_matrix"):
                self.model.encoder.group_attention_matrix = self.model.encoder.group_attention_matrix.to(self.device)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass through the model."""
        x = x.to(self.device)
        preds, M_loss = self.model(x)
        # Apply sigmoid to get probabilities
        prob_preds = torch.sigmoid(preds.squeeze(-1))
        return prob_preds, M_loss  # Already squeezed

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x, y = batch
        
        # Forward pass with sigmoid activation
        preds, M_loss = self.forward(x)
        
        # Calculate MSE loss on probabilities
        mse_loss = F.mse_loss(preds, y)
        
        # Calculate IoU loss if enabled
        iou_loss = 0
        if self.hparams.use_iou_metric:
            # Convert to binary predictions using probability threshold
            binary_preds = (preds > 0.5).float()
            binary_targets = (y > 0.5).float()
            
            # Update metric and let Lightning handle aggregation
            self.train_iou(binary_preds, binary_targets)
            self.log("train/iou", self.train_iou, on_step=True, on_epoch=True, prog_bar=True)
            
            # Use 1 - IoU as a loss component
            iou_loss = 1 - self.train_iou.compute()
            self.log("train/iou_loss", iou_loss, on_step=True, on_epoch=True)
        
        # Combine losses with weights - 0.5 for MSE, 0.4 for IoU, 0.1 for M_loss
        loss = 0.5 * mse_loss + 0.4 * iou_loss + 0.1 * M_loss
        
        # Log other metrics
        self.train_mse(preds, y)
        self.train_r2(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True)
        self.log("train/r2", self.train_r2, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Validation step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
        """
        x, y = batch
        
        # Forward pass with sigmoid activation
        preds, _ = self.forward(x)
        
        # Calculate MSE loss on probabilities
        loss = F.mse_loss(preds, y)
        
        # Update metrics
        self.val_mse(preds, y)
        self.val_r2(preds, y)
        
        # Calculate IoU if enabled
        if self.hparams.use_iou_metric:
            # Convert to binary predictions using probability threshold
            binary_preds = (preds > 0.5).float()
            binary_targets = (y > 0.5).float()
            # Update metric and let Lightning handle aggregation
            self.val_iou(binary_preds, binary_targets)
            self.log("val/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Test step for evaluating on the bu48 dataset.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
        """
        x, y = batch
        
        # Forward pass with sigmoid activation
        preds, _ = self.forward(x)
        
        # Calculate MSE loss on probabilities
        loss = F.mse_loss(preds, y)
        
        # Update metrics
        self.test_mse(preds, y)
        self.test_r2(preds, y)
        
        # Calculate IoU if enabled
        if self.hparams.use_iou_metric:
            # Convert to binary predictions using probability threshold
            binary_preds = (preds > 0.5).float()
            binary_targets = (y > 0.5).float()
            # Update metric and let Lightning handle aggregation
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