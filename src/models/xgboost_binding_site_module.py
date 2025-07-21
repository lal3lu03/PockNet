from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from lightning import LightningModule
from sklearn.preprocessing import StandardScaler
from torchmetrics import MeanSquaredError, R2Score
from torchmetrics.classification import BinaryJaccardIndex  # IoU metric


class XGBoostBindingSiteModule(LightningModule):
    """LightningModule for XGBoost binding site prediction.
    Uses XGBoost for binding site prediction, wrapped in PyTorch Lightning
    Includes IoU metric for comparison.
    """

    def __init__(
        self,
        optimizer,
        scheduler,
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        use_iou_metric=True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False

        # Initialize metrics
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

        if self.hparams.use_iou_metric:
            self.train_iou = BinaryJaccardIndex()
            self.val_iou = BinaryJaccardIndex()
            self.test_iou = BinaryJaccardIndex()

    def setup(self, stage: str) -> None:
        """Setup model when data is available."""
        if stage == "fit" and self.model is None:
            # Initialize XGBoost model
            self.model = xgb.XGBClassifier(
                learning_rate=self.hparams.learning_rate,
                max_depth=self.hparams.max_depth,
                n_estimators=self.hparams.n_estimators,
                subsample=self.hparams.subsample,
                colsample_bytree=self.hparams.colsample_bytree,
                gamma=self.hparams.gamma,
                reg_alpha=self.hparams.reg_alpha,
                reg_lambda=self.hparams.reg_lambda,
                tree_method="hist",
                device="cuda" if torch.cuda.is_available() else "cpu",
                objective="binary:logistic",
                eval_metric="logloss",
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the model.

        For XGBoost, we need to convert to numpy arrays
        """
        x_numpy = x.cpu().numpy()

        if not self.fitted:
            # If model is not fitted yet, return zeros
            return torch.zeros(x.shape[0], device=x.device)

        # Scale the features
        x_scaled = self.scaler.transform(x_numpy)

        # Predict probabilities
        preds = self.model.predict_proba(x_scaled)[:, 1]  # Get positive class probability

        # Convert back to tensor
        return torch.tensor(preds, device=x.device, dtype=torch.float32)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step.

        For XGBoost, we collect the data and fit the model after all batches
        are processed in the on_train_epoch_end method.
        """
        x, y = batch

        # Store batch data for later training
        if not hasattr(self, "train_features"):
            self.train_features = []
            self.train_labels = []

        self.train_features.append(x.cpu().numpy())
        self.train_labels.append(y.cpu().numpy())

        # Return dummy loss - actual training happens in on_train_epoch_end
        return torch.tensor(0.0, requires_grad=True)

    def on_train_epoch_end(self) -> None:
        """Fit XGBoost model at the end of the training epoch."""
        if not hasattr(self, "train_features") or not self.train_features:
            return

        # Concatenate all batches
        X = np.vstack(self.train_features)
        y = np.hstack(self.train_labels)

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)

        # Convert labels to binary (0 or 1) for classification
        y_binary = (y >= 0.5).astype(int)

        # Fit the model
        self.model.fit(X_scaled, y_binary)
        self.fitted = True

        # Clear stored data
        self.train_features = []
        self.train_labels = []

        # Make predictions on training data for metrics
        y_pred = self.model.predict_proba(X_scaled)[:, 1]

        # Move predictions to GPU and calculate metrics
        y_pred_tensor = torch.tensor(y_pred, device=self.device)
        y_tensor = torch.tensor(y, device=self.device)

        # Calculate metrics
        self.train_mse(y_pred_tensor, y_tensor)
        self.train_r2(y_pred_tensor, y_tensor)

        # Log metrics
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True)
        self.log("train/r2", self.train_r2, on_step=False, on_epoch=True)

        # Calculate IoU if enabled
        if self.hparams.use_iou_metric:
            # Convert to binary predictions - ensure on GPU
            binary_preds = (y_pred_tensor > 0.5).float()
            binary_targets = (y_tensor > 0.5).float()

            # Update IoU metric
            self.train_iou(binary_preds, binary_targets)
            self.log("train/iou", self.train_iou, on_step=False, on_epoch=True)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Input batch
            batch_idx: Batch index
        """
        x, y = batch

        # Skip if model is not fitted yet
        if not self.fitted:
            return

        # Forward pass using XGBoost
        preds = self.forward(x)

        # Move tensors to GPU
        preds = preds.to(self.device)
        y = y.to(self.device)

        # Calculate MSE loss
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
            self.log("val/iou", self.val_iou, on_step=False, on_epoch=True)

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation to compute final metrics."""
        if self.hparams.use_iou_metric:
            # Compute and log final IoU value
            final_iou = self.val_iou.compute()
            self.log("val/iou", final_iou, on_epoch=True)
            # Reset the metric for next epoch
            self.val_iou.reset()

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Test step for evaluating on the bu48 dataset.

        Args:
            batch: Input batch
            batch_idx: Batch index
        """
        x, y = batch

        # Skip if model is not fitted yet
        if not self.fitted:
            return

        # Forward pass using XGBoost
        preds = self.forward(x)

        # Calculate MSE loss
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
            self.log("test/iou", self.test_iou, on_step=False, on_epoch=True)

        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True)
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict:
        """Configure optimizers and learning rate schedulers.

        For XGBoost, we don't use PyTorch optimizers, but we return a dummy optimizer
        to keep the Lightning workflow intact.

        Returns:
            Dictionary with optimizer configuration
        """
        # Return a dummy optimizer since XGBoost handles optimization internally
        return {"optimizer": self.hparams.optimizer(params=[torch.nn.Parameter(torch.zeros(1))])}
