#!/usr/bin/env python3
"""Simple test to verify scheduler configuration without full data loading."""

import sys
from pathlib import Path
import torch
import lightning.pytorch as L

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_scheduler_config():
    """Test scheduler configuration in isolation."""
    print("Testing scheduler configuration...")
    
    # Test the scheduler instantiation as it would be done in the model
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    # Create a simple model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    
    # Create scheduler as configured in the model
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    print("✓ Scheduler created successfully")
    
    # Test PyTorch Lightning configuration format
    optimizer_config = {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": 1,
        },
    }
    
    print("✓ Lightning scheduler config created successfully")
    print(f"  - Monitor metric: {optimizer_config['lr_scheduler']['monitor']}")
    print(f"  - Interval: {optimizer_config['lr_scheduler']['interval']}")
    
    # Test scheduler stepping with mock validation loss
    print("\nTesting scheduler step with mock validation metrics...")
    
    # Simulate multiple epochs with validation loss
    mock_val_losses = [0.8, 0.7, 0.6, 0.65, 0.63, 0.62, 0.61, 0.6, 0.59, 0.58]
    
    for epoch, val_loss in enumerate(mock_val_losses):
        print(f"Epoch {epoch}: val/loss = {val_loss:.3f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(val_loss)
    
    return True

def test_actual_training_minimal():
    """Test with minimal actual training setup."""
    print("\n" + "="*50)
    print("Testing minimal training setup...")
    
    # Create a simple mock datamodule for testing
    class MockDataModule(L.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.batch_size = 32
            
        def setup(self, stage=None):
            # Create tiny synthetic dataset
            X_tab = torch.randn(100, 42)
            X_esm = torch.randn(100, 2560) 
            y = torch.randint(0, 2, (100,)).float()
            
            # Split into train/val
            self.train_data = torch.utils.data.TensorDataset(X_tab[:80], X_esm[:80], y[:80])
            self.val_data = torch.utils.data.TensorDataset(X_tab[80:], X_esm[80:], y[80:])
            
        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_data, batch_size=16, shuffle=True)
            
        def val_dataloader(self):
            return torch.utils.data.DataLoader(self.val_data, batch_size=16, shuffle=False)
    
    # Create simple mock model that mimics the structure
    class MockModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(42 + 128, 64),  # tabular + projected ESM
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
                torch.nn.Sigmoid()
            )
            self.esm_projection = torch.nn.Linear(2560, 128)
            self.criterion = torch.nn.BCELoss()
            
        def forward(self, tabular_features, esm_features):
            esm_proj = self.esm_projection(esm_features)
            combined = torch.cat([tabular_features, esm_proj], dim=1)
            return self.model(combined)
            
        def model_step(self, batch):
            tabular_features, esm_features, targets = batch
            preds = self(tabular_features, esm_features).squeeze()
            loss = self.criterion(preds, targets)
            return loss, preds, targets
            
        def training_step(self, batch, batch_idx):
            loss, preds, targets = self.model_step(batch)
            self.log("train/loss", loss, on_epoch=True, prog_bar=True)
            return loss
            
        def validation_step(self, batch, batch_idx):
            loss, preds, targets = self.model_step(batch)
            self.log("val/loss", loss, on_epoch=True, prog_bar=True)
            
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=0.003, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
    
    # Test the setup
    datamodule = MockDataModule()
    model = MockModel()
    
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        check_val_every_n_epoch=1,  # Validate every epoch
        limit_train_batches=2,
        limit_val_batches=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True
    )
    
    try:
        print("🚀 Starting minimal training test...")
        trainer.fit(model, datamodule)
        print("✓ Minimal training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Minimal training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Scheduler Configuration")
    print("=" * 60)
    
    success1 = test_scheduler_config()
    success2 = test_actual_training_minimal()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed! Scheduler should work correctly now.")
    else:
        print("💥 Some tests failed! Check the errors above.")
    print("=" * 60)
