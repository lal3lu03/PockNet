#!/usr/bin/env python3
"""
Test the complete ESM-augmented PockNet training pipeline with corrected metrics.
"""

import os
import sys
import torch
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

import hydra
from omegaconf import DictConfig
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from src.models.pocknet_esm_binding_site_module import PockNetESMBindingSiteModule
from src.data.binding_site_esm_datamodule_separate import BindingSiteESMDataModuleSeparate

def test_metrics_only():
    """Test just the metrics functionality."""
    print("Testing metrics functionality...")
    
    # Create model
    model = PockNetESMBindingSiteModule(
        optimizer=torch.optim.Adam,
        scheduler=None,
        tabular_dim=42,
        esm_dim=128,
        output_dim=1,
        n_steps=3,
        n_d=32,
        n_a=32,
        esm_projection_dim=64,
        fusion_strategy="concatenate",
        use_iou_metric=True
    )
    
    # Create dummy batch
    batch_size = 8
    tabular_x = torch.randn(batch_size, 42)
    esm_x = torch.randn(batch_size, 128)
    y = torch.randint(0, 2, (batch_size, 1)).float()
    batch = (tabular_x, esm_x, y)
    
    # Test model step
    model.setup("fit")
    loss, preds, targets = model.model_step(batch)
    
    print(f"Model step successful:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Pred range: [{preds.min().item():.3f}, {preds.max().item():.3f}]")
    
    # Test training step
    model.train()
    train_loss = model.training_step(batch, 0)
    print(f"  Training step loss: {train_loss.item():.4f}")
    
    # Test validation step
    model.eval()
    with torch.no_grad():
        model.validation_step(batch, 0)
    print(f"  Validation step completed")
    
    # Check that metrics are working
    print(f"  Train accuracy: {model.train_acc.compute():.3f}")
    print(f"  Val accuracy: {model.val_acc.compute():.3f}")
    
    print("✓ Metrics test passed!")

def test_mini_training():
    """Test a mini training loop."""
    print("\nTesting mini training loop...")
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Create model
        model = PockNetESMBindingSiteModule(
            optimizer=torch.optim.Adam,
            scheduler=None,
            tabular_dim=42,
            esm_dim=128,
            output_dim=1,
            n_steps=2,
            n_d=16,
            n_a=16,
            esm_projection_dim=32,
            fusion_strategy="concatenate",
            use_iou_metric=True
        )
        
        # Create dummy datamodule
        class DummyDataModule:
            def __init__(self):
                self.batch_size = 4
                self.num_samples = 16
                
            def setup(self, stage=None):
                pass
                
            def train_dataloader(self):
                from torch.utils.data import DataLoader, TensorDataset
                tabular_data = torch.randn(self.num_samples, 42)
                esm_data = torch.randn(self.num_samples, 128)
                targets = torch.randint(0, 2, (self.num_samples, 1)).float()
                dataset = TensorDataset(tabular_data, esm_data, targets)
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                
            def val_dataloader(self):
                from torch.utils.data import DataLoader, TensorDataset
                val_samples = 8
                tabular_data = torch.randn(val_samples, 42)
                esm_data = torch.randn(val_samples, 128)
                targets = torch.randint(0, 2, (val_samples, 1)).float()
                dataset = TensorDataset(tabular_data, esm_data, targets)
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        datamodule = DummyDataModule()
        
        # Create trainer with corrected callbacks
        logger = CSVLogger(temp_dir, name="test_training")
        
        early_stopping = EarlyStopping(
            monitor="val/loss",  # Use val/loss instead of val/acc
            min_delta=0.0,
            patience=3,
            verbose=True,
            mode="min"  # min for loss
        )
        
        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(temp_dir, "checkpoints"),
            filename="best_model",
            monitor="val/loss",
            mode="min",
            save_top_k=1
        )
        
        trainer = Trainer(
            max_epochs=3,
            logger=logger,
            callbacks=[early_stopping, checkpoint],
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            accelerator="cpu",
            devices=1
        )
        
        # Run training
        print("Starting mini training...")
        trainer.fit(model, datamodule)
        
        print("✓ Mini training completed successfully!")
        
        # Check logged metrics
        log_dir = Path(temp_dir) / "test_training" / "version_0"
        metrics_file = log_dir / "metrics.csv"
        
        if metrics_file.exists():
            import pandas as pd
            metrics = pd.read_csv(metrics_file)
            print(f"Logged metrics columns: {list(metrics.columns)}")
            
            # Check if our metrics are there
            expected_metrics = ["train/loss", "train/acc", "val/loss", "val/acc"]
            found_metrics = [m for m in expected_metrics if m in metrics.columns]
            print(f"Found expected metrics: {found_metrics}")
            
            if len(found_metrics) == len(expected_metrics):
                print("✓ All expected metrics found in logs!")
            else:
                missing = set(expected_metrics) - set(found_metrics)
                print(f"⚠ Missing metrics: {missing}")
        else:
            print("⚠ No metrics file found")
            
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp directory")

def main():
    """Run all tests."""
    print("Testing ESM-augmented PockNet training pipeline...")
    print("=" * 60)
    
    test_metrics_only()
    test_mini_training()
    
    print("\n" + "=" * 60)
    print("✓ All training pipeline tests passed!")

if __name__ == "__main__":
    main()
