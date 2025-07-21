#!/usr/bin/env python3
"""
Test training script for ESM + Tabular model using H5 datasets.

This script provides a quick test of the full training pipeline with
properly mapped ESM embeddings and tabular features.
"""

import os
import sys
sys.path.append(os.getcwd())

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.data.h5_esm_datamodule import H5EsmDataModule
from src.models.esm_tabular_module import EsmTabularModule


def main():
    """Test training pipeline."""
    
    # Set seed for reproducibility
    L.seed_everything(42)
    
    # Initialize datamodule
    print("🔄 Initializing datamodule...")
    dm = H5EsmDataModule(
        train_h5="data/h5/chen11_with_esm.h5",
        test_h5="data/h5/bu48_with_esm.h5",
        batch_size=512,  # Smaller batch for testing
        num_workers=8,
        val_split=0.1
    )
    
    # Initialize model
    print("🔄 Initializing model...")
    model = EsmTabularModule(
        tabular_dim=45,
        esm_dim=2560,
        hidden_dims=[512, 256, 128],  # Smaller model for testing
        dropout=0.2,
        fusion_method="attention",
        fusion_dim=256,
        num_classes=2,
        optimizer={
            "_target_": "torch.optim.AdamW",
            "_partial_": True,
            "lr": 0.001,
            "weight_decay": 0.01
        }
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        filename="best-model-{epoch:02d}-{val/f1:.4f}",
        dirpath="test_checkpoints"
    )
    
    early_stopping = EarlyStopping(
        monitor="val/f1",
        mode="max", 
        patience=5,
        min_delta=0.001
    )
    
    # Setup logger
    logger = TensorBoardLogger("test_logs", name="esm_tabular_test")
    
    # Initialize trainer
    print("🔄 Initializing trainer...")
    trainer = L.Trainer(
        max_epochs=3,  # Just a few epochs for testing
        devices=1,  # Single GPU for testing
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=0.5,
        enable_progress_bar=True
    )
    
    # Setup data
    print("🔄 Setting up data...")
    dm.setup("fit")
    
    # Print dataset info
    print(f"📊 Dataset Information:")
    print(f"   Train samples: {len(dm.train_dataset):,}")
    print(f"   Val samples: {len(dm.val_dataset):,}")
    print(f"   Test samples: {len(dm.test_dataset):,}")
    
    # Test one batch
    print("🔄 Testing data loading...")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"   Batch tabular shape: {batch['tabular'].shape}")
    print(f"   Batch ESM shape: {batch['esm'].shape}")
    print(f"   Batch labels shape: {batch['label'].shape}")
    print(f"   Label distribution: {torch.bincount(batch['label'])}")
    
    # Test model forward pass
    print("🔄 Testing model forward pass...")
    with torch.no_grad():
        logits = model(batch['tabular'], batch['esm'])
        print(f"   Output shape: {logits.shape}")
        print(f"   Output sample: {logits[0]}")
    
    # Start training
    print("🚀 Starting training...")
    trainer.fit(model, dm)
    
    # Test
    print("🧪 Running test...")
    trainer.test(model, dm)
    
    print("✅ Test completed successfully!")


if __name__ == "__main__":
    main()
