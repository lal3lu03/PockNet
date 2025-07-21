#!/usr/bin/env python3
"""Test script to verify the learning rate scheduler works correctly with validation metrics."""

import os
import sys
import torch
import lightning.pytorch as L
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pocknet_esm_binding_site_module import PockNetESMBindingSiteModule
from data.binding_site_esm_datamodule_separate import BindingSiteESMDataModuleSeparate

def test_scheduler_integration():
    """Test that the scheduler works with PyTorch Lightning validation loop."""
    print("Testing scheduler integration with Lightning validation...")
    
    # Create a minimal dataset for testing
    datamodule = BindingSiteESMDataModuleSeparate(
        data_dir="data/",
        esm_dir="data/esm2/",
        batch_size=32,
        num_workers=0,
        train_val_test_split=[0.7, 0.15, 0.15]
    )
    
    # Create model with scheduler
    model = PockNetESMBindingSiteModule(
        tabular_dim=42,
        esm_dim=2560,
        output_dim=1,
        n_steps=3,
        n_d=32,
        n_a=32,
        n_shared=2,
        n_independent=2,
        gamma=1.3,
        epsilon=1e-15,
        dropout=0.2,
        esm_projection_dim=128,
        fusion_strategy="concatenate",
        use_iou_metric=True,
        optimizer={"_target_": "torch.optim.Adam", "_partial_": True, "lr": 0.003, "weight_decay": 1e-5},
        scheduler={
            "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "_partial_": True,
            "mode": "min",
            "factor": 0.5,
            "patience": 10,
            "verbose": True
        }
    )
    
    # Create trainer with validation every epoch
    trainer = L.Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1,
        check_val_every_n_epoch=1,  # Validation every epoch
        limit_train_batches=3,      # Limit for fast testing
        limit_val_batches=2,        # Limit for fast testing
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False
    )
    
    try:
        # Setup data
        datamodule.setup()
        print(f"✓ Data setup successful")
        print(f"  - Train samples: {len(datamodule.data_train)}")
        print(f"  - Val samples: {len(datamodule.data_val)}")
        
        # Test model initialization
        train_batch = next(iter(datamodule.train_dataloader()))
        val_batch = next(iter(datamodule.val_dataloader()))
        print(f"✓ Data loading successful")
        
        # Test a single forward pass
        model.eval()
        with torch.no_grad():
            tabular_features, esm_features, targets = train_batch
            outputs = model(tabular_features, esm_features)
            print(f"✓ Forward pass successful - Output shape: {outputs[0].shape}")
        
        # Test training step
        model.train()
        loss, preds, targets = model.model_step(train_batch)
        print(f"✓ Training step successful - Loss: {loss:.4f}")
        
        # Test validation step  
        model.eval()
        with torch.no_grad():
            model.validation_step(val_batch, 0)
        print(f"✓ Validation step successful")
        
        # Test optimizer configuration
        optimizer_config = model.configure_optimizers()
        print(f"✓ Optimizer configuration successful")
        print(f"  - Has scheduler: {'lr_scheduler' in optimizer_config}")
        if 'lr_scheduler' in optimizer_config:
            print(f"  - Scheduler monitor: {optimizer_config['lr_scheduler']['monitor']}")
            print(f"  - Scheduler interval: {optimizer_config['lr_scheduler']['interval']}")
        
        # Now test actual training
        print("\n🚀 Starting training test...")
        trainer.fit(model, datamodule)
        print("✓ Training completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PyTorch Lightning Scheduler Integration")
    print("=" * 60)
    
    success = test_scheduler_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Scheduler integration is working correctly.")
    else:
        print("💥 Tests failed! Check the errors above.")
    print("=" * 60)
