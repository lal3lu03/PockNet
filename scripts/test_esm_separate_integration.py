#!/usr/bin/env python3
"""Test script for ESM-augmented PockNet with separate feature handling.

This script tests the integration of:
1. BindingSiteESMDataModuleSeparate (separate tabular/ESM tensors)
2. PockNetESMBindingSiteModule (updated for separate inputs)
3. PockNetESM architecture (fusion strategies)
"""

import sys
import os
import torch
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_separate_datamodule():
    """Test the separate ESM DataModule."""
    print("=" * 60)
    print("Testing BindingSiteESMDataModuleSeparate...")
    print("=" * 60)
    
    try:
        from data.binding_site_esm_datamodule_separate import BindingSiteESMDataModuleSeparate
        
        # Initialize DataModule with small batch size for testing
        dm = BindingSiteESMDataModuleSeparate(
            data_dir="data/",
            esm_dir="data/esm2/",
            embedding_type="mean",
            train_val_test_split=(0.8, 0.1, 0.1),
            batch_size=32,
            num_workers=0,
            sampling_strategy="none",
            eval_dataset="chen11"
        )
        
        print("✓ DataModule created successfully")
        
        # Setup the data
        print("Setting up data...")
        dm.setup()
        
        print("✓ Data setup completed")
        
        # Test train dataloader
        train_loader = dm.train_dataloader()
        print(f"✓ Train loader created: {len(train_loader)} batches")
        
        # Test a batch
        batch = next(iter(train_loader))
        tabular_x, esm_x, y = batch
        
        print(f"✓ Batch loaded successfully:")
        print(f"  - Tabular features: {tabular_x.shape}")
        print(f"  - ESM features: {esm_x.shape}")
        print(f"  - Labels: {y.shape}")
        print(f"  - Label distribution: {torch.bincount(y.long())}")
        
        return tabular_x.shape[1], esm_x.shape[1]
        
    except Exception as e:
        print(f"✗ DataModule test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_esm_model_separate(tabular_dim, esm_dim):
    """Test the ESM-augmented PockNet model with separate inputs."""
    print("\n" + "=" * 60)
    print("Testing PockNetESMBindingSiteModule with separate inputs...")
    print("=" * 60)
    
    try:
        from models.pocknet_esm_binding_site_module import PockNetESMBindingSiteModule
        
        # Initialize model
        model = PockNetESMBindingSiteModule(
            tabular_dim=tabular_dim,
            esm_dim=esm_dim,
            output_dim=1,
            n_steps=3,
            n_d=32,
            n_a=32,
            esm_projection_dim=64,
            fusion_strategy="concatenate",
            use_iou_metric=True
        )
        
        print("✓ Model created successfully")
        
        # Setup model
        model.setup("fit")
        print("✓ Model setup completed")
        
        # Create dummy inputs
        batch_size = 16
        tabular_features = torch.randn(batch_size, tabular_dim)
        esm_features = torch.randn(batch_size, esm_dim)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            logits, attention_weights = model.forward(tabular_features, esm_features)
        
        print(f"✓ Forward pass successful:")
        print(f"  - Input shapes: tabular={tabular_features.shape}, esm={esm_features.shape}")
        print(f"  - Output logits: {logits.shape}")
        print(f"  - Attention weights: {attention_weights.shape}")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test different fusion strategies
        strategies = ["concatenate", "attention", "gated"]
        for strategy in strategies:
            test_model = PockNetESMBindingSiteModule(
                tabular_dim=tabular_dim,
                esm_dim=esm_dim,
                output_dim=1,
                n_steps=2,
                n_d=16,
                n_a=16,
                esm_projection_dim=32,
                fusion_strategy=strategy,
                use_iou_metric=False
            )
            test_model.setup("fit")
            
            test_model.eval()
            with torch.no_grad():
                out, _ = test_model.forward(tabular_features, esm_features)
            
            params = sum(p.numel() for p in test_model.parameters())
            print(f"  - {strategy:12s}: {params:6,} params, output shape {out.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_integration():
    """Test the full integration with a small training loop."""
    print("\n" + "=" * 60)
    print("Testing full integration with training step...")
    print("=" * 60)
    
    try:
        from data.binding_site_esm_datamodule_separate import BindingSiteESMDataModuleSeparate
        from models.pocknet_esm_binding_site_module import PockNetESMBindingSiteModule
        
        # Setup DataModule
        dm = BindingSiteESMDataModuleSeparate(
            data_dir="data/",
            esm_dir="data/esm2/",
            embedding_type="mean",
            batch_size=16,
            num_workers=0,
            sampling_strategy="none"
        )
        dm.setup()
        
        # Get data dimensions
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        tabular_x, esm_x, y = batch
        tabular_dim, esm_dim = tabular_x.shape[1], esm_x.shape[1]
        
        # Setup model
        model = PockNetESMBindingSiteModule(
            tabular_dim=tabular_dim,
            esm_dim=esm_dim,
            output_dim=1,
            n_steps=2,
            n_d=16,
            n_a=16,
            esm_projection_dim=32,
            fusion_strategy="concatenate"
        )
        model.setup("fit")
        
        # Test training step
        model.train()
        loss = model.training_step(batch, 0)
        
        print(f"✓ Training step successful:")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Batch size: {len(y)}")
        print(f"  - Feature dimensions: tabular={tabular_dim}, esm={esm_dim}")
        
        # Test validation step
        model.eval()
        val_loss = model.validation_step(batch, 0)
        print(f"  - Validation loss: {val_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("ESM-Augmented PockNet Integration Test (Separate Features)")
    print("=" * 60)
    
    # Test DataModule
    tabular_dim, esm_dim = test_separate_datamodule()
    if tabular_dim is None or esm_dim is None:
        print("\n✗ DataModule test failed. Stopping.")
        return False
    
    # Test Model
    model_success = test_esm_model_separate(tabular_dim, esm_dim)
    if not model_success:
        print("\n✗ Model test failed. Stopping.")
        return False
    
    # Test Integration
    integration_success = test_full_integration()
    if not integration_success:
        print("\n✗ Integration test failed.")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("ESM-augmented PockNet with separate features is ready for training!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
