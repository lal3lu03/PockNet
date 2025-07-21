#!/usr/bin/env python3
"""Quick test script for ESM-augmented PockNet with separate feature handling.

This script uses a small subset of data for fast testing.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_quick_integration():
    """Test with a small subset of data for quick validation."""
    print("Quick ESM-Augmented PockNet Test (Separate Features)")
    print("=" * 60)
    
    try:
        from data.binding_site_esm_datamodule_separate import BindingSiteESMDataModuleSeparate
        from models.pocknet_esm_binding_site_module import PockNetESMBindingSiteModule
        
        print("✓ Imports successful")
        
        # Create a test dataset with a small subset
        print("Creating test subset...")
        
        # Read first 1000 rows from chen11.csv
        full_path = "data/train/chen11.csv"
        if not os.path.exists(full_path):
            print(f"✗ Data file not found: {full_path}")
            return False
            
        # Load just the header to get column info
        sample_data = pd.read_csv(full_path, nrows=1000)
        print(f"✓ Loaded {len(sample_data)} test samples")
        
        # Get feature dimensions
        coords_to_remove = ['file_name', 'x', 'y', 'z', 'chain_id', 'residue_number', 'residue_name', 'class']
        feature_cols = [col for col in sample_data.columns if col not in coords_to_remove]
        tabular_dim = len(feature_cols)
        
        print(f"✓ Tabular features: {tabular_dim} dimensions")
        
        # Test model creation
        print("Testing model...")
        model = PockNetESMBindingSiteModule(
            tabular_dim=tabular_dim,
            esm_dim=2560,  # Standard ESM-2 dimension
            output_dim=1,
            n_steps=2,
            n_d=16,
            n_a=16,
            esm_projection_dim=32,
            fusion_strategy="concatenate"
        )
        
        print("✓ Model created successfully")
        
        # Setup model
        model.setup("fit")
        print("✓ Model setup completed")
        
        # Test with dummy data
        batch_size = 8
        tabular_features = torch.randn(batch_size, tabular_dim)
        esm_features = torch.randn(batch_size, 2560)
        labels = torch.randint(0, 2, (batch_size,)).float()
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            logits, attention_weights = model.forward(tabular_features, esm_features)
        
        print(f"✓ Forward pass successful:")
        print(f"  - Input shapes: tabular={tabular_features.shape}, esm={esm_features.shape}")
        print(f"  - Output logits: {logits.shape}")
        print(f"  - Attention weights: {attention_weights.shape}")
        
        # Test training step
        batch = (tabular_features, esm_features, labels)
        model.train()
        loss = model.training_step(batch, 0)
        
        print(f"✓ Training step successful:")
        print(f"  - Loss: {loss.item():.4f}")
        
        # Test different fusion strategies
        strategies = ["concatenate", "attention", "gated"]
        for strategy in strategies:
            test_model = PockNetESMBindingSiteModule(
                tabular_dim=tabular_dim,
                esm_dim=2560,
                output_dim=1,
                n_steps=2,
                n_d=8,
                n_a=8,
                esm_projection_dim=16,
                fusion_strategy=strategy
            )
            test_model.setup("fit")
            
            test_model.eval()
            with torch.no_grad():
                out, _ = test_model.forward(tabular_features, esm_features)
            
            params = sum(p.numel() for p in test_model.parameters())
            print(f"  - {strategy:12s}: {params:6,} params, output shape {out.shape}")
        
        print("\n🎉 QUICK TEST PASSED! 🎉")
        print("ESM-augmented PockNet with separate features is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quick_integration()
    sys.exit(0 if success else 1)
