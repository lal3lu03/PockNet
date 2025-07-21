#!/usr/bin/env python3
"""
Simple test for ESM-augmented PockNet model.
"""

import os
import sys
import torch

# Add project root to Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

from src.models.components.pocknet_esm import PockNetESM

def test_basic_model():
    """Test basic model creation and forward pass."""
    print("Testing PockNetESM basic functionality...")
    
    # Parameters
    tabular_dim = 42
    esm_dim = 128
    batch_size = 16
    
    # Create model
    model = PockNetESM(
        tabular_dim=tabular_dim,
        esm_dim=esm_dim,
        output_dim=1,
        n_steps=3,
        n_d=32,
        n_a=32,
        esm_projection_dim=64,
        fusion_strategy="concatenate"
    )
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample inputs
    tabular_features = torch.randn(batch_size, tabular_dim)
    esm_features = torch.randn(batch_size, esm_dim)
    
    print(f"Input shapes: tabular={tabular_features.shape}, esm={esm_features.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output, attention_weights = model(tabular_features, esm_features)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Test different fusion strategies
    strategies = ["concatenate", "attention", "gated"]
    for strategy in strategies:
        model_test = PockNetESM(
            tabular_dim=tabular_dim,
            esm_dim=esm_dim,
            output_dim=1,
            n_steps=2,
            n_d=16,
            n_a=16,
            esm_projection_dim=32,
            fusion_strategy=strategy
        )
        
        model_test.eval()
        with torch.no_grad():
            out, _ = model_test(tabular_features, esm_features)
        
        params = sum(p.numel() for p in model_test.parameters())
        print(f"{strategy:12s}: {params:6,} params, output shape {out.shape}")
    
    print("✓ All tests passed!")

if __name__ == "__main__":
    test_basic_model()
