#!/usr/bin/env python3
"""
Test ESM-augmented PockNet model integration.
This script creates a simple test to verify that our ESM-augmented PockNet
model can load data and perform forward passes correctly.
"""

import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add project root to Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

# Import our modules
from src.models.pocknet_esm_binding_site_module import PockNetESMBindingSiteModule
from src.models.components.pocknet_esm import PockNetESM

def load_sample_data():
    """Load a small sample of data for testing."""
    # Load training data
    train_csv_path = os.path.join(proj_root, "data", "train", "chen11.csv")
    print(f"Loading data from {train_csv_path}")
    
    # Read just first 1000 rows for testing
    df = pd.read_csv(train_csv_path, nrows=1000)
    print(f"Loaded {len(df)} rows for testing")
    print(f"Columns: {list(df.columns)}")
    
    # Extract file names and features
    file_names = df['file_name'].values
    features = df.drop(['file_name', 'class'], axis=1).values.astype(np.float32)
    labels = df['class'].values.astype(np.float32)
    
    print(f"Feature shape: {features.shape}")
    print(f"Unique file names: {len(np.unique(file_names))}")
    
    return file_names, features, labels

def create_mock_esm_embeddings(file_names, embedding_dim=128):
    """Create mock ESM embeddings for testing."""
    print(f"Creating mock ESM embeddings with dimension {embedding_dim}")
    
    # Create consistent random embeddings based on file names
    embeddings = {}
    for file_name in np.unique(file_names):
        # Use hash for consistent random generation
        np.random.seed(hash(file_name) % 2**31)
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embeddings[file_name] = torch.from_numpy(embedding)
    
    return embeddings

def test_pocknet_esm_forward_pass():
    """Test that PockNetESM can perform forward passes."""
    print("\n" + "="*50)
    print("Testing PockNetESM Forward Pass")
    print("="*50)
    
    # Model parameters
    tabular_dim = 42
    esm_dim = 128
    projected_esm_dim = 64
    batch_size = 32
    
    # Create model
    model = PockNetESM(
        tabular_dim=tabular_dim,
        esm_dim=esm_dim,
        esm_projection_dim=projected_esm_dim,
        fusion_strategy="concatenate",
        n_steps=3,
        n_d=32,
        n_a=32,
        output_dim=1
    )
    
    print(f"Model created with parameters:")
    print(f"  Tabular input dim: {tabular_dim}")
    print(f"  ESM input dim: {esm_dim}")
    print(f"  Projected ESM dim: {projected_esm_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample inputs
    tabular_features = torch.randn(batch_size, tabular_dim)
    esm_features = torch.randn(batch_size, esm_dim)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(tabular_features, esm_features)
    
    print(f"Input shapes: tabular {tabular_features.shape}, ESM {esm_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    return True

def test_lightning_module():
    """Test the Lightning module integration."""
    print("\n" + "="*50)
    print("Testing PockNetESMBindingSiteModule")
    print("="*50)
    
    # Load sample data
    file_names, features, labels = load_sample_data()
    mock_embeddings = create_mock_esm_embeddings(file_names, embedding_dim=128)
    
    # Prepare combined features
    esm_features_list = []
    for file_name in file_names:
        # Get the base filename without .pdb extension for lookup
        base_name = file_name.replace('.pdb', '') if file_name.endswith('.pdb') else file_name
        if base_name in mock_embeddings:
            esm_features_list.append(mock_embeddings[base_name])
        else:
            # Use first available embedding as fallback
            first_key = list(mock_embeddings.keys())[0]
            esm_features_list.append(mock_embeddings[first_key])
    
    esm_features = torch.stack(esm_features_list)
    
    # Combine tabular and ESM features
    combined_features = torch.cat([
        torch.from_numpy(features),
        esm_features
    ], dim=1)
    
    print(f"Combined feature shape: {combined_features.shape}")
    print(f"Tabular features: {features.shape[1]}, ESM features: {esm_features.shape[1]}")
    
    # Create Lightning module
    model = PockNetESMBindingSiteModule(
        tabular_dim=features.shape[1],
        esm_dim=esm_features.shape[1],
        esm_projection_dim=64,
        fusion_strategy="concatenate",
        n_steps=3,
        n_d=32,
        n_a=32,
        optimizer=torch.optim.Adam,
        lr=0.001
    )
    
    print(f"Lightning module created")
    print(f"Model input dimensions: tabular={features.shape[1]}, esm={esm_features.shape[1]}")
    
    # Create simple dataset and dataloader
    dataset = TensorDataset(combined_features, torch.from_numpy(labels))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Test training step
    model.train()
    batch = next(iter(dataloader))
    features_batch, labels_batch = batch
    
    # Split features back
    tabular_batch = features_batch[:, :features.shape[1]]
    esm_batch = features_batch[:, features.shape[1]:]
    
    # Prepare batch for model
    combined_batch = (torch.cat([tabular_batch, esm_batch], dim=1), labels_batch)
    
    # Training step
    loss = model.training_step(combined_batch, 0)
    print(f"Training step successful, loss: {loss.item():.4f}")
    
    return True

def test_different_fusion_strategies():
    """Test different fusion strategies."""
    print("\n" + "="*50)
    print("Testing Different Fusion Strategies")
    print("="*50)
    
    fusion_strategies = ["concatenate", "attention", "gated"]
    tabular_dim = 42
    esm_dim = 128
    batch_size = 16
    
    for strategy in fusion_strategies:
        print(f"\nTesting {strategy} fusion strategy:")
        
        model = PockNetESM(
            tabular_dim=tabular_dim,
            esm_dim=esm_dim,
            esm_projection_dim=64,
            fusion_strategy=strategy,
            n_steps=3,
            n_d=32,
            n_a=32,
            output_dim=1
        )
        
        # Sample inputs
        tabular_features = torch.randn(batch_size, tabular_dim)
        esm_features = torch.randn(batch_size, esm_dim)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(tabular_features, esm_features)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

def main():
    """Run all tests."""
    print("ESM-PockNet Integration Tests")
    print("="*60)
    
    try:
        # Test 1: Basic forward pass
        test_pocknet_esm_forward_pass()
        print("✓ PockNetESM forward pass test passed")
        
        # Test 2: Lightning module
        test_lightning_module()
        print("✓ Lightning module test passed")
        
        # Test 3: Different fusion strategies
        test_different_fusion_strategies()
        print("✓ Fusion strategies test passed")
        
        print("\n" + "="*60)
        print("🎉 All tests passed! ESM-PockNet integration is working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
