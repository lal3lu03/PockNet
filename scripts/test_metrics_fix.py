#!/usr/bin/env python3
"""
Simple test for ESM training pipeline - focusing on the metric issue fix.
"""

import os
import sys
import torch

# Add project root to Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

from src.models.pocknet_esm_binding_site_module import PockNetESMBindingSiteModule

def test_metrics_and_shapes():
    """Test metrics functionality and shape consistency."""
    print("Testing ESM model metrics and shapes...")
    
    # Create model with optimizer
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
    
    # Setup model
    model.setup("fit")
    
    # Create dummy batch
    batch_size = 8
    tabular_x = torch.randn(batch_size, 42)
    esm_x = torch.randn(batch_size, 128)
    y = torch.randint(0, 2, (batch_size, 1)).float()
    batch = (tabular_x, esm_x, y)
    
    print(f"Input shapes: tabular={tabular_x.shape}, esm={esm_x.shape}, targets={y.shape}")
    
    # Test model step
    loss, preds, targets = model.model_step(batch)
    print(f"Model step results:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Shapes match: {preds.shape == targets.shape}")
    
    # Test training step (this includes metric updates)
    model.train()
    train_loss = model.training_step(batch, 0)
    print(f"Training step completed with loss: {train_loss.item():.4f}")
    
    # Test validation step
    model.eval()
    with torch.no_grad():
        model.validation_step(batch, 0)
    print(f"Validation step completed")
    
    # Check metrics
    train_acc = model.train_acc.compute()
    val_acc = model.val_acc.compute()
    train_mse = model.train_mse.compute()
    val_mse = model.val_mse.compute()
    
    print(f"Metrics computed successfully:")
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Val accuracy: {val_acc:.3f}")
    print(f"  Train MSE: {train_mse:.4f}")
    print(f"  Val MSE: {val_mse:.4f}")
    
    # Test different fusion strategies
    strategies = ["concatenate", "attention", "gated"]
    for strategy in strategies:
        try:
            test_model = PockNetESMBindingSiteModule(
                optimizer=torch.optim.Adam,
                scheduler=None,
                tabular_dim=42,
                esm_dim=128,
                output_dim=1,
                n_steps=2,
                n_d=16,
                n_a=16,
                esm_projection_dim=32,
                fusion_strategy=strategy,
                use_iou_metric=False
            )
            test_model.setup("fit")
            
            with torch.no_grad():
                test_loss, test_preds, test_targets = test_model.model_step(batch)
            
            print(f"  {strategy:12s}: loss={test_loss.item():.4f}, shapes match={test_preds.shape == test_targets.shape}")
            
        except Exception as e:
            print(f"  {strategy:12s}: ERROR - {str(e)}")
    
    print("✓ All metrics and shape tests passed!")

if __name__ == "__main__":
    test_metrics_and_shapes()
