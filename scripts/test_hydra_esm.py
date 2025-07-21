#!/usr/bin/env python3
"""
Test ESM training with actual Hydra configuration to verify the complete pipeline.
"""

import os
import sys
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

# Import your modules
from src.train import train


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def test_hydra_training(cfg: DictConfig) -> None:
    """Test training with Hydra configuration."""
    
    # Override config for testing
    cfg.experiment = "pocknet_esm_separate"
    cfg.trainer.max_epochs = 2
    cfg.trainer.limit_train_batches = 4
    cfg.trainer.limit_val_batches = 2
    cfg.trainer.enable_checkpointing = False
    cfg.trainer.logger = False
    cfg.data.batch_size = 4
    
    # Create temporary data for testing (if needed)
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Override paths to use temp directory
        cfg.paths.output_dir = temp_dir
        cfg.paths.log_dir = temp_dir
        
        print("Starting training with ESM-augmented PockNet...")
        print(f"Fusion strategy: {cfg.model.fusion_strategy}")
        print(f"Max epochs: {cfg.trainer.max_epochs}")
        
        # Run training
        metric_dict, _ = train(cfg)
        
        print("Training completed successfully!")
        print(f"Final metrics: {metric_dict}")
        
        return metric_dict
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Cleaned up temporary directory")


if __name__ == "__main__":
    test_hydra_training()
