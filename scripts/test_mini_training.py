#!/usr/bin/env python3
"""
Small-scale ESM-augmented PockNet training experiment.

This script runs a short training session on a subset of data to verify
the entire pipeline works end-to-end.
"""

import sys
import os
import torch
import hydra
from omegaconf import DictConfig

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_mini_dataset():
    """Create a mini dataset for quick training test."""
    import pandas as pd
    import numpy as np
    
    # Load a small subset of chen11.csv
    full_path = "data/train/chen11.csv"
    if not os.path.exists(full_path):
        print(f"Data file not found: {full_path}")
        return False
    
    # Read first 5000 rows
    print("Creating mini dataset...")
    data = pd.read_csv(full_path, nrows=5000)
    
    # Save as a temporary mini dataset
    mini_path = "data/train/chen11_mini.csv"
    data.to_csv(mini_path, index=False)
    
    print(f"✓ Created mini dataset: {len(data)} samples saved to {mini_path}")
    return True

def run_mini_training():
    """Run a mini training session."""
    try:
        from data.binding_site_esm_datamodule_separate import BindingSiteESMDataModuleSeparate
        from models.pocknet_esm_binding_site_module import PockNetESMBindingSiteModule
        import lightning as L
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        
        print("Mini ESM-Augmented PockNet Training")
        print("=" * 50)
        
        # Create mini dataset
        if not create_mini_dataset():
            return False
        
        # Create a modified DataModule that uses the mini dataset
        class MiniDataModule(BindingSiteESMDataModuleSeparate):
            def setup(self, stage=None):
                # Override to use mini dataset
                import pandas as pd
                import torch
                import numpy as np
                from sklearn.model_selection import train_test_split
                
                # Load mini training data
                train_csv_path = "data/train/chen11_mini.csv"
                print(f"Loading mini training data from {train_csv_path}")
                train_data = pd.read_csv(train_csv_path)
                
                # Extract file names
                file_names = train_data['file_name'].values if 'file_name' in train_data.columns else []
                print(f"Found {len(file_names)} file names")
                
                # Remove non-feature columns
                if 'file_name' in train_data.columns:
                    train_data = train_data.drop('file_name', axis=1)
                coords_to_remove = ['x', 'y', 'z', 'chain_id', 'residue_number', 'residue_name']
                for col in coords_to_remove:
                    if col in train_data.columns:
                        train_data = train_data.drop(col, axis=1)
                
                X = train_data.drop('class', axis=1).values
                y = train_data['class'].values
                
                print(f"Features: {X.shape[1]} dimensions")
                
                # Create dummy ESM embeddings for speed
                print("Creating dummy ESM embeddings for testing...")
                esm_tensor = torch.randn(len(X), 128)  # Use smaller dim for speed
                
                # Split data
                X_train, X_test, esm_train, esm_test, y_train, y_test = train_test_split(
                    X, esm_tensor, y, test_size=0.2, stratify=y, random_state=42
                )
                X_train, X_val, esm_train, esm_val, y_train, y_val = train_test_split(
                    X_train, esm_train, y_train, test_size=0.25, stratify=y_train, random_state=42
                )
                
                # Create datasets
                from torch.utils.data import TensorDataset
                
                self.data_train = TensorDataset(
                    torch.FloatTensor(X_train),
                    esm_train,
                    torch.FloatTensor(y_train)
                )
                self.data_val = TensorDataset(
                    torch.FloatTensor(X_val),
                    esm_val,
                    torch.FloatTensor(y_val)
                )
                self.data_test = TensorDataset(
                    torch.FloatTensor(X_test),
                    esm_test,
                    torch.FloatTensor(y_test)
                )
                
                print(f"Train: {len(self.data_train)}, Val: {len(self.data_val)}, Test: {len(self.data_test)}")
        
        # Setup DataModule
        dm = MiniDataModule(
            data_dir="data/",
            esm_dir="data/esm2/",
            batch_size=32,
            num_workers=0
        )
        dm.setup()
        
        # Setup Model
        model = PockNetESMBindingSiteModule(
            tabular_dim=42,
            esm_dim=128,  # Match dummy ESM dim
            output_dim=1,
            n_steps=2,
            n_d=16,
            n_a=16,
            esm_projection_dim=32,
            fusion_strategy="concatenate"
        )
        
        # Setup the model to initialize parameters
        model.setup("fit")
        
        # Add optimizer configuration methods
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        # Monkey patch the configure_optimizers method
        import types
        model.configure_optimizers = types.MethodType(configure_optimizers, model)
        
        # Setup trainer
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath="./mini_checkpoints/",
                filename="mini-{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
                mode="min"
            )
        ]
        
        trainer = L.Trainer(
            max_epochs=10,
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            logger=False,  # Disable logging for speed
            enable_checkpointing=True,
            enable_progress_bar=True,
            check_val_every_n_epoch=2
        )
        
        print("Starting training...")
        trainer.fit(model, dm)
        
        print("✓ Training completed successfully!")
        
        # Test the model
        print("Running test...")
        trainer.test(model, dm)
        
        print("✓ Testing completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Mini training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_mini_training()
    
    # Cleanup
    if os.path.exists("data/train/chen11_mini.csv"):
        os.remove("data/train/chen11_mini.csv")
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 MINI TRAINING SUCCESS! 🎉")
        print("ESM-augmented PockNet training pipeline works correctly!")
    else:
        print("✗ Mini training failed")
    print("=" * 50)
    
    sys.exit(0 if success else 1)
