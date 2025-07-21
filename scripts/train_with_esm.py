# filepath: /system/user/studentwork/hageneder/MSC/Practical_work/PockNet/scripts/train_with_esm.py
"""
Simple script to test training with ESM embeddings directly.
This bypasses the Hydra configuration system.
"""
import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import numpy as np

# Add the project root to the Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

# Import our custom modules
from src.data.binding_site_esm_datamodule import BindingSiteESMDataModule
from src.models.tabnet_binding_site_module import TabNetBindingSiteModule


def main(args):
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Set up paths
    data_dir = os.path.join(proj_root, "data")
    esm_dir = os.path.join(data_dir, "esm2")
    log_dir = os.path.join(proj_root, "logs", "direct_esm_training")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"ESM directory: {esm_dir}")
    print(f"Log directory: {log_dir}")
    
    # Create data module with ESM embeddings
    datamodule = BindingSiteESMDataModule(
        data_dir=data_dir,
        esm_dir=esm_dir,
        embedding_type=args.embedding_type,
        batch_size=args.batch_size,
        sampling_strategy=args.sampling_strategy,
        eval_dataset=args.eval_dataset,
    )
    
    # Set up TabNet model
    model = TabNetBindingSiteModule(
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        n_d=args.n_d,
        n_a=args.n_a,
        n_steps=args.n_steps,
        gamma=args.gamma,
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(log_dir, "checkpoints"),
            filename="epoch_{epoch:03d}",
            monitor="val/acc",
            mode="max",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val/acc",
            mode="max",
            patience=10,
        ),
    ]
    
    # Set up logger
    logger = CSVLogger(log_dir, name="results")
    
    # Create trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1 if args.gpu is None else [args.gpu],
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    
    print("\nTraining Configuration:")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding type: {args.embedding_type}")
    print(f"Sampling strategy: {args.sampling_strategy}")
    print(f"Max epochs: {args.epochs}")
    print(f"TabNet params - n_d: {args.n_d}, n_a: {args.n_a}, n_steps: {args.n_steps}")
    
    # Train the model
    print("\nSetting up data module...")
    datamodule.setup()
    
    print("\nStarting training...")
    trainer.fit(model, datamodule=datamodule)
    
    # Test the model
    if args.test:
        print("\nRunning test evaluation...")
        trainer.test(model, datamodule=datamodule)
    
    print("\nTraining complete!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TabNet with ESM embeddings")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--embedding_type", type=str, default="mean", choices=["mean", "max", "first"], help="ESM embedding type")
    parser.add_argument("--sampling_strategy", type=str, default="combined", choices=["none", "oversample", "undersample", "combined"], help="Sampling strategy")
    parser.add_argument("--eval_dataset", type=str, default="chen11", choices=["chen11", "bu48"], help="Evaluation dataset")
    parser.add_argument("--n_d", type=int, default=128, help="Width of decision layer in TabNet")
    parser.add_argument("--n_a", type=int, default=128, help="Width of attention layer in TabNet")
    parser.add_argument("--n_steps", type=int, default=8, help="Number of steps in TabNet")
    parser.add_argument("--gamma", type=float, default=1.5, help="Coefficient for feature reuse in TabNet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device to use")
    parser.add_argument("--test", action="store_true", help="Run test after training")
    parser.add_argument("--limit_train_batches", type=int, default=None, help="Limit number of training batches")
    parser.add_argument("--limit_val_batches", type=int, default=None, help="Limit number of validation batches")
    
    args = parser.parse_args()
    main(args)
