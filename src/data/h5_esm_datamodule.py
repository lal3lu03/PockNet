#!/usr/bin/env python3
"""
H5 DataModule for PockNet with properly mapped ESM embeddings.

This datamodule loads pre-computed H5 files containing:
1. Tabular features (45 dimensions) 
2. Residue-specific ESM2 embeddings (2560 dimensions)
3. Labels and metadata

Optimized for fast loading during training.
"""

import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import logging

log = logging.getLogger(__name__)


class H5Dataset(Dataset):
    """Dataset that loads data from H5 files."""
    
    def __init__(self, h5_file: str, transform=None):
        self.h5_file = h5_file
        self.transform = transform
        
        # Load data into memory for fast access
        log.info(f"Loading H5 dataset: {h5_file}")
        with h5py.File(h5_file, 'r') as f:
            self.tabular = f['tabular'][:]
            self.esm = f['esm'][:]
            self.labels = f['labels'][:]
            self.residue_numbers = f['residue_numbers'][:]
            self.protein_ids = [pid.decode() for pid in f['protein_ids'][:]]
        
        log.info(f"Loaded {len(self.labels)} samples")
        log.info(f"Tabular shape: {self.tabular.shape}")
        log.info(f"ESM shape: {self.esm.shape}")
        log.info(f"Labels distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {
            'tabular': torch.from_numpy(self.tabular[idx]).float(),
            'esm': torch.from_numpy(self.esm[idx]).float(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'residue_number': self.residue_numbers[idx],
            'protein_id': self.protein_ids[idx]
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class H5EsmDataModule(L.LightningDataModule):
    """Lightning DataModule for H5-based ESM datasets."""
    
    def __init__(
        self,
        train_h5: str = "data/h5/chen11_with_esm.h5",
        test_h5: str = "data/h5/bu48_with_esm.h5",
        val_split: float = 0.1,
        batch_size: int = 1024,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_h5 = train_h5
        self.test_h5 = test_h5
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        
        if stage == "fit" or stage is None:
            # Load full training dataset
            full_train_dataset = H5Dataset(self.train_h5)
            
            # Split into train/val
            total_size = len(full_train_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            log.info(f"Train size: {len(self.train_dataset)}")
            log.info(f"Val size: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            self.test_dataset = H5Dataset(self.test_h5)
            log.info(f"Test size: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
    
    def predict_dataloader(self):
        return self.test_dataloader()
    
    @property
    def num_classes(self) -> int:
        return 2
    
    @property
    def tabular_feature_dim(self) -> int:
        return 45  # Based on our feature extraction
    
    @property
    def esm_feature_dim(self) -> int:
        return 2560  # ESM2 embedding dimension


if __name__ == "__main__":
    # Test the datamodule
    dm = H5EsmDataModule(batch_size=64)
    dm.setup()
    
    # Test train loader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print("Batch structure:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)}")
    
    print(f"\nLabel distribution in batch: {torch.bincount(batch['label'])}")
    print(f"Sample protein IDs: {batch['protein_id'][:5]}")
