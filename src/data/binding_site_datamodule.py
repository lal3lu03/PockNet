from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class BindingSiteDataModule(LightningDataModule):
    """LightningDataModule for Binding Site prediction dataset.

    Supports oversampling using SMOTE and undersampling using random undersampling.
    Also supports evaluation on bu48 dataset.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        sampling_strategy: str = "none",  # Options: "none", "oversample", "undersample", "combined"
        eval_dataset: str = "chen11",  # Options: "chen11", "bu48"
    ) -> None:
        """Initialize a BindingSiteDataModule.

        Args:
            data_dir: Directory containing the data files
            train_val_test_split: Proportions for train/val/test split
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU training
            sampling_strategy: Strategy for handling imbalanced data
            eval_dataset: Dataset to use for evaluation/testing
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[TensorDataset] = None
        self.data_val: Optional[TensorDataset] = None
        self.data_test: Optional[TensorDataset] = None
        
        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. This method is called only from a single GPU."""
        pass # data is expected to be already downloaded

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and apply train/val/test split.
        
        This method is called on every GPU in distributed training.
        """
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) must be divisible by the number of devices ({self.trainer.world_size})"
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        
        # For test-only stages, check if we need to load the bu48 evaluation dataset
        if stage == "test" and self.hparams.eval_dataset == "bu48":
            if not self.data_test:
                print(f"Loading bu48 evaluation dataset from {self.hparams.data_dir}/bu48/vectorsEval.csv")
                # Load bu48 evaluation data
                eval_data = pd.read_csv(f"{self.hparams.data_dir}/bu48/vectorsEval.csv")
                
                # Removing the file_name column if it exists
                if 'file_name' in eval_data.columns:
                    eval_data = eval_data.drop('file_name', axis=1)
                
                X_test = eval_data.drop('class', axis=1).values
                y_test = eval_data['class'].values
                
                # Print class distribution in evaluation set
                unique, counts = np.unique(y_test, return_counts=True)
                print(f"Bu48 evaluation set class distribution: {dict(zip(unique, counts))}")
                
                # Convert to PyTorch tensors - use Float for TabNet compatibility
                self.data_test = TensorDataset(
                    torch.FloatTensor(X_test),
                    torch.FloatTensor(y_test)
                )
                
                print(f"Bu48 evaluation set: {len(self.data_test)} samples")
            return

        # Load training/validation data (if not in test-only mode with bu48)
        if not self.data_train and not self.data_val and not self.data_test:
            # Load training data - updating path to data/data/chen11
            # Updated to use 'class' column instead of 'label'
            train_data = pd.read_csv(f"{self.hparams.data_dir}/data/chen11/vectorsTrain.csv")
            
            # Removing the file_name column as it's not a feature
            if 'file_name' in train_data.columns:
                train_data = train_data.drop('file_name', axis=1)
                
            X = train_data.drop('class', axis=1).values
            y = train_data['class'].values
            
            # Print class distribution before sampling
            unique, counts = np.unique(y, return_counts=True)
            print(f"Class distribution before sampling: {dict(zip(unique, counts))}")
            
            # Train/val/test split
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, 
                test_size=self.hparams.train_val_test_split[2],
                stratify=y,
                random_state=42
            )
            
            val_size = self.hparams.train_val_test_split[1] / (self.hparams.train_val_test_split[0] + self.hparams.train_val_test_split[1])
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size,
                stratify=y_train_val,
                random_state=42
            )
            
            # Apply sampling strategy on training data only
            if self.hparams.sampling_strategy == "oversample":
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print("Applied SMOTE oversampling")
            elif self.hparams.sampling_strategy == "undersample":
                rus = RandomUnderSampler(random_state=42)
                X_train, y_train = rus.fit_resample(X_train, y_train)
                print("Applied random undersampling")
            elif self.hparams.sampling_strategy == "combined":
                # First oversample minority class
                smote = SMOTE(random_state=42)
                X_temp, y_temp = smote.fit_resample(X_train, y_train)
                # Then undersample majority class
                rus = RandomUnderSampler(random_state=42)
                X_train, y_train = rus.fit_resample(X_temp, y_temp)
                print("Applied combined sampling (SMOTE + undersampling)")
            
            # Print class distribution after sampling
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"Class distribution after sampling: {dict(zip(unique, counts))}")
            
            # Convert to PyTorch tensors - convert labels to Float for TabNet compatibility
            self.data_train = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)  # Changed from LongTensor to FloatTensor
            )
            self.data_val = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)  # Changed from LongTensor to FloatTensor
            )
            self.data_test = TensorDataset(
                torch.FloatTensor(X_test),
                torch.FloatTensor(y_test)  # Changed from LongTensor to FloatTensor
            )
            
            print(f"Train set: {len(self.data_train)} samples")
            print(f"Val set: {len(self.data_val)} samples")
            print(f"Test set: {len(self.data_test)} samples")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )