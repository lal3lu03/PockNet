from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class PockNetDataModule(LightningDataModule):
    """LightningDataModule for PockNet binding site prediction.

    Uses chen11.csv for training and bu48.csv for testing.
    The 'class' column contains the target labels (0/1 for non-binding/binding sites).
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_split: float = 0.8,  # Split chen11 into train/val
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        sampling_strategy: str = "none",  # Options: "none", "oversample", "undersample", "combined"
        normalize_features: bool = True,
    ) -> None:
        """Initialize a PockNetDataModule.

        Args:
            data_dir: Directory containing the data files
            train_val_split: Proportion for train/val split of chen11 data
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU training
            sampling_strategy: Strategy for handling imbalanced data
            normalize_features: Whether to normalize features
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[TensorDataset] = None
        self.data_val: Optional[TensorDataset] = None
        self.data_test: Optional[TensorDataset] = None
        self.scaler: Optional[StandardScaler] = None
        self.dims: Optional[int] = None
        
        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. This method is called only from a single GPU."""
        # Check if data files exist
        train_file = f"{self.hparams.data_dir}/train/chen11.csv"
        test_file = f"{self.hparams.data_dir}/test/bu48.csv"
        
        try:
            pd.read_csv(train_file)
            pd.read_csv(test_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data files not found: {e}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        
        # Define feature columns (exclude metadata and target)
        metadata_columns = ['file_name', 'x', 'y', 'z', 'chain_id', 'residue_number', 'residue_name', 'class']
        
        # Exclude problematic feature for better generalization
        excluded_features = ['protrusion.distanceToCenter']
        
        if stage == "fit" or stage is None:
            # Load training data (chen11)
            train_df = pd.read_csv(f"{self.hparams.data_dir}/train/chen11.csv")
            
            # Get feature columns (all except metadata and excluded features)
            feature_columns = [col for col in train_df.columns 
                             if col not in metadata_columns and col not in excluded_features]
            
            # Extract features and targets
            X_train_full = train_df[feature_columns].values.astype(np.float32)
            y_train_full = train_df['class'].values.astype(np.float32)
            
            # Set dimensions
            self.dims = X_train_full.shape[1]
            
            # Split chen11 into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, 
                train_size=self.hparams.train_val_split,
                random_state=42,
                stratify=y_train_full
            )
            
            # Initialize scaler if needed
            if self.hparams.normalize_features:
                self.scaler = StandardScaler()
                X_train = self.scaler.fit_transform(X_train)
                X_val = self.scaler.transform(X_val)
            
            # Apply sampling strategy if requested
            if self.hparams.sampling_strategy != "none":
                X_train, y_train = self._apply_sampling_strategy(X_train, y_train)
            
            # Convert to tensors
            X_train_tensor = torch.from_numpy(X_train)
            y_train_tensor = torch.from_numpy(y_train)
            X_val_tensor = torch.from_numpy(X_val)
            y_val_tensor = torch.from_numpy(y_val)
            
            # Create datasets
            self.data_train = TensorDataset(X_train_tensor, y_train_tensor)
            self.data_val = TensorDataset(X_val_tensor, y_val_tensor)
            
            print(f"Training data: {len(self.data_train)} samples")
            print(f"Validation data: {len(self.data_val)} samples")
            print(f"Feature dimensions: {self.dims}")
            print(f"Excluded features: {excluded_features}")
            print(f"Training class distribution: {np.bincount(y_train.astype(int))}")
            print(f"Validation class distribution: {np.bincount(y_val.astype(int))}")

        if stage == "test" or stage is None:
            # Load test data (bu48)
            test_df = pd.read_csv(f"{self.hparams.data_dir}/test/bu48.csv")
            
            # Get same feature columns as training (excluding metadata and excluded features)
            if hasattr(self, 'dims') and self.dims is not None:
                feature_columns = [col for col in test_df.columns 
                                 if col not in metadata_columns and col not in excluded_features]
            else:
                # If dims not set, infer from test data
                feature_columns = [col for col in test_df.columns 
                                 if col not in metadata_columns and col not in excluded_features]
                self.dims = len(feature_columns)
            
            # Extract features and targets
            X_test = test_df[feature_columns].values.astype(np.float32)
            y_test = test_df['class'].values.astype(np.float32)
            
            # Apply same normalization as training if available
            if self.scaler is not None:
                X_test = self.scaler.transform(X_test)
            
            # Convert to tensors
            X_test_tensor = torch.from_numpy(X_test)
            y_test_tensor = torch.from_numpy(y_test)
            
            # Create dataset
            self.data_test = TensorDataset(X_test_tensor, y_test_tensor)
            
            print(f"Test data: {len(self.data_test)} samples")
            print(f"Test class distribution: {np.bincount(y_test.astype(int))}")

    def _apply_sampling_strategy(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the specified sampling strategy to handle class imbalance."""
        print(f"Original class distribution: {np.bincount(y.astype(int))}")
        
        if self.hparams.sampling_strategy == "oversample":
            # Oversample minority class using SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
        elif self.hparams.sampling_strategy == "undersample":
            # Undersample majority class
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
            
        elif self.hparams.sampling_strategy == "combined":
            # First oversample, then undersample
            smote = SMOTE(random_state=42)
            X_oversampled, y_oversampled = smote.fit_resample(X, y)
            
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X_oversampled, y_oversampled)
            
        else:
            X_resampled, y_resampled = X, y
        
        print(f"Resampled class distribution: {np.bincount(y_resampled.astype(int))}")
        return X_resampled, y_resampled

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `fit`, `validate`, `test`, or `predict`."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state."""
        state = {"scaler": self.scaler, "dims": self.dims}
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given from a checkpoint."""
        self.scaler = state_dict.get("scaler")
        self.dims = state_dict.get("dims")
