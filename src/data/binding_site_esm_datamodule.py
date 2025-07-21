from typing import Any, Dict, Optional, Tuple, List
import os
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from glob import glob


class BindingSiteESMDataModule(LightningDataModule):
    """LightningDataModule for Binding Site prediction with ESM2 embeddings.
    
    This module extends the base BindingSiteDataModule to incorporate ESM2 embeddings
    with the tabular features from the original dataset.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        esm_dir: str = "data/esm2/",
        embedding_type: str = "mean",  # Options: "mean", "max", "first"
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        sampling_strategy: str = "none",  # Options: "none", "oversample", "undersample", "combined"
        eval_dataset: str = "chen11",  # Options: "chen11", "bu48"
    ) -> None:
        """Initialize a BindingSiteESMDataModule.

        Args:
            data_dir: Directory containing the data files
            esm_dir: Directory containing the ESM2 embeddings
            embedding_type: How to process the sequence embeddings ("mean", "max", "first")
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
        self.esm_embeddings = {}  # Cache for loaded embeddings
        
    def _load_esm_embedding(self, file_id: str) -> torch.Tensor:
        """Load ESM embedding for a specific protein file.
        
        Args:
            file_id: Protein ID to load embedding for (e.g., 'a.001.001.001_1s69a.pdb')
            
        Returns:
            A tensor containing the embedding representation
        """
        # Check if already in cache
        if file_id in self.esm_embeddings:
            return self.esm_embeddings[file_id]
            
        # Try to find the embedding file - strip any path and extension
        # Remove .pdb extension if present
        if file_id.endswith('.pdb'):
            base_file_id = file_id[:-4]  # Remove '.pdb'
        else:
            base_file_id = os.path.basename(file_id)
            
        embedding_path = os.path.join(self.hparams.esm_dir, f"{base_file_id}.pt")
        
        if not os.path.exists(embedding_path):
            print(f"Warning: No embedding found for {file_id} (tried {embedding_path})")
            # Return zero tensor with correct embedding dimension (determined from existing embeddings)
            # Use the first available embedding to determine the dimension
            sample_files = glob(os.path.join(self.hparams.esm_dir, "*.pt"))
            if len(sample_files) > 0:
                sample_embed = torch.load(sample_files[0])
                if self.hparams.embedding_type == "mean" or self.hparams.embedding_type == "max":
                    return torch.zeros(sample_embed.size(-1))
                else:  # "first"
                    return torch.zeros(sample_embed.size(-1))
            else:
                # Default ESM2 size
                return torch.zeros(2560)
                
        # Load embedding
        embedding = torch.load(embedding_path)
        
        # Process according to embedding_type
        if self.hparams.embedding_type == "mean":
            # Mean pooling across sequence length
            processed_embedding = torch.mean(embedding, dim=0)
        elif self.hparams.embedding_type == "max":
            # Max pooling across sequence length
            processed_embedding = torch.max(embedding, dim=0)[0]
        elif self.hparams.embedding_type == "first":
            # First token embedding (excluding CLS token if it's the first)
            processed_embedding = embedding[1] if embedding.size(0) > 1 else embedding[0]
        else:
            raise ValueError(f"Unknown embedding type: {self.hparams.embedding_type}")
            
        # Store in cache
        self.esm_embeddings[file_id] = processed_embedding
        return processed_embedding
        
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
                # Try different possible paths for bu48 evaluation data
                eval_csv_path = f"{self.hparams.data_dir}/test/bu48.csv"
                if not os.path.exists(eval_csv_path):
                    eval_csv_path = f"{self.hparams.data_dir}/bu48/vectorsEval.csv"
                    print(f"Trying alternative path {eval_csv_path}...")
                    
                if not os.path.exists(eval_csv_path):
                    raise FileNotFoundError(f"Could not find bu48 CSV file at {eval_csv_path}")
                
                print(f"Loading bu48 evaluation dataset from {eval_csv_path}")
                # Load bu48 evaluation data
                eval_data = pd.read_csv(eval_csv_path)
                
                # Extract file names for loading ESM embeddings
                file_names = eval_data['file_name'].values if 'file_name' in eval_data.columns else []
                
                # Removing the file_name column if it exists
                if 'file_name' in eval_data.columns:
                    eval_data = eval_data.drop('file_name', axis=1)
                
                X_test = eval_data.drop('class', axis=1).values
                y_test = eval_data['class'].values
                
                # Load ESM embeddings for test data
                esm_embeddings_test = []
                if file_names:
                    for file_id in file_names:
                        esm_embeddings_test.append(self._load_esm_embedding(file_id))
                else:
                    # If file_names are not available, we'll have to use protein IDs from the ESM directory
                    print("Warning: No file_name column found, matching by protein IDs might be inaccurate")
                    # Get all available embeddings
                    available_embeddings = [os.path.basename(f).replace('.pt', '') 
                                          for f in glob(os.path.join(self.hparams.esm_dir, "*.pt"))]
                    # Use the first len(X_test) embeddings (this is suboptimal)
                    for i, file_id in enumerate(available_embeddings[:len(X_test)]):
                        esm_embeddings_test.append(self._load_esm_embedding(file_id))
                        
                # Stack ESM embeddings
                if esm_embeddings_test:
                    esm_test_tensor = torch.stack(esm_embeddings_test)
                    
                    # Combine with original features
                    X_test_combined = np.hstack([X_test, esm_test_tensor.numpy()])
                else:
                    print("Warning: No ESM embeddings loaded for test data")
                    X_test_combined = X_test
                
                # Print class distribution in evaluation set
                unique, counts = np.unique(y_test, return_counts=True)
                print(f"Bu48 evaluation set class distribution: {dict(zip(unique, counts))}")
                
                # Convert to PyTorch tensors - use Float for TabNet compatibility
                self.data_test = TensorDataset(
                    torch.FloatTensor(X_test_combined),
                    torch.FloatTensor(y_test)
                )
                
                print(f"Bu48 evaluation set: {len(self.data_test)} samples")
                print(f"Feature dimensions with ESM: {X_test_combined.shape[1]}")
            return

        # Load training/validation data (if not in test-only mode with bu48)
        if not self.data_train and not self.data_val and not self.data_test:
            # Load training data - using correct path structure
            train_csv_path = f"{self.hparams.data_dir}/train/chen11.csv"
            if not os.path.exists(train_csv_path):
                train_csv_path = f"{self.hparams.data_dir}/chen11/vectorsTrain.csv"
                print(f"Trying alternative path {train_csv_path}...")
                
            if not os.path.exists(train_csv_path):
                raise FileNotFoundError(f"Could not find chen11 CSV file at {train_csv_path}")
                
            print(f"Loading training data from {train_csv_path}")
            train_data = pd.read_csv(train_csv_path)
            
            # Extract file names for loading ESM embeddings
            file_names = train_data['file_name'].values if 'file_name' in train_data.columns else []
            
            # Display some file names for debugging
            if file_names:
                print(f"Found {len(file_names)} file names in the CSV. First 5: {file_names[:5]}")
            else:
                print("WARNING: No file_name column found in the CSV!")
            
            # Removing the file_name column as it's not a feature
            if 'file_name' in train_data.columns:
                train_data = train_data.drop('file_name', axis=1)
                
            X = train_data.drop('class', axis=1).values
            y = train_data['class'].values
            
            # Load ESM embeddings for all data
            esm_embeddings_list = []
            matched_count = 0
            missing_count = 0
            
            if file_names:
                print(f"Loading ESM embeddings for {len(file_names)} files...")
                for idx, file_id in enumerate(file_names):
                    embedding = self._load_esm_embedding(file_id)
                    esm_embeddings_list.append(embedding)
                    
                    # Track matches/misses
                    if file_id in self.esm_embeddings:
                        matched_count += 1
                    else:
                        missing_count += 1
                        
                    # Show progress
                    if idx % 1000 == 0 and idx > 0:
                        print(f"Processed {idx}/{len(file_names)} files...")
                
                print(f"ESM embedding loading complete: {matched_count} matches, {missing_count} misses")
            else:
                # If file_names are not available, we'll have to use protein IDs from the ESM directory
                print("Warning: No file_name column found, matching by protein IDs might be inaccurate")
                # Get all available embeddings
                available_embeddings = [os.path.basename(f).replace('.pt', '') 
                                      for f in glob(os.path.join(self.hparams.esm_dir, "*.pt"))]
                print(f"Found {len(available_embeddings)} ESM embeddings in directory")
                # Use the first len(X) embeddings (this is suboptimal)
                for i, file_id in enumerate(available_embeddings[:len(X)]):
                    esm_embeddings_list.append(self._load_esm_embedding(file_id))
                    
            # Stack ESM embeddings
            if esm_embeddings_list:
                esm_tensor = torch.stack(esm_embeddings_list)
                
                # Combine with original features
                X_combined = np.hstack([X, esm_tensor.numpy()])
                print(f"Combined features: Original {X.shape[1]} + ESM {esm_tensor.shape[1]} = {X_combined.shape[1]}")
            else:
                print("Warning: No ESM embeddings loaded")
                X_combined = X
            
            # Print class distribution before sampling
            unique, counts = np.unique(y, return_counts=True)
            print(f"Class distribution before sampling: {dict(zip(unique, counts))}")
            
            # Train/val/test split
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_combined, y, 
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
            print(f"Feature dimensions with ESM: {X_combined.shape[1]}")

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
