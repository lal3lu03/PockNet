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
import json
import pickle
from pathlib import Path


class BindingSiteESMResidueDataModule(LightningDataModule):
    """LightningDataModule for Binding Site prediction with residue-specific ESM2 embeddings.
    
    This module provides residue-specific ESM embeddings rather than global pooling,
    ensuring each SAS point gets the embedding of its corresponding residue.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        esm_dir: str = "data/esm2/",
        mapping_dir: str = "mappings/",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        sampling_strategy: str = "none",  # Options: "none", "oversample", "undersample", "combined"
        eval_dataset: str = "chen11",  # Options: "chen11", "bu48"
        fallback_strategy: str = "neighboring",  # Options: "neighboring", "global_mean", "zero"
    ) -> None:
        """Initialize a BindingSiteESMResidueDataModule.

        Args:
            data_dir: Directory containing the data files
            esm_dir: Directory containing the ESM2 embeddings
            mapping_dir: Directory containing residue mapping files
            train_val_test_split: Proportions for train/val/test split
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU training
            sampling_strategy: Strategy for handling imbalanced data
            eval_dataset: Dataset to use for evaluation/testing
            fallback_strategy: Strategy for handling residues without SAS points
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[TensorDataset] = None
        self.data_val: Optional[TensorDataset] = None
        self.data_test: Optional[TensorDataset] = None
        
        # Caches for embeddings and mappings
        self.esm_embeddings: Dict[str, torch.Tensor] = {}
        self.residue_mappings: Dict[str, Dict] = {}
        
    def _load_esm_embedding(self, file_id: str) -> torch.Tensor:
        """Load ESM embedding for a protein, returning full sequence embeddings."""
        if file_id in self.esm_embeddings:
            return self.esm_embeddings[file_id]
            
        # Convert file_id to embedding file name
        if file_id.endswith('.pdb'):
            file_id = file_id[:-4]  # Remove .pdb extension
            
        esm_file = os.path.join(self.hparams.esm_dir, f"{file_id}.pt")
        
        if not os.path.exists(esm_file):
            raise FileNotFoundError(f"ESM embedding file not found: {esm_file}")
            
        # Load embedding
        embedding = torch.load(esm_file, map_location='cpu')
        
        # Remove BOS and EOS tokens (first and last positions)
        if embedding.size(0) > 2:
            embedding = embedding[1:-1]  # Remove first and last tokens
        
        # Store in cache
        self.esm_embeddings[file_id] = embedding
        return embedding
    
    def _load_residue_mapping(self, protein_id: str) -> Dict:
        """Load residue mapping for a protein."""
        if protein_id in self.residue_mappings:
            return self.residue_mappings[protein_id]
            
        mapping_file = os.path.join(self.hparams.mapping_dir, f"{protein_id}_residue_mapping.json")
        
        if not os.path.exists(mapping_file):
            # Try to create mapping on-the-fly if possible
            print(f"Warning: Mapping file not found: {mapping_file}")
            print("Consider running create_residue_mapping.py to generate mappings")
            return None
            
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
            
        # Convert string keys back to integers for easier access
        if 'esm_to_csv_residue' in mapping:
            mapping['esm_to_csv_residue'] = {int(k): v for k, v in mapping['esm_to_csv_residue'].items()}
        if 'csv_residue_to_esm' in mapping:
            mapping['csv_residue_to_esm'] = {int(k): v for k, v in mapping['csv_residue_to_esm'].items()}
            
        self.residue_mappings[protein_id] = mapping
        return mapping
    
    def _get_residue_embedding(
        self, 
        esm_embedding: torch.Tensor, 
        residue_number: int, 
        protein_mapping: Dict,
        protein_id: str
    ) -> torch.Tensor:
        """Get ESM embedding for a specific residue number."""
        # Check if this residue has a direct ESM mapping
        if residue_number in protein_mapping.get('csv_residue_to_esm', {}):
            esm_pos = protein_mapping['csv_residue_to_esm'][residue_number]
            esm_idx = esm_pos - 1  # Convert to 0-based indexing
            if 0 <= esm_idx < esm_embedding.size(0):
                return esm_embedding[esm_idx]
        
        # Fallback strategies for residues without direct mapping
        if self.hparams.fallback_strategy == "neighboring":
            # Use closest residue with available mapping
            available_residues = list(protein_mapping.get('csv_residue_to_esm', {}).keys())
            if available_residues:
                closest_residue = min(available_residues, key=lambda x: abs(x - residue_number))
                esm_pos = protein_mapping['csv_residue_to_esm'][closest_residue]
                esm_idx = esm_pos - 1
                if 0 <= esm_idx < esm_embedding.size(0):
                    return esm_embedding[esm_idx]
                    
        elif self.hparams.fallback_strategy == "global_mean":
            # Use mean of all residues
            return torch.mean(esm_embedding, dim=0)
            
        elif self.hparams.fallback_strategy == "zero":
            # Return zero vector
            return torch.zeros(esm_embedding.size(1))
        
        # Ultimate fallback: use first residue
        print(f"Warning: Using first residue as fallback for residue {residue_number} in {protein_id}")
        return esm_embedding[0] if esm_embedding.size(0) > 0 else torch.zeros(esm_embedding.size(1))
    
    def _process_data_with_residue_embeddings(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Process dataframe to create features with residue-specific ESM embeddings."""
        print("Processing data with residue-specific ESM embeddings...")
        
        # Extract features and labels
        X_tabular = df.drop(['class', 'file_name', 'residue_number'], axis=1, errors='ignore').values
        y = df['class'].values
        
        # Group by protein for efficient processing
        protein_groups = df.groupby('file_name')
        
        esm_features_list = []
        processed_count = 0
        total_proteins = len(protein_groups)
        
        for protein_file, protein_data in protein_groups:
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processing protein {processed_count}/{total_proteins}: {protein_file}")
            
            # Extract protein ID
            protein_id = protein_file.replace('.pdb', '') if protein_file.endswith('.pdb') else protein_file
            
            try:
                # Load ESM embedding for this protein
                esm_embedding = self._load_esm_embedding(protein_id)
                
                # Load residue mapping
                protein_mapping = self._load_residue_mapping(protein_id)
                
                if protein_mapping is None:
                    # Fallback to global pooling if no mapping available
                    print(f"Warning: No mapping for {protein_id}, using global mean")
                    global_embedding = torch.mean(esm_embedding, dim=0)
                    protein_esm_features = global_embedding.numpy().reshape(1, -1)
                    protein_esm_features = np.tile(protein_esm_features, (len(protein_data), 1))
                else:
                    # Get residue-specific embeddings
                    protein_esm_features = []
                    for _, row in protein_data.iterrows():
                        residue_num = row['residue_number']
                        residue_embedding = self._get_residue_embedding(
                            esm_embedding, residue_num, protein_mapping, protein_id
                        )
                        protein_esm_features.append(residue_embedding.numpy())
                    
                    protein_esm_features = np.array(protein_esm_features)
                
                esm_features_list.append(protein_esm_features)
                
            except Exception as e:
                print(f"Error processing {protein_id}: {e}")
                # Fallback: use zero embeddings
                fallback_dim = 2560  # ESM2 embedding dimension
                protein_esm_features = np.zeros((len(protein_data), fallback_dim))
                esm_features_list.append(protein_esm_features)
        
        # Concatenate all ESM features
        esm_features = np.vstack(esm_features_list)
        
        print(f"ESM features shape: {esm_features.shape}")
        print(f"Tabular features shape: {X_tabular.shape}")
        
        # Combine tabular and ESM features
        X_combined = np.hstack([X_tabular, esm_features])
        
        print(f"Combined features shape: {X_combined.shape}")
        return X_combined, y

    def prepare_data(self) -> None:
        """Download data if needed. This method is called only from a single GPU."""
        pass  # data is expected to be already downloaded

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and apply train/val/test split."""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) must be divisible by the number of devices ({self.trainer.world_size})"
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        
        # For test-only stages, check if we need to load the bu48 evaluation dataset
        if stage == "test" and self.hparams.eval_dataset == "bu48":
            if not self.data_test:
                eval_csv_path = f"{self.hparams.data_dir}/test/bu48.csv"
                if not os.path.exists(eval_csv_path):
                    eval_csv_path = f"{self.hparams.data_dir}/bu48/vectorsEval.csv"
                    
                if not os.path.exists(eval_csv_path):
                    raise FileNotFoundError(f"Could not find bu48 CSV file at {eval_csv_path}")
                
                print(f"Loading bu48 evaluation dataset from {eval_csv_path}")
                eval_data = pd.read_csv(eval_csv_path)
                
                # Process with residue-specific embeddings
                X_test, y_test = self._process_data_with_residue_embeddings(eval_data)
                
                self.data_test = TensorDataset(
                    torch.tensor(X_test, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.long)
                )
            return
        
        # Regular training setup
        if not self.data_train:
            train_csv_path = f"{self.hparams.data_dir}/train/{self.hparams.eval_dataset}.csv"
            if not os.path.exists(train_csv_path):
                train_csv_path = f"{self.hparams.data_dir}/{self.hparams.eval_dataset}/vectors.csv"
                
            if not os.path.exists(train_csv_path):
                raise FileNotFoundError(f"Could not find training CSV file at {train_csv_path}")
                
            print(f"Loading training data from {train_csv_path}")
            train_data = pd.read_csv(train_csv_path)
            
            # Check required columns
            required_cols = ['file_name', 'residue_number', 'class']
            missing_cols = [col for col in required_cols if col not in train_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Process with residue-specific embeddings
            X_combined, y = self._process_data_with_residue_embeddings(train_data)
            
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
            
            val_size = self.hparams.train_val_test_split[1] / (
                self.hparams.train_val_test_split[0] + self.hparams.train_val_test_split[1]
            )
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
                smote = SMOTE(random_state=42)
                X_temp, y_temp = smote.fit_resample(X_train, y_train)
                rus = RandomUnderSampler(random_state=42)
                X_train, y_train = rus.fit_resample(X_temp, y_temp)
                print("Applied combined SMOTE + undersampling")
            
            # Print final class distribution
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"Final training class distribution: {dict(zip(unique, counts))}")
            
            # Create datasets
            self.data_train = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            )
            self.data_val = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            )
            self.data_test = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long)
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the training DataLoader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation DataLoader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test DataLoader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.
        """
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state."""
        pass


if __name__ == "__main__":
    """Test the datamodule."""
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    datamodule = BindingSiteESMResidueDataModule(data_dir=str(root / "data"))
    datamodule.setup()
    
    # Test the dataloader
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch[0].shape}, {batch[1].shape}")
