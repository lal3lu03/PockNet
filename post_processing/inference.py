"""
Model Inference Pipeline
========================

Handles loading PockNet models from checkpoints and extracting residue-level
predictions for post-processing. Supports both single models and multi-seed
ensembling.

Key features:
- PyTorch Lightning checkpoint loading
- Data preprocessing for inference
- Batch prediction extraction
- Integration with WandB model paths

Author: PockNet Team
"""

import os
import glob
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import h5py
import pandas as pd
from dataclasses import dataclass
import sys

# Configure logging first
logger = logging.getLogger(__name__)

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

try:
    from models.esm_tabular_module import EsmTabularModule
    from data.shared_memory_datamodule_v2 import (
        TrueSharedMemoryDataModule, 
        _prepare_memmaps_from_h5, 
        _attach_memmaps,
        _ddp_barrier
    )
    LIGHTNING_AVAILABLE = True
    logger.info("Successfully imported PockNet Lightning modules")
except ImportError as e:
    logger.warning(f"Lightning imports failed: {e}. Using placeholder models.")
    EsmTabularModule = None
    TrueSharedMemoryDataModule = None
    _prepare_memmaps_from_h5 = None
    _attach_memmaps = None
    _ddp_barrier = None
    LIGHTNING_AVAILABLE = False

@dataclass
class ModelConfig:
    """Configuration for model inference."""
    checkpoint_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_workers: int = 4


class SharedMemoryManager:
    """
    Manages shared memory loading for H5 data to eliminate redundant loading
    during ensemble inference.
    """
    
    def __init__(self):
        self._shared_data = None
        self._is_prepared = False
        
    def prepare_shared_memory(self, h5_path: str, rank: int = 0) -> bool:
        """
        Prepare H5 data in shared memory (rank 0) or attach to existing (other ranks).
        
        Args:
            h5_path: Path to H5 file
            rank: Process rank (0 for preparation, others for attachment)
            
        Returns:
            True if successful, False otherwise
        """
        if not LIGHTNING_AVAILABLE or _prepare_memmaps_from_h5 is None:
            logger.warning("Shared memory functionality not available, falling back to normal loading")
            return False
            
        try:
            h5_path = Path(h5_path)
            
            # Prepare shared memory (rank 0 loads data, others wait)
            if rank == 0:
                logger.info(f"🚀 Preparing shared memory for: {h5_path}")
                _prepare_memmaps_from_h5(h5_path, rank)
            
            # All ranks attach to shared memory
            logger.info(f"📎 Attaching to shared memory (rank: {rank})")
            self._shared_data = _attach_memmaps(rank)
            self._is_prepared = True
            
            # Log memory usage
            if self._shared_data:
                esm_shape = self._shared_data["esm"].shape
                tab_shape = self._shared_data["tabular"].shape
                logger.info(f"✅ Shared memory ready - ESM: {esm_shape}, Tabular: {tab_shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare shared memory: {e}")
            self._is_prepared = False
            return False
    
    def get_data_for_proteins(self, protein_list: List[str]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract data for specific proteins from shared memory.
        
        Args:
            protein_list: List of protein IDs to extract
            
        Returns:
            Dictionary with tensors for the specified proteins, or None if not available
        """
        if not self._is_prepared or self._shared_data is None:
            return None
            
        try:
            # Get protein keys (decode from bytes)
            protein_keys = self._shared_data["protein_keys"]
            decoded_keys = [key.decode() if isinstance(key, bytes) else str(key) for key in protein_keys]
            
            # Find indices for requested proteins
            protein_indices = []
            for protein in protein_list:
                indices = [i for i, key in enumerate(decoded_keys) if protein in key]
                protein_indices.extend(indices)
            
            if not protein_indices:
                logger.warning(f"No data found for proteins: {protein_list}")
                return None
            
            # Extract data for these indices
            result = {
                "esm": self._shared_data["esm"][protein_indices],
                "tabular": self._shared_data["tabular"][protein_indices],
                "labels": self._shared_data["labels"][protein_indices],
                "residue_numbers": self._shared_data["residue_numbers"][protein_indices],
                "protein_keys": [decoded_keys[i] for i in protein_indices]
            }
            
            logger.info(f"📊 Extracted {len(protein_indices)} samples for {len(protein_list)} proteins")
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract protein data from shared memory: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if shared memory is prepared and available."""
        return self._is_prepared and self._shared_data is not None
    
    def cleanup(self):
        """Clean up shared memory resources."""
        self._shared_data = None
        self._is_prepared = False


# Global shared memory manager instance
_shared_memory_manager = SharedMemoryManager()


class ModelInference:
    """
    Handles model loading and inference for residue-level predictions.
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 device: Optional[str] = None,
                 model_class: Optional[Any] = None):
        """
        Initialize model inference.
        
        Args:
            checkpoint_path: Path to PyTorch Lightning checkpoint
            device: Computing device ("cuda", "cpu", or None for auto)
            model_class: Model class for loading (if None, auto-detect)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_class = model_class
        
        logger.info(f"Initializing ModelInference on {self.device}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        
        self._load_model()
    
    def _load_model(self):
        """Load model from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        try:
            # Load checkpoint with weights_only=False to handle OmegaConf configs
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            if self.model_class is None:
                # Use EsmTabularModule as default if available
                if LIGHTNING_AVAILABLE and EsmTabularModule is not None:
                    self.model_class = EsmTabularModule
                    logger.info("Using EsmTabularModule for model loading")
                else:
                    # Try to auto-detect model class from checkpoint
                    self.model_class = self._detect_model_class(checkpoint)
            
            # Load model using Lightning's checkpoint loading
            if LIGHTNING_AVAILABLE and hasattr(self.model_class, 'load_from_checkpoint'):
                self.model = self.model_class.load_from_checkpoint(
                    self.checkpoint_path, 
                    map_location=self.device
                )
            else:
                # Fallback for non-Lightning models
                logger.warning("Using fallback model loading (non-Lightning)")
                self.model = self._load_fallback_model(checkpoint)
            
            self.model.eval()
            self.model.to(self.device)
            
            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(f"Available model classes: EsmTabularModule={'available' if EsmTabularModule else 'unavailable'}")
            raise
    
    def _detect_model_class(self, checkpoint: Dict) -> Any:
        """Auto-detect model class from checkpoint."""
        # Check for EsmTabularModule first (our primary model)
        if LIGHTNING_AVAILABLE and EsmTabularModule is not None:
            logger.info("Auto-detected EsmTabularModule from checkpoint")
            return EsmTabularModule
        
        # Check checkpoint metadata for model class information
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            if '_target_' in hparams:
                target = hparams['_target_']
                if 'EsmTabularModule' in target:
                    logger.info(f"Found EsmTabularModule in checkpoint target: {target}")
                    if EsmTabularModule is not None:
                        return EsmTabularModule
        
        # Fallback warning with more helpful information
        logger.warning("Could not auto-detect model class. Available classes:")
        logger.warning(f"  EsmTabularModule: {'available' if EsmTabularModule else 'unavailable'}")
        logger.warning("  Please provide model_class explicitly to ModelInference.")
        
        # Provide more helpful placeholder
        class PlaceholderModel:
            @classmethod
            def load_from_checkpoint(cls, path, map_location=None):
                raise NotImplementedError(
                    "Auto-detection failed. Please provide model_class to ModelInference.\n"
                    f"Available: EsmTabularModule={'✓' if EsmTabularModule else '✗'}"
                )
        
        return PlaceholderModel
    
    def _load_fallback_model(self, checkpoint: Dict) -> Any:
        """Fallback model loading for non-Lightning models."""
        logger.warning("Fallback model loading not yet implemented")
        raise NotImplementedError("Fallback model loading needs implementation")
    
    def predict_residues(self, 
                        data_loader: Any,
                        return_logits: bool = False) -> np.ndarray:
        """
        Extract residue-level predictions from data loader.
        
        Args:
            data_loader: PyTorch DataLoader with protein data
            return_logits: If True, return raw logits; if False, return probabilities
            
        Returns:
            Array of predictions (n_residues,)
        """
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Extract logits/predictions
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('predictions'))
                else:
                    logits = outputs
                
                if logits is None:
                    raise ValueError("Could not extract logits from model output")
                
                # Convert to probabilities if requested
                if return_logits:
                    batch_preds = logits.cpu().numpy()
                else:
                    batch_preds = torch.sigmoid(logits).cpu().numpy()
                
                predictions.append(batch_preds)
                
                if batch_idx % 10 == 0:
                    logger.debug(f"Processed batch {batch_idx}")
        
        # Concatenate all predictions
        all_predictions = np.concatenate(predictions, axis=0)
        
        logger.info(f"Extracted {len(all_predictions)} residue predictions")
        logger.info(f"Prediction range: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]")
        
        return all_predictions
    
    def predict_from_h5(self, 
                       h5_file: str,
                       protein_ids: Optional[List[str]] = None,
                       batch_size: int = 32,
                       use_shared_memory: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict from H5 file containing protein data.
        Uses shared memory when available for improved performance.
        
        Args:
            h5_file: Path to H5 file with protein features (e.g., pocknet_with_esm2_3b.h5)
            protein_ids: List of specific proteins to process (if None, process all)
            batch_size: Batch size for prediction
            use_shared_memory: Whether to use shared memory optimization
            
        Returns:
            Dictionary mapping protein_id -> predictions
        """
        if not LIGHTNING_AVAILABLE or TrueSharedMemoryDataModule is None:
            logger.error("TrueSharedMemoryDataModule not available. Cannot use H5 format.")
            return self._predict_from_h5_fallback(h5_file, protein_ids)
        
        # Try shared memory first if enabled
        if use_shared_memory and _shared_memory_manager.is_available():
            logger.info("🚀 Using shared memory for H5 prediction")
            return self._predict_from_shared_memory(protein_ids, batch_size)
        
        # Fallback to regular H5 loading
        logger.info("📂 Using regular H5 loading for prediction")
        return self._predict_from_h5_regular(h5_file, protein_ids, batch_size)
    
    def _predict_from_shared_memory(self, 
                                   protein_ids: Optional[List[str]], 
                                   batch_size: int) -> Dict[str, np.ndarray]:
        """Predict using shared memory data."""
        if protein_ids is None:
            # Get all available proteins from shared memory
            protein_keys = _shared_memory_manager._shared_data["protein_keys"]
            decoded_keys = [key.decode() if isinstance(key, bytes) else str(key) for key in protein_keys]
            protein_ids = list(set(decoded_keys))
        
        # Extract data for requested proteins
        shared_data = _shared_memory_manager.get_data_for_proteins(protein_ids)
        if shared_data is None:
            logger.error("Failed to extract data from shared memory")
            return {}
        
        # Group by protein
        protein_to_indices = {}
        for idx, protein_id in enumerate(shared_data["protein_keys"]):
            if protein_id not in protein_to_indices:
                protein_to_indices[protein_id] = []
            protein_to_indices[protein_id].append(idx)
        
        results = {}
        
        # Process each protein
        for protein_id in protein_ids:
            if protein_id not in protein_to_indices:
                continue
                
            try:
                indices = protein_to_indices[protein_id]
                n_residues = len(indices)
                
                logger.debug(f"Processing {protein_id}: {n_residues} residues")
                
                # Extract protein data (already filtered)
                protein_tabular = shared_data["tabular"][indices]  # (n_residues, 35)
                protein_esm = shared_data["esm"][indices]          # (n_residues, 2560)
                
                # Process in batches
                protein_predictions = []
                
                for i in range(0, n_residues, batch_size):
                    batch_end = min(i + batch_size, n_residues)
                    
                    # Create batch tensors
                    batch_tabular = protein_tabular[i:batch_end].to(self.device)
                    batch_esm = protein_esm[i:batch_end].to(self.device)
                    
                    # Predict
                    with torch.no_grad():
                        batch_pred = self.model(batch_tabular, batch_esm)
                        if hasattr(batch_pred, 'logits'):
                            batch_pred = batch_pred.logits
                        
                        batch_pred = torch.sigmoid(batch_pred).squeeze()
                        protein_predictions.append(batch_pred.cpu().numpy())
                
                # Combine all batches for this protein
                all_pred = np.concatenate(protein_predictions)
                results[protein_id] = all_pred
                
                logger.debug(f"Completed {protein_id}: {len(all_pred)} predictions")
                
            except Exception as e:
                logger.error(f"Error processing {protein_id}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(results)} proteins using shared memory")
        return results
    
    def _predict_from_h5_regular(self, 
                                h5_file: str,
                                protein_ids: Optional[List[str]],
                                batch_size: int) -> Dict[str, np.ndarray]:
        """Regular H5 loading without shared memory."""
        results = {}
        
        try:
            # Use the actual data loading logic from TrueSharedMemoryDataModule
            logger.info(f"Loading H5 data from: {h5_file}")
            
            with h5py.File(h5_file, 'r') as f:
                # Load data arrays (matching actual H5 structure)
                # Structure found: ['esm', 'feature_names', 'labels', 'protein_keys', 'residue_numbers', 'split', 'tabular']
                tabular_data = f['tabular'][:]  # Shape: (n_samples, 35)
                esm_data = f['esm'][:]          # Shape: (n_samples, 2560)
                labels = f['labels'][:]         # Shape: (n_samples,)
                residue_numbers = f['residue_numbers'][:] # Shape: (n_samples,)
                
                # Load protein keys
                protein_keys = [key.decode() if isinstance(key, bytes) else str(key) 
                               for key in f['protein_keys'][:]]
                
                logger.info(f"Loaded {len(tabular_data)} samples from H5")
                logger.info(f"ESM features shape: {esm_data.shape}")
                logger.info(f"Tabular features shape: {tabular_data.shape}")
                
                # Group by protein
                protein_to_indices = {}
                for idx, protein_id in enumerate(protein_keys):
                    if protein_id not in protein_to_indices:
                        protein_to_indices[protein_id] = []
                    protein_to_indices[protein_id].append(idx)
                
                available_proteins = list(protein_to_indices.keys())
                logger.info(f"Found {len(available_proteins)} unique proteins")
                
                if protein_ids is None:
                    protein_ids = available_proteins
                else:
                    # Check that requested proteins exist
                    missing = set(protein_ids) - set(available_proteins)
                    if missing:
                        logger.warning(f"Missing proteins in H5: {missing}")
                    protein_ids = [pid for pid in protein_ids if pid in available_proteins]
                
                logger.info(f"Processing {len(protein_ids)} proteins")
                
                # Process each protein
                for protein_id in protein_ids:
                    try:
                        indices = protein_to_indices[protein_id]
                        n_residues = len(indices)
                        
                        logger.debug(f"Processing {protein_id}: {n_residues} residues")
                        
                        # Extract protein data
                        protein_tabular = tabular_data[indices]  # (n_residues, 35)
                        protein_esm = esm_data[indices]          # (n_residues, 2560)
                        
                        # Process in batches
                        protein_predictions = []
                        
                        for i in range(0, n_residues, batch_size):
                            batch_end = min(i + batch_size, n_residues)
                            
                            # Create batch tensors
                            batch_tabular = torch.tensor(protein_tabular[i:batch_end], dtype=torch.float32)
                            batch_esm = torch.tensor(protein_esm[i:batch_end], dtype=torch.float32)
                            
                            # Predict
                            with torch.no_grad():
                                batch_pred = self.model(batch_tabular.to(self.device), batch_esm.to(self.device))
                                if hasattr(batch_pred, 'logits'):
                                    batch_pred = batch_pred.logits
                                
                                batch_pred = torch.sigmoid(batch_pred).squeeze()
                                protein_predictions.append(batch_pred.cpu().numpy())
                        
                        # Combine all batches for this protein
                        all_pred = np.concatenate(protein_predictions)
                        results[protein_id] = all_pred
                        
                        logger.debug(f"Completed {protein_id}: {len(all_pred)} predictions")
                        
                    except Exception as e:
                        logger.error(f"Error processing {protein_id}: {e}")
                        continue
                
        except Exception as e:
            logger.error(f"Failed to load H5 file: {e}")
            return self._predict_from_h5_fallback(h5_file, protein_ids)
        
        logger.info(f"Successfully processed {len(results)} proteins using regular H5 loading")
        return results
    
    def _predict_from_h5_fallback(self, h5_file: str, protein_ids: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Fallback H5 prediction method when SharedMemoryDataModuleV2 is not available."""
        logger.warning("Using fallback H5 prediction method")
        
        results = {}
        try:
            with h5py.File(h5_file, 'r') as f:
                # Try to understand the H5 structure
                logger.info(f"H5 file structure: {list(f.keys())}")
                
                # This is a simplified fallback - would need to be adapted based on actual H5 structure
                for key in f.keys():
                    if protein_ids is None or key in protein_ids:
                        try:
                            # Placeholder: create mock predictions
                            # In practice, you'd extract the actual data and run inference
                            mock_predictions = np.random.random(100)  # Replace with actual logic
                            results[key] = mock_predictions
                            logger.warning(f"Using mock predictions for {key}")
                        except Exception as e:
                            logger.error(f"Fallback failed for {key}: {e}")
                            
        except Exception as e:
            logger.error(f"Fallback H5 loading failed: {e}")
        
        return results
    
    def _load_protein_from_h5(self, h5_file: h5py.File, protein_id: str) -> Dict:
        """Load protein data from H5 file."""
        protein_group = h5_file[protein_id]
        
        data = {}
        for key in protein_group.keys():
            data[key] = protein_group[key][()]
        
        return data
    
    def _prepare_batch(self, protein_data: Dict) -> Dict:
        """Prepare protein data for model input."""
        batch = {}
        
        # This needs to be customized based on your model's expected input format
        # Here's a generic example that you should adapt:
        
        for key, value in protein_data.items():
            if isinstance(value, np.ndarray):
                # Add batch dimension and convert to tensor
                tensor_value = torch.from_numpy(value).unsqueeze(0)
                batch[key] = tensor_value
            else:
                batch[key] = value
        
        return batch


def load_checkpoint(checkpoint_path: str, 
                   model_class: Optional[Any] = None,
                   device: Optional[str] = None) -> Any:
    """
    Convenience function to load a single checkpoint.
    
    Args:
        checkpoint_path: Path to PyTorch Lightning checkpoint
        model_class: Model class for loading (if None, use EsmTabularModule if available)
        device: Computing device
        
    Returns:
        Loaded model
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_class is None:
        if LIGHTNING_AVAILABLE and EsmTabularModule is not None:
            model_class = EsmTabularModule
            logger.info("Using EsmTabularModule for checkpoint loading")
        else:
            raise ValueError("No model class provided and EsmTabularModule not available")
    
    # Load with weights_only=False to handle OmegaConf configs
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model.eval()
    model.to(device)
    
    return model


def extract_predictions(model: Any,
                       data_loader: Any,
                       device: Optional[str] = None) -> np.ndarray:
    """
    Extract predictions from a loaded model and data loader.
    
    Args:
        model: Loaded PyTorch model
        data_loader: PyTorch DataLoader
        device: Computing device
        
    Returns:
        Predictions array
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(device)
            
            outputs = model(batch)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('predictions'))
            else:
                logits = outputs
            
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.append(probs)
    
    return np.concatenate(predictions, axis=0)


def prepare_residue_data(predictions: np.ndarray,
                        coordinates: np.ndarray,
                        rsa_values: np.ndarray,
                        chain_ids: List[str],
                        residue_ids: List[int],
                        residue_names: Optional[List[str]] = None) -> List[Dict]:
    """
    Prepare residue data for post-processing.
    
    Args:
        predictions: Model predictions (n_residues,)
        coordinates: 3D coordinates (n_residues, 3)
        rsa_values: RSA values (n_residues,)
        chain_ids: Chain identifiers
        residue_ids: Residue numbers
        residue_names: Residue names (optional)
        
    Returns:
        List of residue dictionaries
    """
    if residue_names is None:
        residue_names = ['UNK'] * len(predictions)
    
    residues = []
    for i in range(len(predictions)):
        residue = {
            'chain': chain_ids[i],
            'res_id': residue_ids[i],
            'xyz': coordinates[i],
            'rsa': rsa_values[i],
            'prob': predictions[i],
            'res_name': residue_names[i]
        }
        residues.append(residue)
    
    return residues


class MultiSeedInference:
    """
    Handles inference across multiple seed models for ensembling.
    Uses ModelInference for consistent model loading.
    """
    
    def __init__(self, 
                 checkpoint_paths: List[str],
                 device: Optional[str] = None,
                 use_shared_memory: bool = True):
        """
        Initialize multi-seed inference.
        
        Args:
            checkpoint_paths: List of checkpoint paths for different seeds
            device: Computing device
            use_shared_memory: Whether to use shared memory optimization
        """
        self.checkpoint_paths = checkpoint_paths
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_instances = []
        self.use_shared_memory = use_shared_memory
        
        self._load_models()
    
    def prepare_shared_memory(self, h5_file: str) -> bool:
        """
        Prepare shared memory for H5 data to avoid redundant loading.
        
        Args:
            h5_file: Path to H5 file
            
        Returns:
            True if shared memory was prepared successfully
        """
        if not self.use_shared_memory:
            return False
            
        logger.info("🚀 Preparing shared memory for ensemble inference")
        return _shared_memory_manager.prepare_shared_memory(h5_file, rank=0)
    
    def _load_models(self):
        """Load all seed models using ModelInference."""
        logger.info(f"Loading {len(self.checkpoint_paths)} seed models")
        
        for i, path in enumerate(self.checkpoint_paths):
            try:
                # Use ModelInference for consistent model loading
                model_inf = ModelInference(
                    checkpoint_path=path,
                    device=self.device,
                    model_class=EsmTabularModule if LIGHTNING_AVAILABLE else None
                )
                self.model_instances.append(model_inf)
                logger.info(f"Loaded seed model {i+1}/{len(self.checkpoint_paths)}")
            except Exception as e:
                logger.error(f"Failed to load model {path}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(self.model_instances)} models")
    
    def predict_ensemble_from_h5(self, 
                                h5_file: str,
                                protein_ids: Optional[List[str]] = None,
                                prepare_shared_memory: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate ensemble predictions from H5 file across all seeds.
        
        Args:
            h5_file: Path to H5 file with protein features
            protein_ids: List of specific proteins to process
            prepare_shared_memory: Whether to prepare shared memory (set False if already prepared)
            
        Returns:
            Dictionary mapping protein_id -> ensemble predictions
        """
        if not self.model_instances:
            raise ValueError("No models loaded for ensemble prediction")
        
        # Prepare shared memory if requested and not already done
        if prepare_shared_memory and self.use_shared_memory:
            success = self.prepare_shared_memory(h5_file)
            if success:
                logger.info("✅ Shared memory prepared successfully")
            else:
                logger.warning("⚠️  Shared memory preparation failed, using regular loading")
        
        logger.info(f"Generating ensemble predictions from {len(self.model_instances)} seeds")
        
        # Collect predictions from all seeds
        all_seed_predictions = []
        
        for i, model_inf in enumerate(self.model_instances):
            logger.info(f"Generating predictions for seed {i+1}/{len(self.model_instances)}")
            
            try:
                # Use shared memory if available, otherwise fall back to regular loading
                seed_predictions = model_inf.predict_from_h5(
                    h5_file, 
                    protein_ids, 
                    use_shared_memory=self.use_shared_memory
                )
                all_seed_predictions.append(seed_predictions)
            except Exception as e:
                logger.error(f"Failed to get predictions from seed {i+1}: {e}")
                continue
        
        if not all_seed_predictions:
            raise RuntimeError("No successful predictions from any seed model")
        
        # Ensemble by averaging across seeds for each protein
        ensemble_results = {}
        
        # Get union of all proteins processed by at least one seed
        all_proteins = set()
        for seed_pred in all_seed_predictions:
            all_proteins.update(seed_pred.keys())
        
        logger.info(f"Ensembling predictions for {len(all_proteins)} proteins")
        
        for protein_id in all_proteins:
            protein_predictions = []
            
            # Collect predictions for this protein from all seeds that processed it
            for seed_pred in all_seed_predictions:
                if protein_id in seed_pred:
                    protein_predictions.append(seed_pred[protein_id])
            
            if protein_predictions:
                # Ensure all predictions have the same length
                lengths = [len(pred) for pred in protein_predictions]
                if len(set(lengths)) > 1:
                    logger.warning(f"Inconsistent prediction lengths for {protein_id}: {lengths}")
                    # Use minimum length to be safe
                    min_len = min(lengths)
                    protein_predictions = [pred[:min_len] for pred in protein_predictions]
                
                # Average across seeds
                ensemble_pred = np.mean(protein_predictions, axis=0)
                ensemble_results[protein_id] = ensemble_pred
                
                logger.debug(f"{protein_id}: ensemble from {len(protein_predictions)} seeds, "
                           f"range [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")
        
        logger.info(f"Ensemble complete for {len(ensemble_results)} proteins")
        return ensemble_results
    
    def predict_ensemble(self, data_loader: Any) -> np.ndarray:
        """
        Generate ensemble predictions using DataLoader (legacy method).
        
        Args:
            data_loader: PyTorch DataLoader
            
        Returns:
            Ensemble predictions
        """
        if not self.model_instances:
            raise ValueError("No models loaded for ensemble prediction")
        
        seed_predictions = []
        
        for i, model_inf in enumerate(self.model_instances):
            logger.info(f"Generating predictions for seed {i+1}")
            try:
                predictions = model_inf.predict_residues(data_loader)
                seed_predictions.append(predictions)
            except Exception as e:
                logger.error(f"Failed to get predictions from seed {i+1}: {e}")
                continue
        
        if not seed_predictions:
            raise RuntimeError("No successful predictions from any seed model")
        
        # Ensemble using mean
        ensemble = np.mean(seed_predictions, axis=0)
        
        logger.info(f"Generated ensemble from {len(seed_predictions)} seeds")
        return ensemble


def find_best_checkpoint(checkpoint_dir: str,
                        metric: str = "val_auprc") -> str:
    """
    Find the best checkpoint in a directory based on validation metric.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        metric: Validation metric to optimize for
        
    Returns:
        Path to best checkpoint
    """
    pattern = os.path.join(checkpoint_dir, f"*{metric}*.ckpt")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        # Fallback to any .ckpt file
        pattern = os.path.join(checkpoint_dir, "*.ckpt")
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
        logger.warning(f"No {metric} checkpoints found, using {checkpoints[0]}")
        return checkpoints[0]
    
    # Extract metric values and find best
    best_checkpoint = None
    best_value = -1
    
    for checkpoint in checkpoints:
        try:
            # Extract metric value from filename
            filename = os.path.basename(checkpoint)
            metric_part = filename.split(f"{metric}=")[1].split(".ckpt")[0]
            value = float(metric_part)
            
            if value > best_value:
                best_value = value
                best_checkpoint = checkpoint
                
        except (IndexError, ValueError):
            continue
    
    if best_checkpoint is None:
        logger.warning("Could not parse metric values, using first checkpoint")
        return checkpoints[0]
    
    logger.info(f"Best checkpoint: {best_checkpoint} ({metric}={best_value:.4f})")
    return best_checkpoint


# Example usage
if __name__ == "__main__":
    # Example usage (would need actual model classes and data)
    print("Model inference utilities initialized.")
    print("To use:")
    print("1. Import your model class")
    print("2. Create ModelInference instance with checkpoint path")
    print("3. Use predict_residues() or predict_from_h5() for inference")
    print("4. Use prepare_residue_data() to format for post-processing")