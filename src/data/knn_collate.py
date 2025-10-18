"""
Enhanced collate functions for handling k-NN data in PyTorch DataLoader.

This module provides collate functions that properly handle k-NN data
when batching samples from the dataset.
"""

import torch
import numpy as np
from typing import List, Dict, Any
import logging

log = logging.getLogger(__name__)


def knn_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that handles k-NN data in addition to standard tensors.
    
    Args:
        batch: List of sample dictionaries from dataset
        
    Returns:
        Batched dictionary with proper tensor formatting
    """
    # Standard keys that should be batched normally
    standard_keys = ['tabular', 'esm', 'label', 'residue_number', 'h5_index']
    
    # k-NN specific keys that need special handling
    knn_keys = ['knn_nearest_residues', 'knn_distances', 'sas_coords']
    
    # String keys that need to be kept as lists
    string_keys = ['protein_id']
    
    batched = {}
    
    # Handle standard tensor keys
    for key in standard_keys:
        if key in batch[0]:
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                batched[key] = torch.stack(values)
            else:
                # Convert to tensor if not already
                batched[key] = torch.tensor(values)
    
    # Handle string keys
    for key in string_keys:
        if key in batch[0]:
            batched[key] = [sample[key] for sample in batch]
    
    # Handle k-NN data if present
    for key in knn_keys:
        if key in batch[0]:
            values = []
            for sample in batch:
                value = sample[key]
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value)
                values.append(value)
            
            try:
                # Try to stack if all have same shape
                batched[key] = torch.stack(values)
            except RuntimeError:
                # If shapes don't match, keep as list
                batched[key] = values
                log.debug(f"Could not stack {key}, keeping as list due to shape mismatch")
    
    return batched


def enhanced_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Enhanced collate function with better error handling and logging.
    Supports both mean and transformer aggregation modes.
    
    For transformer mode, handles [k, D] neighbor tensors with proper padding/masking.
    
    Args:
        batch: List of sample dictionaries from dataset
        
    Returns:
        Batched dictionary with proper tensor formatting
    """
    if not batch:
        return {}
    
    # Get all keys from first sample
    all_keys = set(batch[0].keys())
    
    batched = {}
    
    # ⭐ NEW: Check if this is transformer mode (has neighbor tensors)
    is_transformer_mode = 'esm_neighbors' in batch[0]
    
    for key in all_keys:
        values = [sample.get(key) for sample in batch if key in sample]
        
        if not values:
            continue
        
        try:
            first_value = values[0]
            
            # ⭐ NEW: Special handling for transformer mode neighbor tensors
            if key in ['esm_neighbors', 'neighbor_distances', 'neighbor_resnums'] and is_transformer_mode:
                # These are [k, D] or [k] tensors that need to be stacked to [B, k, D] or [B, k]
                tensors = []
                for v in values:
                    if isinstance(v, np.ndarray):
                        v = torch.from_numpy(v)
                    elif not isinstance(v, torch.Tensor):
                        v = torch.tensor(v)
                    tensors.append(v)
                
                try:
                    # Stack to batch dimension: [B, k, ...] 
                    batched[key] = torch.stack(tensors)
                    
                    # For distances, create mask for valid neighbors
                    if key == 'neighbor_distances':
                        # Finite distances indicate valid neighbors
                        mask = torch.isfinite(batched[key]) & (batched[key] < 1e6)
                        batched['neighbor_mask'] = mask  # [B, k]
                    
                except RuntimeError as e:
                    log.warning(f"Could not stack {key}: {e}. Shapes: {[t.shape for t in tensors[:3]]}")
                    batched[key] = tensors
                continue
            
            # Handle different data types
            if isinstance(first_value, str):
                # ⭐ FIX: Collapse uniform string fields to scalar (e.g., aggregation_mode)
                # Check if all values are identical
                if all(v == first_value for v in values):
                    batched[key] = first_value  # Single scalar string
                else:
                    batched[key] = values  # Keep as list if values differ
            elif isinstance(first_value, (int, float)):
                # Convert scalars to tensor
                batched[key] = torch.tensor(values)
            elif isinstance(first_value, np.ndarray):
                # Convert numpy arrays to tensors and stack
                tensors = [torch.from_numpy(v) for v in values]
                try:
                    batched[key] = torch.stack(tensors)
                except RuntimeError:
                    # Shape mismatch, keep as list
                    batched[key] = tensors
            elif isinstance(first_value, torch.Tensor):
                # Stack tensors
                try:
                    batched[key] = torch.stack(values)
                except RuntimeError:
                    # Shape mismatch, keep as list
                    batched[key] = values
            else:
                # For other types, keep as list
                batched[key] = values
                
        except Exception as e:
            log.debug(f"Failed to batch key '{key}': {e}")
            # Keep as list if batching fails
            batched[key] = values
    
    return batched