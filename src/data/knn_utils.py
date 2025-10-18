"""
k-NN utilities for ESM embedding aggregation in PockNet.

This module provides efficient k-nearest neighbor computation for aggregating
ESM embeddings from multiple residues to SAS points, maintaining the same
2560-D embedding dimension while incorporating local context.
"""

import numpy as np
import torch
import h5py
from functools import lru_cache
from scipy.spatial import cKDTree
from typing import Tuple, Dict, Optional
import logging

log = logging.getLogger(__name__)


@lru_cache(maxsize=4096)
def compute_sas_residue_knn(
    protein_id: str, 
    sas_coords: Tuple[float, ...], 
    res_coords: Tuple[float, ...], 
    k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-nearest neighbors for SAS points to residues.
    
    Args:
        protein_id: Protein identifier (for cache key)
        sas_coords: Flattened SAS coordinates (for cache key)
        res_coords: Flattened residue coordinates (for cache key)
        k: Number of nearest neighbors
        
    Returns:
        Tuple of (indices, distances) arrays:
        - indices: (Ns, k) int64 array of nearest residue indices
        - distances: (Ns, k) float32 array of distances in Angstroms
    """
    # Reconstruct coordinates from flattened tuples
    sas_coords_array = np.array(sas_coords).reshape(-1, 3)
    res_coords_array = np.array(res_coords).reshape(-1, 3)
    
    if len(res_coords_array) == 0:
        # No residues available
        return np.zeros((len(sas_coords_array), k), dtype=np.int64), \
               np.full((len(sas_coords_array), k), np.inf, dtype=np.float32)
    
    # Build k-d tree for efficient nearest neighbor search
    tree = cKDTree(res_coords_array)
    
    # Query k nearest neighbors
    k_actual = min(k, len(res_coords_array))
    distances, indices = tree.query(sas_coords_array, k=k_actual)
    
    # Ensure 2D output even for k=1
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    
    # Pad with last index if k > available residues
    if k_actual < k:
        pad_width = k - k_actual
        # Pad indices with the last available residue
        last_idx = indices[:, -1:] if k_actual > 0 else np.zeros((len(indices), 1), dtype=indices.dtype)
        indices = np.concatenate([indices, np.tile(last_idx, (1, pad_width))], axis=1)
        
        # Pad distances with infinity
        inf_distances = np.full((len(distances), pad_width), np.inf, dtype=np.float32)
        distances = np.concatenate([distances, inf_distances], axis=1)
    
    return indices.astype(np.int64), distances.astype(np.float32)


def extract_residue_coordinates_from_h5(h5_path: str, protein_ids: Optional[list] = None) -> Dict[str, np.ndarray]:
    """
    Extract residue coordinates from H5 file if available.
    
    Args:
        h5_path: Path to H5 file
        protein_ids: List of protein IDs to extract (None for all)
        
    Returns:
        Dictionary mapping protein_id to residue coordinates (R, 3)
    """
    residue_coords = {}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check if residue coordinates are stored in H5
            if 'residue_coords' in f:
                if protein_ids is None:
                    protein_ids = list(f['residue_coords'].keys())
                
                for pid in protein_ids:
                    if pid in f['residue_coords']:
                        residue_coords[pid] = f['residue_coords'][pid][:]
            else:
                log.warning("No residue coordinates found in H5 file")
                
    except Exception as e:
        log.warning(f"Failed to extract residue coordinates from H5: {e}")
        
    return residue_coords


def get_protein_residue_coordinates(protein_id: str, pdb_path: Optional[str] = None) -> np.ndarray:
    """
    Extract CA coordinates for each residue from PDB file.
    
    Args:
        protein_id: Protein identifier
        pdb_path: Path to PDB file (if None, tries to infer from protein_id)
        
    Returns:
        Array of CA coordinates (R, 3)
    """
    from Bio import PDB
    
    if pdb_path is None:
        # Try to infer PDB path from protein_id
        # This is a fallback and may need adjustment based on your file structure
        pdb_path = f"/system/user/studentwork/hageneder/MSC/Practical_work/PockNet/data/pdb/{protein_id}.pdb"
    
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(protein_id, pdb_path)
        
        residue_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ':  # Skip non-standard residues
                        continue
                    try:
                        ca_atom = residue['CA']  # Get alpha carbon
                        residue_coords.append(ca_atom.coord)
                    except KeyError:
                        continue  # Skip residues without CA
            break  # Only use first model
            
        return np.array(residue_coords)
        
    except Exception as e:
        log.warning(f"Failed to extract residue coordinates for {protein_id}: {e}")
        return np.array([]).reshape(0, 3)


def aggregate_knn_embeddings(
    res_embeddings: torch.Tensor,
    knn_indices: torch.Tensor,
    knn_distances: torch.Tensor,
    k: int,
    weighting: str = "softmax",
    temperature: float = 2.0
) -> torch.Tensor:
    """
    Aggregate k nearest residue embeddings for each SAS point.
    
    Args:
        res_embeddings: Residue embeddings (R, D)
        knn_indices: k-NN indices (Ns, k_max) 
        knn_distances: k-NN distances (Ns, k_max)
        k: Number of neighbors to use (≤ k_max)
        weighting: Aggregation method ("softmax", "inverse", "uniform")
        temperature: Temperature for softmax weighting (Angstroms)
        
    Returns:
        Aggregated embeddings for SAS points (Ns, D)
    """
    # Use only first k neighbors
    indices = knn_indices[:, :k]  # (Ns, k)
    distances = knn_distances[:, :k]  # (Ns, k)
    
    # Guard against zero distances (exact overlaps)
    distances = torch.clamp(distances, min=1e-4)
    
    # Gather embeddings: (Ns, k, D)
    gathered_embeddings = res_embeddings.index_select(0, indices.reshape(-1))
    gathered_embeddings = gathered_embeddings.view(indices.size(0), indices.size(1), -1)
    
    # Compute weights based on distances
    if weighting == "softmax":
        # Softmax with temperature (closer residues get higher weight)
        weights = torch.softmax(-distances / temperature, dim=1)  # (Ns, k)
    elif weighting == "inverse":
        # Inverse distance weighting
        weights = 1.0 / distances  # (Ns, k)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    elif weighting == "uniform":
        # Uniform weighting (simple average)
        weights = torch.full_like(distances, 1.0 / distances.size(1))
    else:
        raise ValueError(f"Unknown weighting method: {weighting}")
    
    # Apply weights and sum: (Ns, D)
    aggregated = (gathered_embeddings * weights.unsqueeze(-1)).sum(dim=1)
    
    return aggregated


def prepare_knn_batch_data(
    batch: Dict,
    sas_coords: torch.Tensor,
    residue_coords_dict: Dict[str, np.ndarray],
    k: int = 3
) -> Dict:
    """
    Prepare k-NN data for a batch of samples.
    
    Args:
        batch: Original batch dictionary
        sas_coords: SAS coordinates tensor (B*Ns, 3) or list of coordinates
        residue_coords_dict: Dictionary mapping protein_id to residue coordinates
        k: Number of nearest neighbors
        
    Returns:
        Updated batch with k-NN indices and distances
    """
    # Implementation depends on your batch structure
    # This is a template that can be adapted
    
    protein_ids = batch.get('protein_id', [])
    if isinstance(protein_ids, torch.Tensor):
        protein_ids = [pid.item() if isinstance(pid, torch.Tensor) else pid for pid in protein_ids]
    
    # For each protein in batch, compute k-NN
    knn_indices_list = []
    knn_distances_list = []
    
    for pid in protein_ids:
        if pid in residue_coords_dict:
            res_coords = residue_coords_dict[pid]
            # Get SAS coordinates for this protein (implementation dependent)
            # This is a placeholder - adapt based on your batch structure
            protein_sas_coords = sas_coords  # You'll need to extract per-protein coords
            
            # Compute k-NN
            indices, distances = compute_sas_residue_knn(
                protein_id=pid,
                sas_coords=tuple(protein_sas_coords.flatten().tolist()),
                res_coords=tuple(res_coords.flatten().tolist()),
                k=k
            )
            
            knn_indices_list.append(torch.from_numpy(indices))
            knn_distances_list.append(torch.from_numpy(distances))
        else:
            # Fallback for missing residue coordinates
            log.warning(f"No residue coordinates found for protein {pid}")
            dummy_indices = torch.zeros((len(sas_coords), k), dtype=torch.long)
            dummy_distances = torch.full((len(sas_coords), k), float('inf'))
            knn_indices_list.append(dummy_indices)
            knn_distances_list.append(dummy_distances)
    
    # Add to batch
    if knn_indices_list:
        batch['knn_indices'] = torch.cat(knn_indices_list, dim=0)
        batch['knn_distances'] = torch.cat(knn_distances_list, dim=0)
    
    return batch