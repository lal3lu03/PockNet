"""
Practical k-NN implementation for PockNet using existing H5 structure.

This implementation works with the current H5 file structure where each SAS point
is already mapped to its nearest residue. We simulate k-NN by finding other
SAS points from nearby residues within the same protein.
"""

import numpy as np
import torch
import h5py
from functools import lru_cache
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import os
import logging

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LEGACY_PDB_SEARCH_PATHS = [
    PROJECT_ROOT / "data" / "p2rank-datasets" / "joined" / "bu48",
    PROJECT_ROOT / "data" / "p2rank-datasets" / "joined" / "b210",
    PROJECT_ROOT / "data" / "p2rank-datasets" / "chen11_test_prep",
    PROJECT_ROOT / "data" / "orginal_pdb",
    PROJECT_ROOT / "data" / "pdb_cache",
]

# Try to import Bio for PDB parsing (optional for residue distance calculation)
try:
    from Bio import PDB
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


@lru_cache(maxsize=1024)
def get_protein_residue_sequence(protein_key: str, pdb_base_dir: str = None) -> Dict[int, np.ndarray]:
    """
    Extract residue positions from PDB file for distance calculation.
    
    Args:
        protein_key: Protein identifier (e.g., "1a4j_H")
        pdb_base_dir: Base directory containing PDB files
        
    Returns:
        Dictionary mapping residue_number -> CA coordinates
    """
    if not BIOPYTHON_AVAILABLE:
        return {}
    
    # Extract base protein ID and chain ID
    if '_' in protein_key:
        base_protein_id, chain_id = protein_key.split('_', 1)
    else:
        base_protein_id = protein_key
        chain_id = 'A'
    
    base_protein_id = base_protein_id.lower()
    
    # Try to find PDB file
    possible_paths = []
    
    if pdb_base_dir:
        pdb_base_path = Path(pdb_base_dir)
        possible_paths.extend([
            pdb_base_path / f"{protein_key}.pdb",
            pdb_base_path / f"{base_protein_id}.pdb",
        ])
    
    # Default search directories relative to the project
    for base_path in LEGACY_PDB_SEARCH_PATHS:
        if not base_path.exists():
            continue
        candidate_dirs = [base_path]
        if base_path.is_dir():
            candidate_dirs.extend([p for p in base_path.iterdir() if p.is_dir()])
        for root in candidate_dirs:
            possible_paths.extend([
                root / f"{protein_key}.pdb",
                root / f"{base_protein_id}.pdb",
            ])
    
    for pdb_path in possible_paths:
        if pdb_path.exists():
            try:
                parser = PDB.PDBParser(QUIET=True)
                structure = parser.get_structure(base_protein_id, str(pdb_path))
                
                residue_coords = {}
                for model in structure:
                    target_chain = None
                    
                    # Find the target chain
                    for chain in model:
                        if chain.id == chain_id.upper():
                            target_chain = chain
                            break
                    
                    if target_chain is None and len(list(model.get_chains())) > 0:
                        target_chain = next(iter(model.get_chains()))
                    
                    if target_chain:
                        for residue in target_chain:
                            if residue.id[0] != ' ':
                                continue
                            try:
                                ca_atom = residue['CA']
                                res_num = residue.id[1]
                                residue_coords[res_num] = ca_atom.coord
                            except KeyError:
                                continue
                    break
                
                return residue_coords
                
            except Exception as e:
                log.debug(f"Failed to parse PDB {pdb_path}: {e}")
                continue
    
    return {}


class PracticalKNNCache:
    """
    Practical k-NN implementation using existing H5 structure.
    
    Instead of requiring SAS coordinates, this finds k nearest residues
    by sequence distance and uses their ESM embeddings from the H5 file.
    """
    
    def __init__(self, h5_path: str, k_max: int = 3, pdb_base_dir: str = None):
        self.h5_path = h5_path
        self.k_max = k_max
        self.pdb_base_dir = pdb_base_dir
        self._protein_residue_coords = {}
        self._protein_h5_indices = {}
        self._load_h5_structure()
    
    def _load_h5_structure(self):
        """Load the H5 structure to build protein->residue->h5_index mappings."""
        with h5py.File(self.h5_path, 'r') as f:
            protein_keys = [k.decode('utf-8') if isinstance(k, bytes) else str(k) 
                           for k in f['protein_keys'][:]]
            residue_numbers = f['residue_numbers'][:]
        
        # Build mapping: protein_key -> residue_number -> list of h5_indices
        for h5_idx, (protein_key, res_num) in enumerate(zip(protein_keys, residue_numbers)):
            if protein_key not in self._protein_h5_indices:
                self._protein_h5_indices[protein_key] = {}
            
            if res_num not in self._protein_h5_indices[protein_key]:
                self._protein_h5_indices[protein_key][res_num] = []
            
            self._protein_h5_indices[protein_key][res_num].append(h5_idx)
    
    def get_knn_residues(self, protein_key: str, target_residue: int, k: int) -> List[int]:
        """
        Get k nearest residues for a given protein and target residue.
        
        Uses either:
        1. 3D distance if PDB coordinates are available
        2. Sequence distance as fallback
        
        Args:
            protein_key: Protein identifier
            target_residue: Target residue number
            k: Number of neighbors to find
            
        Returns:
            List of k nearest residue numbers (including target)
        """
        if protein_key not in self._protein_h5_indices:
            return [target_residue] * k
        
        available_residues = list(self._protein_h5_indices[protein_key].keys())
        
        if len(available_residues) <= k:
            # Not enough residues, return what we have
            return available_residues + [target_residue] * (k - len(available_residues))
        
        # Try to get 3D coordinates for distance calculation
        if protein_key not in self._protein_residue_coords:
            self._protein_residue_coords[protein_key] = get_protein_residue_sequence(
                protein_key, self.pdb_base_dir
            )
        
        residue_coords = self._protein_residue_coords[protein_key]
        
        if target_residue in residue_coords and len(residue_coords) >= k:
            # Use 3D distance
            target_coord = residue_coords[target_residue]
            distances = []
            
            for res_num in available_residues:
                if res_num in residue_coords:
                    dist = np.linalg.norm(residue_coords[res_num] - target_coord)
                    distances.append((dist, res_num))
                else:
                    # Use large distance for residues without coordinates
                    seq_dist = abs(res_num - target_residue)
                    distances.append((seq_dist + 1000, res_num))
            
            # Sort by distance and take k nearest
            distances.sort()
            return [res_num for _, res_num in distances[:k]]
        
        else:
            # Fallback to sequence distance
            distances = [(abs(res_num - target_residue), res_num) for res_num in available_residues]
            distances.sort()
            return [res_num for _, res_num in distances[:k]]
    
    def get_knn_embeddings(self, h5_idx: int, protein_key: str, target_residue: int, k: int) -> Optional[Dict]:
        """
        Get k-NN embedding data for a specific H5 index.
        
        Args:
            h5_idx: H5 index of the target SAS point
            protein_key: Protein identifier  
            target_residue: Target residue number
            k: Number of neighbors
            
        Returns:
            Dictionary with k-NN embedding indices and distances
        """
        # Get k nearest residues
        nearest_residues = self.get_knn_residues(protein_key, target_residue, k)
        
        # Get H5 indices for these residues
        knn_h5_indices = []
        knn_distances = []
        
        for i, res_num in enumerate(nearest_residues):
            if res_num in self._protein_h5_indices[protein_key]:
                # Take first H5 index for this residue (could be multiple SAS points per residue)
                res_h5_indices = self._protein_h5_indices[protein_key][res_num]
                knn_h5_indices.append(res_h5_indices[0])  # Take first one
                
                # Distance is either 3D distance or sequence distance
                if res_num == target_residue:
                    knn_distances.append(0.0)
                else:
                    # Use sequence distance as proxy (could be improved with 3D distance)
                    seq_dist = abs(res_num - target_residue)
                    knn_distances.append(float(seq_dist))
            else:
                # Fallback: use target index
                knn_h5_indices.append(h5_idx)
                knn_distances.append(999.0)
        
        # Ensure we have exactly k indices
        while len(knn_h5_indices) < k:
            knn_h5_indices.append(h5_idx)
            knn_distances.append(999.0)
        
        return {
            'knn_h5_indices': np.array(knn_h5_indices[:k], dtype=np.int64),
            'knn_distances': np.array(knn_distances[:k], dtype=np.float32),
            'nearest_residues': np.array(nearest_residues[:k], dtype=np.int64)
        }


def aggregate_knn_embeddings_h5(
    h5_embeddings: torch.Tensor,  # Full H5 ESM embeddings (N, 2560)
    knn_h5_indices: torch.Tensor,  # k-NN H5 indices (B, k)
    knn_distances: torch.Tensor,   # k-NN distances (B, k)
    k: int,
    weighting: str = "softmax",
    temperature: float = 2.0
) -> torch.Tensor:
    """
    Aggregate k-NN embeddings using H5 indices.
    
    Args:
        h5_embeddings: Full H5 ESM embeddings tensor (N, 2560)
        knn_h5_indices: k-NN H5 indices (B, k)  
        knn_distances: k-NN distances (B, k)
        k: Number of neighbors to use
        weighting: Aggregation method
        temperature: Temperature for softmax weighting
        
    Returns:
        Aggregated embeddings (B, 2560)
    """
    batch_size = knn_h5_indices.shape[0]
    embed_dim = h5_embeddings.shape[1]
    device = knn_h5_indices.device
    
    # Use only first k neighbors
    indices = knn_h5_indices[:, :k]  # (B, k)
    distances = knn_distances[:, :k]  # (B, k)
    
    # Guard against zero/negative distances
    distances = torch.clamp(distances, min=1e-4)
    
    # Gather embeddings: (B, k, D)
    gathered_embeddings = h5_embeddings.index_select(0, indices.reshape(-1))
    gathered_embeddings = gathered_embeddings.view(batch_size, k, embed_dim)
    
    # Compute weights based on distances
    if weighting == "softmax":
        # Softmax with temperature (closer residues get higher weight)
        weights = torch.softmax(-distances / temperature, dim=1)  # (B, k)
    elif weighting == "inverse":
        # Inverse distance weighting
        weights = 1.0 / distances  # (B, k)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    elif weighting == "uniform":
        # Uniform weighting (simple average)
        weights = torch.full_like(distances, 1.0 / k)
    else:
        weights = torch.full_like(distances, 1.0 / k)
    
    # Apply weights and sum: (B, D)
    aggregated = (gathered_embeddings * weights.unsqueeze(-1)).sum(dim=1)
    
    return aggregated
