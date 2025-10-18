"""
Enhanced SharedMemoryDataset with k-NN ESM aggregation support.

This extends the existing dataset to add k-NN functionality without breaking
the current pipeline. It extracts SAS coordinates from the existing data
and computes k-NN aggregation when needed.
"""

import torch
import numpy as np
import h5py
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from functools import lru_cache
from scipy.spatial import cKDTree
from pathlib import Path
import os

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_PDB_DIRECTORIES = [
    PROJECT_ROOT / "data" / "p2rank-datasets" / "joined" / "bu48",
    PROJECT_ROOT / "data" / "p2rank-datasets" / "joined" / "b210",
    PROJECT_ROOT / "data" / "p2rank-datasets" / "chen11_test_prep",
    PROJECT_ROOT / "data" / "orginal_pdb",
    PROJECT_ROOT / "data" / "pdb_cache",
]

# Try to import Bio for PDB parsing
try:
    from Bio import PDB
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    log.warning("BioPython not available, k-NN functionality will be limited")

@lru_cache(maxsize=4096)
def get_protein_residue_coords(protein_id: str, pdb_base_dir: str = None) -> np.ndarray:
    """
    Extract CA coordinates for residues from PDB file.
    
    Args:
        protein_id: Protein identifier (e.g., "1a4j_H" or "1a4j")
        pdb_base_dir: Base directory containing PDB files
        
    Returns:
        Array of CA coordinates (R, 3) or empty array if not found
    """
    if not BIOPYTHON_AVAILABLE:
        return np.array([]).reshape(0, 3)
    
    # Extract base protein ID (remove chain suffix if present)
    base_protein_id = protein_id.split('_')[0].lower()
    
    # Try different possible PDB paths
    possible_paths = []
    
    # If pdb_base_dir is provided, try direct paths first
    if pdb_base_dir:
        possible_paths.extend([
            Path(pdb_base_dir) / f"{protein_id}.pdb",
            Path(pdb_base_dir) / f"{base_protein_id}.pdb",
            Path(pdb_base_dir) / f"{protein_id}_A.pdb",
        ])
    
    # Try p2rank-datasets structure (the main location for training data)
    p2rank_base = PROJECT_ROOT / "data" / "p2rank-datasets" / "joined"
    if p2rank_base.exists():
        for subdir in ["bu48", "b210", "dt198", "astex", "conservation"]:
            subdir_path = p2rank_base / subdir
            if subdir_path.exists():
                possible_paths.extend([
                    subdir_path / f"{base_protein_id}.pdb",
                    subdir_path / f"{protein_id}.pdb",
                    subdir_path / f"{protein_id}_A.pdb",
                ])
    
    # Default paths to try
    for base_path in DEFAULT_PDB_DIRECTORIES:
        if not base_path.exists():
            continue
        candidate_dirs = [base_path]
        if base_path.is_dir():
            candidate_dirs.extend([p for p in base_path.iterdir() if p.is_dir()])
        for root in candidate_dirs:
            possible_paths.extend([
                root / f"{protein_id}.pdb",
                root / f"{base_protein_id}.pdb",
            ])
    
    # Try to find the PDB file
    for pdb_path in possible_paths:
        if pdb_path.exists():
            try:
                parser = PDB.PDBParser(QUIET=True)
                structure = parser.get_structure(base_protein_id, str(pdb_path))
                
                coords = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if residue.id[0] != ' ':  # Skip non-standard residues
                                continue
                            try:
                                ca_atom = residue['CA']
                                coords.append(ca_atom.coord)
                            except KeyError:
                                continue  # Skip residues without CA
                    break  # Only use first model
                
                if coords:
                    return np.array(coords)
                    
            except Exception as e:
                log.debug(f"Failed to parse PDB {pdb_path}: {e}")
                continue
    
    log.warning(f"No PDB file found for protein {protein_id}")
    return np.array([]).reshape(0, 3)

@lru_cache(maxsize=4096) 
def compute_sas_knn_indices(
    protein_id: str,
    sas_coords_tuple: Tuple[float, ...],
    res_coords_tuple: Tuple[float, ...], 
    k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-NN indices and distances for SAS points to residues.
    
    Args:
        protein_id: Protein identifier (for cache key)
        sas_coords_tuple: Flattened SAS coordinates (for cache)
        res_coords_tuple: Flattened residue coordinates (for cache)
        k: Number of nearest neighbors
        
    Returns:
        Tuple of (indices, distances):
        - indices: (Ns, k) array of nearest residue indices
        - distances: (Ns, k) array of distances in Angstroms
    """
    # Reconstruct arrays from tuples
    sas_coords = np.array(sas_coords_tuple).reshape(-1, 3)
    res_coords = np.array(res_coords_tuple).reshape(-1, 3)
    
    if len(res_coords) == 0:
        # No residues available
        n_sas = len(sas_coords)
        return (np.zeros((n_sas, k), dtype=np.int64), 
                np.full((n_sas, k), np.inf, dtype=np.float32))
    
    # Build k-d tree for efficient search
    tree = cKDTree(res_coords)
    
    # Query k nearest neighbors
    k_actual = min(k, len(res_coords))
    distances, indices = tree.query(sas_coords, k=k_actual)
    
    # Ensure 2D output
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    
    # Pad if needed
    if k_actual < k:
        pad_width = k - k_actual
        # Pad indices with last available residue
        last_idx = indices[:, -1:] if k_actual > 0 else np.zeros((len(indices), 1), dtype=indices.dtype)
        indices = np.concatenate([indices, np.tile(last_idx, (1, pad_width))], axis=1)
        
        # Pad distances with large values
        large_distances = np.full((len(distances), pad_width), 999.0, dtype=np.float32)
        distances = np.concatenate([distances, large_distances], axis=1)
    
    return indices.astype(np.int64), distances.astype(np.float32)

def extract_sas_coordinates_from_csv():
    """
    Extract SAS coordinates from the original CSV files.
    This is a fallback if coordinates aren't in H5.
    """
    # This would need to be implemented based on your CSV structure
    # For now, return None to indicate coordinates not available
    return None

class KNNCapableDataset:
    """
    Mixin class to add k-NN capabilities to existing datasets.
    """
    
    def __init__(self, enable_knn: bool = False, k_max: int = 3, pdb_base_dir: str = None):
        self.enable_knn = enable_knn
        self.k_max = k_max
        self.pdb_base_dir = pdb_base_dir
        self._protein_coords_cache = {}
        self._csv_coords_cache = {}  # Cache for CSV coordinates
        
    def _load_sas_coords_from_csv(self, protein_key: str, residue_number: int) -> Optional[np.ndarray]:
        """
        Load SAS coordinates from the original CSV file.
        
        Args:
            protein_key: Protein key in format protein_id_chain_id (e.g., "1a4j_H")
            residue_number: Residue number
            
        Returns:
            Array of [x, y, z] coordinates or None if not found
        """
        try:
            # Use the same CSV file that was used to generate the H5
            csv_path = "/system/user/studentwork/hageneder/MSC/Practical_work/PockNet/data/vectorsTrain_all_chainfix.csv"
            
            # Check if we have this protein cached
            if protein_key not in self._csv_coords_cache:
                # Load CSV data for this protein
                try:
                    # Read only the needed columns to save memory
                    df = pd.read_csv(csv_path, 
                                   usecols=['file_name', 'protein_id', 'chain_id', 'residue_number', 'x', 'y', 'z'])
                    
                    # Filter for this protein - need to construct protein_key from CSV columns
                    # First try to find direct matches
                    if 'protein_id' in df.columns and 'chain_id' in df.columns:
                        df['csv_protein_key'] = df['protein_id'].astype(str).str.lower() + '_' + df['chain_id'].astype(str)
                    else:
                        # Fallback: extract from file_name
                        df['csv_protein_key'] = df['file_name'].astype(str).str.replace('.pdb', '').str.lower()
                    
                    protein_df = df[df['csv_protein_key'] == protein_key.lower()]
                    
                    if len(protein_df) > 0:
                        # Create lookup dict: residue_number -> (x, y, z)
                        coords_dict = {}
                        for _, row in protein_df.iterrows():
                            res_num = int(row['residue_number'])
                            coords = np.array([float(row['x']), float(row['y']), float(row['z'])])
                            coords_dict[res_num] = coords
                        
                        self._csv_coords_cache[protein_key] = coords_dict
                        log.debug(f"Loaded {len(coords_dict)} SAS coordinates for {protein_key} from CSV")
                    else:
                        self._csv_coords_cache[protein_key] = {}
                        log.debug(f"No CSV data found for protein {protein_key}")
                        
                except Exception as e:
                    log.debug(f"Failed to load CSV data for {protein_key}: {e}")
                    self._csv_coords_cache[protein_key] = {}
            
            # Get coordinates for specific residue
            protein_coords = self._csv_coords_cache.get(protein_key, {})
            return protein_coords.get(residue_number)
            
        except Exception as e:
            log.debug(f"Failed to load SAS coordinates from CSV for {protein_key}, residue {residue_number}: {e}")
            return None
        
    def _get_knn_data(self, sample: Dict, protein_id: str) -> Dict:
        """
        Add k-NN data to sample if k-NN is enabled.
        
        Args:
            sample: Existing sample dictionary
            protein_id: Protein identifier
            
        Returns:
            Enhanced sample with k-NN data
        """
        if not self.enable_knn:
            return sample
            
        try:
            # Extract SAS coordinates from sample if available
            sas_coords = self._extract_sas_coords(sample)
            if sas_coords is None:
                log.debug(f"No SAS coordinates found for {protein_id}")
                return sample
            
            # Get residue coordinates
            res_coords = self._get_residue_coords(protein_id)
            if len(res_coords) == 0:
                log.debug(f"No residue coordinates found for {protein_id}")
                return sample
            
            # Compute k-NN
            knn_indices, knn_distances = compute_sas_knn_indices(
                protein_id=protein_id,
                sas_coords_tuple=tuple(sas_coords.flatten()),
                res_coords_tuple=tuple(res_coords.flatten()),
                k=self.k_max
            )
            
            # Add to sample
            sample['knn_indices'] = torch.from_numpy(knn_indices)
            sample['knn_distances'] = torch.from_numpy(knn_distances)
            sample['residue_coords'] = torch.from_numpy(res_coords)
            
        except Exception as e:
            log.debug(f"k-NN computation failed for {protein_id}: {e}")
            
        return sample
    
    def _extract_sas_coords(self, sample: Dict) -> Optional[np.ndarray]:
        """
        Extract SAS coordinates from original CSV data.
        
        Since H5 file doesn't contain coordinates, we need to reconstruct them
        from the original CSV file using protein_key and residue_number.
        """
        try:
            protein_key = sample.get('protein_id')  # This is actually protein_key in format protein_id_chain_id
            residue_number = sample.get('residue_number')
            
            if protein_key is None or residue_number is None:
                return None
            
            # Load coordinates from original CSV file
            coords = self._load_sas_coords_from_csv(protein_key, residue_number)
            if coords is not None:
                return coords.reshape(1, 3)  # Single SAS point
                
            return None
            
        except Exception as e:
            log.debug(f"Failed to extract SAS coordinates: {e}")
            return None
    
    def _get_residue_coords(self, protein_id: str) -> np.ndarray:
        """Get residue coordinates for a protein."""
        if protein_id not in self._protein_coords_cache:
            coords = get_protein_residue_coords(protein_id, self.pdb_base_dir)
            self._protein_coords_cache[protein_id] = coords
        return self._protein_coords_cache[protein_id]

# Function to monkey-patch existing SharedMemoryDataset
def enhance_dataset_with_knn(dataset_instance, enable_knn: bool = False, k_max: int = 3):
    """
    Enhance existing dataset instance with k-NN capabilities.
    """
    # Add k-NN functionality to existing dataset
    knn_mixin = KNNCapableDataset(enable_knn=enable_knn, k_max=k_max)
    
    # Store original __getitem__
    original_getitem = dataset_instance.__getitem__
    
    def enhanced_getitem(idx):
        sample = original_getitem(idx)
        if enable_knn and 'protein_id' in sample:
            sample = knn_mixin._get_knn_data(sample, sample['protein_id'])
        return sample
    
    # Replace __getitem__ method
    dataset_instance.__getitem__ = enhanced_getitem
    dataset_instance._knn_mixin = knn_mixin
    
    return dataset_instance
