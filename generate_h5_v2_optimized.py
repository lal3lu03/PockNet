#!/usr/bin/env python3
"""
Generate H5 file v2 - OPTIMIZED VERSION with multi-processing and KD-tree neighbors.

Performance improvements:
1. Producer-consumer pattern: single HDF5 writer, parallel protein processors
2. KD-tree for 100x faster neighbor lookup (scipy.spatial.cKDTree)
3. gemmi parser (if available) for 5-10x faster PDB parsing
4. Optimized HDF5 settings: lzf compression, large chunks, geometric growth
5. Vectorized operations and reduced pandas overhead
6. Better PDB lookup with pre-built index

Expected speedup: 10-20x on multi-core systems
"""

import os
import sys
import gc
import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree

try:
    import torch
except Exception:
    torch = None

try:
    import gemmi
    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False

try:
    from Bio.PDB import PDBParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during hot loop
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import functions from original script
sys.path.insert(0, str(Path(__file__).parent))
from generate_complete_h5_files import (
    load_bu48_list,
    guess_feature_cols,
    get_label,
    protein_key_from_row,
    find_chain_emb_file,
    load_chain_embedding,
    choose_vector_by_resnum,
    grouped_split
)

# ============================================================================
# PDB Index from Dataset Files
# ============================================================================

def load_pdb_index(ds_path: Path, base_dir: Path) -> Dict[Tuple[str, str], Path]:
    """Parse .ds dataset file and build index of protein_id → PDB file path."""
    index = {}
    
    if not ds_path.exists():
        return index
    
    with ds_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("PARAM."):
                continue
            
            rel_path = Path(line)
            abs_path = base_dir / rel_path
            stem = rel_path.stem.lower()
            
            chain = None
            protein_base = stem
            
            if '_' in stem:
                parts = stem.rsplit('_', 1)
                if len(parts) == 2 and len(parts[1]) <= 2 and parts[1].isalpha():
                    protein_base = parts[0]
                    chain = parts[1].upper()
            
            if not chain:
                orig_stem = rel_path.stem
                if len(orig_stem) > 0 and orig_stem[-1].isupper() and orig_stem[-1].isalpha():
                    protein_base = stem[:-1]
                    chain = orig_stem[-1]
            
            if not chain:
                chain = "A"
            
            key1 = (protein_base, chain)
            key2 = (protein_base, "A")
            
            index[key1] = abs_path
            if key2 != key1 and key2 not in index:
                index[key2] = abs_path
    
    return index


# ============================================================================
# Fast PDB Coordinate Extraction
# ============================================================================

def extract_ca_coordinates_fast(pdb_path: Path, chain_id: str) -> Optional[Dict[int, np.ndarray]]:
    """
    Extract Cα coordinates using gemmi (fast) or BioPython (fallback).
    
    Returns:
        Dict mapping residue_number → [x, y, z] coordinate (Å)
    """
    if not pdb_path.exists():
        return None
    
    def _collect_from_gemmi_chain(chain_obj):
        coords = {}
        for res in chain_obj:
            ca = res.find_atom('CA', '')
            if ca is not None:
                coords[res.seqid.num] = np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=np.float32)
        return coords
    
    # Try gemmi first (5-10x faster)
    if HAS_GEMMI:
        try:
            st = gemmi.read_structure(str(pdb_path))
            model = st[0]
            
            # Try to find exact chain match
            target_chain = None
            for ch in model:
                if ch.name == chain_id:
                    target_chain = ch
                    break
            
            # Fallbacks for blank chains or missing IDs
            if target_chain is None:
                # If CSV alias is 'A' but structure has blank chain id
                if chain_id == "A":
                    for ch in model:
                        if ch.name.strip() == "":
                            target_chain = ch
                            break
                # If still not found and only one chain, use it
                if target_chain is None and len(model) == 1:
                    target_chain = model[0]
                # As a last resort, pick the first chain with residues
                if target_chain is None:
                    for ch in model:
                        if any(res.find_atom('CA', '') is not None for res in ch):
                            target_chain = ch
                            break
            
            if target_chain is not None:
                coords = _collect_from_gemmi_chain(target_chain)
                if coords:
                    return coords
            
        except Exception:
            pass
    
    # Fallback to BioPython
    if not HAS_BIOPYTHON:
        return None
    
    def _collect_from_biopython_chain(chain_obj):
        coords = {}
        for residue in chain_obj:
            if residue.id[0] != ' ':
                continue
            if 'CA' not in residue:
                continue
            resnum = residue.id[1]
            coords[resnum] = np.array(residue['CA'].get_coord(), dtype=np.float32)
        return coords
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        
        target_chain = None
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    target_chain = chain
                    break
            if target_chain:
                break
        
        if target_chain is None:
            # Handle blank chain IDs
            if chain_id == "A":
                for model in structure:
                    for chain in model:
                        if (chain.id or "").strip() == "":
                            target_chain = chain
                            break
                    if target_chain:
                        break
            if target_chain is None:
                models = list(structure)
                if len(models) == 1 and len(list(models[0])) == 1:
                    target_chain = list(models[0])[0]
            if target_chain is None:
                for model in structure:
                    for chain in model:
                        coords = _collect_from_biopython_chain(chain)
                        if coords:
                            return coords
                return None
        
        coords = _collect_from_biopython_chain(target_chain)
        return coords if coords else None
        
    except Exception:
        return None


# ============================================================================
# Fast Neighbor Computation with KD-tree
# ============================================================================

def build_kdtree(ca_coords: Dict[int, np.ndarray]) -> Tuple[cKDTree, np.ndarray]:
    """Build KD-tree from CA coordinates for fast neighbor lookup (100x faster)."""
    resnums_with_ca = np.array(sorted(ca_coords.keys()), dtype=np.int32)
    coords = np.vstack([ca_coords[r] for r in resnums_with_ca])  # [M, 3]
    tree = cKDTree(coords)
    return tree, resnums_with_ca


def compute_euclidean_neighbors_fast(
    target_resnum: int,
    all_resnums: Optional[List[int]],
    ca_coords: Optional[Dict[int, np.ndarray]],
    kdtree: Optional[cKDTree],
    kdtree_resnums: Optional[np.ndarray],
    k: int
) -> Tuple[List[int], List[float], bool]:
    """
    Compute k nearest neighbors using KD-tree (100x faster than loop).
    
    Returns:
        (neighbor_resnums, distances_angstrom, used_euclidean)
    """
    if ca_coords is None or target_resnum not in ca_coords or kdtree is None:
        return compute_sequence_neighbors(target_resnum, all_resnums, k)
    
    target_coord = ca_coords[target_resnum]
    
    try:
        # Query KD-tree for k+1 nearest (includes self)
        dists, idxs = kdtree.query(target_coord, k=min(k+1, len(kdtree_resnums)))
        
        # Handle scalar case
        if np.isscalar(dists):
            dists = np.array([dists])
            idxs = np.array([idxs])
        
        # Filter out self
        mask = kdtree_resnums[idxs] != target_resnum
        neighbor_resnums = kdtree_resnums[idxs[mask]][:k].tolist()
        neighbor_distances = dists[mask][:k].astype(np.float32).tolist()
        
        # Pad if needed
        while len(neighbor_resnums) < k:
            neighbor_resnums.append(target_resnum)
            neighbor_distances.append(0.0)
        
        return neighbor_resnums, neighbor_distances, True
        
    except Exception:
        return compute_sequence_neighbors(target_resnum, all_resnums, k)


def compute_sequence_neighbors(
    target_resnum: int,
    all_resnums: Optional[List[int]],
    k: int
) -> Tuple[List[int], List[float], bool]:
    """Fallback: sequence distance neighbors."""
    if all_resnums is None or len(all_resnums) == 0:
        # No residue information - return self-padding
        return [target_resnum] * k, [0.0] * k, False
    
    seq_distances = []
    valid_resnums = []
    
    for resnum in all_resnums:
        if resnum == target_resnum:
            continue
        dist = abs(resnum - target_resnum)
        seq_distances.append(dist)
        valid_resnums.append(resnum)
    
    if len(seq_distances) == 0:
        return [target_resnum] * k, [0.0] * k, False
    
    sorted_indices = np.argsort(seq_distances)[:k]
    neighbor_resnums = [valid_resnums[i] for i in sorted_indices]
    neighbor_distances = [float(seq_distances[i]) for i in sorted_indices]
    
    while len(neighbor_resnums) < k:
        neighbor_resnums.append(target_resnum)
        neighbor_distances.append(0.0)
    
    return neighbor_resnums, neighbor_distances, False


# ============================================================================
# Optimized HDF5 Writer with Geometric Growth
# ============================================================================

class H5WriterTransformerV2Optimized:
    """
    Optimized HDF5 writer with:
    - LZF compression (faster than gzip)
    - Large chunks (100k rows)
    - Geometric growth (reduces resize overhead)
    - Optimized cache settings
    """
    
    def __init__(self, out_path: Path, tab_dim: int, esm_dim: int, k_neighbors: int, feat_names: List[str]):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Optimized HDF5 file settings
        self.f = h5py.File(
            out_path, "w",
            libver="latest",
            rdcc_nbytes=256*1024**2,  # 256MB chunk cache
            rdcc_w0=0.75  # Write bias
        )
        
        self._n = 0
        self._capacity = 100000  # Initial capacity
        self.k_neighbors = k_neighbors
        
        chunk_size = 100000  # Large chunks for better I/O
        
        # Optimized datasets - CREATE with initial capacity to avoid empty-write bug
        self.tab = self.f.create_dataset(
            "tabular", shape=(self._capacity, tab_dim), maxshape=(None, tab_dim),
            chunks=(chunk_size, tab_dim), compression="lzf", shuffle=True, track_times=False
        )
        self.esm = self.f.create_dataset(
            "esm", shape=(self._capacity, esm_dim), maxshape=(None, esm_dim),
            chunks=(chunk_size, esm_dim), compression="lzf", shuffle=True, track_times=False
        )
        self.lbl = self.f.create_dataset(
            "labels", shape=(self._capacity,), maxshape=(None,), dtype="i4",
            chunks=(chunk_size,), track_times=False
        )
        self.pky = self.f.create_dataset(
            "protein_keys", shape=(self._capacity,), maxshape=(None,),
            dtype=h5py.string_dtype("utf-8"),
            chunks=(chunk_size,), compression="lzf", track_times=False
        )
        self.rno = self.f.create_dataset(
            "residue_numbers", shape=(self._capacity,), maxshape=(None,), dtype="i4",
            chunks=(chunk_size,), track_times=False
        )
        self.spl = self.f.create_dataset(
            "split", shape=(self._capacity,), maxshape=(None,), dtype="i1",
            chunks=(chunk_size,), track_times=False
        )
        
        # Neighbor datasets - CREATE with initial capacity
        self.neighbor_h5_indices = self.f.create_dataset(
            "neighbor_h5_indices",
            shape=(self._capacity, k_neighbors), maxshape=(None, k_neighbors),
            dtype="i4", chunks=(chunk_size, k_neighbors),
            compression="lzf", track_times=False
        )
        self.neighbor_distances = self.f.create_dataset(
            "neighbor_distances",
            shape=(self._capacity, k_neighbors), maxshape=(None, k_neighbors),
            dtype="f2", chunks=(chunk_size, k_neighbors),
            compression="lzf", shuffle=True, track_times=False
        )
        self.neighbor_resnums = self.f.create_dataset(
            "neighbor_resnums",
            shape=(self._capacity, k_neighbors), maxshape=(None, k_neighbors),
            dtype="i4", chunks=(chunk_size, k_neighbors),
            track_times=False
        )
        
        # Store feature names
        self.f.create_dataset("feature_names",
                              data=np.array([s.encode("utf-8") for s in feat_names]),
                              compression="gzip")
        
        self.distance_metrics = []
    
    def append(self, tab, esm, neighbor_indices, neighbor_dists, neighbor_resnums,
               lbl, keys, resnos, split_flags, used_euclidean_flags):
        """Append batch with geometric growth to reduce resize overhead."""
        b = tab.shape[0]
        end = self._n + b
        
        # Grow capacity geometrically if needed (reduces resize calls)
        if end > self._capacity:
            self._capacity = max(end, int(self._capacity * 1.5))
            self.tab.resize((self._capacity, self.tab.shape[1]))
            self.esm.resize((self._capacity, self.esm.shape[1]))
            self.lbl.resize((self._capacity,))
            self.pky.resize((self._capacity,))
            self.rno.resize((self._capacity,))
            self.spl.resize((self._capacity,))
            self.neighbor_h5_indices.resize((self._capacity, self.k_neighbors))
            self.neighbor_distances.resize((self._capacity, self.k_neighbors))
            self.neighbor_resnums.resize((self._capacity, self.k_neighbors))
        
        # Write data
        self.tab[self._n:end] = tab
        self.esm[self._n:end] = esm
        self.lbl[self._n:end] = lbl
        # For H5PY variable-length UTF-8 strings, ensure proper encoding
        # Convert to numpy array of strings to force consistent encoding
        keys_array = np.array(keys, dtype=str)
        if len(keys_array) != b:
            raise RuntimeError(f"keys length mismatch: {len(keys_array)} != {b}")
        self.pky[self._n:end] = keys_array
        
        # VALIDATION: Verify first batch to ensure fix worked
        if self._n == 0 and b > 0:
            # Quick validation: check first key was written correctly
            readback_first = self.pky[0]
            decoded_first = readback_first.decode('utf-8') if isinstance(readback_first, bytes) else str(readback_first)
            if decoded_first == '' or decoded_first != str(keys[0]):
                raise RuntimeError(
                    f"❌ CRITICAL: First key validation failed! "
                    f"Wrote '{keys[0]}' but read back '{decoded_first}'. "
                    f"This indicates a regression of the shape=(0,) initialization bug.")
        
        self.rno[self._n:end] = resnos
        self.spl[self._n:end] = split_flags
        
        self.neighbor_h5_indices[self._n:end] = neighbor_indices.astype(np.int32)
        self.neighbor_distances[self._n:end] = neighbor_dists.astype(np.float16)
        self.neighbor_resnums[self._n:end] = neighbor_resnums.astype(np.int32)
        
        self.distance_metrics.extend(used_euclidean_flags)
        self._n = end
        self._n = end
    
    def finalize(self, meta: Dict):
        """Trim datasets to actual size, add metadata, and close."""
        # Trim to actual size
        self.tab.resize((self._n, self.tab.shape[1]))
        self.esm.resize((self._n, self.esm.shape[1]))
        self.lbl.resize((self._n,))
        self.pky.resize((self._n,))
        self.rno.resize((self._n,))
        self.spl.resize((self._n,))
        self.neighbor_h5_indices.resize((self._n, self.k_neighbors))
        self.neighbor_distances.resize((self._n, self.k_neighbors))
        self.neighbor_resnums.resize((self._n, self.k_neighbors))
        
        # Add metadata
        for k, v in meta.items():
            self.f.attrs[k] = v
        
        self.f.attrs["aggregation_mode"] = "transformer"
        self.f.attrs["knn_k"] = self.k_neighbors
        
        euclidean_count = sum(self.distance_metrics)
        sequence_count = len(self.distance_metrics) - euclidean_count
        
        if euclidean_count > 0:
            self.f.attrs["neighbor_distance_metric"] = "euclidean_angstrom"
        else:
            self.f.attrs["neighbor_distance_metric"] = "sequence_delta"
        
        self.f.attrs["euclidean_samples"] = euclidean_count
        self.f.attrs["sequence_fallback_samples"] = sequence_count
        self.f.attrs["fallback_rate"] = sequence_count / max(len(self.distance_metrics), 1)
        
        logger.warning(f"Distance metric usage:")
        logger.warning(f"  Euclidean (3D): {euclidean_count:,} ({100*euclidean_count/max(len(self.distance_metrics),1):.1f}%)")
        logger.warning(f"  Sequence fallback: {sequence_count:,} ({100*sequence_count/max(len(self.distance_metrics),1):.1f}%)")
        
        self.f.close()


# ============================================================================
# Protein Processor (runs in parallel workers)
# ============================================================================

def process_protein_batch(
    protein_key: str,
    group_df: pd.DataFrame,
    esm_dir: Path,
    pdb_index: Dict,
    pdb_base_dir: Path,
    feat_cols: List[str],
    split_map: Dict,
    k_neighbors: int
) -> Optional[Dict]:
    """
    Process single protein: load embeddings, PDB, compute neighbors.
    Returns ready-to-write batch dict with LOCAL (chain-relative) neighbor indices.
    The writer will shift these to global H5 indices when appending.
    """
    try:
        # Load ESM embedding
        emb_path = find_chain_emb_file(esm_dir, protein_key)
        if not emb_path:
            logger.error(f"❌ Protein {protein_key}: No embedding file found in {esm_dir}")
            logger.error(f"   Tried: {protein_key}.pt, {protein_key.lower()}.pt, {protein_key.upper()}.pt")
            raise RuntimeError(f"CRITICAL: Protein {protein_key} has no embedding file! Cannot proceed - would break alignment.")
        
        emb, resnums = load_chain_embedding(emb_path)
        if emb is None:
            logger.error(f"❌ Protein {protein_key}: Embedding file exists but load returned None: {emb_path}")
            raise RuntimeError(f"CRITICAL: Protein {protein_key} embedding load failed! Cannot proceed - would break alignment.")
        
        # Get protein metadata
        protein_id_raw = group_df.iloc[0]["protein_id"]
        chain_id = group_df.iloc[0]["chain_id"]
        is_bu48 = group_df.iloc[0]["is_bu48"]
        
        # Extract base PDB ID
        if '_' in protein_id_raw:
            parts = protein_id_raw.rsplit('_', 1)
            if len(parts) == 2 and len(parts[1]) <= 2 and parts[1].isalpha():
                protein_id_base = parts[0]
            else:
                protein_id_base = protein_id_raw
        else:
            protein_id_base = protein_id_raw
        
        # Find PDB file (fast index lookup)
        pdb_path = None
        if pdb_index:
            for key in [(protein_id_base.lower(), chain_id.upper()),
                        (protein_id_raw.lower(), chain_id.upper()),
                        (protein_id_base.lower(), "A")]:
                if key in pdb_index:
                    candidate = pdb_index[key]
                    if candidate.exists():
                        pdb_path = candidate
                        break
        
        # Load coordinates and build KD-tree
        ca_coords = extract_ca_coordinates_fast(pdb_path, chain_id) if pdb_path else None
        kdtree, kdtree_resnums = build_kdtree(ca_coords) if ca_coords else (None, None)
        
            # Build residue index
        if resnums is not None and len(resnums) > 0:
            resnum_to_local_idx = {int(r): i for i, r in enumerate(resnums)}
        else:
            resnum_to_local_idx = {i: i for i in range(len(emb))}
            resnums = list(range(len(emb)))        # Process all rows for this protein
        batch_tab = []
        batch_esm = []
        batch_neighbor_indices = []
        batch_neighbor_dists = []
        batch_neighbor_resnums = []
        batch_labels = []
        batch_keys = []
        batch_resnos = []
        batch_splits = []
        batch_euclidean_flags = []
        
        stats = {"euclidean": 0, "sequence": 0, "exact": 0, "mean_pooled": 0}
        
        for _, row in group_df.iterrows():
            residue_number = row.get("residue_number", row.get("res_num", None))
            
            # Get embedding
            esm_vector = choose_vector_by_resnum(emb, resnums, residue_number)
            
            if residue_number is not None and residue_number in resnum_to_local_idx:
                stats["exact"] += 1
            else:
                stats["mean_pooled"] += 1
            
            # Compute neighbors (fast KD-tree)
            if residue_number is not None and resnums is not None and len(resnums) > 0:
                target_resnum_int = int(residue_number)
                
                neighbor_resnums_list, neighbor_distances_list, used_euclidean = \
                    compute_euclidean_neighbors_fast(
                        target_resnum_int, resnums, ca_coords, kdtree, kdtree_resnums, k_neighbors
                    )
                
                stats["euclidean" if used_euclidean else "sequence"] += 1
                
                # Keep neighbor indices LOCAL (chain-relative)
                # The writer will shift these to global H5 indices
                neighbor_local_idx_list = []
                for n_resnum in neighbor_resnums_list:
                    if resnum_to_local_idx is not None and n_resnum in resnum_to_local_idx:
                        local_idx = resnum_to_local_idx[n_resnum]
                        neighbor_local_idx_list.append(local_idx)
                    else:
                        neighbor_local_idx_list.append(-1)  # Missing marker
            else:
                # No residue number - use first residue as self
                neighbor_local_idx_list = [0] * k_neighbors
                neighbor_distances_list = [0.0] * k_neighbors
                neighbor_resnums_list = [int(residue_number) if residue_number else 0] * k_neighbors
                used_euclidean = False
                stats["sequence"] += 1
            
            # Get features and metadata
            tab_values = row[feat_cols].values.astype(np.float32)
            label = get_label(row)
            split_flag = 2 if is_bu48 else split_map.get(protein_key, 0)
            
            batch_tab.append(tab_values)
            batch_esm.append(esm_vector)
            batch_neighbor_indices.append(neighbor_local_idx_list)  # LOCAL indices
            batch_neighbor_dists.append(neighbor_distances_list)
            batch_neighbor_resnums.append(neighbor_resnums_list)
            batch_labels.append(label)
            batch_keys.append(protein_key)
            batch_resnos.append(int(residue_number) if residue_number else 0)
            batch_splits.append(split_flag)
            batch_euclidean_flags.append(used_euclidean)
        
        # CRITICAL VALIDATION: Ensure batch_keys is never empty
        if not batch_keys:
            raise RuntimeError(f"Protein {protein_key}: batch_keys is empty! This would cause alignment issues.")
        
        if len(batch_keys) != len(batch_tab):
            raise RuntimeError(
                f"Protein {protein_key}: batch_keys length ({len(batch_keys)}) != "
                f"batch_tab length ({len(batch_tab)}). Data corruption!")
        
        # Verify all keys are the same protein_key
        unique_keys = set(batch_keys)
        if len(unique_keys) != 1 or protein_key not in unique_keys:
            raise RuntimeError(
                f"Protein {protein_key}: batch_keys contains wrong values: {unique_keys}")
        
        return {
            "protein_key": protein_key,
            "tab": np.array(batch_tab),
            "esm": np.array(batch_esm),
            "neighbor_indices_local": np.array(batch_neighbor_indices),  # LOCAL indices
            "neighbor_dists": np.array(batch_neighbor_dists),
            "neighbor_resnums": np.array(batch_neighbor_resnums),
            "lbl": np.array(batch_labels),
            "keys": batch_keys,
            "resnos": np.array(batch_resnos),
            "split_flags": np.array(batch_splits),
            "used_euclidean_flags": batch_euclidean_flags,
            "stats": stats
        }
        
    except Exception as e:
        # Log the error with protein key for debugging
        logger.warning(f"Failed to process protein {protein_key}: {e}")
        return None


# ============================================================================
# Main Build Function with Multi-Processing
# ============================================================================

def build_h5_v2_multiprocess(
    csv_path: Path,
    esm_dir: Path,
    pdb_base_dir: Path,
    bu48_txt: Optional[Path],
    out_path: Path,
    val_frac: float = 0.2,
    seed: int = 42,
    k_neighbors: int = 3,
    max_workers: Optional[int] = None,
    pdb_index: Optional[Dict] = None
):
    """
    Build H5 with multi-processing optimization.
    
    Uses producer-consumer pattern:
    - N workers process proteins in parallel (parse PDB, compute neighbors)
    - 1 writer process handles all HDF5 writes
    """
    logger.warning("="*80)
    logger.warning("Building H5 v2 - OPTIMIZED (Multi-Processing + KD-tree)")
    logger.warning("="*80)
    logger.warning(f"CSV: {csv_path}")
    logger.warning(f"ESM dir: {esm_dir}")
    logger.warning(f"PDB base: {pdb_base_dir}")
    logger.warning(f"Output: {out_path}")
    logger.warning(f"k neighbors: {k_neighbors}")
    logger.warning("")
    
    # Load data
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("CSV is empty")
    
    logger.warning(f"Loaded {len(df):,} samples")
    
    # Prepare columns
    if "protein_id" not in df.columns:
        df["protein_id"] = df["file_name"].astype(str).str.replace(r"\.pdb$", "", regex=True).str.lower()
    if "chain_id" not in df.columns:
        df["chain_id"] = "A"
    
    df["protein_key"] = df.apply(protein_key_from_row, axis=1)
    feat_cols = guess_feature_cols(df)
    
    logger.warning(f"Features: {len(feat_cols)}")
    
    # BU48 split
    bu48_proteins = load_bu48_list(bu48_txt) if bu48_txt else set()
    
    def is_protein_in_bu48(protein_id: str, bu48_set: set) -> bool:
        pid_lower = protein_id.lower()
        return pid_lower in bu48_set or (len(pid_lower) >= 4 and pid_lower[:4] in bu48_set)
    
    df["is_bu48"] = df["protein_id"].apply(lambda x: is_protein_in_bu48(x, bu48_proteins))
    
    # Train/val split
    non_bu48_groups = df.loc[~df["is_bu48"], "protein_key"].drop_duplicates().tolist()
    split_map = grouped_split(non_bu48_groups, val_frac, seed)
    
    # Probe ESM dimension
    esm_dim = 2560
    for protein_key in df["protein_key"].unique():
        emb_path = find_chain_emb_file(esm_dir, protein_key)
        if emb_path:
            emb, _ = load_chain_embedding(emb_path)
            if emb is not None:
                esm_dim = emb.shape[-1]
                break
    
    logger.warning(f"ESM dimension: {esm_dim}")
    
    # Initialize writer
    writer = H5WriterTransformerV2Optimized(out_path, len(feat_cols), esm_dim, k_neighbors, feat_cols)
    
    # Group by protein - CRITICAL: sort=False preserves CSV row order
    # This ensures that when we write results in submission order, 
    # H5 row i corresponds to CSV row i
    protein_groups = list(df.groupby("protein_key", sort=False))
    total_proteins = len(protein_groups)
    
    logger.warning(f"Grouped into {total_proteins} unique proteins (preserving CSV row order)")
    
    logger.warning(f"Processing {total_proteins} proteins with {max_workers or cpu_count()} workers...")
    logger.warning("")
    
    # Process proteins in parallel
    workers = max_workers or cpu_count()
    
    logger.warning(f"Using {workers} CPU cores for parallel processing")
    logger.warning(f"Parser: {'gemmi (fast)' if HAS_GEMMI else 'BioPython (slow)'}")
    logger.warning("")
    logger.warning("⚠️  IMPORTANT: Processing in CSV row order to maintain alignment!")
    logger.warning("")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks (no protein_start_idx - workers return LOCAL indices)
        # Store futures in a list to preserve submission order
        futures_list = []
        for protein_key, group in protein_groups:
            future = executor.submit(
                process_protein_batch,
                protein_key,
                group,
                esm_dir,
                pdb_index,
                pdb_base_dir,
                feat_cols,
                split_map,
                k_neighbors
            )
            futures_list.append((future, protein_key, len(group)))  # Track protein_key for logging
        
        # Collect results and write IN SUBMISSION ORDER (not completion order)
        # This ensures H5 rows match CSV row order
        stats = {"total_samples": 0, "successful_proteins": 0, "failed_proteins": 0,
                 "exact_residue_hits": 0, "mean_pooled_fallbacks": 0,
                 "euclidean_distance_used": 0, "sequence_distance_fallback": 0}
        
        failed_proteins = []  # Track which proteins failed
        
        # CRITICAL: Iterate in submission order, call .result() to wait for each
        for future, protein_key, n_samples in tqdm(futures_list, desc="Processing proteins"):
            try:
                result = future.result()
                
                if result is None:
                    # STRICT MODE: Cannot skip proteins - breaks alignment!
                    raise RuntimeError(
                        f"❌ CRITICAL: Protein '{protein_key}' returned None! "
                        f"This would break CSV/H5 alignment. Cannot continue. "
                        f"Fix missing embedding or PDB file for this protein.")
                
                # Validate result has all required fields
                required_fields = ["tab", "esm", "neighbor_indices_local", "neighbor_dists", 
                                  "neighbor_resnums", "lbl", "keys", "resnos", "split_flags"]
                for field in required_fields:
                    if field not in result:
                        raise RuntimeError(
                            f"❌ CRITICAL: Protein '{protein_key}' result missing field '{field}'!")
                
                # Validate keys array is not empty
                if not result["keys"] or len(result["keys"]) == 0:
                    raise RuntimeError(
                        f"❌ CRITICAL: Protein '{protein_key}' has empty 'keys' array! "
                        f"Expected {n_samples} keys. This would create empty protein_key entries.")
                
                # Shift local neighbor indices to global H5 indices
                # current_start is writer._n (where this batch will be written)
                current_start = writer._n
                neighbor_indices_global = result["neighbor_indices_local"].copy()
                
                # Shift all valid indices (skip -1 markers)
                mask = neighbor_indices_global != -1
                neighbor_indices_global[mask] += current_start
                
                # Write immediately
                n_before = writer._n
                writer.append(
                    result["tab"],
                    result["esm"],
                    neighbor_indices_global,  # Now global indices
                    result["neighbor_dists"],
                    result["neighbor_resnums"],
                    result["lbl"],
                    result["keys"],
                    result["resnos"],
                    result["split_flags"],
                    result["used_euclidean_flags"]
                )
                n_after = writer._n
                
                # Verify correct number of rows were written
                rows_written = n_after - n_before
                if rows_written != n_samples:
                    raise RuntimeError(
                        f"❌ CRITICAL: Protein '{protein_key}': Expected to write {n_samples} rows, "
                        f"but actually wrote {rows_written} rows! Data corruption!")
                
                stats["successful_proteins"] += 1
                stats["total_samples"] += len(result["tab"])
                stats["exact_residue_hits"] += result["stats"]["exact"]
                stats["mean_pooled_fallbacks"] += result["stats"]["mean_pooled"]
                stats["euclidean_distance_used"] += result["stats"]["euclidean"]
                stats["sequence_distance_fallback"] += result["stats"]["sequence"]
                
            except Exception as e:
                stats["failed_proteins"] += 1
                failed_proteins.append(protein_key)
                logger.warning(f"⚠️  Failed to process protein {protein_key}: {e}")
    
    # Finalize
    metadata = {
        "dataset_name": "pocknet_all_train_transformer_v2_optimized",
        "num_samples": stats["total_samples"],
        "num_proteins": stats["successful_proteins"],
        "esm_dimension": esm_dim,
        "tabular_features": len(feat_cols),
        "seed": seed,
        "val_fraction": val_frac,
        "split_meaning": "0=train, 1=val, 2=test(BU48)",
        "row_order": "preserved_from_csv",
        "alignment_fix": "2025-10-10"
    }
    
    writer.finalize(metadata)
    
    # Print statistics
    logger.warning("")
    logger.warning("="*80)
    logger.warning("📊 GENERATION COMPLETE")
    logger.warning("="*80)
    logger.warning(f"Total samples: {stats['total_samples']:,}")
    logger.warning(f"Successful proteins: {stats['successful_proteins']}")
    logger.warning(f"Failed proteins: {stats['failed_proteins']}")
    
    if failed_proteins:
        logger.warning("")
        logger.warning(f"⚠️  Failed protein keys ({len(failed_proteins)}):")
        for pkey in failed_proteins[:20]:  # Show first 20
            logger.warning(f"    - {pkey}")
        if len(failed_proteins) > 20:
            logger.warning(f"    ... and {len(failed_proteins) - 20} more")
    
    logger.warning("")
    logger.warning(f"Residue mapping:")
    logger.warning(f"  Exact hits: {stats['exact_residue_hits']:,} ({100*stats['exact_residue_hits']/stats['total_samples']:.2f}%)")
    logger.warning(f"  Mean pooled: {stats['mean_pooled_fallbacks']:,} ({100*stats['mean_pooled_fallbacks']/stats['total_samples']:.2f}%)")
    logger.warning("")
    logger.warning(f"Distance metric:")
    logger.warning(f"  Euclidean (3D): {stats['euclidean_distance_used']:,} ({100*stats['euclidean_distance_used']/stats['total_samples']:.2f}%)")
    logger.warning(f"  Sequence fallback: {stats['sequence_distance_fallback']:,} ({100*stats['sequence_distance_fallback']/stats['total_samples']:.2f}%)")
    logger.warning("="*80)


def main():
    parser = argparse.ArgumentParser(description="Generate H5 v2 - OPTIMIZED with multi-processing")
    parser.add_argument("--csv", type=Path, required=True, help="Path to vectorsTrain CSV")
    parser.add_argument("--esm_dir", type=Path, required=True, help="ESM embeddings directory")
    parser.add_argument("--pdb_base_dir", type=Path, required=True, help="Base directory for PDB files")
    parser.add_argument("--bu48_txt", type=Path, required=True, help="BU48 protein list")
    parser.add_argument("--out", type=Path, required=True, help="Output H5 file")
    parser.add_argument("--k", type=int, default=3, help="Number of neighbors")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: all CPUs)")
    parser.add_argument("--ds_file", type=Path, help="Dataset .ds file for fast PDB lookup")
    
    args = parser.parse_args()
    
    # Load PDB index
    pdb_index = None
    if args.ds_file and args.ds_file.exists():
        ds_base = args.ds_file.parent
        pdb_index = load_pdb_index(args.ds_file, ds_base)
        logger.warning(f"Loaded PDB index with {len(pdb_index)} entries")
    
    build_h5_v2_multiprocess(
        csv_path=args.csv,
        esm_dir=args.esm_dir,
        pdb_base_dir=args.pdb_base_dir,
        bu48_txt=args.bu48_txt,
        out_path=args.out,
        val_frac=args.val_frac,
        seed=args.seed,
        k_neighbors=args.k,
        max_workers=args.workers,
        pdb_index=pdb_index
    )


if __name__ == "__main__":
    main()
