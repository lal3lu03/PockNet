#!/usr/bin/env python3
"""
Generate comprehensive H5 files with ESM2-3B chain embeddings for PockNet training.

This script:
1. Uses chain-level ESM2-3B embeddings (protein_id_chain_id.pt format)
2. Loads the fixed CSV with proper protein_id:chain_id:residue_number mapping
3. Creates train/val/test splits with BU48 as held-out test set
4. Uses exact residue-level embeddings via residue number lookup
"""

import os
import sys
import gc
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
except Exception:
    torch = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------- Utility Functions ----------------

def load_bu48_list(txt_path: Path) -> set:
    """Load BU48 protein IDs from text file."""
    if not txt_path or not txt_path.exists():
        return set()
    ids = []
    with open(txt_path) as f:
        for line in f:
            protein_id = line.strip()
            if protein_id:
                ids.append(protein_id.lower())
    return set(ids)

def guess_feature_cols(df):
    """Guess which columns contain features vs metadata"""
    exclude = {'file_name', 'pdb_id', 'protein_id', 'chain_id', 'residue_id', 
               'residue_number', 'res_num', 'position', 'residue_name', 
               'atom_name', 'x', 'y', 'z', 'binding_site', 'class', 'protein_key', 'is_bu48'}
    return [c for c in df.columns if c not in exclude]

def get_label(row: pd.Series) -> int:
    """Extract label from row."""
    if "class" in row and pd.notna(row["class"]):
        return int(row["class"])
    if "binding_site" in row and pd.notna(row["binding_site"]):
        return 1 if float(row["binding_site"]) > 0 else 0
    return 0

def protein_key_from_row(row: pd.Series) -> str:
    """Generate protein_chain key matching embedding filenames."""
    pid = str(row["protein_id"]).lower() if "protein_id" in row else Path(str(row["file_name"])).stem.lower()
    cid = str(row["chain_id"]).strip() if "chain_id" in row and pd.notna(row["chain_id"]) else "A"
    return f"{pid}_{cid}"

def find_chain_emb_file(esm_dir: Path, protein_key: str) -> Optional[Path]:
    """Find chain embedding file with case-insensitive search."""
    # Try exact match first
    exact_path = esm_dir / f"{protein_key}.pt"
    if exact_path.exists():
        return exact_path
    
    # Try case variants
    variants = [
        esm_dir / f"{protein_key.lower()}.pt",
        esm_dir / f"{protein_key.upper()}.pt",
    ]
    for p in variants:
        if p.exists():
            return p
    
    # Fallback: glob search
    globs = list(esm_dir.glob(f"{protein_key}*.pt"))
    return globs[0] if globs else None

def load_chain_embedding(path: Path) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    """
    Load chain embedding from .pt file.
    Returns (embedding_matrix, residue_numbers) where:
    - embedding_matrix: (N_residues, 2560) float32 array
    - residue_numbers: list of residue numbers for exact lookup
    """
    if torch is None:
        raise RuntimeError("PyTorch not available for loading .pt embeddings.")
    
    try:
        data = torch.load(path, map_location="cpu")
        
        # Handle dict format from our chain embedding generator
        if isinstance(data, dict):
            emb = data.get("emb", None)
            if emb is not None:
                if hasattr(emb, "detach"):
                    emb = emb.detach()
                emb = emb.float().cpu().numpy().astype(np.float32)
            resnums = data.get("resnums", None)
            return emb, resnums
        
        # Handle raw tensor
        if hasattr(data, "detach"):
            arr = data.detach().float().cpu().numpy().astype(np.float32)
            return arr, None
            
        # Handle numpy array
        if hasattr(data, "dtype"):
            return np.asarray(data, dtype=np.float32), None
            
    except Exception as e:
        logger.warning(f"Failed to load embedding from {path}: {e}")
        return None, None
    
    return None, None

def choose_vector_by_resnum(emb: np.ndarray, resnums: Optional[List[int]], residue_number: Optional[int]) -> np.ndarray:
    """
    Choose the exact residue vector using residue number lookup.
    Falls back to mean pooling if residue not found.
    """
    if emb is None:
        raise ValueError("Empty embedding array.")
    
    if emb.ndim == 1:
        return emb.astype(np.float32)
    
    # Try exact residue lookup
    if residue_number is not None and resnums:
        try:
            idx = resnums.index(int(residue_number))
            return emb[idx].astype(np.float32)
        except (ValueError, IndexError):
            logger.debug(f"Residue {residue_number} not found in chain, using mean")
    
    # Fallback: mean over all residues
    return emb.mean(axis=0).astype(np.float32)

def grouped_split(groups: List[str], val_frac: float, seed: int) -> Dict[str, int]:
    """Split protein groups into train (0) and validation (1) sets."""
    rng = np.random.RandomState(seed)
    arr = np.array(groups)
    rng.shuffle(arr)
    n_val = int(round(len(arr) * val_frac))
    val_set = set(arr[:n_val])
    return {g: (1 if g in val_set else 0) for g in arr}

class H5Writer:
    """Efficient H5 file writer with chunked datasets."""
    
    def __init__(self, out_path: Path, tab_dim: int, esm_dim: int, feat_names: List[str]):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(out_path, "w")
        self._n = 0
        
        # Create resizable datasets
        self.tab = self.f.create_dataset("tabular", shape=(0, tab_dim), maxshape=(None, tab_dim),
                                         chunks=True, compression="gzip")
        self.esm = self.f.create_dataset("esm", shape=(0, esm_dim), maxshape=(None, esm_dim),
                                         chunks=True, compression="gzip")
        self.lbl = self.f.create_dataset("labels", shape=(0,), maxshape=(None,), dtype="i4",
                                         chunks=True, compression="gzip")
        self.pky = self.f.create_dataset("protein_keys", shape=(0,), maxshape=(None,),
                                         dtype=h5py.string_dtype("utf-8"),
                                         chunks=True, compression="gzip")
        self.rno = self.f.create_dataset("residue_numbers", shape=(0,), maxshape=(None,), dtype="i4",
                                         chunks=True, compression="gzip")
        self.spl = self.f.create_dataset("split", shape=(0,), maxshape=(None,), dtype="i1",
                                         chunks=True, compression="gzip")
        
        # Store feature names
        self.f.create_dataset("feature_names",
                              data=np.array([s.encode("utf-8") for s in feat_names]),
                              compression="gzip")
    
    def append(self, tab, esm, lbl, keys, resnos, split_flags):
        """Append batch of data to H5 file."""
        b = tab.shape[0]
        end = self._n + b
        
        # Resize datasets
        self.tab.resize((end, self.tab.shape[1]))
        self.esm.resize((end, self.esm.shape[1]))
        self.lbl.resize((end,))
        self.pky.resize((end,))
        self.rno.resize((end,))
        self.spl.resize((end,))
        
        # Write data
        self.tab[self._n:end] = tab
        self.esm[self._n:end] = esm
        self.lbl[self._n:end] = lbl
        self.pky[self._n:end] = np.array(keys, dtype=object)
        self.rno[self._n:end] = resnos
        self.spl[self._n:end] = split_flags
        
        self._n = end
    
    def finalize(self, meta: Dict):
        """Add metadata and close file."""
        for k, v in meta.items():
            self.f.attrs[k] = v
        self.f.close()


class H5WriterTransformer(H5Writer):
    """
    Extended H5 writer for transformer mode with neighbor tensors.
    
    Stores k neighbor embeddings, distances, and residue numbers per sample
    for attention-based aggregation.
    """
    
    def __init__(self, out_path: Path, tab_dim: int, esm_dim: int, k_neighbors: int, feat_names: List[str]):
        super().__init__(out_path, tab_dim, esm_dim, feat_names)
        self.k_neighbors = k_neighbors
        
        # Create neighbor datasets
        self.esm_neighbors = self.f.create_dataset(
            "esm_neighbors",
            shape=(0, k_neighbors, esm_dim),
            maxshape=(None, k_neighbors, esm_dim),
            chunks=(1000, k_neighbors, esm_dim),
            compression="gzip",
            compression_opts=4
        )
        
        self.neighbor_distances = self.f.create_dataset(
            "neighbor_distances",
            shape=(0, k_neighbors),
            maxshape=(None, k_neighbors),
            chunks=(1000, k_neighbors),
            compression="gzip"
        )
        
        self.neighbor_resnums = self.f.create_dataset(
            "neighbor_resnums",
            shape=(0, k_neighbors),
            maxshape=(None, k_neighbors),
            dtype="i8",
            chunks=(1000, k_neighbors),
            compression="gzip"
        )
    
    def append_with_neighbors(self, tab, esm_fallback, esm_neighbors, neighbor_dists, neighbor_resnums,
                             lbl, keys, resnos, split_flags):
        """Append batch of data with neighbor tensors."""
        b = tab.shape[0]
        start = self._n  # Capture start index BEFORE parent call
        end = start + b
        
        # Resize standard datasets using parent method
        super().append(tab, esm_fallback, lbl, keys, resnos, split_flags)
        
        # Resize neighbor datasets
        self.esm_neighbors.resize((end, self.k_neighbors, self.esm_neighbors.shape[2]))
        self.neighbor_distances.resize((end, self.k_neighbors))
        self.neighbor_resnums.resize((end, self.k_neighbors))
        
        # Write neighbor data using start:end (not self._n which was already advanced)
        self.esm_neighbors[start:end] = esm_neighbors
        self.neighbor_distances[start:end] = neighbor_dists
        self.neighbor_resnums[start:end] = neighbor_resnums

def build_h5(csv_path: Path, esm_dir: Path, bu48_txt: Optional[Path], out_path: Path,
             val_frac: float = 0.2, seed: int = 42, aggregation_mode: str = "mean",
             k_neighbors: int = 3):
    """
    Build H5 file with train/val/test splits using chain-level ESM embeddings.
    
    Args:
        csv_path: Path to vectorsTrain_all_chainfix.csv
        esm_dir: Directory containing chain embedding files (protein_id_chain_id.pt)
        bu48_txt: Path to bu48_proteins.txt (test set)
        out_path: Output H5 file path
        val_frac: Validation fraction for non-BU48 proteins
        seed: Random seed for splits
        aggregation_mode: "mean" (pre-aggregate) or "transformer" (store neighbors with sequence-based k-NN)
        k_neighbors: Number of neighbors to store (for transformer mode)
    """
    logger.info(f"Building H5 from CSV: {csv_path}")
    logger.info(f"ESM embeddings directory: {esm_dir}")
    logger.info(f"BU48 test proteins: {bu48_txt}")
    logger.info(f"Output: {out_path}")
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("CSV is empty.")
    
    logger.info(f"Loaded {len(df):,} samples from CSV")
    
    # Ensure required columns exist
    if "protein_id" not in df.columns:
        df["protein_id"] = df["file_name"].astype(str).str.replace(r"\.pdb$", "", regex=True).str.lower()
    if "chain_id" not in df.columns:
        df["chain_id"] = "A"
    
    # Generate protein keys matching embedding filenames
    df["protein_key"] = df.apply(protein_key_from_row, axis=1)
    
    # Identify tabular feature columns
    feat_cols = guess_feature_cols(df)
    logger.info(f"Found {len(feat_cols)} tabular features: {feat_cols[:10]}...")
    
    # Load BU48 protein list for test split
    bu48_proteins = load_bu48_list(bu48_txt) if bu48_txt else set()
    if bu48_proteins:
        logger.info(f"Loaded {len(bu48_proteins)} BU48 test proteins")
    
    # Mark BU48 proteins for test split (2) - handle potential chain suffixes
    def is_protein_in_bu48(protein_id: str, bu48_set: set) -> bool:
        """Check if protein is in BU48 set, handling potential chain suffixes."""
        pid_lower = protein_id.lower()
        # Direct match first
        if pid_lower in bu48_set:
            return True
        # Try 4-letter PDB code (handle cases like '1udu_a' -> '1udu')
        if len(pid_lower) >= 4:
            base_code = pid_lower[:4]
            if base_code in bu48_set:
                return True
        return False
    
    df["is_bu48"] = df["protein_id"].apply(lambda x: is_protein_in_bu48(x, bu48_proteins))
    
    # Create train/val split for non-BU48 proteins
    non_bu48_groups = df.loc[~df["is_bu48"], "protein_key"].drop_duplicates().tolist()
    split_map = grouped_split(non_bu48_groups, val_frac, seed)
    logger.info(f"Split {len(non_bu48_groups)} non-BU48 protein groups: "
                f"{val_frac:.1%} validation, {1-val_frac:.1%} training")
    
    # Probe one embedding to determine ESM dimension
    esm_dim = None
    for protein_key in df["protein_key"].unique():
        emb_path = find_chain_emb_file(esm_dir, protein_key)
        if emb_path is None:
            continue
        emb, _ = load_chain_embedding(emb_path)
        if emb is not None:
            esm_dim = emb.shape[-1] if emb.ndim >= 1 else None
            logger.info(f"Detected ESM dimension: {esm_dim}")
            break
    
    if esm_dim is None:
        raise RuntimeError("Could not determine ESM dimension; no embeddings found.")
    
    # Initialize appropriate H5 writer based on aggregation mode
    tab_dim = len(feat_cols)
    
    logger.info(f"Aggregation mode: {aggregation_mode}")
    
    if aggregation_mode == "mean":
        writer = H5Writer(out_path, tab_dim, esm_dim, feat_cols)
        use_transformer_mode = False
        logger.info(f"Mean mode: pre-aggregating ESM embeddings (no neighbor storage)")
    elif aggregation_mode == "transformer":
        writer = H5WriterTransformer(out_path, tab_dim, esm_dim, k_neighbors, feat_cols)
        use_transformer_mode = True
        logger.info(f"Transformer mode: storing {k_neighbors} neighbor tensors per sample")
        logger.info(f"Neighbor selection: sequence-based distance (inline computation)")
    else:
        raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")
    
    writer_type = "H5WriterTransformer" if use_transformer_mode else "H5Writer"
    logger.info(f"Using {writer_type} for output")
    
    # Process data in batches
    BATCH_SIZE = 5000
    batch_tab, batch_esm, batch_lbl = [], [], []
    batch_key, batch_res, batch_spl = [], [], []
    
    total_rows = 0
    total_groups = 0
    successful_groups = 0
    exact_hits = 0
    fallback_hits = 0
    
    # Group by protein_key for efficient processing
    protein_groups = df.groupby("protein_key")
    logger.info(f"Processing {len(protein_groups)} protein groups...")
    
    start_time = time.time()
    
    # Create progress bar with better formatting
    pbar = tqdm(protein_groups, 
                desc="Building H5",
                unit="proteins", 
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    for protein_key, group in pbar:
        # Find embedding file
        emb_path = find_chain_emb_file(esm_dir, protein_key)
        if emb_path is None:
            logger.debug(f"No embedding found for {protein_key}")
            continue
        
        # Load embedding
        try:
            emb, resnums = load_chain_embedding(emb_path)
        except Exception as e:
            logger.debug(f"Failed to load embedding for {protein_key}: {e}")
            continue
        
        if emb is None:
            continue
        
        # Create fast residue lookup dict (O(1) instead of O(N) per lookup)
        idx_map = {int(r): i for i, r in enumerate(resnums)} if resnums else {}
        
        # Determine split: BU48 -> test (2), others -> train/val (0/1)
        is_bu48 = group["is_bu48"].iloc[0]
        split_flag = 2 if is_bu48 else split_map.get(protein_key, 0)
        
        # Process each residue in the group
        group_processed = 0
        for _, row in group.iterrows():
            try:
                # Extract tabular features
                tab_features = row[feat_cols].values.astype(np.float32)
                tab_features = np.nan_to_num(tab_features, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Get label
                label = get_label(row)
                
                # Get residue number
                residue_number = None
                if "residue_number" in row and pd.notna(row["residue_number"]):
                    residue_number = int(row["residue_number"])
                
                # Get exact residue embedding (same logic for both modes)
                if residue_number is not None and residue_number in idx_map:
                    target_idx = idx_map[residue_number]
                    esm_vector = emb[target_idx].astype(np.float32)
                    exact_hits += 1
                else:
                    # Fallback to mean pooling if residue not found
                    target_idx = None
                    esm_vector = emb.mean(axis=0).astype(np.float32)
                    fallback_hits += 1
                
                # Get k neighbor embeddings for transformer mode
                if use_transformer_mode:
                    # Initialize neighbor batch lists if needed
                    batch_neighbors = batch_neighbors if 'batch_neighbors' in locals() else []
                    batch_neighbor_dists = batch_neighbor_dists if 'batch_neighbor_dists' in locals() else []
                    batch_neighbor_resnums = batch_neighbor_resnums if 'batch_neighbor_resnums' in locals() else []
                    
                    if target_idx is not None and resnums:
                        # Compute sequence-based distances to all other residues in the protein
                        # (Fast, always available, biologically relevant for local interactions)
                        target_resnum = resnums[target_idx]
                        seq_dists = np.abs(np.array(resnums) - target_resnum).astype(np.float32)
                        
                        # Get k nearest neighbors by sequence distance (excluding self)
                        sorted_indices = np.argsort(seq_dists)
                        neighbor_indices = []
                        neighbor_distances = []
                        neighbor_resnums_list = []
                        
                        for idx in sorted_indices:
                            if len(neighbor_indices) >= k_neighbors:
                                break
                            # Skip self (distance 0)
                            if seq_dists[idx] == 0:
                                continue
                            neighbor_indices.append(idx)
                            neighbor_distances.append(seq_dists[idx])
                            neighbor_resnums_list.append(resnums[idx])
                        
                        # Pad if we have fewer than k neighbors (small proteins)
                        while len(neighbor_indices) < k_neighbors:
                            neighbor_indices.append(target_idx)
                            neighbor_distances.append(0.0)
                            neighbor_resnums_list.append(target_resnum)
                        
                        # Gather neighbor embeddings from the same protein's embedding matrix
                        esm_neighbors_data = emb[neighbor_indices].astype(np.float32)  # [k, D]
                        neighbor_dists_data = np.array(neighbor_distances, dtype=np.float32)  # [k]
                        neighbor_resnums_data = np.array(neighbor_resnums_list, dtype=np.int64)  # [k]
                    else:
                        # Fallback: residue not found, use mean pooling for all neighbors
                        esm_neighbors_data = np.tile(esm_vector, (k_neighbors, 1)).astype(np.float32)
                        neighbor_dists_data = np.full(k_neighbors, 999.0, dtype=np.float32)
                        neighbor_resnums_data = np.full(k_neighbors, -1, dtype=np.int64)
                    
                    batch_neighbors.append(esm_neighbors_data)
                    batch_neighbor_dists.append(neighbor_dists_data)
                    batch_neighbor_resnums.append(neighbor_resnums_data)
                
                # Add to batch
                batch_tab.append(tab_features)
                batch_esm.append(esm_vector)
                batch_lbl.append(int(label))
                batch_key.append(protein_key)
                batch_res.append(residue_number if residue_number is not None else -1)
                batch_spl.append(split_flag)
                
                total_rows += 1
                group_processed += 1
                
                # Write batch if full
                if len(batch_tab) >= BATCH_SIZE:
                    if use_transformer_mode:
                        writer.append_with_neighbors(
                            np.vstack(batch_tab), np.vstack(batch_esm),
                            np.stack(batch_neighbors), np.vstack(batch_neighbor_dists), np.vstack(batch_neighbor_resnums),
                            np.array(batch_lbl, dtype=np.int32), batch_key,
                            np.array(batch_res, dtype=np.int32), np.array(batch_spl, dtype=np.int8)
                        )
                        batch_neighbors.clear()
                        batch_neighbor_dists.clear()
                        batch_neighbor_resnums.clear()
                    else:
                        writer.append(
                            np.vstack(batch_tab), np.vstack(batch_esm), np.array(batch_lbl, dtype=np.int32),
                            batch_key, np.array(batch_res, dtype=np.int32), np.array(batch_spl, dtype=np.int8)
                        )
                    batch_tab.clear()
                    batch_esm.clear()
                    batch_lbl.clear()
                    batch_key.clear()
                    batch_res.clear()
                    batch_spl.clear()
                
            except Exception as e:
                logger.debug(f"Failed to process row in {protein_key}: {e}")
                continue
        
        if group_processed > 0:
            successful_groups += 1
        
        total_groups += 1
        
        # Update progress bar with current stats
        elapsed = time.time() - start_time
        if total_groups > 0:
            rate = total_rows / elapsed
            pbar.set_description(f"Building H5 [{total_rows:,} rows, {rate:.0f} rows/s]")
        
        # Periodic cleanup
        if (total_groups % 100) == 0:
            gc.collect()
    
    # Close progress bar
    pbar.close()
    
    # Write final batch
    if batch_tab:
        if use_transformer_mode:
            writer.append_with_neighbors(
                np.vstack(batch_tab), np.vstack(batch_esm),
                np.stack(batch_neighbors), np.vstack(batch_neighbor_dists), np.vstack(batch_neighbor_resnums),
                np.array(batch_lbl, dtype=np.int32), batch_key,
                np.array(batch_res, dtype=np.int32), np.array(batch_spl, dtype=np.int8)
            )
        else:
            writer.append(
                np.vstack(batch_tab), np.vstack(batch_esm), np.array(batch_lbl, dtype=np.int32),
                batch_key, np.array(batch_res, dtype=np.int32), np.array(batch_spl, dtype=np.int8)
            )
    
    # Prepare metadata
    metadata = {
        "dataset_name": csv_path.stem,
        "num_samples": total_rows,
        "num_proteins": successful_groups,
        "split_meaning": "0=train, 1=val, 2=test_bu48",
        "val_fraction": val_frac,
        "seed": seed,
        "esm_dimension": esm_dim,
        "tabular_features": len(feat_cols),
        "aggregation_mode": aggregation_mode
    }
    
    # Add transformer-specific metadata
    if use_transformer_mode:
        metadata["knn_k"] = k_neighbors
        metadata["neighbor_distance_metric"] = "sequence_distance"
    
    # Finalize H5 file
    writer.finalize(metadata)
    
    # Calculate final statistics
    total_time = time.time() - start_time
    size_gb = out_path.stat().st_size / (1024**3)
    exact_rate = 100 * exact_hits / (exact_hits + fallback_hits + 1e-9)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ H5 FILE GENERATION COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Output file: {out_path}")
    logger.info(f"File size: {size_gb:.2f} GB")
    logger.info(f"Processing time: {total_time/60:.1f} minutes ({total_time:.1f}s)")
    logger.info(f"")
    logger.info(f"📊 DATA STATISTICS:")
    logger.info(f"   Total samples: {total_rows:,}")
    logger.info(f"   Successful proteins: {successful_groups:,}")
    logger.info(f"   ESM dimension: {esm_dim}")
    logger.info(f"   Tabular features: {len(feat_cols)}")
    logger.info(f"")
    logger.info(f"🎯 RESIDUE MAPPING QUALITY:")
    logger.info(f"   Exact residue hits: {exact_hits:,} ({exact_rate:.2f}%)")
    logger.info(f"   Mean pooled fallbacks: {fallback_hits:,} ({100-exact_rate:.2f}%)")
    logger.info(f"")
    logger.info(f"⚡ PERFORMANCE:")
    logger.info(f"   Processing rate: {total_rows/total_time:.0f} samples/second")
    logger.info(f"   Protein rate: {successful_groups/total_time:.1f} proteins/second")
    logger.info(f"{'='*60}")
    
    # Apply PDB4-based split cleanup to avoid data leakage
    logger.info(f"\n🧹 APPLYING PDB4-BASED SPLIT CLEANUP...")
    cleanup_success = _cleanup_pdb4_splits(out_path, val_frac, seed)
    if cleanup_success:
        logger.info(f"✅ PDB4 cleanup completed - no protein leakage between splits")
    else:
        logger.warning(f"⚠️  PDB4 cleanup failed - manual verification recommended")
    
    return True


def _cleanup_pdb4_splits(h5_path: Path, val_frac: float, seed: int) -> bool:
    """
    Rewrite split assignments to ensure no PDB4-level leakage between train/val.
    
    Groups all residues by their 4-character PDB code and assigns entire proteins
    to either train or validation to prevent data leakage.
    
    Args:
        h5_path: Path to H5 file
        val_frac: Validation fraction (e.g., 0.2 for 20%)
        seed: Random seed for reproducibility
        
    Returns:
        True if successful, False otherwise
    """
    import re
    from collections import defaultdict
    
    try:
        # PDB4 regex: strict 4-char PDB code (digit + 3 alphanumeric)
        PDB4 = re.compile(r'(^|[^A-Za-z0-9])([0-9][A-Za-z0-9]{3})(?=[^A-Za-z0-9]|$)')
        
        def pdb4_strict(s: str) -> str:
            """Extract 4-character PDB code from protein key."""
            s = s.replace(":", "_")
            m = PDB4.search(s)
            return m.group(2).lower() if m else None
        
        rng = np.random.RandomState(seed)
        
        with h5py.File(h5_path, "r+") as f:
            # Load data
            keys = np.array([k.decode() if isinstance(k, bytes) else str(k) for k in f["protein_keys"][:]])
            split = f["split"][:]  # 0=train, 1=val, 2=test(BU48)
            
            # Keep BU48 test set (split=2) unchanged
            mask_tv = (split != 2)
            keys_tv = keys[mask_tv]
            idx_tv = np.where(mask_tv)[0]
            
            logger.info(f"   Train/val residues: {len(idx_tv):,}")
            logger.info(f"   Test (BU48) residues: {(split == 2).sum():,}")
            
            # Group indices by PDB4 code
            by_pdb4 = defaultdict(list)
            for i, k in zip(idx_tv, keys_tv):
                p4 = pdb4_strict(k)
                if p4:
                    by_pdb4[p4].append(i)
            
            logger.info(f"   Unique PDB4 codes: {len(by_pdb4)}")
            
            # Shuffle and split PDB4 codes
            p4_list = list(by_pdb4.keys())
            rng.shuffle(p4_list)
            n_val = int(round(len(p4_list) * val_frac))
            val_p4 = set(p4_list[:n_val])
            train_p4 = set(p4_list[n_val:])
            
            logger.info(f"   Train PDB4s: {len(train_p4)}")
            logger.info(f"   Val PDB4s: {len(val_p4)}")
            
            # Rebuild split vector
            new_split = split.copy()
            train_residues = 0
            val_residues = 0
            
            for p4 in by_pdb4:
                new_cls = 1 if p4 in val_p4 else 0
                indices = np.array(by_pdb4[p4], dtype=np.int64)
                new_split[indices] = new_cls
                
                if new_cls == 0:
                    train_residues += len(indices)
                else:
                    val_residues += len(indices)
            
            # Write back to H5
            f["split"][:] = new_split
            
            logger.info(f"   Final train residues: {train_residues:,}")
            logger.info(f"   Final val residues: {val_residues:,}")
            logger.info(f"   Val fraction: {val_residues/(train_residues+val_residues)*100:.1f}%")
            
            # Verify no overlap
            train_pdb4s = {pdb4_strict(k) for k in keys[new_split == 0]}
            val_pdb4s = {pdb4_strict(k) for k in keys[new_split == 1]}
            overlap = train_pdb4s & val_pdb4s
            
            if overlap:
                logger.warning(f"   ⚠️  Found {len(overlap)} PDB4 codes in both train and val!")
                return False
            else:
                logger.info(f"   ✅ Zero PDB4 overlap between train and val")
                return True
                
    except Exception as e:
        logger.error(f"PDB4 cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Build H5 file with chain-level ESM embeddings (dual aggregation support)"
    )
    parser.add_argument("--csv", required=True, help="Path to vectorsTrain_all_chainfix.csv")
    parser.add_argument("--esm_dir", required=True, help="Directory with chain ESM files (protein_id_chain_id.pt)")
    parser.add_argument("--bu48_txt", required=True, help="Path to bu48_proteins.txt")
    parser.add_argument("--out", required=True, help="Output H5 file path")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--aggregation", 
        choices=["mean", "transformer"], 
        default="mean",
        help="Aggregation mode: 'mean' (pre-aggregate) or 'transformer' (store neighbors) (default: mean)"
    )
    parser.add_argument(
        "--k_neighbors", 
        type=int, 
        default=3,
        help="Number of neighbors to store (for transformer mode, selected by sequence distance) (default: 3)"
    )
    args = parser.parse_args()
    
    success = build_h5(
        csv_path=Path(args.csv),
        esm_dir=Path(args.esm_dir),
        bu48_txt=Path(args.bu48_txt) if args.bu48_txt else None,
        out_path=Path(args.out),
        val_frac=args.val_frac,
        seed=args.seed,
        aggregation_mode=args.aggregation,
        k_neighbors=args.k_neighbors
    )
    
    if success:
        logger.info("✅ H5 file generation completed successfully!")
        return 0
    else:
        logger.error("❌ H5 file generation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
