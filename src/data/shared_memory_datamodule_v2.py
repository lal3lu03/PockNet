"""
True Shared Memory DataModule for PyTorch Lightning DDP.
Implements proper inter-process memory sharing using RAM-backed memmap files
where rank 0 loads data once to /dev/shm (tmpfs) and all other processes
access the same memory region via memory-mapped files.
"""

import os
import h5py
import torch
import numpy as np
import json
import time
import fcntl
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable, List, Set
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch import LightningDataModule
from torch.distributed import is_initialized as dist_initialized
import logging
from collections import defaultdict

log = logging.getLogger(__name__)

# RAM-backed shared memory directory (tmpfs) with fallback
import os
fallback_dir = os.environ.get("POCKNET_SHM_DIR", "/dev/shm/pocknet")
SHM_DIR = Path(fallback_dir)
MANIFEST_FILE = SHM_DIR / "manifest.json"
LOCK_FILE = SHM_DIR / "lockfile"

def _bytes_to_str(b) -> str:
    """Handles np.bytes_ and bytes; strips trailing nulls"""
    if isinstance(b, (np.bytes_, bytes, bytearray)):
        return bytes(b).rstrip(b'\x00').decode('utf-8', errors='ignore')
    return str(b)

_PDB4_RE = re.compile(r'([0-9][A-Za-z0-9]{3})')  # e.g., 1a0o

def _norm_id(s: str) -> str:
    """Lowercase, strip whitespace & extensions"""
    s = _bytes_to_str(s).strip()
    s = s.split()[0]
    s = s.split('.')[0]
    s = s.replace(':', '_')
    return s.lower()

def _extract_pdb4(s: str) -> str | None:
    """Extract PDB4 code from normalized string"""
    s = _norm_id(s)
    m = _PDB4_RE.search(s)
    return m.group(1) if m else None

def _load_bu48_ids(path: Path) -> set[str]:
    """Load BU48 IDs from file and normalize them"""
    with open(path) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    return { _extract_pdb4(x) or _norm_id(x) for x in ids }

def _h5_sig(path: Path) -> dict:
    """Get H5 file signature for validation"""
    st = path.stat()
    return {"path": str(path.resolve()), "size": st.st_size, "mtime": int(st.st_mtime)}

def _get_rank(trainer):
    """Get current rank from trainer."""
    return getattr(trainer, "global_rank", 0)

def _get_world_size(trainer):
    """Get world size from trainer."""
    return getattr(trainer, "world_size", 1)

def _ddp_barrier(trainer):
    """Perform distributed barrier if DDP is initialized."""
    if dist_initialized() and hasattr(trainer, 'strategy'):
        try:
            trainer.strategy.barrier()
            log.info(f"[rank: {_get_rank(trainer)}] Passed DDP barrier")
        except Exception as e:
            log.warning(f"[rank: {_get_rank(trainer)}] Barrier failed: {e}")
    else:
        log.info(f"[rank: {_get_rank(trainer)}] No DDP barrier needed")

def _prepare_memmaps_from_h5(h5_path: Path, rank: int):
    """
    Prepare memory-mapped files from H5 file (rank 0 only with file locking).
    This loads data once into RAM-backed tmpfs and creates a manifest.
    """
    if rank != 0:
        return  # Only rank 0 prepares the data
    
    # Create shared memory directory
    SHM_DIR.mkdir(parents=True, exist_ok=True)
    sig = _h5_sig(h5_path)
    
    # Use file locking to ensure only one process writes
    with open(LOCK_FILE, "w") as lock_f:
        try:
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Non-blocking exclusive lock
        except BlockingIOError:
            log.info(f"[rank: {rank}] Another process is preparing data, waiting...")
            fcntl.flock(lock_f, fcntl.LOCK_EX)  # Wait for lock to be released
            
            # 🔧 Re-validate the manifest against the *current* H5
            if MANIFEST_FILE.exists():
                try:
                    with open(MANIFEST_FILE, "r") as f:
                        old = json.load(f)
                    if old.get("source_h5") == sig:
                        log.info(f"[rank: {rank}] Data already prepared for current H5 by another process")
                        return
                    else:
                        log.info(f"[rank: {rank}] Existing manifest is for a different H5 — rebuilding")
                except Exception:
                    log.info(f"[rank: {rank}] Manifest unreadable after wait — rebuilding")
        
        # Check if data already exists and validate against current H5
        if MANIFEST_FILE.exists():
            try:
                with open(MANIFEST_FILE, 'r') as f:
                    old = json.load(f)
                if old.get("source_h5") == sig:
                    log.info(f"[rank: {rank}] Data already prepared for current H5, skipping")
                    return
                else:
                    log.info(f"[rank: {rank}] Manifest points to different H5 — rebuilding")
            except Exception:
                log.info(f"[rank: {rank}] Manifest unreadable — rebuilding")
        
        log.info(f"🚀 [rank: {rank}] Preparing shared memory data from {h5_path}")
        
        with h5py.File(h5_path, 'r') as h5f:
            # Get dataset dimensions
            total_samples = len(h5f['labels'])
            tabular_dim = h5f['tabular'].shape[1]
            esm_dim = h5f['esm'].shape[1]
            
            log.info(f"[rank: {rank}] Dataset dimensions: {total_samples} samples, "
                    f"tabular={tabular_dim}, esm={esm_dim}")
            
            # ⭐ NEW: Detect transformer mode from H5 attributes
            aggregation_mode = h5f.attrs.get('aggregation_mode', 'mean')
            if isinstance(aggregation_mode, bytes):
                aggregation_mode = aggregation_mode.decode('utf-8')
            
            # Check for transformer data in either v1, v2 or v3 format
            has_transformer_v1 = (
                'esm_neighbors' in h5f and 
                'neighbor_distances' in h5f and
                'neighbor_resnums' in h5f
            )
            has_transformer_v2 = (
                'neighbor_h5_indices' in h5f and
                'neighbor_distances' in h5f and
                'neighbor_resnums' in h5f
            )
            has_transformer_v3 = (
                'neighbor_residue_indices' in h5f and
                'neighbor_distances' in h5f and
                'neighbor_resnums' in h5f and
                'residue_embeddings' in h5f
            )
            has_transformer_data = (
                aggregation_mode == 'transformer' and 
                (has_transformer_v1 or has_transformer_v2 or has_transformer_v3)
            )
            
            if has_transformer_data:
                knn_k = int(h5f.attrs.get('knn_k', 3))
                if has_transformer_v3:
                    format_type = "v3 (residue table indices)"
                elif has_transformer_v2:
                    format_type = "v2 (indices)"
                else:
                    format_type = "v1 (pre-computed embeddings)"
                log.info(f"[rank: {rank}] 🔬 Transformer mode detected: k={knn_k} neighbors, format={format_type}")
            else:
                knn_k = None
                log.info(f"[rank: {rank}] Mean aggregation mode")
            
            # Determine max protein ID length for fixed-width storage  
            # Detect which protein ID dataset exists (prefer protein_keys)
            pid_ds = "protein_keys" if "protein_keys" in h5f else "protein_ids"
            protein_ids_raw = h5f[pid_ds][:]
            max_pid_len = max(len(x.decode() if isinstance(x, bytes) else str(x)) 
                            for x in protein_ids_raw)
            
            # Check if split dataset exists
            has_split = "split" in h5f
            if has_split:
                log.info(f"[rank: {rank}] Found H5 split dataset, will use it directly")
            else:
                log.info(f"[rank: {rank}] No H5 split dataset, will rebuild from BU48 list")
            
            def create_memmap(name: str, shape: tuple, dtype: np.dtype):
                """Create a memory-mapped file in shared memory."""
                path = SHM_DIR / f"{name}.npy"
                mm = np.memmap(str(path), mode='w+', dtype=dtype, shape=shape)
                return path, mm
            
            # Create memory-mapped arrays
            tab_path, tab_mm = create_memmap("tabular", (total_samples, tabular_dim), np.float32)
            esm_path, esm_mm = create_memmap("esm", (total_samples, esm_dim), np.float32)
            lab_path, lab_mm = create_memmap("labels", (total_samples,), np.int64)
            res_path, res_mm = create_memmap("residue_numbers", (total_samples,), np.int64)
            pid_path, pid_mm = create_memmap(pid_ds, (total_samples,), f"S{max_pid_len}")
            
            # Optional split (if present in H5)
            if has_split:
                spl_path, spl_mm = create_memmap("split", (total_samples,), np.int8)
            
            residue_emb_mm = residue_key_mm = residue_num_mm = None
            residue_total = 0

            # ⭐ NEW: Create memory-mapped arrays for transformer neighbor tensors
            if has_transformer_data:
                if 'neighbor_residue_indices' in h5f:
                    # V3 format: neighbour indices reference residue table embeddings
                    log.info(f"[rank: {rank}] Detected v3 format (neighbor_residue_indices)")

                    neighbor_idx_path, neighbor_idx_mm = create_memmap(
                        "neighbor_residue_indices", (total_samples, knn_k), np.int32
                    )
                    neighbor_dist_path, neighbor_dist_mm = create_memmap(
                        "neighbor_distances", (total_samples, knn_k), np.float32
                    )
                    neighbor_resnum_path, neighbor_resnum_mm = create_memmap(
                        "neighbor_resnums", (total_samples, knn_k), np.int32
                    )

                    residue_total = h5f['residue_embeddings'].shape[0]
                    residue_emb_path, residue_emb_mm = create_memmap(
                        "residue_embeddings", h5f['residue_embeddings'].shape, np.float32
                    )
                    residue_key_path, residue_key_mm = create_memmap(
                        "residue_protein_keys", (residue_total,), f"S{max_pid_len}"
                    )
                    residue_num_path, residue_num_mm = create_memmap(
                        "residue_numbers_table", (residue_total,), np.int32
                    )

                    log.info(
                        f"[rank: {rank}] Created v3 neighbor memmaps: indices={neighbor_idx_mm.shape}, "
                        f"residue_table={residue_emb_mm.shape}"
                    )

                elif 'neighbor_h5_indices' in h5f:
                    # V2 format: space-efficient indices pointing at sample rows
                    log.info(f"[rank: {rank}] Detected v2 format (neighbor_h5_indices)")

                    neighbor_idx_path, neighbor_idx_mm = create_memmap(
                        "neighbor_h5_indices", (total_samples, knn_k), np.int32)
                    neighbor_dist_path, neighbor_dist_mm = create_memmap(
                        "neighbor_distances", (total_samples, knn_k), np.float32)
                    neighbor_resnum_path, neighbor_resnum_mm = create_memmap(
                        "neighbor_resnums", (total_samples, knn_k), np.int32)

                    log.info(
                        f"[rank: {rank}] Created v2 neighbor memmaps: "
                        f"indices={neighbor_idx_mm.shape}, distances={neighbor_dist_mm.shape}"
                    )

                elif 'esm_neighbors' in h5f:
                    # V1 format: pre-materialized embeddings (legacy)
                    log.info(f"[rank: {rank}] Detected v1 format (esm_neighbors)")
                    
                    neighbor_shape = h5f['esm_neighbors'].shape  # [N, k, D]
                    esm_neigh_path, esm_neigh_mm = create_memmap(
                        "esm_neighbors", neighbor_shape, np.float32)
                    neighbor_dist_path, neighbor_dist_mm = create_memmap(
                        "neighbor_distances", (total_samples, knn_k), np.float32)
                    neighbor_resnum_path, neighbor_resnum_mm = create_memmap(
                        "neighbor_resnums", (total_samples, knn_k), np.int64)
                    
                    log.info(
                        f"[rank: {rank}] Created v1 neighbor memmaps: "
                        f"esm_neighbors={neighbor_shape}, distances={neighbor_dist_mm.shape}"
                    )
                else:
                    log.warning(f"[rank: {rank}] Transformer mode but no neighbor datasets found!")
                    has_transformer_data = False
            
            # Load data in chunks to avoid memory spikes
            chunk_size = 100000
            for i in range(0, total_samples, chunk_size):
                end_idx = min(i + chunk_size, total_samples)
                
                # Load and copy data to memory-mapped arrays
                tab_mm[i:end_idx] = h5f['tabular'][i:end_idx].astype(np.float32, copy=False)
                esm_mm[i:end_idx] = h5f['esm'][i:end_idx].astype(np.float32, copy=False)
                lab_mm[i:end_idx] = h5f['labels'][i:end_idx].astype(np.int64, copy=False)
                res_mm[i:end_idx] = h5f['residue_numbers'][i:end_idx].astype(np.int64, copy=False)
                
                # Handle protein IDs with proper encoding
                raw_pids = h5f[pid_ds][i:end_idx]
                pid_mm[i:end_idx] = np.asarray(
                    [(x.decode() if isinstance(x, bytes) else str(x)) for x in raw_pids],
                    dtype=f"S{max_pid_len}"
                )
                
                # Handle split if present
                if has_split:
                    spl_mm[i:end_idx] = h5f["split"][i:end_idx].astype(np.int8, copy=False)
                
                # ⭐ NEW: Copy transformer neighbor tensors (format-aware)
                if has_transformer_data:
                    if 'neighbor_residue_indices' in h5f:
                        neighbor_idx_mm[i:end_idx] = h5f['neighbor_residue_indices'][i:end_idx].astype(np.int32, copy=False)
                        neighbor_dist_mm[i:end_idx] = h5f['neighbor_distances'][i:end_idx].astype(np.float32, copy=False)
                        neighbor_resnum_mm[i:end_idx] = h5f['neighbor_resnums'][i:end_idx].astype(np.int32, copy=False)
                    elif 'neighbor_h5_indices' in h5f:
                        neighbor_idx_mm[i:end_idx] = h5f['neighbor_h5_indices'][i:end_idx].astype(np.int32, copy=False)
                        neighbor_dist_mm[i:end_idx] = h5f['neighbor_distances'][i:end_idx].astype(np.float32, copy=False)
                        neighbor_resnum_mm[i:end_idx] = h5f['neighbor_resnums'][i:end_idx].astype(np.int32, copy=False)
                    elif 'esm_neighbors' in h5f:
                        esm_neigh_mm[i:end_idx] = h5f['esm_neighbors'][i:end_idx].astype(np.float32, copy=False)
                        neighbor_dist_mm[i:end_idx] = h5f['neighbor_distances'][i:end_idx].astype(np.float32, copy=False)
                        neighbor_resnum_mm[i:end_idx] = h5f['neighbor_resnums'][i:end_idx].astype(np.int64, copy=False)
                
                if (i // chunk_size) % 10 == 0:
                    log.info(f"📥 [rank: {rank}] Loaded {end_idx}/{total_samples} samples "
                            f"({100*end_idx/total_samples:.1f}%)")
            
            # Copy residue lookup table for v3 after sample chunks
            if has_transformer_data and 'neighbor_residue_indices' in h5f and residue_emb_mm is not None:
                residue_chunk = 200000
                residue_embeddings_ds = h5f['residue_embeddings']
                residue_keys_ds = h5f['residue_protein_keys']
                residue_numbers_ds = h5f['residue_numbers_table']
                residue_total = residue_embeddings_ds.shape[0]

                for j in range(0, residue_total, residue_chunk):
                    j_end = min(j + residue_chunk, residue_total)
                    residue_emb_mm[j:j_end] = residue_embeddings_ds[j:j_end].astype(np.float32, copy=False)

                    raw_keys = residue_keys_ds[j:j_end]
                    residue_key_mm[j:j_end] = np.asarray(
                        [(k.decode() if isinstance(k, bytes) else str(k)) for k in raw_keys],
                        dtype=f"S{max_pid_len}"
                    )
                    residue_num_mm[j:j_end] = residue_numbers_ds[j:j_end].astype(np.int32, copy=False)

                residue_emb_mm.flush()
                residue_key_mm.flush()
                residue_num_mm.flush()

            # Ensure all data is written to disk/RAM
            tab_mm.flush()
            esm_mm.flush() 
            lab_mm.flush()
            res_mm.flush()
            pid_mm.flush()
            if has_split:
                spl_mm.flush()
            
            # ⭐ NEW: Flush transformer neighbor memmaps (format-aware)
            if has_transformer_data:
                if 'neighbor_residue_indices' in h5f:
                    neighbor_idx_mm.flush()
                    neighbor_dist_mm.flush()
                    neighbor_resnum_mm.flush()
                elif 'neighbor_h5_indices' in h5f:
                    # V2 format
                    neighbor_idx_mm.flush()
                    neighbor_dist_mm.flush()
                    neighbor_resnum_mm.flush()
                elif 'esm_neighbors' in h5f:
                    # V1 format
                    esm_neigh_mm.flush()
                    neighbor_dist_mm.flush()
                    neighbor_resnum_mm.flush()
            
            # Calculate memory usage
            memory_gb = (
                tab_mm.nbytes + esm_mm.nbytes + lab_mm.nbytes + 
                res_mm.nbytes + pid_mm.nbytes
            ) / (1024**3)
            if has_split:
                memory_gb += spl_mm.nbytes / (1024**3)
            
            # ⭐ NEW: Add transformer neighbor memory usage (format-aware)
            if has_transformer_data:
                if 'neighbor_residue_indices' in h5f and residue_emb_mm is not None:
                    neighbor_memory_gb = (
                        neighbor_idx_mm.nbytes
                        + neighbor_dist_mm.nbytes
                        + neighbor_resnum_mm.nbytes
                        + residue_emb_mm.nbytes
                        + residue_key_mm.nbytes
                        + residue_num_mm.nbytes
                    ) / (1024**3)
                    log.info(f"[rank: {rank}] Neighbor/residue table memory: {neighbor_memory_gb:.2f} GB")
                    memory_gb += neighbor_memory_gb
                elif 'neighbor_h5_indices' in h5f:
                    # V2 format: much smaller (indices instead of embeddings)
                    neighbor_memory_gb = (
                        neighbor_idx_mm.nbytes + neighbor_dist_mm.nbytes + neighbor_resnum_mm.nbytes
                    ) / (1024**3)
                    memory_gb += neighbor_memory_gb
                    log.info(f"[rank: {rank}] Neighbor tensors (v2): {neighbor_memory_gb:.2f} GB")
                elif 'esm_neighbors' in h5f:
                    # V1 format: large (full embeddings)
                    neighbor_memory_gb = (
                        esm_neigh_mm.nbytes + neighbor_dist_mm.nbytes + neighbor_resnum_mm.nbytes
                    ) / (1024**3)
                    memory_gb += neighbor_memory_gb
                    log.info(f"[rank: {rank}] Neighbor tensors (v1): {neighbor_memory_gb:.2f} GB")
            
            log.info(f"✅ [rank: {rank}] Data preparation complete! "
                    f"Shared memory usage: {memory_gb:.2f} GB")
            
            # Create manifest with all necessary metadata
            manifest = {
                "source_h5": sig,
                "total_samples": total_samples,
                "tabular_dim": tabular_dim,
                "esm_dim": esm_dim,
                "max_pid_len": max_pid_len,
                "memory_gb": memory_gb,
                "pid_ds_name": pid_ds,  # Record which dataset name was used
                "has_split": has_split,  # Record if split was present
                "aggregation_mode": aggregation_mode,  # ⭐ NEW: Record aggregation mode
                "paths": {
                    "tabular": str(tab_path),
                    "esm": str(esm_path),
                    "labels": str(lab_path),
                    "residue_numbers": str(res_path),
                    "protein_keys": str(pid_path)
                },
                "shapes": {
                    "tabular": (total_samples, tabular_dim),
                    "esm": (total_samples, esm_dim),
                    "labels": (total_samples,),
                    "residue_numbers": (total_samples,),
                    "protein_keys": (total_samples,)
                },
                "dtypes": {
                    "tabular": "float32",
                    "esm": "float32", 
                    "labels": "int64",
                    "residue_numbers": "int64",
                    "protein_keys": f"S{max_pid_len}"
                }
            }
            
            # ⭐ NEW: Add transformer neighbor tensors to manifest (format-aware)
            if has_transformer_data:
                manifest["knn_k"] = knn_k
                
                if 'neighbor_residue_indices' in h5f:
                    manifest["transformer_format"] = "v3_residue_table"
                    manifest["paths"]["neighbor_residue_indices"] = str(neighbor_idx_path)
                    manifest["paths"]["neighbor_distances"] = str(neighbor_dist_path)
                    manifest["paths"]["neighbor_resnums"] = str(neighbor_resnum_path)
                    manifest["paths"]["residue_embeddings"] = str(residue_emb_path)
                    manifest["paths"]["residue_protein_keys"] = str(residue_key_path)
                    manifest["paths"]["residue_numbers_table"] = str(residue_num_path)
                    manifest["shapes"]["neighbor_residue_indices"] = (total_samples, knn_k)
                    manifest["shapes"]["neighbor_distances"] = (total_samples, knn_k)
                    manifest["shapes"]["neighbor_resnums"] = (total_samples, knn_k)
                    manifest["shapes"]["residue_embeddings"] = h5f['residue_embeddings'].shape
                    manifest["shapes"]["residue_protein_keys"] = (residue_total,)
                    manifest["shapes"]["residue_numbers_table"] = (residue_total,)
                    manifest["dtypes"]["neighbor_residue_indices"] = "int32"
                    manifest["dtypes"]["neighbor_distances"] = "float32"
                    manifest["dtypes"]["neighbor_resnums"] = "int32"
                    manifest["dtypes"]["residue_embeddings"] = "float32"
                    manifest["dtypes"]["residue_protein_keys"] = f"S{max_pid_len}"
                    manifest["dtypes"]["residue_numbers_table"] = "int32"

                elif 'neighbor_h5_indices' in h5f:
                    # V2 format
                    manifest["transformer_format"] = "v2_index_based"
                    manifest["paths"]["neighbor_h5_indices"] = str(neighbor_idx_path)
                    manifest["paths"]["neighbor_distances"] = str(neighbor_dist_path)
                    manifest["paths"]["neighbor_resnums"] = str(neighbor_resnum_path)
                    manifest["shapes"]["neighbor_h5_indices"] = (total_samples, knn_k)
                    manifest["shapes"]["neighbor_distances"] = (total_samples, knn_k)
                    manifest["shapes"]["neighbor_resnums"] = (total_samples, knn_k)
                    manifest["dtypes"]["neighbor_h5_indices"] = "int32"
                    manifest["dtypes"]["neighbor_distances"] = "float32"
                    manifest["dtypes"]["neighbor_resnums"] = "int32"
                    
                elif 'esm_neighbors' in h5f:
                    # V1 format
                    manifest["transformer_format"] = "v1_embedding_based"
                    manifest["paths"]["esm_neighbors"] = str(esm_neigh_path)
                    manifest["paths"]["neighbor_distances"] = str(neighbor_dist_path)
                    manifest["paths"]["neighbor_resnums"] = str(neighbor_resnum_path)
                    manifest["shapes"]["esm_neighbors"] = tuple(neighbor_shape)
                    manifest["shapes"]["neighbor_distances"] = (total_samples, knn_k)
                    manifest["shapes"]["neighbor_resnums"] = (total_samples, knn_k)
                    manifest["dtypes"]["esm_neighbors"] = "float32"
                    manifest["dtypes"]["neighbor_distances"] = "float32"
                    manifest["dtypes"]["neighbor_resnums"] = "int64"
            
            # Add split info if present
            if has_split:
                manifest["paths"]["split"] = str(spl_path)
                manifest["shapes"]["split"] = (total_samples,)
                manifest["dtypes"]["split"] = "int8"
            
            # Write manifest
            with open(MANIFEST_FILE, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            log.info(f"📋 [rank: {rank}] Manifest written to {MANIFEST_FILE}")

def _attach_memmaps(rank: int) -> Dict[str, torch.Tensor]:
    """
    Attach to existing memory-mapped files and return as torch tensors.
    All ranks can call this after rank 0 has prepared the data.
    """
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")
    
    log.info(f"📎 [rank: {rank}] Attaching to shared memory...")
    
    with open(MANIFEST_FILE, 'r') as f:
        manifest = json.load(f)
    
    paths, shapes, dtypes = manifest["paths"], manifest["shapes"], manifest["dtypes"]

    def attach_np(name: str):
        return np.memmap(paths[name], mode='r', dtype=np.dtype(dtypes[name]), shape=tuple(shapes[name]))

    shared_tensors = {}
    memmaps = {}

    # numeric arrays (dtypes already match what we wrote)
    tab_mm = attach_np("tabular");   memmaps["tabular"] = tab_mm
    esm_mm = attach_np("esm");       memmaps["esm"] = esm_mm
    lab_mm = attach_np("labels");    memmaps["labels"] = lab_mm
    res_mm = attach_np("residue_numbers"); memmaps["residue_numbers"] = res_mm

    # Create tensors from memory-mapped arrays (suppress the non-writable warning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*NumPy array is not writable.*")
        shared_tensors["tabular"] = torch.from_numpy(tab_mm)         # float32
        shared_tensors["esm"] = torch.from_numpy(esm_mm)             # float32
        shared_tensors["labels"] = torch.from_numpy(lab_mm)          # int64
        shared_tensors["residue_numbers"] = torch.from_numpy(res_mm) # int64

    # strings: keep as np.memmap (fixed-width bytes)
    shared_tensors["protein_keys"] = attach_np("protein_keys")
    
    # Handle optional split dataset
    if manifest.get("has_split", False):
        spl_mm = attach_np("split")
        memmaps["split"] = spl_mm
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*NumPy array is not writable.*")
            shared_tensors["split"] = torch.from_numpy(spl_mm)  # int8

    # ⭐ NEW: Handle transformer aggregation mode neighbor datasets (format-aware)
    aggregation_mode = manifest.get("aggregation_mode", "mean")
    shared_tensors["aggregation_mode"] = aggregation_mode
    
    if aggregation_mode == "transformer":
        transformer_format = manifest.get("transformer_format", "v1_embedding_based")
        k_neighbors = manifest.get("knn_k", 3)
        
        if transformer_format == "v3_residue_table" and "neighbor_residue_indices" in paths:
            log.info(f"[rank: {rank}] Loading v3 transformer format (residue table)")

            neighbor_idx_mm = attach_np("neighbor_residue_indices")
            neighbor_dist_mm = attach_np("neighbor_distances")
            neighbor_resnum_mm = attach_np("neighbor_resnums")
            residue_emb_mm = attach_np("residue_embeddings")
            residue_key_mm = attach_np("residue_protein_keys")
            residue_num_mm = attach_np("residue_numbers_table")

            memmaps["neighbor_residue_indices"] = neighbor_idx_mm
            memmaps["neighbor_distances"] = neighbor_dist_mm
            memmaps["neighbor_resnums"] = neighbor_resnum_mm
            memmaps["residue_embeddings"] = residue_emb_mm
            memmaps["residue_protein_keys"] = residue_key_mm
            memmaps["residue_numbers_table"] = residue_num_mm

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*NumPy array is not writable.*")
                shared_tensors["neighbor_residue_indices"] = torch.from_numpy(neighbor_idx_mm)
                shared_tensors["neighbor_distances"] = torch.from_numpy(neighbor_dist_mm)
                shared_tensors["neighbor_resnums"] = torch.from_numpy(neighbor_resnum_mm)
                shared_tensors["residue_embeddings"] = torch.from_numpy(residue_emb_mm)
                shared_tensors["residue_numbers_table"] = torch.from_numpy(residue_num_mm)

            # Keep protein keys as numpy bytes (no torch tensor required)
            shared_tensors["residue_protein_keys"] = residue_key_mm

            log.info(f"[rank: {rank}] Transformer v3: neighbor shape {neighbor_idx_mm.shape}, residue table {residue_emb_mm.shape}")

        elif transformer_format == "v2_index_based" and "neighbor_h5_indices" in paths:
            # V2 format: space-efficient indices
            log.info(f"[rank: {rank}] Loading v2 transformer format (index-based)")
            
            neighbor_idx_mm = attach_np("neighbor_h5_indices")
            neighbor_dist_mm = attach_np("neighbor_distances")
            neighbor_resnum_mm = attach_np("neighbor_resnums")
            
            memmaps["neighbor_h5_indices"] = neighbor_idx_mm
            memmaps["neighbor_distances"] = neighbor_dist_mm
            memmaps["neighbor_resnums"] = neighbor_resnum_mm
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*NumPy array is not writable.*")
                shared_tensors["neighbor_h5_indices"] = torch.from_numpy(neighbor_idx_mm)
                shared_tensors["neighbor_distances"] = torch.from_numpy(neighbor_dist_mm)
                shared_tensors["neighbor_resnums"] = torch.from_numpy(neighbor_resnum_mm)
            
            log.info(f"[rank: {rank}] Transformer v2: loaded {k_neighbors} neighbor indices")
            log.info(f"[rank: {rank}] Indices shape: {shared_tensors['neighbor_h5_indices'].shape}, "
                    f"Distances: {shared_tensors['neighbor_distances'].shape}")
            
        elif "esm_neighbors" in paths:
            # V1 format: pre-materialized embeddings (legacy)
            log.info(f"[rank: {rank}] Loading v1 transformer format (embedding-based)")
            
            esm_neigh_mm = attach_np("esm_neighbors")
            neighbor_dist_mm = attach_np("neighbor_distances")
            neighbor_resnum_mm = attach_np("neighbor_resnums")
            
            memmaps["esm_neighbors"] = esm_neigh_mm
            memmaps["neighbor_distances"] = neighbor_dist_mm
            memmaps["neighbor_resnums"] = neighbor_resnum_mm
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*NumPy array is not writable.*")
                shared_tensors["esm_neighbors"] = torch.from_numpy(esm_neigh_mm)
                shared_tensors["neighbor_distances"] = torch.from_numpy(neighbor_dist_mm)
                shared_tensors["neighbor_resnums"] = torch.from_numpy(neighbor_resnum_mm)
            
            log.info(f"[rank: {rank}] Transformer v1: loaded {k_neighbors} neighbor embeddings")
            log.info(f"[rank: {rank}] Neighbor shapes - ESM: {shared_tensors['esm_neighbors'].shape}, "
                    f"Distances: {shared_tensors['neighbor_distances'].shape}")
        else:
            log.warning(f"[rank: {rank}] Transformer mode but no neighbor data found in manifest!")

    shared_tensors["_memmaps"] = memmaps  # keep refs
    log.info(f"✅ [rank: {rank}] Attached. Total: {manifest['memory_gb']:.2f} GB (mode: {aggregation_mode})")
    log.info(f"[rank: {rank}] Tensor shapes - Tabular: {shared_tensors['tabular'].shape}, "
            f"ESM: {shared_tensors['esm'].shape}")
    if manifest.get("has_split", False):
        log.info(f"[rank: {rank}] Split tensor shape: {shared_tensors['split'].shape}")
    
    return shared_tensors

class SharedMemoryDataset(Dataset):
    """Dataset that accesses shared memory tensors directly via memory-mapped files."""
    
    def __init__(
        self,
        indices: List[int],
        shared_tensors: dict,
        transform=None,
        enable_knn: bool = False,
        k_max: int = 10,
        pdb_base_dir: str = None,
        k_neighbors: int = 1,
        neighbor_weighting: str = "softmax",
        neighbor_temp: float = 2.0
    ):
        self.indices = indices
        self.shared_tensors = shared_tensors
        self.transform = transform
        self.enable_knn = enable_knn
        self.k_max = k_max
        self.pdb_base_dir = pdb_base_dir
        self.k_neighbors = k_neighbors
        self.neighbor_weighting = neighbor_weighting
        self.neighbor_temp = neighbor_temp
        self.indices = indices
        self.shared_tensors = shared_tensors
        self.transform = transform
        self.enable_knn = enable_knn
        self.k_max = k_max
        self.pdb_base_dir = pdb_base_dir
        
        # Initialize k-NN functionality if enabled
        if self.enable_knn:
            try:
                from src.data.practical_knn_utils import PracticalKNNCache
                # We'll initialize this in setup() when we have access to h5_path
                self.knn_cache = None
                self.knn_helper = None
                log.info(f"Practical k-NN functionality will be enabled with k_max={k_max}")
            except ImportError as e:
                log.warning(f"Practical k-NN functionality not available: {e}")
                # Fallback to original implementation
                try:
                    from src.data.knn_enhanced_dataset import KNNCapableDataset
                    self.knn_helper = KNNCapableDataset(enable_knn=True, k_max=k_max, pdb_base_dir=pdb_base_dir)
                    self.knn_cache = None
                    log.info(f"Fallback k-NN functionality enabled with k_max={k_max}")
                except ImportError:
                    log.warning("No k-NN functionality available")
                    self.enable_knn = False
                    self.knn_cache = None
                    self.knn_helper = None
        else:
            self.knn_cache = None
            self.knn_helper = None
        
        log.info(f"SharedMemoryDataset initialized with {len(indices)} samples")
        log.info(f"Shared tensors - Tabular: {shared_tensors['tabular'].shape}, "
                f"ESM: {shared_tensors['esm'].shape}")
        if self.enable_knn:
            log.info(f"k-NN enabled with k_max={self.k_max}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map local index to global dataset index
        global_idx = self.indices[idx]
        
        # Decode protein ID using helper function
        pid = _bytes_to_str(self.shared_tensors['protein_keys'][global_idx])
        residue_number = int(self.shared_tensors['residue_numbers'][global_idx])
        
        # Create base sample
        sample = {
            'tabular': torch.from_numpy(np.array(self.shared_tensors['tabular'][global_idx])).float(),
            'label': self.shared_tensors['labels'][global_idx].clone().float(),
            'protein_id': pid,
            'residue_number': residue_number,
            'h5_index': global_idx
        }
        
        # ⭐ NEW: Detect aggregation mode from shared tensors
        aggregation_mode = self.shared_tensors.get('aggregation_mode', 'mean')
        sample['aggregation_mode'] = aggregation_mode
        
        # Handle ESM embeddings based on aggregation mode
        if aggregation_mode == 'transformer':
            # Always include single ESM embedding
            sample['esm'] = torch.from_numpy(np.array(self.shared_tensors['esm'][global_idx])).float()
            
            # ⭐ V3: Residue-table indices (new format)
            if 'neighbor_residue_indices' in self.shared_tensors:
                neighbor_indices = self.shared_tensors['neighbor_residue_indices'][global_idx]
                neighbor_distances = self.shared_tensors['neighbor_distances'][global_idx]
                neighbor_resnums = self.shared_tensors['neighbor_resnums'][global_idx]

                residue_table = self.shared_tensors['residue_embeddings']
                centre_embedding = np.array(self.shared_tensors['esm'][global_idx])

                k = len(neighbor_indices)
                esm_neighbors = np.zeros((k, residue_table.shape[1]), dtype=np.float32)
                valid_mask = neighbor_indices >= 0
                if valid_mask.any():
                    esm_neighbors[valid_mask] = residue_table[neighbor_indices[valid_mask]]
                if not valid_mask.all():
                    esm_neighbors[~valid_mask] = centre_embedding

                distances = np.array(neighbor_distances, dtype=np.float32)
                distances[~valid_mask] = np.inf

                sample['esm_neighbors'] = torch.from_numpy(esm_neighbors).float()
                sample['neighbor_residue_indices'] = torch.from_numpy(np.array(neighbor_indices)).long()
                sample['neighbor_distances'] = torch.from_numpy(distances).float()
                sample['neighbor_resnums'] = torch.from_numpy(np.array(neighbor_resnums)).long()
                sample['knn_aggregated'] = False

            # ⭐ V2: Reconstruct neighbor embeddings from indices
            elif 'neighbor_h5_indices' in self.shared_tensors:
                # Space-efficient v2 format: indices → embeddings
                neighbor_indices = self.shared_tensors['neighbor_h5_indices'][global_idx]  # [k]
                neighbor_distances = self.shared_tensors['neighbor_distances'][global_idx]  # [k]
                neighbor_resnums = self.shared_tensors['neighbor_resnums'][global_idx]  # [k]
                
                # Get full ESM tensor (view, no copy)
                full_esm = self.shared_tensors['esm']
                
                # Reconstruct neighbor embeddings via index_select
                valid_mask = neighbor_indices >= 0  # -1 indicates missing/padding
                
                if valid_mask.any():
                    # Use valid indices only
                    valid_indices = neighbor_indices[valid_mask]
                    
                    # Build neighbor tensor [k, 2560]
                    k = len(neighbor_indices)
                    esm_dim = full_esm.shape[1]
                    esm_neighbors = np.zeros((k, esm_dim), dtype=np.float32)
                    
                    # Fill in valid neighbors
                    esm_neighbors[valid_mask] = full_esm[valid_indices]
                    
                    # For invalid indices, use self embedding as fallback
                    if not valid_mask.all():
                        self_emb = full_esm[global_idx]
                        esm_neighbors[~valid_mask] = self_emb
                    
                    sample['esm_neighbors'] = torch.from_numpy(esm_neighbors).float()
                    
                    # Set invalid distances to inf for proper attention masking
                    distances = np.array(neighbor_distances, dtype=np.float32)
                    distances[~valid_mask] = np.inf
                    sample['neighbor_distances'] = torch.from_numpy(distances).float()
                    sample['neighbor_resnums'] = torch.from_numpy(np.array(neighbor_resnums)).long()
                else:
                    # All invalid - pad with self, but mark distances as inf for masking
                    k = len(neighbor_indices)
                    self_emb = full_esm[global_idx]
                    esm_neighbors = np.tile(self_emb, (k, 1))
                    
                    sample['esm_neighbors'] = torch.from_numpy(esm_neighbors).float()
                    # Set to inf so attention masking treats them as invalid
                    sample['neighbor_distances'] = torch.full((k,), float('inf'), dtype=torch.float32)
                    sample['neighbor_resnums'] = torch.full((k,), residue_number, dtype=torch.long)
                
                sample['knn_aggregated'] = False  # Will be aggregated by model
                
            # ⭐ V1: Pre-materialized neighbor embeddings (legacy)
            elif 'esm_neighbors' in self.shared_tensors:
                sample['esm_neighbors'] = torch.from_numpy(np.array(self.shared_tensors['esm_neighbors'][global_idx])).float()
                sample['neighbor_distances'] = torch.from_numpy(np.array(self.shared_tensors['neighbor_distances'][global_idx])).float()
                sample['neighbor_resnums'] = torch.from_numpy(np.array(self.shared_tensors['neighbor_resnums'][global_idx])).long()
                sample['knn_aggregated'] = False
            
            else:
                # Fallback: no neighbors available, just use single embedding
                log.warning(f"Transformer mode but no neighbor data found for sample {global_idx}")
                sample['knn_aggregated'] = True  # Indicate aggregation not possible
            
        elif aggregation_mode == 'mean' or not ('esm_neighbors' in self.shared_tensors):
            # Mean mode: handle k-NN aggregation (existing implementation)
            if self.enable_knn and self.k_neighbors > 1 and hasattr(self, 'knn_cache') and self.knn_cache is not None:
                try:
                    # Get k-NN data for this SAS point
                    knn_data = self.knn_cache.get_knn_embeddings(
                        h5_idx=global_idx,
                        protein_key=pid, 
                        target_residue=residue_number,
                        k=self.k_max
                    )
                    
                    if knn_data is not None:
                        # Extract neighbor data
                        knn_h5_indices = torch.from_numpy(knn_data['knn_h5_indices']).long()
                        knn_distances = torch.from_numpy(knn_data['knn_distances']).float()
                        
                        # Get full ESM tensor from shared memory (no copy - view only)
                        full_esm = torch.from_numpy(np.array(self.shared_tensors['esm'])).float()
                        
                        # Aggregate k-NN embeddings
                        from src.data.practical_knn_utils import aggregate_knn_embeddings_h5
                        
                        # Use actual k (may be less than k_max)
                        k = min(self.k_neighbors, len(knn_h5_indices))
                        
                        # Aggregate embeddings (adds batch dimension internally)
                        aggregated = aggregate_knn_embeddings_h5(
                            h5_embeddings=full_esm,
                            knn_h5_indices=knn_h5_indices[:k].unsqueeze(0),  # (1, k)
                            knn_distances=knn_distances[:k].unsqueeze(0),     # (1, k)
                            k=k,
                            weighting=self.neighbor_weighting,
                            temperature=self.neighbor_temp
                        )
                        
                        # Store aggregated embedding (remove batch dimension)
                        sample['esm'] = aggregated.squeeze(0)  # (2560,)
                        
                        # Store k-NN metadata for debugging/validation
                        sample['knn_h5_indices'] = knn_h5_indices[:k]
                        sample['knn_distances'] = knn_distances[:k]
                        sample['knn_nearest_residues'] = torch.from_numpy(knn_data['nearest_residues'][:k])
                        sample['knn_aggregated'] = True
                    else:
                        # Fallback to original embedding if k-NN data not available
                        sample['esm'] = torch.from_numpy(np.array(self.shared_tensors['esm'][global_idx])).float()
                        sample['knn_aggregated'] = False
                        
                except Exception as e:
                    log.debug(f"k-NN aggregation failed for idx {idx}, protein {pid}: {e}")
                    # Fallback to original embedding
                    sample['esm'] = torch.from_numpy(np.array(self.shared_tensors['esm'][global_idx])).float()
                    sample['knn_aggregated'] = False
            else:
                # No k-NN aggregation: use original embedding
                sample['esm'] = torch.from_numpy(np.array(self.shared_tensors['esm'][global_idx])).float()
                sample['knn_aggregated'] = False
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class TrueSharedMemoryDataModule(LightningDataModule):
    """
    True shared memory data module where rank 0 loads data into shared memory
    and all other ranks attach to the same memory region.
    
    This avoids data duplication across processes and enables efficient DDP training.
    
    Data Splitting Strategy:
    - Prefers H5 `split` vector (0=train, 1=val, 2=test) if present in H5 file
    - Falls back to protein-based BU48 splits: BU48 proteins for test, others for train/val
    
    k-NN Support:
    - Optionally enables k-NN ESM aggregation for enhanced performance
    """
    
    def __init__(
        self,
        h5_filename: str,
        data_dir: str = "",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_split: float = 0.1,
        test_split: float = 0.1,
        val_split_seed: int = 42,
        transform: Optional[Callable] = None,
        bu48_ids_file: str = "data/output_train/test_vectorsTrain_all_names_bu48.txt",
        normalize_ids: bool = True,
        # k-NN parameters
        enable_knn: bool = False,
        k_max: int = 3,
        pdb_base_dir: str = None,
        hard_positive_indices_path: Optional[str] = None,
        hard_positive_repeat: int = 1,
        train_indices_override_path: Optional[str] = None,
        train_override_shuffle_seed: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        data_root_env = os.environ.get("POCKNET_DATA_ROOT")
        if os.path.isabs(h5_filename):
            self.h5_file = Path(h5_filename)
            base_data_dir = self.h5_file.parent
        else:
            if data_dir:
                base_data_dir = Path(data_dir)
            elif data_root_env:
                base_data_dir = Path(data_root_env)
            else:
                base_data_dir = project_root
            self.h5_file = base_data_dir / h5_filename
        self._base_data_dir = base_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.val_split_seed = val_split_seed
        self.test_split = test_split
        self.transform = transform
        bu48_path = Path(bu48_ids_file)
        if not bu48_path.is_absolute():
            candidate_roots = [self._base_data_dir, project_root]
            resolved = None
            for root in candidate_roots:
                candidate = root / bu48_ids_file
                if candidate.exists():
                    resolved = candidate
                    break
            bu48_path = resolved if resolved is not None else project_root / bu48_path
        self.bu48_ids_file = bu48_path
        self.normalize_ids = normalize_ids
        
        # k-NN parameters
        self.enable_knn = enable_knn
        self.k_max = k_max
        self.pdb_base_dir = pdb_base_dir

        # Hard-example replay parameters
        if hard_positive_indices_path:
            hard_path = Path(hard_positive_indices_path)
            if not hard_path.is_absolute():
                candidate_roots = [self._base_data_dir, project_root]
                resolved = None
                for root in candidate_roots:
                    candidate = root / hard_positive_indices_path
                    if candidate.exists():
                        resolved = candidate
                        break
                hard_path = resolved if resolved is not None else project_root / hard_path
            self.hard_positive_indices_path = hard_path
        else:
            self.hard_positive_indices_path = None
        self.hard_positive_repeat = int(max(1, hard_positive_repeat))
        self._hard_positive_selected = 0

        if train_indices_override_path:
            override_path = Path(train_indices_override_path)
            if not override_path.is_absolute():
                candidate_roots = [self._base_data_dir, project_root]
                resolved = None
                for root in candidate_roots:
                    candidate = root / train_indices_override_path
                    if candidate.exists():
                        resolved = candidate
                        break
                override_path = resolved if resolved is not None else project_root / override_path
            self.train_indices_override_path = override_path
        else:
            self.train_indices_override_path = None
        self.train_override_shuffle_seed = (
            int(train_override_shuffle_seed)
            if train_override_shuffle_seed is not None
            else None
        )
        
        # Will be populated during setup
        self.shared_tensors = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.knn_cache = None  # Initialize k-NN cache attribute
        
        log.info(
            "TrueSharedMemoryDataModule initialized with file: %s (base_data_dir=%s)",
            self.h5_file,
            self._base_data_dir,
        )
        if self.enable_knn:
            log.info(f"k-NN enabled with k_max={self.k_max}, pdb_base_dir={self.pdb_base_dir}")
    
    def prepare_data(self):
        """Verify H5 file and BU48 IDs file exist."""
        if not self.h5_file.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_file}")

        needs_bu48 = True
        try:
            with h5py.File(self.h5_file, "r") as h5f:
                needs_bu48 = "split" not in h5f
        except Exception as e:
            log.warning(f"Could not open H5 to check 'split' ({e}); falling back to BU48 list.")
            needs_bu48 = True

        if needs_bu48:
            if not self.bu48_ids_file.exists():
                raise FileNotFoundError(f"BU48 ids file not found: {self.bu48_ids_file}")
            log.info(f"H5 verified: {self.h5_file} (no 'split' found → will use BU48 splits).")
        else:
            log.info(f"H5 verified: {self.h5_file} (found 'split' vector).")
    
    def setup(self, stage: Optional[str] = None):
        """Load data into shared memory and set up datasets."""
        
        # Get current process rank
        rank = _get_rank(self.trainer)
        
        # Rank 0 prepares memory-mapped files if they don't exist
        _prepare_memmaps_from_h5(self.h5_file, rank)
        
        # All ranks wait for data preparation to complete
        _ddp_barrier(self.trainer)
        
        # All ranks attach to the shared memory-mapped files
        self.shared_tensors = _attach_memmaps(rank)
        
        # Create data splits (same across all ranks due to same seed)
        self._create_data_splits()
        
        # Initialize k-NN cache if enabled (after shared tensors are available)
        if self.enable_knn and self.knn_cache is None:
            try:
                from src.data.practical_knn_utils import PracticalKNNCache
                self.knn_cache = PracticalKNNCache(
                    h5_path=str(self.h5_file),
                    k_max=self.k_max,
                    pdb_base_dir=self.pdb_base_dir
                )
                log.info(f"Practical k-NN cache initialized for {str(self.h5_file)}")
            except Exception as e:
                log.warning(f"Failed to initialize practical k-NN cache: {e}")
                self.enable_knn = False
        
        log.info(f"[rank: {rank}] Setup complete - "
                f"Train: {len(self.train_indices)}, Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")
    
    def _create_data_splits(self):
        """Create protein-based train/val/test splits.
        
        Prefers H5 split vector if available, otherwise uses BU48-based splits.
        Test set: BU48 proteins only  
        Train/Val: Other proteins split according to val_split ratio
        """
        
        # Check if H5 file has split vector and use it directly
        if "split" in self.shared_tensors:
            log.info("Using H5 split vector for data splits")
            split_vector = self.shared_tensors["split"].numpy()
            
            # Split vector encoding: 0=train, 1=val, 2=test
            self.train_indices = np.where(split_vector == 0)[0].astype(np.int64)
            self.val_indices = np.where(split_vector == 1)[0].astype(np.int64)
            self.test_indices = np.where(split_vector == 2)[0].astype(np.int64)
            
            log.info(
                f"H5 splits — Train: {len(self.train_indices)}, "
                f"Val: {len(self.val_indices)}, Test: {len(self.test_indices)}"
            )
            self._apply_train_indices_override()
            self._apply_hard_positive_replay()
            return
        
        # Fallback to BU48-based protein splits
        log.info("Using BU48-based protein splits (H5 split vector not found)")
        
        # 1) read and normalize ids
        raw = self.shared_tensors['protein_keys']
        pids = np.array([_norm_id(x) if self.normalize_ids else _bytes_to_str(x) for x in raw])

        # 2) group indices by protein (by normalized whole id)
        by_prot = defaultdict(list)
        for idx, pid in enumerate(pids):
            by_prot[pid].append(idx)

        # 3) BU48 test set by PDB4 code
        bu48_codes = _load_bu48_ids(self.bu48_ids_file)
        def is_bu48(pid: str) -> bool:
            pdb4 = _extract_pdb4(pid)
            return pdb4 in bu48_codes if pdb4 else False

        test_proteins = [pid for pid in by_prot if is_bu48(pid)]
        trainval_proteins = [pid for pid in by_prot if not is_bu48(pid)]

        # 4) split remaining proteins into train/val
        rng = np.random.RandomState(self.val_split_seed)
        rng.shuffle(trainval_proteins)
        n_tv = len(trainval_proteins)
        val_n = int(self.val_split * n_tv)
        val_proteins = set(trainval_proteins[:val_n])
        train_proteins = set(trainval_proteins[val_n:])

        # 5) materialize residue indices
        self.train_indices = np.concatenate([by_prot[p] for p in train_proteins], dtype=np.int64) if train_proteins else np.array([], dtype=np.int64)
        self.val_indices   = np.concatenate([by_prot[p] for p in val_proteins],   dtype=np.int64) if val_proteins else np.array([], dtype=np.int64)
        self.test_indices  = np.concatenate([by_prot[p] for p in test_proteins],  dtype=np.int64) if test_proteins else np.array([], dtype=np.int64)

        rng.shuffle(self.train_indices); rng.shuffle(self.val_indices); rng.shuffle(self.test_indices)

        # 6) logging + leakage check
        log.info(f"Protein-level splits — Train proteins: {len(train_proteins)}, "
                 f"Val proteins: {len(val_proteins)}, Test (BU48): {len(test_proteins)}")
        log.info(f"Indices — Train: {len(self.train_indices)}, Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")

        # ensure no overlap at the PDB level (note: chain-level embeddings may share same PDB)
        def to_pdb4set(pidset): return { _extract_pdb4(p) for p in pidset if _extract_pdb4(p) }
        tp, vp = to_pdb4set(train_proteins), to_pdb4set(val_proteins)
        overlap = (tp & vp) | (tp & bu48_codes) | (vp & bu48_codes)
        if overlap:
            log.info(f"ℹ️ PDB overlap detected (different chains): {sorted(list(overlap))[:5]} ...")
            log.info("Note: This is expected with chain-level embeddings where different chains from the same PDB are in different splits.")
        else:
            log.info("✅ Verified: No BU48 in train/val and no train↔val protein overlap.")

        self._apply_train_indices_override()
        self._apply_hard_positive_replay()
    
    def _apply_train_indices_override(self):
        """Replace training indices with curated list if provided."""
        if self.train_indices_override_path is None:
            return
        path = self.train_indices_override_path
        if not path.exists():
            log.warning(f"Train override file not found: {path}")
            return
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception as e:
            log.warning(f"Failed to load train override file {path}: {e}")
            return
        if isinstance(payload, dict):
            override = payload.get("train_indices") or payload.get("indices")
            groups = payload.get("groups")
            metadata = payload.get("metadata")
        else:
            override = payload
            groups = None
            metadata = None
        if override is None:
            log.warning(
                f"Train override payload from {path} missing 'train_indices' or 'indices' key."
            )
            return
        override_array = np.array(override, dtype=np.int64)
        if override_array.size == 0:
            log.warning(f"Train override file {path} is empty; keeping original train indices.")
            return

        base_train: Set[int] = set(int(x) for x in np.array(self.train_indices, dtype=np.int64))
        mask = np.array([int(idx) in base_train for idx in override_array], dtype=bool)
        filtered = override_array[mask]
        dropped = int(override_array.size - filtered.size)
        if filtered.size == 0:
            log.warning(
                "No indices from %s overlapped with the baseline train split; "
                "retaining original train indices.",
                path,
            )
            return

        seed = self.train_override_shuffle_seed
        if seed is None:
            seed = self.val_split_seed + 11
        rng = np.random.default_rng(seed)
        rng.shuffle(filtered)
        self.train_indices = filtered.astype(np.int64, copy=False)
        log.info(
            "Curated train override applied: %d samples (dropped %d non-train indices) from %s",
            len(self.train_indices),
            dropped,
            path,
        )

        if groups and isinstance(groups, dict):
            summaries = []
            for name, values in groups.items():
                arr = np.array(values, dtype=np.int64)
                overlap = sum(int(v) in base_train for v in arr)
                summaries.append(f"{name}={overlap}")
            if summaries:
                log.info("Curated group summary: %s", ", ".join(summaries))

        if metadata and isinstance(metadata, dict):
            for key, value in metadata.items():
                log.info(f"Curated metadata — {key}: {value}")
    
    def _apply_hard_positive_replay(self):
        """Optionally replicate hard positive examples to emphasize them during training."""
        if self.hard_positive_indices_path is None:
            return
        path = self.hard_positive_indices_path
        if not path.exists():
            log.warning(f"Hard-positive indices file not found: {path}")
            return
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "indices" in payload:
                hard_indices = payload["indices"]
            else:
                hard_indices = payload
            hard_array = np.array(hard_indices, dtype=np.int64)
        except Exception as e:
            log.warning(f"Failed to load hard-positive indices from {path}: {e}")
            return
        if hard_array.size == 0:
            log.info(f"Hard-positive file {path} is empty; skipping replay.")
            return
        if self.train_indices is None or self.train_indices.size == 0:
            log.warning("Training indices not initialised; skipping hard-positive replay.")
            return
        selected = np.intersect1d(hard_array, self.train_indices, assume_unique=False)
        if selected.size == 0:
            log.info(f"No hard-positive indices overlap with training split (file: {path}).")
            return
        # Repeat each selected index hard_positive_repeat-1 additional times
        repeat_factor = self.hard_positive_repeat
        if repeat_factor <= 1:
            log.info(
                f"Hard-positive list provided ({path}), but repeat factor <=1; keeping original sampling."
            )
            self._hard_positive_selected = selected.size
            return
        extras = np.repeat(selected, repeat_factor - 1)
        augmented = np.concatenate([self.train_indices, extras])
        rng = np.random.default_rng(self.val_split_seed + 17)
        rng.shuffle(augmented)
        self.train_indices = augmented.astype(np.int64, copy=False)
        self._hard_positive_selected = selected.size
        log.info(
            f"Hard-positive replay: using {selected.size} base indices (repeat={repeat_factor}) "
            f"→ added {extras.size} extra samples. New train size: {len(self.train_indices)}"
        )
    
    def train_dataloader(self):
        """Create training dataloader with DistributedSampler."""
        dataset = SharedMemoryDataset(
            indices=self.train_indices,
            shared_tensors=self.shared_tensors,
            transform=self.transform,
            enable_knn=self.enable_knn,
            k_max=self.k_max,
            pdb_base_dir=self.pdb_base_dir,
            k_neighbors=self.hparams.get('k_res_neighbors', 1),
            neighbor_weighting=self.hparams.get('neighbor_weighting', 'softmax'),
            neighbor_temp=self.hparams.get('neighbor_temp', 2.0)
        )
        
        # Pass the k-NN cache to the dataset
        if self.enable_knn and hasattr(self, 'knn_cache'):
            dataset.knn_cache = self.knn_cache
        
        sampler = DistributedSampler(
            dataset, 
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=True,
            drop_last=True
        ) if self.trainer.world_size > 1 else None
        
        # Use enhanced collate function for k-NN or transformer mode
        # (transformer mode needs it to collapse scalar string fields like aggregation_mode)
        collate_fn = None
        aggregation_mode = self.shared_tensors.get('aggregation_mode', 'mean')
        if self.enable_knn or aggregation_mode == 'transformer':
            try:
                from src.data.knn_collate import enhanced_collate_fn
                collate_fn = enhanced_collate_fn
                log.info(f"Using enhanced collate function (mode: {aggregation_mode}, knn: {self.enable_knn})")
            except ImportError:
                log.warning("Enhanced collate function not available, using default")
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        dataset = SharedMemoryDataset(
            indices=self.val_indices,
            shared_tensors=self.shared_tensors,
            transform=self.transform,
            enable_knn=self.enable_knn,
            k_max=self.k_max,
            pdb_base_dir=self.pdb_base_dir,
            k_neighbors=self.hparams.get('k_res_neighbors', 1),
            neighbor_weighting=self.hparams.get('neighbor_weighting', 'softmax'),
            neighbor_temp=self.hparams.get('neighbor_temp', 2.0)
        )
        
        # Pass the k-NN cache to the dataset
        if self.enable_knn and hasattr(self, 'knn_cache'):
            dataset.knn_cache = self.knn_cache
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.trainer.world_size, 
            rank=self.trainer.global_rank,
            shuffle=False,
            drop_last=False
        ) if self.trainer.world_size > 1 else None
        
        # Use enhanced collate function for k-NN or transformer mode
        collate_fn = None
        aggregation_mode = self.shared_tensors.get('aggregation_mode', 'mean')
        if self.enable_knn or aggregation_mode == 'transformer':
            try:
                from src.data.knn_collate import enhanced_collate_fn
                collate_fn = enhanced_collate_fn
            except ImportError:
                pass
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        """Create test dataloader."""
        dataset = SharedMemoryDataset(
            indices=self.test_indices,
            shared_tensors=self.shared_tensors,
            transform=self.transform,
            enable_knn=self.enable_knn,
            k_max=self.k_max,
            pdb_base_dir=self.pdb_base_dir,
            k_neighbors=self.hparams.get('k_res_neighbors', 1),
            neighbor_weighting=self.hparams.get('neighbor_weighting', 'softmax'),
            neighbor_temp=self.hparams.get('neighbor_temp', 2.0)
        )
        
        # Pass the k-NN cache to the dataset
        if self.enable_knn and hasattr(self, 'knn_cache'):
            dataset.knn_cache = self.knn_cache
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank, 
            shuffle=False,
            drop_last=False
        ) if self.trainer.world_size > 1 else None
        
        # Use enhanced collate function for k-NN or transformer mode
        collate_fn = None
        aggregation_mode = self.shared_tensors.get('aggregation_mode', 'mean')
        if self.enable_knn or aggregation_mode == 'transformer':
            try:
                from src.data.knn_collate import enhanced_collate_fn
                collate_fn = enhanced_collate_fn
            except ImportError:
                pass
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn
        )
