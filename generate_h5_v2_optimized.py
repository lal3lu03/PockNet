#!/usr/bin/env python3
"""Streamed H5 generator with residue-table neighbours.

This version avoids holding the full CSV in memory and stores neighbour metadata
as indices into a residue lookup table instead of duplicating embeddings per
sample. The layout is compatible with the shared-memory datamodule once it
recognises the new ``transformer_format = 'v3_residue_table'`` attribute.
"""

import os
import sys
import gc
import csv
import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from multiprocessing import cpu_count

import h5py
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree

try:
    import torch  # noqa: F401  # Used by generate_complete_h5_files helpers
except Exception:  # pragma: no cover - torch is optional for inspection only
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
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("h5-generator")

# make sure we can import helper utilities from the project
sys.path.insert(0, str(Path(__file__).parent))
from generate_complete_h5_files import (  # noqa: E402
    load_bu48_list,
    protein_key_from_row,
    find_chain_emb_file,
    load_chain_embedding,
    choose_vector_by_resnum,
    grouped_split,
    get_label,
)

FEATURE_EXCLUDE = {
    "file_name",
    "pdb_id",
    "protein_id",
    "chain_id",
    "residue_id",
    "residue_number",
    "res_num",
    "position",
    "residue_name",
    "atom_name",
    "x",
    "y",
    "z",
    "binding_site",
    "class",
    "protein_key",
    "is_bu48",
}


def compute_feature_columns(header: List[str]) -> List[str]:
    return [col for col in header if col not in FEATURE_EXCLUDE]


def is_protein_in_bu48(protein_id: str, bu48_set: Set[str]) -> bool:
    if not protein_id:
        return False
    pid_lower = protein_id.lower()
    if pid_lower in bu48_set:
        return True
    if len(pid_lower) >= 4 and pid_lower[:4] in bu48_set:
        return True
    return False


def analyse_dataset(csv_path: Path, bu48_set: Set[str], val_frac: float, seed: int) -> Tuple[List[str], List[str], Dict[str, bool], Dict[str, int], Dict[str, List[Dict[str, str]]]]:
    """Load entire CSV into RAM, group by protein, and return deduplicated data."""
    protein_order: List[str] = []
    protein_is_bu48: Dict[str, bool] = {}
    protein_rows: Dict[str, List[Dict[str, str]]] = {}  # NEW: Store all rows per protein in RAM

    logger.info("Loading entire CSV into RAM and grouping by protein...")
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("CSV appears to be empty – missing header row?")
        feature_cols = compute_feature_columns(reader.fieldnames)

        row_count = 0
        for row in reader:
            row_count += 1
            protein_key = protein_key_from_row(row)
            
            # First time seeing this protein
            if protein_key not in protein_is_bu48:
                protein_id = row.get("protein_id") or Path(str(row.get("file_name", ""))).stem
                protein_is_bu48[protein_key] = is_protein_in_bu48(protein_id, bu48_set)
                protein_order.append(protein_key)
                protein_rows[protein_key] = []
            
            # Store this row for this protein
            protein_rows[protein_key].append(row)
            
            if row_count % 500000 == 0:
                logger.info(f"  Loaded {row_count:,} rows, {len(protein_order)} unique proteins...")

    logger.info(f"Loaded {row_count:,} rows into RAM")
    logger.info(f"Found {len(protein_order)} unique proteins")
    
    non_bu48 = [key for key in protein_order if not protein_is_bu48[key]]
    split_map = grouped_split(non_bu48, val_frac, seed)
    return feature_cols, protein_order, protein_is_bu48, split_map, protein_rows


def iter_protein_batches(csv_path: Path):
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        current_key: Optional[str] = None
        buffer: List[Dict[str, str]] = []

        for row in reader:
            protein_key = protein_key_from_row(row)
            if current_key is None:
                current_key = protein_key

            if protein_key != current_key and buffer:
                yield current_key, buffer
                buffer = []
                current_key = protein_key

            buffer.append(row)

        if buffer:
            assert current_key is not None
            yield current_key, buffer


def load_pdb_index(ds_path: Path, base_dir: Path) -> Dict[str, Path]:
    """Load PDB index from .ds file, mapping base protein IDs to PDB file paths.
    
    Returns a dict mapping lowercase base protein ID (without chain) to absolute PDB path.
    E.g., '5jdw' -> Path('.../dt198/5jdw_A.pdb')
    """
    index: Dict[str, Path] = {}
    if not ds_path or not ds_path.exists():
        return index

    with ds_path.open() as handle:
        for line in handle:
            entry = line.strip()
            if not entry or entry.startswith("#") or entry.startswith("PARAM."):
                continue

            rel_path = Path(entry)
            
            # Strip p2rank-datasets/ prefix if it exists in the path
            path_parts = rel_path.parts
            if path_parts and path_parts[0] == "p2rank-datasets":
                rel_path = Path(*path_parts[1:])
            
            abs_path = base_dir / rel_path
            
            stem = rel_path.stem  # e.g., "5jdw_A" or "1krn" or "a.001.001.001_1s69a"
            
            # Extract base protein ID (strip chain suffix if present)
            base_id = stem
            if "_" in stem:
                parts = stem.rsplit("_", 1)
                # Check if last part is a chain ID (1-2 letters, possibly with uppercase)
                if len(parts) == 2 and len(parts[1]) <= 2 and parts[1].replace("_", "").isalpha():
                    base_id = parts[0]
            
            # Also extract 4-letter PDB code if it's a SCOP-style ID
            pdb_code = None
            if "." in base_id and "_" in base_id:
                # SCOP format like "a.001.001.001_1s69a"
                pdb_part = base_id.split("_")[-1]
                if len(pdb_part) >= 4:
                    pdb_code = pdb_part[:4]
            elif len(base_id) >= 4:
                # Regular PDB code like "5jdw"
                pdb_code = base_id[:4]
            
            # Store with lowercase keys for case-insensitive lookup
            index[base_id.lower()] = abs_path
            if pdb_code:
                index[pdb_code.lower()] = abs_path

    return index


def extract_ca_coordinates_fast(pdb_path: Optional[Path], chain_id: str) -> Optional[Dict[int, np.ndarray]]:
    if pdb_path is None or not pdb_path.exists():
        return None

    def collect_gemmi(chain) -> Dict[int, np.ndarray]:
        coords: Dict[int, np.ndarray] = {}
        for residue in chain:
            atom = residue.find_atom("CA", "")
            if atom is not None:
                coords[residue.seqid.num] = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
        return coords

    if HAS_GEMMI:
        try:
            structure = gemmi.read_structure(str(pdb_path))
            model = structure[0]
            target_chain = None
            
            # Try exact chain match first
            for chain in model:
                if chain.name == chain_id:
                    target_chain = chain
                    break
            
            # Fallback: try empty chain name if looking for 'A'
            if target_chain is None and chain_id == "A":
                for chain in model:
                    if chain.name.strip() == "":
                        target_chain = chain
                        break
            
            # Fallback: if single chain, use it regardless of name
            if target_chain is None and len(model) == 1:
                candidate = list(model)[0]
                if any(res.find_atom("CA", "") for res in candidate):
                    target_chain = candidate
            
            # Fallback: try any chain with CA atoms
            if target_chain is None:
                for chain in model:
                    coords = collect_gemmi(chain)
                    if coords:
                        logger.debug(f"Using chain {chain.name} instead of {chain_id} for {pdb_path.name}")
                        return coords
                return None
            
            coords = collect_gemmi(target_chain)
            if coords:
                return coords
        except Exception as e:
            logger.debug(f"Gemmi failed for {pdb_path.name}: {e}")

    if not HAS_BIOPYTHON:
        return None

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_path)
    except Exception:
        return None

    target_chain = None
    for model in structure:
        for chain in model:
            if (chain.id or "").strip() == chain_id:
                target_chain = chain
                break
        if target_chain:
            break

    if target_chain is None and chain_id == "A":
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
                coords = {
                    residue.id[1]: np.array(residue["CA"].get_coord(), dtype=np.float32)
                    for residue in chain
                    if residue.id[0] == " " and "CA" in residue
                }
                if coords:
                    return coords
        return None

    coords = {}
    for residue in target_chain:
        if residue.id[0] != " ":
            continue
        if "CA" not in residue:
            continue
        coords[residue.id[1]] = np.array(residue["CA"].get_coord(), dtype=np.float32)
    return coords if coords else None


def build_kdtree(ca_coords: Dict[int, np.ndarray]) -> Tuple[cKDTree, np.ndarray]:
    resnums = np.array(sorted(ca_coords.keys()), dtype=np.int32)
    coords = np.vstack([ca_coords[r] for r in resnums])
    return cKDTree(coords), resnums


def compute_euclidean_neighbors_fast(
    target_resnum: int,
    all_resnums: Optional[List[int]],
    ca_coords: Optional[Dict[int, np.ndarray]],
    kdtree: Optional[cKDTree],
    kdtree_resnums: Optional[np.ndarray],
    k: int,
) -> Tuple[List[int], List[float], bool]:
    if ca_coords is None or kdtree is None:
        raise RuntimeError(f"PDB coordinates required but not available for residue {target_resnum}")
    
    # If target residue not in PDB, find the nearest available residue
    if target_resnum not in ca_coords:
        if not ca_coords:
            raise RuntimeError(f"No CA coordinates available")
        
        # Find the closest residue number that exists in the PDB
        available_resnums = sorted(ca_coords.keys())
        closest_resnum = min(available_resnums, key=lambda x: abs(x - target_resnum))
        logger.debug(f"Residue {target_resnum} not in PDB, using nearest residue {closest_resnum}")
        target_resnum = closest_resnum

    try:
        dists, idxs = kdtree.query(ca_coords[target_resnum], k=min(k + 1, len(kdtree_resnums)))
        if np.isscalar(dists):
            dists = np.array([dists])
            idxs = np.array([idxs])
        mask = kdtree_resnums[idxs] != target_resnum
        neighbor_resnums = kdtree_resnums[idxs[mask]][:k].tolist()
        neighbor_distances = dists[mask][:k].astype(np.float32).tolist()
        while len(neighbor_resnums) < k:
            neighbor_resnums.append(target_resnum)
            neighbor_distances.append(0.0)
        return neighbor_resnums, neighbor_distances, True
    except Exception as e:
        raise RuntimeError(f"Failed to compute Euclidean neighbors for residue {target_resnum}: {e}")


def compute_sequence_neighbors(target_resnum: int, all_resnums: Optional[List[int]], k: int) -> Tuple[List[int], List[float], bool]:
    if all_resnums is None or not all_resnums:
        return [target_resnum] * k, [0.0] * k, False

    distances: List[float] = []
    candidates: List[int] = []
    for resnum in all_resnums:
        if resnum == target_resnum:
            continue
        candidates.append(resnum)
        distances.append(abs(resnum - target_resnum))

    if not candidates:
        return [target_resnum] * k, [0.0] * k, False

    order = np.argsort(distances)[:k]
    neighbor_resnums = [int(candidates[i]) for i in order]
    neighbor_distances = [float(distances[i]) for i in order]
    while len(neighbor_resnums) < k:
        neighbor_resnums.append(target_resnum)
        neighbor_distances.append(0.0)
    return neighbor_resnums, neighbor_distances, False


def probe_esm_dim(protein_keys: List[str], esm_dir: Path) -> int:
    for protein_key in protein_keys:
        emb_path = find_chain_emb_file(esm_dir, protein_key)
        if not emb_path:
            continue
        emb, _ = load_chain_embedding(emb_path)
        if emb is not None:
            return emb.shape[-1]
    raise RuntimeError("Unable to determine ESM embedding dimensionality; no embeddings found")


class H5WriterTransformerV2Optimized:
    """Streaming-friendly writer that stores neighbour indices against a residue lookup table."""

    def __init__(self, out_path: Path, tab_dim: int, esm_dim: int, k_neighbors: int, feat_names: List[str]):
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.k_neighbors = k_neighbors
        self.esm_dim = esm_dim

        self.f = h5py.File(
            out_path,
            "w",
            libver="latest",
            rdcc_nbytes=256 * 1024**2,
            rdcc_w0=0.75,
        )

        self._sample_n = 0
        self._sample_capacity = 100_000
        chunk_samples = 100_000

        self.tab = self.f.create_dataset(
            "tabular",
            shape=(self._sample_capacity, tab_dim),
            maxshape=(None, tab_dim),
            chunks=(chunk_samples, tab_dim),
            compression="lzf",
            shuffle=True,
            track_times=False,
        )
        self.esm = self.f.create_dataset(
            "esm",
            shape=(self._sample_capacity, esm_dim),
            maxshape=(None, esm_dim),
            chunks=(chunk_samples, esm_dim),
            compression="lzf",
            shuffle=True,
            track_times=False,
        )
        self.lbl = self.f.create_dataset(
            "labels",
            shape=(self._sample_capacity,),
            maxshape=(None,),
            dtype="i4",
            chunks=(chunk_samples,),
            track_times=False,
        )
        self.pky = self.f.create_dataset(
            "protein_keys",
            shape=(self._sample_capacity,),
            maxshape=(None,),
            dtype=h5py.string_dtype("utf-8"),
            chunks=(chunk_samples,),
            compression="lzf",
            track_times=False,
        )
        self.rno = self.f.create_dataset(
            "residue_numbers",
            shape=(self._sample_capacity,),
            maxshape=(None,),
            dtype="i4",
            chunks=(chunk_samples,),
            track_times=False,
        )
        self.spl = self.f.create_dataset(
            "split",
            shape=(self._sample_capacity,),
            maxshape=(None,),
            dtype="i1",
            chunks=(chunk_samples,),
            track_times=False,
        )
        self.neighbor_idx = self.f.create_dataset(
            "neighbor_residue_indices",
            shape=(self._sample_capacity, k_neighbors),
            maxshape=(None, k_neighbors),
            dtype="i4",
            chunks=(chunk_samples, k_neighbors),
            compression="lzf",
            shuffle=True,
            track_times=False,
        )
        self.neighbor_distances = self.f.create_dataset(
            "neighbor_distances",
            shape=(self._sample_capacity, k_neighbors),
            maxshape=(None, k_neighbors),
            dtype="f2",
            chunks=(chunk_samples, k_neighbors),
            compression="lzf",
            shuffle=True,
            track_times=False,
        )
        self.neighbor_resnums = self.f.create_dataset(
            "neighbor_resnums",
            shape=(self._sample_capacity, k_neighbors),
            maxshape=(None, k_neighbors),
            dtype="i4",
            chunks=(chunk_samples, k_neighbors),
            track_times=False,
        )

        self._residue_n = 0
        self._residue_capacity = 200_000
        chunk_residues = 100_000
        self.residue_embeddings = self.f.create_dataset(
            "residue_embeddings",
            shape=(self._residue_capacity, esm_dim),
            maxshape=(None, esm_dim),
            dtype="f4",
            chunks=(chunk_residues, esm_dim),
            compression="lzf",
            shuffle=True,
            track_times=False,
        )
        self.residue_protein_keys = self.f.create_dataset(
            "residue_protein_keys",
            shape=(self._residue_capacity,),
            maxshape=(None,),
            dtype=h5py.string_dtype("utf-8"),
            chunks=(chunk_residues,),
            compression="lzf",
            track_times=False,
        )
        self.residue_numbers_ds = self.f.create_dataset(
            "residue_numbers_table",
            shape=(self._residue_capacity,),
            maxshape=(None,),
            dtype="i4",
            chunks=(chunk_residues,),
            track_times=False,
        )

        self.f.create_dataset(
            "feature_names",
            data=np.array([name.encode("utf-8") for name in feat_names]),
            compression="gzip",
        )

        self.distance_metrics: List[bool] = []

    # ------------------------------------------------------------------
    def _ensure_sample_capacity(self, additional: int) -> None:
        required = self._sample_n + additional
        if required <= self._sample_capacity:
            return
        new_capacity = max(required, int(self._sample_capacity * 1.5))
        self.tab.resize((new_capacity, self.tab.shape[1]))
        self.esm.resize((new_capacity, self.esm.shape[1]))
        self.lbl.resize((new_capacity,))
        self.pky.resize((new_capacity,))
        self.rno.resize((new_capacity,))
        self.spl.resize((new_capacity,))
        self.neighbor_idx.resize((new_capacity, self.k_neighbors))
        self.neighbor_distances.resize((new_capacity, self.k_neighbors))
        self.neighbor_resnums.resize((new_capacity, self.k_neighbors))
        self._sample_capacity = new_capacity

    def _ensure_residue_capacity(self, additional: int) -> None:
        required = self._residue_n + additional
        if required <= self._residue_capacity:
            return
        new_capacity = max(required, int(self._residue_capacity * 1.5))
        self.residue_embeddings.resize((new_capacity, self.residue_embeddings.shape[1]))
        self.residue_protein_keys.resize((new_capacity,))
        self.residue_numbers_ds.resize((new_capacity,))
        self._residue_capacity = new_capacity

    def add_residues(self, protein_key: str, residue_numbers: np.ndarray, residue_embeddings: np.ndarray) -> np.ndarray:
        count = int(len(residue_numbers))
        if count == 0:
            return np.empty((0,), dtype=np.int64)
        self._ensure_residue_capacity(count)
        start = self._residue_n
        end = start + count

        self.residue_embeddings[start:end] = residue_embeddings.astype(np.float32, copy=False)
        self.residue_protein_keys[start:end] = np.asarray([protein_key] * count, dtype=object)
        self.residue_numbers_ds[start:end] = residue_numbers.astype(np.int32, copy=False)
        self._residue_n = end
        return np.arange(start, end, dtype=np.int64)

    def append_samples(
        self,
        tabular: np.ndarray,
        esm: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_dists: np.ndarray,
        neighbor_resnums: np.ndarray,
        labels: np.ndarray,
        protein_keys: List[str],
        residue_numbers: np.ndarray,
        split_flags: np.ndarray,
        used_euclidean_flags: List[bool],
    ) -> None:
        batch = tabular.shape[0]
        self._ensure_sample_capacity(batch)
        start = self._sample_n
        end = start + batch

        self.tab[start:end] = tabular.astype(np.float32, copy=False)
        self.esm[start:end] = esm.astype(np.float32, copy=False)
        self.lbl[start:end] = labels.astype(np.int32, copy=False)
        self.pky[start:end] = np.asarray(protein_keys, dtype=object)
        self.rno[start:end] = residue_numbers.astype(np.int32, copy=False)
        self.spl[start:end] = split_flags.astype(np.int8, copy=False)
        self.neighbor_idx[start:end] = neighbor_indices.astype(np.int32, copy=False)
        self.neighbor_distances[start:end] = neighbor_dists.astype(np.float16, copy=False)
        self.neighbor_resnums[start:end] = neighbor_resnums.astype(np.int32, copy=False)
        self.distance_metrics.extend(used_euclidean_flags)
        self._sample_n = end

    def finalize(self, meta: Dict) -> None:
        self.tab.resize((self._sample_n, self.tab.shape[1]))
        self.esm.resize((self._sample_n, self.esm.shape[1]))
        self.lbl.resize((self._sample_n,))
        self.pky.resize((self._sample_n,))
        self.rno.resize((self._sample_n,))
        self.spl.resize((self._sample_n,))
        self.neighbor_idx.resize((self._sample_n, self.k_neighbors))
        self.neighbor_distances.resize((self._sample_n, self.k_neighbors))
        self.neighbor_resnums.resize((self._sample_n, self.k_neighbors))

        self.residue_embeddings.resize((self._residue_n, self.residue_embeddings.shape[1]))
        self.residue_protein_keys.resize((self._residue_n,))
        self.residue_numbers_ds.resize((self._residue_n,))

        for key, value in meta.items():
            self.f.attrs[key] = value

        self.f.attrs["aggregation_mode"] = "transformer"
        self.f.attrs["knn_k"] = self.k_neighbors
        self.f.attrs["transformer_format"] = "v3_residue_table"
        self.f.attrs["residue_table_size"] = int(self._residue_n)

        euclidean_count = sum(self.distance_metrics)
        total = max(len(self.distance_metrics), 1)
        sequence_count = total - euclidean_count

        if sequence_count > 0:
            raise RuntimeError(f"Sequence fallback was used {sequence_count} times - this should not happen!")
        
        metric = "euclidean_angstrom"

        self.f.attrs["neighbor_distance_metric"] = metric
        self.f.attrs["euclidean_samples"] = int(euclidean_count)
        self.f.attrs["sequence_fallback_samples"] = int(sequence_count)
        self.f.attrs["fallback_rate"] = float(sequence_count) / float(total)

        logger.info(
            "Distance metric: 100%% Euclidean (%s samples)",
            f"{euclidean_count:,}",
        )

        self.f.close()


def process_protein_batch(
    protein_key: str,
    rows: List[Dict[str, str]],
    feature_cols: List[str],
    split_flag: int,
    esm_dir: Path,
    pdb_index: Dict[str, Path],
    pdb_base_dir: Path,
    k_neighbors: int,
) -> Optional[Dict]:
    try:
        if not rows:
            return None

        first_row = rows[0]
        protein_id_raw = str(first_row.get("protein_id", "")).strip() or Path(str(first_row.get("file_name", ""))).stem
        chain_id = str(first_row.get("chain_id", "A")).strip() or "A"

        emb_path = find_chain_emb_file(esm_dir, protein_key)
        if not emb_path:
            raise RuntimeError(f"Embedding file missing for {protein_key}")
        emb, resnums = load_chain_embedding(emb_path)
        if emb is None:
            raise RuntimeError(f"Failed to load embeddings for {protein_key}")

        if resnums is None or len(resnums) == 0:
            resnums = list(range(emb.shape[0]))
        resnums = [int(r) for r in resnums]
        resnum_to_local_idx = {resnum: idx for idx, resnum in enumerate(resnums)}

        pdb_path = None
        
        # Extract base protein ID (without chain suffix)
        base_pdb_id = protein_id_raw
        if "_" in protein_id_raw:
            parts = protein_id_raw.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) <= 2 and parts[1].replace("_", "").isalpha():
                base_pdb_id = parts[0]
        
        # Try PDB index first (most reliable - uses paths from .ds file)
        if pdb_index:
            # Try various lookups in order of specificity
            lookup_keys = [
                base_pdb_id.lower(),  # e.g., "5jdw", "a.001.001.001_1s69a"
                protein_id_raw.lower(),  # e.g., "5jdw_a" (unlikely but try anyway)
            ]
            
            # Also try 4-letter PDB code if available
            if len(base_pdb_id) >= 4:
                pdb_code = base_pdb_id[:4].lower()
                if pdb_code not in lookup_keys:
                    lookup_keys.append(pdb_code)
            
            for key in lookup_keys:
                candidate = pdb_index.get(key)
                if candidate and candidate.exists():
                    pdb_path = candidate
                    break
        
        # Direct file lookup as fallback (for files not in .ds index)
        if pdb_path is None:
            search_locations = [
                pdb_base_dir / f"{base_pdb_id}.pdb",
                pdb_base_dir / "joined" / "bu48" / f"{base_pdb_id}.pdb",
                pdb_base_dir / "joined" / "chen11" / f"{base_pdb_id}.pdb",
                pdb_base_dir / "joined" / "coach420" / f"{base_pdb_id}.pdb",
                pdb_base_dir / "joined" / "holo4k" / f"{base_pdb_id}.pdb",
                pdb_base_dir / "joined" / "dt198" / f"{protein_id_raw}.pdb",  # dt198 keeps chain suffix
                pdb_base_dir / "joined" / "astex" / f"{base_pdb_id}.pdb",
                pdb_base_dir / "joined" / "b210" / f"{base_pdb_id}.pdb",
            ]
            for candidate in search_locations:
                if candidate.exists():
                    pdb_path = candidate
                    break

        if pdb_path is None:
            logger.warning(f"No PDB file found for {protein_key} - skipping protein")
            return None
        
        ca_coords = extract_ca_coordinates_fast(pdb_path, chain_id)
        if ca_coords is None or not ca_coords:
            logger.warning(f"Failed to extract CA coordinates from PDB {pdb_path} for {protein_key} - skipping protein")
            return None
        
        try:
            kdtree, kdtree_resnums = build_kdtree(ca_coords)
        except Exception as e:
            logger.warning(f"Failed to build KDTree for {protein_key}: {e} - skipping protein")
            return None

        num_samples = len(rows)
        num_features = len(feature_cols)

        tabular = np.zeros((num_samples, num_features), dtype=np.float32)
        esm_vectors = np.zeros((num_samples, emb.shape[1]), dtype=np.float32)
        labels = np.zeros((num_samples,), dtype=np.int32)
        residue_numbers = np.full((num_samples,), -1, dtype=np.int32)
        split_flags = np.full((num_samples,), split_flag, dtype=np.int8)
        neighbor_resnums = np.full((num_samples, k_neighbors), -1, dtype=np.int32)
        neighbor_dists = np.full((num_samples, k_neighbors), 999.0, dtype=np.float32)
        used_euclidean_flags: List[bool] = []

        stats = {"euclidean": 0, "sequence": 0, "exact": 0, "mean_pooled": 0}

        for idx, row in enumerate(rows):
            feature_values = []
            for col in feature_cols:
                value = row.get(col, "")
                if value in (None, ""):
                    feature_values.append(0.0)
                else:
                    try:
                        feature_values.append(float(value))
                    except ValueError:
                        feature_values.append(0.0)
            tabular[idx] = np.nan_to_num(np.asarray(feature_values, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)

            labels[idx] = int(get_label(row))

            residue_val = row.get("residue_number", row.get("res_num", ""))
            if residue_val in ("", None):
                resno = -1
            else:
                try:
                    resno = int(float(residue_val))
                except ValueError:
                    resno = -1
            residue_numbers[idx] = resno

            centre = choose_vector_by_resnum(emb, resnums, resno if resno >= 0 else None)
            esm_vectors[idx] = centre.astype(np.float32, copy=False)

            if resno >= 0 and resno in resnum_to_local_idx:
                stats["exact"] += 1
            else:
                stats["mean_pooled"] += 1

            if resno >= 0 and resnums:
                try:
                    neigh_resnums, neigh_dists, used_euclidean = compute_euclidean_neighbors_fast(
                        resno, resnums, ca_coords, kdtree, kdtree_resnums, k_neighbors
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute neighbors for residue {resno} in {protein_key}: {e}")
                    # Use fallback: duplicate the center residue
                    neigh_resnums = [resno] * k_neighbors
                    neigh_dists = [0.0] * k_neighbors
                    used_euclidean = True  # Still counts as euclidean since we have PDB coords
            else:
                # Invalid or missing residue number - use first available residue as fallback
                if ca_coords and kdtree_resnums is not None and len(kdtree_resnums) > 0:
                    fallback_resnum = int(kdtree_resnums[0])
                    logger.debug(f"Invalid residue number {resno}, using fallback {fallback_resnum}")
                    try:
                        neigh_resnums, neigh_dists, used_euclidean = compute_euclidean_neighbors_fast(
                            fallback_resnum, resnums, ca_coords, kdtree, kdtree_resnums, k_neighbors
                        )
                    except Exception as e:
                        logger.warning(f"Fallback failed for {protein_key}: {e}")
                        neigh_resnums = [fallback_resnum] * k_neighbors
                        neigh_dists = [0.0] * k_neighbors
                        used_euclidean = True
                else:
                    logger.warning(f"No valid residues available for {protein_key}")
                    neigh_resnums = [-1] * k_neighbors
                    neigh_dists = [999.0] * k_neighbors
                    used_euclidean = True

            stats["euclidean" if used_euclidean else "sequence"] += 1
            used_euclidean_flags.append(used_euclidean)

            for j in range(k_neighbors):
                if j < len(neigh_resnums):
                    neighbor_resnums[idx, j] = int(neigh_resnums[j])
                    neighbor_dists[idx, j] = float(neigh_dists[j])
                else:
                    neighbor_resnums[idx, j] = -1
                    neighbor_dists[idx, j] = 999.0

        return {
            "protein_key": protein_key,
            "tab": tabular,
            "esm": esm_vectors,
            "labels": labels,
            "residue_numbers": residue_numbers,
            "split_flags": split_flags,
            "neighbor_resnums": neighbor_resnums,
            "neighbor_dists": neighbor_dists,
            "used_euclidean_flags": used_euclidean_flags,
            "residue_embeddings": emb.astype(np.float32, copy=False),
            "residue_resnums": np.asarray(resnums, dtype=np.int32),
            "stats": stats,
        }
    except Exception as exc:
        logger.warning("Failed to process protein %s: %s", protein_key, exc)
        return None


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
    pdb_index: Optional[Dict] = None,
) -> bool:
    logger.info("=" * 80)
    logger.info("Building transformer H5 (streaming mode, residue table)")
    logger.info("=" * 80)
    logger.info("CSV: %s", csv_path)
    logger.info("ESM dir: %s", esm_dir)
    logger.info("PDB base: %s", pdb_base_dir)
    logger.info("Output: %s", out_path)
    logger.info("Neighbors (k): %d", k_neighbors)

    bu48_set = load_bu48_list(bu48_txt) if bu48_txt else set()
    feature_cols, protein_order, protein_is_bu48, split_map, protein_rows = analyse_dataset(csv_path, bu48_set, val_frac, seed)
    logger.info("Features: %d", len(feature_cols))
    logger.info("Proteins: %d", len(protein_order))
    logger.info("Total rows in RAM: %d", sum(len(rows) for rows in protein_rows.values()))

    if not pdb_index:
        pdb_index = {}

    workers = max_workers if max_workers is not None else cpu_count()
    workers = max(1, min(workers, cpu_count()))
    logger.info("Using %d worker(s) for per-protein processing (max available: %d)", workers, cpu_count())

    esm_dim = probe_esm_dim(protein_order, esm_dir)
    writer = H5WriterTransformerV2Optimized(out_path, len(feature_cols), esm_dim, k_neighbors, feature_cols)

    total_samples = 0
    stats = {
        "successful_proteins": 0,
        "failed_proteins": 0,
        "skipped_proteins": 0,
        "exact_residue_hits": 0,
        "mean_pooled_fallbacks": 0,
        "euclidean_distance_used": 0,
        "sequence_distance_fallback": 0,
    }

    start_time = time.time()
    def handle_result(protein_key: str, result: Optional[Dict], rows_count: int) -> bool:
        nonlocal total_samples
        if result is None:
            stats["skipped_proteins"] += 1
            logger.debug("Protein %s skipped (no PDB or failed to process)", protein_key)
            return True  # Continue processing - don't abort

        residue_indices = writer.add_residues(
            protein_key,
            result["residue_resnums"],
            result["residue_embeddings"],
        )
        resnum_to_global = {int(resnum): int(idx) for resnum, idx in zip(result["residue_resnums"], residue_indices)}

        neighbor_indices = np.full_like(result["neighbor_resnums"], -1, dtype=np.int32)
        for i in range(rows_count):
            for j in range(k_neighbors):
                resnum = int(result["neighbor_resnums"][i, j])
                neighbor_indices[i, j] = resnum_to_global.get(resnum, -1)

        writer.append_samples(
            result["tab"],
            result["esm"],
            neighbor_indices,
            result["neighbor_dists"],
            result["neighbor_resnums"],
            result["labels"],
            [protein_key] * rows_count,
            result["residue_numbers"],
            result["split_flags"],
            result["used_euclidean_flags"],
        )

        stats["successful_proteins"] += 1
        stats["exact_residue_hits"] += result["stats"]["exact"]
        stats["mean_pooled_fallbacks"] += result["stats"]["mean_pooled"]
        stats["euclidean_distance_used"] += result["stats"]["euclidean"]
        stats["sequence_distance_fallback"] += result["stats"]["sequence"]
        total_samples += rows_count
        return True

    # Process proteins from RAM (not streaming from CSV)
    progress = tqdm(total=len(protein_order), desc="Proteins")

    if workers == 1:
        for protein_key in protein_order:
            rows = protein_rows[protein_key]
            split_flag = 2 if protein_is_bu48.get(protein_key, False) else split_map.get(protein_key, 0)
            result = process_protein_batch(
                protein_key,
                rows,
                feature_cols,
                split_flag,
                esm_dir,
                pdb_index,
                pdb_base_dir,
                k_neighbors,
            )
            rows_count = len(rows)
            if not handle_result(protein_key, result, rows_count):
                progress.close()
                return False

            if stats["successful_proteins"] % 100 == 0:
                gc.collect()
            progress.update(1)
    else:
        from concurrent.futures import ProcessPoolExecutor

        pending: List[Tuple] = []
        max_pending = workers * 2
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for protein_key in protein_order:
                rows = protein_rows[protein_key]
                split_flag = 2 if protein_is_bu48.get(protein_key, False) else split_map.get(protein_key, 0)
                future = pool.submit(
                    process_protein_batch,
                    protein_key,
                    rows,
                    feature_cols,
                    split_flag,
                    esm_dir,
                    pdb_index,
                    pdb_base_dir,
                    k_neighbors,
                )
                pending.append((future, protein_key, len(rows)))

                if len(pending) >= max_pending:
                    future, key, count = pending.pop(0)
                    result = future.result()
                    if not handle_result(key, result, count):
                        pool.shutdown(cancel_futures=True)
                        progress.close()
                        return False
                    if stats["successful_proteins"] % 100 == 0:
                        gc.collect()
                    progress.update(1)

            for future, key, count in pending:
                result = future.result()
                if not handle_result(key, result, count):
                    progress.close()
                    return False
                if stats["successful_proteins"] % 100 == 0:
                    gc.collect()
                progress.update(1)

    progress.close()

    elapsed = time.time() - start_time
    metadata = {
        "dataset_name": csv_path.stem,
        "num_samples": total_samples,
        "num_proteins": stats["successful_proteins"],
        "split_meaning": "0=train, 1=val, 2=test_bu48",
        "val_fraction": float(val_frac),
        "seed": int(seed),
        "esm_dimension": int(esm_dim),
        "tabular_features": int(len(feature_cols)),
        "processing_time_sec": float(elapsed),
        "processing_strategy": "streaming_v3",
    }
    writer.finalize(metadata)

    logger.info("=" * 80)
    logger.info("Completed: %d samples | %d proteins | %d skipped", total_samples, stats["successful_proteins"], stats["skipped_proteins"])
    logger.info("Time: %.1f minutes", elapsed / 60.0)
    logger.info("Residue exact hits: %d", stats["exact_residue_hits"])
    logger.info("Mean pooled fallbacks: %d", stats["mean_pooled_fallbacks"])
    logger.info("Euclidean neighbour usage: %d", stats["euclidean_distance_used"])
    logger.info("Sequence neighbour fallback: %d", stats["sequence_distance_fallback"])
    logger.info("=" * 80)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate transformer-ready H5 with streaming neighbours")
    parser.add_argument("--csv", type=Path, required=True, help="Path to vectorsTrain CSV")
    parser.add_argument("--esm_dir", type=Path, required=True, help="Directory holding chain embeddings (.pt)")
    parser.add_argument("--pdb_base_dir", type=Path, required=True, help="Base directory with PDB files")
    parser.add_argument("--bu48_txt", type=Path, required=True, help="BU48 protein list")
    parser.add_argument("--out", type=Path, required=True, help="Output H5 file path")
    parser.add_argument("--k", type=int, default=3, help="Number of neighbour residues to store")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction for non-BU48 proteins")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: all available CPUs)")
    parser.add_argument("--ds_file", type=Path, help="Optional .ds dataset file for fast PDB lookup")

    args = parser.parse_args()

    pdb_index = load_pdb_index(args.ds_file, args.pdb_base_dir) if args.ds_file else {}
    if pdb_index:
        logger.info("Loaded PDB index with %d entries", len(pdb_index))

    success = build_h5_v2_multiprocess(
        csv_path=args.csv,
        esm_dir=args.esm_dir,
        pdb_base_dir=args.pdb_base_dir,
        bu48_txt=args.bu48_txt,
        out_path=args.out,
        val_frac=args.val_frac,
        seed=args.seed,
        k_neighbors=args.k,
        max_workers=args.workers,
        pdb_index=pdb_index,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
