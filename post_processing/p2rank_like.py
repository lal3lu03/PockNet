#!/usr/bin/env python3
"""
P2Rank-Inspired Post-Processing
==============================

Python reimplementation of the pocket aggregation logic used by P2Rank.
Transforms SAS point predictions into ranked pocket outputs that mirror
the original Groovy pipeline.
"""

from __future__ import annotations

import csv
import io
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Parameter configuration
# --------------------------------------------------------------------------- #

@dataclass
class P2RankParams:
    """Configuration mirroring key P2Rank parameters."""

    pred_point_threshold: float = 0.35
    pred_min_cluster_size: int = 3
    pred_clustering_dist: float = 3.0
    extended_pocket_cutoff: float = 3.5
    pred_protein_surface_cutoff: float = 3.5
    point_score_pow: float = 2.0
    balance_density: bool = False
    balance_density_radius: float = 2.0
    score_point_limit: int = 0


@dataclass
class SASPoint:
    """Single SAS sample point."""

    index: int
    residue_number: int
    coords: np.ndarray  # shape (3,)
    prob: float


@dataclass
class P2RankPocket:
    """Pocket aggregate with metadata."""

    name: str
    rank: int
    score: float
    center: np.ndarray  # shape (3,)
    member_indices: np.ndarray
    cluster_indices: np.ndarray
    sum_prob: float
    surface_points: int

    def to_csv_row(self, protein_id: str, pocket_count: int) -> List[str]:
        """
        Render pocket row in the same column order as P2Rank's pockets.csv.

        Columns:
            file, #ligands, #pockets, pocket, ligand, rank, score,
            newRank, oldScore, zScoreTP, probaTP, samplePoints,
            rawNewScore, pocketVolume, surfaceAtoms
        """
        return [
            protein_id,
            "0",  # #ligands (unknown without GT)
            str(pocket_count),
            self.name,
            "",  # ligand name unknown
            str(self.rank),
            f"{self.score:.6f}",
            str(self.rank),            # newRank
            f"{self.score:.6f}",       # oldScore (mirror)
            "",                        # zScoreTP
            "",                        # probaTP
            str(len(self.cluster_indices)),
            f"{self.sum_prob:.6f}",    # rawNewScore proxy
            "",                        # pocketVolume unavailable
            str(self.surface_points),
        ]


# --------------------------------------------------------------------------- #
# CSV indexing utilities
# --------------------------------------------------------------------------- #

class VectorsCSVIndex:
    """
    Efficient reader for vectorsTrain_all_chainfix*.csv.

    Builds a per-protein byte-offset index to avoid repeated full scans.
    """

    def __init__(self, csv_path: Path, cache_dir: Optional[Path] = None):
        self.csv_path = Path(csv_path)
        if cache_dir is None:
            cache_dir = self.csv_path.parent / "tmp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / "vectors_index.pkl"

        self.header: List[str] = []
        self._index: Dict[str, Tuple[int, int]] = {}
        self._residue_idx: int = -1
        self._coord_idx: Tuple[int, int, int] = (-1, -1, -1)

        if self.cache_file.exists() and self.cache_file.stat().st_mtime > self.csv_path.stat().st_mtime:
            self._load_index()
        else:
            self._build_index()
            self._save_index()

    def _build_index(self) -> None:
        logger.info("Building CSV index for %s (this may take a few minutes)...", self.csv_path)
        index: Dict[str, Tuple[int, int]] = {}

        with self.csv_path.open("r", newline="") as fh:
            reader = csv.reader(fh)
            self.header = next(reader)
            key_idx = self.header.index("protein_key")
            self._residue_idx = self.header.index("residue_number")
            self._coord_idx = (
                self.header.index("x"),
                self.header.index("y"),
                self.header.index("z"),
            )

            current_key: Optional[str] = None
            start_pos: int = fh.tell()
            count: int = 0

            while True:
                pos_before = fh.tell()
                try:
                    row = next(reader)
                except StopIteration:
                    if current_key is not None:
                        index[current_key] = (start_pos, count)
                    break

                protein_key = row[key_idx]

                if protein_key != current_key:
                    if current_key is not None:
                        index[current_key] = (start_pos, count)
                    current_key = protein_key
                    start_pos = pos_before
                    count = 1
                else:
                    count += 1

        self._index = index
        logger.info("Indexed %d protein keys from CSV", len(self._index))

    def _save_index(self) -> None:
        payload = {
            "header": self.header,
            "index": self._index,
            "residue_idx": self._residue_idx,
            "coord_idx": self._coord_idx,
        }
        with self.cache_file.open("wb") as fh:
            pickle.dump(payload, fh)

    def _load_index(self) -> None:
        with self.cache_file.open("rb") as fh:
            payload = pickle.load(fh)
        self.header = payload["header"]
        self._index = payload["index"]
        self._residue_idx = payload["residue_idx"]
        self._coord_idx = tuple(payload["coord_idx"])
        logger.info("Loaded cached CSV index with %d protein keys", len(self._index))

    def has_protein(self, protein_key: str) -> bool:
        return protein_key in self._index

    def load_points(self, protein_key: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load SAS coordinates and residue numbers for a specific protein.

        Returns:
            coords: (N, 3) array
            residue_numbers: (N,) array
        """
        if protein_key not in self._index:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

        start, count = self._index[protein_key]
        coords: List[List[float]] = []
        residues: List[int] = []

        with self.csv_path.open("rb") as raw:
            raw.seek(start)
            text = io.TextIOWrapper(raw, newline="")
            text.seek(0, 1)  # sync after seek
            reader = csv.reader(text)

            for _ in range(count):
                try:
                    row = next(reader)
                except StopIteration:
                    break

                residues.append(int(float(row[self._residue_idx])))
                coords.append([
                    float(row[self._coord_idx[0]]),
                    float(row[self._coord_idx[1]]),
                    float(row[self._coord_idx[2]]),
                ])

        return (
            np.asarray(coords, dtype=np.float32),
            np.asarray(residues, dtype=np.int32),
        )


# --------------------------------------------------------------------------- #
# H5 helper
# --------------------------------------------------------------------------- #

class H5ProteinIndex:
    """Locate contiguous index ranges for each protein within the H5 file."""

    def __init__(self, h5_path: Path):
        self.h5_path = Path(h5_path)
        self._index: Dict[str, Tuple[int, int]] = {}
        self._build_index()

    def _build_index(self) -> None:
        mapping: Dict[str, Tuple[int, int]] = {}
        with h5py.File(self.h5_path, "r") as fh:
            keys = fh["protein_keys"]
            total = len(keys)

            current: Optional[str] = None
            start = 0

            for idx in range(total):
                raw = keys[idx]
                protein = raw.decode() if isinstance(raw, bytes) else str(raw)

                if protein != current:
                    if current is not None:
                        mapping[current] = (start, idx)
                    current = protein
                    start = idx

            if current is not None:
                mapping[current] = (start, total)

        self._index = mapping
        logger.info("Indexed %d proteins from H5", len(self._index))

    def has_protein(self, protein_id: str) -> bool:
        return protein_id in self._index

    def proteins(self) -> List[str]:
        return list(self._index.keys())

    def slice(self, protein_id: str) -> Tuple[int, int]:
        return self._index[protein_id]


class ProteinPointLoader:
    """
    Fetch aligned SAS coordinates, residue numbers, labels and predictions
    for a given protein.
    """

    def __init__(self, h5_path: Path, csv_path: Path, cache_dir: Optional[Path] = None):
        self.h5_path = Path(h5_path)
        self.h5_index = H5ProteinIndex(self.h5_path)
        self.csv_index = VectorsCSVIndex(Path(csv_path), cache_dir=cache_dir)

    def get_protein_arrays(self, protein_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            coords: (N, 3)
            residue_numbers: (N,)
            labels: (N,)
        """
        if not self.h5_index.has_protein(protein_id):
            logger.warning("Protein %s not present in H5 file", protein_id)
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int8)

        start, end = self.h5_index.slice(protein_id)
        length = end - start

        with h5py.File(self.h5_path, "r") as fh:
            residue_numbers = fh["residue_numbers"][start:end].astype(np.int32)
            labels = fh["labels"][start:end].astype(np.int8) if "labels" in fh else np.zeros(length, dtype=np.int8)

        coords, csv_residues = self.csv_index.load_points(protein_id)

        if len(coords) != length:
            logger.warning(
                "Coordinate mismatch for %s: H5=%d vs CSV=%d",
                protein_id,
                length,
                len(coords),
            )

        if len(coords) == length:
            return coords, residue_numbers, labels

        # Fallback: align by position up to min length
        limit = min(length, len(coords))
        return coords[:limit], residue_numbers[:limit], labels[:limit]


# --------------------------------------------------------------------------- #
# Core P2Rank-like processor
# --------------------------------------------------------------------------- #

class P2RankPostProcessor:
    """Python implementation of P2Rank's pocket aggregation."""

    def __init__(self, params: Optional[P2RankParams] = None):
        self.params = params or P2RankParams()

    # -- clustering ----------------------------------------------------- #

    @staticmethod
    def _single_linkage_clusters(coords: np.ndarray, threshold: float) -> List[np.ndarray]:
        """Return list of index arrays using single-linkage clustering."""
        n = len(coords)
        if n == 0:
            return []

        thr2 = threshold * threshold
        parent = np.arange(n, dtype=np.int32)
        size = np.ones(n, dtype=np.int32)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri == rj:
                return
            if size[ri] < size[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            size[ri] += size[rj]

        for i in range(n):
            diff = coords[i + 1 :] - coords[i]  # (n-i-1, 3)
            if diff.size == 0:
                continue
            dist2 = np.einsum("ij,ij->i", diff, diff)
            neighbours = np.where(dist2 <= thr2)[0]
            for offset in neighbours:
                j = i + 1 + offset
                union(i, j)

        clusters: Dict[int, List[int]] = {}
        for idx in range(n):
            root = find(idx)
            clusters.setdefault(root, []).append(idx)

        return [np.asarray(indices, dtype=np.int32) for indices in clusters.values()]

    # -- main routine --------------------------------------------------- #

    def process(self, protein_id: str, coords: np.ndarray, probs: np.ndarray, residue_numbers: np.ndarray) -> List[P2RankPocket]:
        if len(coords) == 0 or len(probs) == 0:
            return []

        if len(coords) != len(probs):
            limit = min(len(coords), len(probs))
            logger.warning("Length mismatch for %s, truncating to %d entries", protein_id, limit)
            coords = coords[:limit]
            probs = probs[:limit]
            residue_numbers = residue_numbers[:limit]

        mask = probs >= self.params.pred_point_threshold
        ligandable_idx = np.where(mask)[0]
        if len(ligandable_idx) < self.params.pred_min_cluster_size:
            return []

        ligandable_coords = coords[ligandable_idx]
        clusters = self._single_linkage_clusters(ligandable_coords, self.params.pred_clustering_dist)

        pockets: List[P2RankPocket] = []

        for cluster in clusters:
            global_cluster_idx = ligandable_idx[cluster]
            if len(global_cluster_idx) < self.params.pred_min_cluster_size:
                continue

            pocket_idx = self._grow_cluster(coords, global_cluster_idx)

            if pocket_idx.size == 0:
                continue

            pocket = self._build_pocket(protein_id, coords, probs, residue_numbers, pocket_idx, global_cluster_idx)
            pockets.append(pocket)

        pockets.sort(key=lambda p: p.score, reverse=True)
        for rank, pocket in enumerate(pockets, start=1):
            pocket.rank = rank
            pocket.name = f"pocket{rank}"

        return pockets

    # -- helpers -------------------------------------------------------- #

    def _grow_cluster(self, coords: np.ndarray, cluster_idx: np.ndarray) -> np.ndarray:
        """Extend cluster with neighbouring SAS points."""
        cluster_coords = coords[cluster_idx]
        diff = coords[:, None, :] - cluster_coords[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        mask = np.any(dist2 <= (self.params.extended_pocket_cutoff ** 2), axis=1)
        return np.where(mask)[0]

    def _balance_scores(self, coords: np.ndarray, transformed: np.ndarray) -> np.ndarray:
        """Optional density balancing."""
        if not self.params.balance_density or len(coords) == 0:
            return transformed

        diff = coords[:, None, :] - coords[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        radius2 = self.params.balance_density_radius ** 2
        neighbour_counts = np.sum(dist2 <= radius2, axis=1)
        neighbour_counts = np.maximum(neighbour_counts, 1)
        return transformed / neighbour_counts

    def _build_pocket(
        self,
        protein_id: str,
        coords: np.ndarray,
        probs: np.ndarray,
        residue_numbers: np.ndarray,
        pocket_idx: np.ndarray,
        cluster_idx: np.ndarray,
    ) -> P2RankPocket:
        pocket_coords = coords[pocket_idx]
        pocket_probs = probs[pocket_idx]

        transformed = pocket_probs ** self.params.point_score_pow
        balanced = self._balance_scores(pocket_coords, transformed)

        order = np.argsort(-balanced)
        if self.params.score_point_limit > 0:
            order = order[: self.params.score_point_limit]

        score = float(np.sum(balanced[order]))
        sum_prob = float(np.sum(pocket_probs))

        cluster_coords = coords[cluster_idx]
        center = np.mean(cluster_coords, axis=0)

        pocket = P2RankPocket(
            name="",
            rank=0,
            score=score,
            center=center.astype(np.float32),
            member_indices=pocket_idx.astype(np.int32),
            cluster_indices=cluster_idx.astype(np.int32),
            sum_prob=sum_prob,
            surface_points=len(pocket_idx),
        )
        logger.debug(
            "Protein %s pocket: members=%d score=%.3f sumP=%.3f",
            protein_id,
            len(pocket_idx),
            score,
            sum_prob,
        )
        return pocket


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #

def pockets_to_csv(protein_id: str, pockets: List[P2RankPocket]) -> str:
    """Return CSV text matching P2Rank's pockets.csv layout."""
    if not pockets:
        header = "file,#ligands,#pockets,pocket,ligand,rank,score,newRank,oldScore,zScoreTP,probaTP,samplePoints,rawNewScore,pocketVolume,surfaceAtoms"
        return header + "\n"

    lines = [
        "file,#ligands,#pockets,pocket,ligand,rank,score,newRank,oldScore,zScoreTP,probaTP,samplePoints,rawNewScore,pocketVolume,surfaceAtoms"
    ]
    pocket_count = len(pockets)
    for pocket in pockets:
        lines.append(",".join(pocket.to_csv_row(protein_id, pocket_count)))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Ground-truth pocket construction and evaluation helpers
# --------------------------------------------------------------------------- #

def create_ground_truth_pockets(
    protein_id: str,
    coords: np.ndarray,
    labels: np.ndarray,
    residue_numbers: np.ndarray,
    base_params: P2RankParams,
) -> List[P2RankPocket]:
    """
    Construct ground-truth pockets by clustering positive labels.

    Uses the same clustering radius while disabling pocket growth so that only
    labeled SAS points are included in each ground-truth pocket.
    """
    if len(coords) == 0 or len(labels) == 0:
        return []

    gt_params_dict = asdict(base_params)
    gt_params_dict.update(
        {
            "pred_point_threshold": 0.5,
            "extended_pocket_cutoff": 0.0,
            "point_score_pow": 1.0,
            "balance_density": False,
            "score_point_limit": 0,
        }
    )
    gt_processor = P2RankPostProcessor(P2RankParams(**gt_params_dict))
    gt_probs = labels.astype(float)
    return gt_processor.process(protein_id, coords, gt_probs, residue_numbers)


def _pockets_to_sets(pockets: List[P2RankPocket]) -> List[set]:
    return [set(p.member_indices.tolist()) for p in pockets]


def pocket_iou_metrics(
    predicted: List[P2RankPocket],
    ground_truth: List[P2RankPocket],
) -> Dict[str, float]:
    """
    Compute IoU-based metrics between predicted and ground-truth pockets.

    Returns:
        mean_gt_iou: Average of best IoU per ground-truth pocket.
        best_iou:    Maximum IoU achieved by any predicted/GT pair.
        gt_coverage: Fraction of GT pockets with IoU > 0.
    """
    if not ground_truth:
        return {"mean_gt_iou": 0.0, "best_iou": 0.0, "gt_coverage": 0.0}

    pred_sets = _pockets_to_sets(predicted)
    gt_sets = _pockets_to_sets(ground_truth)

    if not pred_sets:
        return {"mean_gt_iou": 0.0, "best_iou": 0.0, "gt_coverage": 0.0}

    best_per_gt: List[float] = []
    best_overall = 0.0
    coverage_hits = 0

    for gt in gt_sets:
        best = 0.0
        for pred in pred_sets:
            union = len(pred | gt)
            if union == 0:
                continue
            inter = len(pred & gt)
            iou = inter / union
            if iou > best:
                best = iou
            if iou > best_overall:
                best_overall = iou
        best_per_gt.append(best)
        if best > 0.0:
            coverage_hits += 1

    mean_gt_iou = float(np.mean(best_per_gt)) if best_per_gt else 0.0
    gt_coverage = coverage_hits / len(gt_sets)

    return {
        "mean_gt_iou": mean_gt_iou,
        "best_iou": best_overall,
        "gt_coverage": gt_coverage,
    }
