#!/usr/bin/env python3
"""
PockNet Post-Processing Pipeline
===============================

Complete post-processing pipeline for converting residue-level predictions
to pocket predictions with IoU evaluation and optimization.

Implements the full recipe:
1. Mean ensemble across seeds
2. Surface filtering and spatial clustering  
3. Pocket formation and scoring
4. IoU and pocket-level metrics
5. Adaptive thresholding and optimization

Author: PockNet Team
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import defaultdict
import time

# Scientific computing
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PostProcessingConfig:
    """Configuration for post-processing pipeline"""
    # Surface filtering
    rsa_min: float = 0.2
    surface_percentile: float = 60.0  # fallback if no absolute RSA
    
    # Graph construction
    graph_radius: float = 8.0
    knn_fallback: int = 16
    
    # Clustering
    min_cluster_size: int = 5
    sump_min: float = 2.0
    
    # Thresholding
    threshold_mode: str = "adaptive"  # "global", "adaptive", "percentile"
    global_threshold: Optional[float] = None
    target_prevalence: float = 0.03  # 3% for adaptive
    percentile_threshold: float = 95.0
    
    # Smoothing
    enable_smoothing: bool = True
    smoothing_alpha: float = 0.7
    
    # Pocket filtering
    max_pockets_per_protein: int = 5
    min_pocket_score: float = 0.1
    
    # NMS
    enable_nms: bool = True
    nms_radius: float = 6.0


@dataclass 
class Residue:
    """Single residue with prediction data"""
    chain: str
    res_id: int
    xyz: np.ndarray  # (3,) Cα/Cβ coordinates
    rsa: float  # Relative solvent accessibility
    prob: float  # Model prediction probability
    
    def __post_init__(self):
        if isinstance(self.xyz, (list, tuple)):
            self.xyz = np.array(self.xyz, dtype=np.float32)


@dataclass
class Pocket:
    """Predicted pocket with metadata"""
    members: List[Tuple[str, int]]  # [(chain, res_id), ...]
    size: int
    sump: float  # Sum of probabilities
    surface_fraction: float
    score: float
    center: np.ndarray  # (3,) weighted centroid
    
    def __post_init__(self):
        if isinstance(self.center, (list, tuple)):
            self.center = np.array(self.center, dtype=np.float32)


class GraphBuilder:
    """Builds spatial graphs from residue coordinates"""
    
    @staticmethod
    def make_graph(coords: np.ndarray, 
                   radius: float = 8.0, 
                   knn: int = 16) -> csr_matrix:
        """
        Build spatial graph connecting residues within radius.
        
        Args:
            coords: (N, 3) array of coordinates
            radius: Maximum distance for connections (Å)
            knn: Fallback k-nearest neighbors
            
        Returns:
            Sparse adjacency matrix
        """
        n = len(coords)
        if n == 0:
            return csr_matrix((0, 0))
            
        tree = KDTree(coords)
        
        # Query k-nearest neighbors
        k = min(knn, n)
        distances, indices = tree.query(coords, k=k)
        
        rows, cols = [], []
        for i, (dists, neighs) in enumerate(zip(distances, indices)):
            for d, j in zip(dists, neighs):
                if i != j and d <= radius:
                    rows.append(i)
                    cols.append(j)
        
        # Create symmetric adjacency matrix
        data = np.ones(len(rows), dtype=np.uint8)
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        graph = graph.maximum(graph.transpose())
        
        return graph


class SpatialSmoother:
    """Applies spatial smoothing to predictions"""
    
    @staticmethod
    def smooth_predictions(predictions: np.ndarray,
                          graph: csr_matrix,
                          alpha: float = 0.7) -> np.ndarray:
        """
        Apply spatial smoothing: p' = α*p + (1-α)*mean(neighbors)
        
        Args:
            predictions: (N,) array of prediction probabilities
            graph: Sparse adjacency matrix
            alpha: Smoothing parameter (0=full smoothing, 1=no smoothing)
            
        Returns:
            Smoothed predictions
        """
        if len(predictions) == 0:
            return predictions
            
        # Calculate neighbor means
        degree = np.array(graph.sum(axis=1)).flatten()
        neighbor_sums = graph.dot(predictions)
        
        # Avoid division by zero
        neighbor_means = np.divide(neighbor_sums, degree, 
                                 out=predictions.copy(), 
                                 where=degree > 0)
        
        # Apply smoothing
        smoothed = alpha * predictions + (1 - alpha) * neighbor_means
        
        return smoothed


class ThresholdSelector:
    """Selects optimal thresholds for pocket formation"""
    
    @staticmethod
    def global_threshold(predictions: np.ndarray,
                        threshold: float) -> float:
        """Use fixed global threshold"""
        return threshold
    
    @staticmethod
    def adaptive_threshold(predictions: np.ndarray,
                          target_prevalence: float = 0.03) -> float:
        """Select threshold to achieve target positive rate"""
        if len(predictions) == 0:
            return 0.5
            
        sorted_preds = np.sort(predictions)[::-1]  # Descending
        n_target = max(1, int(len(predictions) * target_prevalence))
        
        if n_target >= len(sorted_preds):
            return sorted_preds[-1]
        
        return float(sorted_preds[n_target - 1])
    
    @staticmethod
    def percentile_threshold(predictions: np.ndarray,
                           percentile: float = 95.0) -> float:
        """Use percentile-based threshold"""
        if len(predictions) == 0:
            return 0.5
        return float(np.percentile(predictions, percentile))


class PocketFormer:
    """Forms pockets from thresholded residue predictions"""
    
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        
    def form_pockets(self, 
                    residues: List[Residue],
                    threshold: float) -> List[Pocket]:
        """
        Form pockets from residue predictions using spatial clustering.
        
        Args:
            residues: List of residues with predictions
            threshold: Probability threshold for positive residues
            
        Returns:
            List of predicted pockets
        """
        if not residues:
            return []
            
        # Convert to arrays
        coords = np.stack([r.xyz for r in residues])
        rsa_values = np.array([r.rsa for r in residues])
        predictions = np.array([r.prob for r in residues])
        
        # Apply surface filter
        surface_mask = rsa_values >= self.config.rsa_min
        
        # Apply threshold
        positive_mask = predictions >= threshold
        
        # Combined mask
        keep_mask = surface_mask & positive_mask
        keep_indices = np.where(keep_mask)[0]
        
        if len(keep_indices) == 0:
            return []
        
        # Build graph on all residues
        graph = GraphBuilder.make_graph(coords, 
                                      radius=self.config.graph_radius,
                                      knn=self.config.knn_fallback)
        
        # Extract subgraph of positive residues
        subgraph = graph[keep_indices][:, keep_indices]
        
        # Find connected components
        n_components, labels = connected_components(subgraph, directed=False)
        
        pockets = []
        for comp_id in range(n_components):
            # Get component members (in subgraph indices)
            comp_local = np.where(labels == comp_id)[0]
            # Map back to original indices
            comp_global = keep_indices[comp_local]
            
            if len(comp_global) < self.config.min_cluster_size:
                continue
                
            # Calculate pocket properties
            comp_probs = predictions[comp_global]
            comp_rsa = rsa_values[comp_global] 
            comp_coords = coords[comp_global]
            
            sump = float(comp_probs.sum())
            if sump < self.config.sump_min:
                continue
            
            # Calculate weighted center
            weights = comp_probs / (sump + 1e-8)
            center = (weights[:, None] * comp_coords).sum(axis=0)
            
            # Surface fraction
            surface_fraction = float((comp_rsa >= self.config.rsa_min).mean())
            
            # Pocket score
            score = sump * surface_fraction
            
            if score < self.config.min_pocket_score:
                continue
                
            # Create pocket
            members = [(residues[i].chain, residues[i].res_id) for i in comp_global]
            
            pocket = Pocket(
                members=members,
                size=len(comp_global),
                sump=sump,
                surface_fraction=surface_fraction,
                score=score,
                center=center
            )
            
            pockets.append(pocket)
        
        # Sort by score
        pockets.sort(key=lambda p: p.score, reverse=True)
        
        # Apply NMS if enabled
        if self.config.enable_nms:
            pockets = self._apply_nms(pockets)
        
        # Limit number of pockets
        return pockets[:self.config.max_pockets_per_protein]
    
    def _apply_nms(self, pockets: List[Pocket]) -> List[Pocket]:
        """Apply non-maximum suppression to remove nearby pockets"""
        if len(pockets) <= 1:
            return pockets
            
        kept = []
        centers = np.stack([p.center for p in pockets])
        
        for i, pocket in enumerate(pockets):
            # Check if too close to any already kept pocket
            suppress = False
            for kept_pocket in kept:
                dist = np.linalg.norm(pocket.center - kept_pocket.center)
                if dist <= self.config.nms_radius:
                    suppress = True
                    break
            
            if not suppress:
                kept.append(pocket)
                
        return kept


class IoUCalculator:
    """Calculates Intersection over Union metrics"""
    
    @staticmethod
    def residue_iou(pred_residues: List[Tuple[str, int]],
                   gt_residues: List[Tuple[str, int]]) -> float:
        """Calculate IoU between predicted and ground truth residue sets"""
        pred_set = set(pred_residues)
        gt_set = set(gt_residues)
        
        intersection = len(pred_set & gt_set)
        union = len(pred_set | gt_set)
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    @staticmethod
    def pocket_iou(pred_pockets: List[Pocket],
                  gt_pockets: List[List[Tuple[str, int]]],
                  distance_threshold: float = 6.0) -> Dict[str, float]:
        """
        Calculate pocket-level IoU metrics.
        
        Args:
            pred_pockets: Predicted pockets
            gt_pockets: Ground truth pockets (list of residue lists)
            distance_threshold: Max distance for pocket matching (Å)
            
        Returns:
            Dictionary with IoU metrics
        """
        if not pred_pockets or not gt_pockets:
            return {
                'mean_iou': 0.0,
                'max_iou': 0.0,
                'pocket_recall': 0.0,
                'pocket_precision': 0.0
            }
        
        # Calculate IoU matrix
        ious = np.zeros((len(pred_pockets), len(gt_pockets)))
        
        for i, pred_pocket in enumerate(pred_pockets):
            for j, gt_pocket in enumerate(gt_pockets):
                ious[i, j] = IoUCalculator.residue_iou(pred_pocket.members, gt_pocket)
        
        # Best matching
        best_pred_ious = ious.max(axis=1) if ious.size > 0 else np.array([])
        best_gt_ious = ious.max(axis=0) if ious.size > 0 else np.array([])
        
        # Metrics
        mean_iou = best_pred_ious.mean() if len(best_pred_ious) > 0 else 0.0
        max_iou = best_pred_ious.max() if len(best_pred_ious) > 0 else 0.0
        
        # Pocket recall: fraction of GT pockets with IoU > 0
        pocket_recall = (best_gt_ious > 0).mean() if len(best_gt_ious) > 0 else 0.0
        
        # Pocket precision: fraction of pred pockets with IoU > 0  
        pocket_precision = (best_pred_ious > 0).mean() if len(best_pred_ious) > 0 else 0.0
        
        return {
            'mean_iou': float(mean_iou),
            'max_iou': float(max_iou),
            'pocket_recall': float(pocket_recall),
            'pocket_precision': float(pocket_precision),
            'n_pred_pockets': len(pred_pockets),
            'n_gt_pockets': len(gt_pockets)
        }


class EnsembleProcessor:
    """Handles multi-seed ensemble predictions"""
    
    @staticmethod
    def mean_ensemble(predictions_list: List[np.ndarray]) -> np.ndarray:
        """Average predictions across multiple seeds"""
        if not predictions_list:
            return np.array([])
            
        if len(predictions_list) == 1:
            return predictions_list[0]
            
        # Ensure all have same length
        min_len = min(len(p) for p in predictions_list)
        truncated = [p[:min_len] for p in predictions_list]
        
        return np.mean(truncated, axis=0)
    
    @staticmethod
    def ensemble_residue_data(residue_lists: List[List[Residue]]) -> List[Residue]:
        """
        Create ensemble residue data with averaged predictions.
        
        Args:
            residue_lists: List of residue lists from different seeds
            
        Returns:
            Single residue list with ensembled predictions
        """
        if not residue_lists:
            return []
            
        if len(residue_lists) == 1:
            return residue_lists[0]
        
        # Use first seed as template (assume same residues across seeds)
        template = residue_lists[0]
        
        ensembled = []
        for i, template_res in enumerate(template):
            # Collect predictions from all seeds for this residue
            seed_probs = []
            for seed_residues in residue_lists:
                if i < len(seed_residues):
                    seed_probs.append(seed_residues[i].prob)
            
            if seed_probs:
                # Create new residue with ensemble prediction
                ensemble_prob = np.mean(seed_probs)
                
                ensemble_res = Residue(
                    chain=template_res.chain,
                    res_id=template_res.res_id,
                    xyz=template_res.xyz.copy(),
                    rsa=template_res.rsa,
                    prob=ensemble_prob
                )
                
                ensembled.append(ensemble_res)
        
        return ensembled