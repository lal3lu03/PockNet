"""
Pocket-Level Evaluation Metrics
===============================

Implements comprehensive evaluation metrics for pocket-level prediction assessment,
going beyond residue-level IoU to evaluate the final use-case performance.

Key metrics:
- Pocket recall @ top-k
- Center distance metrics  
- Pocket IoU (residue-set overlap)
- Pocket-level PR curves
- Spatial overlap metrics

Author: PockNet Team
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class GroundTruthPocket:
    """Ground truth pocket definition."""
    residues: List[Tuple[str, int]]  # [(chain, res_id), ...]
    center: np.ndarray               # 3D center coordinates
    pocket_id: str                   # Unique identifier
    source: str = "unknown"          # Source annotation (fpocket, CASTp, etc.)


@dataclass 
class PredictedPocket:
    """Predicted pocket from post-processing."""
    residues: List[Tuple[str, int]]  # [(chain, res_id), ...]
    center: np.ndarray               # 3D center coordinates
    score: float                     # Prediction score
    confidence: Optional[float] = None


@dataclass
class PocketMatch:
    """Represents a match between predicted and ground truth pocket."""
    predicted_pocket: PredictedPocket
    ground_truth_pocket: GroundTruthPocket
    residue_overlap: int
    residue_iou: float
    center_distance: float
    spatial_overlap: float


class PocketEvaluator:
    """
    Comprehensive pocket-level evaluation toolkit.
    """
    
    def __init__(self, 
                 center_distance_threshold: float = 6.0,
                 residue_overlap_threshold: int = 1,
                 spatial_distance_threshold: float = 4.0):
        """
        Initialize pocket evaluator.
        
        Args:
            center_distance_threshold: Max distance for center-based matching (Å)
            residue_overlap_threshold: Min residues for overlap-based matching  
            spatial_distance_threshold: Max distance for spatial overlap (Å)
        """
        self.center_distance_threshold = center_distance_threshold
        self.residue_overlap_threshold = residue_overlap_threshold  
        self.spatial_distance_threshold = spatial_distance_threshold
        
        logger.info(f"Initialized PocketEvaluator:")
        logger.info(f"  Center distance threshold: {center_distance_threshold} Å")
        logger.info(f"  Residue overlap threshold: {residue_overlap_threshold}")
        logger.info(f"  Spatial distance threshold: {spatial_distance_threshold} Å")
    
    def evaluate_protein(self,
                        predicted_pockets: List[PredictedPocket],
                        ground_truth_pockets: List[GroundTruthPocket],
                        residue_coordinates: Optional[Dict[Tuple[str, int], np.ndarray]] = None
                        ) -> Dict[str, Any]:
        """
        Evaluate pocket predictions for a single protein.
        
        Args:
            predicted_pockets: List of predicted pockets
            ground_truth_pockets: List of ground truth pockets
            residue_coordinates: Mapping from (chain, res_id) to 3D coordinates
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            'n_predicted': len(predicted_pockets),
            'n_ground_truth': len(ground_truth_pockets),
            'matches': [],
            'unmatched_predicted': list(predicted_pockets),
            'unmatched_ground_truth': list(ground_truth_pockets)
        }
        
        if not predicted_pockets or not ground_truth_pockets:
            logger.warning("Empty pocket lists for evaluation")
            return self._compute_summary_metrics(results)
        
        # Find all possible matches
        potential_matches = []
        for pred_pocket in predicted_pockets:
            for gt_pocket in ground_truth_pockets:
                match = self._evaluate_pocket_pair(
                    pred_pocket, gt_pocket, residue_coordinates
                )
                if self._is_valid_match(match):
                    potential_matches.append(match)
        
        # Greedy matching (highest IoU first)
        potential_matches.sort(key=lambda m: m.residue_iou, reverse=True)
        
        used_predicted = set()
        used_ground_truth = set()
        
        for match in potential_matches:
            pred_id = id(match.predicted_pocket)
            gt_id = id(match.ground_truth_pocket)
            
            if pred_id not in used_predicted and gt_id not in used_ground_truth:
                results['matches'].append(match)
                used_predicted.add(pred_id)
                used_ground_truth.add(gt_id)
        
        # Update unmatched lists
        results['unmatched_predicted'] = [
            p for p in predicted_pockets if id(p) not in used_predicted
        ]
        results['unmatched_ground_truth'] = [
            p for p in ground_truth_pockets if id(p) not in used_ground_truth
        ]
        
        return self._compute_summary_metrics(results)
    
    def _evaluate_pocket_pair(self,
                             predicted: PredictedPocket,
                             ground_truth: GroundTruthPocket,
                             residue_coordinates: Optional[Dict] = None
                             ) -> PocketMatch:
        """Evaluate a single predicted-ground truth pocket pair."""
        
        # Residue overlap
        pred_residues = set(predicted.residues)
        gt_residues = set(ground_truth.residues)
        
        overlap = len(pred_residues & gt_residues)
        union = len(pred_residues | gt_residues)
        iou = overlap / (union + 1e-8)
        
        # Center distance
        center_distance = np.linalg.norm(predicted.center - ground_truth.center)
        
        # Spatial overlap (if coordinates available)
        spatial_overlap = 0.0
        if residue_coordinates is not None:
            spatial_overlap = self._compute_spatial_overlap(
                predicted, ground_truth, residue_coordinates
            )
        
        return PocketMatch(
            predicted_pocket=predicted,
            ground_truth_pocket=ground_truth,
            residue_overlap=overlap,
            residue_iou=iou,
            center_distance=center_distance,
            spatial_overlap=spatial_overlap
        )
    
    def _compute_spatial_overlap(self,
                                predicted: PredictedPocket,
                                ground_truth: GroundTruthPocket,
                                residue_coordinates: Dict) -> float:
        """Compute spatial overlap based on 3D distance."""
        
        pred_coords = []
        for chain, res_id in predicted.residues:
            if (chain, res_id) in residue_coordinates:
                pred_coords.append(residue_coordinates[(chain, res_id)])
        
        gt_coords = []
        for chain, res_id in ground_truth.residues:
            if (chain, res_id) in residue_coordinates:
                gt_coords.append(residue_coordinates[(chain, res_id)])
        
        if not pred_coords or not gt_coords:
            return 0.0
        
        pred_coords = np.array(pred_coords)
        gt_coords = np.array(gt_coords)
        
        # Count spatial overlaps
        overlapping_pred = 0
        for pred_coord in pred_coords:
            distances = np.linalg.norm(gt_coords - pred_coord, axis=1)
            if np.any(distances <= self.spatial_distance_threshold):
                overlapping_pred += 1
        
        spatial_overlap = overlapping_pred / len(pred_coords)
        return spatial_overlap
    
    def _is_valid_match(self, match: PocketMatch) -> bool:
        """Check if a pocket match meets minimum criteria."""
        return (
            match.residue_overlap >= self.residue_overlap_threshold or
            match.center_distance <= self.center_distance_threshold
        )
    
    def _compute_summary_metrics(self, results: Dict) -> Dict[str, Any]:
        """Compute summary metrics from match results."""
        n_matches = len(results['matches'])
        n_predicted = results['n_predicted']
        n_ground_truth = results['n_ground_truth']
        
        # Basic counts
        precision = n_matches / (n_predicted + 1e-8)
        recall = n_matches / (n_ground_truth + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Match quality metrics
        if n_matches > 0:
            avg_iou = np.mean([m.residue_iou for m in results['matches']])
            avg_center_distance = np.mean([m.center_distance for m in results['matches']])
            avg_spatial_overlap = np.mean([m.spatial_overlap for m in results['matches']])
        else:
            avg_iou = 0.0
            avg_center_distance = float('inf')
            avg_spatial_overlap = 0.0
        
        summary = {
            'n_predicted': n_predicted,
            'n_ground_truth': n_ground_truth,
            'n_matches': n_matches,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_iou': avg_iou,
            'avg_center_distance': avg_center_distance,
            'avg_spatial_overlap': avg_spatial_overlap,
            'matches': results['matches'],
            'unmatched_predicted': results['unmatched_predicted'],
            'unmatched_ground_truth': results['unmatched_ground_truth']
        }
        
        return summary


def pocket_recall_at_k(predictions: List[List[PredictedPocket]],
                      ground_truths: List[List[GroundTruthPocket]],
                      k_values: List[int] = [1, 3, 5],
                      evaluator: Optional[PocketEvaluator] = None) -> Dict[int, float]:
    """
    Compute pocket recall @ top-k across multiple proteins.
    
    Args:
        predictions: List of predicted pocket lists (one per protein)
        ground_truths: List of ground truth pocket lists (one per protein)
        k_values: List of k values to evaluate
        evaluator: PocketEvaluator instance (if None, create default)
        
    Returns:
        Dictionary mapping k -> recall@k
    """
    if evaluator is None:
        evaluator = PocketEvaluator()
    
    if len(predictions) != len(ground_truths):
        raise ValueError("Mismatched number of proteins")
    
    recalls = {k: [] for k in k_values}
    
    for protein_preds, protein_gts in zip(predictions, ground_truths):
        if not protein_gts:  # Skip proteins with no ground truth pockets
            continue
        
        # Evaluate all predictions for this protein
        results = evaluator.evaluate_protein(protein_preds, protein_gts)
        
        # For each k, check if any of top-k predictions match GT
        for k in k_values:
            top_k_preds = protein_preds[:k]
            
            # Check if any top-k prediction has a match
            has_match = False
            for pred in top_k_preds:
                # Check if this prediction matched any GT
                for match in results['matches']:
                    if match.predicted_pocket is pred:
                        has_match = True
                        break
                if has_match:
                    break
            
            recalls[k].append(1.0 if has_match else 0.0)
    
    # Average across proteins
    recall_at_k = {}
    for k in k_values:
        if recalls[k]:
            recall_at_k[k] = np.mean(recalls[k])
        else:
            recall_at_k[k] = 0.0
    
    logger.info("Pocket Recall @ K:")
    for k, recall in recall_at_k.items():
        logger.info(f"  Recall@{k}: {recall:.3f}")
    
    return recall_at_k


def center_distance_metrics(predictions: List[List[PredictedPocket]],
                           ground_truths: List[List[GroundTruthPocket]],
                           evaluator: Optional[PocketEvaluator] = None) -> Dict[str, float]:
    """
    Compute center distance metrics across multiple proteins.
    
    Args:
        predictions: List of predicted pocket lists
        ground_truths: List of ground truth pocket lists  
        evaluator: PocketEvaluator instance
        
    Returns:
        Dictionary of distance metrics
    """
    if evaluator is None:
        evaluator = PocketEvaluator()
    
    all_distances = []
    min_distances = []  # Min distance from each GT to closest prediction
    
    for protein_preds, protein_gts in zip(predictions, ground_truths):
        if not protein_preds or not protein_gts:
            continue
        
        results = evaluator.evaluate_protein(protein_preds, protein_gts)
        
        # Collect matched distances
        for match in results['matches']:
            all_distances.append(match.center_distance)
        
        # Compute min distance for each GT pocket
        for gt_pocket in protein_gts:
            min_dist = float('inf')
            for pred_pocket in protein_preds:
                dist = np.linalg.norm(pred_pocket.center - gt_pocket.center)
                min_dist = min(min_dist, dist)
            min_distances.append(min_dist)
    
    if not all_distances:
        return {
            'mean_matched_distance': float('inf'),
            'median_matched_distance': float('inf'),
            'mean_min_distance': float('inf'),
            'median_min_distance': float('inf')
        }
    
    metrics = {
        'mean_matched_distance': np.mean(all_distances),
        'median_matched_distance': np.median(all_distances),
        'mean_min_distance': np.mean(min_distances),
        'median_min_distance': np.median(min_distances)
    }
    
    logger.info("Center Distance Metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.2f} Å")
    
    return metrics


def pocket_iou_metrics(predictions: List[List[PredictedPocket]],
                      ground_truths: List[List[GroundTruthPocket]],
                      evaluator: Optional[PocketEvaluator] = None) -> Dict[str, float]:
    """
    Compute pocket IoU metrics across multiple proteins.
    
    Args:
        predictions: List of predicted pocket lists
        ground_truths: List of ground truth pocket lists
        evaluator: PocketEvaluator instance
        
    Returns:
        Dictionary of IoU metrics
    """
    if evaluator is None:
        evaluator = PocketEvaluator()
    
    all_ious = []
    
    for protein_preds, protein_gts in zip(predictions, ground_truths):
        if not protein_preds or not protein_gts:
            continue
        
        results = evaluator.evaluate_protein(protein_preds, protein_gts)
        
        # Collect matched IoUs
        for match in results['matches']:
            all_ious.append(match.residue_iou)
    
    if not all_ious:
        return {
            'mean_iou': 0.0,
            'median_iou': 0.0,
            'iou_std': 0.0
        }
    
    metrics = {
        'mean_iou': np.mean(all_ious),
        'median_iou': np.median(all_ious),
        'iou_std': np.std(all_ious)
    }
    
    logger.info("Pocket IoU Metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.3f}")
    
    return metrics


def pocket_pr_curve(predictions: List[List[PredictedPocket]],
                   ground_truths: List[List[GroundTruthPocket]],
                   evaluator: Optional[PocketEvaluator] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pocket-level precision-recall curve.
    
    Args:
        predictions: List of predicted pocket lists
        ground_truths: List of ground truth pocket lists
        evaluator: PocketEvaluator instance
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    if evaluator is None:
        evaluator = PocketEvaluator()
    
    # Collect all pockets with their scores and match status
    all_pockets = []
    
    for protein_preds, protein_gts in zip(predictions, ground_truths):
        if not protein_gts:  # Skip proteins with no GT
            continue
        
        results = evaluator.evaluate_protein(protein_preds, protein_gts)
        
        # Mark which predictions are matches
        matched_preds = {match.predicted_pocket for match in results['matches']}
        
        for pred in protein_preds:
            is_match = pred in matched_preds
            all_pockets.append((pred.score, is_match))
    
    if not all_pockets:
        return np.array([1, 0]), np.array([0, 0]), np.array([0])
    
    # Sort by score (descending)
    all_pockets.sort(key=lambda x: x[0], reverse=True)
    
    scores = np.array([x[0] for x in all_pockets])
    is_positive = np.array([x[1] for x in all_pockets])
    
    # Compute precision and recall at each threshold
    precisions = []
    recalls = []
    
    total_positives = is_positive.sum()
    
    for i in range(len(all_pockets)):
        # Consider top i+1 predictions
        tp = is_positive[:i+1].sum()
        fp = (i + 1) - tp
        
        precision = tp / (tp + fp)
        recall = tp / total_positives if total_positives > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls), scores


def plot_pocket_metrics(metrics_dict: Dict[str, Any],
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization of pocket-level metrics.
    
    Args:
        metrics_dict: Dictionary containing various pocket metrics
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Pocket-Level Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # Recall @ K plot
    if 'recall_at_k' in metrics_dict:
        ax = axes[0, 0]
        recall_data = metrics_dict['recall_at_k']
        k_values = list(recall_data.keys())
        recalls = list(recall_data.values())
        
        ax.bar(k_values, recalls, alpha=0.7, color='skyblue', edgecolor='navy')
        ax.set_xlabel('K (Top-K Predictions)')
        ax.set_ylabel('Recall @ K')
        ax.set_title('Pocket Recall @ Top-K')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for k, recall in zip(k_values, recalls):
            ax.text(k, recall + 0.02, f'{recall:.3f}', ha='center', va='bottom')
    
    # Distance metrics plot
    if 'center_distances' in metrics_dict:
        ax = axes[0, 1]
        dist_data = metrics_dict['center_distances']
        
        metrics = ['mean_matched_distance', 'median_matched_distance', 
                  'mean_min_distance', 'median_min_distance']
        values = [dist_data.get(m, 0) for m in metrics]
        labels = [m.replace('_', ' ').title() for m in metrics]
        
        bars = ax.bar(range(len(metrics)), values, alpha=0.7, 
                     color=['coral', 'lightcoral', 'gold', 'khaki'])
        ax.set_xlabel('Distance Metric')
        ax.set_ylabel('Distance (Å)')
        ax.set_title('Center Distance Metrics')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, values):
            if value != float('inf'):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{value:.1f}', ha='center', va='bottom')
    
    # IoU distribution
    if 'pocket_ious' in metrics_dict:
        ax = axes[1, 0]
        iou_data = metrics_dict['pocket_ious']
        
        metrics = ['mean_iou', 'median_iou']
        values = [iou_data.get(m, 0) for m in metrics]
        
        ax.bar(metrics, values, alpha=0.7, color=['lightgreen', 'mediumseagreen'])
        ax.set_ylabel('IoU Score')
        ax.set_title('Pocket IoU Metrics')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for i, (metric, value) in enumerate(zip(metrics, values)):
            ax.text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom')
    
    # PR curve
    if 'pr_curve' in metrics_dict:
        ax = axes[1, 1]
        precision, recall, _ = metrics_dict['pr_curve']
        
        ax.plot(recall, precision, 'b-', linewidth=2, label='Pocket PR Curve')
        ax.fill_between(recall, precision, alpha=0.2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Pocket-Level PR Curve')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Compute and display AUPRC
        auprc = np.trapz(precision, recall)
        ax.text(0.05, 0.95, f'AUPRC: {auprc:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved pocket metrics plot to {save_path}")
    
    return fig


def evaluate_dbscan_success_metrics(
    protein_cache: Dict[str, Dict[str, object]],
    threshold: float,
    eps: float,
    min_samples: int,
    dcc_threshold: float,
    dca_threshold: float,
    topk: Tuple[int, ...] = (1, 2, 3),
) -> Dict[str, Any]:
    """
    Cluster residue-level predictions with DBSCAN and evaluate DCC/DCA success.

    Args:
        protein_cache: Mapping protein_id -> cached arrays (coords, probs, gt_pockets, ...)
        threshold: Probability cutoff for selecting residues prior to clustering.
        eps: DBSCAN epsilon (Å).
        min_samples: Minimum samples for DBSCAN clusters.
        dcc_threshold: Success distance (Å) for distance-to-centre (DCC).
        dca_threshold: Success distance (Å) for distance-of-closest-approach (DCA).
        topk: Tuple of K values for top-K success evaluation.

    Returns:
        Dict containing per-protein outcomes and dataset aggregates.
    """
    topk = tuple(sorted(set(topk)))
    per_protein: Dict[str, Dict[str, Any]] = {}
    total_gt = 0
    total_clusters = 0
    dcc_hits = {k: 0 for k in topk}
    dca_hits = {k: 0 for k in topk}
    dcc_distances: List[float] = []
    dca_distances: List[float] = []

    for protein_id, pdata in protein_cache.items():
        coords = np.asarray(pdata.get("coords", np.empty((0, 3))), dtype=np.float32)
        probs = np.asarray(pdata.get("probs", np.empty((0,))), dtype=np.float32)
        gt_pockets = pdata.get("gt_pockets", [])

        if coords.ndim != 2 or coords.shape[1] != 3:
            logger.warning("Skipping DBSCAN metrics for %s: invalid coords shape %s", protein_id, coords.shape)
            continue

        mask = probs >= threshold
        selected_coords = coords[mask]
        selected_probs = probs[mask]

        clusters: List[Dict[str, Any]] = []
        if len(selected_coords) >= max(min_samples, 1):
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(selected_coords)
            for label in np.unique(labels):
                if label < 0:
                    continue
                cluster_idx = np.where(labels == label)[0]
                cluster_points = selected_coords[cluster_idx]
                cluster_probs = selected_probs[cluster_idx]
                clusters.append(
                    {
                        "size": int(cluster_points.shape[0]),
                        "center": cluster_points.mean(axis=0),
                        "coords": cluster_points,
                        "score": float(cluster_probs.sum()),
                        "mean_prob": float(cluster_probs.mean()),
                        "max_prob": float(cluster_probs.max()),
                    }
                )

        clusters.sort(key=lambda c: c["score"], reverse=True)
        total_clusters += len(clusters)

        gt_centers = []
        gt_members = []
        if gt_pockets:
            for pocket in gt_pockets:  # pockets are P2RankPocket objects
                center = np.asarray(getattr(pocket, "center", None))
                members = getattr(pocket, "member_indices", None)
                if center is None or center.shape != (3,):
                    continue
                gt_centers.append(center.astype(np.float32, copy=False))
                if members is not None and len(members) > 0:
                    gt_members.append(coords[np.asarray(members, dtype=np.int64)])
                else:
                    gt_members.append(np.empty((0, 3), dtype=np.float32))

        gt_centers_arr = np.asarray(gt_centers, dtype=np.float32)
        gt_count = len(gt_centers_arr)
        total_gt += gt_count

        dcc_min = np.array([], dtype=np.float32)
        per_gt_dca: List[float] = []

        if len(clusters) > 0 and gt_count > 0:
            pred_centers_arr = np.asarray([c["center"] for c in clusters], dtype=np.float32)
            # DCC distances (best cluster among all predictions)
            dcc_matrix = np.linalg.norm(
                pred_centers_arr[:, None, :] - gt_centers_arr[None, :, :],
                axis=2,
            )
            dcc_min = dcc_matrix.min(axis=0)
            dcc_distances.extend(dcc_min.tolist())

            # DCA distances
            for gt_pts in gt_members:
                if gt_pts.size == 0:
                    per_gt_dca.append(float("inf"))
                    continue
                best = float("inf")
                for cluster in clusters:
                    pred_pts = cluster["coords"]
                    if pred_pts.size == 0:
                        continue
                    dists = np.linalg.norm(pred_pts[:, None, :] - gt_pts[None, :, :], axis=2)
                    best = min(best, float(dists.min()))
                    if best <= dca_threshold:
                        break
                per_gt_dca.append(best)
            dca_distances.extend(per_gt_dca)

            # Success rates per top-k
            for k in topk:
                top_pred_centers = pred_centers_arr[:k] if k <= len(pred_centers_arr) else pred_centers_arr
                if top_pred_centers.size == 0:
                    continue
                top_dcc = np.linalg.norm(
                    top_pred_centers[:, None, :] - gt_centers_arr[None, :, :],
                    axis=2,
                )
                hits = (top_dcc.min(axis=0) <= dcc_threshold).sum()
                dcc_hits[k] += int(hits)

                # DCA success
                hit_count = 0
                for gt_idx, gt_pts in enumerate(gt_members):
                    if gt_pts.size == 0:
                        continue
                    success = False
                    for cluster in clusters[:k]:
                        pred_pts = cluster["coords"]
                        if pred_pts.size == 0:
                            continue
                        dists = np.linalg.norm(pred_pts[:, None, :] - gt_pts[None, :, :], axis=2)
                        if float(dists.min()) <= dca_threshold:
                            success = True
                            break
                    if success:
                        hit_count += 1
                dca_hits[k] += hit_count

        per_protein[protein_id] = {
            "n_selected": int(mask.sum()),
            "n_clusters": len(clusters),
            "n_gt_pockets": gt_count,
            "dcc_distances": dcc_min.tolist() if gt_count > 0 and len(clusters) > 0 else [],
            "dca_distances": per_gt_dca if gt_count > 0 and len(clusters) > 0 else [],
        }

    aggregate: Dict[str, Any] = {
        "config": {
            "threshold": threshold,
            "eps": eps,
            "min_samples": min_samples,
            "dcc_threshold": dcc_threshold,
            "dca_threshold": dca_threshold,
            "topk": topk,
        },
        "total_gt_pockets": total_gt,
        "total_clusters": total_clusters,
        "dcc_mean_distance": float(np.mean(dcc_distances)) if dcc_distances else None,
        "dca_mean_distance": float(np.mean(dca_distances)) if dca_distances else None,
    }

    if total_gt > 0:
        for k in topk:
            aggregate[f"dcc_success@{k}"] = dcc_hits[k] / total_gt
            aggregate[f"dca_success@{k}"] = dca_hits[k] / total_gt
    else:
        for k in topk:
            aggregate[f"dcc_success@{k}"] = 0.0
            aggregate[f"dca_success@{k}"] = 0.0

    aggregate["dcc_success_counts"] = dcc_hits
    aggregate["dca_success_counts"] = dca_hits
    aggregate["dcc_distance_samples"] = dcc_distances
    aggregate["dca_distance_samples"] = dca_distances

    return {
        "aggregate": aggregate,
        "per_protein": per_protein,
    }


def plot_dbscan_success_rates(
    aggregate: Dict[str, Any],
    save_path: Path,
) -> Optional[plt.Figure]:
    """
    Plot DCC/DCA top-k success rates from aggregate metrics.
    """
    topk = aggregate.get("config", {}).get("topk", (1, 2, 3))
    if isinstance(topk, list):
        topk = tuple(topk)

    dcc_rates = [aggregate.get(f"dcc_success@{k}", 0.0) for k in topk]
    dca_rates = [aggregate.get(f"dca_success@{k}", 0.0) for k in topk]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(topk, dcc_rates, marker="o", label="DCC success")
    ax.plot(topk, dca_rates, marker="s", label="DCA success")
    ax.set_xlabel("Top-K predicted pockets")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("DBSCAN-based pocket success rates")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved DBSCAN success plot to %s", save_path)
    return fig


# Example usage and testing
if __name__ == "__main__":
    # Create some test data
    import random
    
    # Generate test ground truth pockets
    gt_pockets = [
        GroundTruthPocket(
            residues=[('A', i) for i in range(10, 20)],
            center=np.array([0, 0, 0]),
            pocket_id="GT1"
        ),
        GroundTruthPocket(
            residues=[('A', i) for i in range(30, 40)],
            center=np.array([10, 0, 0]),
            pocket_id="GT2"
        )
    ]
    
    # Generate test predictions
    pred_pockets = [
        PredictedPocket(
            residues=[('A', i) for i in range(12, 22)],  # Overlaps with GT1
            center=np.array([1, 1, 1]),
            score=0.8
        ),
        PredictedPocket(
            residues=[('A', i) for i in range(50, 60)],  # No overlap
            center=np.array([20, 0, 0]),
            score=0.6
        )
    ]
    
    # Test evaluation
    evaluator = PocketEvaluator()
    results = evaluator.evaluate_protein(pred_pockets, gt_pockets)
    
    print(f"Evaluation Results:")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall: {results['recall']:.3f}")
    print(f"  F1: {results['f1']:.3f}")
    print(f"  Matches: {results['n_matches']}")
    print(f"  Avg IoU: {results['avg_iou']:.3f}")
    
    print("Pocket metrics module test completed!")
