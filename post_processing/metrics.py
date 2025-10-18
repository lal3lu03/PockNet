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
import matplotlib.pyplot as plt
import seaborn as sns

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