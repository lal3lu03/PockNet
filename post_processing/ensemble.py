"""
Ensemble and Smoothing Utilities
================================

Implements multi-seed ensembling, spatial smoothing, and advanced thresholding
strategies for improved pocket prediction performance.

Key features:
- Mean ensemble across multiple seeds
- Spatial smoothing on residue graphs
- Adaptive thresholding strategies
- Probability calibration techniques

Author: PockNet Team
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

def ensemble_predictions(seed_predictions: List[np.ndarray], 
                        method: str = "mean",
                        weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Ensemble predictions from multiple seeds.
    
    Args:
        seed_predictions: List of prediction arrays, one per seed
        method: Ensembling method ("mean", "weighted_mean", "median")
        weights: Optional weights for weighted_mean (should sum to 1)
        
    Returns:
        Ensembled predictions array
    """
    if not seed_predictions:
        raise ValueError("Empty prediction list")
    
    # Stack predictions
    predictions = np.stack(seed_predictions, axis=0)  # (n_seeds, n_residues)
    
    logger.info(f"Ensembling {len(seed_predictions)} seed predictions "
               f"with method '{method}'")
    
    if method == "mean":
        result = np.mean(predictions, axis=0)
    elif method == "weighted_mean":
        if weights is None:
            raise ValueError("Weights required for weighted_mean")
        if len(weights) != len(seed_predictions):
            raise ValueError("Number of weights must match number of predictions")
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        result = np.average(predictions, axis=0, weights=weights)
    elif method == "median":
        result = np.median(predictions, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    logger.info(f"Ensemble range: [{result.min():.4f}, {result.max():.4f}]")
    return result


def ensemble_logits(seed_logits: List[np.ndarray],
                   temperature: float = 1.0) -> np.ndarray:
    """
    Ensemble raw logits (before sigmoid) from multiple seeds.
    This is generally better than ensembling probabilities.
    
    Args:
        seed_logits: List of logit arrays, one per seed
        temperature: Temperature scaling parameter
        
    Returns:
        Ensembled probabilities (after sigmoid)
    """
    if not seed_logits:
        raise ValueError("Empty logits list")
    
    # Average logits
    mean_logits = np.mean(seed_logits, axis=0)
    
    # Apply temperature scaling and sigmoid
    scaled_logits = mean_logits / temperature
    probabilities = 1.0 / (1.0 + np.exp(-scaled_logits))
    
    logger.info(f"Ensembled {len(seed_logits)} sets of logits with T={temperature}")
    return probabilities


def spatial_smoothing(probabilities: np.ndarray,
                     graph: csr_matrix,
                     alpha: float = 0.7,
                     iterations: int = 1) -> np.ndarray:
    """
    Apply spatial smoothing on residue graph to reduce salt-and-pepper noise.
    
    Formula: p' = α·p + (1-α)·mean(p_neighbors)
    
    Args:
        probabilities: Original prediction probabilities
        graph: Residue connectivity graph (sparse matrix)
        alpha: Weight for original prediction (0.7 = 70% original, 30% neighbors)
        iterations: Number of smoothing iterations
        
    Returns:
        Smoothed probabilities
    """
    smoothed = probabilities.copy()
    
    logger.info(f"Applying spatial smoothing: α={alpha}, iterations={iterations}")
    
    for iter_idx in range(iterations):
        # Compute neighbor means
        neighbor_sums = graph.dot(smoothed)
        neighbor_counts = np.array(graph.sum(axis=1)).flatten()
        
        # Avoid division by zero
        neighbor_counts = np.maximum(neighbor_counts, 1)
        neighbor_means = neighbor_sums / neighbor_counts
        
        # Apply smoothing
        smoothed = alpha * smoothed + (1 - alpha) * neighbor_means
        
        logger.debug(f"Iteration {iter_idx + 1}: "
                    f"range [{smoothed.min():.4f}, {smoothed.max():.4f}]")
    
    return smoothed


def adaptive_threshold(probabilities: np.ndarray,
                      strategy: str = "percentile95",
                      target_prevalence: float = 0.03,
                      validation_labels: Optional[np.ndarray] = None) -> float:
    """
    Compute adaptive threshold based on different strategies.
    
    Args:
        probabilities: Prediction probabilities
        strategy: Thresholding strategy
            - "percentile95": 95th percentile
            - "prevalence": Target prevalence-based
            - "global_opt": Validation-optimal (needs validation_labels)
        target_prevalence: Target positive rate for prevalence strategy
        validation_labels: Ground truth labels for global_opt strategy
        
    Returns:
        Computed threshold
    """
    if strategy == "percentile95":
        threshold = float(np.percentile(probabilities, 95.0))
        
    elif strategy == "prevalence":
        # Choose threshold to achieve target prevalence
        percentile = (1 - target_prevalence) * 100
        threshold = float(np.percentile(probabilities, percentile))
        
    elif strategy == "global_opt":
        if validation_labels is None:
            raise ValueError("validation_labels required for global_opt strategy")
        threshold = find_optimal_threshold(probabilities, validation_labels)
        
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")
    
    # Compute resulting prevalence
    predicted_prevalence = (probabilities >= threshold).mean()
    
    logger.info(f"Adaptive threshold ({strategy}): {threshold:.4f}")
    logger.info(f"Resulting prevalence: {predicted_prevalence:.3f}")
    
    return threshold


def find_optimal_threshold(probabilities: np.ndarray,
                          labels: np.ndarray,
                          metric: str = "f1") -> float:
    """
    Find optimal threshold by maximizing a validation metric.
    
    Args:
        probabilities: Prediction probabilities
        labels: Ground truth binary labels
        metric: Optimization metric ("f1", "iou", "auprc")
        
    Returns:
        Optimal threshold
    """
    from sklearn.metrics import f1_score, precision_recall_curve, auc
    
    if metric == "f1":
        # Grid search for best F1
        thresholds = np.linspace(0.1, 0.9, 100)
        best_score = 0
        best_thresh = 0.5
        
        for thresh in thresholds:
            preds = (probabilities >= thresh).astype(int)
            score = f1_score(labels, preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_thresh = thresh
                
        logger.info(f"Optimal F1 threshold: {best_thresh:.4f} (F1: {best_score:.4f})")
        return best_thresh
        
    elif metric == "iou":
        # Grid search for best IoU
        thresholds = np.linspace(0.1, 0.9, 100)
        best_score = 0
        best_thresh = 0.5
        
        for thresh in thresholds:
            preds = (probabilities >= thresh).astype(int)
            intersection = (labels & preds).sum()
            union = (labels | preds).sum()
            iou = intersection / (union + 1e-8)
            if iou > best_score:
                best_score = iou
                best_thresh = thresh
                
        logger.info(f"Optimal IoU threshold: {best_thresh:.4f} (IoU: {best_score:.4f})")
        return best_thresh
        
    elif metric == "auprc":
        # Use precision-recall curve to find best threshold
        precision, recall, thresholds = precision_recall_curve(labels, probabilities)
        # Find threshold that maximizes F1 on PR curve
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # Exclude last point
        best_thresh = thresholds[best_idx]
        
        logger.info(f"PR-optimal threshold: {best_thresh:.4f} "
                   f"(F1: {f1_scores[best_idx]:.4f})")
        return best_thresh
        
    else:
        raise ValueError(f"Unknown optimization metric: {metric}")


def calibrate_predictions(probabilities: np.ndarray,
                         labels: np.ndarray,
                         method: str = "platt") -> Tuple[np.ndarray, Any]:
    """
    Calibrate prediction probabilities using Platt scaling or isotonic regression.
    
    Note: This won't change AUPRC (rank-preserving) but can improve IoU/F1.
    
    Args:
        probabilities: Raw prediction probabilities
        labels: Ground truth binary labels
        method: Calibration method ("platt", "isotonic")
        
    Returns:
        Tuple of (calibrated_probabilities, calibrator_object)
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    
    if method == "platt":
        # Platt scaling (sigmoid)
        calibrator = LogisticRegression()
        calibrator.fit(probabilities.reshape(-1, 1), labels)
        calibrated = calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        
    elif method == "isotonic":
        # Isotonic regression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrated = calibrator.fit_transform(probabilities, labels)
        
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    logger.info(f"Applied {method} calibration")
    logger.info(f"Before: range [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    logger.info(f"After:  range [{calibrated.min():.4f}, {calibrated.max():.4f}]")
    
    return calibrated, calibrator


def multi_seed_consensus(seed_pockets: List[List[Dict]],
                        center_threshold: float = 6.0) -> List[Dict]:
    """
    Build consensus pockets across multiple seeds by clustering pocket centers.
    
    Args:
        seed_pockets: List of pocket lists, one per seed
        center_threshold: Maximum distance for pocket center clustering (Å)
        
    Returns:
        Consensus pocket list with averaged scores
    """
    from sklearn.cluster import DBSCAN
    
    if not seed_pockets:
        return []
    
    # Collect all pocket centers and metadata
    all_centers = []
    all_metadata = []
    
    for seed_idx, pockets in enumerate(seed_pockets):
        for pocket in pockets:
            all_centers.append(pocket['center'])
            all_metadata.append({
                'seed': seed_idx,
                'original_pocket': pocket
            })
    
    if not all_centers:
        return []
    
    # Cluster pocket centers
    centers_array = np.array(all_centers)
    clustering = DBSCAN(eps=center_threshold, min_samples=1)
    cluster_labels = clustering.fit_predict(centers_array)
    
    # Build consensus pockets
    consensus_pockets = []
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:  # Noise cluster
            continue
            
        cluster_mask = (cluster_labels == cluster_id)
        cluster_metadata = [all_metadata[i] for i in np.where(cluster_mask)[0]]
        
        # Average properties
        cluster_centers = centers_array[cluster_mask]
        cluster_pockets = [meta['original_pocket'] for meta in cluster_metadata]
        
        consensus_center = cluster_centers.mean(axis=0)
        consensus_score = np.mean([p['score'] for p in cluster_pockets])
        consensus_size = np.mean([p['size'] for p in cluster_pockets])
        consensus_sump = np.mean([p['sump'] for p in cluster_pockets])
        
        # Count how many seeds agree
        participating_seeds = set(meta['seed'] for meta in cluster_metadata)
        seed_agreement = len(participating_seeds) / len(seed_pockets)
        
        consensus_pocket = {
            'center': consensus_center.tolist(),
            'score': consensus_score,
            'size': int(consensus_size),
            'sump': consensus_sump,
            'seed_agreement': seed_agreement,
            'participating_seeds': sorted(participating_seeds),
            'source_pockets': cluster_pockets
        }
        
        consensus_pockets.append(consensus_pocket)
    
    # Sort by score
    consensus_pockets.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"Built {len(consensus_pockets)} consensus pockets from "
               f"{len(all_centers)} individual pockets across {len(seed_pockets)} seeds")
    
    return consensus_pockets


def non_maximum_suppression(pockets: List[Dict],
                           distance_threshold: float = 6.0) -> List[Dict]:
    """
    Apply non-maximum suppression to remove nearby pockets.
    
    Args:
        pockets: List of pocket dictionaries (must have 'center' and 'score')
        distance_threshold: Minimum distance between pocket centers
        
    Returns:
        Filtered pocket list
    """
    if len(pockets) <= 1:
        return pockets
    
    # Sort by score (highest first)
    sorted_pockets = sorted(pockets, key=lambda x: x['score'], reverse=True)
    
    keep_pockets = []
    centers = np.array([p['center'] for p in sorted_pockets])
    
    for i, pocket in enumerate(sorted_pockets):
        current_center = centers[i]
        
        # Check if too close to any already kept pocket
        too_close = False
        for kept_pocket in keep_pockets:
            kept_center = np.array(kept_pocket['center'])
            distance = np.linalg.norm(current_center - kept_center)
            if distance < distance_threshold:
                too_close = True
                break
        
        if not too_close:
            keep_pockets.append(pocket)
    
    logger.info(f"NMS: kept {len(keep_pockets)}/{len(pockets)} pockets "
               f"(threshold: {distance_threshold} Å)")
    
    return keep_pockets


# Example usage
if __name__ == "__main__":
    # Test ensemble functionality
    np.random.seed(42)
    
    # Simulate multi-seed predictions
    n_residues = 1000
    n_seeds = 7
    
    seed_predictions = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        # Simulate correlated but slightly different predictions
        base_signal = np.random.rand(n_residues)
        noise = np.random.randn(n_residues) * 0.1
        predictions = np.clip(base_signal + noise, 0, 1)
        seed_predictions.append(predictions)
    
    # Test ensembling
    ensemble = ensemble_predictions(seed_predictions, method="mean")
    print(f"Individual seed ranges: {[f'[{p.min():.3f}, {p.max():.3f}]' for p in seed_predictions[:3]]}")
    print(f"Ensemble range: [{ensemble.min():.3f}, {ensemble.max():.3f}]")
    
    # Test adaptive thresholding
    threshold = adaptive_threshold(ensemble, strategy="percentile95")
    print(f"95th percentile threshold: {threshold:.4f}")
    
    print("Ensemble utilities test completed!")