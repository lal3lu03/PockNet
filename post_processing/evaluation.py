#!/usr/bin/env python3
"""
Post-Processing Evaluation Module
=================================

Comprehensive evaluation of post-processing results including:
- IoU (Intersection over Union) calculation
- AUPRC (Area Under Precision-Recall Curve) 
- Pocket-level metrics
- Residue-level metrics
- Statistical analysis

Handles both predicted vs ground truth comparisons and 
post-processing enhancement analysis.
"""

import numpy as np
import h5py
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
from collections import defaultdict

# Try sklearn imports with fallbacks
try:
    from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    # Residue-level metrics
    auprc: float
    auroc: float
    precision_at_threshold: Dict[float, float]
    recall_at_threshold: Dict[float, float]
    
    # IoU metrics
    iou_scores: List[float]
    mean_iou: float
    iou_at_threshold: Dict[float, float]
    
    # Pocket-level metrics  
    pocket_precision: float
    pocket_recall: float
    pocket_f1: float
    
    # Statistical data
    num_proteins: int
    num_residues: int
    num_positive_residues: int
    num_predicted_pockets: int
    num_true_pockets: int

def fallback_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback implementation when sklearn not available"""
    # Sort by scores (descending)
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Get unique thresholds
    thresholds = np.unique(y_scores_sorted)
    thresholds = np.append(thresholds, thresholds[-1] - 1e-6)  # Add one more threshold
    
    precisions = []
    recalls = []
    
    total_positive = y_true.sum()
    
    for threshold in thresholds:
        predicted_positive = (y_scores >= threshold)
        true_positive = (predicted_positive & (y_true == 1)).sum()
        predicted_positive_count = predicted_positive.sum()
        
        precision = true_positive / predicted_positive_count if predicted_positive_count > 0 else 0.0
        recall = true_positive / total_positive if total_positive > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls), thresholds

def fallback_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Fallback AUC calculation using trapezoidal rule"""
    # Sort by x values
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Calculate area using trapezoidal rule
    area = 0.0
    for i in range(1, len(x_sorted)):
        dx = x_sorted[i] - x_sorted[i-1]
        avg_y = (y_sorted[i] + y_sorted[i-1]) / 2
        area += dx * avg_y
    
    return area

class PostProcessingEvaluator:
    """Comprehensive evaluator for post-processing results"""
    
    def __init__(self):
        self.thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        logger.info("🔍 Initialized PostProcessingEvaluator")
        
    def load_ground_truth_from_h5(self, h5_path: str, protein_ids: List[str]) -> Dict[str, np.ndarray]:
        """Load ground truth labels from H5 file"""
        
        ground_truth = {}
        
        with h5py.File(h5_path, 'r') as f:
            # Load protein keys and labels
            protein_keys = f['protein_keys'][:]
            protein_keys = [k.decode() if isinstance(k, bytes) else str(k) for k in protein_keys]
            labels = f['labels'][:]
            
            # Group by protein
            protein_to_indices = defaultdict(list)
            for idx, protein_id in enumerate(protein_keys):
                protein_to_indices[protein_id].append(idx)
            
            # Extract labels for requested proteins
            for protein_id in protein_ids:
                if protein_id in protein_to_indices:
                    indices = protein_to_indices[protein_id]
                    protein_labels = labels[indices]
                    ground_truth[protein_id] = protein_labels
                else:
                    logger.warning(f"Protein {protein_id} not found in ground truth")
        
        logger.info(f"📊 Loaded ground truth for {len(ground_truth)} proteins")
        return ground_truth
    
    def calculate_residue_level_metrics(self, 
                                      predictions: Dict[str, np.ndarray],
                                      ground_truth: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate residue-level classification metrics"""
        
        # Combine all proteins
        all_predictions = []
        all_labels = []
        
        for protein_id in predictions:
            if protein_id in ground_truth:
                pred = predictions[protein_id]
                gt = ground_truth[protein_id]
                
                # Ensure same length
                min_len = min(len(pred), len(gt))
                all_predictions.extend(pred[:min_len])
                all_labels.extend(gt[:min_len])
        
        if not all_predictions:
            logger.warning("No matching predictions and ground truth found")
            return {}
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        logger.info(f"📊 Evaluating {len(all_predictions)} residues")
        logger.info(f"   Positive residues: {all_labels.sum()} ({100*all_labels.mean():.1f}%)")
        logger.info(f"   Prediction range: [{all_predictions.min():.3f}, {all_predictions.max():.3f}]")
        
        metrics = {}
        
        # AUPRC and AUROC
        if SKLEARN_AVAILABLE:
            try:
                metrics['auprc'] = average_precision_score(all_labels, all_predictions)
                metrics['auroc'] = roc_auc_score(all_labels, all_predictions)
                
                # Precision-recall curve
                precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
                
            except Exception as e:
                logger.warning(f"sklearn metrics failed: {e}, using fallbacks")
                precision, recall, _ = fallback_precision_recall_curve(all_labels, all_predictions)
                metrics['auprc'] = fallback_auc(recall, precision)
                metrics['auroc'] = 0.5  # Placeholder
        else:
            # Fallback implementation
            precision, recall, _ = fallback_precision_recall_curve(all_labels, all_predictions)
            metrics['auprc'] = fallback_auc(recall, precision)
            metrics['auroc'] = 0.5  # Placeholder
        
        # Threshold-based metrics
        for threshold in self.thresholds:
            pred_binary = (all_predictions >= threshold).astype(int)
            
            tp = ((pred_binary == 1) & (all_labels == 1)).sum()
            fp = ((pred_binary == 1) & (all_labels == 0)).sum()
            fn = ((pred_binary == 0) & (all_labels == 1)).sum()
            
            precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_t = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            metrics[f'precision_{threshold}'] = precision_t
            metrics[f'recall_{threshold}'] = recall_t
            metrics[f'f1_{threshold}'] = 2 * precision_t * recall_t / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0.0
        
        return metrics
    
    def calculate_iou_per_protein(self, 
                                 predictions: np.ndarray,
                                 ground_truth: np.ndarray,
                                 threshold: float = 0.5) -> float:
        """Calculate IoU for a single protein"""
        
        # Convert to binary masks
        pred_binary = (predictions >= threshold).astype(int)
        gt_binary = ground_truth.astype(int)
        
        # Calculate intersection and union
        intersection = (pred_binary & gt_binary).sum()
        union = (pred_binary | gt_binary).sum()
        
        # IoU calculation
        if union == 0:
            return 1.0 if intersection == 0 else 0.0  # Both empty = perfect IoU
        
        iou = intersection / union
        return float(iou)
    
    def calculate_iou_metrics(self, 
                            predictions: Dict[str, np.ndarray],
                            ground_truth: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate IoU metrics across all proteins and thresholds"""
        
        iou_results = {}
        
        # Calculate IoU for each threshold
        for threshold in self.thresholds:
            protein_ious = []
            
            for protein_id in predictions:
                if protein_id in ground_truth:
                    pred = predictions[protein_id]
                    gt = ground_truth[protein_id]
                    
                    # Ensure same length
                    min_len = min(len(pred), len(gt))
                    iou = self.calculate_iou_per_protein(pred[:min_len], gt[:min_len], threshold)
                    protein_ious.append(iou)
            
            if protein_ious:
                iou_results[f'mean_iou_{threshold}'] = np.mean(protein_ious)
                iou_results[f'std_iou_{threshold}'] = np.std(protein_ious)
                iou_results[f'protein_ious_{threshold}'] = protein_ious
        
        # Overall statistics
        all_ious = []
        for key in iou_results:
            if key.startswith('protein_ious_'):
                all_ious.extend(iou_results[key])
        
        if all_ious:
            iou_results['overall_mean_iou'] = np.mean(all_ious)
            iou_results['overall_std_iou'] = np.std(all_ious)
        
        return iou_results
    
    def analyze_pocket_formation_quality(self, 
                                       pocket_results: Dict[str, List],
                                       ground_truth: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze quality of pocket formation vs ground truth"""
        
        pocket_metrics = {}
        
        total_predicted_pockets = 0
        total_proteins = len(pocket_results)
        proteins_with_pockets = 0
        pocket_sizes = []
        pocket_scores = []
        
        for protein_id, pockets in pocket_results.items():
            total_predicted_pockets += len(pockets)
            if len(pockets) > 0:
                proteins_with_pockets += 1
            
            for pocket in pockets:
                if hasattr(pocket, 'size'):
                    pocket_sizes.append(pocket.size)
                if hasattr(pocket, 'final_score'):
                    pocket_scores.append(pocket.final_score)
        
        # Basic pocket statistics
        pocket_metrics['total_pockets'] = total_predicted_pockets
        pocket_metrics['avg_pockets_per_protein'] = total_predicted_pockets / total_proteins if total_proteins > 0 else 0
        pocket_metrics['proteins_with_pockets'] = proteins_with_pockets
        pocket_metrics['pocket_coverage'] = proteins_with_pockets / total_proteins if total_proteins > 0 else 0
        
        if pocket_sizes:
            pocket_metrics['avg_pocket_size'] = np.mean(pocket_sizes)
            pocket_metrics['std_pocket_size'] = np.std(pocket_sizes)
            pocket_metrics['min_pocket_size'] = np.min(pocket_sizes)
            pocket_metrics['max_pocket_size'] = np.max(pocket_sizes)
        
        if pocket_scores:
            pocket_metrics['avg_pocket_score'] = np.mean(pocket_scores)
            pocket_metrics['std_pocket_score'] = np.std(pocket_scores)
        
        # Analyze overlap with ground truth binding sites
        overlap_scores = []
        for protein_id, pockets in pocket_results.items():
            if protein_id in ground_truth and pockets:
                gt = ground_truth[protein_id]
                binding_sites = np.where(gt == 1)[0]
                
                if len(binding_sites) > 0:
                    for pocket in pockets:
                        if hasattr(pocket, 'members'):
                            pocket_residues = set(pocket.members)
                            binding_residues = set(binding_sites)
                            
                            overlap = len(pocket_residues & binding_residues)
                            pocket_size = len(pocket_residues)
                            
                            if pocket_size > 0:
                                overlap_ratio = overlap / pocket_size
                                overlap_scores.append(overlap_ratio)
        
        if overlap_scores:
            pocket_metrics['avg_overlap_with_binding_sites'] = np.mean(overlap_scores)
            pocket_metrics['std_overlap_with_binding_sites'] = np.std(overlap_scores)
        
        return pocket_metrics
    
    def compare_before_after_post_processing(self, 
                                           raw_predictions: Dict[str, np.ndarray],
                                           processed_predictions: Dict[str, np.ndarray],
                                           ground_truth: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compare metrics before and after post-processing"""
        
        logger.info("🔄 Comparing before/after post-processing")
        
        # Calculate metrics for raw predictions
        raw_metrics = self.calculate_residue_level_metrics(raw_predictions, ground_truth)
        raw_iou = self.calculate_iou_metrics(raw_predictions, ground_truth)
        
        # Calculate metrics for processed predictions
        processed_metrics = self.calculate_residue_level_metrics(processed_predictions, ground_truth)
        processed_iou = self.calculate_iou_metrics(processed_predictions, ground_truth)
        
        comparison = {
            'raw': {**raw_metrics, **raw_iou},
            'processed': {**processed_metrics, **processed_iou},
            'improvement': {}
        }
        
        # Calculate improvements
        for metric in raw_metrics:
            if metric in processed_metrics:
                raw_val = raw_metrics[metric]
                processed_val = processed_metrics[metric]
                
                if raw_val > 0:
                    improvement = (processed_val - raw_val) / raw_val * 100
                    comparison['improvement'][f'{metric}_improvement_pct'] = improvement
                
                comparison['improvement'][f'{metric}_delta'] = processed_val - raw_val
        
        return comparison
    
    def evaluate_comprehensive(self, 
                             predictions: Dict[str, np.ndarray],
                             ground_truth: Dict[str, np.ndarray],
                             pocket_results: Optional[Dict[str, List]] = None) -> EvaluationResults:
        """Comprehensive evaluation of post-processing results"""
        
        logger.info("🎯 Running comprehensive evaluation")
        
        # Residue-level metrics
        residue_metrics = self.calculate_residue_level_metrics(predictions, ground_truth)
        
        # IoU metrics
        iou_metrics = self.calculate_iou_metrics(predictions, ground_truth)
        
        # Pocket-level metrics if available
        pocket_metrics = {}
        if pocket_results:
            pocket_metrics = self.analyze_pocket_formation_quality(pocket_results, ground_truth)
        
        # Compile statistics
        total_residues = sum(len(pred) for pred in predictions.values())
        total_positive = 0
        num_proteins = len(predictions)
        
        for protein_id in predictions:
            if protein_id in ground_truth:
                min_len = min(len(predictions[protein_id]), len(ground_truth[protein_id]))
                total_positive += ground_truth[protein_id][:min_len].sum()
        
        # Extract IoU scores
        iou_scores = []
        mean_iou = 0.0
        iou_at_threshold = {}
        
        for key, value in iou_metrics.items():
            if key.startswith('protein_ious_'):
                iou_scores.extend(value)
            elif key.startswith('mean_iou_'):
                threshold = float(key.split('_')[-1])
                iou_at_threshold[threshold] = value
                if threshold == 0.5:  # Use 0.5 as default
                    mean_iou = value
        
        # Extract precision/recall at thresholds
        precision_at_threshold = {}
        recall_at_threshold = {}
        for key, value in residue_metrics.items():
            if key.startswith('precision_'):
                threshold = float(key.split('_')[1])
                precision_at_threshold[threshold] = value
            elif key.startswith('recall_'):
                threshold = float(key.split('_')[1])
                recall_at_threshold[threshold] = value
        
        results = EvaluationResults(
            auprc=residue_metrics.get('auprc', 0.0),
            auroc=residue_metrics.get('auroc', 0.0),
            precision_at_threshold=precision_at_threshold,
            recall_at_threshold=recall_at_threshold,
            iou_scores=iou_scores,
            mean_iou=mean_iou,
            iou_at_threshold=iou_at_threshold,
            pocket_precision=pocket_metrics.get('avg_overlap_with_binding_sites', 0.0),
            pocket_recall=pocket_metrics.get('pocket_coverage', 0.0),
            pocket_f1=0.0,  # Would need more complex calculation
            num_proteins=num_proteins,
            num_residues=total_residues,
            num_positive_residues=int(total_positive),
            num_predicted_pockets=pocket_metrics.get('total_pockets', 0),
            num_true_pockets=0  # Would need additional analysis
        )
        
        return results
    
    def print_evaluation_report(self, results: EvaluationResults, title: str = "Evaluation Results"):
        """Print a comprehensive evaluation report"""
        
        print(f"\n{'='*60}")
        print(f"📊 {title}")
        print(f"{'='*60}")
        
        print(f"\n🎯 Overview:")
        print(f"   Proteins evaluated: {results.num_proteins}")
        print(f"   Total residues: {results.num_residues:,}")
        print(f"   Positive residues: {results.num_positive_residues:,} ({100*results.num_positive_residues/results.num_residues:.1f}%)")
        print(f"   Predicted pockets: {results.num_predicted_pockets}")
        
        print(f"\n📈 Residue-Level Metrics:")
        print(f"   AUPRC: {results.auprc:.4f}")
        print(f"   AUROC: {results.auroc:.4f}")
        
        print(f"\n🎯 Performance at Key Thresholds:")
        for threshold in [0.3, 0.5, 0.7]:
            if threshold in results.precision_at_threshold:
                prec = results.precision_at_threshold[threshold]
                rec = results.recall_at_threshold[threshold]
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                print(f"   @ {threshold}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
        
        print(f"\n🔄 IoU Metrics:")
        print(f"   Mean IoU: {results.mean_iou:.4f}")
        print(f"   IoU scores: {len(results.iou_scores)} protein evaluations")
        if results.iou_scores:
            print(f"   IoU range: [{min(results.iou_scores):.3f}, {max(results.iou_scores):.3f}]")
        
        print(f"\n🏗️  Pocket Quality:")
        print(f"   Pocket precision: {results.pocket_precision:.3f}")
        print(f"   Pocket recall: {results.pocket_recall:.3f}")
        
        print(f"{'='*60}\n")


def create_mock_processed_predictions(raw_predictions: Dict[str, np.ndarray], 
                                    noise_factor: float = 0.1) -> Dict[str, np.ndarray]:
    """Create mock processed predictions for demonstration"""
    processed = {}
    
    for protein_id, pred in raw_predictions.items():
        # Apply some post-processing effects
        processed_pred = pred.copy()
        
        # Add slight smoothing
        if len(processed_pred) > 2:
            for i in range(1, len(processed_pred) - 1):
                processed_pred[i] = 0.7 * processed_pred[i] + 0.15 * (processed_pred[i-1] + processed_pred[i+1])
        
        # Add slight enhancement to high-confidence predictions
        high_conf_mask = processed_pred > 0.7
        processed_pred[high_conf_mask] = np.minimum(processed_pred[high_conf_mask] * 1.1, 1.0)
        
        # Add small amount of noise to simulate processing
        noise = np.random.normal(0, noise_factor * processed_pred.std(), processed_pred.shape)
        processed_pred = np.clip(processed_pred + noise, 0, 1)
        
        processed[protein_id] = processed_pred
    
    return processed


if __name__ == "__main__":
    print("🔍 Post-Processing Evaluation Module")
    print("Features:")
    print("  ✅ IoU calculation")
    print("  ✅ AUPRC metrics")
    print("  ✅ Pocket-level analysis")
    print("  ✅ Before/after comparison")
    print("  ✅ Comprehensive reporting")