"""
PockNet Post-Processing Pipeline - Main Interface
================================================

Main pipeline class that orchestrates the complete post-processing workflow
from residue predictions to pocket formation and evaluation.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import time

from .pocket_formation import (
    PostProcessingConfig, Residue, Pocket, 
    GraphBuilder, SpatialSmoother, ThresholdSelector, 
    PocketFormer, IoUCalculator, EnsembleProcessor
)

logger = logging.getLogger(__name__)


class PostProcessingPipeline:
    """
    Complete post-processing pipeline for PockNet predictions.
    
    Converts residue-level predictions to pocket predictions with:
    - Multi-seed ensembling
    - Spatial smoothing
    - Adaptive thresholding
    - Pocket formation and scoring
    - IoU evaluation
    """
    
    def __init__(self, config: Optional[PostProcessingConfig] = None):
        self.config = config or PostProcessingConfig()
        self.pocket_former = PocketFormer(self.config)
        
    def process_protein(self,
                       residue_data: List[Residue],
                       gt_pockets: Optional[List[List[Tuple[str, int]]]] = None) -> Dict[str, Any]:
        """
        Process a single protein through the complete pipeline.
        
        Args:
            residue_data: List of residues with predictions
            gt_pockets: Ground truth pockets for evaluation (optional)
            
        Returns:
            Dictionary with predictions and metrics
        """
        if not residue_data:
            return {'pockets': [], 'metrics': {}}
        
        start_time = time.time()
        
        # Extract predictions and coordinates
        predictions = np.array([r.prob for r in residue_data])
        coords = np.stack([r.xyz for r in residue_data])
        
        # Step 1: Spatial smoothing (if enabled)
        if self.config.enable_smoothing:
            graph = GraphBuilder.make_graph(coords, 
                                          radius=self.config.graph_radius,
                                          knn=self.config.knn_fallback)
            
            smoothed_preds = SpatialSmoother.smooth_predictions(
                predictions, graph, alpha=self.config.smoothing_alpha
            )
            
            # Update residue predictions
            for i, residue in enumerate(residue_data):
                residue.prob = smoothed_preds[i]
        
        # Step 2: Threshold selection
        threshold = self._select_threshold(predictions)
        
        # Step 3: Pocket formation
        pockets = self.pocket_former.form_pockets(residue_data, threshold)
        
        # Step 4: Calculate metrics
        metrics = {
            'n_residues': len(residue_data),
            'n_surface_residues': sum(1 for r in residue_data if r.rsa >= self.config.rsa_min),
            'n_positive_residues': sum(1 for r in residue_data if r.prob >= threshold),
            'n_pockets': len(pockets),
            'threshold_used': threshold,
            'processing_time': time.time() - start_time
        }
        
        # Residue-level metrics
        predictions_final = np.array([r.prob for r in residue_data])
        metrics.update({
            'pred_mean': float(predictions_final.mean()),
            'pred_std': float(predictions_final.std()),
            'pred_max': float(predictions_final.max()),
            'positive_rate': float((predictions_final >= threshold).mean())
        })
        
        # Add pocket scores
        if pockets:
            pocket_scores = [p.score for p in pockets]
            metrics.update({
                'pocket_scores_mean': float(np.mean(pocket_scores)),
                'pocket_scores_max': float(np.max(pocket_scores)),
                'total_pocket_score': float(np.sum(pocket_scores))
            })
        
        # IoU evaluation (if ground truth provided)
        if gt_pockets is not None:
            iou_metrics = IoUCalculator.pocket_iou(pockets, gt_pockets)
            metrics.update(iou_metrics)
        
        return {
            'pockets': pockets,
            'metrics': metrics,
            'threshold': threshold,
            'residue_predictions': predictions_final
        }
    
    def process_ensemble(self,
                        protein_predictions: List[List[Residue]],
                        gt_pockets: Optional[List[List[Tuple[str, int]]]] = None) -> Dict[str, Any]:
        """
        Process ensemble predictions for a single protein.
        
        Args:
            protein_predictions: List of residue lists from different seeds
            gt_pockets: Ground truth pockets for evaluation
            
        Returns:
            Dictionary with ensemble predictions and metrics
        """
        logger.info(f"Processing ensemble of {len(protein_predictions)} seeds")
        
        # Step 1: Create ensemble residue data
        ensemble_residues = EnsembleProcessor.ensemble_residue_data(protein_predictions)
        
        # Step 2: Process ensemble through pipeline
        results = self.process_protein(ensemble_residues, gt_pockets)
        
        # Add ensemble-specific metrics
        if len(protein_predictions) > 1:
            # Calculate agreement between seeds
            seed_predictions = [np.array([r.prob for r in seed_res]) 
                              for seed_res in protein_predictions]
            
            # Pairwise correlations
            correlations = []
            for i in range(len(seed_predictions)):
                for j in range(i + 1, len(seed_predictions)):
                    min_len = min(len(seed_predictions[i]), len(seed_predictions[j]))
                    if min_len > 0:
                        corr = np.corrcoef(seed_predictions[i][:min_len], 
                                         seed_predictions[j][:min_len])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
            
            results['metrics'].update({
                'n_seeds': len(protein_predictions),
                'seed_correlation_mean': float(np.mean(correlations)) if correlations else 0.0,
                'seed_correlation_std': float(np.std(correlations)) if correlations else 0.0
            })
        
        return results
    
    def _select_threshold(self, predictions: np.ndarray) -> float:
        """Select threshold based on configuration"""
        if self.config.threshold_mode == "global" and self.config.global_threshold is not None:
            return ThresholdSelector.global_threshold(predictions, self.config.global_threshold)
        elif self.config.threshold_mode == "adaptive":
            return ThresholdSelector.adaptive_threshold(predictions, self.config.target_prevalence)
        elif self.config.threshold_mode == "percentile":
            return ThresholdSelector.percentile_threshold(predictions, self.config.percentile_threshold)
        else:
            # Default to 95th percentile
            return ThresholdSelector.percentile_threshold(predictions, 95.0)
    
    def evaluate_dataset(self,
                        protein_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate performance across a dataset of proteins.
        
        Args:
            protein_results: Dictionary mapping protein_id -> results
            
        Returns:
            Dataset-level aggregated metrics
        """
        if not protein_results:
            return {}
        
        # Collect all metrics
        all_metrics = [result['metrics'] for result in protein_results.values()]
        
        # Aggregate metrics
        aggregated = {}
        
        # Simple averages
        avg_metrics = [
            'n_residues', 'n_surface_residues', 'n_positive_residues', 'n_pockets',
            'pred_mean', 'pred_std', 'positive_rate', 'processing_time'
        ]
        
        for metric in avg_metrics:
            values = [m.get(metric, 0) for m in all_metrics]
            if values:
                aggregated[f'{metric}_mean'] = float(np.mean(values))
                aggregated[f'{metric}_std'] = float(np.std(values))
        
        # IoU metrics (only for proteins with ground truth)
        iou_metrics = ['mean_iou', 'max_iou', 'pocket_recall', 'pocket_precision']
        for metric in iou_metrics:
            values = [m.get(metric) for m in all_metrics if m.get(metric) is not None]
            if values:
                aggregated[f'{metric}_mean'] = float(np.mean(values))
                aggregated[f'{metric}_std'] = float(np.std(values))
        
        # Total counts
        aggregated['total_proteins'] = len(protein_results)
        aggregated['total_residues'] = sum(m.get('n_residues', 0) for m in all_metrics)
        aggregated['total_pockets'] = sum(m.get('n_pockets', 0) for m in all_metrics)
        
        # Pocket score statistics
        all_pocket_scores = []
        for result in protein_results.values():
            if 'pockets' in result:
                all_pocket_scores.extend([p.score for p in result['pockets']])
        
        if all_pocket_scores:
            aggregated['all_pocket_scores_mean'] = float(np.mean(all_pocket_scores))
            aggregated['all_pocket_scores_std'] = float(np.std(all_pocket_scores))
            aggregated['all_pocket_scores_max'] = float(np.max(all_pocket_scores))
        
        return aggregated


def create_residue_from_prediction(protein_id: str,
                                  chain: str,
                                  res_id: int,
                                  coordinates: np.ndarray,
                                  rsa: float,
                                  prediction: float) -> Residue:
    """
    Helper function to create Residue objects from prediction data.
    
    Args:
        protein_id: Protein identifier
        chain: Chain identifier  
        res_id: Residue number
        coordinates: (3,) array of Cα/Cβ coordinates
        rsa: Relative solvent accessibility
        prediction: Model prediction probability
        
    Returns:
        Residue object
    """
    return Residue(
        chain=chain,
        res_id=res_id,
        xyz=np.array(coordinates, dtype=np.float32),
        rsa=float(rsa),
        prob=float(prediction)
    )


def load_ground_truth_pockets(pocket_file: str) -> Dict[str, List[List[Tuple[str, int]]]]:
    """
    Load ground truth pocket definitions from file.
    
    Expected format: Each line contains protein_id, pocket_id, chain, res_id
    
    Args:
        pocket_file: Path to ground truth file
        
    Returns:
        Dictionary mapping protein_id -> list of pockets (each pocket is list of (chain, res_id))
    """
    gt_pockets = {}
    
    try:
        with open(pocket_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if len(parts) >= 4:
                    protein_id = parts[0]
                    pocket_id = int(parts[1])
                    chain = parts[2]
                    res_id = int(parts[3])
                    
                    if protein_id not in gt_pockets:
                        gt_pockets[protein_id] = []
                    
                    # Extend pocket list to accommodate pocket_id
                    while len(gt_pockets[protein_id]) <= pocket_id:
                        gt_pockets[protein_id].append([])
                    
                    gt_pockets[protein_id][pocket_id].append((chain, res_id))
        
        logger.info(f"Loaded ground truth for {len(gt_pockets)} proteins")
        
    except Exception as e:
        logger.error(f"Failed to load ground truth pockets: {e}")
        
    return gt_pockets