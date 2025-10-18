"""
Hyperparameter Sweep Framework
==============================

Implements automated hyperparameter optimization for pocket post-processing,
including grid search and objective function optimization for pocket-level metrics.

Optimizes:
- Graph radius (6, 7, 8 Å)
- Min cluster size (3, 5, 8)  
- Sum probability filter (1.5, 2.0, 3.0)
- Thresholding strategy (global-opt, adaptive prevalence, 95th percentile)

Author: PockNet Team
"""

import itertools
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
import json
import pickle
from pathlib import Path
import time

from .core import PocketPostProcessor
from .metrics import PocketEvaluator, pocket_recall_at_k, center_distance_metrics, pocket_iou_metrics

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SweepConfig:
    """Configuration for hyperparameter sweep."""
    graph_radii: List[float] = None
    min_cluster_sizes: List[int] = None
    sump_thresholds: List[float] = None
    threshold_strategies: List[str] = None
    rsa_thresholds: List[float] = None
    max_pockets: List[int] = None
    
    def __post_init__(self):
        # Set defaults if not provided
        if self.graph_radii is None:
            self.graph_radii = [6.0, 7.0, 8.0]
        if self.min_cluster_sizes is None:
            self.min_cluster_sizes = [3, 5, 8]
        if self.sump_thresholds is None:
            self.sump_thresholds = [1.5, 2.0, 3.0]
        if self.threshold_strategies is None:
            self.threshold_strategies = ["percentile95", "prevalence"]
        if self.rsa_thresholds is None:
            self.rsa_thresholds = [0.2]
        if self.max_pockets is None:
            self.max_pockets = [5]


@dataclass
class SweepResult:
    """Results from a single hyperparameter configuration."""
    config: Dict[str, Any]
    metrics: Dict[str, float]
    execution_time: float
    n_proteins: int
    
    def __post_init__(self):
        # Compute combined score
        self.combined_score = self._compute_combined_score()
    
    def _compute_combined_score(self) -> float:
        """Compute combined score for ranking configurations."""
        # Weighted combination of key metrics
        weights = {
            'recall_at_1': 0.3,
            'recall_at_3': 0.3, 
            'mean_iou': 0.2,
            'precision': 0.2
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in self.metrics:
                score += weight * self.metrics[metric]
        
        return score


class HyperparameterSweep:
    """
    Automated hyperparameter optimization for pocket post-processing.
    """
    
    def __init__(self,
                 validation_data: List[Dict[str, Any]],
                 sweep_config: Optional[SweepConfig] = None,
                 objective_function: Optional[Callable] = None,
                 n_jobs: int = 1):
        """
        Initialize hyperparameter sweep.
        
        Args:
            validation_data: List of validation examples with residues and ground truth
            sweep_config: Configuration for parameter ranges
            objective_function: Custom objective function (if None, use default)
            n_jobs: Number of parallel jobs (currently not implemented)
        """
        self.validation_data = validation_data
        self.sweep_config = sweep_config or SweepConfig()
        self.objective_function = objective_function or self._default_objective
        self.n_jobs = n_jobs
        
        self.results = []
        self.best_config = None
        self.best_score = -1
        
        logger.info(f"Initialized hyperparameter sweep with {len(validation_data)} proteins")
        logger.info(f"Parameter ranges:")
        for attr, value in asdict(self.sweep_config).items():
            logger.info(f"  {attr}: {value}")
    
    def run_sweep(self, 
                  max_configs: Optional[int] = None,
                  save_path: Optional[str] = None) -> List[SweepResult]:
        """
        Run the hyperparameter sweep.
        
        Args:
            max_configs: Maximum number of configurations to test
            save_path: Path to save results
            
        Returns:
            List of SweepResult objects sorted by performance
        """
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations()
        
        if max_configs is not None:
            # Subsample if too many combinations
            if len(param_combinations) > max_configs:
                logger.info(f"Subsampling {max_configs} from {len(param_combinations)} configs")
                np.random.shuffle(param_combinations)
                param_combinations = param_combinations[:max_configs]
        
        logger.info(f"Testing {len(param_combinations)} parameter configurations")
        
        # Evaluate each configuration
        for i, config in enumerate(param_combinations):
            logger.info(f"Evaluating config {i+1}/{len(param_combinations)}: {config}")
            
            start_time = time.time()
            try:
                metrics = self._evaluate_config(config)
                execution_time = time.time() - start_time
                
                result = SweepResult(
                    config=config,
                    metrics=metrics,
                    execution_time=execution_time,
                    n_proteins=len(self.validation_data)
                )
                
                self.results.append(result)
                
                # Update best config
                if result.combined_score > self.best_score:
                    self.best_score = result.combined_score
                    self.best_config = config
                    logger.info(f"New best config (score: {self.best_score:.4f}): {config}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate config {config}: {e}")
                continue
        
        # Sort results by combined score
        self.results.sort(key=lambda r: r.combined_score, reverse=True)
        
        logger.info(f"Sweep completed! Best config: {self.best_config}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        # Save results if requested
        if save_path:
            self.save_results(save_path)
        
        return self.results
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        config = self.sweep_config
        
        # Create parameter grid
        param_grid = {
            'graph_radius': config.graph_radii,
            'min_cluster_size': config.min_cluster_sizes,
            'min_sump': config.sump_thresholds,
            'adaptive_strategy': config.threshold_strategies,
            'rsa_min': config.rsa_thresholds,
            'max_pockets': config.max_pockets
        }
        
        # Generate all combinations
        combinations = []
        for values in itertools.product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), values))
            combinations.append(param_dict)
        
        return combinations
    
    def _evaluate_config(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single parameter configuration."""
        
        # Extract adaptive strategy (not a PocketPostProcessor parameter)
        adaptive_strategy = config.pop('adaptive_strategy')
        
        # Create post-processor with current config
        processor = PocketPostProcessor(**config)
        evaluator = PocketEvaluator()
        
        # Process all validation proteins
        all_predictions = []
        all_ground_truths = []
        
        for protein_data in self.validation_data:
            try:
                # Extract data
                residues = protein_data['residues']
                ground_truth_pockets = protein_data['ground_truth_pockets']
                
                # Process with current configuration
                predicted_pockets = processor.process(
                    residues, 
                    threshold=None,  # Use adaptive
                    adaptive_strategy=adaptive_strategy
                )
                
                all_predictions.append(predicted_pockets)
                all_ground_truths.append(ground_truth_pockets)
                
            except Exception as e:
                logger.warning(f"Failed to process protein: {e}")
                # Add empty lists to maintain alignment
                all_predictions.append([])
                all_ground_truths.append(protein_data['ground_truth_pockets'])
        
        # Compute metrics
        metrics = {}
        
        # Pocket recall @ K
        recall_at_k = pocket_recall_at_k(
            all_predictions, all_ground_truths, k_values=[1, 3, 5], evaluator=evaluator
        )
        for k, recall in recall_at_k.items():
            metrics[f'recall_at_{k}'] = recall
        
        # Center distance metrics
        center_metrics = center_distance_metrics(
            all_predictions, all_ground_truths, evaluator=evaluator
        )
        metrics.update(center_metrics)
        
        # IoU metrics
        iou_metrics = pocket_iou_metrics(
            all_predictions, all_ground_truths, evaluator=evaluator
        )
        metrics.update(iou_metrics)
        
        # Overall precision/recall/F1
        total_predicted = sum(len(preds) for preds in all_predictions)
        total_ground_truth = sum(len(gts) for gts in all_ground_truths)
        
        # Count total matches across all proteins
        total_matches = 0
        for preds, gts in zip(all_predictions, all_ground_truths):
            if not preds or not gts:
                continue
            results = evaluator.evaluate_protein(preds, gts)
            total_matches += results['n_matches']
        
        metrics['precision'] = total_matches / (total_predicted + 1e-8)
        metrics['recall'] = total_matches / (total_ground_truth + 1e-8)
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
            metrics['precision'] + metrics['recall'] + 1e-8
        )
        
        # Add configuration parameters to metrics for tracking
        config['adaptive_strategy'] = adaptive_strategy  # Restore for saving
        
        return metrics
    
    def _default_objective(self, metrics: Dict[str, float]) -> float:
        """Default objective function for optimization."""
        # Weighted combination emphasizing recall@1 and IoU
        weights = {
            'recall_at_1': 0.4,
            'recall_at_3': 0.3,
            'mean_iou': 0.2,
            'f1': 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            score += weight * metrics.get(metric, 0.0)
        
        return score
    
    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Get the best configuration found."""
        return self.best_config
    
    def get_top_configs(self, n: int = 5) -> List[SweepResult]:
        """Get top N configurations."""
        return self.results[:n]
    
    def save_results(self, save_path: str):
        """Save sweep results to disk."""
        results_data = {
            'sweep_config': asdict(self.sweep_config),
            'best_config': self.best_config,
            'best_score': self.best_score,
            'results': [asdict(result) for result in self.results],
            'n_proteins': len(self.validation_data)
        }
        
        save_path = Path(save_path)
        
        # Save as JSON for human readability
        json_path = save_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save as pickle for full object preservation
        pickle_path = save_path.with_suffix('.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Saved sweep results to {json_path} and {pickle_path}")
    
    @classmethod
    def load_results(cls, pickle_path: str) -> 'HyperparameterSweep':
        """Load sweep results from pickle file."""
        with open(pickle_path, 'rb') as f:
            sweep = pickle.load(f)
        return sweep


def pocket_optimization_objective(predictions: List[List],
                                 ground_truths: List[List],
                                 evaluator: Optional[PocketEvaluator] = None,
                                 weights: Optional[Dict[str, float]] = None) -> float:
    """
    Pocket-focused optimization objective.
    
    Args:
        predictions: List of predicted pocket lists
        ground_truths: List of ground truth pocket lists
        evaluator: PocketEvaluator instance
        weights: Metric weights for objective
        
    Returns:
        Combined objective score
    """
    if evaluator is None:
        evaluator = PocketEvaluator()
    
    if weights is None:
        weights = {
            'recall_at_1': 0.4,
            'recall_at_3': 0.3,
            'mean_iou': 0.2,
            'f1': 0.1
        }
    
    # Compute metrics
    recall_at_k = pocket_recall_at_k(predictions, ground_truths, [1, 3], evaluator)
    iou_metrics = pocket_iou_metrics(predictions, ground_truths, evaluator)
    
    # Compute overall F1
    total_predicted = sum(len(preds) for preds in predictions)
    total_ground_truth = sum(len(gts) for gts in ground_truths)
    total_matches = 0
    
    for preds, gts in zip(predictions, ground_truths):
        if not preds or not gts:
            continue
        results = evaluator.evaluate_protein(preds, gts)
        total_matches += results['n_matches']
    
    precision = total_matches / (total_predicted + 1e-8)
    recall = total_matches / (total_ground_truth + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Combine metrics
    metrics = {
        'recall_at_1': recall_at_k.get(1, 0),
        'recall_at_3': recall_at_k.get(3, 0),
        'mean_iou': iou_metrics.get('mean_iou', 0),
        'f1': f1
    }
    
    objective = sum(weights.get(k, 0) * v for k, v in metrics.items())
    return objective


def run_validation_sweep(validation_data: List[Dict],
                        sweep_config: Optional[SweepConfig] = None,
                        save_path: Optional[str] = None,
                        max_configs: Optional[int] = None) -> HyperparameterSweep:
    """
    Convenience function to run a complete validation sweep.
    
    Args:
        validation_data: Validation dataset
        sweep_config: Parameter ranges to sweep
        save_path: Path to save results
        max_configs: Maximum configurations to test
        
    Returns:
        HyperparameterSweep object with results
    """
    logger.info("Starting validation hyperparameter sweep")
    
    sweep = HyperparameterSweep(
        validation_data=validation_data,
        sweep_config=sweep_config
    )
    
    results = sweep.run_sweep(max_configs=max_configs, save_path=save_path)
    
    logger.info("Validation sweep completed!")
    logger.info(f"Best configuration: {sweep.get_best_config()}")
    
    # Print top 3 configurations
    logger.info("Top 3 configurations:")
    for i, result in enumerate(sweep.get_top_configs(3)):
        logger.info(f"  {i+1}. Score: {result.combined_score:.4f}")
        logger.info(f"     Config: {result.config}")
        logger.info(f"     Key metrics: R@1={result.metrics.get('recall_at_1', 0):.3f}, "
                   f"IoU={result.metrics.get('mean_iou', 0):.3f}")
    
    return sweep


# Example usage and testing
if __name__ == "__main__":
    # Create mock validation data for testing
    import random
    np.random.seed(42)
    
    def create_mock_protein_data(protein_id: str, n_residues: int = 100) -> Dict:
        """Create mock protein data for testing."""
        
        # Generate random residues
        residues = []
        for i in range(n_residues):
            residue = {
                'chain': 'A',
                'res_id': i + 1,
                'xyz': np.random.randn(3) * 10,
                'rsa': np.random.rand(),
                'prob': np.random.rand()
            }
            residues.append(residue)
        
        # Generate mock ground truth pockets
        from .metrics import GroundTruthPocket
        
        gt_pockets = []
        for pocket_id in range(2):  # 2 pockets per protein
            center = np.random.randn(3) * 10
            pocket_residues = [('A', i) for i in range(pocket_id*20, (pocket_id+1)*20)]
            
            gt_pocket = GroundTruthPocket(
                residues=pocket_residues,
                center=center,
                pocket_id=f"{protein_id}_pocket_{pocket_id}"
            )
            gt_pockets.append(gt_pocket)
        
        return {
            'protein_id': protein_id,
            'residues': residues,
            'ground_truth_pockets': gt_pockets
        }
    
    # Create mock validation dataset
    validation_data = [
        create_mock_protein_data(f"protein_{i}", 100) 
        for i in range(5)  # 5 proteins for testing
    ]
    
    # Test sweep with limited configurations
    sweep_config = SweepConfig(
        graph_radii=[7.0, 8.0],           # Reduced for testing
        min_cluster_sizes=[3, 5],         # Reduced for testing
        sump_thresholds=[2.0],            # Single value for testing
        threshold_strategies=["percentile95"]  # Single strategy for testing
    )
    
    logger.info("Running test hyperparameter sweep...")
    
    sweep = run_validation_sweep(
        validation_data=validation_data,
        sweep_config=sweep_config,
        max_configs=4  # Small number for testing
    )
    
    print("Test sweep completed!")
    print(f"Best config: {sweep.get_best_config()}")
    print(f"Number of results: {len(sweep.results)}")