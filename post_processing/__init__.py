"""
PockNet Post-Processing Pipeline
================================

This module provides comprehensive post-processing functionality for converting
residue-level predictions to pocket-level predictions using spatial clustering,
surface filtering, and ensemble techniques.

Main Components:
- residue_to_pockets: Core conversion pipeline
- ensemble: Multi-seed ensembling and smoothing
- metrics: Pocket-level evaluation metrics
- inference: Model loading and prediction extraction
- sweep: Hyperparameter optimization

Usage:
    from post_processing import residue_to_pockets, ensemble_predictions
    
    # Convert residue scores to pockets
    pockets = residue_to_pockets(residues, threshold=0.5)
    
    # Ensemble multiple seeds
    ensemble_probs = ensemble_predictions(seed_predictions)
"""

from .core import (
    residue_to_pockets,
    make_residue_graph,
    postprocess_residues,
    PocketPostProcessor
)

from .ensemble import (
    ensemble_predictions,
    spatial_smoothing,
    adaptive_threshold,
    calibrate_predictions
)

from .metrics import (
    PocketEvaluator,
    pocket_recall_at_k,
    center_distance_metrics,
    pocket_iou_metrics,
    pocket_pr_curve
)

from .inference import (
    ModelInference,
    load_checkpoint,
    extract_predictions,
    prepare_residue_data
)

from .sweep import (
    HyperparameterSweep,
    pocket_optimization_objective,
    run_validation_sweep
)

__version__ = "1.0.0"
__author__ = "PockNet Team"

__all__ = [
    # Core functionality
    'residue_to_pockets',
    'make_residue_graph', 
    'postprocess_residues',
    'PocketPostProcessor',
    
    # Ensemble techniques
    'ensemble_predictions',
    'spatial_smoothing',
    'adaptive_threshold',
    'calibrate_predictions',
    
    # Evaluation metrics
    'PocketEvaluator',
    'pocket_recall_at_k',
    'center_distance_metrics',
    'pocket_iou_metrics',
    'pocket_pr_curve',
    
    # Model inference
    'ModelInference',
    'load_checkpoint',
    'extract_predictions',
    'prepare_residue_data',
    
    # Hyperparameter optimization
    'HyperparameterSweep',
    'pocket_optimization_objective',
    'run_validation_sweep'
]