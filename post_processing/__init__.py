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

import warnings

try:
    from .core import (
        residue_to_pockets,
        make_residue_graph,
        postprocess_residues,
        PocketPostProcessor,
    )
except ImportError as exc:  # pragma: no cover - optional dependency (scikit-learn)
    residue_to_pockets = make_residue_graph = postprocess_residues = PocketPostProcessor = None  # type: ignore[assignment]
    warnings.warn(f"post_processing.core unavailable ({exc}); core pocket utilities disabled.", RuntimeWarning)

try:
    from .ensemble import (
        ensemble_predictions,
        spatial_smoothing,
        adaptive_threshold,
        calibrate_predictions,
    )
except ImportError as exc:  # pragma: no cover
    ensemble_predictions = spatial_smoothing = adaptive_threshold = calibrate_predictions = None  # type: ignore[assignment]
    warnings.warn(f"post_processing.ensemble unavailable ({exc}); ensemble utilities disabled.", RuntimeWarning)

try:
    from .metrics import (
        PocketEvaluator,
        pocket_recall_at_k,
        center_distance_metrics,
        pocket_iou_metrics,
        pocket_pr_curve,
    )
except ImportError as exc:  # pragma: no cover
    PocketEvaluator = pocket_recall_at_k = center_distance_metrics = pocket_iou_metrics = pocket_pr_curve = None  # type: ignore[assignment]
    warnings.warn(f"post_processing.metrics unavailable ({exc}); metrics utilities disabled.", RuntimeWarning)

try:
    from .inference import (
        ModelInference,
        load_checkpoint,
        extract_predictions,
        prepare_residue_data,
    )
except ImportError as exc:  # pragma: no cover
    ModelInference = load_checkpoint = extract_predictions = prepare_residue_data = None  # type: ignore[assignment]
    warnings.warn(f"post_processing.inference unavailable ({exc}); inference helpers disabled.", RuntimeWarning)

try:
    from .sweep import (
        HyperparameterSweep,
        pocket_optimization_objective,
        run_validation_sweep,
    )
except ImportError as exc:  # pragma: no cover
    HyperparameterSweep = pocket_optimization_objective = run_validation_sweep = None  # type: ignore[assignment]
    warnings.warn(f"post_processing.sweep unavailable ({exc}); sweep utilities disabled.", RuntimeWarning)

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
