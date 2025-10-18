#!/usr/bin/env python3
"""
Example Usage: PockNet Post-Processing Pipeline
==============================================

This script demonstrates how to use the post-processing pipeline with the best
performing model from the notebook analysis. It shows the complete workflow from
model loading to pocket prediction and evaluation.

Usage:
    python example_usage.py --checkpoint_path /path/to/best_model.ckpt
    python example_usage.py --config config.json

Author: PockNet Team
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from post_processing import (
    residue_to_pockets,
    ensemble_predictions, 
    PocketEvaluator,
    ModelInference,
    run_validation_sweep
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def example_single_seed_inference(checkpoint_path: str, 
                                 data_file: str,
                                 output_dir: str):
    """
    Example 1: Single seed model inference and post-processing.
    """
    logger.info("=== Example 1: Single Seed Inference ===")
    
    # Step 1: Load model (you need to provide your model class)
    logger.info(f"Loading model from {checkpoint_path}")
    
    # TODO: Replace with your actual model class
    # model_inference = ModelInference(checkpoint_path, model_class=YourModelClass)
    
    # For demonstration, we'll create mock predictions
    logger.info("Creating mock predictions for demonstration...")
    predictions = create_mock_predictions()
    
    # Step 2: Prepare residue data
    logger.info("Preparing residue data...")
    residues = prepare_mock_residue_data(predictions)
    
    # Step 3: Post-process to get pockets
    logger.info("Converting residues to pockets...")
    pockets = residue_to_pockets(
        residues,
        rsa_min=0.2,
        graph_radius=8.0,
        min_cluster_size=5,
        min_sump=2.0,
        max_pockets=5
    )
    
    logger.info(f"Found {len(pockets)} pockets")
    for i, pocket in enumerate(pockets):
        logger.info(f"  Pocket {i+1}: size={pocket.size}, score={pocket.score:.3f}")
    
    # Step 4: Save results
    output_file = os.path.join(output_dir, "single_seed_pockets.json")
    save_pockets_to_json(pockets, output_file)
    logger.info(f"Results saved to {output_file}")

def example_multi_seed_ensemble(checkpoint_paths: List[str],
                               data_file: str, 
                               output_dir: str):
    """
    Example 2: Multi-seed ensemble inference and post-processing.
    """
    logger.info("=== Example 2: Multi-Seed Ensemble ===")
    
    # Step 1: Generate predictions from multiple seeds
    logger.info(f"Processing {len(checkpoint_paths)} seed models...")
    
    seed_predictions = []
    for i, checkpoint_path in enumerate(checkpoint_paths):
        logger.info(f"Processing seed {i+1}: {checkpoint_path}")
        
        # TODO: Replace with actual model inference
        # model_inference = ModelInference(checkpoint_path, model_class=YourModelClass)
        # predictions = model_inference.predict_from_h5(data_file)
        
        # For demonstration, create mock predictions
        predictions = create_mock_predictions(seed=i)
        seed_predictions.append(predictions)
    
    # Step 2: Ensemble predictions
    logger.info("Ensembling predictions across seeds...")
    ensemble = ensemble_predictions(seed_predictions, method="mean")
    
    logger.info(f"Ensemble stats: mean={ensemble.mean():.4f}, std={ensemble.std():.4f}")
    
    # Step 3: Prepare residue data with ensemble predictions
    residues = prepare_mock_residue_data(ensemble)
    
    # Step 4: Post-process ensemble predictions
    logger.info("Post-processing ensemble predictions...")
    pockets = residue_to_pockets(
        residues,
        rsa_min=0.2,
        graph_radius=8.0,
        min_cluster_size=5,
        min_sump=2.0,
        max_pockets=5
    )
    
    logger.info(f"Ensemble found {len(pockets)} pockets")
    for i, pocket in enumerate(pockets):
        logger.info(f"  Pocket {i+1}: size={pocket.size}, score={pocket.score:.3f}")
    
    # Step 5: Save ensemble results
    output_file = os.path.join(output_dir, "ensemble_pockets.json")
    save_pockets_to_json(pockets, output_file)
    logger.info(f"Ensemble results saved to {output_file}")

def example_hyperparameter_optimization(validation_data: List[Dict],
                                      output_dir: str):
    """
    Example 3: Hyperparameter optimization on validation data.
    """
    logger.info("=== Example 3: Hyperparameter Optimization ===")
    
    # Run hyperparameter sweep
    logger.info("Running hyperparameter sweep...")
    
    sweep = run_validation_sweep(
        validation_data=validation_data,
        save_path=os.path.join(output_dir, "hyperparameter_sweep"),
        max_configs=12  # Limit for demonstration
    )
    
    best_config = sweep.get_best_config()
    logger.info(f"Best configuration found: {best_config}")
    
    # Apply best configuration
    logger.info("Applying best configuration to test data...")
    
    # Create test predictions
    test_predictions = create_mock_predictions(seed=42)
    test_residues = prepare_mock_residue_data(test_predictions)
    
    # Post-process with optimized parameters
    optimized_pockets = residue_to_pockets(
        test_residues,
        **best_config
    )
    
    logger.info(f"Optimized post-processing found {len(optimized_pockets)} pockets")
    
    # Save optimized results
    output_file = os.path.join(output_dir, "optimized_pockets.json")
    save_pockets_to_json(optimized_pockets, output_file)

def example_evaluation_metrics(predicted_pockets, ground_truth_pockets):
    """
    Example 4: Comprehensive evaluation with pocket-level metrics.
    """
    logger.info("=== Example 4: Evaluation Metrics ===")
    
    # Initialize evaluator
    evaluator = PocketEvaluator(
        center_distance_threshold=6.0,
        residue_overlap_threshold=1,
        spatial_distance_threshold=4.0
    )
    
    # Evaluate single protein
    results = evaluator.evaluate_protein(predicted_pockets, ground_truth_pockets)
    
    logger.info("Evaluation Results:")
    logger.info(f"  Precision: {results['precision']:.3f}")
    logger.info(f"  Recall: {results['recall']:.3f}")
    logger.info(f"  F1-Score: {results['f1']:.3f}")
    logger.info(f"  Average IoU: {results['avg_iou']:.3f}")
    logger.info(f"  Average Center Distance: {results['avg_center_distance']:.2f} Å")
    logger.info(f"  Matches: {results['n_matches']}/{results['n_predicted']} predicted")

def create_mock_predictions(n_residues: int = 500, seed: int = 42) -> np.ndarray:
    """Create mock prediction data for demonstration."""
    np.random.seed(seed)
    
    # Create realistic-looking predictions with some high-scoring regions
    base_noise = np.random.rand(n_residues) * 0.3  # Low background
    
    # Add some high-scoring pocket regions
    n_pockets = 3
    for i in range(n_pockets):
        center = np.random.randint(50, n_residues - 50)
        width = np.random.randint(15, 30)
        
        # Create Gaussian-like pocket scores
        indices = np.arange(max(0, center - width), min(n_residues, center + width))
        distances = np.abs(indices - center)
        scores = 0.8 * np.exp(-distances**2 / (2 * (width/3)**2))
        
        base_noise[indices] = np.maximum(base_noise[indices], scores)
    
    return np.clip(base_noise, 0, 1)

def prepare_mock_residue_data(predictions: np.ndarray) -> List[Dict]:
    """Prepare mock residue data with coordinates and RSA values."""
    n_residues = len(predictions)
    
    residues = []
    for i in range(n_residues):
        # Generate protein-like 3D structure (linear with some noise)
        base_coord = np.array([i * 3.8, 0, 0])  # ~3.8Å between Cα atoms
        noise = np.random.randn(3) * 1.0
        coord = base_coord + noise
        
        # Generate realistic RSA values (more surface residues at ends)
        distance_from_center = abs(i - n_residues/2) / (n_residues/2)
        rsa = 0.1 + 0.7 * distance_from_center + np.random.rand() * 0.2
        rsa = np.clip(rsa, 0, 1)
        
        residue = {
            'chain': 'A',
            'res_id': i + 1,
            'xyz': coord,
            'rsa': rsa,
            'prob': predictions[i],
            'res_name': 'ALA'  # Placeholder
        }
        residues.append(residue)
    
    return residues

def create_mock_validation_data(n_proteins: int = 5) -> List[Dict]:
    """Create mock validation data for hyperparameter optimization."""
    from post_processing.metrics import GroundTruthPocket
    
    validation_data = []
    
    for protein_idx in range(n_proteins):
        # Create mock predictions and residues
        predictions = create_mock_predictions(seed=protein_idx)
        residues = prepare_mock_residue_data(predictions)
        
        # Create mock ground truth pockets
        gt_pockets = []
        n_gt_pockets = np.random.randint(1, 4)  # 1-3 pockets per protein
        
        for pocket_idx in range(n_gt_pockets):
            # Random pocket location
            center_residue = np.random.randint(50, len(residues) - 50)
            pocket_size = np.random.randint(10, 30)
            
            # Define pocket residues
            start_idx = max(0, center_residue - pocket_size // 2)
            end_idx = min(len(residues), center_residue + pocket_size // 2)
            pocket_residues = [('A', i) for i in range(start_idx, end_idx)]
            
            # Pocket center
            pocket_center = np.mean([residues[i]['xyz'] for i in range(start_idx, end_idx)], axis=0)
            
            gt_pocket = GroundTruthPocket(
                residues=pocket_residues,
                center=pocket_center,
                pocket_id=f"protein_{protein_idx}_pocket_{pocket_idx}"
            )
            gt_pockets.append(gt_pocket)
        
        protein_data = {
            'protein_id': f"protein_{protein_idx}",
            'residues': residues,
            'ground_truth_pockets': gt_pockets
        }
        validation_data.append(protein_data)
    
    return validation_data

def save_pockets_to_json(pockets: List, output_file: str):
    """Save pocket predictions to JSON file."""
    # Convert pockets to serializable format
    serializable_pockets = []
    
    for pocket in pockets:
        if hasattr(pocket, '__dict__'):
            # Convert dataclass to dict
            pocket_dict = pocket.__dict__.copy()
        else:
            pocket_dict = pocket.copy()
        
        # Convert numpy arrays to lists
        if 'center' in pocket_dict and isinstance(pocket_dict['center'], np.ndarray):
            pocket_dict['center'] = pocket_dict['center'].tolist()
        
        serializable_pockets.append(pocket_dict)
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(serializable_pockets, f, indent=2)

def main():
    """Main function demonstrating the complete post-processing workflow."""
    parser = argparse.ArgumentParser(description="PockNet Post-Processing Examples")
    parser.add_argument("--checkpoint_path", type=str, 
                       help="Path to best model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str,
                       help="Directory containing multiple seed checkpoints")
    parser.add_argument("--data_file", type=str,
                       help="Path to H5 data file")
    parser.add_argument("--output_dir", type=str, default="./post_processing_results",
                       help="Output directory for results")
    parser.add_argument("--config", type=str,
                       help="JSON configuration file")
    parser.add_argument("--example", type=str, choices=["single", "ensemble", "optimize", "all"],
                       default="all", help="Which example to run")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting PockNet Post-Processing Examples")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    
    try:
        if args.example in ["single", "all"]:
            # Example 1: Single seed inference
            if args.checkpoint_path:
                example_single_seed_inference(
                    args.checkpoint_path,
                    args.data_file or "mock_data.h5",
                    args.output_dir
                )
            else:
                logger.info("Skipping single seed example (no checkpoint_path provided)")
        
        if args.example in ["ensemble", "all"]:
            # Example 2: Multi-seed ensemble
            if args.checkpoint_dir:
                # Find all checkpoints in directory
                checkpoint_paths = [
                    os.path.join(args.checkpoint_dir, f) 
                    for f in os.listdir(args.checkpoint_dir)
                    if f.endswith('.ckpt')
                ]
                
                if checkpoint_paths:
                    example_multi_seed_ensemble(
                        checkpoint_paths[:7],  # Use up to 7 seeds
                        args.data_file or "mock_data.h5",
                        args.output_dir
                    )
                else:
                    logger.warning("No checkpoint files found in directory")
            else:
                logger.info("Skipping ensemble example (no checkpoint_dir provided)")
        
        if args.example in ["optimize", "all"]:
            # Example 3: Hyperparameter optimization
            logger.info("Creating mock validation data for optimization example...")
            validation_data = create_mock_validation_data(n_proteins=5)
            
            example_hyperparameter_optimization(
                validation_data,
                args.output_dir
            )
        
        # Example 4: Evaluation (always run as demonstration)
        logger.info("Running evaluation metrics example...")
        
        # Create mock data for evaluation
        from post_processing.metrics import PredictedPocket, GroundTruthPocket
        
        predicted_pockets = [
            PredictedPocket(
                residues=[('A', i) for i in range(10, 25)],
                center=np.array([0, 0, 0]),
                score=0.8
            ),
            PredictedPocket(
                residues=[('A', i) for i in range(50, 60)],
                center=np.array([10, 0, 0]),
                score=0.6
            )
        ]
        
        ground_truth_pockets = [
            GroundTruthPocket(
                residues=[('A', i) for i in range(12, 28)],
                center=np.array([1, 1, 0]),
                pocket_id="GT1"
            ),
            GroundTruthPocket(
                residues=[('A', i) for i in range(80, 95)],
                center=np.array([20, 0, 0]),
                pocket_id="GT2"
            )
        ]
        
        example_evaluation_metrics(predicted_pockets, ground_truth_pockets)
        
        logger.info("All examples completed successfully!")
        logger.info(f"Results saved in: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise

if __name__ == "__main__":
    main()