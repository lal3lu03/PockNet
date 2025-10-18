#!/usr/bin/env python3
"""
End-to-End PockNet Post-Processing Pipeline

This script provides a complete workflow for running PockNet pocket detection
from model loading through final evaluation. It combines:

1. Model inference on H5 data
2. Multi-seed ensembling (optional)
3. Pocket post-processing
4. Performance evaluation
5. Results export

Usage:
    python run_post_processing_pipeline.py --config configs/post_processing/thesis.yaml
    python run_post_processing_pipeline.py --ensemble --output results/
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from post_processing.inference import ModelInference, MultiSeedInference
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import hydra


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('post_processing_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class PockNetPipeline:
    """Complete PockNet post-processing pipeline"""
    
    def __init__(self, config: DictConfig):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        logger.info(f"Initializing PockNet Pipeline on {self.device}")
        logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")
    
    def run_single_model_inference(self, 
                                 checkpoint_path: str,
                                 h5_path: str,
                                 output_prefix: str = "single_model") -> Dict[str, np.ndarray]:
        """
        Run inference with a single model
        
        Args:
            checkpoint_path: Path to model checkpoint
            h5_path: Path to H5 data file
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary of {protein_id: predictions}
        """
        logger.info(f"Starting single model inference")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        logger.info(f"  H5 data: {h5_path}")
        
        try:
            # Initialize model inference
            model_inference = ModelInference(
                checkpoint_path=checkpoint_path,
                device=self.device
            )
            
            # Run predictions
            predictions = model_inference.predict_from_h5(
                h5_file=h5_path,
                protein_ids=self.config.get('protein_ids', None),
                batch_size=self.config.get('batch_size', 32)
            )
            
            logger.info(f"Single model inference completed: {len(predictions)} proteins")
            
            # Save intermediate results
            if self.config.get('save_intermediate', True):
                output_path = f"{output_prefix}_predictions.npz"
                np.savez_compressed(output_path, **predictions)
                logger.info(f"Saved single model predictions to {output_path}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Single model inference failed: {e}")
            raise
    
    def run_ensemble_inference(self,
                             checkpoint_paths: List[str],
                             h5_path: str,
                             output_prefix: str = "ensemble") -> Dict[str, np.ndarray]:
        """
        Run ensemble inference with multiple models
        
        Args:
            checkpoint_paths: List of model checkpoint paths
            h5_path: Path to H5 data file
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary of {protein_id: ensemble_predictions}
        """
        logger.info(f"Starting ensemble inference with {len(checkpoint_paths)} models")
        
        try:
            # Initialize ensemble inference
            ensemble_inference = MultiSeedInference(
                checkpoint_paths=checkpoint_paths,
                device=self.device
            )
            
            # Run ensemble predictions
            predictions = ensemble_inference.predict_ensemble_from_h5(
                h5_file=h5_path,
                protein_ids=self.config.get('protein_ids', None)
            )
            
            logger.info(f"Ensemble inference completed: {len(predictions)} proteins")
            
            # Save intermediate results
            if self.config.get('save_intermediate', True):
                output_path = f"{output_prefix}_predictions.npz"
                np.savez_compressed(output_path, **predictions)
                logger.info(f"Saved ensemble predictions to {output_path}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Ensemble inference failed: {e}")
            raise
    
    def evaluate_predictions(self,
                           predictions: Dict[str, np.ndarray],
                           h5_path: str,
                           output_prefix: str = "evaluation") -> Dict[str, float]:
        """
        Evaluate predictions against ground truth labels
        
        Args:
            predictions: Dictionary of protein predictions
            h5_path: Path to H5 data file with labels
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting prediction evaluation")
        
        try:
            # Load ground truth labels from H5
            import h5py
            
            with h5py.File(h5_path, 'r') as f:
                protein_keys = f['protein_keys'][:].astype(str)
                labels = f['labels'][:]
                
                # Create protein -> labels mapping
                gt_labels = {}
                start_idx = 0
                
                for protein_id in np.unique(protein_keys):
                    protein_mask = protein_keys == protein_id
                    protein_labels = labels[protein_mask]
                    gt_labels[protein_id] = protein_labels
            
            # Calculate metrics for each protein
            protein_metrics = {}
            all_predictions = []
            all_labels = []
            
            for protein_id, pred in predictions.items():
                if protein_id in gt_labels:
                    gt = gt_labels[protein_id]
                    
                    # Ensure same length
                    min_len = min(len(pred), len(gt))
                    pred_truncated = pred[:min_len]
                    gt_truncated = gt[:min_len]
                    
                    # Calculate AUC, AUPRC
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    
                    if len(np.unique(gt_truncated)) > 1:  # Need both classes for AUC
                        auc = roc_auc_score(gt_truncated, pred_truncated)
                        auprc = average_precision_score(gt_truncated, pred_truncated)
                    else:
                        auc = np.nan
                        auprc = np.nan
                    
                    protein_metrics[protein_id] = {
                        'auc': auc,
                        'auprc': auprc,
                        'n_residues': len(pred_truncated),
                        'n_positive': int(np.sum(gt_truncated))
                    }
                    
                    # Add to global arrays
                    all_predictions.extend(pred_truncated)
                    all_labels.extend(gt_truncated)
            
            # Calculate overall metrics
            if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
                overall_auc = roc_auc_score(all_labels, all_predictions)
                overall_auprc = average_precision_score(all_labels, all_predictions)
            else:
                overall_auc = np.nan
                overall_auprc = np.nan
            
            # Aggregate metrics
            valid_aucs = [m['auc'] for m in protein_metrics.values() if not np.isnan(m['auc'])]
            valid_auprcs = [m['auprc'] for m in protein_metrics.values() if not np.isnan(m['auprc'])]
            
            evaluation_results = {
                'overall_auc': overall_auc,
                'overall_auprc': overall_auprc,
                'mean_protein_auc': np.mean(valid_aucs) if valid_aucs else np.nan,
                'mean_protein_auprc': np.mean(valid_auprcs) if valid_auprcs else np.nan,
                'n_proteins_evaluated': len(protein_metrics),
                'total_residues': len(all_labels),
                'total_positive_residues': sum(all_labels)
            }
            
            logger.info(f"Evaluation completed:")
            logger.info(f"  Overall AUC: {evaluation_results['overall_auc']:.4f}")
            logger.info(f"  Overall AUPRC: {evaluation_results['overall_auprc']:.4f}")
            logger.info(f"  Mean protein AUC: {evaluation_results['mean_protein_auc']:.4f}")
            logger.info(f"  Mean protein AUPRC: {evaluation_results['mean_protein_auprc']:.4f}")
            logger.info(f"  Proteins evaluated: {evaluation_results['n_proteins_evaluated']}")
            
            # Save detailed results
            if self.config.get('save_detailed_results', True):
                # Save protein-level metrics
                df_proteins = pd.DataFrame.from_dict(protein_metrics, orient='index')
                protein_results_path = f"{output_prefix}_protein_metrics.csv"
                df_proteins.to_csv(protein_results_path)
                logger.info(f"Saved protein metrics to {protein_results_path}")
                
                # Save overall metrics
                df_overall = pd.DataFrame([evaluation_results])
                overall_results_path = f"{output_prefix}_overall_metrics.csv"
                df_overall.to_csv(overall_results_path, index=False)
                logger.info(f"Saved overall metrics to {overall_results_path}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def run_complete_pipeline(self) -> Dict:
        """
        Run the complete post-processing pipeline
        
        Returns:
            Dictionary with all results
        """
        logger.info("=" * 60)
        logger.info("STARTING POCKNET POST-PROCESSING PIPELINE")
        logger.info("=" * 60)
        
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'config': OmegaConf.to_container(self.config),
                'device': str(self.device)
            }
            
            # Determine run mode
            run_ensemble = self.config.get('ensemble', {}).get('enabled', False)
            h5_path = self.config.data.h5_file
            
            if run_ensemble:
                logger.info("Running in ENSEMBLE mode")
                checkpoint_paths = self.config.model.checkpoint_paths
                
                # Run ensemble inference
                predictions = self.run_ensemble_inference(
                    checkpoint_paths=checkpoint_paths,
                    h5_path=h5_path,
                    output_prefix="ensemble"
                )
                results['mode'] = 'ensemble'
                results['n_models'] = len(checkpoint_paths)
                
            else:
                logger.info("Running in SINGLE MODEL mode")
                checkpoint_path = self.config.model.checkpoint_paths[0]  # Use first checkpoint
                
                # Run single model inference
                predictions = self.run_single_model_inference(
                    checkpoint_path=checkpoint_path,
                    h5_path=h5_path,
                    output_prefix="single_model"
                )
                results['mode'] = 'single_model'
                results['n_models'] = 1
            
            # Store predictions
            results['predictions'] = predictions
            results['n_proteins'] = len(predictions)
            
            # Evaluate predictions
            if self.config.get('evaluation', {}).get('enabled', True):
                logger.info("Running evaluation")
                evaluation_results = self.evaluate_predictions(
                    predictions=predictions,
                    h5_path=h5_path,
                    output_prefix=results['mode']
                )
                results['evaluation'] = evaluation_results
            
            # Export final results
            if self.config.get('export', {}).get('enabled', True):
                self.export_results(results)
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def export_results(self, results: Dict):
        """Export pipeline results to files"""
        logger.info("Exporting results")
        
        try:
            # Create output directory
            output_dir = Path(self.config.get('output_dir', 'results'))
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export predictions
            predictions_path = output_dir / f"predictions_{timestamp}.npz"
            np.savez_compressed(predictions_path, **results['predictions'])
            logger.info(f"Exported predictions to {predictions_path}")
            
            # Export summary
            summary = {k: v for k, v in results.items() if k != 'predictions'}
            summary_path = output_dir / f"summary_{timestamp}.yaml"
            OmegaConf.save(summary, summary_path)
            logger.info(f"Exported summary to {summary_path}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="PockNet Post-Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model with default config
  python run_post_processing_pipeline.py
  
  # Ensemble mode
  python run_post_processing_pipeline.py --ensemble
  
  # Custom config file
  python run_post_processing_pipeline.py --config custom_config.yaml
  
  # Specify output directory
  python run_post_processing_pipeline.py --output results/experiment_1/
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/post_processing/thesis.yaml',
        help='Path to configuration file (default: configs/post_processing/thesis.yaml)'
    )
    
    parser.add_argument(
        '--ensemble', '-e',
        action='store_true',
        help='Run ensemble inference with multiple models'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--limit-proteins',
        type=int,
        help='Limit number of proteins to process (for testing)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Batch size for inference (default: 1024)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use for inference (default: auto)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


@hydra.main(version_base=None, config_path="configs/post_processing", config_name="thesis")
def main(cfg: DictConfig) -> None:
    """Main entry point using Hydra"""
    
    # Parse command line arguments
    args = parse_args()
    
    # Update config with command line overrides
    if args.ensemble:
        cfg.ensemble.enabled = True
    
    if args.output:
        cfg.output_dir = args.output
    
    if args.limit_proteins:
        cfg.limit_proteins = args.limit_proteins
    
    if args.batch_size:
        cfg.batch_size = args.batch_size
    
    if args.device != 'auto':
        cfg.device = args.device
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run pipeline
    pipeline = PockNetPipeline(cfg)
    results = pipeline.run_complete_pipeline()
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Mode: {results['mode']}")
    print(f"Models used: {results['n_models']}")
    print(f"Proteins processed: {results['n_proteins']}")
    print(f"Device: {results['device']}")
    
    if 'evaluation' in results:
        eval_results = results['evaluation']
        print(f"\nEvaluation Results:")
        print(f"  Overall AUC: {eval_results.get('overall_auc', 'N/A'):.4f}")
        print(f"  Overall AUPRC: {eval_results.get('overall_auprc', 'N/A'):.4f}")
        print(f"  Mean Protein AUC: {eval_results.get('mean_protein_auc', 'N/A'):.4f}")
        print(f"  Mean Protein AUPRC: {eval_results.get('mean_protein_auprc', 'N/A'):.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()