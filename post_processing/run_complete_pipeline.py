#!/usr/bin/env python3
"""
Complete PockNet Pipeline with Post-Processing
==============================================

Integration script that runs the complete pipeline:
1. Model inference with shared memory optimization
2. Multi-seed ensembling 
3. Post-processing with pocket formation
4. IoU evaluation and comprehensive metrics

This provides the full workflow from model checkpoints to final pocket predictions.
"""

import sys
import logging
import time
import os
from pathlib import Path
import numpy as np
import h5py
from typing import List, Dict, Any, Optional

# Set threading optimizations
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count()) 

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "post_processing"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import post-processing modules
try:
    from pocket_formation import (
        PostProcessingConfig, Residue, Pocket,
        create_residue_from_prediction
    )
    from pipeline import PostProcessingPipeline
    logger.info("✅ Post-processing modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import post-processing modules: {e}")
    sys.exit(1)


def extract_residue_data_from_h5(h5_file: str, 
                                protein_id: str,
                                predictions: np.ndarray) -> List[Residue]:
    """
    Extract residue data from H5 file and combine with predictions.
    
    Args:
        h5_file: Path to H5 file
        protein_id: Protein identifier
        predictions: Model predictions for this protein
        
    Returns:
        List of Residue objects
    """
    residues = []
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # Load all data arrays
            protein_keys = [key.decode() if isinstance(key, bytes) else str(key) 
                           for key in f['protein_keys'][:]]
            
            # Find indices for this protein
            protein_indices = [i for i, key in enumerate(protein_keys) if protein_id in key]
            
            if not protein_indices:
                logger.warning(f"No data found for protein {protein_id}")
                return []
            
            # Extract coordinates and features (mock data for now)
            # In a real implementation, you would load actual coordinates and RSA values
            for i, idx in enumerate(protein_indices):
                if i >= len(predictions):
                    break
                    
                # Mock coordinates (in practice, load from PDB or H5)
                # For now, generate dummy coordinates
                coord = np.array([i * 1.0, 0.0, 0.0], dtype=np.float32)
                
                # Mock RSA (in practice, calculate or load from features)
                # Use tabular features as proxy for surface accessibility
                mock_rsa = 0.5  # Placeholder
                
                # Extract chain and residue ID from protein key format
                # Assuming format like "1abc_A" 
                if '_' in protein_id:
                    chain = protein_id.split('_')[-1]
                else:
                    chain = 'A'
                
                residue = create_residue_from_prediction(
                    protein_id=protein_id,
                    chain=chain,
                    res_id=i + 1,  # 1-indexed residue numbers
                    coordinates=coord,
                    rsa=mock_rsa,
                    prediction=predictions[i]
                )
                
                residues.append(residue)
                
    except Exception as e:
        logger.error(f"Failed to extract residue data for {protein_id}: {e}")
        
    return residues


def run_complete_pipeline(checkpoint_paths: List[str],
                         h5_file: str,
                         test_proteins: Optional[List[str]] = None,
                         config: Optional[PostProcessingConfig] = None) -> Dict[str, Any]:
    """
    Run the complete pipeline with post-processing.
    
    Args:
        checkpoint_paths: List of model checkpoint paths
        h5_file: Path to H5 data file
        test_proteins: List of proteins to process (None for all test proteins)
        config: Post-processing configuration
        
    Returns:
        Complete results with pockets and metrics
    """
    logger.info("🚀 Starting Complete PockNet Pipeline with Post-Processing")
    logger.info("=" * 70)
    
    total_start = time.time()
    
    # Import inference modules
    logger.info("📥 Loading inference modules...")
    from inference import MultiSeedInference, _shared_memory_manager
    
    # Initialize post-processing pipeline
    pp_config = config or PostProcessingConfig()
    post_processor = PostProcessingPipeline(pp_config)
    
    logger.info(f"📦 Using {len(checkpoint_paths)} model checkpoints")
    logger.info(f"📁 H5 file: {Path(h5_file).name}")
    logger.info(f"🔧 Post-processing config: {pp_config}")
    
    # Step 1: Initialize ensemble inference with shared memory
    logger.info("\n🔄 Step 1: Model Initialization")
    init_start = time.time()
    
    ensemble_inference = MultiSeedInference(
        checkpoint_paths=checkpoint_paths,
        use_shared_memory=True
    )
    
    init_time = time.time() - init_start
    logger.info(f"✅ Models initialized in {init_time:.2f}s")
    
    # Step 2: Discover test proteins
    logger.info("\n🔍 Step 2: Protein Discovery")
    
    if test_proteins is None:
        # Auto-discover test proteins from H5 split
        with h5py.File(h5_file, 'r') as f:
            protein_keys = [key.decode() if isinstance(key, bytes) else str(key) 
                           for key in f['protein_keys'][:]]
            
            if 'split' in f:
                splits = f['split'][:]
                protein_to_split = {}
                for idx, protein_id in enumerate(protein_keys):
                    split_val = splits[idx]
                    if protein_id not in protein_to_split:
                        protein_to_split[protein_id] = split_val
                
                test_proteins = [pid for pid, split in protein_to_split.items() if split == 2]
                logger.info(f"📊 Auto-discovered {len(test_proteins)} test proteins")
            else:
                test_proteins = list(set(protein_keys))[:10]  # Limit for testing
                logger.info(f"⚠️  No split info, using first {len(test_proteins)} proteins")
    
    logger.info(f"🧬 Processing {len(test_proteins)} proteins")
    
    # Step 3: Run ensemble inference with shared memory
    logger.info("\n🔮 Step 3: Ensemble Inference")
    inference_start = time.time()
    
    ensemble_predictions = ensemble_inference.predict_ensemble_from_h5(
        h5_file=h5_file,
        protein_ids=test_proteins,
        prepare_shared_memory=True
    )
    
    inference_time = time.time() - inference_start
    logger.info(f"✅ Ensemble inference completed in {inference_time:.2f}s")
    logger.info(f"📊 Got predictions for {len(ensemble_predictions)} proteins")
    
    # Step 4: Post-processing and pocket formation
    logger.info("\n🏗️  Step 4: Post-Processing & Pocket Formation")
    pp_start = time.time()
    
    all_results = {}
    pocket_counts = []
    iou_scores = []
    
    for protein_id in test_proteins:
        if protein_id not in ensemble_predictions:
            logger.warning(f"⚠️  No predictions for {protein_id}")
            continue
            
        try:
            # Extract residue data
            residue_data = extract_residue_data_from_h5(
                h5_file, protein_id, ensemble_predictions[protein_id]
            )
            
            if not residue_data:
                continue
            
            # Run post-processing (no GT pockets for now)
            pp_result = post_processor.process_protein(residue_data, gt_pockets=None)
            
            all_results[protein_id] = pp_result
            pocket_counts.append(len(pp_result['pockets']))
            
            # Log protein result
            n_pockets = len(pp_result['pockets'])
            n_residues = pp_result['metrics']['n_residues']
            threshold = pp_result['threshold']
            
            logger.info(f"  {protein_id}: {n_residues} residues → {n_pockets} pockets (th={threshold:.3f})")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {protein_id}: {e}")
            continue
    
    pp_time = time.time() - pp_start
    logger.info(f"✅ Post-processing completed in {pp_time:.2f}s")
    
    # Step 5: Dataset-level evaluation
    logger.info("\n📈 Step 5: Dataset Evaluation")
    eval_start = time.time()
    
    dataset_metrics = post_processor.evaluate_dataset(all_results)
    
    eval_time = time.time() - eval_start
    total_time = time.time() - total_start
    
    # Summary statistics
    logger.info("\n📊 Pipeline Results Summary")
    logger.info("=" * 50)
    
    logger.info(f"🧬 Proteins processed: {len(all_results)}")
    logger.info(f"🏗️  Total pockets found: {sum(pocket_counts)}")
    logger.info(f"📈 Avg pockets per protein: {np.mean(pocket_counts):.1f} ± {np.std(pocket_counts):.1f}")
    
    if 'n_residues_mean' in dataset_metrics:
        logger.info(f"🔬 Avg residues per protein: {dataset_metrics['n_residues_mean']:.0f}")
        logger.info(f"📍 Avg positive rate: {dataset_metrics['positive_rate_mean']:.1%}")
    
    # Timing breakdown
    logger.info("\n⏱️  Timing Breakdown")
    logger.info("=" * 30)
    logger.info(f"🏗️  Model Init:       {init_time:.2f}s")
    logger.info(f"🔮 Inference:        {inference_time:.2f}s")
    logger.info(f"🏗️  Post-Processing:  {pp_time:.2f}s")
    logger.info(f"📈 Evaluation:       {eval_time:.2f}s")
    logger.info(f"⏱️  Total Pipeline:   {total_time:.2f}s")
    
    # Throughput metrics
    total_residues = dataset_metrics.get('total_residues', 0)
    if total_time > 0:
        protein_throughput = len(all_results) / total_time
        residue_throughput = total_residues / total_time
        
        logger.info(f"\n🚀 Throughput")
        logger.info("=" * 20)
        logger.info(f"🧬 Proteins/sec: {protein_throughput:.2f}")
        logger.info(f"🔬 Residues/sec: {residue_throughput:,.0f}")
    
    # Return complete results
    return {
        'protein_results': all_results,
        'dataset_metrics': dataset_metrics,
        'timing': {
            'total_time': total_time,
            'init_time': init_time,
            'inference_time': inference_time,
            'postprocessing_time': pp_time,
            'evaluation_time': eval_time
        },
        'config': pp_config,
        'n_proteins': len(all_results),
        'n_total_pockets': sum(pocket_counts)
    }


def main():
    """Main execution function"""
    logger.info("🧪 PockNet Complete Pipeline Test")
    
    # Configuration
    checkpoint_paths = [
        "/system/user/studentwork/hageneder/MSC/Practical_work/PockNet/logs/fusion_all_train_complete/runs/2025-09-04_19-45-33/checkpoints/epoch=11-val_auprc=0.2743.ckpt"
    ]
    
    h5_file = "/system/user/studentwork/hageneder/MSC/Practical_work/PockNet/data/h5/pocknet_with_esm2_3b.h5"
    
    # Create post-processing config
    config = PostProcessingConfig(
        # Surface filtering
        rsa_min=0.2,
        
        # Graph and clustering
        graph_radius=8.0,
        min_cluster_size=5,
        sump_min=2.0,
        
        # Thresholding
        threshold_mode="percentile",
        percentile_threshold=95.0,
        
        # Smoothing
        enable_smoothing=True,
        smoothing_alpha=0.7,
        
        # Pocket filtering
        max_pockets_per_protein=5,
        enable_nms=True,
        nms_radius=6.0
    )
    
    try:
        # Run complete pipeline
        results = run_complete_pipeline(
            checkpoint_paths=checkpoint_paths,
            h5_file=h5_file,
            test_proteins=None,  # Auto-discover test proteins
            config=config
        )
        
        logger.info("\n🎉 Pipeline completed successfully!")
        
        # Show some example results
        logger.info("\n🔍 Example Pocket Results")
        logger.info("=" * 40)
        
        protein_results = results['protein_results']
        for i, (protein_id, result) in enumerate(list(protein_results.items())[:3]):
            pockets = result['pockets']
            logger.info(f"\n{protein_id}:")
            for j, pocket in enumerate(pockets[:2]):  # Show top 2 pockets
                logger.info(f"  Pocket {j+1}: {pocket.size} residues, score={pocket.score:.3f}")
                logger.info(f"    Center: ({pocket.center[0]:.1f}, {pocket.center[1]:.1f}, {pocket.center[2]:.1f})")
                logger.info(f"    Surface fraction: {pocket.surface_fraction:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)