#!/usr/bin/env python3
"""
Enhanced Post-Processing with PyMOL Visualization and Structured Results
========================================================================

Complete post-processing system that:
✅ Uses shared memory for optimal performance  
✅ Generates PyMOL scripts for visualization
✅ Creates structured results per protein
✅ Shows true vs predicted pockets in metrics
✅ Comprehensive evaluation with IoU and AUPRC
✅ All results organized in post_processing_results/

Directory structure:
post_processing_results/
├── summary/                    # Overall summary
│   ├── summary_report.txt      # Main results
│   ├── all_pockets.csv        # Combined results
│   └── performance_metrics.json
├── protein_1a4j_H/            # Individual protein results
│   ├── pockets.csv            # Pocket details
│   ├── visualization.pml      # PyMOL script
│   ├── predictions.txt        # Raw predictions
│   ├── true_pockets.txt       # Ground truth
│   └── metrics.json           # Protein-specific metrics
└── protein_1a6u_H/           # Another protein...
    └── ...
"""

import sys
import logging
import time
import os
import json
import shutil
from pathlib import Path
import numpy as np
import h5py
import yaml
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict
import csv

# Try to import BioPython 
try:
    from Bio.PDB import PDBParser
    BIOPYTHON_AVAILABLE = True
except ImportError:
    print("Warning: BioPython not available - PDB coordinate extraction disabled")
    BIOPYTHON_AVAILABLE = False
    PDBParser = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set optimized threading
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Try to import modules with fallbacks
try:
    from post_processing.evaluation import PostProcessingEvaluator  # type: ignore
    EVALUATION_AVAILABLE = True
except Exception:
    try:
        from evaluation import PostProcessingEvaluator  # type: ignore
        EVALUATION_AVAILABLE = True
    except Exception:
        EVALUATION_AVAILABLE = False
        logger.warning("Evaluation module not available")

try:
    from post_processing.inference import (  # type: ignore
        ModelInference,
        MultiSeedInference,
        _shared_memory_manager,
    )
    INFERENCE_AVAILABLE = True
except Exception:
    try:
        from inference import ModelInference, MultiSeedInference, _shared_memory_manager  # type: ignore
        INFERENCE_AVAILABLE = True
    except Exception:
        INFERENCE_AVAILABLE = False
        logger.warning("Advanced inference modules not available")

try:
    from post_processing.production_pipeline import (  # type: ignore
        ProductionPostProcessor,
        ProductionConfig,
        AdvancedPocket,
        StructuralDataLoader,
    )
    PRODUCTION_AVAILABLE = True
except Exception:
    try:
        from production_pipeline import (  # type: ignore
            ProductionPostProcessor,
            ProductionConfig,
            AdvancedPocket,
            StructuralDataLoader,
        )
        PRODUCTION_AVAILABLE = True
    except Exception:
        PRODUCTION_AVAILABLE = False
        logger.warning("Production pipeline components not available")

# Scientific computing with fallbacks
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available - using fallback neighbor finding")

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - using fallback clustering")

# Fallback implementations
class FallbackDBSCAN:
    """Simple fallback clustering when sklearn not available"""
    def __init__(self, eps=6.0, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def fit(self, X):
        n_points = len(X)
        self.labels_ = np.full(n_points, -1)
        current_label = 0
        
        for i in range(n_points):
            if self.labels_[i] != -1:
                continue
                
            neighbors = []
            for j in range(n_points):
                if i != j:
                    dist = np.linalg.norm(X[i] - X[j])
                    if dist <= self.eps:
                        neighbors.append(j)
            
            if len(neighbors) >= self.min_samples - 1:
                self.labels_[i] = current_label
                for neighbor in neighbors:
                    if self.labels_[neighbor] == -1:
                        self.labels_[neighbor] = current_label
                current_label += 1
        
        return self

def fallback_kdtree_query(coords, radius):
    """Fallback neighbor finding when SciPy not available"""
    n_points = len(coords)
    neighbors = []
    
    for i in range(n_points):
        point_neighbors = []
        for j in range(n_points):
            if i != j:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= radius:
                    point_neighbors.append(j)
        neighbors.append(point_neighbors)
    
    return neighbors

@dataclass
class EnhancedConfig:
    """Enhanced configuration with visualization options"""
    # Processing parameters
    mode: str = "residue"
    rsa_threshold: float = 0.2
    neighbor_radius: float = 8.0
    crf_alpha: float = 0.7
    crf_iterations: int = 2
    crf_sigma: float = 3.0
    adaptive_percentiles: List[float] = field(default_factory=lambda: [95, 90, 85, 80])
    fallback_threshold: float = 0.3
    seed_separation: float = 6.0
    grow_radius: float = 8.0
    min_pocket_size: int = 5
    min_sum_prob: float = 1.5
    nms_radius: float = 6.0
    max_pockets: int = 5
    
    # Visualization parameters
    create_pymol: bool = True
    pymol_colors: List[str] = field(default_factory=lambda: ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan'])
    sphere_size: float = 2.0
    transparency: float = 0.3
    
    # Output parameters
    create_individual_folders: bool = True
    generate_summary: bool = True
    save_raw_predictions: bool = True
    save_true_pockets: bool = True

@dataclass
class ProteinResult:
    """Complete results for a single protein"""
    protein_id: str
    num_residues: int
    predictions: np.ndarray
    ground_truth: np.ndarray
    coordinates: np.ndarray
    rsa_values: np.ndarray
    residue_numbers: np.ndarray
    pockets: List[Any]
    true_pockets: List[Dict]
    metrics: Dict[str, float]
    processing_time: float


@dataclass
class EnhancedPipelineRunConfig:
    """Runtime configuration for executing the enhanced pipeline."""

    checkpoint_paths: List[str]
    h5_file: str
    protein_ids: Optional[List[str]] = None
    output_dir: str = "post_processing_results"
    enhanced: EnhancedConfig = field(default_factory=EnhancedConfig)


def load_pipeline_config(config_path: Union[str, Path]) -> EnhancedPipelineRunConfig:
    """Load pipeline configuration from a YAML file."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    base_dir = path.parent.resolve()

    with open(path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    checkpoint_paths = raw_config.get("checkpoint_paths")
    if not checkpoint_paths:
        raise ValueError("Config must define at least one checkpoint path under 'checkpoint_paths'.")

    if isinstance(checkpoint_paths, (str, Path)):
        checkpoint_paths = [checkpoint_paths]

    checkpoint_list = []
    for item in checkpoint_paths:
        item_path = Path(item)
        if not item_path.is_absolute():
            item_path = (base_dir / item_path).resolve()
        checkpoint_list.append(str(item_path))

    h5_file = raw_config.get("h5_file")
    if not h5_file:
        raise ValueError("Config must define 'h5_file'.")
    h5_path = Path(h5_file)
    if not h5_path.is_absolute():
        h5_path = (base_dir / h5_path).resolve()

    protein_ids = raw_config.get("protein_ids")
    output_dir_raw = raw_config.get("output_dir", "post_processing_results")
    output_dir_path = Path(output_dir_raw)
    if not output_dir_path.is_absolute():
        output_dir_path = (base_dir / output_dir_path).resolve()

    enhanced_cfg_dict = raw_config.get("enhanced", {})
    enhanced_config = EnhancedConfig(**enhanced_cfg_dict)

    return EnhancedPipelineRunConfig(
    checkpoint_paths=checkpoint_list,
        h5_file=str(h5_path),
        protein_ids=list(protein_ids) if protein_ids else None,
        output_dir=str(output_dir_path),
        enhanced=enhanced_config,
    )

class PyMOLVisualizer:
    """Generates PyMOL scripts for pocket visualization"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
    
    def create_pymol_script(self, protein_result: ProteinResult, output_path: Path) -> str:
        """Create PyMOL script for visualizing predicted vs true pockets"""
        
        pymol_script = f'''#!/usr/bin/env pymol
# PyMOL Visualization Script for {protein_result.protein_id}
# Generated by Enhanced Post-Processing Pipeline
# Shows predicted pockets (red spheres) vs true binding sites (blue spheres)

# Load protein structure (you'll need to provide the PDB file)
# load {protein_result.protein_id}.pdb, {protein_result.protein_id}

# Basic protein display
hide everything
show cartoon, {protein_result.protein_id}
color grey80, {protein_result.protein_id}
set transparency, 0.1, {protein_result.protein_id}

# Show all residues as lines for context
show lines, {protein_result.protein_id}

'''
        
        # Add predicted pockets
        if protein_result.pockets:
            pymol_script += "# PREDICTED POCKETS\n"
            for i, pocket in enumerate(protein_result.pockets):
                color = self.config.pymol_colors[i % len(self.config.pymol_colors)]
                
                pymol_script += f"\\n# Predicted Pocket {i+1} (Score: {pocket.final_score:.3f})\\n"
                pymol_script += f"select pocket_pred_{i+1}, none\\n"
                
                # Add residues to selection
                for res_idx in pocket.members:
                    res_num = protein_result.residue_numbers[res_idx]
                    pymol_script += f"select pocket_pred_{i+1}, pocket_pred_{i+1} or (resi {res_num})\\n"
                
                # Visualize pocket
                pymol_script += f"show spheres, pocket_pred_{i+1}\\n"
                pymol_script += f"color {color}, pocket_pred_{i+1}\\n"
                pymol_script += f"set sphere_scale, {self.config.sphere_size}, pocket_pred_{i+1}\\n"
                pymol_script += f"set sphere_transparency, {self.config.transparency}, pocket_pred_{i+1}\\n"
        
        # Add true pockets
        true_pocket_residues = np.where(protein_result.ground_truth == 1)[0]
        if len(true_pocket_residues) > 0:
            pymol_script += "\\n# TRUE BINDING SITES\\n"
            pymol_script += "select true_pockets, none\\n"
            
            for res_idx in true_pocket_residues:
                res_num = protein_result.residue_numbers[res_idx]
                pymol_script += f"select true_pockets, true_pockets or (resi {res_num})\\n"
            
            pymol_script += "show spheres, true_pockets\\n"
            pymol_script += "color marine, true_pockets\\n"
            pymol_script += f"set sphere_scale, {self.config.sphere_size * 0.8}, true_pockets\\n"
            pymol_script += f"set sphere_transparency, {self.config.transparency + 0.2}, true_pockets\\n"
        
        # Add pocket centers as larger spheres
        if protein_result.pockets:
            pymol_script += "\\n# POCKET CENTERS\\n"
            for i, pocket in enumerate(protein_result.pockets):
                x, y, z = pocket.center
                pymol_script += f"pseudoatom center_{i+1}, pos=[{x:.3f}, {y:.3f}, {z:.3f}]\\n"
                pymol_script += f"show spheres, center_{i+1}\\n"
                pymol_script += f"color white, center_{i+1}\\n"
                pymol_script += f"set sphere_scale, {self.config.sphere_size * 1.5}, center_{i+1}\\n"
        
        # Final visualization settings
        pymol_script += '''
# Final settings
bg_color white
set depth_cue, 0
set ray_opaque_background, 0
set antialias, 2

# Legend
print "Legend:"
print "  Red/Green/Yellow spheres: Predicted pockets"
print "  Blue spheres: True binding sites"  
print "  White spheres: Pocket centers"

# Center view
zoom all
'''
        
        # Save script
        script_file = output_path / "visualization.pml"
        with open(script_file, 'w') as f:
            f.write(pymol_script)
        
        logger.info(f"📄 Created PyMOL script: {script_file}")
        return str(script_file)

class StructuredResultsManager:
    """Manages structured output organization"""
    
    def __init__(self, base_output_dir: str, config: EnhancedConfig):
        self.base_dir = Path(base_output_dir)
        self.config = config
        self.summary_dir = self.base_dir / "summary"
        
        # Create directory structure
        self._setup_directories()
    
    def _setup_directories(self):
        """Create the directory structure"""
        # Remove existing results if they exist
        if self.base_dir.exists():
            logger.info(f"🧹 Cleaning existing results directory: {self.base_dir}")
            shutil.rmtree(self.base_dir)
        
        # Create main directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Created results directory structure at: {self.base_dir}")
    
    def create_protein_folder(self, protein_id: str) -> Path:
        """Create individual protein folder"""
        protein_dir = self.base_dir / f"protein_{protein_id}"
        protein_dir.mkdir(parents=True, exist_ok=True)
        return protein_dir
    
    def save_protein_results(self, protein_result: ProteinResult, pymol_visualizer: PyMOLVisualizer):
        """Save all results for a single protein"""
        
        protein_dir = self.create_protein_folder(protein_result.protein_id)
        
        # 1. Save pocket details
        self._save_protein_pockets(protein_result, protein_dir)
        
        # 2. Save raw predictions
        if self.config.save_raw_predictions:
            self._save_raw_predictions(protein_result, protein_dir)
        
        # 3. Save true pockets
        if self.config.save_true_pockets:
            self._save_true_pockets(protein_result, protein_dir)
        
        # 4. Save metrics
        self._save_protein_metrics(protein_result, protein_dir)
        
        # 5. Create PyMOL visualization
        if self.config.create_pymol:
            pymol_visualizer.create_pymol_script(protein_result, protein_dir)
        
        logger.info(f"💾 Saved complete results for {protein_result.protein_id} in {protein_dir}")
    
    def _save_protein_pockets(self, protein_result: ProteinResult, protein_dir: Path):
        """Save pocket details to CSV"""
        pockets_file = protein_dir / "pockets.csv"
        
        with open(pockets_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'pocket_id', 'size', 'center_x', 'center_y', 'center_z', 
                'final_score', 'sum_prob', 'mean_prob', 'max_prob',
                'compactness', 'surface_fraction', 'shape_score', 'density', 'volume_est',
                'member_residues'
            ])
            
            for i, pocket in enumerate(protein_result.pockets):
                # Convert member indices to residue numbers
                member_residues = [str(protein_result.residue_numbers[idx]) for idx in pocket.members]
                member_str = ";".join(member_residues)
                
                writer.writerow([
                    i, pocket.size,
                    pocket.center[0], pocket.center[1], pocket.center[2],
                    pocket.final_score, pocket.sum_prob, pocket.mean_prob, pocket.max_prob,
                    pocket.compactness, pocket.surface_fraction, pocket.shape_score,
                    pocket.density, pocket.volume_est,
                    member_str
                ])
    
    def _save_raw_predictions(self, protein_result: ProteinResult, protein_dir: Path):
        """Save raw prediction scores"""
        pred_file = protein_dir / "predictions.txt"
        
        with open(pred_file, 'w') as f:
            f.write(f"# Raw predictions for {protein_result.protein_id}\\n")
            f.write(f"# Total residues: {len(protein_result.predictions)}\\n")
            f.write(f"# Format: residue_number, prediction_score, rsa_value, x, y, z\\n")
            
            for i, (pred, rsa, coord, res_num) in enumerate(zip(
                protein_result.predictions, 
                protein_result.rsa_values,
                protein_result.coordinates,
                protein_result.residue_numbers
            )):
                f.write(f"{res_num}, {pred:.6f}, {rsa:.4f}, {coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}\\n")
    
    def _save_true_pockets(self, protein_result: ProteinResult, protein_dir: Path):
        """Save true binding sites"""
        true_file = protein_dir / "true_pockets.txt"
        
        true_residues = np.where(protein_result.ground_truth == 1)[0]
        
        with open(true_file, 'w') as f:
            f.write(f"# True binding sites for {protein_result.protein_id}\\n")
            f.write(f"# Total binding residues: {len(true_residues)}\\n")
            f.write(f"# Format: residue_number, x, y, z\\n")
            
            for res_idx in true_residues:
                res_num = protein_result.residue_numbers[res_idx]
                coord = protein_result.coordinates[res_idx]
                f.write(f"{res_num}, {coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}\\n")
    
    def _save_protein_metrics(self, protein_result: ProteinResult, protein_dir: Path):
        """Save protein-specific metrics"""
        metrics_file = protein_dir / "metrics.json"
        
        # Add additional computed metrics
        enhanced_metrics = protein_result.metrics.copy()
        enhanced_metrics.update({
            "protein_id": protein_result.protein_id,
            "num_residues": protein_result.num_residues,
            "num_predicted_pockets": len(protein_result.pockets),
            "num_true_binding_residues": int(protein_result.ground_truth.sum()),
            "processing_time_seconds": float(protein_result.processing_time),
            "prediction_range": {
                "min": float(protein_result.predictions.min()),
                "max": float(protein_result.predictions.max()),
                "mean": float(protein_result.predictions.mean())
            },
            "pocket_details": [
                {
                    "pocket_id": i,
                    "size": pocket.size,
                    "score": float(pocket.final_score),
                    "center": [float(x) for x in pocket.center.tolist()]
                }
                for i, pocket in enumerate(protein_result.pockets)
            ]
        })

        def _to_serializable(value: Any):
            if isinstance(value, dict):
                return {k: _to_serializable(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_to_serializable(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_to_serializable(v) for v in value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value

        with open(metrics_file, 'w') as f:
            json.dump(_to_serializable(enhanced_metrics), f, indent=2)
    
    def create_summary_report(self, all_results: List[ProteinResult], overall_metrics: Dict):
        """Create comprehensive summary report"""
        
        # 1. Text summary
        self._create_text_summary(all_results, overall_metrics)
        
        # 2. Combined CSV
        self._create_combined_csv(all_results)
        
        # 3. Performance metrics JSON
        self._create_performance_json(overall_metrics)
        
        logger.info(f"📊 Created summary report in {self.summary_dir}")
    
    def _create_text_summary(self, all_results: List[ProteinResult], overall_metrics: Dict):
        """Create human-readable text summary"""
        summary_file = self.summary_dir / "summary_report.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Enhanced Post-Processing Results Summary\\n")
            f.write("=" * 50 + "\\n\\n")
            
            # Overall statistics
            f.write("OVERALL PERFORMANCE\\n")
            f.write("-" * 20 + "\\n")
            f.write(f"Proteins processed: {len(all_results)}\\n")
            f.write(f"Total residues: {sum(r.num_residues for r in all_results):,}\\n")
            f.write(f"Total pockets found: {sum(len(r.pockets) for r in all_results)}\\n")
            f.write(f"Average pockets per protein: {sum(len(r.pockets) for r in all_results) / len(all_results):.2f}\\n")
            
            if 'auprc' in overall_metrics:
                f.write(f"\\nPERFORMANCE METRICS\\n")
                f.write("-" * 18 + "\\n")
                f.write(f"AUPRC: {overall_metrics['auprc']:.4f}\\n")
                f.write(f"AUROC: {overall_metrics.get('auroc', 0):.4f}\\n")
                f.write(f"Mean IoU: {overall_metrics.get('mean_iou', 0):.4f}\\n")
            
            # Per-protein breakdown
            f.write("\\nPER-PROTEIN BREAKDOWN\\n")
            f.write("-" * 23 + "\\n")
            
            for result in all_results:
                f.write(f"\\n{result.protein_id}:\\n")
                f.write(f"  Residues: {result.num_residues}\\n")
                f.write(f"  Pockets: {len(result.pockets)}\\n")
                f.write(f"  True binding sites: {result.ground_truth.sum()}\\n")
                f.write(f"  Processing time: {result.processing_time:.2f}s\\n")
                
                if result.pockets:
                    f.write(f"  Best pocket score: {max(p.final_score for p in result.pockets):.3f}\\n")
                
                if 'iou' in result.metrics:
                    f.write(f"  IoU: {result.metrics['iou']:.3f}\\n")
    
    def _create_combined_csv(self, all_results: List[ProteinResult]):
        """Create CSV with all pockets from all proteins"""
        csv_file = self.summary_dir / "all_pockets.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'protein_id', 'pocket_id', 'size', 'center_x', 'center_y', 'center_z',
                'final_score', 'sum_prob', 'compactness', 'surface_fraction',
                'shape_score', 'density', 'volume_est', 'processing_time'
            ])
            
            for result in all_results:
                for i, pocket in enumerate(result.pockets):
                    writer.writerow([
                        result.protein_id, i, pocket.size,
                        pocket.center[0], pocket.center[1], pocket.center[2],
                        pocket.final_score, pocket.sum_prob, pocket.compactness,
                        pocket.surface_fraction, pocket.shape_score,
                        pocket.density, pocket.volume_est, result.processing_time
                    ])
    
    def _create_performance_json(self, overall_metrics: Dict):
        """Create machine-readable performance metrics"""
        json_file = self.summary_dir / "performance_metrics.json"
        
        with open(json_file, 'w') as f:
            json.dump(overall_metrics, f, indent=2)

# Import the existing ProductionPostProcessor from the original file
# (We'll inherit from it and enhance it)

class EnhancedPostProcessor:
    """Enhanced post-processor with visualization and structured results"""

    def __init__(self, config: EnhancedConfig = None, h5_path: Optional[str] = None):
        self.config = config or EnhancedConfig()
        self.h5_path = h5_path
        
        # Initialize components
        self.pymol_visualizer = PyMOLVisualizer(self.config) if self.config.create_pymol else None
        self.evaluator = PostProcessingEvaluator() if EVALUATION_AVAILABLE else None
        
        # Initialize production processor if available
        if PRODUCTION_AVAILABLE:
            production_config = ProductionConfig(
                mode=self.config.mode,
                rsa_threshold=self.config.rsa_threshold,
                neighbor_radius=self.config.neighbor_radius,
                crf_alpha=self.config.crf_alpha,
                crf_iterations=self.config.crf_iterations,
                crf_sigma=self.config.crf_sigma,
                adaptive_percentiles=self.config.adaptive_percentiles,
                fallback_threshold=self.config.fallback_threshold,
                seed_separation=self.config.seed_separation,
                grow_radius=self.config.grow_radius,
                min_pocket_size=self.config.min_pocket_size,
                min_sum_prob=self.config.min_sum_prob,
                nms_radius=self.config.nms_radius,
                max_pockets=self.config.max_pockets
            )
            self.production_processor = ProductionPostProcessor(production_config, self.h5_path)
        else:
            self.production_processor = None
        
        logger.info("🔧 Initialized EnhancedPostProcessor")
        logger.info(f"   PyMOL visualization: {self.config.create_pymol}")
        logger.info(f"   Individual folders: {self.config.create_individual_folders}")
        logger.info(f"   Summary generation: {self.config.generate_summary}")
        logger.info(f"   Production processor: {PRODUCTION_AVAILABLE}")
    
    def run_complete_pipeline(self,
                            checkpoint_paths: List[str],
                            h5_file: Optional[str] = None,
                            protein_ids: Optional[List[str]] = None,
                            output_dir: str = "post_processing_results") -> Dict:
        """Run the complete enhanced post-processing pipeline"""
        
        logger.info("🚀 Starting Enhanced Post-Processing Pipeline")
        logger.info(f"🎯 Output directory: {output_dir}")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        effective_h5_file = h5_file or self.h5_path
        if effective_h5_file is None:
            raise ValueError("An H5 file path must be provided either during initialization or run_complete_pipeline().")

        # Initialize results manager
        results_manager = StructuredResultsManager(output_dir, self.config)
        
        # Get test proteins if not specified
        if protein_ids is None:
            protein_ids = self._get_test_proteins(effective_h5_file)
        
        logger.info(f"🧬 Processing {len(protein_ids)} proteins")
        
        # Step 1: Ensemble inference with shared memory
        logger.info("\\n📊 Step 1: Ensemble inference")
        ensemble_predictions, raw_predictions = self._run_ensemble_inference(
            checkpoint_paths, effective_h5_file, protein_ids
        )
        
        # Step 2: Load ground truth
        logger.info("\\n📊 Step 2: Loading ground truth")
        ground_truth = self._load_ground_truth(effective_h5_file, protein_ids)
        
        # Step 3: Process each protein individually
        logger.info("\\n🏗️ Step 3: Processing individual proteins")
        all_protein_results = []
        
        for protein_id in protein_ids:
            if protein_id not in ensemble_predictions:
                logger.warning(f"⚠️ Skipping {protein_id} - no predictions")
                continue
            
            protein_start = time.time()
            
            # Process protein
            protein_result = self._process_single_protein(
                protein_id, 
                ensemble_predictions[protein_id],
                ground_truth.get(protein_id, np.array([])),
                effective_h5_file
            )
            
            protein_result.processing_time = time.time() - protein_start
            all_protein_results.append(protein_result)
            
            # Save protein results
            results_manager.save_protein_results(protein_result, self.pymol_visualizer)
            
            logger.info(f"  ✅ {protein_id}: {protein_result.num_residues} res → {len(protein_result.pockets)} pockets ({protein_result.processing_time:.2f}s)")
        
        # Step 4: Comprehensive evaluation and summary
        logger.info("\\n🎯 Step 4: Creating comprehensive evaluation")
        overall_metrics = self._create_overall_evaluation(all_protein_results)
        
        # Step 5: Generate summary
        if self.config.generate_summary:
            results_manager.create_summary_report(all_protein_results, overall_metrics)
        
        total_time = time.time() - start_time
        
        # Final report
        logger.info(f"\\n📈 Enhanced Pipeline Complete!")
        logger.info("=" * 50)
        logger.info(f"🧬 Proteins processed: {len(all_protein_results)}")
        logger.info(f"🏗️ Total pockets: {sum(len(r.pockets) for r in all_protein_results)}")
        logger.info(f"⏱️ Total time: {total_time:.2f}s")
        
        if 'auprc' in overall_metrics:
            logger.info(f"🎯 AUPRC: {overall_metrics['auprc']:.4f}")
            logger.info(f"🔄 Mean IoU: {overall_metrics.get('mean_iou', 0):.4f}")
        
        logger.info(f"📁 Results saved to: {output_dir}")
        
        return {
            "protein_results": all_protein_results,
            "overall_metrics": overall_metrics,
            "output_directory": output_dir,
            "total_time": total_time
        }
    
    def _get_test_proteins(self, h5_file: str, max_proteins: int = 6) -> List[str]:
        """Get test proteins from H5 file"""
        with h5py.File(h5_file, 'r') as f:
            protein_keys = f['protein_keys'][:]
            protein_keys = [k.decode() if isinstance(k, bytes) else str(k) for k in protein_keys]
            
            if 'split' in f:
                splits = f['split'][:]
                protein_to_split = {protein_keys[i]: splits[i] for i in range(len(protein_keys))}
                test_proteins = [pid for pid, split in protein_to_split.items() if split == 2][:max_proteins]
            else:
                test_proteins = list(set(protein_keys))[:max_proteins]
        
        return test_proteins
    
    def _run_ensemble_inference(self, checkpoint_paths: List[str], h5_file: str, protein_ids: List[str]) -> Tuple[Dict, Dict]:
        """Run ensemble inference using shared memory"""
        
        if not INFERENCE_AVAILABLE:
            logger.error("❌ Inference modules not available")
            return {}, {}

        # Preferred path: use production components when present
        if PRODUCTION_AVAILABLE and self.production_processor:
            ensemble_predictions, raw_predictions = self.production_processor.ensemble_average_logits_all_seeds(
                checkpoint_paths, h5_file, protein_ids
            )
            logger.info(f"✅ Ensemble inference completed for {len(ensemble_predictions)} proteins")
            return ensemble_predictions, raw_predictions

        # Fallback: perform multi-seed inference directly
        logger.info("⚙️ Using MultiSeedInference fallback for ensemble generation")
        multi_seed = MultiSeedInference(checkpoint_paths, use_shared_memory=True)

        # Attempt shared memory preparation but continue if it fails
        try:
            multi_seed.prepare_shared_memory(h5_file)
        except Exception as exc:
            logger.debug(f"Shared memory preparation failed: {exc}")

        seed_predictions: List[Dict[str, np.ndarray]] = []

        for idx, model_inf in enumerate(multi_seed.model_instances):
            try:
                preds = model_inf.predict_from_h5(
                    h5_file,
                    protein_ids,
                    use_shared_memory=multi_seed.use_shared_memory,
                )
                if preds:
                    seed_predictions.append(preds)
                logger.info(f"Seed {idx + 1}/{len(multi_seed.model_instances)} produced predictions for {len(preds)} proteins")
            except Exception as exc:
                logger.error(f"Seed {idx + 1} prediction failed: {exc}")

        if not seed_predictions:
            logger.error("❌ No predictions generated by any seed models")
            return {}, {}

        all_proteins: Set[str] = set()
        for preds in seed_predictions:
            all_proteins.update(preds.keys())

        ensemble_predictions: Dict[str, np.ndarray] = {}
        raw_predictions: Dict[str, np.ndarray] = {}

        for protein_id in sorted(all_proteins):
            protein_arrays = [preds[protein_id] for preds in seed_predictions if protein_id in preds]
            if not protein_arrays:
                continue
            stacked = np.vstack(protein_arrays)
            ensemble_predictions[protein_id] = stacked.mean(axis=0)
            raw_predictions[protein_id] = protein_arrays[0]

        logger.info(f"✅ Fallback ensemble completed for {len(ensemble_predictions)} proteins")
        return ensemble_predictions, raw_predictions
    
    def _load_ground_truth(self, h5_file: str, protein_ids: List[str]) -> Dict[str, np.ndarray]:
        """Load ground truth labels"""
        if not self.evaluator:
            return {}
        
        return self.evaluator.load_ground_truth_from_h5(h5_file, protein_ids)
    
    def _process_single_protein(self, protein_id: str, predictions: np.ndarray, 
                               ground_truth: np.ndarray, h5_file: str) -> ProteinResult:
        """Process a single protein using production pipeline and return complete results"""
        
        # Load structural data using production loader
        coords, rsa_values, residue_numbers = self._load_protein_structure(protein_id, len(predictions), h5_file)
        
        # Use production processor for advanced pocket detection
        if self.production_processor:
            pockets = self.production_processor.process_protein_production(protein_id, predictions)
        else:
            # Fallback to simplified detection
            pockets = self._detect_pockets_fallback(predictions, coords, rsa_values)
        
        # Extract true pockets from ground truth
        true_pockets = self._extract_true_pockets(ground_truth, coords, residue_numbers)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_protein_metrics(predictions, ground_truth, pockets, true_pockets)
        
        return ProteinResult(
            protein_id=protein_id,
            num_residues=len(predictions),
            predictions=predictions,
            ground_truth=ground_truth,
            coordinates=coords,
            rsa_values=rsa_values,
            residue_numbers=residue_numbers,
            pockets=pockets,
            true_pockets=true_pockets,
            metrics=metrics,
            processing_time=0  # Will be set by caller
        )
    
    def _load_protein_structure(self, protein_id: str, n_residues: int, h5_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load protein structural data aligned with ensemble predictions."""

        loader: Optional[StructuralDataLoader] = None

        if self.production_processor and hasattr(self.production_processor, "loader"):
            loader = self.production_processor.loader
        elif PRODUCTION_AVAILABLE:
            loader = StructuralDataLoader()

        if loader is None:
            raise RuntimeError(
                "StructuralDataLoader unavailable; install production components to run enhanced pipeline"
            )

        residue_numbers: Optional[np.ndarray] = None
        try:
            residue_numbers = loader.get_residue_numbers_from_h5(h5_file, protein_id)
        except Exception as exc:
            logger.debug(f"Could not read residue numbers from H5 for {protein_id}: {exc}")

        coords, rsa, residue_numbers = loader.load_struct_with_residue_numbers(
            protein_id, residue_numbers=residue_numbers
        )

        if len(coords) != n_residues:
            raise ValueError(
                f"Prediction length mismatch for {protein_id}: predictions={n_residues}, coordinates={len(coords)}"
            )

        return coords.astype(np.float32), rsa.astype(np.float32), residue_numbers.astype(np.int32)

    def _calculate_protein_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        pockets: List[Any],
        true_pockets: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate per-protein evaluation metrics."""

        metrics: Dict[str, Any] = {
            "num_residues": int(len(predictions)),
            "num_pockets": len(pockets),
            "num_true_pockets": len(true_pockets),
            "prediction_mean": float(predictions.mean()) if len(predictions) else 0.0,
            "prediction_max": float(predictions.max()) if len(predictions) else 0.0,
            "true_positive_residues": int(ground_truth.sum()) if len(ground_truth) else 0,
        }

        if len(predictions) == 0 or len(ground_truth) == 0:
            return metrics

        try:
            min_len = min(len(predictions), len(ground_truth))
            preds_slice = predictions[:min_len]
            gt_slice = ground_truth[:min_len].astype(int)

            threshold = getattr(self.config, "fallback_threshold", 0.5)
            pred_binary = (preds_slice >= threshold).astype(int)
            gt_binary = (gt_slice > 0).astype(int)

            tp = int(np.logical_and(pred_binary == 1, gt_binary == 1).sum())
            fp = int(np.logical_and(pred_binary == 1, gt_binary == 0).sum())
            fn = int(np.logical_and(pred_binary == 0, gt_binary == 1).sum())

            metrics["predicted_positive_residues"] = int(pred_binary.sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1

            if self.evaluator:
                try:
                    metrics["iou@0.5"] = float(
                        self.evaluator.calculate_iou_per_protein(preds_slice, gt_slice, threshold=0.5)
                    )
                except Exception as exc:
                    logger.debug(f"Failed to compute IoU for protein metrics: {exc}")

        except Exception as exc:
            logger.warning(f"Failed to calculate protein metrics: {exc}")

        return metrics
    
    def _create_overall_evaluation(self, all_results: List[ProteinResult]) -> Dict:
        """Create overall evaluation metrics"""
        
        if not self.evaluator:
            return {"total_proteins": len(all_results)}
        
        try:
            # Combine all predictions and ground truth
            all_predictions = {}
            all_ground_truth = {}
            
            for result in all_results:
                if len(result.ground_truth) > 0:
                    all_predictions[result.protein_id] = result.predictions
                    all_ground_truth[result.protein_id] = result.ground_truth
            
            if not all_predictions:
                return {"total_proteins": len(all_results)}
            
            # Calculate overall metrics
            overall_results = self.evaluator.evaluate_comprehensive(
                all_predictions, all_ground_truth
            )
            
            return {
                "auprc": overall_results.auprc,
                "auroc": overall_results.auroc, 
                "mean_iou": overall_results.mean_iou,
                "total_proteins": len(all_results),
                "total_pockets": sum(len(r.pockets) for r in all_results),
                "avg_pockets_per_protein": sum(len(r.pockets) for r in all_results) / len(all_results),
                "total_processing_time": sum(r.processing_time for r in all_results)
            }
            
        except Exception as e:
            logger.warning(f"Failed to create overall evaluation: {e}")
            return {"total_proteins": len(all_results)}


def run_enhanced_pipeline(
    config: Optional[EnhancedPipelineRunConfig] = None,
    config_path: Optional[Union[str, Path]] = None
):
    """Run the enhanced post-processing pipeline using the provided configuration."""

    logger.info("🚀 Enhanced Post-Processing with PyMOL Visualization")
    logger.info("🎯 Structured results + True vs Predicted pockets")
    logger.info("=" * 70)

    if config is None:
        if config_path is None:
            default_config = Path(__file__).resolve().parent / "configs" / "sota_default.yaml"
            if default_config.exists():
                config_path = default_config
            else:
                raise FileNotFoundError(
                    "No configuration provided and default config missing at post_processing/configs/sota_default.yaml."
                )
        config = load_pipeline_config(config_path)

    processor = EnhancedPostProcessor(config.enhanced, config.h5_file)
    results = processor.run_complete_pipeline(
        checkpoint_paths=config.checkpoint_paths,
        h5_file=config.h5_file,
        protein_ids=config.protein_ids,
        output_dir=config.output_dir,
    )

    return results


if __name__ == "__main__":
    try:
        results = run_enhanced_pipeline()
        logger.info("✅ Enhanced pipeline completed successfully!")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)