# PockNet Post-Processing Pipeline

A comprehensive post-processing system for converting residue-level predictions from PockNet models into high-quality pocket predictions. This pipeline implements the default recipe for optimal pocket detection performance.

## Overview

The post-processing pipeline transforms raw residue probability scores into biologically meaningful pocket predictions through:

1. **Surface filtering** (RSA ≥ 0.2)
2. **Spatial clustering** (Cα–Cα ≤ 8Å graph construction)
3. **Connected components** analysis
4. **Cluster filtering** (size ≥ 5, ∑p ≥ 2.0)
5. **Pocket scoring** and ranking

## Enhanced Pipeline Workflow

An end-to-end, state-of-the-art workflow is available via the enhanced pipeline. It performs shared-memory inference, multi-checkpoint ensembling, adaptive pocket formation, and rich reporting in a single command.

### Run the enhanced pipeline

```bash
python -m post_processing.run_enhanced_pipeline --config post_processing/configs/sota_default.yaml
```

- **Configuration** – edit the YAML file to point to your checkpoints, dataset (`h5_file`), and output directory. All EnhancedConfig options are exposed under the `enhanced` key.
- **Path handling** – any relative paths in the YAML are resolved relative to the configuration file itself, keeping configs portable across machines.
- **Results** – the pipeline writes structured artefacts under `post_processing_results/`, including per-protein folders, PyMOL scripts, and a `summary/` directory with machine-friendly metrics.
- **Dry run** – append `--dry-run` to validate the configuration without performing inference (useful for CI).

`advanced_post_processing.py` now delegates to this entry point for backwards compatibility.

## Installation

The post-processing module requires the following dependencies:

```bash
pip install numpy scipy scikit-learn matplotlib seaborn
```

For PyTorch model inference:
```bash
pip install torch pytorch-lightning h5py
```

## Quick Start

### Basic Usage

```python
from post_processing import residue_to_pockets

# Your residue data
residues = [
    {
        'chain': 'A',
        'res_id': 1,
        'xyz': np.array([0, 0, 0]),  # 3D coordinates
        'rsa': 0.8,                   # Relative surface accessibility
        'prob': 0.9                   # Model prediction
    },
    # ... more residues
]

# Convert to pockets
pockets = residue_to_pockets(residues)

# Results
for i, pocket in enumerate(pockets):
    print(f"Pocket {i+1}: size={pocket.size}, score={pocket.score:.3f}")
```

### Multi-Seed Ensemble

```python
from post_processing import ensemble_predictions, residue_to_pockets

# Predictions from multiple seeds
seed_predictions = [pred1, pred2, pred3, ...]  # numpy arrays

# Ensemble (mean)
ensemble = ensemble_predictions(seed_predictions, method="mean")

# Convert to pockets
pockets = residue_to_pockets(residues_with_ensemble_probs)
```

### Hyperparameter Optimization

```python
from post_processing import run_validation_sweep, SweepConfig

# Define parameter ranges
sweep_config = SweepConfig(
    graph_radii=[6.0, 7.0, 8.0],
    min_cluster_sizes=[3, 5, 8],
    sump_thresholds=[1.5, 2.0, 3.0]
)

# Run optimization
sweep = run_validation_sweep(
    validation_data=validation_proteins,
    sweep_config=sweep_config
)

best_config = sweep.get_best_config()
```

## Module Structure

```
post_processing/
├── __init__.py           # Main module imports
├── core.py              # Core pocket conversion pipeline
├── ensemble.py          # Multi-seed ensembling utilities
├── inference.py         # Model loading and prediction
├── metrics.py           # Pocket-level evaluation metrics
├── sweep.py             # Hyperparameter optimization
├── example_usage.py     # Usage examples
└── notebook_integration.py  # Integration with notebook analysis
```

## Core Components

### 1. Core Pipeline (`core.py`)

The main residue-to-pocket conversion engine:

- **`PocketPostProcessor`**: Main processing class
- **`residue_to_pockets()`**: Convenience function
- **`make_residue_graph()`**: Spatial graph construction
- **`postprocess_residues()`**: Minimal implementation

### 2. Ensemble Utilities (`ensemble.py`)

Multi-seed prediction combination:

- **`ensemble_predictions()`**: Average predictions across seeds
- **`spatial_smoothing()`**: Reduce salt-and-pepper noise
- **`adaptive_threshold()`**: Dynamic thresholding strategies
- **`multi_seed_consensus()`**: Pocket-level consensus across seeds

### 3. Model Inference (`inference.py`)

PyTorch model loading and prediction:

- **`ModelInference`**: Single model inference class
- **`MultiSeedInference`**: Multi-seed ensemble inference
- **`load_checkpoint()`**: Checkpoint loading utilities
- **`prepare_residue_data()`**: Data formatting for post-processing

### 4. Evaluation Metrics (`metrics.py`)

Pocket-level evaluation beyond residue IoU:

- **`PocketEvaluator`**: Comprehensive evaluation class
- **`pocket_recall_at_k()`**: Pocket recall @ top-k predictions
- **`center_distance_metrics()`**: Spatial distance evaluation
- **`pocket_iou_metrics()`**: Residue-set IoU for pockets
- **`pocket_pr_curve()`**: Pocket-level precision-recall curves

### 5. Hyperparameter Optimization (`sweep.py`)

Automated parameter tuning:

- **`HyperparameterSweep`**: Grid search framework
- **`SweepConfig`**: Parameter range specification
- **`run_validation_sweep()`**: Complete optimization pipeline
- **`pocket_optimization_objective()`**: Pocket-focused objective function

## Default Recipe Parameters

The pipeline uses the following well-tested defaults:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `rsa_min` | 0.2 | Minimum RSA for surface filtering |
| `graph_radius` | 8.0 Å | Maximum Cα–Cα distance for edges |
| `min_cluster_size` | 5 | Minimum residues per cluster |
| `min_sump` | 2.0 | Minimum sum of probabilities |
| `max_pockets` | 5 | Maximum pockets to return |
| `threshold_strategy` | "percentile95" | Adaptive thresholding method |

## Advanced Features

### Spatial Smoothing

Reduce false positives with neighbor averaging:

```python
from post_processing import spatial_smoothing, make_residue_graph

# Build residue graph
graph = make_residue_graph(coordinates, radius=8.0)

# Apply smoothing
smoothed_probs = spatial_smoothing(
    probabilities, 
    graph, 
    alpha=0.7  # 70% original, 30% neighbors
)
```

### Adaptive Thresholding

Multiple strategies for dynamic threshold selection:

```python
from post_processing import adaptive_threshold

# 95th percentile (default)
threshold = adaptive_threshold(predictions, strategy="percentile95")

# Target prevalence (e.g., 3% of residues)
threshold = adaptive_threshold(predictions, strategy="prevalence", target_prevalence=0.03)

# Validation-optimal (requires labels)
threshold = adaptive_threshold(predictions, strategy="global_opt", validation_labels=labels)
```

## 📊 Analysis Notebook

After running the enhanced pipeline, explore the aggregated metrics and pocket characteristics with the notebook `post_processing/sota_post_processing_analysis.ipynb`. It loads the artefacts from `post_processing_results/summary/` and produces:

- headline metrics (AUPRC, AUROC, IoU)
- distribution plots of pocket scores
- per-protein summaries and scatter plots relating pocket yield to quality indicators

Open the notebook in Jupyter or VS Code, execute the cells, and the figures will update automatically based on the latest pipeline output.

### Non-Maximum Suppression

Remove nearby duplicate pockets:

```python
from post_processing.ensemble import non_maximum_suppression

# Filter pockets within 6Å of each other
filtered_pockets = non_maximum_suppression(
    pockets, 
    distance_threshold=6.0
)
```

## Integration with Notebook Analysis

Use the best model identified in your notebook analysis:

```bash
# Run post-processing with notebook results
python post_processing/notebook_integration.py --mode all --output_dir results/

# Single best model only
python post_processing/notebook_integration.py --mode single

# 7-seed ensemble
python post_processing/notebook_integration.py --mode ensemble --use_top_n 7

# Hyperparameter optimization
python post_processing/notebook_integration.py --mode optimize
```

The integration script automatically uses:
- **Best model**: Seed 55 (AUPRC: 0.4048)
- **Model path**: `/logs/fusion_all_train_complete/runs/2025-09-04_19-45-33/checkpoints`
- **All 7 seeds**: For ensemble processing
- **WandB run IDs**: For traceability

## Example Workflows

### Workflow 1: Single Model Processing

```python
# 1. Load model
from post_processing import ModelInference
model = ModelInference("path/to/best_checkpoint.ckpt", model_class=YourModel)

# 2. Generate predictions
predictions = model.predict_from_h5("data.h5")

# 3. Prepare residue data
residues = prepare_residue_data(predictions, coordinates, rsa_values, chains, res_ids)

# 4. Convert to pockets
pockets = residue_to_pockets(residues)
```

### Workflow 2: Multi-Seed Ensemble

```python
# 1. Load multiple models
from post_processing import MultiSeedInference
multi_model = MultiSeedInference(checkpoint_paths, YourModel)

# 2. Generate ensemble predictions
ensemble_predictions = multi_model.predict_ensemble(data_loader)

# 3. Post-process ensemble
pockets = residue_to_pockets(residues_with_ensemble)
```

### Workflow 3: Optimization + Evaluation

```python
# 1. Optimize parameters
sweep = run_validation_sweep(validation_data)
best_config = sweep.get_best_config()

# 2. Apply to test data
test_pockets = residue_to_pockets(test_residues, **best_config)

# 3. Evaluate results
from post_processing import PocketEvaluator
evaluator = PocketEvaluator()
results = evaluator.evaluate_protein(test_pockets, ground_truth_pockets)
```

## Performance Tips

### Easy Gains (Implement First)

1. **Mean-ensemble** across 7 seeds → most reliable AUPRC improvement
2. **Surface gating** → use RSA ≥ 0.2 filter → better precision
3. **Spatial smoothing** → reduces salt-and-pepper false positives
4. **Adaptive thresholding** → better than fixed thresholds

### Parameter Tuning Priority

1. **Graph radius** (6-8 Å): Most sensitive parameter
2. **Min cluster size** (3-8): Affects precision vs recall trade-off
3. **Sum threshold** (1.5-3.0): Controls pocket confidence
4. **Threshold strategy**: Protein-dependent optimal choice

### Computational Efficiency

- Use `max_configs` parameter to limit hyperparameter search
- Cache ensemble predictions to avoid recomputation
- Process proteins in batches for large datasets
- Use sparse matrices for graph operations

## Evaluation Metrics

The pipeline provides comprehensive pocket-level evaluation:

### Pocket Recall @ Top-K
- Does any predicted pocket overlap GT pocket?
- Standard evaluation: K=1, 3, 5

### Center Distance
- Minimum distance from predicted to GT centers
- Spatial accuracy assessment

### Pocket IoU
- Residue-set intersection over union
- Direct overlap quantification

### Pocket-Level PR Curves
- Precision-recall at pocket level
- Better reflects final use-case than residue-level metrics

## File Formats

### Input: Residue Data
```python
residue = {
    'chain': 'A',           # Chain identifier
    'res_id': 123,          # Residue number
    'xyz': np.array([x,y,z]), # 3D coordinates (Cβ/Cα)
    'rsa': 0.8,             # Relative surface accessibility
    'prob': 0.9,            # Model prediction probability
    'res_name': 'ALA'       # Residue name (optional)
}
```

### Output: Pocket Data
```python
pocket = {
    'members': [('A', 123), ('A', 124), ...],  # Residue members
    'size': 15,                                # Number of residues
    'center': [x, y, z],                       # 3D center coordinates
    'score': 12.5,                            # Pocket score
    'sump': 8.3,                              # Sum of probabilities
    'surface_fraction': 0.8                    # Fraction of surface residues
}
```

## Troubleshooting

### Common Issues

1. **Empty pocket lists**: Check RSA values and thresholds
2. **Too many small pockets**: Increase `min_cluster_size` or `min_sump`
3. **Low precision**: Apply surface gating or increase thresholds
4. **Low recall**: Decrease thresholds or `min_cluster_size`

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('post_processing').setLevel(logging.DEBUG)
```

### Performance Monitoring

```python
# Monitor processing stats
processor = PocketPostProcessor()
pockets = processor.process(residues)

print(f"Input residues: {len(residues)}")
print(f"Surface residues: {processor.surface_count}")
print(f"Positive residues: {processor.positive_count}")
print(f"Graph edges: {processor.graph_edges}")
print(f"Components found: {processor.n_components}")
print(f"Final pockets: {len(pockets)}")
```

## Citation

If you use this post-processing pipeline in your research, please cite:

```bibtex
@misc{pocknet_postprocessing,
  title={PockNet Post-Processing Pipeline},
  author={PockNet Team},
  year={2025},
  url={https://github.com/your-repo/PockNet}
}
```

## Contributing

To contribute to the post-processing pipeline:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.