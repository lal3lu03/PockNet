# DDP Logging Improvements for PockNet

## Overview

This document describes the enhanced DDP (Distributed Data Parallel) logging implementation for PockNet, specifically addressing Weights & Biases (W&B) logging challenges when using `find_unused_parameters=True`.

## Problem

When using PyTorch DDP with `find_unused_parameters=True` and logging to W&B, several issues can occur:

1. **Autograd Hook Conflicts**: W&B's `wandb.watch()` relies on autograd hooks that may not trigger for unused parameters
2. **Parameter Count Logging Failures**: Some layers may not be tracked properly
3. **Silent Failures**: Subprocesses may fail without visible errors
4. **Multi-process Logging Conflicts**: Multiple processes trying to log simultaneously

## Solution

### 1. Enhanced Logging Utilities (`src/utils/logging_utils.py`)

#### Key Functions:

- **`is_main_process()`**: Safe rank checking for distributed environments
- **`log_model_architecture()`**: DDP-safe model parameter counting and architecture logging
- **`safe_wandb_watch()`**: Robust W&B model watching with error handling
- **`log_ddp_info()`**: Comprehensive DDP configuration logging
- **`setup_ddp_logging_best_practices()`**: Automated DDP optimization detection
- **`recover_wandb_logging()`**: W&B connection recovery for unstable environments
- **`validate_ddp_config()`**: Configuration validation with optimization suggestions

#### Key Features:

```python
# Manual parameter counting (bypasses wandb.watch() dependencies)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# DDP-safe model unwrapping
if hasattr(model, 'module'):
    unwrapped_model = model.module
else:
    unwrapped_model = model

# Main process only logging
@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    # ... enhanced logging logic
```

### 2. W&B DDP Configuration (`configs/logger/wandb_ddp.yaml`)

Optimized W&B settings for DDP:

```yaml
logger:
  wandb:
    settings:
      start_method: "thread"  # DDP-compatible initialization
      console: "off"          # Reduce multi-process spam
    log_model: false          # Prevent DDP conflicts
    watch_model: true         # Handled by custom code
    resume: "auto"            # Auto-resume capability
```

### 3. Environment Variables

For optimal DDP performance:

```bash
export NCCL_DEBUG=INFO
export WANDB_DEBUG=TRUE  # For debugging W&B issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Best Practices Implemented

### 1. **Main Process Only Logging**
```python
if is_main_process():
    wandb.log({"metric": value})
```

### 2. **Manual Parameter Logging**
```python
# Bypass wandb.watch() dependency
model_info = {
    "model/total_params": total_params,
    "model/trainable_params": trainable_params,
    "model/total_MB": total_params * 4 / (1024 * 1024)
}
```

### 3. **Delayed W&B Watch Setup**
```python
# Avoid early initialization conflicts
time.sleep(1)
safe_wandb_watch(model, logger, log_freq=100, log_graph=False)
```

### 4. **Error Recovery**
```python
# Automatic recovery for unstable connections
if recover_wandb_logging(trainer, model):
    safe_wandb_watch(model, logger)
else:
    log.warning("W&B recovery failed, continuing without watching")
```

### 5. **Configuration Validation**
```python
# Automatic optimization suggestions
validation = validate_ddp_config(trainer)
if validation["warnings"]:
    for warning in validation["warnings"]:
        log.warning(f"DDP Warning: {warning}")
```

## Usage

### 1. Standard Training with Enhanced Logging
```bash
python src/train.py experiment=pocknet logger=wandb_ddp
```

### 2. Test Enhanced DDP
```bash
./test_enhanced_ddp.sh
```

### 3. Debug Mode
```bash
WANDB_DEBUG=TRUE python src/train.py experiment=pocknet logger=wandb_ddp
```

## DDP Strategy Configuration

Current strategy: `ddp_find_unused_parameters_true`

```yaml
# configs/trainer/default.yaml
strategy: ddp_find_unused_parameters_true
```

### Optimization Recommendations:

1. **If no unused parameters**: Change to `ddp` strategy for better performance
2. **For stable models**: Set `find_unused_parameters: false`
3. **For large models**: Consider `ddp_sharded` strategy

## Monitoring and Debugging

### 1. **DDP Configuration Logging**
The enhanced logging automatically reports:
- World size and rank information
- Strategy configuration
- find_unused_parameters setting
- GPU utilization and memory usage
- Batch size per GPU vs. effective batch size

### 2. **W&B Integration Status**
- Connection status validation
- Model watching status
- Gradient logging frequency
- Error recovery attempts

### 3. **Performance Warnings**
- Small batch size per GPU warnings
- Low GPU utilization alerts
- Memory usage optimization suggestions

## Integration with Modified PockNet

The enhanced logging is fully integrated with the modified PockNet experiment:

1. **Feature exclusion**: `protrusion.distanceToCenter` removed
2. **Input dimension**: Reduced from 42 to 41 features
3. **Scheduler**: Changed to CosineAnnealingLR
4. **DDP logging**: Enhanced W&B integration

## Troubleshooting

### Common Issues:

1. **W&B not initializing**
   - Check `WANDB_DEBUG=TRUE` output
   - Verify main process logging
   - Check network connectivity

2. **Parameter counts missing**
   - Manual logging bypasses hook dependencies
   - Check model unwrapping logic

3. **Silent subprocess failures**
   - Enhanced error logging captures subprocess issues
   - Recovery mechanisms handle transient failures

4. **Performance degradation**
   - Configuration validation suggests optimizations
   - Automatic detection of suboptimal settings

## Testing

Run the test script to validate the implementation:

```bash
./test_enhanced_ddp.sh
```

This runs a short experiment (5 epochs, 10% data) to verify:
- DDP initialization
- W&B logging
- Feature exclusion
- Cosine scheduler
- Enhanced logging utilities

## Migration from Standard Logging

To upgrade existing experiments:

1. Replace `logger: wandb` with `logger: wandb_ddp`
2. No code changes required - enhanced logging is automatic
3. Existing configurations remain compatible

The enhanced logging is backward-compatible and provides graceful degradation if W&B is unavailable.
