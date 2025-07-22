from typing import Any, Dict, Optional
import os
import time

import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def is_main_process() -> bool:
    """Check if current process is the main process in distributed training.

    Returns:
        bool: True if main process (rank 0) or not distributed, False otherwise
    """
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def log_model_architecture(model: torch.nn.Module, input_size: tuple = None) -> Dict[str, Any]:
    """Log detailed model architecture information for DDP-safe logging.

    Args:
        model: PyTorch model to analyze
        input_size: Input tensor size for model summary (optional)

    Returns:
        Dict containing model architecture information
    """
    # Get the underlying model if wrapped in DDP
    if hasattr(model, 'module'):
        unwrapped_model = model.module
    else:
        unwrapped_model = model

    model_info = {}

    # Basic parameter counts
    total_params = sum(p.numel() for p in unwrapped_model.parameters())
    trainable_params = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    model_info.update({
        "model/params/total": total_params,
        "model/params/trainable": trainable_params,
        "model/params/non_trainable": non_trainable_params,
        "model/params/total_MB": total_params * 4 / (1024 * 1024),  # Assuming float32
    })

    # Layer-wise parameter counts
    layer_info = {}
    for name, module in unwrapped_model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                layer_info[f"model/layers/{name}/params"] = module_params

    model_info.update(layer_info)

    # Model architecture string
    model_info["model/architecture"] = str(unwrapped_model)

    # Try to get model summary if torchinfo is available
    try:
        from torchinfo import summary
        if input_size is not None:
            model_summary = summary(unwrapped_model, input_size=input_size, verbose=0)
            model_info["model/summary"] = str(model_summary)
    except ImportError:
        log.warning("torchinfo not available. Install with 'pip install torchinfo' for detailed model summaries.")
    except Exception as e:
        log.warning(f"Could not generate model summary: {e}")

    return model_info


def safe_wandb_watch(model: torch.nn.Module, logger=None, log_freq: int = 100, log_graph: bool = True) -> None:
    """Safely apply wandb.watch() for DDP models with additional robustness.

    Args:
        model: PyTorch model to watch
        logger: W&B logger instance (optional)
        log_freq: Frequency of gradient logging
        log_graph: Whether to log computational graph
    """
    if not is_main_process():
        return

    try:
        import wandb

        # Get the underlying model if wrapped in DDP
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        else:
            unwrapped_model = model

        # Only watch if wandb is initialized
        if wandb.run is not None:
            # Check if model is already being watched to avoid duplicates
            if not hasattr(wandb.run, '_watched_models'):
                wandb.run._watched_models = set()
            
            model_id = id(unwrapped_model)
            if model_id not in wandb.run._watched_models:
                wandb.watch(
                    unwrapped_model, 
                    log="gradients", 
                    log_freq=log_freq,
                    log_graph=log_graph
                )
                wandb.run._watched_models.add(model_id)
                log.info(f"W&B watching model with log_freq={log_freq}, log_graph={log_graph}")
            else:
                log.info("Model already being watched by W&B, skipping duplicate watch")
        else:
            log.warning("W&B not initialized, skipping wandb.watch()")

    except ImportError:
        log.warning("wandb not available, skipping wandb.watch()")
    except Exception as e:
        log.warning(f"Failed to set up wandb.watch(): {e}")


def log_ddp_info(trainer) -> Dict[str, Any]:
    """Log DDP-specific information for debugging.

    Args:
        trainer: PyTorch Lightning trainer

    Returns:
        Dict containing DDP information
    """
    ddp_info = {}

    if torch.distributed.is_initialized():
        ddp_info.update({
            "ddp/world_size": torch.distributed.get_world_size(),
            "ddp/rank": torch.distributed.get_rank(),
            "ddp/local_rank": torch.distributed.get_rank() % torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "ddp/backend": torch.distributed.get_backend(),
            "ddp/master_addr": os.environ.get("MASTER_ADDR", "localhost"),
            "ddp/master_port": os.environ.get("MASTER_PORT", "unknown"),
        })
        
        # Check for potential issues
        if torch.distributed.get_world_size() == 1:
            log.warning("DDP initialized but world_size=1. Consider using single GPU training instead.")
            
    else:
        ddp_info["ddp/enabled"] = False

    # Lightning strategy info
    if hasattr(trainer, 'strategy'):
        strategy_name = str(type(trainer.strategy).__name__)
        ddp_info["ddp/strategy"] = strategy_name
        
        if hasattr(trainer.strategy, 'find_unused_parameters'):
            ddp_info["ddp/find_unused_parameters"] = trainer.strategy.find_unused_parameters
            
        # Log additional strategy-specific info
        if 'ddp' in strategy_name.lower():
            if hasattr(trainer.strategy, 'bucket_cap_mb'):
                ddp_info["ddp/bucket_cap_mb"] = trainer.strategy.bucket_cap_mb
            if hasattr(trainer.strategy, 'static_graph'):
                ddp_info["ddp/static_graph"] = getattr(trainer.strategy, 'static_graph', False)

    # GPU information
    if torch.cuda.is_available():
        ddp_info.update({
            "hardware/gpu_count": torch.cuda.device_count(),
            "hardware/current_device": torch.cuda.current_device(),
            "hardware/device_name": torch.cuda.get_device_name(),
        })

    return ddp_info


def setup_ddp_logging_best_practices(trainer, model) -> None:
    """Apply DDP logging best practices and optimizations.
    
    Args:
        trainer: PyTorch Lightning trainer
        model: PyTorch Lightning model
    """
    if not is_main_process():
        return
        
    log.info("Applying DDP logging best practices...")
    
    # Log comprehensive DDP summary
    log_ddp_best_practices_summary(trainer, model)
    
    # Check for common DDP issues
    ddp_info = log_ddp_info(trainer)
    if ddp_info.get("ddp/find_unused_parameters", False):
        log.warning(
            "find_unused_parameters=True detected. This can slow down training. "
            "Consider setting to False if your model doesn't have unused parameters."
        )
    
    # Setup W&B watching with proper timing and error recovery
    if hasattr(trainer, 'loggers'):
        for logger in trainer.loggers:
            if hasattr(logger, 'experiment') and logger.__class__.__name__ == 'WandbLogger':
                # Delay W&B watch setup to avoid early initialization issues
                time.sleep(1)  # Brief delay to ensure proper initialization
                
                # Attempt W&B setup with recovery
                if recover_wandb_logging(trainer, model):
                    safe_wandb_watch(model, logger, log_freq=100, log_graph=False)
                else:
                    log.warning("W&B recovery failed, continuing without W&B watching")
                break


def log_training_environment() -> Dict[str, Any]:
    """Log comprehensive training environment information.
    
    Returns:
        Dict containing environment information
    """
    env_info = {}
    
    # PyTorch version info
    env_info.update({
        "environment/torch_version": torch.__version__,
        "environment/cuda_available": torch.cuda.is_available(),
        "environment/cudnn_enabled": torch.backends.cudnn.enabled,
        "environment/cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.enabled else None,
    })
    
    if torch.cuda.is_available():
        env_info.update({
            "environment/cuda_version": torch.version.cuda,
            "environment/gpu_count": torch.cuda.device_count(),
            "environment/gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
        })
    
    # System info
    env_info.update({
        "environment/python_version": os.sys.version.split()[0],
        "environment/hostname": os.environ.get("HOSTNAME", "unknown"),
        "environment/slurm_job_id": os.environ.get("SLURM_JOB_ID", None),
        "environment/slurm_procid": os.environ.get("SLURM_PROCID", None),
    })
    
    return env_info


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.
    
    Enhanced for DDP compatibility with proper parameter counting and W&B integration.

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # Enhanced model parameter logging (DDP-safe)
    model_info = log_model_architecture(model)
    hparams.update(model_info)
    # Also log model parameter counts as metrics (e.g., total, trainable, non_trainable, total_MB)
    metric_keys = ["model/params/total", "model/params/trainable", "model/params/non_trainable", "model/params/total_MB"]
    param_metrics = {k: model_info[k] for k in metric_keys if k in model_info}
    for logger in trainer.loggers:
        try:
            if hasattr(logger, 'log_metrics'):
                logger.log_metrics(param_metrics, step=0)
                log.info(f"Logged model parameter metrics to {type(logger).__name__}")
        except Exception as e:
            log.warning(f"Failed to log model parameter metrics to {type(logger).__name__}: {e}")
    
    # Log DDP information
    ddp_info = log_ddp_info(trainer)
    hparams.update(ddp_info)
    
    # Log training environment
    env_info = log_training_environment()
    hparams.update(env_info)

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # Log basic model info to console
    log.info(f"Model parameters: {hparams.get('model/params/total', 0):,}")
    log.info(f"Trainable parameters: {hparams.get('model/params/trainable', 0):,}")
    log.info(f"Model size: {hparams.get('model/params/total_MB', 0):.2f} MB")
    
    if ddp_info.get("ddp/world_size"):
        log.info(f"DDP enabled with {ddp_info['ddp/world_size']} processes")
        log.info(f"DDP strategy: {ddp_info.get('ddp/strategy', 'unknown')}")

    # send hparams to all loggers
    for logger in trainer.loggers:
        try:
            logger.log_hyperparameters(hparams)
            log.info(f"Logged hyperparameters to {type(logger).__name__}")
        except Exception as e:
            log.warning(f"Failed to log hyperparameters to {type(logger).__name__}: {e}")
        
    # Apply DDP logging best practices
    setup_ddp_logging_best_practices(trainer, model)


def recover_wandb_logging(trainer, model, attempt_count: int = 3) -> bool:
    """Attempt to recover W&B logging if it fails during DDP training.
    
    Args:
        trainer: PyTorch Lightning trainer
        model: PyTorch Lightning model  
        attempt_count: Number of recovery attempts
        
    Returns:
        bool: True if recovery successful, False otherwise
    """
    if not is_main_process():
        return True
        
    for attempt in range(attempt_count):
        try:
            import wandb
            
            # Check if W&B run is still active
            if wandb.run is None:
                log.warning(f"W&B run lost, attempting recovery (attempt {attempt + 1}/{attempt_count})")
                
                # Try to reinitialize (this should ideally not happen in production)
                # This is mainly for debugging purposes
                log.warning("W&B run is None - this suggests improper initialization")
                return False
            
            # Test W&B connectivity
            wandb.log({"recovery_test": attempt}, step=0)
            log.info(f"W&B logging recovery successful on attempt {attempt + 1}")
            return True
            
        except Exception as e:
            log.warning(f"W&B recovery attempt {attempt + 1} failed: {e}")
            if attempt < attempt_count - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            
    log.error("W&B logging recovery failed after all attempts")
    return False


def validate_ddp_config(trainer) -> Dict[str, Any]:
    """Validate DDP configuration and suggest optimizations.
    
    Args:
        trainer: PyTorch Lightning trainer
        
    Returns:
        Dict containing validation results and suggestions
    """
    validation = {
        "status": "ok",
        "warnings": [],
        "suggestions": [],
        "config": {}
    }
    
    if not torch.distributed.is_initialized():
        validation["config"]["ddp_enabled"] = False
        return validation
        
    validation["config"]["ddp_enabled"] = True
    world_size = torch.distributed.get_world_size()
    
    # Check world size
    if world_size == 1:
        validation["warnings"].append("DDP initialized with world_size=1")
        validation["suggestions"].append("Consider using single GPU training instead of DDP")
    
    # Check find_unused_parameters
    if hasattr(trainer.strategy, 'find_unused_parameters'):
        if trainer.strategy.find_unused_parameters:
            validation["warnings"].append("find_unused_parameters=True may slow down training")
            validation["suggestions"].append("Set find_unused_parameters=False if no unused parameters")
    
    # Check batch size per GPU
    if hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'batch_size'):
        batch_size_per_gpu = trainer.datamodule.batch_size
        effective_batch_size = batch_size_per_gpu * world_size
        
        validation["config"]["batch_size_per_gpu"] = batch_size_per_gpu
        validation["config"]["effective_batch_size"] = effective_batch_size
        
        if batch_size_per_gpu < 16:
            validation["warnings"].append(f"Small batch size per GPU: {batch_size_per_gpu}")
            validation["suggestions"].append("Consider increasing batch size per GPU for better GPU utilization")
    
    # Check GPU memory utilization if available
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        memory_utilization = current_memory / max_memory
        
        validation["config"]["gpu_memory_utilization"] = memory_utilization
        
        if memory_utilization < 0.3:
            validation["suggestions"].append("Low GPU memory utilization - consider increasing batch size")
        elif memory_utilization > 0.9:
            validation["warnings"].append("High GPU memory utilization - risk of OOM")
    
    return validation


@rank_zero_only
def log_ddp_best_practices_summary(trainer, model) -> None:
    """Log a comprehensive summary of DDP best practices and current configuration.
    
    Args:
        trainer: PyTorch Lightning trainer
        model: PyTorch Lightning model
    """
    log.info("=== DDP Configuration Summary ===")
    
    # Validate configuration
    validation = validate_ddp_config(trainer)
    
    # Log configuration
    for key, value in validation["config"].items():
        log.info(f"{key}: {value}")
    
    # Log warnings
    if validation["warnings"]:
        log.warning("Configuration warnings:")
        for warning in validation["warnings"]:
            log.warning(f"  - {warning}")
    
    # Log suggestions
    if validation["suggestions"]:
        log.info("Optimization suggestions:")
        for suggestion in validation["suggestions"]:
            log.info(f"  - {suggestion}")
    
    # Log DDP-specific info
    ddp_info = log_ddp_info(trainer)
    if ddp_info.get("ddp/enabled", False):
        log.info(f"DDP Backend: {ddp_info.get('ddp/backend', 'unknown')}")
        log.info(f"World Size: {ddp_info.get('ddp/world_size', 1)}")
        log.info(f"Strategy: {ddp_info.get('ddp/strategy', 'unknown')}")
    
    log.info("=== End DDP Summary ===")
