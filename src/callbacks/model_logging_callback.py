"""Callback to log model parameters after setup is complete."""

from typing import Any
from lightning import Callback, LightningModule, Trainer
from src.utils.logging_utils import log_model_architecture
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class ModelLoggingCallback(Callback):
    """Callback to log model architecture and parameters after setup."""
    
    def __init__(self):
        super().__init__()
        self._logged = False
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log model parameters once training starts and model is initialized."""
        if not self._logged and trainer.is_global_zero:
            try:
                # Get model info
                model_info = log_model_architecture(pl_module)
                
                # Log to all loggers
                for logger in trainer.loggers:
                    try:
                        if hasattr(logger, 'log_metrics'):
                            # Extract only the parameter metrics
                            param_metrics = {k: v for k, v in model_info.items() 
                                           if k.startswith('model/params/')}
                            # Use current step instead of step=0 to avoid conflicts
                            current_step = trainer.global_step if trainer.global_step > 0 else None
                            logger.log_metrics(param_metrics, step=current_step)
                            log.info(f"Logged model parameters to {type(logger).__name__}")
                            
                    except Exception as e:
                        log.warning(f"Failed to log model parameters to {type(logger).__name__}: {e}")
                
                # Log to console
                log.info(f"Model parameters: {model_info.get('model/params/total', 0):,}")
                log.info(f"Trainable parameters: {model_info.get('model/params/trainable', 0):,}")
                log.info(f"Model size: {model_info.get('model/params/total_MB', 0):.2f} MB")
                
                self._logged = True
                
            except Exception as e:
                log.warning(f"Failed to log model architecture in callback: {e}")
