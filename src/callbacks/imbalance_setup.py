"""
Callback to automatically set pos_weight and initial bias for imbalanced datasets.
Computes actual class distribution from training data at fit start.
"""

import torch
import math
import logging
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


class ImbalanceSetup(Callback):
    """
    Automatically configure model for imbalanced binary classification.
    
    At training start:
    1. Computes pos_weight from actual train split class distribution
    2. Sets initial bias for single-logit heads based on prevalence
    3. Logs statistics for monitoring
    """

    def on_fit_start(self, trainer, pl_module):
        """Set up imbalanced learning parameters from training data."""
        
        # Get training labels from datamodule
        dm = trainer.datamodule
        if not hasattr(dm, 'train_indices') or not hasattr(dm, 'shared_tensors'):
            log.warning("DataModule missing train_indices or shared_tensors - skipping imbalance setup")
            return
            
        # Extract training labels (use numpy since shared_tensors are memory-mapped)
        all_labels = dm.shared_tensors['labels']  # This is a numpy memmap or torch tensor
        train_labels = all_labels[dm.train_indices]
        
        # Convert to torch tensor if needed
        if hasattr(train_labels, 'numpy'):
            y = torch.from_numpy(train_labels.numpy())
        else:
            y = torch.as_tensor(train_labels, dtype=torch.long)
        
        # Compute class statistics
        pos = (y == 1).sum().item()
        neg = (y == 0).sum().item()
        total = pos + neg
        
        if total == 0:
            log.warning("No training samples found - skipping imbalance setup")
            return
            
        prevalence = pos / total
        pos_weight = neg / max(1, pos)  # neg/pos ratio for BCE loss weighting
        
        log.info(f"Training set statistics:")
        log.info(f"  Positive samples: {pos:,} ({100*prevalence:.2f}%)")
        log.info(f"  Negative samples: {neg:,} ({100*(1-prevalence):.2f}%)")
        log.info(f"  Total samples: {total:,}")
        log.info(f"  Computed pos_weight: {pos_weight:.3f}")
        
        # Update model's pos_weight buffer
        if hasattr(pl_module, "pos_weight"):
            device = pl_module.device if hasattr(pl_module, 'device') else 'cpu'
            pl_module.pos_weight.data = torch.tensor(pos_weight, device=device, dtype=torch.float32)
            log.info(f"✅ Updated model pos_weight to {pos_weight:.3f}")
        else:
            log.warning("Model missing pos_weight buffer - cannot set automatic weighting")
        
        # Set initial bias for single-logit classifier heads
        bias_init = math.log(prevalence / max(1e-6, 1 - prevalence))
        
        # Handle different fusion architectures
        if hasattr(pl_module, "classifier") and pl_module.classifier is not None:
            # Early fusion (concat, gated, attention, film)
            last_layer = pl_module.classifier[-1]
            if isinstance(last_layer, torch.nn.Linear) and last_layer.out_features == 1:
                with torch.no_grad():
                    last_layer.bias.data.fill_(bias_init)
                log.info(f"✅ Set classifier bias to {bias_init:.4f}")
                    
        elif hasattr(pl_module, "late_head"):
            # Late fusion - set bias for both heads
            if hasattr(pl_module.late_head, 'head_tab') and hasattr(pl_module.late_head.head_tab, 'bias'):
                with torch.no_grad():
                    pl_module.late_head.head_tab.bias.data.fill_(bias_init)
                    pl_module.late_head.head_esm.bias.data.fill_(bias_init)
                log.info(f"✅ Set late fusion head biases to {bias_init:.4f}")
        
        # Log metrics to wandb/logger
        if trainer.logger is not None:
            metrics = {
                "imbalance/pos_weight": pos_weight,
                "imbalance/train_prevalence": prevalence,
                "imbalance/train_pos_samples": pos,
                "imbalance/train_neg_samples": neg,
                "imbalance/bias_init": bias_init
            }
            
            try:
                trainer.logger.log_metrics(metrics, step=trainer.global_step)
                log.info("✅ Logged imbalance metrics to wandb")
            except Exception as e:
                log.warning(f"Failed to log imbalance metrics: {e}")
        
        log.info("🎯 Imbalance setup complete!")