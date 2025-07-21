#!/usr/bin/env python3
"""
Training script for ESM + Tabular model using H5 datasets.

This script uses the existing PockNet training infrastructure with Hydra
and PyTorch Lightning to train a model that combines:
1. Tabular features (45D) from physicochemical properties  
2. Residue-specific ESM2 embeddings (2560D) with proper mapping
3. Attention-based fusion for optimal performance

Usage:
    python train_esm_h5.py experiment=esm_tabular_h5
    python train_esm_h5.py experiment=esm_tabular_h5 model.fusion_method=concat
    python train_esm_h5.py experiment=esm_tabular_h5 trainer.devices=[0,1]
"""

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the ESM + Tabular model using H5 datasets.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("=" * 80)
    log.info("🚀 ESM + Tabular H5 Training Pipeline")
    log.info("=" * 80)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Setup data and log dataset information
    log.info("Setting up datasets...")
    datamodule.setup("fit")
    
    log.info("📊 Dataset Information:")
    log.info(f"   Train samples: {len(datamodule.train_dataset):,}")
    log.info(f"   Val samples: {len(datamodule.val_dataset):,}")
    
    if hasattr(datamodule, 'test_dataset') and datamodule.test_dataset:
        datamodule.setup("test")
        log.info(f"   Test samples: {len(datamodule.test_dataset):,}")
    
    # Log model information
    log.info("🧠 Model Information:")
    log.info(f"   Tabular dim: {model.hparams.tabular_dim}")
    log.info(f"   ESM dim: {model.hparams.esm_dim}")
    log.info(f"   Fusion method: {model.hparams.fusion_method}")
    log.info(f"   Hidden dims: {model.hparams.hidden_dims}")
    log.info(f"   Dropout: {model.hparams.dropout}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"   Total parameters: {total_params:,}")
    log.info(f"   Trainable parameters: {trainable_params:,}")

    if cfg.get("train"):
        log.info("🏋️ Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("🧪 Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    
    # Log final results
    log.info("📈 Final Results:")
    if "val/f1" in metric_dict:
        log.info(f"   Best Val F1: {metric_dict['val/f1']:.4f}")
    if "val/auroc" in metric_dict:
        log.info(f"   Best Val AUROC: {metric_dict['val/auroc']:.4f}")
    if "test/f1" in metric_dict:
        log.info(f"   Test F1: {metric_dict['test/f1']:.4f}")
    if "test/auroc" in metric_dict:
        log.info(f"   Test AUROC: {metric_dict['test/auroc']:.4f}")

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for ESM + Tabular H5 training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
