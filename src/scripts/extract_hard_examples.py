#!/usr/bin/env python
"""
Extract hard positive training examples (false negatives) from a checkpoint.

Usage example:

python src/scripts/extract_hard_examples.py \
  --checkpoint logs/fusion_transformer_aggressive_oct17/runs/2025-10-23_blend_sweep/selective_swa_epoch09_12.ckpt \
  --experiment fusion_transformer_aggressive_oct17 \
  --output logs/hard_examples/train_hard_pos.json \
  --split train \
  --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf, DictConfig, ListConfig  # noqa: E402
from omegaconf.base import ContainerMetadata  # noqa: E402
from collections import defaultdict  # noqa: E402
import hydra.utils  # noqa: E402

from src.data.shared_memory_datamodule_v2 import SharedMemoryDataset, TrueSharedMemoryDataModule  # noqa: E402
try:
    from src.data.knn_collate import enhanced_collate_fn
except ImportError:
    enhanced_collate_fn = None

import torch.serialization

torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, Any, list, dict, defaultdict])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract hard positive examples from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint (.ckpt).")
    parser.add_argument(
        "--experiment",
        default="fusion_transformer_aggressive_oct17",
        help="Hydra experiment config to apply (default: fusion_transformer_aggressive_oct17).",
    )
    parser.add_argument(
        "--output",
        default="logs/hard_examples/train_hard_pos.json",
        help="Output JSON file to store hard-positive indices.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Dataset split to analyse for hard positives (default: train).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for positives (default: model.best_thr or 0.5).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep only the top-K hardest positives (lowest probabilities).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device (default: auto).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    return parser.parse_args()


def instantiate_config(overrides: List[str]) -> OmegaConf:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), job_name="extract_hard_examples", version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides, return_hydra_config=False)
    return cfg


def _build_dataset(dm: TrueSharedMemoryDataModule, split: str) -> SharedMemoryDataset:
    indices = {
        "train": dm.train_indices,
        "val": dm.val_indices,
    }[split]
    return SharedMemoryDataset(
        indices=indices,
        shared_tensors=dm.shared_tensors,
        transform=dm.transform,
        enable_knn=dm.enable_knn,
        k_max=dm.k_max,
        pdb_base_dir=dm.pdb_base_dir,
        k_neighbors=dm.hparams.get("k_res_neighbors", 1),
        neighbor_weighting=dm.hparams.get("neighbor_weighting", "softmax"),
        neighbor_temp=dm.hparams.get("neighbor_temp", 2.0),
    )


def _prepare_datamodule(cfg: OmegaConf, batch_size: int, num_workers: int) -> TrueSharedMemoryDataModule:
    dm: TrueSharedMemoryDataModule = hydra.utils.instantiate(cfg.data, _convert_="partial")
    dm.batch_size = batch_size
    dm.num_workers = num_workers
    dm.pin_memory = True
    dm.trainer = SimpleNamespace(world_size=1, global_rank=0, strategy=None)
    dm.prepare_data()
    dm.setup(stage="fit")
    return dm


def _select_device(device_flag: str) -> torch.device:
    if device_flag == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_flag)


def _tensor_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def main() -> None:
    args = parse_args()

    overrides = [
        f"experiment={args.experiment}",
        "train=false",
        "test=false",
        "logger=csv",
        "callbacks=default",  # callback config ignored; no trainer instantiated
    ]
    cfg = instantiate_config(overrides)

    device = _select_device(args.device)
    torch.set_grad_enabled(False)

    dm = _prepare_datamodule(cfg, args.batch_size, args.num_workers)
    dataset = _build_dataset(dm, args.split)
    collate_fn = enhanced_collate_fn if enhanced_collate_fn is not None else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    threshold = args.threshold
    if threshold is None:
        threshold = float(getattr(model, "best_thr", 0.5))
    threshold = float(threshold)

    hard_examples: List[tuple[int, float]] = []

    progress = tqdm(loader, disable=args.no_progress, desc=f"Scanning {args.split} split")
    for batch in progress:
        h5_indices = batch["h5_index"]
        labels = batch["label"].float()
        batch_dev = _tensor_to_device(batch, device)
        logits = model(batch_dev["tabular"], batch=batch_dev)
        probs = torch.sigmoid(logits).detach().cpu()
        labels_cpu = labels.long().cpu()

        for idx, label, prob in zip(h5_indices, labels_cpu, probs):
            if label.item() == 1 and prob.item() < threshold:
                hard_examples.append((int(idx), float(prob.item())))

    hard_examples.sort(key=lambda x: x[1])
    if args.top_k is not None and args.top_k < len(hard_examples):
        hard_examples = hard_examples[: args.top_k]

    indices_only = [idx for idx, _ in hard_examples]
    probs_only = [prob for _, prob in hard_examples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "experiment": args.experiment,
        "split": args.split,
        "threshold": threshold,
        "count": len(indices_only),
        "indices": indices_only,
        "probabilities": probs_only,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Wrote {len(indices_only)} hard positive indices "
        f"(lowest prob {probs_only[0] if probs_only else math.nan:.4f}) to {output_path}"
    )


if __name__ == "__main__":
    main()
