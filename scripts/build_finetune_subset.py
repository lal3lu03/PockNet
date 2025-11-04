#!/usr/bin/env python3
"""
Build a targeted fine-tuning subset focused on uncertain positives and border negatives.

The script runs a checkpoint over the training split, mines samples near the decision
boundary, and writes a JSON file with curated indices that can be consumed through
`data.train_indices_override_path`.

Example:
    python scripts/build_finetune_subset.py \
        --checkpoint path/to/model.ckpt \
        --experiment fusion_transformer_aggressive \
        --output logs/finetune/curated_indices.json \
        --per-protein-metrics outputs/p2rank_like_run/summary/per_protein_metrics.csv \
        --hard-positive-file logs/hard_examples/train_hard_pos.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from random import Random
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import DictConfig, ListConfig, OmegaConf  # noqa: E402
from omegaconf.base import ContainerMetadata  # noqa: E402
import hydra.utils  # noqa: E402

from src.data.shared_memory_datamodule_v2 import (  # noqa: E402
    SharedMemoryDataset,
    TrueSharedMemoryDataModule,
    _bytes_to_str,
)

try:  # noqa: E402
    from src.data.knn_collate import enhanced_collate_fn  # type: ignore
except ImportError:  # pragma: no cover
    enhanced_collate_fn = None  # type: ignore

torch.serialization.add_safe_globals(
    [DictConfig, ListConfig, ContainerMetadata, Dict, list, dict, defaultdict]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate a fine-tuning subset around decision boundaries.")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint (.ckpt) to analyse.")
    parser.add_argument(
        "--experiment",
        default="fusion_transformer_aggressive",
        help="Hydra experiment config to reuse (default: fusion_transformer_aggressive).",
    )
    parser.add_argument(
        "--output",
        default="logs/finetune/curated_indices.json",
        help="Destination JSON for curated indices.",
    )
    parser.add_argument(
        "--per-protein-metrics",
        default="outputs/p2rank_like_run/summary/per_protein_metrics.csv",
        help="CSV with per-protein IoU metrics to identify underperformers.",
    )
    parser.add_argument(
        "--underperforming-top-k",
        type=int,
        default=30,
        help="Number of lowest-IoU proteins to prioritise (default: 30).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold override (default: checkpoint best_thr or 0.5).",
    )
    parser.add_argument(
        "--pos-band",
        type=float,
        default=0.15,
        help="Probability window |p-thr| <= pos-band for uncertain positives.",
    )
    parser.add_argument(
        "--neg-band",
        type=float,
        default=0.12,
        help="Margin below threshold allowed for border negatives (default: 0.12).",
    )
    parser.add_argument(
        "--safe-max-prob",
        type=float,
        default=0.05,
        help="Upper bound on probability for \"safe\" negatives.",
    )
    parser.add_argument(
        "--positives-limit",
        type=int,
        default=1800,
        help="Maximum number of unique positive indices to keep before repeats.",
    )
    parser.add_argument(
        "--border-neg-per-pos",
        type=float,
        default=1.0,
        help="Border negatives per selected positive (default 1.0).",
    )
    parser.add_argument(
        "--safe-neg-per-pos",
        type=float,
        default=0.5,
        help="Safe negatives per selected positive (default 0.5).",
    )
    parser.add_argument(
        "--positive-repeat",
        type=int,
        default=1,
        help="Repeat factor applied to each positive index in the final list.",
    )
    parser.add_argument(
        "--border-repeat",
        type=int,
        default=1,
        help="Repeat factor applied to each border negative index.",
    )
    parser.add_argument(
        "--safe-repeat",
        type=int,
        default=1,
        help="Repeat factor applied to each safe negative index.",
    )
    parser.add_argument(
        "--safe-neg-limit",
        type=int,
        default=2000,
        help="Hard cap on the number of unique safe negatives sampled.",
    )
    parser.add_argument(
        "--hard-positive-file",
        default=None,
        help="Optional JSON from extract_hard_examples.py to seed positives.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size (default: 512).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device for inference (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for sampling and shuffling.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle final train_indices (default: True).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_false",
        dest="shuffle",
        help="Disable shuffling of the final train_indices.",
    )
    parser.set_defaults(shuffle=True)
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bar (default: True).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="Disable tqdm progress bar.",
    )
    parser.set_defaults(progress=True)
    return parser.parse_args()


def _resolve_config(overrides: Sequence[str]) -> OmegaConf:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), job_name="build_finetune_subset", version_base="1.3"):
        cfg = compose(config_name="train", overrides=list(overrides), return_hydra_config=False)
    return cfg


def _prepare_datamodule(cfg: OmegaConf, batch_size: int, num_workers: int) -> TrueSharedMemoryDataModule:
    dm: TrueSharedMemoryDataModule = hydra.utils.instantiate(cfg.data, _convert_="partial")
    dm.batch_size = batch_size
    dm.num_workers = num_workers
    dm.pin_memory = True
    # Ensure we scan the baseline train split (ignore any override from config).
    dm.train_indices_override_path = None
    dm.train_override_shuffle_seed = None
    dm.trainer = SimpleNamespace(world_size=1, global_rank=0, strategy=None)
    dm.prepare_data()
    dm.setup(stage="fit")
    return dm


def _select_device(flag: str) -> torch.device:
    if flag == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(flag)


def _tensor_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _load_underperformers(csv_path: Path, top_k: int) -> Set[str]:
    if top_k <= 0 or not csv_path.exists():
        return set()
    scores: List[Tuple[float, str]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pid = row.get("protein_id")
            if not pid:
                continue
            try:
                score = float(row.get("mean_gt_iou_default", row.get("best_iou", "inf")))
            except ValueError:
                continue
            scores.append((score, pid))
    scores.sort(key=lambda item: item[0])
    return {pid for _, pid in scores[:top_k]}


def _load_hard_positive_indices(path: Optional[Path], train_set: Set[int]) -> List[int]:
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to parse hard-positive file {path}: {exc}") from exc
    if isinstance(payload, dict):
        indices = payload.get("indices", [])
    else:
        indices = payload
    filtered = []
    for idx in indices:
        try:
            val = int(idx)
        except (TypeError, ValueError):
            continue
        if val in train_set:
            filtered.append(val)
    return filtered


def _reservoir_add(
    pool: List[Tuple[float, int, float, str]],
    capacity: int,
    item: Tuple[float, int, float, str],
    seen: int,
    rng: Random,
) -> int:
    seen += 1
    if capacity <= 0:
        return seen
    if len(pool) < capacity:
        pool.append(item)
    else:
        j = rng.randint(0, seen - 1)
        if j < capacity:
            pool[j] = item
    return seen


def main() -> None:
    args = parse_args()

    overrides = [
        f"experiment={args.experiment}",
        "train=false",
        "test=false",
        "logger=csv",
        "callbacks=default",
    ]
    cfg = _resolve_config(overrides)

    device = _select_device(args.device)
    torch.set_grad_enabled(False)

    dm = _prepare_datamodule(cfg, args.batch_size, args.num_workers)
    dataset = SharedMemoryDataset(
        indices=dm.train_indices,
        shared_tensors=dm.shared_tensors,
        transform=dm.transform,
        enable_knn=dm.enable_knn,
        k_max=dm.k_max,
        pdb_base_dir=dm.pdb_base_dir,
        k_neighbors=dm.hparams.get("k_res_neighbors", 1),
        neighbor_weighting=dm.hparams.get("neighbor_weighting", "softmax"),
        neighbor_temp=dm.hparams.get("neighbor_temp", 2.0),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=enhanced_collate_fn if enhanced_collate_fn is not None else None,
    )

    train_index_set: Set[int] = {int(x) for x in np.asarray(dm.train_indices, dtype=np.int64)}  # type: ignore[name-defined]

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    threshold = float(args.threshold) if args.threshold is not None else float(getattr(model, "best_thr", 0.5))
    pos_band = max(args.pos_band, 0.0)
    neg_band = max(args.neg_band, 0.0)
    safe_max_prob = max(min(args.safe_max_prob, 0.5), 0.0)

    per_protein_path = Path(args.per_protein_metrics)
    underperformers = _load_underperformers(per_protein_path, args.underperforming_top_k)

    rng = Random(args.seed)

    pos_targeted: List[Tuple[float, int, float, str]] = []
    pos_other: List[Tuple[float, int, float, str]] = []
    border_targeted: List[Tuple[float, int, float, str]] = []
    border_other: List[Tuple[float, int, float, str]] = []
    safe_targeted: List[Tuple[float, int, float, str]] = []
    safe_other: List[Tuple[float, int, float, str]] = []
    safe_target_seen = 0
    safe_other_seen = 0

    stats = {
        "samples_processed": 0,
        "positive_seen": 0,
        "negative_seen": 0,
    }

    progress = tqdm(loader, disable=not args.progress, desc="Scanning train split")
    for batch in progress:
        batch_dev = _tensor_to_device(batch, device)
        logits = model(batch_dev["tabular"], batch=batch_dev)
        probs = torch.sigmoid(logits).detach().cpu()
        labels = torch.as_tensor(batch["label"]).long().cpu()
        indices = torch.as_tensor(batch["h5_index"]).long().cpu()

        protein_field = batch.get("protein_id")
        if isinstance(protein_field, list):
            proteins = [str(p) for p in protein_field]
        elif protein_field is None:
            proteins = ["unknown"] * len(indices)
        else:
            proteins = [str(protein_field)] * len(indices)

        stats["samples_processed"] += len(indices)

        for idx_t, label_t, prob_t, protein in zip(indices, labels, probs, proteins):
            idx = int(idx_t.item())
            prob = float(prob_t.item())
            label = int(label_t.item())
            is_underperformer = protein in underperformers
            margin = abs(prob - threshold)

            if label == 1:
                stats["positive_seen"] += 1
                if prob < threshold or margin <= pos_band:
                    target = pos_targeted if is_underperformer else pos_other
                    target.append((margin, idx, prob, protein))
            else:
                stats["negative_seen"] += 1
                if prob >= threshold - neg_band:
                    target = border_targeted if is_underperformer else border_other
                    target.append((margin, idx, prob, protein))
                elif prob <= safe_max_prob:
                    candidate = (prob, idx, prob, protein)
                    if is_underperformer:
                        safe_target_seen = _reservoir_add(
                            safe_targeted, max(args.safe_neg_limit, 1), candidate, safe_target_seen, rng
                        )
                    else:
                        safe_other_seen = _reservoir_add(
                            safe_other, max(args.safe_neg_limit, 1), candidate, safe_other_seen, rng
                        )

    hard_positive_file = Path(args.hard_positive_file) if args.hard_positive_file else None
    hard_positive_indices = _load_hard_positive_indices(hard_positive_file, train_index_set)

    def _select_candidates(
        primary: List[Tuple[float, int, float, str]],
        secondary: List[Tuple[float, int, float, str]],
        limit: int,
    ) -> List[int]:
        ordered: List[Tuple[float, int, float, str]] = sorted(primary, key=lambda x: x[0])
        if len(ordered) < limit:
            ordered.extend(sorted(secondary, key=lambda x: x[0])[: max(0, limit - len(ordered))])
        selection: List[int] = []
        seen_local: Set[int] = set()
        for _, idx, _, _ in ordered:
            if idx in seen_local:
                continue
            selection.append(idx)
            seen_local.add(idx)
            if len(selection) >= limit:
                break
        return selection

    positives_cap = args.positives_limit if args.positives_limit > 0 else (
        len(pos_targeted) + len(pos_other) + len(hard_positive_indices)
    )

    selected_positive: List[int] = []
    # Hard positives first (respecting cap)
    for idx in hard_positive_indices:
        if idx not in selected_positive:
            selected_positive.append(idx)
        if len(selected_positive) >= positives_cap:
            break

    if len(selected_positive) < positives_cap:
        remaining = positives_cap - len(selected_positive)
        mined = _select_candidates(pos_targeted, pos_other, remaining)
        for idx in mined:
            if idx not in selected_positive:
                selected_positive.append(idx)

    border_cap = int(math.ceil(len(selected_positive) * max(args.border_neg_per_pos, 0.0)))
    border_cap = max(border_cap, 0)
    selected_border = _select_candidates(border_targeted, border_other, border_cap)

    safe_cap = int(math.ceil(len(selected_positive) * max(args.safe_neg_per_pos, 0.0)))
    safe_cap = min(max(safe_cap, 0), args.safe_neg_limit)

    def _sample_safe(pool: List[Tuple[float, int, float, str]], remaining: int) -> List[int]:
        if remaining <= 0:
            return []
        if len(pool) <= remaining:
            return [idx for (_, idx, _, _) in sorted(pool, key=lambda x: x[0])]
        return [idx for (_, idx, _, _) in rng.sample(pool, remaining)]

    safe_selected: List[int] = []
    if safe_cap > 0:
        safe_primary = _sample_safe(safe_targeted, min(safe_cap, len(safe_targeted)))
        safe_selected.extend(safe_primary)
        residual = safe_cap - len(safe_selected)
        if residual > 0:
            safe_selected.extend(_sample_safe(safe_other, residual))

    def _repeat_indices(indices: List[int], repeat: int) -> List[int]:
        repeat = max(1, int(repeat))
        if repeat == 1:
            return list(indices)
        expanded: List[int] = []
        for idx in indices:
            expanded.extend([idx] * repeat)
        return expanded

    train_indices: List[int] = []
    train_indices.extend(_repeat_indices(selected_positive, args.positive_repeat))
    train_indices.extend(_repeat_indices(selected_border, args.border_repeat))
    train_indices.extend(_repeat_indices(safe_selected, args.safe_repeat))

    if args.shuffle:
        rng.shuffle(train_indices)

    shared = dm.shared_tensors

    def _decode_protein(idx: int) -> str:
        return _bytes_to_str(shared["protein_keys"][idx])

    groups = {
        "positives": sorted({idx for idx in selected_positive}),
        "border_negatives": sorted({idx for idx in selected_border}),
        "safe_negatives": sorted({idx for idx in safe_selected}),
    }

    counts_after_repeat = {
        "positives": len(selected_positive) * max(1, args.positive_repeat),
        "border_negatives": len(selected_border) * max(1, args.border_repeat),
        "safe_negatives": len(safe_selected) * max(1, args.safe_repeat),
    }

    proteins_in_set = sorted({ _decode_protein(idx) for idx in set(train_indices) })

    output_payload = {
        "train_indices": [int(idx) for idx in train_indices],
        "groups": {name: [int(idx) for idx in indices] for name, indices in groups.items()},
        "metadata": {
            "threshold": threshold,
            "pos_band": pos_band,
            "neg_band": neg_band,
            "safe_max_prob": safe_max_prob,
            "underperforming_top_k": args.underperforming_top_k,
            "underperforming_proteins": sorted(list(underperformers)),
            "positive_repeat": max(1, args.positive_repeat),
            "border_repeat": max(1, args.border_repeat),
            "safe_repeat": max(1, args.safe_repeat),
            "counts_unique": {name: len(indices) for name, indices in groups.items()},
            "counts_after_repeat": counts_after_repeat,
            "samples_processed": stats["samples_processed"],
            "positives_scanned": stats["positive_seen"],
            "negatives_scanned": stats["negative_seen"],
            "scan_split_size": len(dm.train_indices),
            "source_checkpoint": os.path.abspath(args.checkpoint),
            "hard_positive_file": str(hard_positive_file) if hard_positive_file else None,
            "proteins_covered": proteins_in_set,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2))

    print(f"Wrote curated fine-tune indices to {output_path} "
          f"({len(train_indices)} total samples; "
          f"{counts_after_repeat['positives']} positives / "
          f"{counts_after_repeat['border_negatives']} border negs / "
          f"{counts_after_repeat['safe_negatives']} safe negs).")


if __name__ == "__main__":
    main()
