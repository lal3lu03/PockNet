#!/usr/bin/env python3
"""
Production P2Rank-Style Post-Processing
======================================

Runs PockNet inference on an H5 dataset and applies a faithful Python
implementation of the P2Rank pocket aggregation stage so that the final
outputs match the original tool's `pockets.csv` format.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from post_processing.inference import ModelInference
from post_processing.p2rank_like import (
    P2RankParams,
    P2RankPostProcessor,
    ProteinPointLoader,
    create_ground_truth_pockets,
    pocket_iou_metrics,
    pockets_to_csv,
)
from post_processing.score_transformers import ScoreTransformer, load_score_transformer

logger = logging.getLogger("pocknet.p2rank_production")

_DEFAULT_ZSCORE_REL = Path("tmp/p2rank/distro/models/_score_transform/default_ZscoreTpTransformer.json")
_DEFAULT_PROB_REL = Path("tmp/p2rank/distro/models/_score_transform/default_ProbabilityScoreTransformer.json")
_DEFAULT_POSTPROC_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))


def _resolve_default_transform(relative_path: Path) -> Optional[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / relative_path
    return candidate if candidate.exists() else None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _select_proteins_by_split(
    h5_path: Path,
    protein_index: ProteinPointLoader,
    split_mode: str,
) -> List[str]:
    """Filter proteins according to the split metadata stored in the H5 file."""
    proteins = protein_index.h5_index.proteins()
    if not proteins:
        return []

    split_mode = split_mode.lower()
    if split_mode == "all":
        return proteins

    code_map = {
        "train": {0},
        "val": {1},
        "test": {2},
        "trainval": {0, 1},
    }

    codes = code_map.get(split_mode)
    if codes is None:
        logger.warning("Unknown split mode '%s'; processing all proteins.", split_mode)
        return proteins

    with h5py.File(h5_path, "r") as fh:
        if "split" not in fh:
            return proteins

        split = fh["split"]
        keep: List[str] = []
        for protein in proteins:
            start, _ = protein_index.h5_index.slice(protein)
            label = int(split[start])
            if label in codes:
                keep.append(protein)

        if not keep:
            logger.warning("No proteins found for split '%s'; falling back to all.", split_mode)
            return proteins
        return keep


def _prepare_predictions(
    checkpoint: Path,
    h5_path: Path,
    protein_ids: List[str],
    device: Optional[str],
    batch_size: int,
    use_shared_memory: bool,
) -> Tuple[Dict[str, np.ndarray], str]:
    """Run model inference and return sigmoid probabilities per protein."""
    start = time.time()
    model = ModelInference(str(checkpoint), device=device)
    preds = model.predict_from_h5(
        str(h5_path),
        protein_ids,
        batch_size=batch_size,
        use_shared_memory=use_shared_memory,
    )
    elapsed = time.time() - start
    logger.info("Inference completed for %d proteins in %.2fs", len(preds), elapsed)
    return preds, str(model.device)


def _write_pockets_csv(output_dir: Path, protein_id: str, csv_text: str) -> None:
    protein_dir = output_dir / protein_id
    protein_dir.mkdir(parents=True, exist_ok=True)
    target = protein_dir / "pockets.csv"
    target.write_text(csv_text)


def _write_summary(output_dir: Path, metrics: Dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    lines = ["metric,value"]
    for key, value in metrics.items():
        lines.append(f"{key},{value}")
    summary_path.write_text("\n".join(lines) + "\n")


def _write_per_protein_metrics(output_dir: Path, results: Dict[str, Dict[str, float]]) -> None:
    if not results:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "per_protein_metrics.csv"
    field_order = [
        "n_points",
        "n_pockets_total",
        "n_pockets_eval",
        "max_score",
        "max_score_eval",
        "sum_scores_total",
        "sum_scores_eval",
        "mean_gt_iou_default",
        "best_iou",
        "gt_coverage",
    ]
    header = ",".join(["protein_id"] + field_order)
    lines = [header]
    for protein_id in sorted(results.keys()):
        entry = results[protein_id]
        cells = [protein_id]
        for field in field_order:
            value = entry.get(field, "")
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")


def _write_threshold_csv(output_dir: Path, sweep_results: List[Dict[str, float]]) -> None:
    if not sweep_results:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["threshold,mean_gt_iou"]
    for entry in sweep_results:
        lines.append(f"{entry['threshold']:.4f},{entry['mean_gt_iou']:.6f}")
    (output_dir / "threshold_sweep.csv").write_text("\n".join(lines) + "\n")


def _parse_threshold_grid(arg: Optional[str]) -> Optional[List[float]]:
    if arg is None:
        return None
    arg = arg.strip().lower()
    if not arg:
        return None
    if arg == "auto":
        return [round(x, 3) for x in np.linspace(0.1, 0.9, 17)]
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid threshold grid: {arg}") from exc


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _platt_fit(scores: np.ndarray, labels: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> Tuple[float, float]:
    a = 1.0
    b = 0.0
    y = labels.astype(np.float64)
    x = scores.astype(np.float64)
    lr = 5e-3
    reg = 1e-4

    for _ in range(max_iter):
        z = np.clip(a * x + b, -50.0, 50.0)
        s = _sigmoid(z)
        diff = s - y
        grad_a = np.sum(diff * x) + reg * a
        grad_b = np.sum(diff) + reg * b

        a_prev, b_prev = a, b
        a -= lr * grad_a
        b -= lr * grad_b

        if max(abs(a - a_prev), abs(b - b_prev)) < tol:
            break

    return float(a), float(b)


def _apply_platt(probabilities: np.ndarray, a: float, b: float) -> np.ndarray:
    scores = np.clip(probabilities, 0.0, 1.0)
    adjusted = _sigmoid(a * scores + b)
    return adjusted.astype(np.float32)


def _save_calibration(path: Path, a: float, b: float) -> None:
    payload = {"type": "platt", "a": a, "b": b}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _load_calibration(path: Path) -> Optional[Tuple[float, float]]:
    if not path.exists():
        logger.warning("Calibration file not found: %s", path)
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load calibration %s: %s", path, exc)
        return None
    if payload.get("type") != "platt":
        logger.error("Unsupported calibration type in %s", path)
        return None
    try:
        return float(payload["a"]), float(payload["b"])
    except Exception as exc:  # noqa: BLE001
        logger.error("Invalid calibration payload in %s: %s", path, exc)
        return None


def _fit_platt_calibration(
    loader: ProteinPointLoader,
    predictions: Dict[str, np.ndarray],
    proteins: List[str],
) -> Optional[Tuple[float, float]]:
    score_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for pid in proteins:
        probs = predictions.get(pid)
        if probs is None or len(probs) == 0:
            continue
        _, _, labels = loader.get_protein_arrays(pid)
        if len(labels) == 0:
            continue
        n = min(len(labels), len(probs))
        score_list.append(np.clip(probs[:n], 0.0, 1.0))
        labels_list.append(labels[:n].astype(np.float64))

    if not score_list:
        logger.warning("Calibration requested but no predictions with labels were found.")
        return None

    scores = np.concatenate(score_list)
    labels = np.concatenate(labels_list)
    unique = np.unique(labels)
    if len(unique) < 2:
        logger.warning("Calibration skipped: labels contain only a single class.")
        return None

    a, b = _platt_fit(scores, labels)
    logger.info("Fitted Platt calibration: a=%.4f b=%.4f", a, b)
    return a, b


def run_production_pipeline(
    checkpoint: Path,
    h5_path: Path,
    csv_path: Path,
    output_root: Path,
    max_proteins: Optional[int] = None,
    params: Optional[P2RankParams] = None,
    threshold_grid: Optional[List[float]] = None,
    split_mode: str = "test",
    zscore_transformer: Optional[ScoreTransformer] = None,
    probability_transformer: Optional[ScoreTransformer] = None,
    device: str = "auto",
    batch_size: int = 2048,
    use_shared_memory: bool = True,
    postproc_workers: int = 1,
    eval_topk: Optional[int] = None,
    min_pocket_score: Optional[float] = None,
    min_pocket_size: Optional[int] = None,
    max_pockets_all: Optional[int] = None,
    calibration_path: Optional[Path] = None,
    fit_calibration_path: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Execute the full inference + post-processing pipeline.

    Returns a dictionary with per-protein metadata for further analysis.
    """
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Checkpoint: %s", checkpoint)
    logger.info("Dataset:    %s", h5_path)
    logger.info("CSV source: %s", csv_path)
    logger.info("Output dir: %s", output_root)

    params = params or P2RankParams()

    loader = ProteinPointLoader(h5_path, csv_path, cache_dir=output_root / "cache")
    all_proteins = _select_proteins_by_split(h5_path, loader, split_mode)
    if max_proteins is not None:
        all_proteins = all_proteins[:max_proteins]

    if not all_proteins:
        logger.error("No proteins found to process.")
        return {}

    logger.info("Processing %d proteins", len(all_proteins))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("First proteins scheduled: %s", all_proteins[:5])

    device_for_model: Optional[str] = None if device == "auto" else device

    try:
        predictions, resolved_device = _prepare_predictions(
            checkpoint,
            h5_path,
            all_proteins,
            device_for_model,
            batch_size,
            use_shared_memory,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Inference failed: %s", exc)
        raise

    logger.info(
        "Inference device resolved to %s (batch_size=%d, shared_memory=%s)",
        resolved_device,
        batch_size,
        use_shared_memory,
    )

    calibration_params: Optional[Tuple[float, float]] = None
    if calibration_path is not None:
        calibration_params = _load_calibration(calibration_path)
        if calibration_params is not None:
            logger.info("Loaded calibration from %s", calibration_path)

    if fit_calibration_path is not None:
        fitted = _fit_platt_calibration(loader, predictions, all_proteins)
        if fitted is not None:
            calibration_params = fitted
            _save_calibration(fit_calibration_path, *fitted)
            logger.info("Saved calibration to %s", fit_calibration_path)

    if calibration_params is not None:
        a, b = calibration_params
        for pid, probs in predictions.items():
            predictions[pid] = _apply_platt(probs, a, b)
        logger.info("Applied Platt calibration (a=%.4f, b=%.4f) to predictions", a, b)

    pockets_output_dir = output_root / "cases"
    pockets_output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}
    total_pockets_eval = 0
    total_pockets_all = 0
    pocket_scores_eval: List[float] = []
    pocket_scores_all: List[float] = []
    gt_mean_ious: List[float] = []
    best_iou_vals: List[float] = []
    coverage_vals: List[float] = []
    protein_cache: Dict[str, Dict[str, object]] = {}

    postproc_workers = max(1, postproc_workers)

    def _process_single(protein_id: str) -> Dict[str, object]:
        logger.debug("Starting aggregation for %s", protein_id)
        probs = predictions.get(protein_id)
        if probs is None or len(probs) == 0:
            return {
                "protein_id": protein_id,
                "skip": f"Skipping {protein_id} (no predictions returned)",
            }

        coords, residue_numbers, labels = loader.get_protein_arrays(protein_id)
        if len(coords) == 0:
            return {
                "protein_id": protein_id,
                "skip": f"Skipping {protein_id} (no coordinates)",
            }

        local_processor = P2RankPostProcessor(
            params,
            zscore_transformer=zscore_transformer,
            probability_transformer=probability_transformer,
        )

        gt_pockets = create_ground_truth_pockets(
            protein_id,
            coords,
            labels,
            residue_numbers,
            params,
        )

        pockets = local_processor.process(protein_id, coords, probs, residue_numbers)

        if min_pocket_score is not None:
            pockets = [p for p in pockets if p.sum_prob >= min_pocket_score]
        if min_pocket_size is not None:
            pockets = [p for p in pockets if p.sample_points >= min_pocket_size]

        pockets_sorted = sorted(pockets, key=lambda p: p.score, reverse=True)
        if max_pockets_all is not None:
            pockets_sorted = pockets_sorted[:max_pockets_all]
        pockets = pockets_sorted

        csv_text = pockets_to_csv(protein_id, pockets)

        pockets_eval = pockets if eval_topk is None else pockets[:eval_topk]

        gt_metrics = pocket_iou_metrics(pockets_eval, gt_pockets)
        logger.debug(
            "Finished %s: pockets=%d mean_gt_iou=%.4f",
            protein_id,
            len(pockets),
            gt_metrics["mean_gt_iou"],
        )

        return {
            "protein_id": protein_id,
            "coords": coords,
            "residue_numbers": residue_numbers,
            "labels": labels,
            "probs": probs,
            "pockets": pockets,
            "pockets_eval": pockets_eval,
            "csv_text": csv_text,
            "gt_pockets": gt_pockets,
            "gt_mean_iou": gt_metrics["mean_gt_iou"],
            "gt_best_iou": gt_metrics["best_iou"],
            "gt_coverage": gt_metrics["gt_coverage"],
            "metrics_entry": {
                "n_points": len(coords),
                "n_pockets_total": len(pockets),
                "n_pockets_eval": len(pockets_eval),
                "max_score": max((p.score for p in pockets), default=0.0),
                "max_score_eval": max((p.score for p in pockets_eval), default=0.0),
                "sum_scores_total": sum(p.score for p in pockets),
                "sum_scores_eval": sum(p.score for p in pockets_eval),
                "mean_gt_iou_default": gt_metrics["mean_gt_iou"],
                "best_iou": gt_metrics["best_iou"],
                "gt_coverage": gt_metrics["gt_coverage"],
            },
        }

    progress_desc = "Post-processing proteins"
    if postproc_workers > 1:
        logger.info("Post-processing proteins with %d worker threads", postproc_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=postproc_workers) as executor:
            processed = list(
                tqdm(
                    executor.map(_process_single, all_proteins),
                    total=len(all_proteins),
                    desc=progress_desc,
                    unit="protein",
                )
            )
    else:
        processed = []
        for pid in tqdm(all_proteins, desc=progress_desc, unit="protein"):
            processed.append(_process_single(pid))

    for item in processed:
        protein_id = item["protein_id"]
        if item.get("skip"):
            logger.warning(item["skip"])
            continue

        pockets = item["pockets"]  # type: ignore[assignment]
        pockets_eval = item["pockets_eval"]  # type: ignore[assignment]
        csv_text = item["csv_text"]  # type: ignore[assignment]
        _write_pockets_csv(pockets_output_dir, protein_id, csv_text)

        total_pockets_eval += len(pockets_eval)
        total_pockets_all += len(pockets)
        pocket_scores_eval.extend([p.score for p in pockets_eval])
        pocket_scores_all.extend([p.score for p in pockets])
        gt_mean_ious.append(item["gt_mean_iou"])  # type: ignore[arg-type]
        best_iou_vals.append(item["gt_best_iou"])  # type: ignore[arg-type]
        coverage_vals.append(item["gt_coverage"])  # type: ignore[arg-type]

        results[protein_id] = item["metrics_entry"]  # type: ignore[assignment]

        logger.info(
            "%s: points=%d pockets_eval=%d pockets_total=%d mean_iou=%.3f",
            protein_id,
            results[protein_id]["n_points"],
            results[protein_id]["n_pockets_eval"],
            results[protein_id]["n_pockets_total"],
            results[protein_id]["mean_gt_iou_default"],
        )

        protein_cache[protein_id] = {
            "coords": item["coords"],
            "residue_numbers": item["residue_numbers"],
            "labels": item["labels"],
            "probs": item["probs"],
            "gt_pockets": item["gt_pockets"],
        }

    n_proteins = len(results)
    metrics = {
        "total_proteins": n_proteins,
        "proteins_attempted": len(all_proteins),
        "total_pockets_eval": total_pockets_eval,
        "total_pockets_all": total_pockets_all,
        "avg_pockets_per_protein_eval": total_pockets_eval / n_proteins if n_proteins else 0.0,
        "avg_pockets_per_protein_all": total_pockets_all / n_proteins if n_proteins else 0.0,
        "avg_pocket_score_eval": float(np.mean(pocket_scores_eval)) if pocket_scores_eval else 0.0,
        "avg_pocket_score_all": float(np.mean(pocket_scores_all)) if pocket_scores_all else 0.0,
        "max_pocket_score_eval": max(pocket_scores_eval) if pocket_scores_eval else 0.0,
        "max_pocket_score_all": max(pocket_scores_all) if pocket_scores_all else 0.0,
        "avg_mean_gt_iou_default": float(np.mean(gt_mean_ious)) if gt_mean_ious else 0.0,
        "avg_best_iou": float(np.mean(best_iou_vals)) if best_iou_vals else 0.0,
        "avg_gt_coverage": float(np.mean(coverage_vals)) if coverage_vals else 0.0,
    }
    metrics["split_mode"] = split_mode
    metrics["inference_device"] = resolved_device
    metrics["inference_batch_size"] = batch_size
    metrics["use_shared_memory"] = int(use_shared_memory)
    metrics["postproc_workers"] = postproc_workers
    if eval_topk is not None:
        metrics["eval_topk"] = eval_topk
    threshold_sweep_results: List[Dict[str, float]] = []
    if threshold_grid:
        base_param_dict = params.__dict__.copy()
        sweep_results: List[Dict[str, float]] = []
        for threshold in tqdm(threshold_grid, desc="Threshold sweep", unit="thr"):
            base_param_dict["pred_point_threshold"] = threshold
            sweep_processor = P2RankPostProcessor(
                P2RankParams(**base_param_dict),
                zscore_transformer=zscore_transformer,
                probability_transformer=probability_transformer,
            )
            per_protein = []
            for protein_id, pdata in protein_cache.items():
                pockets_thresh = sweep_processor.process(
                    protein_id,
                    pdata["coords"],  # type: ignore[arg-type]
                    pdata["probs"],  # type: ignore[arg-type]
                    pdata["residue_numbers"],  # type: ignore[arg-type]
                )
                pockets_thresh_sorted = sorted(pockets_thresh, key=lambda p: p.score, reverse=True)
                if eval_topk is not None:
                    pockets_thresh_sorted = pockets_thresh_sorted[:eval_topk]
                metrics_thresh = pocket_iou_metrics(
                    pockets_thresh_sorted,
                    pdata["gt_pockets"],  # type: ignore[arg-type]
                )
                per_protein.append(metrics_thresh["mean_gt_iou"])
            mean_iou = float(np.mean(per_protein)) if per_protein else 0.0
            sweep_results.append({"threshold": threshold, "mean_gt_iou": mean_iou})

        threshold_sweep_results = sorted(sweep_results, key=lambda x: x["threshold"])
        if threshold_sweep_results:
            best_entry = max(threshold_sweep_results, key=lambda x: x["mean_gt_iou"])
            metrics["best_threshold"] = best_entry["threshold"]
            metrics["best_threshold_iou"] = best_entry["mean_gt_iou"]
            _write_threshold_csv(output_root / "summary", threshold_sweep_results)

    _write_summary(output_root / "summary", metrics)
    _write_per_protein_metrics(output_root / "summary", results)

    logger.info(
        "Finished. Total pockets (eval): %d (all: %d)",
        total_pockets_eval,
        total_pockets_all,
    )
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P2Rank-style production pipeline.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to Lightning checkpoint.")
    parser.add_argument("--h5", type=Path, required=True, help="Inference H5 dataset.")
    parser.add_argument("--csv", type=Path, required=True, help="vectorsTrain_all_chainfix CSV file.")
    parser.add_argument("--output", type=Path, default=Path("post_processing_results/p2rank_production"), help="Output directory.")
    parser.add_argument("--max-proteins", type=int, default=None, help="Limit number of proteins to process.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...).")
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test", "trainval"],
        default="test",
        help="Subset of proteins based on H5 split labels (default: test).",
    )
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default=None,
        help="Comma-separated thresholds to evaluate (e.g. '0.2,0.25,0.3') or 'auto' for 0.1-0.9 sweep.",
    )
    parser.add_argument(
        "--zscore-transform",
        type=Path,
        default=None,
        help="Path to ZscoreTpTransformer JSON (defaults to bundled P2Rank model if available).",
    )
    parser.add_argument(
        "--probability-transform",
        type=Path,
        default=None,
        help="Path to ProbabilityScoreTransformer JSON (defaults to bundled P2Rank model if available).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run inference on (auto -> CUDA if available).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for inference (increase to better utilise GPU/CPU throughput).",
    )
    parser.add_argument(
        "--disable-shared-memory",
        action="store_true",
        help="Disable shared-memory staging for H5 inference (useful if unsupported on the system).",
    )
    parser.add_argument(
        "--postproc-workers",
        type=int,
        default=_DEFAULT_POSTPROC_WORKERS,
        help=f"Number of threads for pocket aggregation (default: {_DEFAULT_POSTPROC_WORKERS}).",
    )
    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=None,
        help="Override P2Rank pred_point_threshold (default: 0.35).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Override minimum cluster size (default: 3).",
    )
    parser.add_argument(
        "--cluster-dist",
        type=float,
        default=None,
        help="Override clustering distance in Å (default: 3.0).",
    )
    parser.add_argument(
        "--eval-topk",
        type=int,
        default=None,
        help="Limit IoU metrics to top-K pockets per protein (default: all).",
    )
    parser.add_argument(
        "--min-pocket-score",
        type=float,
        default=None,
        help="Drop pockets with sum_prob below this threshold before evaluation.",
    )
    parser.add_argument(
        "--min-pocket-size",
        type=int,
        default=None,
        help="Drop pockets with fewer sample points than this threshold.",
    )
    parser.add_argument(
        "--max-pockets",
        type=int,
        default=None,
        help="Keep only the top-N pockets per protein before evaluation/output.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Path to Platt calibration JSON to apply to predictions.",
    )
    parser.add_argument(
        "--fit-calibration",
        type=Path,
        default=None,
        help="Fit Platt calibration on this run and save to the given path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        threshold_grid = _parse_threshold_grid(args.threshold_grid)
        zscore_path = args.zscore_transform or _resolve_default_transform(_DEFAULT_ZSCORE_REL)
        prob_path = args.probability_transform or _resolve_default_transform(_DEFAULT_PROB_REL)

        z_transform = load_score_transformer(zscore_path)
        if z_transform is not None and zscore_path is not None:
            logger.info("Loaded z-score transformer from %s", zscore_path)
        elif zscore_path is not None:
            logger.warning("Could not load z-score transformer from %s", zscore_path)

        prob_transform = load_score_transformer(prob_path)
        if prob_transform is not None and prob_path is not None:
            logger.info("Loaded probability transformer from %s", prob_path)
        elif prob_path is not None:
            logger.warning("Could not load probability transformer from %s", prob_path)

        device_request = args.device
        use_shared_memory = not args.disable_shared_memory
        postproc_workers = max(1, args.postproc_workers)
        base_params = P2RankParams()
        if args.pred_threshold is not None:
            base_params.pred_point_threshold = args.pred_threshold
        if args.min_cluster_size is not None:
            base_params.pred_min_cluster_size = args.min_cluster_size
        if args.cluster_dist is not None:
            base_params.pred_clustering_dist = args.cluster_dist

        run_production_pipeline(
            checkpoint=args.checkpoint,
            h5_path=args.h5,
            csv_path=args.csv,
            output_root=args.output,
            max_proteins=args.max_proteins,
            params=base_params,
            threshold_grid=threshold_grid,
            split_mode=args.split,
            zscore_transformer=z_transform,
            probability_transformer=prob_transform,
            device=device_request,
            batch_size=args.batch_size,
            use_shared_memory=use_shared_memory,
            postproc_workers=postproc_workers,
            eval_topk=args.eval_topk,
            min_pocket_score=args.min_pocket_score,
            min_pocket_size=args.min_pocket_size,
            max_pockets_all=args.max_pockets,
            calibration_path=args.calibration,
            fit_calibration_path=args.fit_calibration,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
