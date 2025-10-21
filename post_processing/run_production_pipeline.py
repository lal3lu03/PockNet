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
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from post_processing.inference import ModelInference
from post_processing.p2rank_like import (
    P2RankParams,
    P2RankPostProcessor,
    ProteinPointLoader,
    create_ground_truth_pockets,
    pocket_iou_metrics,
    pockets_to_csv,
)

logger = logging.getLogger("pocknet.p2rank_production")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _detect_test_proteins(h5_path: Path, protein_index: ProteinPointLoader) -> List[str]:
    """Return proteins with split==2 if the split dataset is available."""
    proteins = protein_index.h5_index.proteins()
    if not proteins:
        return []

    with h5py.File(h5_path, "r") as fh:
        if "split" not in fh:
            return proteins

        split = fh["split"]
        keep: List[str] = []
        for protein in proteins:
            start, _ = protein_index.h5_index.slice(protein)
            if split[start] == 2:
                keep.append(protein)

        return keep if keep else proteins


def _prepare_predictions(
    checkpoint: Path,
    h5_path: Path,
    protein_ids: List[str],
) -> Dict[str, np.ndarray]:
    """Run model inference and return sigmoid probabilities per protein."""
    start = time.time()
    model = ModelInference(str(checkpoint))
    preds = model.predict_from_h5(str(h5_path), protein_ids)
    elapsed = time.time() - start
    logger.info("Inference completed for %d proteins in %.2fs", len(preds), elapsed)
    return preds


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

def run_production_pipeline(
    checkpoint: Path,
    h5_path: Path,
    csv_path: Path,
    output_root: Path,
    max_proteins: Optional[int] = None,
    params: Optional[P2RankParams] = None,
    threshold_grid: Optional[List[float]] = None,
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
    all_proteins = _detect_test_proteins(h5_path, loader)
    if max_proteins is not None:
        all_proteins = all_proteins[:max_proteins]

    if not all_proteins:
        logger.error("No proteins found to process.")
        return {}

    logger.info("Processing %d proteins", len(all_proteins))

    try:
        predictions = _prepare_predictions(checkpoint, h5_path, all_proteins)
    except Exception as exc:  # noqa: BLE001
        logger.error("Inference failed: %s", exc)
        raise

    processor = P2RankPostProcessor(params)
    pockets_output_dir = output_root / "cases"
    pockets_output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}
    total_pockets = 0
    pocket_scores: List[float] = []
    gt_mean_ious: List[float] = []
    protein_cache: Dict[str, Dict[str, object]] = {}

    for protein_id in all_proteins:
        probs = predictions.get(protein_id)
        if probs is None or len(probs) == 0:
            logger.warning("Skipping %s (no predictions returned)", protein_id)
            continue

        coords, residue_numbers, labels = loader.get_protein_arrays(protein_id)
        if len(coords) == 0:
            logger.warning("Skipping %s (no coordinates)", protein_id)
            continue

        gt_pockets = create_ground_truth_pockets(
            protein_id,
            coords,
            labels,
            residue_numbers,
            params,
        )

        pockets = processor.process(protein_id, coords, probs, residue_numbers)
        csv_text = pockets_to_csv(protein_id, pockets)
        _write_pockets_csv(pockets_output_dir, protein_id, csv_text)

        total_pockets += len(pockets)
        pocket_scores.extend([p.score for p in pockets])

        gt_metrics = pocket_iou_metrics(pockets, gt_pockets)
        gt_mean_ious.append(gt_metrics["mean_gt_iou"])

        results[protein_id] = {
            "n_points": len(coords),
            "n_pockets": len(pockets),
            "max_score": max((p.score for p in pockets), default=0.0),
            "sum_scores": sum(p.score for p in pockets),
            "mean_gt_iou_default": gt_metrics["mean_gt_iou"],
        }

        logger.info(
            "%s: points=%d pockets=%d max_score=%.3f",
            protein_id,
            len(coords),
            len(pockets),
            results[protein_id]["max_score"],
        )

        protein_cache[protein_id] = {
            "coords": coords,
            "residue_numbers": residue_numbers,
            "labels": labels,
            "probs": probs,
            "gt_pockets": gt_pockets,
        }

    n_proteins = len(results)
    metrics = {
        "total_proteins": n_proteins,
        "proteins_attempted": len(all_proteins),
        "total_pockets": total_pockets,
        "avg_pockets_per_protein": total_pockets / n_proteins if n_proteins else 0.0,
        "avg_pocket_score": float(np.mean(pocket_scores)) if pocket_scores else 0.0,
        "max_pocket_score": max(pocket_scores) if pocket_scores else 0.0,
        "avg_mean_gt_iou_default": float(np.mean(gt_mean_ious)) if gt_mean_ious else 0.0,
    }
    threshold_sweep_results: List[Dict[str, float]] = []
    if threshold_grid:
        base_param_dict = params.__dict__.copy()
        sweep_results: List[Dict[str, float]] = []
        for threshold in threshold_grid:
            base_param_dict["pred_point_threshold"] = threshold
            sweep_processor = P2RankPostProcessor(P2RankParams(**base_param_dict))
            per_protein = []
            for protein_id, pdata in protein_cache.items():
                pockets_thresh = sweep_processor.process(
                    protein_id,
                    pdata["coords"],  # type: ignore[arg-type]
                    pdata["probs"],  # type: ignore[arg-type]
                    pdata["residue_numbers"],  # type: ignore[arg-type]
                )
                metrics_thresh = pocket_iou_metrics(
                    pockets_thresh,
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

    logger.info("Finished. Total pockets: %d", total_pockets)
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
        "--threshold-grid",
        type=str,
        default=None,
        help="Comma-separated thresholds to evaluate (e.g. '0.2,0.25,0.3') or 'auto' for 0.1-0.9 sweep.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        threshold_grid = _parse_threshold_grid(args.threshold_grid)
        run_production_pipeline(
            checkpoint=args.checkpoint,
            h5_path=args.h5,
            csv_path=args.csv,
            output_root=args.output,
            max_proteins=args.max_proteins,
            params=P2RankParams(),
            threshold_grid=threshold_grid,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
