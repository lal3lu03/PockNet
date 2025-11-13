#!/usr/bin/env python3
"""
PockNet End-to-End Orchestrator
===============================

This CLI script stitches together the most common project workflows so that a single
entry point can:

1. Launch Hydra-powered training runs (with overrides just like `src/train.py`).
2. Execute the full production post-processing pipeline on an H5 dataset so that
   the outputs match the historical P2Rank artefacts.
3. Run a lightweight single-protein inference pass (accepting either a raw PDB file
   path or just a protein identifier) while saving the resulting `pockets.csv`.
4. Chain training + production inference via the `full-run` command when you want
   a one-liner for release builds or smoke tests.

Each command prints human-readable status information and also writes machine-
readable JSON summaries so that automation (CI/CD, Docker health checks, etc.)
can detect regressions without parsing log files.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import click
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import rootutils

from post_processing.inference import ModelInference
from post_processing.pocketnet_aggregation import (
    PocketAggregationParams,
    PocketAggregationProcessor,
    ProteinPointLoader,
    pockets_to_csv,
)
from post_processing.run_production_pipeline import run_production_pipeline
from src.train import train as train_workflow
from src.utils import extras

# Ensure imports work no matter where the script is launched from.
PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_TRAIN_CONFIG = "train.yaml"
DEFAULT_PDB_ROOT = PROJECT_ROOT / "data" / "p2rank-datasets"
DEFAULT_BU48_LIST = PROJECT_ROOT / "data" / "bu48_proteins.txt"
DEFAULT_RELEASE_CHECKPOINT = PROJECT_ROOT / "logs" / "fusion_transformer_aggressive_oct17" / "runs" / "2025-10-23_blend_sweep" / "selective_swa_epoch09_12.ckpt"

LOG = logging.getLogger("pocknet.e2e")


def _compose_cfg(config_name: str, overrides: Sequence[str]) -> DictConfig:
    """Load a Hydra configuration the same way the standalone entry points do."""

    if not CONFIG_DIR.exists():
        raise FileNotFoundError(f"Could not find Hydra config directory at {CONFIG_DIR}")

    overrides = list(overrides or [])

    # Hydra refuses to re-initialize unless the singleton is cleared.
    GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def _tensor_to_float(value: Any) -> float:
    """Convert Lightning/Torch metric values to plain floats for JSON dumping."""
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # noqa: BLE001
            return float(value)  # fallback to generic cast
    if isinstance(value, (int, float)):
        return float(value)
    return float(value) if value is not None else 0.0


def _metrics_to_dict(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Best-effort conversion of metric dicts returned by Lightning trainers."""
    return {key: _tensor_to_float(val) for key, val in metrics.items()}


def _resolve_checkpoint(objects: Dict[str, Any], cfg: DictConfig) -> Optional[str]:
    """Return the checkpoint path that should be used after training."""
    ckpt_from_cfg = cfg.get("ckpt_path")
    if ckpt_from_cfg:
        return ckpt_from_cfg

    trainer = objects.get("trainer")
    if trainer is None:
        return None

    callback = getattr(trainer, "checkpoint_callback", None)
    if callback is None:
        return None

    best_path = getattr(callback, "best_model_path", "")
    return best_path or None


def _ensure_path(path: Path, kind: str) -> Path:
    """Validate that a user-supplied path exists before proceeding."""
    if not path.exists():
        raise click.ClickException(f"Expected {kind} at '{path}', but it does not exist.")
    return path


def _infer_protein_id(target: str) -> str:
    """Derive a protein identifier from either a raw string or a file path."""
    candidate = Path(target)
    if candidate.exists():
        return candidate.stem.lower()
    return target.strip().lower()


def _write_json(payload: Dict[str, Any], destination: Path) -> None:
    """Write dictionaries as pretty-printed JSON for downstream automation."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_step(command: Sequence[str], step_name: str) -> float:
    """Execute a subprocess step and report its duration."""
    LOG.info(">>> %s", step_name)
    start = time.time()
    try:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - exercised in integration
        raise click.ClickException(f"{step_name} failed (exit code {exc.returncode}).") from exc
    duration = time.time() - start
    LOG.info("<<< %s completed in %.1fs", step_name, duration)
    return duration


def _resolve_entry_path(entry: str, ds_dir: Path, pdb_root: Path) -> Optional[Path]:
    """Resolve a dataset entry to an absolute PDB path."""
    raw = Path(entry.strip())
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append((ds_dir / raw).resolve())
    candidates.append((pdb_root / raw).resolve())
    if "p2rank-datasets" in raw.parts:
        try:
            idx = raw.parts.index("p2rank-datasets")
            remaining = Path(*raw.parts[idx + 1 :])
            candidates.append((pdb_root / remaining).resolve())
        except ValueError:
            pass

    seen: List[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.append(candidate)
        if candidate.exists():
            return candidate
    return None


def _collect_pdb_paths(input_path: Path, pdb_root: Path) -> List[Path]:
    """Collect absolute PDB file paths from a .ds file, directory, or single file."""
    input_path = input_path.resolve()
    pdb_root = pdb_root.resolve()

    if input_path.is_file() and input_path.suffix.lower() == ".ds":
        resolved: List[Path] = []
        with input_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("PARAM."):
                    continue
                match = _resolve_entry_path(line, input_path.parent, pdb_root)
                if match is None:
                    raise click.ClickException(f"Could not resolve '{line}' from {input_path} using {pdb_root}.")
                resolved.append(match)
        return list(dict.fromkeys(resolved))

    if input_path.is_dir():
        pdbs = sorted(input_path.glob("*.pdb"))
        if not pdbs:
            raise click.ClickException(f"No .pdb files found inside {input_path}.")
        return [p.resolve() for p in dict.fromkeys(pdbs)]

    if input_path.is_file() and input_path.suffix.lower() in {".pdb", ".ent"}:
        return [input_path.resolve()]

    raise click.ClickException(f"Unsupported data source: {input_path}. Provide a .ds file, directory, or .pdb file.")


def _materialize_dataset(pdb_paths: Sequence[Path], destination: Path) -> Path:
    """Write a normalized dataset file with absolute PDB paths."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for path in pdb_paths:
            handle.write(f"{path}\n")
    LOG.info("Normalized dataset with %d proteins -> %s", len(pdb_paths), destination)
    return destination


def _resolve_bu48_file(user_path: Optional[Path], work_dir: Path) -> Path:
    """Return a BU48 list path, creating an empty placeholder if necessary."""
    if user_path:
        user_path = user_path.resolve()
        if not user_path.exists():
            raise click.ClickException(f"BU48 list not found: {user_path}")
        return user_path

    if DEFAULT_BU48_LIST.exists():
        return DEFAULT_BU48_LIST

    placeholder = work_dir / "bu48_placeholder.txt"
    placeholder.write_text("", encoding="utf-8")
    return placeholder


def _has_embeddings(emb_dir: Path) -> bool:
    """Check whether an embedding directory already contains .pt files."""
    if not emb_dir.exists():
        return False
    try:
        next(emb_dir.glob("*.pt"))
        return True
    except StopIteration:  # pragma: no cover - depends on filesystem state
        return False


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Unified CLI for training, evaluation, and production inference."""


@cli.command("train-model")
@click.option(
    "--config-name",
    default=DEFAULT_TRAIN_CONFIG,
    show_default=True,
    help="Name of the Hydra config under ./configs to use (e.g., train.yaml).",
)
@click.option(
    "-o",
    "--override",
    multiple=True,
    help="Hydra override in KEY=VALUE form (can be specified multiple times).",
)
@click.option(
    "--summary",
    type=click.Path(dir_okay=False, path_type=Path),
    default=PROJECT_ROOT / "outputs" / "train_summary.json",
    show_default=True,
    help="Where to store the JSON summary of the training run.",
)
@click.option(
    "--show-config",
    is_flag=True,
    help="Print the resolved Hydra config before running (useful for debugging).",
)
def train_model(config_name: str, override: Sequence[str], summary: Path, show_config: bool) -> None:
    """Train a model via Hydra and emit a JSON summary + checkpoint hint."""
    cfg = _compose_cfg(config_name, override)

    if show_config:
        click.echo("# -------- Hydrated config --------")
        click.echo(OmegaConf.to_yaml(cfg))
        click.echo("# ---------------------------------")

    extras(cfg)
    metrics, objects = train_workflow(cfg)

    metric_dict = _metrics_to_dict(metrics)
    ckpt_path = _resolve_checkpoint(objects, cfg)

    summary_payload = {
        "config": config_name,
        "overrides": list(override),
        "metrics": metric_dict,
        "checkpoint": ckpt_path,
    }
    _write_json(summary_payload, summary)

    click.echo(f"Training finished with {len(metric_dict)} logged metrics.")
    if ckpt_path:
        click.echo(f"Best checkpoint: {ckpt_path}")
    else:
        click.echo("No checkpoint reported by the trainer (check callback configuration).")


@cli.command("predict-dataset")
@click.option("--checkpoint", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Path to the Lightning checkpoint to load.")
@click.option("--h5", "h5_path", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Path to the merged H5 file with residues + embeddings.")
@click.option("--csv", "csv_path", required=True, type=click.Path(path_type=Path), help="vectorsTrain CSV directory or file that matches the H5 file.")
@click.option("--output", "output_dir", default=PROJECT_ROOT / "outputs" / "production_run", show_default=True, type=click.Path(path_type=Path), help="Destination directory for pockets, metrics and summaries.")
@click.option("--split", default="test", show_default=True, help="Dataset split to evaluate (train|val|test|all).")
@click.option("--max-proteins", type=int, default=None, help="Limit the number of proteins processed (useful for smoke tests).")
@click.option("--device", default="auto", show_default=True, help="Device hint passed to the inference engine (auto|cpu|cuda:0|...).")
@click.option("--batch-size", default=2048, show_default=True, type=int, help="Batch size for the Lightning inference pass.")
@click.option("--threshold", type=float, default=None, help="Override the ligandable residue threshold used by the aggregator.")
@click.option("--postproc-workers", type=int, default=4, show_default=True, help="Number of CPU threads used for pocket aggregation.")
def predict_dataset(
    checkpoint: Path,
    h5_path: Path,
    csv_path: Path,
    output_dir: Path,
    split: str,
    max_proteins: Optional[int],
    device: str,
    batch_size: int,
    threshold: Optional[float],
    postproc_workers: int,
) -> None:
    """Run the historical P2Rank-like post-processing pipeline on an H5 dataset."""
    checkpoint = _ensure_path(checkpoint, "checkpoint")
    h5_path = _ensure_path(h5_path, "H5 dataset")
    csv_path = _ensure_path(csv_path, "feature CSV directory/file")

    output_dir.mkdir(parents=True, exist_ok=True)

    params = PocketAggregationParams()
    if threshold is not None:
        params.pred_point_threshold = threshold

    results = run_production_pipeline(
        checkpoint=checkpoint,
        h5_path=h5_path,
        csv_path=csv_path,
        output_root=output_dir,
        max_proteins=max_proteins,
        params=params,
        split_mode=split,
        device=device,
        batch_size=batch_size,
        postproc_workers=max(1, postproc_workers),
    )

    summary_payload = {
        "processed_proteins": len(results),
        "checkpoint": str(checkpoint),
        "h5": str(h5_path),
        "csv": str(csv_path),
        "output_dir": str(output_dir),
        "split": split,
        "threshold": params.pred_point_threshold,
    }
    _write_json(summary_payload, output_dir / "dataset_inference_summary.json")
    click.echo(f"Dataset inference finished for {len(results)} proteins. Outputs live in {output_dir}.")


@cli.command("predict-pdb")
@click.argument("target")
@click.option("--checkpoint", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Checkpoint used for residue-level inference.")
@click.option("--h5", "h5_path", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Existing H5 dataset that already contains the protein of interest.")
@click.option("--csv", "csv_path", required=True, type=click.Path(path_type=Path), help="CSV/dir with SAS coordinates that correspond to the H5 dataset.")
@click.option("--output", "output_dir", default=PROJECT_ROOT / "outputs" / "single_protein", show_default=True, type=click.Path(path_type=Path), help="Where to save the generated pockets + metadata.")
@click.option("--device", default="auto", show_default=True, help="Device hint for Lightning inference (auto|cpu|cuda:0|...).")
@click.option("--batch-size", default=1024, show_default=True, type=int, help="Batch size used for the residue predictions.")
@click.option("--threshold", type=float, default=None, help="Optional override for the residue probability threshold.")
def predict_single_protein(
    target: str,
    checkpoint: Path,
    h5_path: Path,
    csv_path: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    threshold: Optional[float],
) -> None:
    """
    Run inference for a single protein.

    The command accepts either a literal protein identifier (matching the keys inside
    the H5 file) or a path to a PDB file. When a PDB path is given, the filename
    (without extension) is used as the lookup key and the file itself is copied into
    the output directory so downstream tools understand what was processed.
    """
    checkpoint = _ensure_path(checkpoint, "checkpoint")
    h5_path = _ensure_path(h5_path, "H5 dataset")
    csv_path = _ensure_path(csv_path, "feature CSV directory/file")

    output_dir.mkdir(parents=True, exist_ok=True)

    protein_id = _infer_protein_id(target)
    click.echo(f"Resolved protein identifier: {protein_id}")

    pdb_candidate = Path(target)
    if pdb_candidate.exists():
        shutil.copy2(pdb_candidate, output_dir / pdb_candidate.name)

    loader = ProteinPointLoader(h5_path, csv_path, cache_dir=output_dir / "cache")
    coords, residue_numbers, _ = loader.get_protein_arrays(protein_id)
    if coords.size == 0:
        raise click.ClickException(
            f"No coordinates found for '{protein_id}'. "
            "Ensure the protein is present in the provided H5/CSV pair."
        )

    inference_device = None if device == "auto" else device
    predictor = ModelInference(str(checkpoint), device=inference_device)
    predictions = predictor.predict_from_h5(
        str(h5_path),
        protein_ids=[protein_id],
        batch_size=batch_size,
        use_shared_memory=False,
    )
    probs = predictions.get(protein_id)
    if probs is None or len(probs) == 0:
        raise click.ClickException(
            f"The checkpoint did not return predictions for '{protein_id}'. "
            "Double-check that the identifier matches the dataset."
        )

    if len(probs) != len(coords):
        click.echo(
            "Warning: prediction length mismatch detected; truncating to the shortest length.",
            err=True,
        )
        limit = min(len(probs), len(coords))
        probs = probs[:limit]
        coords = coords[:limit]
        residue_numbers = residue_numbers[:limit]

    params = PocketAggregationParams()
    if threshold is not None:
        params.pred_point_threshold = threshold
    processor = PocketAggregationProcessor(params)
    pockets = processor.process(protein_id, coords, probs, residue_numbers)

    csv_text = pockets_to_csv(protein_id, pockets)
    pockets_path = output_dir / f"{protein_id}_pockets.csv"
    pockets_path.write_text(csv_text, encoding="utf-8")

    summary_payload = {
        "protein_id": protein_id,
        "checkpoint": str(checkpoint),
        "total_points": int(len(coords)),
        "pocket_count": len(pockets),
        "max_pocket_score": max((p.score for p in pockets), default=0.0),
        "sum_pocket_scores": sum(p.score for p in pockets),
        "threshold": params.pred_point_threshold,
        "pockets_csv": str(pockets_path),
    }
    _write_json(summary_payload, output_dir / f"{protein_id}_summary.json")
    click.echo(f"Pockets for {protein_id} saved to {pockets_path}")


@cli.command("auto-run")
@click.argument("data_path", type=click.Path(path_type=Path))
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store all generated artefacts (defaults to outputs/auto_<dataset>).",
)
@click.option(
    "--checkpoint",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Checkpoint used for inference (defaults to the released selective_swa_epoch09_12).",
)
@click.option(
    "--pdb-root",
    type=click.Path(path_type=Path),
    default=DEFAULT_PDB_ROOT,
    show_default=True,
    help="Root directory containing the PDB hierarchy referenced by the dataset.",
)
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="Device hint used for both embedding generation and inference (cpu, cuda:0, ...).",
)
@click.option(
    "--threads",
    type=int,
    default=os.cpu_count() or 4,
    show_default=True,
    help="Number of CPU workers used by feature extraction and H5 generation.",
)
@click.option(
    "--batch-size",
    type=int,
    default=2048,
    show_default=True,
    help="Batch size for Lightning inference.",
)
@click.option(
    "--max-proteins",
    type=int,
    default=None,
    help="Limit the number of proteins processed during inference.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Recompute every stage even if intermediate artefacts already exist.",
)
@click.option(
    "--bu48-list",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional BU48 list to preserve held-out proteins. Defaults to data/bu48_proteins.txt when available.",
)
def auto_run(
    data_path: Path,
    output: Optional[Path],
    checkpoint: Optional[Path],
    pdb_root: Path,
    device: str,
    threads: int,
    batch_size: int,
    max_proteins: Optional[int],
    force: bool,
    bu48_list: Optional[Path],
) -> None:
    """
    Run the full feature -> embedding -> H5 -> inference pipeline with minimal inputs.

    Provide a dataset (.ds), directory, or single PDB file and receive production-ready
    pocket predictions in <output>/predictions.
    """
    threads = max(1, threads or 1)
    data_path = data_path.resolve()
    pdb_root = pdb_root.resolve()
    output_dir = (output.resolve() if output else (PROJECT_ROOT / "outputs" / f"auto_{data_path.stem.lower()}"))
    artifacts_dir = output_dir / "artifacts"
    features_dir = artifacts_dir / "features"
    embeddings_dir = artifacts_dir / "embeddings"
    h5_dir = artifacts_dir / "h5"
    predictions_dir = output_dir / "predictions"

    if force:
        for path in [features_dir, embeddings_dir, h5_dir, predictions_dir]:
            if path.exists():
                shutil.rmtree(path)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for path in [features_dir, embeddings_dir, h5_dir, predictions_dir]:
        path.mkdir(parents=True, exist_ok=True)

    pdb_paths = _collect_pdb_paths(data_path, pdb_root)
    if not pdb_paths:
        raise click.ClickException(f"No PDB files resolved from {data_path}.")

    normalized_ds = _materialize_dataset(pdb_paths, artifacts_dir / f"{data_path.stem}_normalized.ds")
    features_csv = features_dir / "vectorsTrain.csv"

    durations: Dict[str, float] = {}

    if not features_csv.exists() or force:
        feature_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src/datagen/extract_protein_features.py"),
            str(normalized_ds),
            str(features_dir),
            "--threads",
            str(threads),
            "--skip_existing",
        ]
        durations["feature_extraction"] = _run_step(feature_cmd, "Feature extraction")
    else:
        LOG.info("Reusing existing features at %s", features_csv)

    if not _has_embeddings(embeddings_dir) or force:
        embed_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src/datagen/generate_esm2_embeddings.py"),
            "--ds-file",
            str(normalized_ds),
            "--out-dir",
            str(embeddings_dir),
            "--device",
            device,
            "--pdb-base",
            str(Path("/")),
        ]
        if force:
            embed_cmd.append("--force")
        durations["embeddings"] = _run_step(embed_cmd, "ESM2 embedding generation")
    else:
        LOG.info("Reusing embeddings in %s", embeddings_dir)

    if not features_csv.exists():
        raise click.ClickException(f"Combined feature CSV not found at {features_csv}.")

    bu48_path = _resolve_bu48_file(bu48_list, artifacts_dir)
    h5_path = h5_dir / "pocknet_inputs.h5"
    if not h5_path.exists() or force:
        h5_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "generate_h5_v2_optimized.py"),
            "--csv",
            str(features_csv),
            "--esm_dir",
            str(embeddings_dir),
            "--pdb_base_dir",
            str(pdb_root),
            "--bu48_txt",
            str(bu48_path),
            "--out",
            str(h5_path),
            "--k",
            "3",
            "--workers",
            str(threads),
            "--ds_file",
            str(normalized_ds),
        ]
        durations["h5_generation"] = _run_step(h5_cmd, "H5 generation")
    else:
        LOG.info("Reusing H5 dataset at %s", h5_path)

    if not h5_path.exists():
        raise click.ClickException(f"H5 dataset not found at {h5_path}.")

    checkpoint_path = (checkpoint.resolve() if checkpoint else DEFAULT_RELEASE_CHECKPOINT)
    if not checkpoint_path.exists():
        raise click.ClickException(
            f"Checkpoint {checkpoint_path} not found. Pass --checkpoint to point at a valid .ckpt file."
        )

    LOG.info("Running production inference with checkpoint %s", checkpoint_path)
    inference_start = time.time()
    results = run_production_pipeline(
        checkpoint=checkpoint_path,
        h5_path=h5_path,
        csv_path=features_csv,
        output_root=predictions_dir,
        max_proteins=max_proteins,
        params=PocketAggregationParams(),
        split_mode="all",
        device=device,
        batch_size=batch_size,
        postproc_workers=threads,
    )
    durations["inference"] = time.time() - inference_start

    summary_csv = predictions_dir / "summary" / "summary.csv"
    summary = {
        "inputs": len(pdb_paths),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "features_csv": str(features_csv),
        "h5_path": str(h5_path),
        "predictions_dir": str(predictions_dir),
        "processed_proteins": len(results),
        "durations_sec": durations,
    }
    if summary_csv.exists():
        summary["summary_csv"] = str(summary_csv)
    _write_json(summary, output_dir / "auto_run_summary.json")
    click.echo(f"End-to-end pipeline finished. Outputs available under {predictions_dir}.")

@cli.command("full-run")
@click.option(
    "--train-config",
    default=DEFAULT_TRAIN_CONFIG,
    show_default=True,
    help="Hydra config to use for the training phase.",
)
@click.option(
    "-o",
    "--override",
    multiple=True,
    help="Hydra override applied during the training phase.",
)
@click.option(
    "--checkpoint",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Existing checkpoint to skip training (if omitted a new run is launched).",
)
@click.option("--h5", "h5_path", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Dataset used for the production inference phase.")
@click.option("--csv", "csv_path", required=True, type=click.Path(path_type=Path), help="vectorsTrain CSV directory/file used alongside the H5 dataset.")
@click.option("--output", "output_dir", default=PROJECT_ROOT / "outputs" / "release_candidate", show_default=True, type=click.Path(path_type=Path), help="Directory that will hold both the training summary and the inference artefacts.")
def full_run(
    train_config: str,
    override: Sequence[str],
    checkpoint: Optional[Path],
    h5_path: Path,
    csv_path: Path,
    output_dir: Path,
) -> None:
    """
    Convenience command that (optionally) trains a model and immediately runs the
    production post-processing pipeline with the resulting checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint
    summaries: Dict[str, Any] = {}

    if ckpt_path is None:
        cfg = _compose_cfg(train_config, override)
        extras(cfg)
        metrics, objects = train_workflow(cfg)
        summary_payload = {
            "config": train_config,
            "overrides": list(override),
            "metrics": _metrics_to_dict(metrics),
        }
        ckpt_path_str = _resolve_checkpoint(objects, cfg)
        if ckpt_path_str is None:
            raise click.ClickException(
                "Training finished but no checkpoint path was reported; "
                "please ensure ModelCheckpoint is enabled or pass --checkpoint explicitly."
            )
        ckpt_path = Path(ckpt_path_str)
        summary_payload["checkpoint"] = ckpt_path_str
        summaries["training"] = summary_payload
        _write_json(summary_payload, output_dir / "training_summary.json")
        click.echo(f"Training complete. Using checkpoint {ckpt_path_str} for inference.")
    else:
        ckpt_path = _ensure_path(ckpt_path, "checkpoint")

    dataset_output_dir = output_dir / "production_pipeline"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    results = run_production_pipeline(
        checkpoint=ckpt_path,
        h5_path=_ensure_path(h5_path, "H5 dataset"),
        csv_path=_ensure_path(csv_path, "feature CSV directory/file"),
        output_root=dataset_output_dir,
        params=PocketAggregationParams(),
    )

    summaries["production"] = {
        "checkpoint": str(ckpt_path),
        "processed_proteins": len(results),
        "output_dir": str(dataset_output_dir),
    }
    _write_json(summaries["production"], output_dir / "production_summary.json")

    rollup = {
        "training": summaries.get("training"),
        "production": summaries["production"],
    }
    _write_json(rollup, output_dir / "full_run_summary.json")
    click.echo(
        f"Full run completed. All artefacts are inside {output_dir} "
        "(see full_run_summary.json for a machine-readable overview)."
    )


if __name__ == "__main__":
    cli()
