#!/usr/bin/env python3
"""Command-line entry point for the enhanced post-processing pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from post_processing.enhanced_pipeline import run_enhanced_pipeline, load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the enhanced PockNet post-processing pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "sota_default.yaml"),
        help="Path to the YAML configuration file describing the pipeline run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate the configuration without executing the pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    config = load_pipeline_config(config_path)

    logging.info("Loaded enhanced pipeline configuration from %s", config_path)

    if args.dry_run:
        logging.info("Dry run requested; exiting after config validation.")
        return

    run_enhanced_pipeline(config=config)


if __name__ == "__main__":
    main()
