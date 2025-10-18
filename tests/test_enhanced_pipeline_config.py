"""Tests for the enhanced post-processing configuration utilities."""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "post_processing" / "enhanced_pipeline.py"

spec = importlib.util.spec_from_file_location("enhanced_pipeline", MODULE_PATH)
enhanced_pipeline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(enhanced_pipeline)


def test_load_pipeline_config_reads_yaml(tmp_path: Path) -> None:
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        textwrap.dedent(
            """
            checkpoint_paths:
              - logs/run/checkpoint.ckpt
            h5_file: data/h5/sample.h5
            output_dir: custom_results
            enhanced:
              create_pymol: false
              max_pockets: 3
            """
        ).strip()
    )

    config = enhanced_pipeline.load_pipeline_config(config_yaml)

    expected_checkpoint = (config_yaml.parent / "logs/run/checkpoint.ckpt").resolve()
    expected_h5 = (config_yaml.parent / "data/h5/sample.h5").resolve()
    expected_output = (config_yaml.parent / "custom_results").resolve()

    assert config.checkpoint_paths == [str(expected_checkpoint)]
    assert config.h5_file == str(expected_h5)
    assert config.output_dir == str(expected_output)
    assert config.enhanced.create_pymol is False
    assert config.enhanced.max_pockets == 3


def test_load_pipeline_config_requires_checkpoint(tmp_path: Path) -> None:
    config_yaml = tmp_path / "missing.yaml"
    config_yaml.write_text("h5_file: data/h5/sample.h5\n")

    with pytest.raises(ValueError):
        enhanced_pipeline.load_pipeline_config(config_yaml)


def test_load_pipeline_config_preserves_absolute_paths(tmp_path: Path) -> None:
    absolute_ckpt = (tmp_path / "ckpt.ckpt").resolve()
    absolute_h5 = (tmp_path / "dataset.h5").resolve()
    absolute_output = (tmp_path / "results").resolve()

    config_yaml = tmp_path / "absolute.yaml"
    config_yaml.write_text(
        textwrap.dedent(
            f"""
            checkpoint_paths:
              - {absolute_ckpt}
            h5_file: {absolute_h5}
            output_dir: {absolute_output}
            """
        ).strip()
    )

    config = enhanced_pipeline.load_pipeline_config(config_yaml)

    assert config.checkpoint_paths == [str(absolute_ckpt)]
    assert config.h5_file == str(absolute_h5)
    assert config.output_dir == str(absolute_output)


def test_run_enhanced_pipeline_uses_provided_config(monkeypatch: pytest.MonkeyPatch) -> None:
    invoked = {}

    class DummyProcessor:
        def __init__(self, config, h5_path):
            invoked["init"] = {
                "config": config,
                "h5_path": h5_path,
            }

        def run_complete_pipeline(self, **kwargs):
            invoked["run"] = kwargs
            return {"status": "ok"}

    monkeypatch.setattr(enhanced_pipeline, "EnhancedPostProcessor", DummyProcessor)

    runtime_config = enhanced_pipeline.EnhancedPipelineRunConfig(
        checkpoint_paths=["ckpt-a.ckpt"],
        h5_file="dataset.h5",
        protein_ids=["proteinA"],
        output_dir="results_dir",
    )

    result = enhanced_pipeline.run_enhanced_pipeline(config=runtime_config)

    assert result == {"status": "ok"}
    assert invoked["init"]["h5_path"] == "dataset.h5"
    assert invoked["run"]["checkpoint_paths"] == ["ckpt-a.ckpt"]
    assert invoked["run"]["protein_ids"] == ["proteinA"]
    assert invoked["run"]["output_dir"] == "results_dir"
