"""Smoke tests for the Click-based end-to-end pipeline CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from src.scripts import end_to_end_pipeline as e2e


@pytest.fixture(autouse=True)
def stub_hydra_and_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub Hydra + Lightning training so CLI commands remain lightweight."""

    def fake_compose(config_name: str, overrides) -> dict:
        return {"config": config_name, "overrides": list(overrides)}

    def fake_extras(cfg: dict) -> None:  # pragma: no cover - nothing to do
        return None

    def fake_train_workflow(cfg: dict) -> tuple[dict, dict]:
        trainer = SimpleNamespace(
            checkpoint_callback=SimpleNamespace(best_model_path="dummy.ckpt")
        )
        metrics = {"val/auprc": 0.123}
        return metrics, {"trainer": trainer}

    monkeypatch.setattr(e2e, "_compose_cfg", fake_compose)
    monkeypatch.setattr(e2e, "extras", fake_extras)
    monkeypatch.setattr(e2e, "train_workflow", fake_train_workflow)


@pytest.fixture(autouse=True)
def stub_production_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the expensive production pipeline + inference machinery."""

    def fake_run_production_pipeline(**kwargs):
        return {"proteinA": {"iou": 0.42}}

    class DummyLoader:
        def __init__(self, h5_path, csv_path, cache_dir):
            self._coords = np.zeros((4, 3), dtype=np.float32)
            self._res_nums = np.arange(4, dtype=np.int32)

        def get_protein_arrays(self, protein_id):
            labels = np.zeros(len(self._coords), dtype=np.int8)
            return (
                self._coords.copy(),
                self._res_nums.copy(),
                labels,
            )

    class DummyInference:
        def __init__(self, checkpoint: str, device=None):
            self.checkpoint = checkpoint
            self.device = device

        def predict_from_h5(self, h5_path: str, protein_ids, **kwargs):
            count = 4
            return {protein_ids[0]: np.linspace(0.1, 0.9, count, dtype=np.float32)}

    class DummyProcessor:
        def __init__(self, params):
            self.params = params

        def process(self, protein_id, coords, probs, residue_numbers):
            return [SimpleNamespace(score=0.5)]

    monkeypatch.setattr(e2e, "run_production_pipeline", fake_run_production_pipeline)
    monkeypatch.setattr(e2e, "ProteinPointLoader", DummyLoader)
    monkeypatch.setattr(e2e, "ModelInference", DummyInference)
    monkeypatch.setattr(e2e, "PocketAggregationProcessor", DummyProcessor)
    monkeypatch.setattr(e2e, "pockets_to_csv", lambda protein_id, pockets: "csv")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _touch(path: Path) -> Path:
    path.write_text("stub")
    return path


def test_train_model_command(tmp_path: Path, runner: CliRunner) -> None:
    summary_path = tmp_path / "train_summary.json"
    result = runner.invoke(
        e2e.cli,
        ["train-model", "--summary", str(summary_path), "-o", "trainer.fast_dev_run=true"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    payload = json.loads(summary_path.read_text())
    assert payload["checkpoint"] == "dummy.ckpt"


def test_predict_dataset_command(tmp_path: Path, runner: CliRunner) -> None:
    ckpt = _touch(tmp_path / "model.ckpt")
    h5 = _touch(tmp_path / "dataset.h5")
    csv = _touch(tmp_path / "vectors.csv")
    out_dir = tmp_path / "prod"

    result = runner.invoke(
        e2e.cli,
        [
            "predict-dataset",
            "--checkpoint",
            str(ckpt),
            "--h5",
            str(h5),
            "--csv",
            str(csv),
            "--output",
            str(out_dir),
            "--max-proteins",
            "1",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    summary_file = out_dir / "dataset_inference_summary.json"
    assert summary_file.exists()
    payload = json.loads(summary_file.read_text())
    assert payload["split"] == "test"


def test_predict_single_protein_command(tmp_path: Path, runner: CliRunner) -> None:
    ckpt = _touch(tmp_path / "model.ckpt")
    h5 = _touch(tmp_path / "dataset.h5")
    csv = _touch(tmp_path / "vectors.csv")
    out_dir = tmp_path / "single"

    result = runner.invoke(
        e2e.cli,
        [
            "predict-pdb",
            "1abc_A",
            "--checkpoint",
            str(ckpt),
            "--h5",
            str(h5),
            "--csv",
            str(csv),
            "--output",
            str(out_dir),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    summary_file = out_dir / "1abc_a_summary.json"
    assert summary_file.exists()


def test_full_run_command(tmp_path: Path, runner: CliRunner) -> None:
    h5 = _touch(tmp_path / "dataset.h5")
    csv = _touch(tmp_path / "vectors.csv")
    out_dir = tmp_path / "release"

    result = runner.invoke(
        e2e.cli,
        [
            "full-run",
            "--h5",
            str(h5),
            "--csv",
            str(csv),
            "--output",
            str(out_dir),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert (out_dir / "full_run_summary.json").exists()
