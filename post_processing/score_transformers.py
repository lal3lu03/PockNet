"""
Score transformer utilities mirroring P2Rank's JSON transformers.

This module provides light-weight reimplementations of the score
transformers that ship with P2Rank so that the Python post-processing
pipeline can emit comparable auxiliary pocket scores (`zScoreTP`,
`probaTP`, etc.).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class ScoreTransformer:
    """Base interface for score transformers."""

    def transform(self, score: float) -> float:
        raise NotImplementedError


@dataclass
class ZScoreTransformer(ScoreTransformer):
    """Linear z-score normalisation."""

    mean: float
    stdev: float

    def transform(self, score: float) -> float:
        if self.stdev <= 0:
            return 0.0
        return (score - self.mean) / self.stdev


@dataclass
class ProbabilityScoreTransformer(ScoreTransformer):
    """
    Histogram-based probability transformer.

    Interpolates between cumulative histograms of true/false pockets to
    approximate P(true pocket | raw score).
    """

    min_value: float
    max_value: float
    nbins: int
    tp_cumul_hist: np.ndarray
    fp_cumul_hist: np.ndarray

    def __post_init__(self) -> None:
        if self.tp_cumul_hist.shape != self.fp_cumul_hist.shape:
            raise ValueError("Histogram shapes for TP/FP differ.")
        if self.tp_cumul_hist.size != self.nbins:
            raise ValueError("Histogram size does not match nbins.")
        if self.max_value <= self.min_value:
            raise ValueError("Invalid min/max for probability transformer.")

    def transform(self, score: float) -> float:
        if self.nbins <= 0:
            return 0.0

        span = self.max_value - self.min_value
        if span <= 0:
            return 0.0

        step = span / self.nbins
        if step <= 0:
            return 0.0

        idx = int((score - self.min_value) / step)
        idx = max(0, min(self.nbins - 1, idx))

        if idx == self.nbins - 1:
            tp = float(self.tp_cumul_hist[idx])
            fp = float(self.fp_cumul_hist[idx])
        else:
            base = self.min_value + step * idx
            frac = (score - base) / step
            frac = max(0.0, min(1.0, frac))

            tp = float(
                self.tp_cumul_hist[idx]
                + (self.tp_cumul_hist[idx + 1] - self.tp_cumul_hist[idx]) * frac
            )
            fp = float(
                self.fp_cumul_hist[idx]
                + (self.fp_cumul_hist[idx + 1] - self.fp_cumul_hist[idx]) * frac
            )

        denom = tp + fp
        if denom <= 0.0:
            return 0.0
        return tp / denom


def _ensure_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def load_score_transformer(path: Optional[Path]) -> Optional[ScoreTransformer]:
    """
    Load a score transformer from a P2Rank JSON file.

    Args:
        path: Path to the JSON payload. If None or missing, returns None.
    """
    if path is None:
        return None

    json_path = Path(path)
    if not json_path.exists():
        logger.warning("Score transformer not found: %s", json_path)
        return None

    try:
        payload = json.loads(json_path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse score transformer %s: %s", json_path, exc)
        return None

    name = payload.get("name")
    params = payload.get("params", {})

    if name == "ZscoreTpTransformer":
        mean = float(params.get("mean", 0.0))
        stdev = float(params.get("stdev", 1.0))
        if stdev <= 0:
            logger.warning(
                "ZScore transformer %s has non-positive stdev (%.4f); returning None.",
                json_path,
                stdev,
            )
            return None
        return ZScoreTransformer(mean=mean, stdev=stdev)

    if name == "ProbabilityScoreTransformer":
        required = {"min", "max", "nbins", "tp_cumul_hist", "fp_cumul_hist"}
        if not required.issubset(params.keys()):
            logger.warning(
                "Probability transformer %s missing keys (%s)", json_path, required - params.keys()
            )
            return None

        try:
            transformer = ProbabilityScoreTransformer(
                min_value=float(params["min"]),
                max_value=float(params["max"]),
                nbins=int(params["nbins"]),
                tp_cumul_hist=_ensure_array(params["tp_cumul_hist"]),
                fp_cumul_hist=_ensure_array(params["fp_cumul_hist"]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Invalid probability transformer %s: %s", json_path, exc)
            return None
        return transformer

    logger.warning("Unsupported score transformer '%s' in %s", name, json_path)
    return None
