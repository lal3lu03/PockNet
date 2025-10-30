#!/usr/bin/env python3
"""
Pocket visualization utilities.

Provides lightweight Matplotlib helpers to render predicted and ground-truth
pockets for qualitative inspection. Figures show a sampled subset of the SAS
points (grey) alongside coloured pocket assignments and pocket centres.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:  # Lazy import – matplotlib is optional for headless environments
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection

    HAS_MPL = True
except Exception:  # pragma: no cover - visualization is optional
    HAS_MPL = False

from .p2rank_like import P2RankPocket

logger = logging.getLogger(__name__)

_DEFAULT_MAX_POINTS = 6000
_DEFAULT_MAX_POCKETS = 5


def _set_axes_equal(ax) -> None:
    """Set 3D axis limits to equal scale for better proportion perception."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)
    x_centre = np.mean(x_limits)
    y_centre = np.mean(y_limits)
    z_centre = np.mean(z_limits)

    half = max_range / 2.0
    ax.set_xlim3d([x_centre - half, x_centre + half])
    ax.set_ylim3d([y_centre - half, y_centre + half])
    ax.set_zlim3d([z_centre - half, z_centre + half])


def _choose_background_indices(n_points: int, max_points: int, seed: int) -> np.ndarray:
    """Down-sample background SAS points deterministically."""
    if n_points <= max_points:
        return np.arange(n_points, dtype=np.int32)
    rng = np.random.default_rng(seed)
    return rng.choice(n_points, size=max_points, replace=False)


def _iter_pocket_points(
    pockets: Sequence[P2RankPocket],
    coords: np.ndarray,
    limit: int,
) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (label, points, centre) for the first ``limit`` pockets with members."""
    for idx, pocket in enumerate(pockets[:limit], start=1):
        if pocket.member_indices.size == 0:
            continue
        pocket_pts = coords[pocket.member_indices]
        if pocket_pts.size == 0:
            continue
        centre = np.asarray(pocket.center, dtype=np.float32)
        yield f"P{idx}", pocket_pts, centre


def save_pocket_visualization(
    protein_id: str,
    coords: np.ndarray,
    predicted: Sequence[P2RankPocket],
    ground_truth: Sequence[P2RankPocket],
    out_dir: Path,
    *,
    max_points: int = _DEFAULT_MAX_POINTS,
    max_pockets: int = _DEFAULT_MAX_POCKETS,
) -> Optional[Path]:
    """
    Render side-by-side 3D scatter plots of predicted vs ground-truth pockets.

    Args:
        protein_id: Identifier used in output filename.
        coords: (N, 3) SAS coordinates array.
        predicted: Iterable of predicted pockets (sorted by score).
        ground_truth: Iterable of ground-truth pockets.
        out_dir: Directory to write the PNG figure into.
        max_points: Maximum number of SAS points to plot for context.
        max_pockets: Maximum pockets per panel to colour individually.

    Returns:
        Path to the saved PNG figure, or ``None`` if visualization is disabled.
    """
    if not HAS_MPL:
        logger.debug("Matplotlib not available; skipping visualization for %s", protein_id)
        return None

    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        logger.warning("Invalid coordinate array for %s; expected (N,3), got %s", protein_id, coords.shape)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{protein_id}_pockets.png"

    seed = abs(hash(protein_id)) % (2**32)
    bg_idx = _choose_background_indices(len(coords), max_points, seed)
    background = coords[bg_idx]

    cmap = plt.get_cmap("tab20")

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"{protein_id} – Pocket Predictions vs Ground Truth", fontsize=14, fontweight="bold")

    panels = [
        ("Predicted Pockets", predicted),
        ("Ground Truth Pockets", ground_truth),
    ]

    for col, (title, pocket_list) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, col, projection="3d")
        ax.set_title(title)
        ax.scatter(
            background[:, 0],
            background[:, 1],
            background[:, 2],
            s=4,
            c="#D3D3D3",
            alpha=0.15,
            linewidths=0,
        )

        for pocket_idx, (label, points, centre) in enumerate(
            _iter_pocket_points(pocket_list, coords, max_pockets)
        ):
            colour = cmap(pocket_idx % cmap.N)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=14,
                color=colour,
                alpha=0.7,
                label=label,
            )
            ax.scatter(
                centre[0],
                centre[1],
                centre[2],
                marker="*",
                s=90,
                color=colour,
                edgecolor="black",
                linewidths=0.6,
            )

        if len(pocket_list) > 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        _set_axes_equal(ax)
        ax.grid(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    logger.debug("Saved pocket visualization for %s -> %s", protein_id, out_path)
    return out_path


__all__ = ["save_pocket_visualization"]

