from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class CoordinateScaleVerdict:
    scale: float
    median_radius: float
    max_radius: float
    verdict: str


def _valid_xyz(electrode_df: pd.DataFrame) -> np.ndarray:
    coord_cols = [c for c in ["x", "y", "z"] if c in electrode_df.columns]
    if len(coord_cols) < 2:
        return np.empty((0, 0))
    xyz = electrode_df[coord_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return xyz[np.isfinite(xyz).all(axis=1)]


def radii_after_scale(electrode_df: pd.DataFrame, scale: float) -> np.ndarray:
    xyz = _valid_xyz(electrode_df)
    if xyz.size == 0:
        return np.array([])
    return np.linalg.norm(xyz * scale, axis=1)


def infer_scale_plausibility(electrode_df: pd.DataFrame, scales: Iterable[float]) -> list[CoordinateScaleVerdict]:
    verdicts: list[CoordinateScaleVerdict] = []
    for scale in scales:
        radii = radii_after_scale(electrode_df, scale)
        if len(radii) == 0:
            verdicts.append(CoordinateScaleVerdict(scale=scale, median_radius=float("nan"), max_radius=float("nan"), verdict="no_data"))
            continue
        med = float(np.nanmedian(radii))
        mx = float(np.nanmax(radii))
        # Broad heuristic ranges only; final decision should still use coordsystem metadata.
        if 0.06 <= med <= 0.13:
            verdict = "plausible_head_scale"
        elif 6 <= med <= 13:
            verdict = "looks_like_cm_or_mm_not_converted"
        elif med < 0.01:
            verdict = "too_small"
        else:
            verdict = "unlikely_scale"
        verdicts.append(CoordinateScaleVerdict(scale=scale, median_radius=med, max_radius=mx, verdict=verdict))
    return verdicts


def plot_xy_layout(electrode_df: pd.DataFrame, save_path: str | Path, title: str) -> Path:
    save_path = Path(save_path)
    xyz = _valid_xyz(electrode_df)
    fig, ax = plt.subplots(figsize=(6, 6))
    if xyz.size:
        ax.scatter(xyz[:, 0], xyz[:, 1], s=18)
        if "name" in electrode_df.columns:
            labels = electrode_df.loc[np.isfinite(xyz).all(axis=1) if False else electrode_df.index[:0], "name"]  # no-op, labels skipped
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.text(0.5, 0.5, "No valid x/y/z coordinates", ha="center", va="center")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_scale_comparison(electrode_df: pd.DataFrame, scales: Iterable[float], save_path: str | Path, title: str) -> Path:
    save_path = Path(save_path)
    scales = list(scales)
    fig, axes = plt.subplots(1, len(scales), figsize=(4 * len(scales), 4), squeeze=False)
    xyz = _valid_xyz(electrode_df)

    for ax, scale in zip(axes[0], scales):
        if xyz.size:
            scaled = xyz * scale
            ax.scatter(scaled[:, 0], scaled[:, 1], s=15)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
            med = np.nanmedian(np.linalg.norm(scaled, axis=1))
            ax.set_title(f"scale={scale:g}\nmedian r={med:.4f}")
        else:
            ax.text(0.5, 0.5, "No valid coords", ha="center", va="center")
            ax.set_title(f"scale={scale:g}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def montage_sanity_markdown(session_label: str, verdicts: list[CoordinateScaleVerdict]) -> str:
    lines = [f"## {session_label}", ""]
    if not verdicts:
        lines.append("- No verdicts available.")
        return "\n".join(lines)
    for verdict in verdicts:
        lines.append(
            f"- scale={verdict.scale:g}: median_radius={verdict.median_radius:.4f}, "
            f"max_radius={verdict.max_radius:.4f}, verdict={verdict.verdict}"
        )
    plausible = [v for v in verdicts if v.verdict == "plausible_head_scale"]
    if plausible:
        best = plausible[0]
        lines.extend(
            [
                "",
                f"Interpretation: scale `{best.scale:g}` falls into a plausible human-head range "
                "for sensor radius. This does not prove correctness, but it is a strong sanity check.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Interpretation: none of the tested scales landed in a clearly plausible head-radius range. "
                "This should be investigated before topographies are trusted.",
            ]
        )
    return "\n".join(lines)
