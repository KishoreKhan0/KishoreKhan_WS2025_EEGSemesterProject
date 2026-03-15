from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import yaml

from .audit_bids import save_dataframe, write_text


DEFAULT_PIPELINES = ("ours", "authors")
DEFAULT_SESSION_GROUPS = ("ses-01", "ses-02", "all_sessions")
DEFAULT_CONDITION_ORDER = ("standing", "walking_alone", "walking_together")
DEFAULT_ROI_CHANNELS = ("Cz", "Pz")
DEFAULT_TOPO_TIMES = (0.1, 0.2, 0.3, 0.4)


@dataclass
class PipelineComparisonConfig:
    outputs_root: Path
    ours_preprocessing_root: Path
    authors_preprocessing_root: Path
    ours_oddball_root: Path
    authors_oddball_root: Path
    pipeline_order: tuple[str, ...] = DEFAULT_PIPELINES
    session_groups: tuple[str, ...] = DEFAULT_SESSION_GROUPS
    condition_order: tuple[str, ...] = DEFAULT_CONDITION_ORDER
    roi_channels: tuple[str, ...] = DEFAULT_ROI_CHANNELS
    topomap_times: tuple[float, ...] = DEFAULT_TOPO_TIMES

    @classmethod
    def from_yaml(cls, config_path: str | Path, project_root: str | Path | None = None) -> "PipelineComparisonConfig":
        config_path = Path(config_path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        project_root = Path(project_root).resolve() if project_root else config_path.resolve().parents[1]
        return cls(
            outputs_root=(project_root / payload["outputs_root"]).resolve(),
            ours_preprocessing_root=(project_root / payload["ours_preprocessing_root"]).resolve(),
            authors_preprocessing_root=(project_root / payload["authors_preprocessing_root"]).resolve(),
            ours_oddball_root=(project_root / payload["ours_oddball_root"]).resolve(),
            authors_oddball_root=(project_root / payload["authors_oddball_root"]).resolve(),
            pipeline_order=tuple(payload.get("pipeline_order", list(DEFAULT_PIPELINES))),
            session_groups=tuple(payload.get("session_groups", list(DEFAULT_SESSION_GROUPS))),
            condition_order=tuple(payload.get("condition_order", list(DEFAULT_CONDITION_ORDER))),
            roi_channels=tuple(payload.get("roi_channels", list(DEFAULT_ROI_CHANNELS))),
            topomap_times=tuple(float(x) for x in payload.get("topomap_times", list(DEFAULT_TOPO_TIMES))),
        )


def ensure_dirs(config: PipelineComparisonConfig) -> dict[str, Path]:
    root = config.outputs_root
    tables = root / "tables"
    figures = root / "figures"
    reports = root / "reports"
    for p in (root, tables, figures, reports):
        p.mkdir(parents=True, exist_ok=True)
    return {"root": root, "tables": tables, "figures": figures, "reports": reports}


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _safe_json(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _pipeline_paths(config: PipelineComparisonConfig, pipeline: str) -> dict[str, Path]:
    if pipeline == "ours":
        pre_root = config.ours_preprocessing_root
        odd_root = config.ours_oddball_root
    elif pipeline == "authors":
        pre_root = config.authors_preprocessing_root
        odd_root = config.authors_oddball_root
    else:
        raise ValueError(f"Unsupported pipeline: {pipeline}")
    return {
        "pre_manifest": pre_root / "tables" / "preprocessing_manifest.csv",
        "odd_run_manifest": odd_root / "tables" / "run_level_evoked_manifest.csv",
        "odd_subject_manifest": odd_root / "tables" / "subject_level_evoked_manifest.csv",
        "odd_group_manifest": odd_root / "tables" / "group_level_evoked_manifest.csv",
        "odd_diff_manifest": odd_root / "tables" / "difference_wave_manifest.csv",
        "odd_peak_metrics": odd_root / "tables" / "peak_metrics.csv",
    }


def read_pipeline_outputs(config: PipelineComparisonConfig) -> dict[str, dict[str, pd.DataFrame]]:
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for pipeline in config.pipeline_order:
        paths = _pipeline_paths(config, pipeline)
        outputs[pipeline] = {name: _safe_csv(path) for name, path in paths.items()}
    return outputs


def build_availability_table(config: PipelineComparisonConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pipeline in config.pipeline_order:
        for label, path in _pipeline_paths(config, pipeline).items():
            rows.append({
                "pipeline": pipeline,
                "artifact": label,
                "path": str(path),
                "exists": bool(path.exists()),
            })
    return pd.DataFrame(rows)


def summarize_preprocessing(config: PipelineComparisonConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pipeline in config.pipeline_order:
        manifest = _safe_csv(_pipeline_paths(config, pipeline)["pre_manifest"])
        if manifest.empty:
            rows.append({
                "pipeline": pipeline,
                "processed_runs": 0,
                "mean_annotations": np.nan,
                "mean_sampling_rate_after": np.nan,
                "mean_bad_channels_union": np.nan,
                "mean_interpolated_channels": np.nan,
                "mean_ica_n_components": np.nan,
                "mean_ica_excluded_components": np.nan,
            })
            continue
        ica_n = []
        ica_excl = []
        bad_union = []
        interpolated = []
        for rec in manifest.to_dict(orient="records"):
            summary = _safe_json(Path(str(rec.get("summary_json", ""))))
            ica_summary = summary.get("ica_summary", {}) if isinstance(summary, dict) else {}
            ica_n.append(ica_summary.get("n_components", np.nan))
            excl = ica_summary.get("excluded_components", [])
            ica_excl.append(len(excl) if isinstance(excl, list) else np.nan)
            bad_union.append(len(summary.get("bads_union", []) or []))
            interpolated.append(len(summary.get("interpolated_channels", []) or []))
        bad_union_arr = pd.to_numeric(pd.Series(bad_union, dtype=float), errors="coerce").to_numpy(dtype=float)
        interpolated_arr = pd.to_numeric(pd.Series(interpolated, dtype=float), errors="coerce").to_numpy(dtype=float)
        ica_n_arr = pd.to_numeric(pd.Series(ica_n, dtype=float), errors="coerce").to_numpy(dtype=float)
        ica_excl_arr = pd.to_numeric(pd.Series(ica_excl, dtype=float), errors="coerce").to_numpy(dtype=float)
        rows.append({
            "pipeline": pipeline,
            "processed_runs": int(len(manifest)),
            "mean_annotations": float(pd.to_numeric(manifest.get("n_annotations", pd.Series(dtype=float)), errors="coerce").mean()),
            "mean_sampling_rate_after": float(pd.to_numeric(manifest.get("sampling_rate_hz_after", pd.Series(dtype=float)), errors="coerce").mean()),
            "mean_bad_channels_union": float(np.nanmean(bad_union_arr)) if np.isfinite(bad_union_arr).any() else np.nan,
            "mean_interpolated_channels": float(np.nanmean(interpolated_arr)) if np.isfinite(interpolated_arr).any() else np.nan,
            "mean_ica_n_components": float(np.nanmean(ica_n_arr)) if np.isfinite(ica_n_arr).any() else np.nan,
            "mean_ica_excluded_components": float(np.nanmean(ica_excl_arr)) if np.isfinite(ica_excl_arr).any() else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_oddball(config: PipelineComparisonConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    peak_rows: list[pd.DataFrame] = []
    for pipeline in config.pipeline_order:
        paths = _pipeline_paths(config, pipeline)
        run_manifest = _safe_csv(paths["odd_run_manifest"])
        subject_manifest = _safe_csv(paths["odd_subject_manifest"])
        group_manifest = _safe_csv(paths["odd_group_manifest"])
        diff_manifest = _safe_csv(paths["odd_diff_manifest"])
        peak_metrics = _safe_csv(paths["odd_peak_metrics"])
        summary_rows.append({
            "pipeline": pipeline,
            "run_level_cells": int(len(run_manifest)),
            "subject_session_cells": int(len(subject_manifest)),
            "group_average_cells": int(len(group_manifest)),
            "difference_waves": int(len(diff_manifest)),
            "peak_metric_rows": int(len(peak_metrics)),
        })
        if not peak_metrics.empty:
            tmp = peak_metrics.copy()
            tmp["pipeline"] = pipeline
            peak_rows.append(tmp)
    peak_long = pd.concat(peak_rows, ignore_index=True) if peak_rows else pd.DataFrame()
    return pd.DataFrame(summary_rows), peak_long


def make_peak_wide_table(peak_long: pd.DataFrame) -> pd.DataFrame:
    if peak_long.empty:
        return pd.DataFrame()
    value_cols = ["peak_amplitude_uv", "peak_latency_s"]
    wide_frames = []
    index_cols = ["session_group", "condition", "channel"]
    for value_col in value_cols:
        pivot = peak_long.pivot_table(index=index_cols, columns="pipeline", values=value_col, aggfunc="mean")
        pivot = pivot.reset_index()
        rename = {col: f"{value_col}_{col}" for col in pivot.columns if col not in index_cols}
        pivot = pivot.rename(columns=rename)
        wide_frames.append(pivot)
    merged = wide_frames[0]
    for other in wide_frames[1:]:
        merged = merged.merge(other, on=index_cols, how="outer")
    if {"peak_amplitude_uv_ours", "peak_amplitude_uv_authors"}.issubset(merged.columns):
        merged["amplitude_delta_ours_minus_authors_uv"] = merged["peak_amplitude_uv_ours"] - merged["peak_amplitude_uv_authors"]
    if {"peak_latency_s_ours", "peak_latency_s_authors"}.issubset(merged.columns):
        merged["latency_delta_ours_minus_authors_s"] = merged["peak_latency_s_ours"] - merged["peak_latency_s_authors"]
    return merged.sort_values(index_cols).reset_index(drop=True)


def _plot_preprocessing_summary(pre_summary: pd.DataFrame, save_path: Path) -> Path:
    metrics = [
        "processed_runs",
        "mean_bad_channels_union",
        "mean_interpolated_channels",
        "mean_ica_n_components",
        "mean_ica_excluded_components",
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3 * len(metrics)))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    x = np.arange(len(pre_summary))
    labels = pre_summary["pipeline"].astype(str).tolist()
    for ax, metric in zip(axes, metrics):
        vals = pd.to_numeric(pre_summary[metric], errors="coerce").to_numpy(dtype=float)
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(metric.replace("_", " "))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _plot_peak_comparison(peak_wide: pd.DataFrame, save_path: Path, pipeline_order: tuple[str, ...]) -> Path:
    if peak_wide.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No peak metrics available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = np.arange(len(peak_wide))
    labels = peak_wide.apply(lambda r: f"{r['session_group']}\n{r['condition']}\n{r['channel']}", axis=1)
    for pipeline in pipeline_order:
        amp_col = f"peak_amplitude_uv_{pipeline}"
        lat_col = f"peak_latency_s_{pipeline}"
        if amp_col in peak_wide.columns:
            axes[0].plot(x, peak_wide[amp_col], marker="o", label=pipeline)
        if lat_col in peak_wide.columns:
            axes[1].plot(x, peak_wide[lat_col], marker="o", label=pipeline)
    axes[0].set_ylabel("Peak amplitude (µV)")
    axes[1].set_ylabel("Peak latency (s)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    axes[0].set_title("Difference-wave peak amplitude comparison")
    axes[1].set_title("Difference-wave peak latency comparison")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _safe_read_evoked(path: str | Path) -> mne.Evoked | None:
    try:
        return mne.read_evokeds(path, condition=0, verbose="ERROR")
    except Exception:
        return None


def _plot_difference_wave_overlay(config: PipelineComparisonConfig, diff_tables: dict[str, pd.DataFrame],
                                  save_path: Path, session_group: str, condition: str) -> Path:
    fig, axes = plt.subplots(len(config.roi_channels), 1, figsize=(10, 3.5 * max(1, len(config.roi_channels))), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    any_curve = False
    for pipeline, manifest in diff_tables.items():
        if manifest.empty:
            continue
        row = manifest[(manifest["session_group"].astype(str) == session_group) & (manifest["condition"].astype(str) == condition)]
        if row.empty:
            continue
        evoked = _safe_read_evoked(row.iloc[0]["evoked_file"])
        if evoked is None:
            continue
        for ax, ch in zip(axes, config.roi_channels):
            if ch not in evoked.ch_names:
                continue
            ax.plot(evoked.times, evoked.get_data(picks=[ch])[0] * 1e6, label=pipeline)
            any_curve = True
    for ax, ch in zip(axes, config.roi_channels):
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_ylabel("µV")
        ax.set_title(f"{session_group} | {condition} | {ch}")
        if any_curve:
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "No evoked files available", ha="center", va="center", transform=ax.transAxes)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _plot_topomap_comparison(config: PipelineComparisonConfig, diff_tables: dict[str, pd.DataFrame],
                             save_path: Path, session_group: str, condition: str) -> Path:
    n_rows = len(config.pipeline_order)
    n_cols = len(config.topomap_times)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.9 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    for r, pipeline in enumerate(config.pipeline_order):
        manifest = diff_tables.get(pipeline, pd.DataFrame())
        row = manifest[(manifest.get("session_group", pd.Series(dtype=str)).astype(str) == session_group) &
                       (manifest.get("condition", pd.Series(dtype=str)).astype(str) == condition)] if not manifest.empty else pd.DataFrame()
        if row.empty:
            for c in range(n_cols):
                ax = axes[r, c]
                ax.text(0.5, 0.5, f"No data\n{pipeline}", ha="center", va="center")
                ax.axis("off")
            continue
        evoked = _safe_read_evoked(row.iloc[0]["evoked_file"])
        if evoked is None:
            for c in range(n_cols):
                ax = axes[r, c]
                ax.text(0.5, 0.5, f"Unreadable\n{pipeline}", ha="center", va="center")
                ax.axis("off")
            continue
        try:
            evoked.plot_topomap(times=list(config.topomap_times), axes=list(axes[r]), show=False, colorbar=False, time_unit="s")
            axes[r, 0].set_ylabel(pipeline)
        except Exception:
            for c in range(n_cols):
                ax = axes[r, c]
                ax.text(0.5, 0.5, f"Topomap unavailable\n{pipeline}", ha="center", va="center")
                ax.axis("off")
    fig.suptitle(f"Topomap comparison | {session_group} | {condition}", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def write_report(config: PipelineComparisonConfig, availability: pd.DataFrame, pre_summary: pd.DataFrame,
                 odd_summary: pd.DataFrame, peak_wide: pd.DataFrame, figure_paths: list[Path], dirs: dict[str, Path]) -> Path:
    lines = [
        "# Pipeline comparison report",
        "",
        "This report compares the current outputs from the `ours` and `authors` branches.",
        "",
        "## Availability",
        "",
    ]
    if availability.empty:
        lines.append("No comparison inputs were found.")
    else:
        for rec in availability.to_dict(orient="records"):
            state = "present" if rec["exists"] else "missing"
            lines.append(f"- {rec['pipeline']} | {rec['artifact']}: **{state}**")
    lines.extend(["", "## Preprocessing summary", ""])
    if pre_summary.empty:
        lines.append("No preprocessing manifests found.")
    else:
        for rec in pre_summary.to_dict(orient="records"):
            lines.append(
                f"- {rec['pipeline']}: runs={rec['processed_runs']}, "
                f"mean bad-channel union={rec['mean_bad_channels_union']}, "
                f"mean interpolated={rec['mean_interpolated_channels']}, "
                f"mean ICA comps={rec['mean_ica_n_components']}"
            )
    lines.extend(["", "## Oddball replication summary", ""])
    if odd_summary.empty:
        lines.append("No oddball summary tables found.")
    else:
        for rec in odd_summary.to_dict(orient="records"):
            lines.append(
                f"- {rec['pipeline']}: run cells={rec['run_level_cells']}, "
                f"subject/session cells={rec['subject_session_cells']}, "
                f"group averages={rec['group_average_cells']}, "
                f"difference waves={rec['difference_waves']}, "
                f"peak rows={rec['peak_metric_rows']}"
            )
    lines.extend(["", "## Interpretation prompts", ""])
    lines.extend([
        "- This seems correct because both pipelines are being compared on the same exported manifest structure.",
        "- This is strange because any large mismatch in run counts usually means one pipeline skipped files or failed during epoching.",
        "- This is important because peak amplitude/latency differences can come from preprocessing choices, not only neural differences.",
        "",
        "## Generated figures",
        "",
    ])
    for fig in figure_paths:
        lines.append(f"- `{fig}`")
    if not peak_wide.empty:
        lines.extend(["", "## Peak deltas preview", ""])
        preview = peak_wide.head(12).to_dict(orient="records")
        for rec in preview:
            lines.append(
                f"- {rec['session_group']} | {rec['condition']} | {rec['channel']}: "
                f"Δamp={rec.get('amplitude_delta_ours_minus_authors_uv', np.nan)}, "
                f"Δlat={rec.get('latency_delta_ours_minus_authors_s', np.nan)}"
            )
    return write_text(dirs["reports"] / "pipeline_comparison_report.md", "\n".join(lines))


def run_pipeline_comparison(config: PipelineComparisonConfig) -> dict[str, pd.DataFrame]:
    dirs = ensure_dirs(config)
    availability = build_availability_table(config)
    pre_summary = summarize_preprocessing(config)
    odd_summary, peak_long = summarize_oddball(config)
    peak_wide = make_peak_wide_table(peak_long)

    save_dataframe(availability, dirs["tables"] / "artifact_availability.csv")
    save_dataframe(pre_summary, dirs["tables"] / "preprocessing_summary.csv")
    save_dataframe(odd_summary, dirs["tables"] / "oddball_summary.csv")
    save_dataframe(peak_long, dirs["tables"] / "peak_metrics_long.csv")
    save_dataframe(peak_wide, dirs["tables"] / "peak_metrics_comparison.csv")

    figure_paths: list[Path] = []
    figure_paths.append(_plot_preprocessing_summary(pre_summary, dirs["figures"] / "preprocessing_summary.png"))
    figure_paths.append(_plot_peak_comparison(peak_wide, dirs["figures"] / "peak_metrics_comparison.png", config.pipeline_order))

    diff_tables = {pipeline: _safe_csv(_pipeline_paths(config, pipeline)["odd_diff_manifest"]) for pipeline in config.pipeline_order}
    for session_group in config.session_groups:
        for condition in config.condition_order:
            figure_paths.append(
                _plot_difference_wave_overlay(
                    config,
                    diff_tables,
                    dirs["figures"] / f"difference_overlay_{session_group}_{condition}.png",
                    session_group=session_group,
                    condition=condition,
                )
            )
            figure_paths.append(
                _plot_topomap_comparison(
                    config,
                    diff_tables,
                    dirs["figures"] / f"topomap_comparison_{session_group}_{condition}.png",
                    session_group=session_group,
                    condition=condition,
                )
            )

    write_report(config, availability, pre_summary, odd_summary, peak_wide, figure_paths, dirs)
    return {
        "availability": availability,
        "preprocessing_summary": pre_summary,
        "oddball_summary": odd_summary,
        "peak_metrics_long": peak_long,
        "peak_metrics_comparison": peak_wide,
    }
