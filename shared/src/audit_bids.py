from __future__ import annotations

import json
import math
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml


DEFAULT_EVENT_COLUMNS = ["trial_type", "value", "event_type", "HED", "stimulus", "condition"]


@dataclass
class AuditConfig:
    dataset_root: Path
    audit_output_root: Path
    dataset_name: str = "ds004033"
    expected_sampling_rate_hz: float = 500.0
    raw_spotcheck_max_files: int = 4
    montage_preview_max_runs: int = 2
    random_seed: int = 42
    eeg_extensions: tuple[str, ...] = (".set", ".vhdr", ".edf", ".bdf")
    event_candidate_columns: tuple[str, ...] = tuple(DEFAULT_EVENT_COLUMNS)
    coordinate_scale_checks: tuple[float, ...] = (1.0, 0.1, 0.01, 10.0, 100.0)

    @classmethod
    def from_yaml(cls, config_path: str | Path, project_root: str | Path | None = None) -> "AuditConfig":
        config_path = Path(config_path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        project_root = Path(project_root).resolve() if project_root else config_path.resolve().parents[2]
        dataset_root = (project_root / payload["dataset_root"]).resolve()
        output_root = (project_root / payload["audit_output_root"]).resolve()
        return cls(
            dataset_root=dataset_root,
            audit_output_root=output_root,
            dataset_name=payload.get("dataset_name", "ds004033"),
            expected_sampling_rate_hz=float(payload.get("expected_sampling_rate_hz", 500)),
            raw_spotcheck_max_files=int(payload.get("raw_spotcheck_max_files", 4)),
            montage_preview_max_runs=int(payload.get("montage_preview_max_runs", 2)),
            random_seed=int(payload.get("random_seed", 42)),
            eeg_extensions=tuple(payload.get("eeg_extensions", [".set", ".vhdr", ".edf", ".bdf"])),
            event_candidate_columns=tuple(payload.get("event_candidate_columns", DEFAULT_EVENT_COLUMNS)),
            coordinate_scale_checks=tuple(float(x) for x in payload.get("coordinate_scale_checks", [1.0, 0.1, 0.01, 10.0, 100.0])),
        )


def ensure_output_dirs(config: AuditConfig) -> dict[str, Path]:
    root = config.audit_output_root
    tables = root / "tables"
    figures = root / "figures"
    reports = root / "reports"
    for p in (root, tables, figures, reports):
        p.mkdir(parents=True, exist_ok=True)
    return {"root": root, "tables": tables, "figures": figures, "reports": reports}


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)
    return path


def safe_read_tsv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep="\t", encoding="latin1")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="latin1"))
    except Exception:
        return {}


def parse_bids_entities(file_path: str | Path) -> dict[str, str | None]:
    file_path = Path(file_path)
    stem = file_path.name
    parts = stem.split("_")
    entities: dict[str, str | None] = {
        "subject": None,
        "session": None,
        "task": None,
        "run": None,
        "suffix": None,
    }
    for part in parts:
        if part.startswith("sub-"):
            entities["subject"] = part
        elif part.startswith("ses-"):
            entities["session"] = part
        elif part.startswith("task-"):
            entities["task"] = part.replace("task-", "", 1)
        elif part.startswith("run-"):
            entities["run"] = part.replace("run-", "", 1)
    if "_eeg" in stem:
        entities["suffix"] = "eeg"
    return entities


def list_subjects(dataset_root: Path) -> list[str]:
    return sorted(p.name for p in dataset_root.glob("sub-*") if p.is_dir())


def collect_run_inventory(config: AuditConfig) -> pd.DataFrame:
    dataset_root = config.dataset_root
    rows: list[dict[str, Any]] = []
    eeg_files: list[Path] = []
    for ext in config.eeg_extensions:
        eeg_files.extend(sorted(dataset_root.rglob(f"*{ext}")))
    eeg_files = [p for p in eeg_files if "_eeg" in p.name and "/eeg/" in p.as_posix()]

    for eeg_path in eeg_files:
        entities = parse_bids_entities(eeg_path)
        base_no_ext = eeg_path.name.replace(eeg_path.suffix, "")
        events_tsv = eeg_path.with_name(base_no_ext.replace("_eeg", "_events.tsv"))
        events_json = eeg_path.with_name(base_no_ext.replace("_eeg", "_events.json"))
        channels_tsv = eeg_path.with_name(base_no_ext.replace("_eeg", "_channels.tsv"))
        electrodes_tsv = eeg_path.with_name(base_no_ext.replace("_eeg", "_electrodes.tsv"))
        coordsystem_json = eeg_path.with_name(base_no_ext.replace("_eeg", "_coordsystem.json"))
        if not electrodes_tsv.exists():
            electrodes_tsv = eeg_path.parent.parent / "eeg" / electrodes_tsv.name
        if not coordsystem_json.exists():
            coordsystem_json = eeg_path.parent.parent / "eeg" / coordsystem_json.name

        rows.append(
            {
                "subject": entities["subject"],
                "session": entities["session"],
                "task": entities["task"],
                "run": entities["run"],
                "eeg_file": str(eeg_path),
                "eeg_extension": eeg_path.suffix.lower(),
                "events_tsv": str(events_tsv),
                "events_json": str(events_json),
                "channels_tsv": str(channels_tsv),
                "electrodes_tsv": str(electrodes_tsv),
                "coordsystem_json": str(coordsystem_json),
                "events_tsv_exists": events_tsv.exists(),
                "events_json_exists": events_json.exists(),
                "channels_tsv_exists": channels_tsv.exists(),
                "electrodes_tsv_exists": electrodes_tsv.exists(),
                "coordsystem_json_exists": coordsystem_json.exists(),
            }
        )
    columns = [
        "subject", "session", "task", "run", "eeg_file", "eeg_extension",
        "events_tsv", "events_json", "channels_tsv", "electrodes_tsv", "coordsystem_json",
        "events_tsv_exists", "events_json_exists", "channels_tsv_exists",
        "electrodes_tsv_exists", "coordsystem_json_exists",
    ]
    inventory = pd.DataFrame(rows, columns=columns)
    if inventory.empty:
        return inventory
    return inventory.sort_values(["subject", "session", "task", "run"], na_position="last").reset_index(drop=True)


def presence_check(config: AuditConfig, inventory: pd.DataFrame) -> str:
    dataset_root = config.dataset_root
    checks = {
        "dataset_root_exists": dataset_root.exists(),
        "dataset_description_exists": (dataset_root / "dataset_description.json").exists(),
        "participants_tsv_exists": (dataset_root / "participants.tsv").exists(),
        "has_subject_folders": any(dataset_root.glob("sub-*")),
        "inventory_nonempty": not inventory.empty,
        "all_runs_have_channels_tsv": bool(inventory["channels_tsv_exists"].all()) if not inventory.empty else False,
        "all_runs_have_events_tsv": bool(inventory["events_tsv_exists"].all()) if not inventory.empty else False,
    }
    lines = ["BIDS presence check", "==================", ""]
    for key, value in checks.items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    return "\n".join(lines)


def environment_snapshot(config: AuditConfig) -> pd.DataFrame:
    rows = [
        {"key": "python_version", "value": sys.version.replace("\n", " ")},
        {"key": "platform", "value": platform.platform()},
        {"key": "mne_version", "value": mne.__version__},
        {"key": "numpy_version", "value": np.__version__},
        {"key": "pandas_version", "value": pd.__version__},
        {"key": "dataset_root", "value": str(config.dataset_root.resolve())},
        {"key": "expected_sampling_rate_hz", "value": str(config.expected_sampling_rate_hz)},
    ]
    return pd.DataFrame(rows)


def build_inventory_summary(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if inventory.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    runs_per_subject = inventory.groupby("subject", dropna=False).size().reset_index(name="n_runs")
    runs_per_session = inventory.groupby("session", dropna=False).size().reset_index(name="n_runs")
    runs_per_task = inventory.groupby("task", dropna=False).size().reset_index(name="n_runs")
    return runs_per_subject, runs_per_session, runs_per_task


def plot_count_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, save_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df[x_col].astype(str), df[y_col].astype(float))
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def collect_event_definitions(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if inventory.empty:
        return pd.DataFrame()

    for row in inventory.itertuples(index=False):
        events_json = read_json(row.events_json)
        for key, value in events_json.items():
            if isinstance(value, dict):
                rows.append(
                    {
                        "subject": row.subject,
                        "session": row.session,
                        "task": row.task,
                        "run": row.run,
                        "event_key": key,
                        "description": value.get("Description"),
                        "levels": json.dumps(value.get("Levels")) if "Levels" in value else None,
                        "units": value.get("Units"),
                        "source_json": row.events_json,
                    }
                )
            else:
                rows.append(
                    {
                        "subject": row.subject,
                        "session": row.session,
                        "task": row.task,
                        "run": row.run,
                        "event_key": key,
                        "description": str(value),
                        "levels": None,
                        "units": None,
                        "source_json": row.events_json,
                    }
                )
    return pd.DataFrame(rows)


def _pick_event_label_column(df: pd.DataFrame, candidate_columns: Iterable[str]) -> str | None:
    for col in candidate_columns:
        if col in df.columns:
            return col
    return None


def collect_event_occurrences(inventory: pd.DataFrame, candidate_columns: Iterable[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if inventory.empty:
        return pd.DataFrame()

    for row in inventory.itertuples(index=False):
        event_df = safe_read_tsv(row.events_tsv)
        if event_df.empty:
            continue
        event_df = event_df.copy()
        label_col = _pick_event_label_column(event_df, candidate_columns)
        if label_col is None:
            label_col = "UNRESOLVED_LABEL"
            event_df[label_col] = "UNRESOLVED_LABEL"
        meta = {
            "subject": row.subject,
            "session": row.session,
            "task": row.task,
            "run": row.run,
            "source_events_tsv": row.events_tsv,
            "label_column_used": label_col,
        }
        for k, v in meta.items():
            event_df[k] = v
        event_df["event_label"] = event_df[label_col].astype(str)
        rows.append(event_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarise_event_counts(event_occurrences: pd.DataFrame) -> pd.DataFrame:
    if event_occurrences.empty:
        return pd.DataFrame()
    group_cols = ["subject", "session", "task", "run", "event_label"]
    return (
        event_occurrences.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_events")
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def plot_event_label_counts(event_counts: pd.DataFrame, save_path: Path, top_n: int = 30) -> Path:
    if event_counts.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("No event counts available")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    counts = event_counts.groupby("event_label")["n_events"].sum().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(counts.index.astype(str), counts.values.astype(float))
    ax.set_title("Most frequent event labels")
    ax.set_ylabel("Total count")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_event_heatmap(event_counts: pd.DataFrame, save_path: Path, max_labels: int = 20) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    if not event_counts.empty:
        totals = event_counts.groupby("event_label")["n_events"].sum().sort_values(ascending=False)
        labels = list(totals.head(max_labels).index)
        pivot = (
            event_counts[event_counts["event_label"].isin(labels)]
            .pivot_table(index="subject", columns="event_label", values="n_events", aggfunc="sum", fill_value=0)
            .sort_index()
        )
        img = ax.imshow(pivot.to_numpy(), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=75)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Event count heatmap by subject")
        fig.colorbar(img, ax=ax, label="Count")
    else:
        ax.set_title("No event counts available")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def infer_event_group(label: str) -> str:
    text = str(label).lower()
    if any(x in text for x in ["eyes", "open", "closed"]):
        return "eyes_open_closed"
    if "oddball" in text or "standard" in text or "deviant" in text:
        if "stand" in text:
            return "oddball_standing"
        if "alone" in text:
            return "oddball_walk_alone"
        if "together" in text or "partner" in text:
            return "oddball_walk_together"
        return "oddball_other"
    if "blocked" in text:
        return "sync_blocked"
    if "natural" in text:
        return "sync_natural"
    if "sync" in text:
        return "sync_sync"
    if "rhs" in text or "right heel strike" in text:
        return "gait_rhs"
    if "rto" in text or "right toe off" in text:
        return "gait_rto"
    if "heel" in text or "toe" in text or "hs" in text or "to" in text:
        return "gait_other"
    if any(x in text for x in ["countdown", "instruction", "start", "stop", "beep"]):
        return "instruction_or_boundary"
    return "unknown"


def build_provisional_event_groups(event_counts: pd.DataFrame) -> pd.DataFrame:
    if event_counts.empty:
        return pd.DataFrame(columns=["event_label", "provisional_group", "total_count"])
    totals = event_counts.groupby("event_label")["n_events"].sum().reset_index(name="total_count")
    totals["provisional_group"] = totals["event_label"].map(infer_event_group)
    return totals.sort_values(["provisional_group", "event_label"]).reset_index(drop=True)


def collect_channel_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in inventory.itertuples(index=False):
        ch_df = safe_read_tsv(row.channels_tsv)
        if ch_df.empty:
            continue
        ch_df = ch_df.copy()
        ch_df["subject"] = row.subject
        ch_df["session"] = row.session
        ch_df["task"] = row.task
        ch_df["run"] = row.run
        ch_df["source_channels_tsv"] = row.channels_tsv
        frames.append(ch_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise_channel_status(channel_inventory: pd.DataFrame) -> pd.DataFrame:
    if channel_inventory.empty:
        return pd.DataFrame()
    status_col = "status" if "status" in channel_inventory.columns else None
    type_col = "type" if "type" in channel_inventory.columns else None
    group_cols = [c for c in [type_col, status_col] if c is not None]
    if not group_cols:
        return pd.DataFrame({"note": ["No type/status columns found in channels.tsv"]})
    return channel_inventory.groupby(group_cols, dropna=False).size().reset_index(name="n_channels")


def collect_electrode_metadata(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for row in inventory.itertuples(index=False):
        elec_df = safe_read_tsv(row.electrodes_tsv)
        coord_json = read_json(row.coordsystem_json)
        if elec_df.empty:
            continue
        elec_df = elec_df.copy()
        for coord_col in ["x", "y", "z"]:
            if coord_col in elec_df.columns:
                elec_df[coord_col] = pd.to_numeric(elec_df[coord_col], errors="coerce")
        for key, value in {
            "subject": row.subject,
            "session": row.session,
            "task": row.task,
            "run": row.run,
            "source_electrodes_tsv": row.electrodes_tsv,
            "source_coordsystem_json": row.coordsystem_json,
        }.items():
            elec_df[key] = value
        long_rows.append(elec_df)

        coord_cols = [c for c in ["x", "y", "z"] if c in elec_df.columns]
        xyz = elec_df[coord_cols].to_numpy(dtype=float) if coord_cols else np.empty((0, 0))
        valid = xyz[np.isfinite(xyz).all(axis=1)] if xyz.size else np.empty((0, 0))
        radii = np.linalg.norm(valid, axis=1) if valid.size else np.array([])
        summary_rows.append(
            {
                "subject": row.subject,
                "session": row.session,
                "task": row.task,
                "run": row.run,
                "n_electrodes": int(len(elec_df)),
                "x_min": float(np.nanmin(elec_df["x"])) if "x" in elec_df.columns else math.nan,
                "x_max": float(np.nanmax(elec_df["x"])) if "x" in elec_df.columns else math.nan,
                "y_min": float(np.nanmin(elec_df["y"])) if "y" in elec_df.columns else math.nan,
                "y_max": float(np.nanmax(elec_df["y"])) if "y" in elec_df.columns else math.nan,
                "z_min": float(np.nanmin(elec_df["z"])) if "z" in elec_df.columns else math.nan,
                "z_max": float(np.nanmax(elec_df["z"])) if "z" in elec_df.columns else math.nan,
                "median_radius": float(np.nanmedian(radii)) if len(radii) else math.nan,
                "mean_radius": float(np.nanmean(radii)) if len(radii) else math.nan,
                "max_radius": float(np.nanmax(radii)) if len(radii) else math.nan,
                "EEGCoordinateSystem": coord_json.get("EEGCoordinateSystem"),
                "EEGCoordinateUnits": coord_json.get("EEGCoordinateUnits"),
            }
        )
    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return long_df, summary_df


def plot_electrode_radius_histogram(summary_df: pd.DataFrame, save_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not summary_df.empty and "median_radius" in summary_df.columns:
        data = pd.to_numeric(summary_df["median_radius"], errors="coerce").dropna()
        if not data.empty:
            ax.hist(data, bins=min(20, max(5, len(data))))
            ax.set_xlabel("Median electrode radius")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of run-wise median electrode radius")
        else:
            ax.set_title("No valid radius values found")
    else:
        ax.set_title("No electrode coordinate summary available")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def collect_coordsystem_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in inventory.itertuples(index=False):
        payload = read_json(row.coordsystem_json)
        if not payload:
            continue
        payload = dict(payload)
        payload["subject"] = row.subject
        payload["session"] = row.session
        payload["task"] = row.task
        payload["run"] = row.run
        payload["source_coordsystem_json"] = row.coordsystem_json
        rows.append(payload)
    return pd.DataFrame(rows)


def select_montage_preview_runs(inventory: pd.DataFrame, max_runs: int = 2) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    dedup = inventory.drop_duplicates(subset=["session", "task"]).copy()
    if len(dedup) <= max_runs:
        return dedup
    preferred: list[pd.DataFrame] = []
    for session_name in ["ses-active", "ses-passive"]:
        chunk = dedup[dedup["session"] == session_name]
        if not chunk.empty:
            preferred.append(chunk.iloc[[0]])
    if preferred:
        out = pd.concat(preferred, ignore_index=True)
        return out.head(max_runs)
    return dedup.head(max_runs)


def read_raw_metadata(eeg_path: str | Path) -> dict[str, Any]:
    eeg_path = Path(eeg_path)
    info: dict[str, Any] = {
        "eeg_file": str(eeg_path),
        "reader_status": "ok",
        "reader_error": None,
        "sfreq": math.nan,
        "n_channels": math.nan,
        "duration_sec": math.nan,
        "annotation_count": math.nan,
    }
    try:
        ext = eeg_path.suffix.lower()
        if ext == ".set":
            raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose="ERROR")
        elif ext == ".vhdr":
            raw = mne.io.read_raw_brainvision(eeg_path, preload=False, verbose="ERROR")
        elif ext == ".edf":
            raw = mne.io.read_raw_edf(eeg_path, preload=False, verbose="ERROR")
        elif ext == ".bdf":
            raw = mne.io.read_raw_bdf(eeg_path, preload=False, verbose="ERROR")
        else:
            raise ValueError(f"Unsupported EEG extension: {ext}")
        info.update(
            {
                "sfreq": float(raw.info["sfreq"]),
                "n_channels": int(len(raw.ch_names)),
                "duration_sec": float(raw.n_times / raw.info["sfreq"]),
                "annotation_count": int(len(raw.annotations)),
            }
        )
    except Exception as exc:
        info["reader_status"] = "error"
        info["reader_error"] = repr(exc)
    return info


def spot_check_raw_files(inventory: pd.DataFrame, max_files: int, random_seed: int) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    rng = random.Random(random_seed)
    eeg_files = inventory["eeg_file"].tolist()
    if len(eeg_files) > max_files:
        eeg_files = rng.sample(eeg_files, max_files)
    rows = [read_raw_metadata(p) for p in eeg_files]
    meta_df = pd.DataFrame(rows)
    inventory_lookup = inventory[["eeg_file", "subject", "session", "task", "run"]]
    return meta_df.merge(inventory_lookup, on="eeg_file", how="left")


def inventory_summary_markdown(
    config: AuditConfig,
    inventory: pd.DataFrame,
    runs_per_subject: pd.DataFrame,
    runs_per_session: pd.DataFrame,
    runs_per_task: pd.DataFrame,
) -> str:
    lines = [
        "# Inventory summary",
        "",
        f"- Dataset root: `{config.dataset_root}`",
        f"- Number of detected EEG runs: **{len(inventory)}**",
        f"- Number of detected subjects: **{inventory['subject'].nunique() if not inventory.empty else 0}**",
        f"- Sessions found: **{', '.join(sorted(inventory['session'].dropna().unique())) if not inventory.empty else 'none'}**",
        f"- Tasks found: **{', '.join(sorted(inventory['task'].dropna().astype(str).unique())) if not inventory.empty else 'none'}**",
        "",
        "## Runs per session",
        "",
    ]
    if not runs_per_session.empty:
        for row in runs_per_session.itertuples(index=False):
            lines.append(f"- {row.session}: {row.n_runs}")
    else:
        lines.append("- No runs found")
    lines.extend(["", "## Runs per task", ""])
    if not runs_per_task.empty:
        for row in runs_per_task.itertuples(index=False):
            lines.append(f"- {row.task}: {row.n_runs}")
    else:
        lines.append("- No tasks found")
    return "\n".join(lines)


def audit_conclusion_markdown(
    config: AuditConfig,
    inventory: pd.DataFrame,
    provisional_groups: pd.DataFrame,
    electrode_summary: pd.DataFrame,
    raw_meta: pd.DataFrame,
) -> str:
    lines = [
        "# Data audit report",
        "",
        "## High-level findings",
        "",
        f"- Dataset root checked: `{config.dataset_root}`",
        f"- EEG runs detected: **{len(inventory)}**",
        f"- Unique subjects detected: **{inventory['subject'].nunique() if not inventory.empty else 0}**",
        f"- Unique tasks detected: **{inventory['task'].nunique() if not inventory.empty else 0}**",
        "",
        "## Event structure",
        "",
    ]
    if provisional_groups.empty:
        lines.append("- No event labels were extracted; event mapping cannot proceed yet.")
    else:
        for row in provisional_groups.groupby("provisional_group")["total_count"].sum().reset_index(name="n").itertuples(index=False):
            lines.append(f"- {row.provisional_group}: total count {row.n}")
    lines.extend(["", "## Electrode / montage sanity", ""])
    if electrode_summary.empty:
        lines.append("- No electrode metadata found.")
    else:
        med = pd.to_numeric(electrode_summary["median_radius"], errors="coerce").dropna()
        if med.empty:
            lines.append("- Electrode metadata exists, but coordinate radii could not be computed.")
        else:
            lines.append(
                f"- Median of run-wise median electrode radii: **{float(med.median()):.4f}**"
            )
            lines.append(
                "- This should be checked against the expected unit in `coordsystem.json` before any topomap interpretation."
            )
    lines.extend(["", "## Raw-file spot check", ""])
    if raw_meta.empty:
        lines.append("- No raw files were spot-checked.")
    else:
        failures = int((raw_meta["reader_status"] != "ok").sum())
        lines.append(f"- Spot-checked files: **{len(raw_meta)}**")
        lines.append(f"- Read failures: **{failures}**")
        ok_sf = pd.to_numeric(raw_meta.loc[raw_meta["reader_status"] == "ok", "sfreq"], errors="coerce").dropna()
        if not ok_sf.empty:
            lines.append(f"- Observed sampling rates: **{sorted(ok_sf.unique())}**")
    lines.extend(
        [
            "",
            "## Why this matters next",
            "",
            "1. The event structure must be finalized before condition-wise ERP or TFR results are interpreted.",
            "2. Electrode coordinate plausibility must be checked before trusting topographies.",
            "3. Only after those two issues are locked should preprocessing parameters and replication figures be treated as final.",
        ]
    )
    return "\n".join(lines)


def save_environment_snapshot(config: AuditConfig, reports_dir: Path) -> Path:
    env_df = environment_snapshot(config)
    lines = [f"{row.key}: {row.value}" for row in env_df.itertuples(index=False)]
    return write_text(reports_dir / "environment_snapshot.txt", "\n".join(lines))
