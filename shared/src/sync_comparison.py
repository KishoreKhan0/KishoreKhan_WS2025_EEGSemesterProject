from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .audit_bids import save_dataframe, write_text


@dataclass
class SyncComparisonConfig:
    authors_root: Path
    ours_root: Path
    outputs_root: Path
    authors_label: str = 'authors'
    ours_label: str = 'ours'
    roi_order: tuple[str, ...] = ('alpha_mu', 'beta')
    condition_order: tuple[str, ...] = ('natural', 'blocked', 'sync')
    report_title: str = 'Walking synchronization analysis comparison'

    @classmethod
    def from_yaml(cls, config_path: str | Path, project_root: str | Path | None = None) -> 'SyncComparisonConfig':
        config_path = Path(config_path)
        payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        project_root = Path(project_root).resolve() if project_root else config_path.resolve().parents[1]
        return cls(
            authors_root=(project_root / payload['authors_root']).resolve(),
            ours_root=(project_root / payload['ours_root']).resolve(),
            outputs_root=(project_root / payload['outputs_root']).resolve(),
            authors_label=str(payload.get('authors_label', 'authors')),
            ours_label=str(payload.get('ours_label', 'ours')),
            roi_order=tuple(str(x) for x in payload.get('roi_order', ['alpha_mu', 'beta'])),
            condition_order=tuple(str(x) for x in payload.get('condition_order', ['natural', 'blocked', 'sync'])),
            report_title=str(payload.get('report_title', 'Walking synchronization analysis comparison')),
        )


def ensure_dirs(config: SyncComparisonConfig) -> dict[str, Path]:
    root = config.outputs_root
    paths = {
        'root': root,
        'tables': root / 'tables',
        'figures': root / 'figures',
        'reports': root / 'reports',
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.stat().st_size == 0:
            return pd.DataFrame()
    except OSError:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _pipeline_paths(root: Path) -> dict[str, Path]:
    return {
        'stride_manifest': root / 'tables' / 'stride_manifest.csv',
        'run_level_tfr_manifest': root / 'tables' / 'run_level_tfr_manifest.csv',
        'run_level_bandpower': root / 'tables' / 'run_level_bandpower.csv',
        'subject_session_bandpower': root / 'tables' / 'subject_session_bandpower.csv',
        'subject_pooled_bandpower': root / 'tables' / 'subject_pooled_bandpower.csv',
        'stats_results': root / 'tables' / 'stats_results.csv',
        'cycle_curve_long': root / 'tables' / 'cycle_curve_long.csv',
        'figure_manifest': root / 'tables' / 'figure_manifest.csv',
        'summary_json': root / 'reports' / 'sync_tfr_summary.json',
        'report_md': root / 'reports' / 'sync_tfr_report.md',
    }


def summarize_pipeline(label: str, root: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    paths = _pipeline_paths(root)
    tables = {name: _safe_csv(path) for name, path in paths.items() if path.suffix == '.csv'}
    summary_json = _safe_json(paths['summary_json'])

    stride_df = tables['stride_manifest']
    run_df = tables['run_level_tfr_manifest']
    subject_session_df = tables['subject_session_bandpower']
    subject_pooled_df = tables['subject_pooled_bandpower']
    stats_df = tables['stats_results']
    fig_df = tables['figure_manifest']

    stride_duration = stride_df['duration_sec'].to_numpy(dtype=float) if 'duration_sec' in stride_df.columns else np.array([])
    toe_percent = stride_df['toe_percent_actual'].to_numpy(dtype=float) if 'toe_percent_actual' in stride_df.columns else np.array([])
    n_strides = run_df['n_strides'].to_numpy(dtype=float) if 'n_strides' in run_df.columns else np.array([])

    row = {
        'pipeline': label,
        'n_stride_rows': int(len(stride_df)),
        'n_run_level_cells': int(len(run_df)),
        'n_subject_session_rows': int(len(subject_session_df)),
        'n_subject_pooled_rows': int(len(subject_pooled_df)),
        'n_stats_rows': int(len(stats_df)),
        'n_figures_listed': int(len(fig_df)),
        'mean_stride_duration_sec': float(np.nanmean(stride_duration)) if stride_duration.size else np.nan,
        'mean_toe_percent_actual': float(np.nanmean(toe_percent)) if toe_percent.size else np.nan,
        'mean_strides_per_run_cell': float(np.nanmean(n_strides)) if n_strides.size else np.nan,
        'summary_json_n_runs_input': summary_json.get('n_runs_input', np.nan),
    }
    return pd.DataFrame([row]), tables, summary_json


def _ordered_cat(series: pd.Series, order: tuple[str, ...]) -> pd.Series:
    return pd.Categorical(series.astype(str), categories=list(order), ordered=True)


def compare_subject_bandpower(config: SyncComparisonConfig, authors_tables: dict[str, pd.DataFrame], ours_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    a = authors_tables['subject_pooled_bandpower'].copy()
    o = ours_tables['subject_pooled_bandpower'].copy()
    if a.empty or o.empty:
        return pd.DataFrame()
    keep = ['subject', 'condition', 'roi_band', 'mean_power']
    a = a[keep].rename(columns={'mean_power': 'authors_mean_power'})
    o = o[keep].rename(columns={'mean_power': 'ours_mean_power'})
    merged = a.merge(o, on=['subject', 'condition', 'roi_band'], how='outer')
    merged['power_delta_ours_minus_authors'] = merged['ours_mean_power'] - merged['authors_mean_power']
    merged['condition'] = _ordered_cat(merged['condition'], config.condition_order)
    merged['roi_band'] = _ordered_cat(merged['roi_band'], config.roi_order)
    return merged.sort_values(['roi_band', 'condition', 'subject']).reset_index(drop=True)


def compare_stats(config: SyncComparisonConfig, authors_tables: dict[str, pd.DataFrame], ours_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    a = authors_tables['stats_results'].copy()
    o = ours_tables['stats_results'].copy()
    if a.empty and o.empty:
        return pd.DataFrame()
    if not a.empty:
        a = a.rename(columns={c: f'authors_{c}' for c in a.columns if c not in {'roi_band', 'comparison'}})
    if not o.empty:
        o = o.rename(columns={c: f'ours_{c}' for c in o.columns if c not in {'roi_band', 'comparison'}})
    if a.empty:
        merged = o.copy()
    elif o.empty:
        merged = a.copy()
    else:
        merged = a.merge(o, on=['roi_band', 'comparison'], how='outer')
    if 'roi_band' in merged.columns:
        merged['roi_band'] = _ordered_cat(merged['roi_band'], config.roi_order)
    return merged.sort_values(['roi_band', 'comparison']).reset_index(drop=True)


def _group_curve(df: pd.DataFrame, pipeline_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['pipeline', 'condition', 'roi_band', 'cycle_percent', 'value'])
    out = (
        df.groupby(['condition', 'roi_band', 'cycle_percent'], as_index=False)
        .agg(value=('value', 'mean'))
    )
    out['pipeline'] = pipeline_label
    return out


def plot_group_curve_overlays(config: SyncComparisonConfig, authors_tables: dict[str, pd.DataFrame], ours_tables: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    a = _group_curve(authors_tables['cycle_curve_long'], config.authors_label)
    o = _group_curve(ours_tables['cycle_curve_long'], config.ours_label)
    long_df = pd.concat([a, o], ignore_index=True)
    if long_df.empty:
        return pd.DataFrame()
    rows = []
    for roi_band in config.roi_order:
        roi_df = long_df[long_df['roi_band'].astype(str).eq(roi_band)]
        if roi_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for condition in config.condition_order:
            for pipeline_label in (config.authors_label, config.ours_label):
                sub = roi_df[(roi_df['condition'].astype(str) == condition) & (roi_df['pipeline'].astype(str) == pipeline_label)]
                if sub.empty:
                    continue
                ax.plot(sub['cycle_percent'], sub['value'], label=f'{pipeline_label} | {condition}')
        ax.set_xlabel('Stride cycle (%)')
        ax.set_ylabel('Band power')
        ax.set_title(f'{roi_band} group cycle curves')
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        out_path = out_dir / f'group_curve_overlay_{roi_band}.png'
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        rows.append({'figure_type': 'group_curve_overlay', 'roi_band': roi_band, 'path': str(out_path)})
    return pd.DataFrame(rows)


def plot_delta_boxplots(config: SyncComparisonConfig, subject_compare_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if subject_compare_df.empty:
        return pd.DataFrame()
    rows = []
    for roi_band in config.roi_order:
        roi_df = subject_compare_df[subject_compare_df['roi_band'].astype(str).eq(roi_band)].copy()
        if roi_df.empty:
            continue
        roi_df['condition'] = roi_df['condition'].astype(str)
        fig, ax = plt.subplots(figsize=(8, 4))
        data = []
        labels = []
        for condition in config.condition_order:
            vals = roi_df.loc[roi_df['condition'].eq(condition), 'power_delta_ours_minus_authors'].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            data.append(vals)
            labels.append(condition)
        if not data:
            plt.close(fig)
            continue
        ax.boxplot(data, labels=labels)
        ax.axhline(0.0, linestyle='--', linewidth=1)
        ax.set_ylabel('Ours - Authors mean power')
        ax.set_title(f'{roi_band} subject deltas')
        fig.tight_layout()
        out_path = out_dir / f'subject_delta_boxplot_{roi_band}.png'
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        rows.append({'figure_type': 'subject_delta_boxplot', 'roi_band': roi_band, 'path': str(out_path)})
    return pd.DataFrame(rows)


def run_sync_comparison(config: SyncComparisonConfig) -> dict[str, Any]:
    dirs = ensure_dirs(config)
    authors_summary_df, authors_tables, authors_json = summarize_pipeline(config.authors_label, config.authors_root)
    ours_summary_df, ours_tables, ours_json = summarize_pipeline(config.ours_label, config.ours_root)

    summary_df = pd.concat([ours_summary_df, authors_summary_df], ignore_index=True)
    subject_compare_df = compare_subject_bandpower(config, authors_tables, ours_tables)
    stats_compare_df = compare_stats(config, authors_tables, ours_tables)

    figure_manifest_df = pd.concat([
        plot_group_curve_overlays(config, authors_tables, ours_tables, dirs['figures']),
        plot_delta_boxplots(config, subject_compare_df, dirs['figures']),
    ], ignore_index=True)

    report_lines = [
        config.report_title,
        '=' * len(config.report_title),
        '',
        'Pipeline summaries',
        '------------------',
        summary_df.to_string(index=False),
        '',
        'Authors JSON summary',
        '--------------------',
        json.dumps(authors_json, indent=2),
        '',
        'Ours JSON summary',
        '-----------------',
        json.dumps(ours_json, indent=2),
    ]
    if not stats_compare_df.empty:
        report_lines.extend(['', 'Statistics comparison', '---------------------', stats_compare_df.to_string(index=False)])
    if not subject_compare_df.empty:
        brief = (
            subject_compare_df.groupby(['roi_band', 'condition'], as_index=False)
            .agg(mean_delta=('power_delta_ours_minus_authors', 'mean'))
        )
        report_lines.extend(['', 'Mean subject-level deltas (ours - authors)', '----------------------------------------', brief.to_string(index=False)])

    write_text(dirs['reports'] / 'sync_comparison_report.md', '\n'.join(report_lines))

    summary_json = {
        'authors': authors_json,
        'ours': ours_json,
        'summary_rows': int(len(summary_df)),
        'subject_comparison_rows': int(len(subject_compare_df)),
        'stats_comparison_rows': int(len(stats_compare_df)),
        'figure_manifest_rows': int(len(figure_manifest_df)),
    }
    (dirs['reports'] / 'sync_comparison_summary.json').write_text(json.dumps(summary_json, indent=2), encoding='utf-8')

    save_dataframe(summary_df, dirs['tables'] / 'sync_pipeline_summary.csv')
    save_dataframe(subject_compare_df, dirs['tables'] / 'subject_bandpower_comparison.csv')
    save_dataframe(stats_compare_df, dirs['tables'] / 'stats_comparison.csv')
    save_dataframe(figure_manifest_df, dirs['tables'] / 'figure_manifest.csv')

    return {
        'summary': summary_df,
        'subject_bandpower_comparison': subject_compare_df,
        'stats_comparison': stats_compare_df,
        'figure_manifest': figure_manifest_df,
        'summary_json': summary_json,
    }
