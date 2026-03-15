
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
class DecodingComparisonConfig:
    ours_root: Path
    authors_root: Path
    outputs_root: Path
    chance_level: float = 0.5

    @classmethod
    def from_yaml(cls, config_path: str | Path, project_root: str | Path | None = None) -> 'DecodingComparisonConfig':
        config_path = Path(config_path)
        payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        project_root = Path(project_root).resolve() if project_root else config_path.resolve().parents[1]
        return cls(
            ours_root=(project_root / payload['ours_root']).resolve(),
            authors_root=(project_root / payload['authors_root']).resolve(),
            outputs_root=(project_root / payload['outputs_root']).resolve(),
            chance_level=float(payload.get('chance_level', 0.5)),
        )


def ensure_dirs(root: Path) -> dict[str, Path]:
    tables = root / 'tables'
    figures = root / 'figures'
    reports = root / 'reports'
    for p in (root, tables, figures, reports):
        p.mkdir(parents=True, exist_ok=True)
    return {'root': root, 'tables': tables, 'figures': figures, 'reports': reports}


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _pipeline_tables(root: Path) -> dict[str, Path]:
    t = root / 'tables'
    return {
        'epoch_counts': t / 'epoch_count_summary.csv',
        'time_auc': t / 'subject_time_auc.csv',
        'window_auc': t / 'subject_window_auc.csv',
        'group_auc': t / 'group_auc_summary.csv',
        'stats': t / 'stats_results.csv',
        'figures': t / 'figure_manifest.csv',
        'summary': root / 'reports' / 'decoding_summary.json',
    }


def _summarize_pipeline(name: str, root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = _pipeline_tables(root)
    epoch_df = _safe_csv(paths['epoch_counts'])
    time_df = _safe_csv(paths['time_auc'])
    window_df = _safe_csv(paths['window_auc'])
    group_df = _safe_csv(paths['group_auc'])
    stats_df = _safe_csv(paths['stats'])

    summary = {
        'pipeline': name,
        'n_epoch_rows': int(len(epoch_df)),
        'n_subject_time_rows': int(len(time_df)),
        'n_subject_window_rows': int(len(window_df)),
        'n_group_rows': int(len(group_df)),
        'n_stats_rows': int(len(stats_df)),
        'n_subjects': int(window_df['subject'].nunique()) if 'subject' in window_df.columns else 0,
        'n_pairs': int(window_df['pair'].nunique()) if 'pair' in window_df.columns else 0,
        'n_windows': int(window_df['window'].nunique()) if 'window' in window_df.columns else 0,
    }
    if not stats_df.empty and 'mean_auc' in stats_df.columns:
        summary['mean_window_auc'] = float(pd.to_numeric(stats_df['mean_auc'], errors='coerce').mean())
    else:
        summary['mean_window_auc'] = np.nan

    for df in (window_df, group_df, stats_df):
        if not df.empty:
            df['pipeline'] = name
    return summary, window_df, group_df, stats_df


def _plot_group_overlay(group_long: pd.DataFrame, figures_dir: Path, chance_level: float) -> list[Path]:
    outputs: list[Path] = []
    if group_long.empty:
        return outputs
    for pair in sorted(group_long['pair'].dropna().unique()):
        sub = group_long[group_long['pair'].eq(pair)].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        for pipeline, sdf in sub.groupby('pipeline'):
            sdf = sdf.sort_values('time')
            ax.plot(sdf['time'], sdf['mean_auc'], label=f'{pipeline} mean AUC')
            if 'sem_auc' in sdf.columns:
                ax.fill_between(sdf['time'], sdf['mean_auc'] - sdf['sem_auc'], sdf['mean_auc'] + sdf['sem_auc'], alpha=0.2)
        ax.axhline(chance_level, color='k', lw=0.8, ls='--', label='Chance')
        ax.axvline(0.0, color='k', lw=0.8, ls=':')
        ax.set_title(f'Decoding comparison: {pair.replace("_vs_", " vs ")}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AUC')
        ax.set_ylim(0.35, 0.8)
        ax.legend(loc='best', fontsize=8)
        fig.tight_layout()
        out = figures_dir / f'{pair}_group_auc_overlay.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        outputs.append(out)
    return outputs


def _plot_window_boxplots(window_long: pd.DataFrame, figures_dir: Path, chance_level: float) -> list[Path]:
    outputs: list[Path] = []
    if window_long.empty:
        return outputs
    for pair in sorted(window_long['pair'].dropna().unique()):
        sub_pair = window_long[window_long['pair'].eq(pair)]
        for window in sorted(sub_pair['window'].dropna().unique()):
            sub = sub_pair[sub_pair['window'].eq(window)].copy()
            if sub.empty:
                continue
            pivot = sub.pivot_table(index='subject', columns='pipeline', values='mean_auc', aggfunc='mean')
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            cols = [c for c in ['authors', 'ours'] if c in pivot.columns] + [c for c in pivot.columns if c not in {'authors', 'ours'}]
            data = [pivot[c].dropna().to_numpy(dtype=float) for c in cols]
            if not data:
                plt.close(fig)
                continue
            ax.boxplot(data, labels=cols)
            ax.axhline(chance_level, color='k', lw=0.8, ls='--')
            ax.set_title(f'{pair.replace("_vs_", " vs ")} — {window}')
            ax.set_ylabel('Mean AUC')
            fig.tight_layout()
            out = figures_dir / f'{pair}_{window}_window_boxplot.png'
            fig.savefig(out, dpi=150)
            plt.close(fig)
            outputs.append(out)
    return outputs


def run_decoding_comparison(config: DecodingComparisonConfig) -> dict[str, Any]:
    dirs = ensure_dirs(config.outputs_root)
    summaries = []
    window_frames = []
    group_frames = []
    stats_frames = []

    for name, root in [('ours', config.ours_root), ('authors', config.authors_root)]:
        summary, window_df, group_df, stats_df = _summarize_pipeline(name, root)
        summaries.append(summary)
        if not window_df.empty:
            window_frames.append(window_df)
        if not group_df.empty:
            group_frames.append(group_df)
        if not stats_df.empty:
            stats_frames.append(stats_df)

    summary_df = pd.DataFrame(summaries)
    window_long = pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()
    group_long = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    stats_long = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()

    # Pairwise comparison of window AUC between pipelines
    if window_long.empty:
        window_compare = pd.DataFrame(columns=['pair', 'window', 'n_subjects', 'authors_mean_auc', 'ours_mean_auc', 'delta_ours_minus_authors'])
    else:
        pivot = window_long.pivot_table(index=['subject', 'pair', 'window'], columns='pipeline', values='mean_auc', aggfunc='mean').reset_index()
        rows = []
        for (pair, window), sub in pivot.groupby(['pair', 'window']):
            authors_vals = pd.to_numeric(sub.get('authors'), errors='coerce') if 'authors' in sub.columns else pd.Series(dtype=float)
            ours_vals = pd.to_numeric(sub.get('ours'), errors='coerce') if 'ours' in sub.columns else pd.Series(dtype=float)
            valid = np.isfinite(authors_vals.to_numpy(dtype=float, na_value=np.nan)) & np.isfinite(ours_vals.to_numpy(dtype=float, na_value=np.nan))
            n = int(valid.sum())
            authors_mean = float(np.nanmean(authors_vals.to_numpy(dtype=float))) if len(authors_vals) else np.nan
            ours_mean = float(np.nanmean(ours_vals.to_numpy(dtype=float))) if len(ours_vals) else np.nan
            delta = float(np.nanmean(ours_vals.to_numpy(dtype=float) - authors_vals.to_numpy(dtype=float))) if n else np.nan
            rows.append({
                'pair': pair,
                'window': window,
                'n_subjects': n,
                'authors_mean_auc': authors_mean,
                'ours_mean_auc': ours_mean,
                'delta_ours_minus_authors': delta,
            })
        window_compare = pd.DataFrame(rows)

    # Stats comparison side-by-side
    if stats_long.empty:
        stats_compare = pd.DataFrame(columns=['pair', 'window', 'authors_mean_auc', 'ours_mean_auc', 'authors_p_fdr', 'ours_p_fdr'])
    else:
        keep_cols = ['pair', 'window', 'pipeline', 'mean_auc', 'p_fdr', 'cohens_d', 'n_subjects']
        keep_cols = [c for c in keep_cols if c in stats_long.columns]
        pivot = stats_long[keep_cols].pivot_table(index=['pair', 'window'], columns='pipeline', values=[c for c in keep_cols if c not in {'pair', 'window', 'pipeline'}], aggfunc='first')
        pivot.columns = [f'{pipe}_{metric}' for metric, pipe in pivot.columns]
        stats_compare = pivot.reset_index()

    figure_paths = []
    figure_paths.extend(_plot_group_overlay(group_long, dirs['figures'], config.chance_level))
    figure_paths.extend(_plot_window_boxplots(window_long, dirs['figures'], config.chance_level))
    figure_manifest = pd.DataFrame({'figure_file': [str(p) for p in figure_paths]})

    save_dataframe(summary_df, dirs['tables'] / 'decoding_pipeline_summary.csv')
    save_dataframe(window_compare, dirs['tables'] / 'subject_window_auc_comparison.csv')
    save_dataframe(stats_compare, dirs['tables'] / 'stats_comparison.csv')
    save_dataframe(figure_manifest, dirs['tables'] / 'figure_manifest.csv')

    report_lines = [
        '# Decoding comparison',
        '',
        'This report compares the authors-inspired and ours decoding-analysis branches.',
        '',
        f'- Pipelines summarized: {len(summary_df)}',
        f'- Window-comparison rows: {len(window_compare)}',
        f'- Statistics-comparison rows: {len(stats_compare)}',
        f'- Figures written: {len(figure_paths)}',
    ]
    write_text(dirs['reports'] / 'decoding_comparison_report.md', '\n'.join(report_lines))

    summary = {
        'n_pipelines': int(len(summary_df)),
        'n_window_rows': int(len(window_compare)),
        'n_stats_rows': int(len(stats_compare)),
        'n_figures': int(len(figure_paths)),
        'tables': {
            'decoding_pipeline_summary': str((dirs['tables'] / 'decoding_pipeline_summary.csv').resolve()),
            'subject_window_auc_comparison': str((dirs['tables'] / 'subject_window_auc_comparison.csv').resolve()),
            'stats_comparison': str((dirs['tables'] / 'stats_comparison.csv').resolve()),
        },
    }
    summary_path = dirs['reports'] / 'decoding_comparison_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    return {
        'pipeline_summary': summary_df,
        'window_comparison': window_compare,
        'stats_comparison': stats_compare,
        'summary': summary,
        'summary_file': summary_path,
    }
