# 18 subjects × 2 sessions = 36 expected runs; used to validate pipeline output completeness

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import yaml


@dataclass
class ProjectValidationConfig:
    project_root: Path
    outputs_root: Path
    ours_root: Path
    authors_root: Path
    expected_subjects: int = 18
    expected_sessions: int = 2
    expected_oddball_conditions: int = 3
    expected_oddball_stimulus_classes: int = 2
    expected_sync_conditions: int = 3
    expected_sync_bands: int = 2
    expected_decoding_pairs: int = 3
    expected_decoding_windows: int = 3

    @classmethod
    def from_yaml(cls, path: str | Path, project_root: str | Path | None = None) -> 'ProjectValidationConfig':
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        if project_root is None:
            project_root = data.get('project_root', path.parents[1])
        project_root = Path(project_root).resolve()
        def _p(key: str, default: str) -> Path:
            return (project_root / data.get(key, default)).resolve()
        return cls(
            project_root=project_root,
            outputs_root=_p('outputs_root', 'shared/outputs/project_validation'),
            ours_root=_p('ours_root', 'ours_pipeline/outputs'),
            authors_root=_p('authors_root', 'authors_pipeline/outputs'),
            expected_subjects=int(data.get('expected_subjects', 18)),
            expected_sessions=int(data.get('expected_sessions', 2)),
            expected_oddball_conditions=int(data.get('expected_oddball_conditions', 3)),
            expected_oddball_stimulus_classes=int(data.get('expected_oddball_stimulus_classes', 2)),
            expected_sync_conditions=int(data.get('expected_sync_conditions', 3)),
            expected_sync_bands=int(data.get('expected_sync_bands', 2)),
            expected_decoding_pairs=int(data.get('expected_decoding_pairs', 3)),
            expected_decoding_windows=int(data.get('expected_decoding_windows', 3)),
        )


def _ensure_dirs(cfg: ProjectValidationConfig) -> dict[str, Path]:
    tables = cfg.outputs_root / 'tables'
    reports = cfg.outputs_root / 'reports'
    figures = cfg.outputs_root / 'figures'
    for p in (tables, reports, figures):
        p.mkdir(parents=True, exist_ok=True)
    return {'tables': tables, 'reports': reports, 'figures': figures}


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _row(component: str, pipeline: str, metric: str, value: Any, expected: Any, status: str, note: str = '') -> dict[str, Any]:
    return {
        'component': component,
        'pipeline': pipeline,
        'metric': metric,
        'value': value,
        'expected': expected,
        'status': status,
        'note': note,
    }


def _check_equal(value: Any, expected: Any) -> str:
    return 'pass' if value == expected else 'warn'


def _check_nonempty(value: int) -> str:
    return 'pass' if int(value) > 0 else 'fail'


def _pipeline_paths(root: Path) -> dict[str, Path]:
    return {
        'preproc_manifest': root / 'tables' / 'preprocessing_manifest.csv',
        'oddball_run': root / 'oddball_replication' / 'tables' / 'run_level_evoked_manifest.csv',
        'oddball_subject': root / 'oddball_replication' / 'tables' / 'subject_level_evoked_manifest.csv',
        'oddball_group': root / 'oddball_replication' / 'tables' / 'group_level_evoked_manifest.csv',
        'oddball_diff': root / 'oddball_replication' / 'tables' / 'difference_wave_manifest.csv',
        'oddball_peaks': root / 'oddball_replication' / 'tables' / 'peak_metrics.csv',
        'sync_stride': root / 'sync_tfr_analysis' / 'tables' / 'stride_manifest.csv',
        'sync_run': root / 'sync_tfr_analysis' / 'tables' / 'run_level_tfr_manifest.csv',
        'sync_subject_pooled': root / 'sync_tfr_analysis' / 'tables' / 'subject_pooled_bandpower.csv',
        'sync_stats': root / 'sync_tfr_analysis' / 'tables' / 'stats_results.csv',
        'dec_epoch': root / 'decoding_analysis' / 'tables' / 'epoch_count_summary.csv',
        'dec_subject_time': root / 'decoding_analysis' / 'tables' / 'subject_time_auc.csv',
        'dec_subject_window': root / 'decoding_analysis' / 'tables' / 'subject_window_auc.csv',
        'dec_group': root / 'decoding_analysis' / 'tables' / 'group_auc_summary.csv',
        'dec_stats': root / 'decoding_analysis' / 'tables' / 'stats_results.csv',
        'dec_summary': root / 'decoding_analysis' / 'reports' / 'decoding_summary.json',
        'sync_summary': root / 'sync_tfr_analysis' / 'reports' / 'sync_tfr_summary.json',
    }


def _validate_pipeline(name: str, root: Path, cfg: ProjectValidationConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    p = _pipeline_paths(root)
    rows: list[dict[str, Any]] = []

    preproc = _safe_csv(p['preproc_manifest'])
    odd_run = _safe_csv(p['oddball_run'])
    odd_subject = _safe_csv(p['oddball_subject'])
    odd_group = _safe_csv(p['oddball_group'])
    odd_diff = _safe_csv(p['oddball_diff'])
    odd_peaks = _safe_csv(p['oddball_peaks'])
    sync_stride = _safe_csv(p['sync_stride'])
    sync_run = _safe_csv(p['sync_run'])
    sync_subject = _safe_csv(p['sync_subject_pooled'])
    sync_stats = _safe_csv(p['sync_stats'])
    dec_epoch = _safe_csv(p['dec_epoch'])
    dec_subject_time = _safe_csv(p['dec_subject_time'])
    dec_subject_window = _safe_csv(p['dec_subject_window'])
    dec_group = _safe_csv(p['dec_group'])
    dec_stats = _safe_csv(p['dec_stats'])
    dec_summary = _safe_json(p['dec_summary'])
    sync_summary = _safe_json(p['sync_summary'])

    expected_runs = cfg.expected_subjects * cfg.expected_sessions
    expected_odd_run = expected_runs * cfg.expected_oddball_conditions * cfg.expected_oddball_stimulus_classes
    expected_odd_group = cfg.expected_sessions * cfg.expected_oddball_conditions * cfg.expected_oddball_stimulus_classes
    expected_odd_diff = cfg.expected_sessions * cfg.expected_oddball_conditions
    expected_sync_subject = cfg.expected_subjects * cfg.expected_sync_conditions * cfg.expected_sync_bands
    expected_sync_stats = 3 * cfg.expected_sync_bands
    expected_dec_subject_window = cfg.expected_subjects * cfg.expected_decoding_pairs * cfg.expected_decoding_windows
    expected_dec_stats = cfg.expected_decoding_pairs * cfg.expected_decoding_windows

    rows += [
        _row('preprocessing', name, 'n_runs', len(preproc), expected_runs, _check_equal(len(preproc), expected_runs)),
        _row('oddball', name, 'run_level_cells', len(odd_run), expected_odd_run, _check_equal(len(odd_run), expected_odd_run)),
        _row('oddball', name, 'subject_session_cells', len(odd_subject), expected_odd_run, _check_equal(len(odd_subject), expected_odd_run)),
        _row('oddball', name, 'group_average_cells', len(odd_group), expected_odd_group, _check_equal(len(odd_group), expected_odd_group)),
        _row('oddball', name, 'difference_waves', len(odd_diff), expected_odd_diff, _check_equal(len(odd_diff), expected_odd_diff)),
        _row('oddball', name, 'peak_metric_rows', len(odd_peaks), '>0', _check_nonempty(len(odd_peaks))),
        _row('sync_tfr', name, 'stride_rows', len(sync_stride), '>0', _check_nonempty(len(sync_stride))),
        _row('sync_tfr', name, 'run_level_cells', len(sync_run), expected_runs * cfg.expected_sync_conditions, _check_equal(len(sync_run), expected_runs * cfg.expected_sync_conditions)),
        _row('sync_tfr', name, 'subject_pooled_rows', len(sync_subject), expected_sync_subject, _check_equal(len(sync_subject), expected_sync_subject)),
        _row('sync_tfr', name, 'stats_rows', len(sync_stats), expected_sync_stats, _check_equal(len(sync_stats), expected_sync_stats)),
        _row('decoding', name, 'epoch_rows', len(dec_epoch), expected_runs * cfg.expected_oddball_conditions, _check_equal(len(dec_epoch), expected_runs * cfg.expected_oddball_conditions)),
        _row('decoding', name, 'subject_time_rows', len(dec_subject_time), '>0', _check_nonempty(len(dec_subject_time))),
        _row('decoding', name, 'subject_window_rows', len(dec_subject_window), expected_dec_subject_window, _check_equal(len(dec_subject_window), expected_dec_subject_window)),
        _row('decoding', name, 'group_rows', len(dec_group), '>0', _check_nonempty(len(dec_group))),
        _row('decoding', name, 'stats_rows', len(dec_stats), expected_dec_stats, _check_equal(len(dec_stats), expected_dec_stats)),
    ]

    mean_auc = dec_summary.get('mean_window_auc', np.nan)
    auc_status = 'pass' if np.isfinite(mean_auc) else 'warn'
    auc_note = 'Near chance is scientifically acceptable for an exploratory extension; this checks only that a value exists.'
    rows.append(_row('decoding', name, 'mean_window_auc', mean_auc, 'finite', auc_status, auc_note))

    summary = {
        'pipeline': name,
        'n_preprocessing_runs': int(len(preproc)),
        'n_oddball_run_cells': int(len(odd_run)),
        'n_sync_stride_rows': int(len(sync_stride)),
        'n_decoding_subject_window_rows': int(len(dec_subject_window)),
        'mean_window_auc': mean_auc,
        'sync_summary_file_present': bool(sync_summary),
        'decoding_summary_file_present': bool(dec_summary),
    }
    return pd.DataFrame(rows), summary


def run_project_validation(cfg: ProjectValidationConfig) -> dict[str, pd.DataFrame]:
    dirs = _ensure_dirs(cfg)

    ours_df, ours_summary = _validate_pipeline('ours', cfg.ours_root, cfg)
    authors_df, authors_summary = _validate_pipeline('authors', cfg.authors_root, cfg)
    validation = pd.concat([ours_df, authors_df], ignore_index=True)
    summary_df = pd.DataFrame([ours_summary, authors_summary])

    pass_count = int((validation['status'] == 'pass').sum())
    warn_count = int((validation['status'] == 'warn').sum())
    fail_count = int((validation['status'] == 'fail').sum())

    validation.to_csv(dirs['tables'] / 'project_validation_checks.csv', index=False)
    summary_df.to_csv(dirs['tables'] / 'project_validation_summary.csv', index=False)

    lines = [
        '# Project validation report',
        '',
        f'- Pass checks: {pass_count}',
        f'- Warn checks: {warn_count}',
        f'- Fail checks: {fail_count}',
        '',
        '## Pipeline summaries',
        '',
    ]
    for rec in [ours_summary, authors_summary]:
        lines.extend([
            f"### {rec['pipeline']}",
            '',
            f"- Preprocessing runs: {rec['n_preprocessing_runs']}",
            f"- Oddball run-level cells: {rec['n_oddball_run_cells']}",
            f"- Sync stride rows: {rec['n_sync_stride_rows']}",
            f"- Decoding subject-window rows: {rec['n_decoding_subject_window_rows']}",
            f"- Mean decoding window AUC: {rec['mean_window_auc']}",
            '',
        ])
    lines.extend(['## Notes', '', '- This step is a freeze/QA pass before report writing.', '- Decoding AUC is checked only for existence here; chance-level results are still valid negative findings.', ''])
    (dirs['reports'] / 'project_validation_report.md').write_text('\n'.join(lines), encoding='utf-8')

    summary = {
        'pass_checks': pass_count,
        'warn_checks': warn_count,
        'fail_checks': fail_count,
        'summary_file': str((dirs['tables'] / 'project_validation_summary.csv').resolve()),
        'checks_file': str((dirs['tables'] / 'project_validation_checks.csv').resolve()),
        'report_file': str((dirs['reports'] / 'project_validation_report.md').resolve()),
    }
    (dirs['reports'] / 'project_validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    return {'validation_checks': validation, 'validation_summary': summary_df}
