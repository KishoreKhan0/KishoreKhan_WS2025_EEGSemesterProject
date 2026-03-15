
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml
from mne.decoding import SlidingEstimator, cross_val_multiscore
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .audit_bids import save_dataframe, write_text
from .erp_replication import annotations_to_event_info, read_oddball_mapping

DEFAULT_CONDITIONS = ('standing', 'walking_alone', 'walking_together')
DEFAULT_WINDOWS = {
    'early': (0.0, 0.2),
    'p3_window': (0.25, 0.5),
    'late': (0.5, 0.8),
}


@dataclass
class DecodingConfig:
    pipeline_name: str
    preprocessing_manifest_path: Path
    event_mapping_root: Path
    outputs_root: Path
    condition_order: tuple[str, ...] = DEFAULT_CONDITIONS
    tmin: float = -0.2
    tmax: float = 0.8
    baseline_start: float | None = -0.2
    baseline_end: float | None = 0.0
    reject_by_annotation: bool = True
    detrend: int | None = None
    resample_sfreq: float = 100.0
    decim: int = 1
    min_epochs_per_condition: int = 40
    max_epochs_per_condition: int = 250
    cv_splits: int = 5
    test_size: float = 0.2
    random_seed: int = 42
    chance_level: float = 0.5
    stats_windows: dict[str, tuple[float, float]] = None

    @classmethod
    def from_yaml(cls, config_path: str | Path, project_root: str | Path | None = None) -> 'DecodingConfig':
        config_path = Path(config_path)
        payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        project_root = Path(project_root).resolve() if project_root else config_path.resolve().parents[1]
        raw_windows = payload.get('stats_windows', DEFAULT_WINDOWS)
        windows = {str(k): (float(v[0]), float(v[1])) for k, v in raw_windows.items()}
        return cls(
            pipeline_name=str(payload['pipeline_name']),
            preprocessing_manifest_path=(project_root / payload['preprocessing_manifest_path']).resolve(),
            event_mapping_root=(project_root / payload['event_mapping_root']).resolve(),
            outputs_root=(project_root / payload['outputs_root']).resolve(),
            condition_order=tuple(payload.get('condition_order', list(DEFAULT_CONDITIONS))),
            tmin=float(payload.get('tmin', -0.2)),
            tmax=float(payload.get('tmax', 0.8)),
            baseline_start=(float(payload['baseline_start']) if payload.get('baseline_start') not in (None, '', 'null') else None),
            baseline_end=(float(payload['baseline_end']) if payload.get('baseline_end') not in (None, '', 'null') else None),
            reject_by_annotation=bool(payload.get('reject_by_annotation', True)),
            detrend=(int(payload['detrend']) if payload.get('detrend') not in (None, '', 'null') else None),
            resample_sfreq=float(payload.get('resample_sfreq', 100.0)),
            decim=int(payload.get('decim', 1)),
            min_epochs_per_condition=int(payload.get('min_epochs_per_condition', 40)),
            max_epochs_per_condition=int(payload.get('max_epochs_per_condition', 250)),
            cv_splits=int(payload.get('cv_splits', 5)),
            test_size=float(payload.get('test_size', 0.2)),
            random_seed=int(payload.get('random_seed', 42)),
            chance_level=float(payload.get('chance_level', 0.5)),
            stats_windows=windows,
        )

    @property
    def baseline(self) -> tuple[float | None, float | None] | None:
        if self.baseline_start is None and self.baseline_end is None:
            return None
        return (self.baseline_start, self.baseline_end)


def ensure_decoding_dirs(config: DecodingConfig) -> dict[str, Path]:
    root = config.outputs_root
    tables = root / 'tables'
    figures = root / 'figures'
    reports = root / 'reports'
    for p in (root, tables, figures, reports):
        p.mkdir(parents=True, exist_ok=True)
    return {'root': root, 'tables': tables, 'figures': figures, 'reports': reports}


def read_preprocessing_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    for col in ['run_stem', 'output_fif']:
        if col not in df.columns:
            df[col] = np.nan
    return df


def select_runs(manifest: pd.DataFrame, subject: str | None = None, max_runs: int | None = None) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()
    df = manifest.copy()
    if subject:
        subj = subject if str(subject).startswith('sub-') else f'sub-{subject}'
        df = df[df['run_stem'].astype(str).str.contains(subj, regex=False)]
    df = df.sort_values('run_stem').reset_index(drop=True)
    if max_runs is not None:
        df = df.head(max_runs).copy()
    return df


def _subject_from_run_stem(run_stem: str) -> str:
    for part in str(run_stem).split('_'):
        if part.startswith('sub-'):
            return part
    return 'sub-unknown'


def _session_from_run_stem(run_stem: str) -> str:
    for part in str(run_stem).split('_'):
        if part.startswith('ses-'):
            return part
    return 'ses-unknown'


def _condition_from_label(label: str) -> str:
    return str(label).split('/')[0]


def _pairwise_conditions(condition_order: tuple[str, ...]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for ix, a in enumerate(condition_order):
        for b in condition_order[ix + 1:]:
            pairs.append((a, b))
    return pairs


def _load_run_epochs(raw_path: Path, mapping_df: pd.DataFrame, config: DecodingConfig) -> mne.Epochs | None:
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose='ERROR')
    raw.pick('eeg')
    if config.resample_sfreq and abs(raw.info['sfreq'] - config.resample_sfreq) > 1e-6:
        raw.resample(config.resample_sfreq, npad='auto', verbose='ERROR')
    events, event_id, _ = annotations_to_event_info(raw, mapping_df)
    if len(events) == 0 or not event_id:
        return None
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=config.baseline,
        preload=True,
        picks='eeg',
        reject_by_annotation=config.reject_by_annotation,
        detrend=config.detrend,
        decim=config.decim,
        event_repeated='drop',
        verbose='ERROR',
    )
    return epochs


def _collect_subject_condition_data(subject_manifest: pd.DataFrame, mapping_df: pd.DataFrame, config: DecodingConfig) -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray | None]:
    per_condition: dict[str, list[np.ndarray]] = {c: [] for c in config.condition_order}
    rows: list[dict[str, Any]] = []
    times = None
    for _, rec in subject_manifest.iterrows():
        raw_path = Path(str(rec['output_fif']))
        if not raw_path.exists():
            continue
        run_stem = str(rec['run_stem'])
        subject = _subject_from_run_stem(run_stem)
        session = _session_from_run_stem(run_stem)
        epochs = _load_run_epochs(raw_path, mapping_df, config)
        if epochs is None:
            rows.append({'subject': subject, 'session': session, 'run_stem': run_stem, 'condition': 'none', 'n_epochs': 0})
            continue
        times = epochs.times.copy()
        labels = list(epochs.event_id)
        for cond in config.condition_order:
            cond_labels = [lab for lab in labels if _condition_from_label(lab) == cond]
            if not cond_labels:
                rows.append({'subject': subject, 'session': session, 'run_stem': run_stem, 'condition': cond, 'n_epochs': 0})
                continue
            data_parts = [epochs[lab].get_data(copy=True) for lab in cond_labels if len(epochs[lab]) > 0]
            if not data_parts:
                rows.append({'subject': subject, 'session': session, 'run_stem': run_stem, 'condition': cond, 'n_epochs': 0})
                continue
            data = np.concatenate(data_parts, axis=0)
            per_condition[cond].append(data)
            rows.append({'subject': subject, 'session': session, 'run_stem': run_stem, 'condition': cond, 'n_epochs': int(data.shape[0])})
    out: dict[str, np.ndarray] = {}
    for cond, arrays in per_condition.items():
        if arrays:
            out[cond] = np.concatenate(arrays, axis=0)
        else:
            out[cond] = np.empty((0, 0, 0), dtype=float)
    return out, pd.DataFrame(rows), times


def _balance_pairwise(a: np.ndarray, b: np.ndarray, max_epochs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b), max_epochs)
    if n <= 0:
        return a[:0], b[:0]
    idx_a = rng.choice(len(a), size=n, replace=False)
    idx_b = rng.choice(len(b), size=n, replace=False)
    return a[idx_a], b[idx_b]


def _decode_pair(X: np.ndarray, y: np.ndarray, config: DecodingConfig) -> np.ndarray:
    cv = StratifiedShuffleSplit(
        n_splits=config.cv_splits,
        test_size=config.test_size,
        random_state=config.random_seed,
    )
    estimator = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    )
    time_decoder = SlidingEstimator(estimator, scoring='roc_auc', n_jobs=1, verbose=False)
    scores = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=1)
    return scores.mean(axis=0)


def _plot_group_curves(group_df: pd.DataFrame, times: np.ndarray, config: DecodingConfig, figures_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    if group_df.empty or times is None or len(times) == 0:
        return outputs
    for pair_name in sorted(group_df['pair'].dropna().unique()):
        sub = group_df[group_df['pair'].eq(pair_name)].sort_values('time')
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sub['time'], sub['mean_auc'], label='Mean AUC')
        ax.fill_between(sub['time'], sub['mean_auc'] - sub['sem_auc'], sub['mean_auc'] + sub['sem_auc'], alpha=0.25)
        ax.axhline(config.chance_level, color='k', lw=0.8, ls='--', label='Chance')
        ax.axvline(0.0, color='k', lw=0.8, ls=':')
        ax.set_title(f'{config.pipeline_name}: {pair_name.replace("_vs_", " vs ")}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AUC')
        ax.set_ylim(0.35, 0.75)
        ax.legend(loc='best', fontsize=8)
        fig.tight_layout()
        out = figures_dir / f'{pair_name}_group_auc.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        outputs.append(out)
    return outputs


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    if pvals.size == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = len(pvals)
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def run_decoding_analysis(config: DecodingConfig, subject: str | None = None, max_subjects: int | None = None) -> dict[str, Path | pd.DataFrame | Any]:
    dirs = ensure_decoding_dirs(config)
    manifest = select_runs(read_preprocessing_manifest(config.preprocessing_manifest_path), subject=subject)
    mapping_df = read_oddball_mapping(config.event_mapping_root)

    if manifest.empty:
        empty = pd.DataFrame()
        save_dataframe(empty, dirs['tables'] / 'epoch_count_summary.csv')
        save_dataframe(empty, dirs['tables'] / 'subject_time_auc.csv')
        save_dataframe(empty, dirs['tables'] / 'subject_window_auc.csv')
        save_dataframe(empty, dirs['tables'] / 'group_auc_summary.csv')
        save_dataframe(empty, dirs['tables'] / 'stats_results.csv')
        summary = {
            'pipeline_name': config.pipeline_name,
            'n_runs_input': 0,
            'n_subjects': 0,
            'n_subject_pair_rows': 0,
            'n_stats_rows': 0,
        }
        summary_path = dirs['reports'] / 'decoding_summary.json'
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        write_text(dirs['reports'] / 'decoding_report.md', '# Decoding analysis\n\nNo runs available.')
        return {'summary': summary, 'summary_file': summary_path}

    manifest['subject'] = manifest['run_stem'].astype(str).map(_subject_from_run_stem)
    subjects = sorted(manifest['subject'].dropna().unique())
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    epoch_rows = []
    time_rows = []
    window_rows = []
    times_ref = None
    rng = np.random.default_rng(config.random_seed)
    pairs = _pairwise_conditions(config.condition_order)

    for subj in subjects:
        subj_manifest = manifest[manifest['subject'].eq(subj)].copy()
        cond_data, subj_epoch_df, times = _collect_subject_condition_data(subj_manifest, mapping_df, config)
        if not subj_epoch_df.empty:
            epoch_rows.extend(subj_epoch_df.to_dict(orient='records'))
        if times_ref is None and times is not None:
            times_ref = times
        for cond_a, cond_b in pairs:
            A = cond_data.get(cond_a)
            B = cond_data.get(cond_b)
            if A is None or B is None or A.size == 0 or B.size == 0:
                continue
            if len(A) < config.min_epochs_per_condition or len(B) < config.min_epochs_per_condition:
                continue
            A_bal, B_bal = _balance_pairwise(A, B, config.max_epochs_per_condition, rng)
            if len(A_bal) < config.min_epochs_per_condition or len(B_bal) < config.min_epochs_per_condition:
                continue
            X = np.concatenate([A_bal, B_bal], axis=0)
            y = np.concatenate([np.zeros(len(A_bal), dtype=int), np.ones(len(B_bal), dtype=int)])
            auc = _decode_pair(X, y, config)
            pair_name = f'{cond_a}_vs_{cond_b}'
            for t, score in zip(times_ref, auc):
                time_rows.append({'subject': subj, 'pair': pair_name, 'time': float(t), 'auc': float(score)})
            for window_name, (t0, t1) in config.stats_windows.items():
                mask = (times_ref >= t0) & (times_ref <= t1)
                if np.any(mask):
                    window_rows.append({
                        'subject': subj,
                        'pair': pair_name,
                        'window': window_name,
                        'tmin': float(t0),
                        'tmax': float(t1),
                        'mean_auc': float(np.nanmean(auc[mask])),
                        'chance_level': float(config.chance_level),
                        'n_epochs_per_class': int(min(len(A_bal), len(B_bal))),
                    })

    epoch_df = pd.DataFrame(epoch_rows)
    time_df = pd.DataFrame(time_rows)
    window_df = pd.DataFrame(window_rows)

    if time_df.empty:
        group_df = pd.DataFrame(columns=['pair', 'time', 'mean_auc', 'sem_auc', 'n_subjects'])
    else:
        group_df = (
            time_df.groupby(['pair', 'time'], as_index=False)
            .agg(mean_auc=('auc', 'mean'), sem_auc=('auc', lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0), n_subjects=('subject', 'nunique'))
        )

    if window_df.empty:
        stats_df = pd.DataFrame(columns=['pair', 'window', 'tmin', 'tmax', 'n_subjects', 'mean_auc', 't_value', 'p_value', 'p_fdr', 'cohens_d'])
    else:
        stat_rows = []
        for (pair, window), sub in window_df.groupby(['pair', 'window']):
            vals = sub['mean_auc'].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            diff = vals - config.chance_level
            t_val, p_val = stats.ttest_1samp(vals, popmean=config.chance_level, nan_policy='omit')
            d = float(np.nanmean(diff) / np.nanstd(diff, ddof=1)) if len(diff) > 1 and np.nanstd(diff, ddof=1) > 0 else np.nan
            stat_rows.append({
                'pair': pair,
                'window': window,
                'tmin': float(sub['tmin'].iloc[0]),
                'tmax': float(sub['tmax'].iloc[0]),
                'n_subjects': int(sub['subject'].nunique()),
                'mean_auc': float(np.nanmean(vals)),
                't_value': float(t_val) if np.isfinite(t_val) else np.nan,
                'p_value': float(p_val) if np.isfinite(p_val) else np.nan,
                'cohens_d': d,
            })
        stats_df = pd.DataFrame(stat_rows)
        if not stats_df.empty:
            stats_df['p_fdr'] = _fdr_bh(stats_df['p_value'].fillna(1.0).to_numpy())

    figure_paths = _plot_group_curves(group_df, times_ref, config, dirs['figures'])
    figure_manifest = pd.DataFrame({'figure_file': [str(p) for p in figure_paths]})

    save_dataframe(epoch_df, dirs['tables'] / 'epoch_count_summary.csv')
    save_dataframe(time_df, dirs['tables'] / 'subject_time_auc.csv')
    save_dataframe(window_df, dirs['tables'] / 'subject_window_auc.csv')
    save_dataframe(group_df, dirs['tables'] / 'group_auc_summary.csv')
    save_dataframe(stats_df, dirs['tables'] / 'stats_results.csv')
    save_dataframe(figure_manifest, dirs['tables'] / 'figure_manifest.csv')

    summary = {
        'pipeline_name': config.pipeline_name,
        'n_runs_input': int(len(manifest)),
        'n_subjects': int(len(subjects)),
        'n_subject_pair_rows': int(window_df.shape[0]),
        'n_stats_rows': int(stats_df.shape[0]),
        'figure_count': int(len(figure_paths)),
        'summary_tables': {
            'epoch_count_summary': str((dirs['tables'] / 'epoch_count_summary.csv').resolve()),
            'subject_time_auc': str((dirs['tables'] / 'subject_time_auc.csv').resolve()),
            'subject_window_auc': str((dirs['tables'] / 'subject_window_auc.csv').resolve()),
            'group_auc_summary': str((dirs['tables'] / 'group_auc_summary.csv').resolve()),
            'stats_results': str((dirs['tables'] / 'stats_results.csv').resolve()),
        },
    }
    summary_path = dirs['reports'] / 'decoding_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    report_lines = [
        f'# Decoding analysis — {config.pipeline_name}',
        '',
        f'- Input runs: {summary["n_runs_input"]}',
        f'- Subjects: {summary["n_subjects"]}',
        f'- Subject/window rows: {summary["n_subject_pair_rows"]}',
        f'- Statistics rows: {summary["n_stats_rows"]}',
        f'- Figures: {summary["figure_count"]}',
        '',
        'Pairwise condition decoding was run using a time-resolved LDA with stratified shuffle-split cross-validation.',
        'Chance level is 0.5 AUC.',
    ]
    write_text(dirs['reports'] / 'decoding_report.md', '\n'.join(report_lines))

    return {
        'epoch_counts': epoch_df,
        'subject_time_auc': time_df,
        'subject_window_auc': window_df,
        'group_auc_summary': group_df,
        'stats_results': stats_df,
        'summary': summary,
        'summary_file': summary_path,
    }
