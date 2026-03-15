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

from .audit_bids import save_dataframe, write_text
from .preprocessing import build_run_stem


DEFAULT_CONDITION_ORDER = ('standing', 'walking_alone', 'walking_together')
DEFAULT_STIMULUS_ORDER = ('standard', 'deviant')
DEFAULT_TOPO_TIMES = (0.1, 0.2, 0.3, 0.4)
DEFAULT_ROI_CHANNELS = ('Cz', 'Pz')


@dataclass
class OddballERPConfig:
    pipeline_name: str
    preprocessing_manifest_path: Path
    event_mapping_root: Path
    outputs_root: Path
    oddball_task_filters: tuple[str, ...] = ('task2', 'oddball')
    condition_order: tuple[str, ...] = DEFAULT_CONDITION_ORDER
    stimulus_order: tuple[str, ...] = DEFAULT_STIMULUS_ORDER
    tmin: float = -0.2
    tmax: float = 0.8
    baseline_start: float | None = -0.2
    baseline_end: float | None = 0.0
    reject_by_annotation: bool = True
    detrend: int | None = None
    decim: int = 1
    min_epochs_per_label: int = 5
    equalize_within_condition: bool = False
    roi_channels: tuple[str, ...] = DEFAULT_ROI_CHANNELS
    peak_window_start: float = 0.25
    peak_window_end: float = 0.5
    topomap_times: tuple[float, ...] = DEFAULT_TOPO_TIMES

    @classmethod
    def from_yaml(cls, config_path: str | Path, project_root: str | Path | None = None) -> 'OddballERPConfig':
        config_path = Path(config_path)
        payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        project_root = Path(project_root).resolve() if project_root else config_path.resolve().parents[1]
        return cls(
            pipeline_name=str(payload['pipeline_name']),
            preprocessing_manifest_path=(project_root / payload['preprocessing_manifest_path']).resolve(),
            event_mapping_root=(project_root / payload['event_mapping_root']).resolve(),
            outputs_root=(project_root / payload['outputs_root']).resolve(),
            oddball_task_filters=tuple(payload.get('oddball_task_filters', ['task2', 'oddball'])),
            condition_order=tuple(payload.get('condition_order', list(DEFAULT_CONDITION_ORDER))),
            stimulus_order=tuple(payload.get('stimulus_order', list(DEFAULT_STIMULUS_ORDER))),
            tmin=float(payload.get('tmin', -0.2)),
            tmax=float(payload.get('tmax', 0.8)),
            baseline_start=(float(payload['baseline_start']) if payload.get('baseline_start') not in (None, '', 'null') else None),
            baseline_end=(float(payload['baseline_end']) if payload.get('baseline_end') not in (None, '', 'null') else None),
            reject_by_annotation=bool(payload.get('reject_by_annotation', True)),
            detrend=(int(payload['detrend']) if payload.get('detrend') not in (None, '', 'null') else None),
            decim=int(payload.get('decim', 1)),
            min_epochs_per_label=int(payload.get('min_epochs_per_label', 5)),
            equalize_within_condition=bool(payload.get('equalize_within_condition', False)),
            roi_channels=tuple(payload.get('roi_channels', list(DEFAULT_ROI_CHANNELS))),
            peak_window_start=float(payload.get('peak_window_start', 0.25)),
            peak_window_end=float(payload.get('peak_window_end', 0.5)),
            topomap_times=tuple(float(x) for x in payload.get('topomap_times', list(DEFAULT_TOPO_TIMES))),
        )

    @property
    def baseline(self) -> tuple[float | None, float | None] | None:
        if self.baseline_start is None and self.baseline_end is None:
            return None
        return (self.baseline_start, self.baseline_end)


def ensure_oddball_dirs(config: OddballERPConfig) -> dict[str, Path]:
    root = config.outputs_root
    epochs = root / 'epochs'
    evokeds = root / 'evokeds'
    figures = root / 'figures'
    tables = root / 'tables'
    reports = root / 'reports'
    for p in (root, epochs, evokeds, figures, tables, reports):
        p.mkdir(parents=True, exist_ok=True)
    return {
        'root': root,
        'epochs': epochs,
        'evokeds': evokeds,
        'figures': figures,
        'tables': tables,
        'reports': reports,
    }


def read_preprocessing_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ['input_eeg_file', 'output_fif', 'summary_json', 'run_stem']:
        if col not in df.columns:
            df[col] = np.nan
    return df


def select_oddball_runs(
    manifest: pd.DataFrame,
    subject: str | None = None,
    session: str | None = None,
    max_runs: int | None = None,
) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()

    df = manifest.copy()

    if subject:
        subj = subject if str(subject).startswith('sub-') else f'sub-{subject}'
        df = df[df['run_stem'].astype(str).str.contains(subj, regex=False)]

    if session:
        ses = session if str(session).startswith('ses-') else f'ses-{session}'
        df = df[df['run_stem'].astype(str).str.contains(ses, regex=False)]

    # Keep all preprocessed session files.
    # Oddball membership will be determined from annotations/events inside each file.
    df = df.sort_values('run_stem').reset_index(drop=True)

    if max_runs is not None:
        df = df.head(max_runs).copy()

    return df


def read_oddball_mapping(event_mapping_root: str | Path) -> pd.DataFrame:
    path = Path(event_mapping_root) / 'tables' / 'oddball_condition_dict.csv'
    if not path.exists():
        return pd.DataFrame(columns=['source_label', 'condition', 'stimulus_class'])
    df = pd.read_csv(path)
    for col in ['source_label', 'condition', 'stimulus_class']:
        if col not in df.columns:
            df[col] = ''
    df['source_label_norm'] = df['source_label'].astype(str).str.strip().str.lower()
    return df


def _fallback_condition(label: str) -> str | None:
    lowered = label.lower()
    if 'standing' in lowered:
        return 'standing'
    if 'walkinga' in lowered or 'walking alone' in lowered or 'walk alone' in lowered:
        return 'walking_alone'
    if 'walkingt' in lowered or 'walking together' in lowered or 'walk together' in lowered:
        return 'walking_together'
    return None


def _fallback_stimulus(label: str) -> str | None:
    lowered = label.lower()
    if 'deviant' in lowered:
        return 'deviant'
    if 'standard' in lowered:
        return 'standard'
    return None


def canonicalize_annotation(label: str, mapping_df: pd.DataFrame) -> str | None:
    label = str(label).strip()
    if not label:
        return None
    lowered = label.lower()
    if not mapping_df.empty:
        match = mapping_df[mapping_df['source_label_norm'].eq(lowered)]
        if not match.empty:
            row = match.iloc[0]
            condition = str(row.get('condition', '')).strip()
            stimulus = str(row.get('stimulus_class', '')).strip()
            if condition and condition != 'unknown' and stimulus in {'standard', 'deviant'}:
                return f'{condition}/{stimulus}'
    condition = _fallback_condition(label)
    stimulus = _fallback_stimulus(label)
    if condition and stimulus:
        return f'{condition}/{stimulus}'
    return None


def annotations_to_event_info(raw: mne.io.BaseRaw, mapping_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, int], dict[str, str]]:
    desc_to_canonical: dict[str, str] = {}
    canonical_to_id: dict[str, int] = {}
    for desc in map(str, raw.annotations.description):
        canonical = canonicalize_annotation(desc, mapping_df)
        if canonical is None:
            continue
        desc_to_canonical[desc] = canonical
        if canonical not in canonical_to_id:
            canonical_to_id[canonical] = len(canonical_to_id) + 1
    if not desc_to_canonical:
        return np.empty((0, 3), dtype=int), {}, {}

    def mapper(desc: str) -> int | None:
        canonical = desc_to_canonical.get(str(desc))
        if canonical is None:
            return None
        return canonical_to_id[canonical]

    events, _ = mne.events_from_annotations(raw, event_id=mapper, verbose='ERROR')
    return events, canonical_to_id, desc_to_canonical


def maybe_equalize_epochs(epochs: mne.Epochs, config: OddballERPConfig) -> mne.Epochs:
    if not config.equalize_within_condition:
        return epochs
    labels_by_condition: dict[str, list[str]] = {}
    for label in epochs.event_id:
        condition = label.split('/')[0]
        labels_by_condition.setdefault(condition, []).append(label)
    for labels in labels_by_condition.values():
        if len(labels) >= 2:
            try:
                epochs.equalize_event_counts(labels)
            except Exception:
                pass
    return epochs


def _plot_run_erps(epochs: mne.Epochs, save_path: Path, roi_channels: tuple[str, ...]) -> Path:
    fig, axes = plt.subplots(len(roi_channels), 1, figsize=(10, 3.5 * max(1, len(roi_channels))), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, ch in zip(axes, roi_channels):
        found = False
        for label in sorted(epochs.event_id):
            if len(epochs[label]) == 0:
                continue
            evoked = epochs[label].average()
            if ch in evoked.ch_names:
                found = True
                ax.plot(evoked.times, evoked.get_data(picks=[ch])[0] * 1e6, label=label)
        ax.axvline(0.0, color='k', lw=0.8, ls='--')
        ax.axhline(0.0, color='k', lw=0.5)
        ax.set_title(f'Run ERP at {ch}')
        ax.set_ylabel('µV')
        if found:
            ax.legend(loc='best', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'{ch} not available', ha='center', va='center', transform=ax.transAxes)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _save_evoked(evoked: mne.Evoked, path: Path) -> Path:
    mne.write_evokeds(path, evoked, overwrite=True)
    return path


def _extract_entities_from_manifest_row(row: dict[str, Any]) -> dict[str, str]:
    run_stem = str(row.get('run_stem', ''))
    parts = run_stem.split('_')
    out = {'subject': 'sub-unknown', 'session': 'ses-unknown', 'task': 'unknown', 'run': 'unknown'}
    for part in parts:
        if part.startswith('sub-'):
            out['subject'] = part
        elif part.startswith('ses-'):
            out['session'] = part
        elif part.startswith('task-'):
            out['task'] = part.replace('task-', '', 1)
        elif part.startswith('run-'):
            out['run'] = part.replace('run-', '', 1)
    return out


def process_oddball_run(row: pd.Series | dict[str, Any], config: OddballERPConfig,
                        mapping_df: pd.DataFrame, dirs: dict[str, Path]) -> list[dict[str, Any]]:
    row = dict(row)
    run_stem = str(row.get('run_stem') or build_run_stem(row))
    entities = _extract_entities_from_manifest_row(row)
    subject = entities['subject']
    session = entities['session']
    raw_path = Path(row['output_fif'])

    subject_epoch_dir = dirs['epochs'] / subject / session
    subject_evoked_dir = dirs['evokeds'] / subject / session
    subject_fig_dir = dirs['figures'] / subject / session
    subject_report_dir = dirs['reports'] / subject / session
    for p in (subject_epoch_dir, subject_evoked_dir, subject_fig_dir, subject_report_dir):
        p.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose='ERROR')
    events, event_id, desc_map = annotations_to_event_info(raw, mapping_df)
    if len(events) == 0 or not event_id:
        report = {
            'run_stem': run_stem,
            'status': 'no_mapped_events',
            'raw_file': str(raw_path),
            'available_annotation_labels': sorted(set(map(str, raw.annotations.description))),
        }
        report_path = subject_report_dir / f'{run_stem}_oddball_report.json'
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        return []

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    dup_samples = int(len(events) - len(np.unique(events[:, 0])))

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=config.baseline,
        preload=True,
        reject_by_annotation=config.reject_by_annotation,
        detrend=config.detrend,
        decim=max(1, int(config.decim)),
        picks=picks,
        event_repeated='drop',
        verbose='ERROR',
    )
    epochs = maybe_equalize_epochs(epochs, config)
    epochs_path = subject_epoch_dir / f'{run_stem}_oddball-epo.fif'
    epochs.save(epochs_path, overwrite=True)
    _plot_run_erps(epochs, subject_fig_dir / f'{run_stem}_roi_erps.png', config.roi_channels)

    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for label in sorted(epochs.event_id):
        condition, stimulus = label.split('/')
        n_epochs = len(epochs[label])
        event_rows.append({'label': label, 'n_epochs': int(n_epochs)})
        if n_epochs < config.min_epochs_per_label:
            continue
        evoked = epochs[label].average()
        evoked.comment = label
        evoked_path = subject_evoked_dir / f'{run_stem}_{condition}_{stimulus}-ave.fif'
        _save_evoked(evoked, evoked_path)
        rows.append({
            'pipeline_name': config.pipeline_name,
            'subject': subject,
            'session': session,
            'task': entities['task'],
            'run': entities['run'],
            'run_stem': run_stem,
            'condition': condition,
            'stimulus_class': stimulus,
            'label': label,
            'n_epochs': int(n_epochs),
            'epochs_file': str(epochs_path),
            'evoked_file': str(evoked_path),
            'raw_file': str(raw_path),
        })

    report = {
        'run_stem': run_stem,
        'status': 'ok',
        'raw_file': str(raw_path),
        'epochs_file': str(epochs_path),
        'event_id': event_id,
        'desc_to_canonical': desc_map,
        'label_epoch_counts': event_rows,
        'n_input_events': int(len(events)),
        'n_duplicate_sample_events_dropped': dup_samples,
        'baseline': config.baseline,
        'tmin': config.tmin,
        'tmax': config.tmax,
    }
    report_path = subject_report_dir / f'{run_stem}_oddball_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return rows


def _load_evoked(path: str | Path) -> mne.Evoked:
    return mne.read_evokeds(path, condition=0, verbose='ERROR')


def combine_subject_evokeds(manifest: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=['subject', 'session', 'condition', 'stimulus_class', 'label', 'n_runs', 'n_epochs_total', 'evoked_file'])
    rows: list[dict[str, Any]] = []
    for (subject, session, condition, stimulus), grp in manifest.groupby(['subject', 'session', 'condition', 'stimulus_class']):
        evokeds = [_load_evoked(path) for path in grp['evoked_file']]
        combined = mne.combine_evoked(evokeds, weights='nave')
        combined.comment = f'{condition}/{stimulus}'
        out_dir = dirs['evokeds'] / subject / session
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{subject}_{session}_{condition}_{stimulus}_subject-avg-ave.fif'
        _save_evoked(combined, out_path)
        rows.append({
            'subject': subject,
            'session': session,
            'condition': condition,
            'stimulus_class': stimulus,
            'label': f'{condition}/{stimulus}',
            'n_runs': int(len(grp)),
            'n_epochs_total': int(grp['n_epochs'].sum()),
            'evoked_file': str(out_path),
        })
    return pd.DataFrame(rows).sort_values(['subject', 'session', 'condition', 'stimulus_class']).reset_index(drop=True)


def combine_group_evokeds(subject_manifest: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    if subject_manifest.empty:
        return pd.DataFrame(columns=['session_group', 'condition', 'stimulus_class', 'label', 'n_subject_cells', 'evoked_file'])
    rows: list[dict[str, Any]] = []
    expanded = []
    for rec in subject_manifest.to_dict(orient='records'):
        expanded.append(rec)
        rec_all = dict(rec)
        rec_all['session'] = 'all_sessions'
        expanded.append(rec_all)
    expanded_df = pd.DataFrame(expanded)
    for (session, condition, stimulus), grp in expanded_df.groupby(['session', 'condition', 'stimulus_class']):
        evokeds = [_load_evoked(path) for path in grp['evoked_file']]
        grand = mne.grand_average(evokeds, interpolate_bads=False, drop_bads=False)
        grand.comment = f'{condition}/{stimulus}'
        out_dir = dirs['evokeds'] / 'group' / session
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'grand_{session}_{condition}_{stimulus}-ave.fif'
        _save_evoked(grand, out_path)
        rows.append({
            'session_group': session,
            'condition': condition,
            'stimulus_class': stimulus,
            'label': f'{condition}/{stimulus}',
            'n_subject_cells': int(len(grp)),
            'evoked_file': str(out_path),
        })
    return pd.DataFrame(rows).sort_values(['session_group', 'condition', 'stimulus_class']).reset_index(drop=True)


def make_difference_manifest(group_manifest: pd.DataFrame, dirs: dict[str, Path], condition_order: tuple[str, ...]) -> pd.DataFrame:
    if group_manifest.empty:
        return pd.DataFrame(columns=['session_group', 'condition', 'label', 'evoked_file'])
    rows: list[dict[str, Any]] = []
    for (session, condition), grp in group_manifest.groupby(['session_group', 'condition']):
        have = set(grp['stimulus_class'])
        if not {'standard', 'deviant'}.issubset(have):
            continue
        dev_path = grp.loc[grp['stimulus_class'].eq('deviant'), 'evoked_file'].iloc[0]
        std_path = grp.loc[grp['stimulus_class'].eq('standard'), 'evoked_file'].iloc[0]
        difference = mne.combine_evoked([_load_evoked(dev_path), _load_evoked(std_path)], weights=[1, -1])
        difference.comment = f'{condition}/difference_deviant_minus_standard'
        out_dir = dirs['evokeds'] / 'group' / session
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'difference_{session}_{condition}_deviant_minus_standard-ave.fif'
        _save_evoked(difference, out_path)
        rows.append({
            'session_group': session,
            'condition': condition,
            'label': difference.comment,
            'evoked_file': str(out_path),
        })
    return pd.DataFrame(rows).sort_values(['session_group', 'condition']).reset_index(drop=True)


def _plot_group_waveforms(group_manifest: pd.DataFrame, diff_manifest: pd.DataFrame, save_path: Path,
                          session_group: str, roi_channels: tuple[str, ...], condition_order: tuple[str, ...]) -> Path:
    fig, axes = plt.subplots(len(roi_channels), 2, figsize=(14, 4 * max(1, len(roi_channels))), sharex='col')
    if len(roi_channels) == 1:
        axes = np.array([axes])
    group_slice = group_manifest[group_manifest['session_group'].eq(session_group)]
    diff_slice = diff_manifest[diff_manifest['session_group'].eq(session_group)]
    for row_idx, ch in enumerate(roi_channels):
        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]
        for condition in condition_order:
            cond_grp = group_slice[group_slice['condition'].eq(condition)]
            for stimulus in ('standard', 'deviant'):
                stim_grp = cond_grp[cond_grp['stimulus_class'].eq(stimulus)]
                if stim_grp.empty:
                    continue
                evoked = _load_evoked(stim_grp['evoked_file'].iloc[0])
                if ch not in evoked.ch_names:
                    continue
                ax_left.plot(evoked.times, evoked.get_data(picks=[ch])[0] * 1e6, label=f'{condition}/{stimulus}')
            diff_grp = diff_slice[diff_slice['condition'].eq(condition)]
            if not diff_grp.empty:
                diff_evoked = _load_evoked(diff_grp['evoked_file'].iloc[0])
                if ch in diff_evoked.ch_names:
                    ax_right.plot(diff_evoked.times, diff_evoked.get_data(picks=[ch])[0] * 1e6, label=condition)
        for ax in (ax_left, ax_right):
            ax.axvline(0.0, color='k', lw=0.8, ls='--')
            ax.axhline(0.0, color='k', lw=0.5)
            ax.set_ylabel('µV')
        ax_left.set_title(f'{session_group}: ERP at {ch}')
        ax_right.set_title(f'{session_group}: Difference wave at {ch}')
        ax_left.legend(loc='best', fontsize=8)
        ax_right.legend(loc='best', fontsize=8)
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _plot_topomaps(diff_manifest: pd.DataFrame, save_path: Path, session_group: str,
                   topomap_times: tuple[float, ...], condition_order: tuple[str, ...]) -> Path:
    selected = diff_manifest[diff_manifest['session_group'].eq(session_group)]
    n_rows = max(1, len(condition_order))
    n_cols = max(1, len(topomap_times))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.8 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    for row_idx, condition in enumerate(condition_order):
        row = selected[selected['condition'].eq(condition)]
        if row.empty:
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, f'No data\n{condition}', ha='center', va='center')
                ax.axis('off')
            continue
        evoked = _load_evoked(row['evoked_file'].iloc[0])
        try:
            evoked.plot_topomap(times=list(topomap_times), axes=list(axes[row_idx]), show=False, colorbar=False, time_unit='s')
            axes[row_idx, 0].set_ylabel(condition)
        except Exception:
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, f'Topomap unavailable\n{condition}', ha='center', va='center')
                ax.axis('off')
    fig.suptitle(f'Difference-wave topomaps: {session_group}', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def compute_peak_metrics(diff_manifest: pd.DataFrame, config: OddballERPConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in diff_manifest.to_dict(orient='records'):
        evoked = _load_evoked(rec['evoked_file'])
        time_mask = (evoked.times >= config.peak_window_start) & (evoked.times <= config.peak_window_end)
        if not np.any(time_mask):
            continue
        for ch in config.roi_channels:
            if ch not in evoked.ch_names:
                continue
            data = evoked.get_data(picks=[ch])[0]
            window = data[time_mask]
            win_times = evoked.times[time_mask]
            peak_idx = int(np.argmax(window))
            rows.append({
                'session_group': rec['session_group'],
                'condition': rec['condition'],
                'channel': ch,
                'peak_window_start': config.peak_window_start,
                'peak_window_end': config.peak_window_end,
                'peak_amplitude_uv': float(window[peak_idx] * 1e6),
                'peak_latency_s': float(win_times[peak_idx]),
                'evoked_file': rec['evoked_file'],
            })
    return pd.DataFrame(rows)


def write_oddball_report(config: OddballERPConfig, run_manifest: pd.DataFrame,
                         subject_manifest: pd.DataFrame, group_manifest: pd.DataFrame,
                         diff_manifest: pd.DataFrame, peak_df: pd.DataFrame,
                         dirs: dict[str, Path]) -> Path:
    lines = [
        f'# Oddball ERP replication report: {config.pipeline_name}',
        '',
        f'- Preprocessing manifest: `{config.preprocessing_manifest_path}`',
        f'- Event mapping root: `{config.event_mapping_root}`',
        f'- Baseline: `{config.baseline}`',
        f'- Epoch window: `{config.tmin}` to `{config.tmax}` s',
        f'- ROI channels: `{list(config.roi_channels)}`',
        f'- Run-level evoked cells: {len(run_manifest)}',
        f'- Subject/session combined cells: {len(subject_manifest)}',
        f'- Group grand-average cells: {len(group_manifest)}',
        f'- Difference waves: {len(diff_manifest)}',
        '',
        '## Notes',
        '',
        '- The dataset contains a separate oddball task and a separate walking-synchronization task; this step analyzes the oddball branch only.',
        '- The baseline remains configurable because the milestone feedback questioned whether a neutral baseline always exists during continuous walking.',
        '- Difference waves are computed as deviant minus standard within each oddball condition.',
        '',
    ]
    if not peak_df.empty:
        lines.extend(['## Peak summary', ''])
        for rec in peak_df.sort_values(['session_group', 'condition', 'channel']).to_dict(orient='records'):
            lines.append(f"- {rec['session_group']} | {rec['condition']} | {rec['channel']}: {rec['peak_amplitude_uv']:.2f} µV at {rec['peak_latency_s']:.3f} s")
    return write_text(dirs['reports'] / 'oddball_replication_report.md', '\n'.join(lines))


def run_oddball_replication(config: OddballERPConfig, subject: str | None = None,
                            session: str | None = None, max_runs: int | None = None) -> dict[str, pd.DataFrame]:
    dirs = ensure_oddball_dirs(config)
    manifest = read_preprocessing_manifest(config.preprocessing_manifest_path)
    selected = select_oddball_runs(manifest, subject=subject, session=session, max_runs=max_runs)
    mapping_df = read_oddball_mapping(config.event_mapping_root)

    run_rows: list[dict[str, Any]] = []
    for rec in selected.to_dict(orient='records'):
        run_rows.extend(process_oddball_run(rec, config, mapping_df, dirs))
    run_manifest = pd.DataFrame(run_rows)
    if run_manifest.empty:
        empty = pd.DataFrame()
        save_dataframe(empty, dirs['tables'] / 'run_level_evoked_manifest.csv')

        diagnostic = {
            'status': 'no_run_level_evokeds_created',
            'n_preprocessing_rows': int(len(manifest)),
            'n_selected_rows': int(len(selected)),
            'selected_run_stems': selected['run_stem'].tolist() if not selected.empty else [],
            'mapping_columns': mapping_df.columns.tolist() if not mapping_df.empty else [],
            'n_mapping_rows': int(len(mapping_df)),
        }
        (dirs['reports'] / 'selection_diagnostic.json').write_text(
            json.dumps(diagnostic, indent=2),
            encoding='utf-8',
        )

        write_oddball_report(config, empty, empty, empty, empty, empty, dirs)
        return {
            'run_manifest': empty,
            'subject_manifest': empty,
            'group_manifest': empty,
            'difference_manifest': empty,
            'peak_metrics': empty,
        }
    subject_manifest = combine_subject_evokeds(run_manifest, dirs)
    group_manifest = combine_group_evokeds(subject_manifest, dirs)
    diff_manifest = make_difference_manifest(group_manifest, dirs, config.condition_order)
    peak_df = compute_peak_metrics(diff_manifest, config)

    save_dataframe(run_manifest, dirs['tables'] / 'run_level_evoked_manifest.csv')
    save_dataframe(subject_manifest, dirs['tables'] / 'subject_level_evoked_manifest.csv')
    save_dataframe(group_manifest, dirs['tables'] / 'group_level_evoked_manifest.csv')
    save_dataframe(diff_manifest, dirs['tables'] / 'difference_wave_manifest.csv')
    save_dataframe(peak_df, dirs['tables'] / 'peak_metrics.csv')

    epoch_counts = run_manifest.groupby(['subject', 'session', 'condition', 'stimulus_class'], as_index=False)['n_epochs'].sum()
    save_dataframe(epoch_counts, dirs['tables'] / 'epoch_count_summary.csv')

    for session_group in sorted(group_manifest['session_group'].unique()):
        _plot_group_waveforms(group_manifest, diff_manifest,
                              dirs['figures'] / f'grand_average_waveforms_{session_group}.png',
                              session_group, config.roi_channels, config.condition_order)
        if not diff_manifest.empty:
            _plot_topomaps(diff_manifest,
                           dirs['figures'] / f'difference_topomaps_{session_group}.png',
                           session_group, config.topomap_times, config.condition_order)

    write_oddball_report(config, run_manifest, subject_manifest, group_manifest, diff_manifest, peak_df, dirs)
    return {
        'run_manifest': run_manifest,
        'subject_manifest': subject_manifest,
        'group_manifest': group_manifest,
        'difference_manifest': diff_manifest,
        'peak_metrics': peak_df,
    }
