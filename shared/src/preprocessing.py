from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
import mne
from typing import Any

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import yaml
from mne.preprocessing import ICA

from .audit_bids import parse_bids_entities, safe_read_tsv, save_dataframe, write_text



DEFAULT_SCALE_CANDIDATES = (1.0, 0.1, 0.01, 10.0, 100.0, 0.001)
PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    'ours': {
        'highpass_hz': 0.1, # preserves ERP morphology; 1Hz high-pass distorts slow components
        'lowpass_hz': 40.0, # removes muscle artifact while retaining all ERP-relevant frequencies
        'notch_freqs_hz': [50.0, 100.0], # removes European mains noise and harmonic
        'resample_hz': None, # no resampling to avoid interpolation artifacts at preprocessing stage
        'average_reference': True,
        'ica_highpass_hz': 1.0,
        'ica_method_preference': 'picard', # faster and robust compared to extended infomax
        'ica_n_components': 0.99,
        'iclabel_threshold': 0.80, # more aggressive with conserving brain signals at the cost of more artifact
        'use_asr': False,
        'extended_infomax': False,
    },
    'authors': {
        'highpass_hz': 1.0,
        'lowpass_hz': 120.0,
        'notch_freqs_hz': [],
        'resample_hz': 250.0,
        'average_reference': True,
        'ica_highpass_hz': 1.0,
        'ica_method_preference': 'infomax',
        'ica_n_components': 0.99,
        'iclabel_threshold': 0.70,
        'use_asr': True,
        'extended_infomax': True,
    },
}


@dataclass
class PreprocessingConfig:
    pipeline_name: str
    profile_name: str
    dataset_root: Path
    outputs_root: Path
    run_inventory_path: Path
    event_mapping_root: Path | None = None
    random_state: int = 97
    max_runs: int | None = None
    raw_segment_start_sec: float = 60.0
    raw_segment_duration_sec: float = 15.0
    robust_std_z: float = 6.0
    apply_ransac_if_available: bool = True
    interpolate_bads: bool = True
    save_intermediate_ica_fit_raw: bool = False
    coordinate_scale_candidates: tuple[float, ...] = DEFAULT_SCALE_CANDIDATES
    highpass_hz: float | None = None
    lowpass_hz: float | None = None
    notch_freqs_hz: tuple[float, ...] = tuple()
    resample_hz: float | None = None
    average_reference: bool = True
    ica_highpass_hz: float | None = 1.0
    ica_method_preference: str = 'picard'
    ica_n_components: float | int | None = 0.99
    iclabel_threshold: float = 0.80
    use_asr: bool = False
    extended_infomax: bool = False
    n_jobs: int = 1

    @classmethod
    def from_yaml(cls, config_path: str | Path, project_root: str | Path | None = None) -> 'PreprocessingConfig':
        config_path = Path(config_path)
        payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        project_root = Path(project_root).resolve() if project_root else config_path.resolve().parents[1]
        profile_name = str(payload.get('profile_name', 'ours')).strip().lower()
        defaults = PROFILE_DEFAULTS.get(profile_name, PROFILE_DEFAULTS['ours']).copy()
        merged = {**defaults, **payload}
        return cls(
            pipeline_name=merged['pipeline_name'],
            profile_name=profile_name,
            dataset_root=(project_root / merged['dataset_root']).resolve(),
            outputs_root=(project_root / merged['outputs_root']).resolve(),
            run_inventory_path=(project_root / merged['run_inventory_path']).resolve(),
            event_mapping_root=((project_root / merged['event_mapping_root']).resolve() if merged.get('event_mapping_root') else None),
            random_state=int(merged.get('random_state', 97)),
            max_runs=(int(merged['max_runs']) if merged.get('max_runs') not in (None, '', 'null') else None),
            raw_segment_start_sec=float(merged.get('raw_segment_start_sec', 60.0)),
            raw_segment_duration_sec=float(merged.get('raw_segment_duration_sec', 15.0)),
            robust_std_z=float(merged.get('robust_std_z', 6.0)),
            apply_ransac_if_available=bool(merged.get('apply_ransac_if_available', True)),
            interpolate_bads=bool(merged.get('interpolate_bads', True)),
            save_intermediate_ica_fit_raw=bool(merged.get('save_intermediate_ica_fit_raw', False)),
            coordinate_scale_candidates=tuple(float(x) for x in merged.get('coordinate_scale_candidates', list(DEFAULT_SCALE_CANDIDATES))),
            highpass_hz=(float(merged['highpass_hz']) if merged.get('highpass_hz') not in (None, '') else None),
            lowpass_hz=(float(merged['lowpass_hz']) if merged.get('lowpass_hz') not in (None, '') else None),
            notch_freqs_hz=tuple(float(x) for x in merged.get('notch_freqs_hz', [])),
            resample_hz=(float(merged['resample_hz']) if merged.get('resample_hz') not in (None, '') else None),
            average_reference=bool(merged.get('average_reference', True)),
            ica_highpass_hz=(float(merged['ica_highpass_hz']) if merged.get('ica_highpass_hz') not in (None, '') else None),
            ica_method_preference=str(merged.get('ica_method_preference', 'picard')),
            ica_n_components=merged.get('ica_n_components', 0.99),
            iclabel_threshold=float(merged.get('iclabel_threshold', 0.80)),
            use_asr=bool(merged.get('use_asr', False)),
            extended_infomax=bool(merged.get('extended_infomax', False)),
            n_jobs=int(merged.get('n_jobs', 1)),
        )


def ensure_preprocessing_dirs(config: PreprocessingConfig) -> dict[str, Path]:
    root = config.outputs_root
    preprocessed = root / 'preprocessed'
    qc = root / 'qc'
    ica_dir = root / 'ica'
    reports = root / 'reports'
    tables = root / 'tables'
    logs = root / 'logs'
    for p in (root, preprocessed, qc, ica_dir, reports, tables, logs):
        p.mkdir(parents=True, exist_ok=True)
    return {
        'root': root,
        'preprocessed': preprocessed,
        'qc': qc,
        'ica': ica_dir,
        'reports': reports,
        'tables': tables,
        'logs': logs,
    }


def read_run_inventory(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def select_runs(inventory: pd.DataFrame, subject: str | None = None, session: str | None = None,
                task: str | None = None, run: str | None = None, max_runs: int | None = None) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()
    df = inventory.copy()
    if subject:
        subj = subject if str(subject).startswith('sub-') else f'sub-{subject}'
        df = df[df['subject'].astype(str) == subj]
    if session:
        ses = session if str(session).startswith('ses-') else f'ses-{session}'
        df = df[df['session'].astype(str) == ses]
    if task:
        df = df[df['task'].astype(str) == str(task)]
    if run:
        df = df[df['run'].astype(str) == str(run)]
    df = df.sort_values(['subject', 'session', 'task', 'run'], na_position='last').reset_index(drop=True)
    if max_runs is not None:
        df = df.head(max_runs).copy()
    return df


def build_run_stem(row: pd.Series | dict[str, Any]) -> str:
    row = dict(row)
    parts = [str(row.get('subject', '')).strip(), str(row.get('session', '')).strip(), f"task-{str(row.get('task', '')).strip()}"]
    run_val = str(row.get('run', '')).strip()
    if run_val and run_val.lower() != 'nan':
        parts.append(f'run-{run_val}')
    return '_'.join(p for p in parts if p and p != 'nan')


def read_raw_any(eeg_path: str | Path, preload: bool = False) -> mne.io.BaseRaw:
    eeg_path = Path(eeg_path)
    ext = eeg_path.suffix.lower()
    if ext == '.set':
        return mne.io.read_raw_eeglab(eeg_path, preload=preload, verbose='ERROR')
    if ext == '.vhdr':
        return mne.io.read_raw_brainvision(eeg_path, preload=preload, verbose='ERROR')
    if ext == '.edf':
        return mne.io.read_raw_edf(eeg_path, preload=preload, verbose='ERROR')
    if ext == '.bdf':
        return mne.io.read_raw_bdf(eeg_path, preload=preload, verbose='ERROR')
    raise ValueError(f'Unsupported EEG format: {eeg_path}')


def apply_channel_types_from_tsv(raw: mne.io.BaseRaw, channels_tsv: str | Path | None) -> dict[str, str]:
    if channels_tsv is None:
        return {}
    df = safe_read_tsv(channels_tsv)
    mapping: dict[str, str] = {}
    if df.empty or 'name' not in df.columns:
        dir_chs = {ch: 'misc' for ch in raw.ch_names if ch.lower() in {'x_dir', 'y_dir', 'z_dir'}}
        if dir_chs:
            raw.set_channel_types(dir_chs, on_unit_change='ignore')
        return dir_chs

    type_map = {
        'EEG': 'eeg', 'EOG': 'eog', 'ECG': 'ecg', 'EMG': 'emg', 'MISC': 'misc',
        'ACCELEROMETER': 'misc', 'TRIG': 'stim', 'STIM': 'stim', 'RESP': 'resp',
    }
    for rec in df.to_dict(orient='records'):
        name = str(rec.get('name', '')).strip()
        if not name or name not in raw.ch_names:
            continue
        kind = str(rec.get('type', '')).strip().upper()
        if name.lower() in {'x_dir', 'y_dir', 'z_dir'}:
            mapping[name] = 'misc'
        elif kind in type_map:
            mapping[name] = type_map[kind]
    if mapping:
        raw.set_channel_types(mapping, on_unit_change='ignore')
    return mapping


def _annotation_description_from_row(rec: dict[str, Any]) -> str:
    for key in ('trial_type', 'value', 'event_type', 'type', 'condition'):
        val = rec.get(key)
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
    return 'n/a'


def attach_annotations_from_events_tsv(raw: mne.io.BaseRaw, events_tsv: str | Path | None) -> int:
    if events_tsv is None:
        return len(raw.annotations)
    df = safe_read_tsv(events_tsv)
    if df.empty or 'onset' not in df.columns:
        return len(raw.annotations)
    onset = pd.to_numeric(df['onset'], errors='coerce').fillna(0.0).to_numpy(float)
    if 'duration' in df.columns:
        duration = pd.to_numeric(df['duration'], errors='coerce').fillna(0.0).to_numpy(float)
    else:
        duration = np.zeros_like(onset)
    desc = [ _annotation_description_from_row(rec) for rec in df.to_dict(orient='records') ]
    annotations = mne.Annotations(onset=onset, duration=duration, description=desc)
    raw.set_annotations(annotations)
    return len(raw.annotations)


def choose_coordinate_scale(norms: np.ndarray, candidates: tuple[float, ...] = DEFAULT_SCALE_CANDIDATES,
                            target_radius_m: float = 0.095) -> tuple[float, float]:
    norms = np.asarray(norms, dtype=float)
    norms = norms[np.isfinite(norms)]
    if norms.size == 0:
        return 1.0, float('inf')
    best_scale = 1.0
    best_score = float('inf')
    for scale in candidates:
        scaled = norms * scale
        med = float(np.nanmedian(scaled))
        score = abs(med - target_radius_m)
        if score < best_score:
            best_score = score
            best_scale = float(scale)
    return best_scale, best_score


def build_montage_from_sidecars(electrodes_tsv: str | Path | None, coordsystem_json: str | Path | None,
                                candidates: tuple[float, ...] = DEFAULT_SCALE_CANDIDATES) -> tuple[mne.channels.DigMontage | None, dict[str, Any]]:
    electrodes = safe_read_tsv(electrodes_tsv) if electrodes_tsv else pd.DataFrame()
    report: dict[str, Any] = {
        'electrodes_tsv': str(electrodes_tsv) if electrodes_tsv else None,
        'coordsystem_json': str(coordsystem_json) if coordsystem_json else None,
        'n_electrodes': 0,
        'chosen_scale_to_meters': None,
        'median_radius_before_scale': None,
        'median_radius_after_scale': None,
        'coordinate_system': None,
        'units_reported': None,
    }
    if electrodes.empty or 'name' not in electrodes.columns:
        return None, report
    if not {'x', 'y', 'z'}.issubset(electrodes.columns):
        return None, report
    coords = electrodes[['x', 'y', 'z']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    names = electrodes['name'].astype(str).tolist()
    mask = np.isfinite(coords).all(axis=1)
    coords = coords[mask]
    names = [name for name, keep in zip(names, mask) if keep]
    report['n_electrodes'] = int(len(names))
    if len(names) == 0:
        return None, report
    norms = np.linalg.norm(coords, axis=1)
    report['median_radius_before_scale'] = float(np.median(norms))
    scale, _ = choose_coordinate_scale(norms, candidates=candidates)
    coords_m = coords * scale
    report['chosen_scale_to_meters'] = float(scale)
    report['median_radius_after_scale'] = float(np.median(np.linalg.norm(coords_m, axis=1)))

    coordsys = {}
    if coordsystem_json and Path(coordsystem_json).exists():
        try:
            coordsys = json.loads(Path(coordsystem_json).read_text(encoding='utf-8'))
        except Exception:
            coordsys = {}
    report['coordinate_system'] = coordsys.get('EEGCoordinateSystem') or coordsys.get('iEEGCoordinateSystem')
    report['units_reported'] = coordsys.get('EEGCoordinateUnits') or coordsys.get('iEEGCoordinateUnits')

    ch_pos = {name: coords_m[idx] for idx, name in enumerate(names)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    return montage, report


def apply_montage_from_sidecars(raw: mne.io.BaseRaw, electrodes_tsv: str | Path | None,
                                coordsystem_json: str | Path | None,
                                candidates: tuple[float, ...] = DEFAULT_SCALE_CANDIDATES) -> dict[str, Any]:
    montage, report = build_montage_from_sidecars(electrodes_tsv, coordsystem_json, candidates=candidates)
    if montage is None:
        report['montage_applied'] = False
        return report
    try:
        raw.set_montage(montage, on_missing='ignore', match_case=False)
        report['montage_applied'] = True
    except Exception as exc:
        report['montage_applied'] = False
        report['montage_error'] = str(exc)
    return report


def robust_std_bad_channels(raw: mne.io.BaseRaw, z_thresh: float = 6.0) -> list[str]:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return []
    data = raw.get_data(picks=picks)
    ch_std = np.std(data, axis=1)
    med = np.median(ch_std)
    mad = np.median(np.abs(ch_std - med)) + 1e-12
    rz = 0.6745 * (ch_std - med) / mad
    bad_idx = np.where(np.abs(rz) > z_thresh)[0]
    return [raw.ch_names[picks[i]] for i in bad_idx]


def ransac_bad_channels(raw: mne.io.BaseRaw) -> list[str]:
    try:
        from autoreject import Ransac
    except Exception:
        return []
    try:
        events = mne.make_fixed_length_events(raw, duration=2.0, overlap=0.0)
        epochs = mne.Epochs(raw, events, tmin=0.0, tmax=2.0, baseline=None,
                            preload=True, reject_by_annotation=True, verbose='ERROR')
        ransac = Ransac(random_state=97, verbose=False)
        ransac.fit(epochs)
        return list(getattr(ransac, 'bad_chs_', []))
    except Exception:
        return []


def _has_valid_dig(raw: mne.io.BaseRaw) -> bool:
    if raw.get_montage() is None:
        return False
    pos = raw.get_montage().get_positions().get('ch_pos', {})
    return len(pos) > 0


def interpolate_bad_channels(raw: mne.io.BaseRaw) -> list[str]:
    if not raw.info.get('bads'):
        return []
    if not _has_valid_dig(raw):
        return []
    bads_before = list(raw.info['bads'])
    try:
        raw.interpolate_bads(reset_bads=False, method=dict(eeg='spline'), verbose='ERROR')
    except TypeError:
        raw.interpolate_bads(reset_bads=False, verbose='ERROR')
    return bads_before


def maybe_apply_asr(raw: mne.io.BaseRaw, config: PreprocessingConfig) -> tuple[mne.io.BaseRaw, str | None]:
    if not config.use_asr:
        return raw, None
    try:
        import asrpy  # noqa: F401
    except Exception:
        return raw, 'ASR requested by profile but asrpy is not installed; skipped.'
    return raw, 'ASR support detected but not enabled in this scaffold yet; skipped to keep the pipeline stable.'


def choose_ica_method(config: PreprocessingConfig) -> tuple[str, dict[str, Any]]:
    preference = str(config.ica_method_preference).lower()
    fit_params: dict[str, Any] = {}
    if preference == 'picard':
        try:
            import picard  # noqa: F401
            return 'picard', fit_params
        except Exception:
            return 'fastica', fit_params
    if preference == 'infomax':
        if config.extended_infomax:
            fit_params = {'extended': True}
        return 'infomax', fit_params
    return preference, fit_params


def fit_ica(raw: mne.io.BaseRaw, config: PreprocessingConfig) -> tuple[ICA, dict[str, Any]]:
    raw_ica = raw.copy()
    if config.ica_highpass_hz is not None:
        raw_ica.filter(l_freq=config.ica_highpass_hz, h_freq=None, phase='zero', n_jobs=config.n_jobs, verbose='ERROR')
    method, fit_params = choose_ica_method(config)
    ica = ICA(
        n_components=config.ica_n_components,
        method=method,
        fit_params=fit_params,
        random_state=config.random_state,
        max_iter='auto',
    )
    ica.fit(raw_ica, picks='eeg', verbose='ERROR')
    summary = {
        'method': method,
        'fit_params': fit_params,
        'n_components_': int(getattr(ica, 'n_components_', 0) or 0),
        'ica_highpass_hz': config.ica_highpass_hz,
    }
    return ica, summary


def auto_exclude_ica_components(raw: mne.io.BaseRaw, ica: ICA, config: PreprocessingConfig) -> tuple[list[int], dict[str, Any]]:
    summary: dict[str, Any] = {
        'strategy': 'none',
        'excluded_components': [],
        'details': None,
    }
    try:
        from mne_icalabel import label_components
    except Exception:
        return [], summary
    try:
        labels = label_components(raw, ica, method='iclabel')
        excluded = []
        class_names = list(labels['classes'])
        brain_idx = class_names.index('brain') if 'brain' in class_names else None
        for comp_idx, pred_label in enumerate(labels['labels']):
            if brain_idx is None:
                continue
            brain_prob = float(labels['y_pred_proba'][comp_idx, brain_idx])
            if brain_prob < float(config.iclabel_threshold):
                excluded.append(int(comp_idx))
        summary.update({
            'strategy': 'iclabel',
            'excluded_components': excluded,
            'details': {
                'threshold': float(config.iclabel_threshold),
                'labels': list(labels['labels']),
                'classes': class_names,
            },
        })
        return excluded, summary
    except Exception as exc:
        summary['details'] = str(exc)
        return [], summary


def plot_segment(raw: mne.io.BaseRaw, save_path: str | Path, start_sec: float = 60.0, duration_sec: float = 15.0,
                 n_channels: int = 25) -> Path:
    save_path = Path(save_path)
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    picks = picks[:n_channels]
    if len(picks) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No EEG channels available', ha='center', va='center')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    sfreq = float(raw.info['sfreq'])
    start = max(0, int(round(start_sec * sfreq)))
    stop = min(raw.n_times, int(round((start_sec + duration_sec) * sfreq)))
    if stop <= start:
        start = 0
        stop = min(raw.n_times, int(round(duration_sec * sfreq)))
    data = raw.get_data(picks=picks, start=start, stop=stop) * 1e6
    times = np.arange(data.shape[1]) / sfreq + (start / sfreq)
    centered = data - np.median(data, axis=1, keepdims=True)
    spread = np.percentile(np.abs(centered), 95)
    spread = float(spread) if np.isfinite(spread) and spread > 0 else 20.0
    offsets = np.arange(len(picks))[::-1] * spread * 3.0
    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, ch_idx in enumerate(picks):
        ax.plot(times, centered[idx] + offsets[idx], lw=0.6)
    ax.set_yticks(offsets)
    ax.set_yticklabels([raw.ch_names[p] for p in picks])
    ax.set_xlabel('Time (s)')
    ax.set_title('Continuous EEG segment (µV offsets)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_psd_summary(raw: mne.io.BaseRaw, save_path: str | Path, fmin: float = 1.0, fmax: float = 60.0) -> Path:
    save_path = Path(save_path)
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(picks) == 0:
        ax.text(0.5, 0.5, 'No EEG channels available', ha='center', va='center')
        ax.axis('off')
    else:
        psd = raw.compute_psd(picks=picks, fmin=fmin, fmax=fmax, verbose='ERROR')
        data, freqs = psd.get_data(return_freqs=True)
        db = 10 * np.log10(np.maximum(data, np.finfo(float).tiny))
        ax.plot(freqs, np.median(db, axis=0), lw=1.5, label='Median')
        ax.fill_between(freqs, np.percentile(db, 25, axis=0), np.percentile(db, 75, axis=0), alpha=0.3)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB)')
        ax.set_title('EEG PSD summary')
        ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_montage_topdown(raw: mne.io.BaseRaw, save_path: str | Path) -> Path:
    save_path = Path(save_path)
    fig, ax = plt.subplots(figsize=(5, 5))
    montage = raw.get_montage()
    if montage is None:
        ax.text(0.5, 0.5, 'No montage available', ha='center', va='center')
        ax.axis('off')
    else:
        pos = montage.get_positions().get('ch_pos', {})
        xs, ys, labels = [], [], []
        for ch in raw.ch_names:
            xyz = pos.get(ch)
            if xyz is None:
                continue
            xs.append(float(xyz[0]))
            ys.append(float(xyz[1]))
            labels.append(ch)
        if xs:
            ax.scatter(xs, ys, s=20)
            for x, y, label in zip(xs, ys, labels):
                ax.text(x, y, label, fontsize=6)
            ax.set_title('Top-down sensor layout')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.axis('equal')
        else:
            ax.text(0.5, 0.5, 'No channel positions available', ha='center', va='center')
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _collect_annotation_counts(raw: mne.io.BaseRaw) -> dict[str, int]:
    counts: dict[str, int] = {}
    for desc in raw.annotations.description:
        key = str(desc)
        counts[key] = counts.get(key, 0) + 1
    return counts


def process_run(row: pd.Series | dict[str, Any], config: PreprocessingConfig, dirs: dict[str, Path]) -> dict[str, Any]:
    row = dict(row)
    stem = build_run_stem(row)
    subject = str(row.get('subject', 'sub-unknown'))
    session = str(row.get('session', 'ses-unknown'))
    eeg_path = Path(row['eeg_file'])
    out_preproc = dirs['preprocessed'] / subject / session
    out_qc = dirs['qc'] / subject / session
    out_ica = dirs['ica'] / subject / session
    out_reports = dirs['reports'] / subject / session
    for p in (out_preproc, out_qc, out_ica, out_reports):
        p.mkdir(parents=True, exist_ok=True)

    raw = read_raw_any(eeg_path, preload=False)
    raw.load_data()
    type_mapping = apply_channel_types_from_tsv(raw, row.get('channels_tsv'))
    n_annotations = attach_annotations_from_events_tsv(raw, row.get('events_tsv'))
    montage_report = apply_montage_from_sidecars(raw, row.get('electrodes_tsv'), row.get('coordsystem_json'),
                                                 candidates=config.coordinate_scale_candidates)

    qc_raw_before = plot_segment(raw, out_qc / f'{stem}_raw_segment_before.png',
                                 start_sec=config.raw_segment_start_sec,
                                 duration_sec=config.raw_segment_duration_sec)
    qc_montage = plot_montage_topdown(raw, out_qc / f'{stem}_montage.png')

    raw_proc = raw.copy()
    asr_note = None
    raw_proc, asr_note = maybe_apply_asr(raw_proc, config)
    if config.highpass_hz is not None or config.lowpass_hz is not None:
        raw_proc.filter(l_freq=config.highpass_hz, h_freq=config.lowpass_hz, phase='zero',
                        n_jobs=config.n_jobs, verbose='ERROR')
    if config.notch_freqs_hz:
        raw_proc.notch_filter(freqs=list(config.notch_freqs_hz), n_jobs=config.n_jobs, verbose='ERROR')
    if config.resample_hz is not None and float(raw_proc.info['sfreq']) != float(config.resample_hz):
        raw_proc.resample(config.resample_hz, npad='auto', verbose='ERROR')

    bads_robust = robust_std_bad_channels(raw_proc, z_thresh=config.robust_std_z)
    bads_ransac = ransac_bad_channels(raw_proc) if config.apply_ransac_if_available else []
    bads_union = sorted(set(bads_robust) | set(bads_ransac))
    raw_proc.info['bads'] = bads_union
    interpolated = interpolate_bad_channels(raw_proc) if (config.interpolate_bads and bads_union) else []
    if config.average_reference:
        raw_proc.set_eeg_reference('average', projection=False, verbose='ERROR')

    ica, ica_summary = fit_ica(raw_proc, config)
    excluded, label_summary = auto_exclude_ica_components(raw_proc, ica, config)
    ica.exclude = excluded
    raw_clean = raw_proc.copy()
    ica.apply(raw_clean)

    fif_path = out_preproc / f'{stem}_clean_raw.fif'
    raw_clean.save(fif_path, overwrite=True)
    ica_path = out_ica / f'{stem}_ica.fif'
    ica.save(ica_path, overwrite=True)

    try:
        fig = ica.plot_components(show=False)
        if isinstance(fig, list):
            for idx, one in enumerate(fig, start=1):
                one.savefig(out_ica / f'{stem}_ica_components_{idx:02d}.png', dpi=150, bbox_inches='tight')
                plt.close(one)
        else:
            fig.savefig(out_ica / f'{stem}_ica_components.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
    except Exception:
        pass

    qc_raw_after = plot_segment(raw_clean, out_qc / f'{stem}_raw_segment_after.png',
                                start_sec=config.raw_segment_start_sec,
                                duration_sec=config.raw_segment_duration_sec)
    qc_psd = plot_psd_summary(raw_clean, out_qc / f'{stem}_psd.png')

    summary = {
        'pipeline_name': config.pipeline_name,
        'profile_name': config.profile_name,
        'run_stem': stem,
        'input_eeg_file': str(eeg_path),
        'output_fif': str(fif_path),
        'output_ica': str(ica_path),
        'channel_type_mapping': type_mapping,
        'n_annotations': int(n_annotations),
        'annotation_counts': _collect_annotation_counts(raw_clean),
        'sampling_rate_hz_before': float(raw.info['sfreq']),
        'sampling_rate_hz_after': float(raw_clean.info['sfreq']),
        'n_channels_total': int(len(raw_clean.ch_names)),
        'n_eeg_channels': int(sum(t == 'eeg' for t in raw_clean.get_channel_types())),
        'n_misc_channels': int(sum(t == 'misc' for t in raw_clean.get_channel_types())),
        'bads_robust': bads_robust,
        'bads_ransac': bads_ransac,
        'bads_union': bads_union,
        'interpolated_channels': interpolated,
        'montage_report': montage_report,
        'filter': {
            'highpass_hz': config.highpass_hz,
            'lowpass_hz': config.lowpass_hz,
            'notch_freqs_hz': list(config.notch_freqs_hz),
            'resample_hz': config.resample_hz,
        },
        'average_reference': bool(config.average_reference),
        'asr_note': asr_note,
        'ica_summary': ica_summary,
        'ica_auto_label_summary': label_summary,
        'qc_files': {
            'raw_segment_before': str(qc_raw_before),
            'raw_segment_after': str(qc_raw_after),
            'psd': str(qc_psd),
            'montage': str(qc_montage),
        },
        'platform': platform.platform(),
        'python_version': sys.version.replace('\n', ' '),
        'mne_version': mne.__version__,
    }
    summary_path = out_reports / f'{stem}_preprocessing_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    summary['summary_json'] = str(summary_path)
    return summary


def run_preprocessing(config: PreprocessingConfig, subject: str | None = None, session: str | None = None,
                      task: str | None = None, run: str | None = None, max_runs: int | None = None) -> pd.DataFrame:
    dirs = ensure_preprocessing_dirs(config)
    inventory = read_run_inventory(config.run_inventory_path)
    selected = select_runs(inventory, subject=subject, session=session, task=task, run=run,
                           max_runs=max_runs if max_runs is not None else config.max_runs)
    rows = []
    for rec in selected.to_dict(orient='records'):
        rows.append(process_run(rec, config, dirs))
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        save_dataframe(manifest, dirs['tables'] / 'preprocessing_manifest.csv')
        write_text(dirs['reports'] / 'preprocessing_overview.md',
                   '\n'.join([
                       f'# Preprocessing overview: {config.pipeline_name}',
                       '',
                       f'- Profile: `{config.profile_name}`',
                       f'- Processed runs: {len(manifest)}',
                       f'- Dataset root: `{config.dataset_root}`',
                   ]))
    return manifest
