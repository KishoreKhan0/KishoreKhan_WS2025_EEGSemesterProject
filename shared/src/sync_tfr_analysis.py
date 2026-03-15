from __future__ import annotations
import gc

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .audit_bids import save_dataframe, write_text

DEFAULT_CONDITION_ORDER = ('natural', 'blocked', 'sync')
DEFAULT_ALPHA_CHANNELS = ('C4', 'C6', 'CP4', 'CP6')
DEFAULT_BETA_CHANNELS = ('C1', 'Cz', 'C2')
DEFAULT_COMPARISONS = (('natural', 'blocked'), ('natural', 'sync'), ('blocked', 'sync'))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, '', 'null'):
        return default
    return float(value)


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, '', 'null'):
        return default
    return int(value)


@dataclass
class SyncTFRConfig:
    pipeline_name: str
    preprocessing_manifest_path: Path
    event_mapping_root: Path
    outputs_root: Path
    analysis_variant: str = 'ours'
    condition_order: tuple[str, ...] = DEFAULT_CONDITION_ORDER
    segment_pre_sec: float = 0.2
    baseline_start: float | None = -0.2
    baseline_end: float | None = 0.0
    baseline_mode: str = 'logratio'
    all_cycle_fallback_baseline: bool = True
    min_stride_sec: float = 0.6
    max_stride_sec: float = 2.0
    cycle_points: int = 200
    warp_mode: str = 'linear_cycle'
    fixed_toe_percent: float = 0.68
    freqs: tuple[float, ...] = tuple(np.linspace(3.0, 35.0, 17))
    n_cycles: tuple[float, ...] = tuple(np.linspace(4.0, 10.0, 17))
    alpha_band: tuple[float, float] = (7.5, 12.5)
    beta_band: tuple[float, float] = (16.0, 32.0)
    alpha_channels: tuple[str, ...] = DEFAULT_ALPHA_CHANNELS
    beta_channels: tuple[str, ...] = DEFAULT_BETA_CHANNELS
    participant_role_aliases: tuple[str, ...] = ('participant', 'par')
    experimenter_role_aliases: tuple[str, ...] = ('experimenter', 'exp')
    hs_aliases: tuple[str, ...] = ('hs', 'heel_strike', 'heelstrike', 'rhs')
    to_aliases: tuple[str, ...] = ('to', 'toe_off', 'toeoff', 'rto')
    stats_method: str = 'paired_t'
    fdr_alpha: float = 0.05
    bootstrap_iterations: int = 1000
    random_state: int = 97
    max_runs: int | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path, project_root: str | Path | None = None) -> 'SyncTFRConfig':
        config_path = Path(config_path)
        payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        project_root = Path(project_root).resolve() if project_root else config_path.resolve().parents[1]
        freqs_payload = payload.get('freqs')
        if freqs_payload:
            freqs = tuple(float(x) for x in freqs_payload)
        else:
            fmin = float(payload.get('freq_min_hz', 3.0))
            fmax = float(payload.get('freq_max_hz', 35.0))
            n_freqs = int(payload.get('n_freqs', 17))
            spacing = str(payload.get('freq_spacing', 'linear')).lower()
            if spacing == 'log':
                freqs = tuple(float(x) for x in np.geomspace(fmin, fmax, n_freqs))
            else:
                freqs = tuple(float(x) for x in np.linspace(fmin, fmax, n_freqs))
        n_cycles_payload = payload.get('n_cycles')
        if n_cycles_payload:
            n_cycles = tuple(float(x) for x in n_cycles_payload)
        else:
            nc_start = float(payload.get('n_cycles_start', 4.0))
            nc_end = float(payload.get('n_cycles_end', 10.0))
            n_cycles = tuple(float(x) for x in np.linspace(nc_start, nc_end, len(freqs)))
        return cls(
            pipeline_name=str(payload['pipeline_name']),
            preprocessing_manifest_path=(project_root / payload['preprocessing_manifest_path']).resolve(),
            event_mapping_root=(project_root / payload['event_mapping_root']).resolve(),
            outputs_root=(project_root / payload['outputs_root']).resolve(),
            analysis_variant=str(payload.get('analysis_variant', 'ours')),
            condition_order=tuple(str(x) for x in payload.get('condition_order', list(DEFAULT_CONDITION_ORDER))),
            segment_pre_sec=float(payload.get('segment_pre_sec', 0.2)),
            baseline_start=_safe_float(payload.get('baseline_start'), -0.2),
            baseline_end=_safe_float(payload.get('baseline_end'), 0.0),
            baseline_mode=str(payload.get('baseline_mode', 'logratio')),
            all_cycle_fallback_baseline=bool(payload.get('all_cycle_fallback_baseline', True)),
            min_stride_sec=float(payload.get('min_stride_sec', 0.6)),
            max_stride_sec=float(payload.get('max_stride_sec', 2.0)),
            cycle_points=int(payload.get('cycle_points', 200)),
            warp_mode=str(payload.get('warp_mode', 'linear_cycle')),
            fixed_toe_percent=float(payload.get('fixed_toe_percent', 0.68)),
            freqs=freqs,
            n_cycles=n_cycles,
            alpha_band=tuple(float(x) for x in payload.get('alpha_band', [7.5, 12.5])),
            beta_band=tuple(float(x) for x in payload.get('beta_band', [16.0, 32.0])),
            alpha_channels=tuple(str(x) for x in payload.get('alpha_channels', list(DEFAULT_ALPHA_CHANNELS))),
            beta_channels=tuple(str(x) for x in payload.get('beta_channels', list(DEFAULT_BETA_CHANNELS))),
            participant_role_aliases=tuple(str(x).lower() for x in payload.get('participant_role_aliases', ['participant', 'par'])),
            experimenter_role_aliases=tuple(str(x).lower() for x in payload.get('experimenter_role_aliases', ['experimenter', 'exp'])),
            hs_aliases=tuple(str(x).lower() for x in payload.get('hs_aliases', ['hs', 'heel_strike', 'heelstrike', 'rhs'])),
            to_aliases=tuple(str(x).lower() for x in payload.get('to_aliases', ['to', 'toe_off', 'toeoff', 'rto'])),
            stats_method=str(payload.get('stats_method', 'paired_t')),
            fdr_alpha=float(payload.get('fdr_alpha', 0.05)),
            bootstrap_iterations=int(payload.get('bootstrap_iterations', 1000)),
            random_state=int(payload.get('random_state', 97)),
            max_runs=_safe_int(payload.get('max_runs')),
        )



def ensure_sync_dirs(config: SyncTFRConfig) -> dict[str, Path]:
    root = config.outputs_root
    paths = {
        'root': root,
        'arrays': root / 'arrays',
        'figures': root / 'figures',
        'tables': root / 'tables',
        'reports': root / 'reports',
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths



def read_preprocessing_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ['run_stem', 'output_fif']:
        if col not in df.columns:
            df[col] = np.nan
    df = df.dropna(subset=['output_fif']).copy()
    return df



def select_runs(manifest: pd.DataFrame, subject: str | None = None,
                session: str | None = None, max_runs: int | None = None) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()
    df = manifest.copy()
    if subject:
        subj = subject if str(subject).startswith('sub-') else f'sub-{subject}'
        df = df[df['run_stem'].astype(str).str.contains(subj, regex=False)]
    if session:
        ses = session if str(session).startswith('ses-') else f'ses-{session}'
        df = df[df['run_stem'].astype(str).str.contains(ses, regex=False)]
    df = df.sort_values('run_stem').reset_index(drop=True)
    if max_runs is not None:
        df = df.head(max_runs).copy()
    return df



def read_sync_mapping(event_mapping_root: str | Path) -> pd.DataFrame:
    path = Path(event_mapping_root) / 'tables' / 'sync_gait_dict.csv'
    if not path.exists():
        return pd.DataFrame(columns=['source_label', 'condition', 'role', 'gait_event', 'event_family', 'block_marker'])
    df = pd.read_csv(path)
    for col in ['source_label', 'condition', 'role', 'gait_event', 'event_family', 'block_marker']:
        if col not in df.columns:
            df[col] = ''
    df['source_label_norm'] = df['source_label'].astype(str).str.strip().str.lower()
    return df



def _normalize_condition(value: str) -> str:
    text = str(value).strip().lower().replace(' ', '_')
    aliases = {
        'control': 'blocked',
        'walk_control': 'blocked',
        'walking_control': 'blocked',
        'wc': 'blocked',
        'blocked': 'blocked',
        'walk_blocked': 'blocked',
        'natural': 'natural',
        'walk_natural': 'natural',
        'wn': 'natural',
        'sync': 'sync',
        'walk_sync': 'sync',
        'ws': 'sync',
    }
    return aliases.get(text, text)



def _normalize_role(value: str) -> str:
    text = str(value).strip().lower().replace(' ', '_')
    if text.startswith('par') or text == 'participant':
        return 'participant'
    if text.startswith('exp') or text == 'experimenter':
        return 'experimenter'
    return text



def _normalize_gait_event(value: str) -> str:
    text = str(value).strip().lower().replace(' ', '_')
    aliases = {
        'hs': 'hs', 'heelstrike': 'hs', 'heel_strike': 'hs', 'rhs': 'hs',
        'to': 'to', 'toeoff': 'to', 'toe_off': 'to', 'rto': 'to',
    }
    return aliases.get(text, text)



def _fallback_sync_parse(label: str) -> dict[str, str]:
    lowered = str(label).strip().lower().replace(' ', '_')
    condition = 'unknown'
    if any(tok in lowered for tok in ['_wn', 'walk_natural', 'natural']):
        condition = 'natural'
    elif any(tok in lowered for tok in ['_wc', 'control', 'blocked']):
        condition = 'blocked'
    elif any(tok in lowered for tok in ['_ws', 'walk_sync', 'sync']):
        condition = 'sync'
    role = 'unknown'
    if lowered.startswith('par') or 'participant' in lowered:
        role = 'participant'
    elif lowered.startswith('exp') or 'experimenter' in lowered:
        role = 'experimenter'
    gait_event = 'unknown'
    if any(tok in lowered for tok in ['rhs', 'heel_strike', 'heelstrike', 'hs']):
        gait_event = 'hs'
    elif any(tok in lowered for tok in ['rto', 'toe_off', 'toeoff', 'to']):
        gait_event = 'to'
    event_family = 'gait_event' if gait_event in {'hs', 'to'} else 'other'
    block_marker = 'condition_onset' if ('walk_' in lowered or lowered.startswith('walk')) and gait_event == 'unknown' else 'unknown'
    return {
        'condition': condition,
        'role': role,
        'gait_event': gait_event,
        'event_family': event_family,
        'block_marker': block_marker,
    }



def canonicalize_sync_annotation(label: str, mapping_df: pd.DataFrame) -> dict[str, str] | None:
    label = str(label).strip()
    if not label:
        return None
    lowered = label.lower()
    if not mapping_df.empty:
        match = mapping_df[mapping_df['source_label_norm'].eq(lowered)]
        if not match.empty:
            row = match.iloc[0]
            info = {
                'condition': _normalize_condition(row.get('condition', 'unknown')),
                'role': _normalize_role(row.get('role', 'unknown')),
                'gait_event': _normalize_gait_event(row.get('gait_event', 'unknown')),
                'event_family': str(row.get('event_family', 'other')).strip().lower() or 'other',
                'block_marker': str(row.get('block_marker', 'unknown')).strip().lower() or 'unknown',
            }
            return info
    info = _fallback_sync_parse(label)
    if info['condition'] == 'unknown' and info['gait_event'] == 'unknown' and info['block_marker'] == 'unknown':
        return None
    return info



def annotations_to_sync_events(raw: mne.io.BaseRaw, mapping_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sfreq = raw.info['sfreq']
    for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        parsed = canonicalize_sync_annotation(str(desc), mapping_df)
        if parsed is None:
            continue
        rows.append({
            'description': str(desc),
            'onset_sec': float(onset),
            'duration_sec': float(duration),
            'sample': int(round(float(onset) * sfreq)),
            'condition': parsed['condition'],
            'role': parsed['role'],
            'gait_event': parsed['gait_event'],
            'event_family': parsed['event_family'],
            'block_marker': parsed['block_marker'],
        })
    if not rows:
        return pd.DataFrame(columns=['description', 'onset_sec', 'duration_sec', 'sample', 'condition', 'role', 'gait_event', 'event_family', 'block_marker'])
    return pd.DataFrame(rows).sort_values('sample').reset_index(drop=True)



def build_stride_manifest(events_df: pd.DataFrame, config: SyncTFRConfig, run_stem: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if events_df.empty:
        return pd.DataFrame()
    participant_df = events_df[events_df['role'].eq('participant')].copy()
    for condition in config.condition_order:
        cond_df = participant_df[participant_df['condition'].eq(condition)].sort_values('sample').reset_index(drop=True)
        hs_df = cond_df[cond_df['gait_event'].eq('hs')].reset_index(drop=True)
        to_df = cond_df[cond_df['gait_event'].eq('to')].reset_index(drop=True)
        if len(hs_df) < 2:
            continue
        for idx in range(len(hs_df) - 1):
            start = int(hs_df.loc[idx, 'sample'])
            end = int(hs_df.loc[idx + 1, 'sample'])
            duration_sec = float(hs_df.loc[idx + 1, 'onset_sec'] - hs_df.loc[idx, 'onset_sec'])
            if not (config.min_stride_sec <= duration_sec <= config.max_stride_sec):
                continue
            candidates = to_df[(to_df['sample'] > start) & (to_df['sample'] < end)]
            if candidates.empty:
                continue
            toe_sample = int(candidates.iloc[0]['sample'])
            toe_pct = float((toe_sample - start) / max(end - start, 1))
            rows.append({
                'run_stem': run_stem,
                'condition': condition,
                'stride_index': len(rows) + 1,
                'start_sample': start,
                'toe_sample': toe_sample,
                'end_sample': end,
                'duration_sec': duration_sec,
                'toe_percent_actual': toe_pct,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)



def _apply_baseline(power: np.ndarray, times: np.ndarray, config: SyncTFRConfig) -> np.ndarray:
    power = np.asarray(power, dtype=np.float32, order='C')

    baseline_mask = np.zeros(times.shape, dtype=bool)
    if config.baseline_start is not None and config.baseline_end is not None:
        baseline_mask = (times >= config.baseline_start) & (times <= config.baseline_end)
    if not baseline_mask.any() and config.all_cycle_fallback_baseline:
        baseline_mask = np.ones(times.shape, dtype=bool)
    if not baseline_mask.any():
        return power

    baseline = np.nanmean(
        power[..., baseline_mask],
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    ).astype(np.float32, copy=False)

    baseline = np.where(
        np.abs(baseline) < np.float32(1e-12),
        np.float32(1e-12),
        baseline,
    ).astype(np.float32, copy=False)

    mode = config.baseline_mode.lower()

    if mode == 'ratio':
        out = np.empty_like(power)
        np.divide(power, baseline, out=out)
        return out

    if mode == 'percent':
        out = np.empty_like(power)
        np.subtract(power, baseline, out=out)
        np.divide(out, baseline, out=out)
        out *= np.float32(100.0)
        return out

    if mode == 'zscore':
        std = np.nanstd(
            power[..., baseline_mask],
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        ).astype(np.float32, copy=False)
        std = np.where(std < np.float32(1e-12), np.float32(1.0), std).astype(np.float32, copy=False)

        out = np.empty_like(power)
        np.subtract(power, baseline, out=out)
        np.divide(out, std, out=out)
        return out

    # logratio / default
    out = np.empty_like(power)
    np.divide(power, baseline, out=out)
    np.maximum(out, np.float32(1e-12), out=out)
    np.log10(out, out=out)
    return out



def _resample_curve(y: np.ndarray, new_len: int) -> np.ndarray:
    old_x = np.linspace(0.0, 1.0, y.shape[-1])
    new_x = np.linspace(0.0, 1.0, new_len)
    flat = y.reshape(-1, y.shape[-1])
    out = np.empty((flat.shape[0], new_len), dtype=float)
    for i, row in enumerate(flat):
        out[i] = np.interp(new_x, old_x, row)
    return out.reshape(*y.shape[:-1], new_len)


def _piecewise_warp(cycle_power: np.ndarray, toe_index: int, target_toe_index: int, cycle_points: int) -> np.ndarray:
    toe_index = max(1, min(int(toe_index), cycle_power.shape[-1] - 2))
    target_toe_index = max(1, min(int(target_toe_index), cycle_points - 2))
    first = _resample_curve(cycle_power[..., :toe_index + 1], target_toe_index + 1)
    second = _resample_curve(cycle_power[..., toe_index:], cycle_points - target_toe_index)
    return np.concatenate([first[..., :-1], second], axis=-1)



def extract_stride_tfr(raw: mne.io.BaseRaw, stride_row: pd.Series, config: SyncTFRConfig) -> dict[str, Any] | None:
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(picks) == 0:
        return None

    if not getattr(raw, 'preload', False):
        raw.load_data()

    sfreq = float(raw.info['sfreq'])
    start_sample = int(stride_row['start_sample'])
    toe_sample = int(stride_row['toe_sample'])
    end_sample = int(stride_row['end_sample'])
    pre_samples = int(round(config.segment_pre_sec * sfreq))
    segment_start = max(0, start_sample - pre_samples)
    segment_stop = min(raw.n_times, end_sample)

    if segment_stop - segment_start < 8:
        return None

    data = np.asarray(raw._data[picks, segment_start:segment_stop], dtype=np.float32)

    freqs = np.asarray(config.freqs, dtype=float)
    n_cycles = np.asarray(config.n_cycles, dtype=float)

    wavelets = mne.time_frequency.morlet(
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        zero_mean=True,
    )
    max_wavelet_len = max(len(w) for w in wavelets)

    tfr_input = data
    pad_left = 0
    pad_right = 0

    if data.shape[1] < max_wavelet_len:
        pad_total = max_wavelet_len - data.shape[1]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        tfr_input = np.pad(
            data,
            ((0, 0), (pad_left, pad_right)),
            mode='reflect',
        )

    power = mne.time_frequency.tfr_array_morlet(
        tfr_input[np.newaxis, :, :],
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output='power',
        zero_mean=True,
        n_jobs=1,
    )[0]

    if pad_left or pad_right:
        power = power[:, :, pad_left:pad_left + data.shape[1]]

    power = power.astype(np.float32, copy=False)

    del tfr_input, wavelets, data

    times = (
        np.arange(data.shape[1], dtype=np.float32) / np.float32(sfreq)
        - np.float32((start_sample - segment_start) / sfreq)
    )

    power = _apply_baseline(power, times, config)

    cycle_start_idx = start_sample - segment_start
    cycle_end_idx = end_sample - segment_start
    toe_idx = toe_sample - start_sample
    cycle_power = power[..., cycle_start_idx:cycle_end_idx]

    if cycle_power.shape[-1] < 8:
        return None

    if config.warp_mode == 'fixed_toe_percent':
        target_toe_idx = int(round(config.fixed_toe_percent * (config.cycle_points - 1)))
        warped = _piecewise_warp(cycle_power, toe_idx, target_toe_idx, config.cycle_points)
    else:
        warped = _resample_curve(cycle_power, config.cycle_points)

    return {
        'power_warped': np.asarray(warped, dtype=np.float32),
        'channels': [raw.ch_names[p] for p in picks],
        'toe_percent_actual': float(stride_row['toe_percent_actual']),
        'duration_sec': float(stride_row['duration_sec']),
    }



def _band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs <= band[1])



def _channel_indices(ch_names: list[str], wanted: tuple[str, ...]) -> np.ndarray:
    idx = [i for i, ch in enumerate(ch_names) if ch in set(wanted)]
    return np.asarray(idx, dtype=int)



def _mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float('nan')
    return float(np.nanmean(values))


def _init_running_nanmean(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {
        'sum': np.zeros(shape, dtype=np.float64),
        'count': np.zeros(shape, dtype=np.uint32),
        'n': np.uint32(0),
    }


def _update_running_nanmean(state: dict[str, np.ndarray], arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(a)
    state['sum'] += np.where(valid, a, 0.0)
    state['count'] += valid.astype(np.uint32)
    state['n'] += np.uint32(1)


def _finalize_running_nanmean(state: dict[str, np.ndarray]) -> np.ndarray:
    out = np.full(state['sum'].shape, np.nan, dtype=np.float32)
    np.divide(state['sum'], state['count'], out=out, where=state['count'] > 0)
    return out

def _paired_effect_size_and_ci(x: np.ndarray, y: np.ndarray, iterations: int, seed: int) -> tuple[float, float, float]:
    diff = np.asarray(y, dtype=float) - np.asarray(x, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return np.nan, np.nan, np.nan
    sd = np.nanstd(diff, ddof=1)
    d = float(np.nanmean(diff) / sd) if sd > 1e-12 else np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(iterations):
        sample = rng.choice(diff, size=diff.size, replace=True)
        ssd = np.nanstd(sample, ddof=1)
        boots.append(np.nanmean(sample) / ssd if ssd > 1e-12 else np.nan)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return d, float(lo), float(hi)



def _fdr_bh(pvals: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    pvals = np.asarray(pvals, dtype=float)
    valid = np.isfinite(pvals)
    qvals = np.full_like(pvals, np.nan)
    reject = np.zeros_like(pvals, dtype=bool)
    if valid.sum() == 0:
        return reject, qvals
    p = pvals[valid]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    thresh = alpha * np.arange(1, m + 1) / m
    rej = ranked <= thresh
    rej = np.array([np.any(rej[i:]) for i in range(m)], dtype=bool)
    q_back = np.empty_like(q)
    q_back[order] = q
    rej_back = np.empty_like(rej)
    rej_back[order] = rej
    qvals[valid] = q_back
    reject[valid] = rej_back
    return reject, qvals



def _plot_group_tfr(group_tfr: np.ndarray, freqs: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(group_tfr, aspect='auto', origin='lower',
                   extent=[0, 100, float(freqs[0]), float(freqs[-1])])
    ax.set_xlabel('Stride cycle (%)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Power (baseline corrected)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)



def _plot_band_curves(curve_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for cond in curve_df['condition'].unique():
        sub = curve_df[curve_df['condition'].eq(cond)]
        ax.plot(sub['cycle_percent'], sub['value'], label=cond)
    ax.set_xlabel('Stride cycle (%)')
    ax.set_ylabel('Band power')
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)



def run_sync_tfr_analysis(config: SyncTFRConfig, subject: str | None = None,
                          session: str | None = None, max_runs: int | None = None) -> dict[str, Any]:
    dirs = ensure_sync_dirs(config)
    manifest = select_runs(read_preprocessing_manifest(config.preprocessing_manifest_path), subject, session, max_runs or config.max_runs)
    mapping_df = read_sync_mapping(config.event_mapping_root)
    freqs = np.asarray(config.freqs, dtype=float)

    stride_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    group_states: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    figure_manifest: list[dict[str, Any]] = []

    for rec in manifest.to_dict(orient='records'):
        output_fif = Path(str(rec['output_fif']))
        run_stem = str(rec['run_stem'])
        if not output_fif.exists():
            report_rows.append({'run_stem': run_stem, 'status': 'missing_clean_fif'})
            continue
        raw = mne.io.read_raw_fif(output_fif, preload=False, verbose='ERROR')
        raw.load_data()
        events_df = annotations_to_sync_events(raw, mapping_df)
        if events_df.empty:
            report_rows.append({'run_stem': run_stem, 'status': 'no_sync_events'})
            continue
        strides_df = build_stride_manifest(events_df, config, run_stem)
        if strides_df.empty:
            report_rows.append({'run_stem': run_stem, 'status': 'no_valid_strides'})
            continue
        subj = run_stem.split('_')[0]
        ses = next((part for part in run_stem.split('_') if part.startswith('ses-')), 'ses-unknown')
        run_array_dir = dirs['arrays'] / subj / ses
        run_array_dir.mkdir(parents=True, exist_ok=True)

        for row in strides_df.to_dict(orient='records'):
            stride_row = pd.Series(row)
            extracted = extract_stride_tfr(raw, stride_row, config)
            if extracted is None:
                continue

            condition = str(row['condition'])
            power = np.asarray(extracted['power_warped'], dtype=np.float32)

            stride_rows.append({
                **row,
                'subject': subj,
                'session': ses,
                'n_channels': power.shape[0],
                'n_freqs': power.shape[1],
                'cycle_points': power.shape[2],
            })

            npz_path = run_array_dir / f"{run_stem}_{condition}_stride-{int(row['stride_index']):03d}_tfr.npz"
            np.savez_compressed(
                npz_path,
                power=power,
                freqs=freqs,
                channels=np.asarray(extracted['channels'], dtype=object),
            )

            key = (run_stem, condition)
            if key not in group_states:
                group_states[key] = _init_running_nanmean(power.shape)
            _update_running_nanmean(group_states[key], power)

            del power, extracted, stride_row

        per_run_conditions = sorted({k[1] for k in group_states.keys() if k[0] == run_stem})
        if not per_run_conditions:
            report_rows.append({'run_stem': run_stem, 'status': 'no_extractable_strides'})
            del raw, events_df, strides_df
            gc.collect()
            continue

        for condition in per_run_conditions:
            key = (run_stem, condition)
            state = group_states[key]
            run_mean = _finalize_running_nanmean(state)

            run_npz = run_array_dir / f'{run_stem}_{condition}_run_mean_tfr.npz'
            np.savez_compressed(run_npz, power=run_mean, freqs=freqs)

            n_strides = int(state['n'])

            run_rows.append({
                'run_stem': run_stem,
                'subject': subj,
                'session': ses,
                'condition': condition,
                'n_strides': n_strides,
                'array_file': str(run_npz),
            })
            ch_names = extracted['channels']
            alpha_idx = _channel_indices(ch_names, config.alpha_channels)
            beta_idx = _channel_indices(ch_names, config.beta_channels)
            alpha_band_mask = _band_mask(freqs, config.alpha_band)
            beta_band_mask = _band_mask(freqs, config.beta_band)
            alpha_curve = np.nanmean(run_mean[alpha_idx][:, alpha_band_mask, :], axis=(0, 1)) if len(alpha_idx) and alpha_band_mask.any() else np.full(config.cycle_points, np.nan)
            beta_curve = np.nanmean(run_mean[beta_idx][:, beta_band_mask, :], axis=(0, 1)) if len(beta_idx) and beta_band_mask.any() else np.full(config.cycle_points, np.nan)
            alpha_roi_tfr = np.nanmean(run_mean[alpha_idx], axis=0) if len(alpha_idx) else np.full((len(freqs), config.cycle_points), np.nan)
            beta_roi_tfr = np.nanmean(run_mean[beta_idx], axis=0) if len(beta_idx) else np.full((len(freqs), config.cycle_points), np.nan)
            run_fig_dir = dirs['figures'] / subj / ses
            run_fig_dir.mkdir(parents=True, exist_ok=True)
            alpha_tfr_path = run_fig_dir / f'{run_stem}_{condition}_alpha_mu_tfr.png'
            beta_tfr_path = run_fig_dir / f'{run_stem}_{condition}_beta_tfr.png'
            _plot_group_tfr(alpha_roi_tfr, freqs, alpha_tfr_path, f'{run_stem} | {condition} | alpha-mu ROI')
            _plot_group_tfr(beta_roi_tfr, freqs, beta_tfr_path, f'{run_stem} | {condition} | beta ROI')
            figure_manifest.append({'figure_type': 'roi_tfr', 'run_stem': run_stem, 'condition': condition, 'path': str(alpha_tfr_path)})
            figure_manifest.append({'figure_type': 'roi_tfr', 'run_stem': run_stem, 'condition': condition, 'path': str(beta_tfr_path)})
            for cycle_percent, val in zip(np.linspace(0, 100, config.cycle_points), alpha_curve):
                curve_rows.append({'pipeline': config.pipeline_name, 'run_stem': run_stem, 'subject': subj, 'session': ses, 'condition': condition, 'roi_band': 'alpha_mu', 'cycle_percent': cycle_percent, 'value': float(val)})
            for cycle_percent, val in zip(np.linspace(0, 100, config.cycle_points), beta_curve):
                curve_rows.append({'pipeline': config.pipeline_name, 'run_stem': run_stem, 'subject': subj, 'session': ses, 'condition': condition, 'roi_band': 'beta', 'cycle_percent': cycle_percent, 'value': float(val)})
            band_rows.extend([
                {'pipeline': config.pipeline_name, 'run_stem': run_stem, 'subject': subj, 'session': ses, 'condition': condition, 'roi_band': 'alpha_mu', 'mean_power': _mean_or_nan(alpha_curve), 'n_strides': n_strides},
                {'pipeline': config.pipeline_name, 'run_stem': run_stem, 'subject': subj, 'session': ses, 'condition': condition, 'roi_band': 'beta', 'mean_power': _mean_or_nan(beta_curve), 'n_strides': n_strides},
            ])
        report_rows.append({'run_stem': run_stem, 'status': 'ok', 'n_conditions': len(per_run_conditions)})
        for condition in per_run_conditions:
            key = (run_stem, condition)
            if key in group_states:
                del group_states[key]

        del raw, events_df, strides_df
        gc.collect()

    stride_df = pd.DataFrame(stride_rows)
    run_manifest_df = pd.DataFrame(run_rows)
    band_run_df = pd.DataFrame(band_rows)
    curves_df = pd.DataFrame(curve_rows)
    report_df = pd.DataFrame(report_rows)
    fig_manifest_df = pd.DataFrame(figure_manifest)

    subject_session_df = pd.DataFrame(columns=['pipeline','subject','session','condition','roi_band','mean_power'])
    subject_pooled_df = pd.DataFrame(columns=['pipeline','subject','condition','roi_band','mean_power'])
    stats_df = pd.DataFrame(columns=['pipeline','roi_band','comparison','n_subjects','statistic','p_value','effect_size_dz','effect_ci_low','effect_ci_high'])

    if not band_run_df.empty:
        subject_session_df = (
            band_run_df.groupby(['pipeline', 'subject', 'session', 'condition', 'roi_band'], as_index=False)
            .agg(mean_power=('mean_power', 'mean'), n_strides=('n_strides', 'sum'))
        )
        subject_pooled_df = (
            subject_session_df.groupby(['pipeline', 'subject', 'condition', 'roi_band'], as_index=False)
            .agg(mean_power=('mean_power', 'mean'), n_sessions=('session', 'nunique'))
        )
        stat_rows = []
        for roi_band in sorted(subject_pooled_df['roi_band'].unique()):
            roi_df = subject_pooled_df[subject_pooled_df['roi_band'].eq(roi_band)]
            pivot = roi_df.pivot(index='subject', columns='condition', values='mean_power')
            for c1, c2 in DEFAULT_COMPARISONS:
                if c1 not in pivot.columns or c2 not in pivot.columns:
                    continue
                pair = pivot[[c1, c2]].dropna()
                if len(pair) < 2:
                    continue
                x = pair[c1].to_numpy(dtype=float)
                y = pair[c2].to_numpy(dtype=float)
                if config.stats_method.lower().startswith('wilcox'):
                    stat, p = stats.wilcoxon(x, y, alternative='two-sided')
                else:
                    stat, p = stats.ttest_rel(x, y, nan_policy='omit')
                d, lo, hi = _paired_effect_size_and_ci(x, y, config.bootstrap_iterations, config.random_state)
                stat_rows.append({
                    'pipeline': config.pipeline_name,
                    'roi_band': roi_band,
                    'comparison': f'{c1}_vs_{c2}',
                    'n_subjects': len(pair),
                    'statistic': float(stat),
                    'p_value': float(p),
                    'effect_size_dz': d,
                    'effect_ci_low': lo,
                    'effect_ci_high': hi,
                })
        stats_df = pd.DataFrame(stat_rows)
        if not stats_df.empty:
            reject, qvals = _fdr_bh(stats_df['p_value'].to_numpy(dtype=float), config.fdr_alpha)
            stats_df['p_fdr'] = qvals
            stats_df['reject_fdr'] = reject

    if not curves_df.empty:
        for roi_band in sorted(curves_df['roi_band'].unique()):
            group_curve = (
                curves_df[curves_df['roi_band'].eq(roi_band)]
                .groupby(['condition', 'cycle_percent'], as_index=False)
                .agg(value=('value', 'mean'))
            )
            out_path = dirs['figures'] / f'group_{config.pipeline_name}_{roi_band}_cycle_curves.png'
            _plot_band_curves(group_curve, out_path, f'{config.pipeline_name} | {roi_band} | group cycle curves')
            fig_manifest_df = pd.concat([fig_manifest_df, pd.DataFrame([{'figure_type': 'group_curve', 'run_stem': 'group', 'condition': 'all', 'path': str(out_path)}])], ignore_index=True)

    save_dataframe(stride_df, dirs['tables'] / 'stride_manifest.csv')
    save_dataframe(run_manifest_df, dirs['tables'] / 'run_level_tfr_manifest.csv')
    save_dataframe(band_run_df, dirs['tables'] / 'run_level_bandpower.csv')
    save_dataframe(subject_session_df, dirs['tables'] / 'subject_session_bandpower.csv')
    save_dataframe(subject_pooled_df, dirs['tables'] / 'subject_pooled_bandpower.csv')
    save_dataframe(stats_df, dirs['tables'] / 'stats_results.csv')
    save_dataframe(curves_df, dirs['tables'] / 'cycle_curve_long.csv')
    save_dataframe(fig_manifest_df, dirs['tables'] / 'figure_manifest.csv')
    save_dataframe(report_df, dirs['reports'] / 'run_reports.csv')

    summary = {
        'pipeline_name': config.pipeline_name,
        'analysis_variant': config.analysis_variant,
        'n_runs_input': int(len(manifest)),
        'n_stride_rows': int(len(stride_df)),
        'n_run_level_cells': int(len(run_manifest_df)),
        'n_subject_session_rows': int(len(subject_session_df)),
        'n_subject_pooled_rows': int(len(subject_pooled_df)),
        'n_stats_rows': int(len(stats_df)),
        'figure_manifest_rows': int(len(fig_manifest_df)),
    }
    summary_path = dirs['reports'] / 'sync_tfr_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = [
        f"Pipeline: {config.pipeline_name}",
        f"Variant: {config.analysis_variant}",
        f"Input runs: {len(manifest)}",
        f"Stride rows: {len(stride_df)}",
        f"Run-level condition cells: {len(run_manifest_df)}",
        f"Subject/session bandpower rows: {len(subject_session_df)}",
        f"Subject pooled bandpower rows: {len(subject_pooled_df)}",
        f"Stats rows: {len(stats_df)}",
        '',
        'Condition order: ' + ', '.join(config.condition_order),
        'Alpha ROI: ' + ', '.join(config.alpha_channels),
        'Beta ROI: ' + ', '.join(config.beta_channels),
    ]
    if not stats_df.empty:
        lines.extend(['', 'Top stats rows:', stats_df.head(10).to_string(index=False)])
    write_text(dirs['reports'] / 'sync_tfr_report.md', '\n'.join(lines))

    return {
        'summary': summary,
        'stride_manifest': stride_df,
        'run_level_tfr_manifest': run_manifest_df,
        'run_level_bandpower': band_run_df,
        'subject_session_bandpower': subject_session_df,
        'subject_pooled_bandpower': subject_pooled_df,
        'stats_results': stats_df,
        'cycle_curve_long': curves_df,
        'figure_manifest': fig_manifest_df,
        'summary_path': summary_path,
    }
