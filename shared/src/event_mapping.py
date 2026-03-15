from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .audit_bids import save_dataframe, write_text
from .authors_repo_parser import extract_event_like_strings, parse_matlab_arrays


TASK_LABEL_PATTERNS = {
    'oddball': [r'\bstanding\b', r'\bwalkinga\b', r'\bwalkingt\b', r'\bodd\b', r'\bdeviant\b', r'\bstandard\b', r'\btone\b'],
    'synchronization': [r'\bwalk natural\b', r'\bwalk control\b', r'\bwalk sync\b', r'_wn\b', r'_wc\b', r'_ws\b', r'\brhs\b', r'\brto\b', r'\bhs\b', r'\bto\b'],
    'eyes_open_closed': [r'\beyes\b', r'\beo\b', r'\bec\b'],
}

CONDITION_RULES = [
    ('standing', [r'odd\s*standing', r'^standing$']),
    ('walking_alone', [r'odd\s*walkinga', r'walkinga']),
    ('walking_together', [r'odd\s*walkingt', r'walkingt']),
    ('natural', [r'walk natural', r'_wn\b']),
    ('control_blocked', [r'walk control', r'_wc\b', r'blocked']),
    ('sync', [r'walk sync', r'_ws\b', r'synchron']),
]

ROLE_RULES = [
    ('experimenter', [r'^exp', r'\bexperimenter\b']),
    ('participant', [r'^par', r'\bparticipant\b']),
]

GAIT_RULES = [
    ('heel_strike', [r'\bhs\b', r'hs', r'heel[- ]?strike']),
    ('toe_off', [r'\bto\b', r'to', r'toe[- ]?off', r'\brto\b']),
]

BLOCK_MARKER_RULES = [
    ('block_start', [r'^start_', r'walk\d+_start', r'walkinginstruct', r'^standing$', r'^walkinga$', r'^walkingt$']),
    ('block_end', [r'^end_', r'^end$', r'end_odd']),
]

STIMULUS_RULES = [
    ('deviant', [r'deviant']),
    ('standard', [r'standard']),
    ('countdown', [r'countdown']),
]


@dataclass
class EventMappingConfig:
    dataset_root: Path
    audit_output_root: Path
    event_mapping_output_root: Path
    authors_repo_root: Path
    event_occurrences_filename: str = 'event_occurrences_long.csv'
    event_definitions_filename: str = 'event_definitions_from_json.csv'

    @classmethod
    def from_yaml(cls, config_path: str | Path, project_root: str | Path | None = None) -> 'EventMappingConfig':
        config_path = Path(config_path)
        payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        project_root = Path(project_root).resolve() if project_root else config_path.resolve().parents[1]
        return cls(
            dataset_root=(project_root / payload['dataset_root']).resolve(),
            audit_output_root=(project_root / payload['audit_output_root']).resolve(),
            event_mapping_output_root=(project_root / payload['event_mapping_output_root']).resolve(),
            authors_repo_root=(project_root / payload['authors_repo_root']).resolve(),
            event_occurrences_filename=payload.get('event_occurrences_filename', 'event_occurrences_long.csv'),
            event_definitions_filename=payload.get('event_definitions_filename', 'event_definitions_from_json.csv'),
        )


def ensure_event_mapping_dirs(config: EventMappingConfig) -> dict[str, Path]:
    root = config.event_mapping_output_root
    tables = root / 'tables'
    reports = root / 'reports'
    for p in (root, tables, reports):
        p.mkdir(parents=True, exist_ok=True)
    return {'root': root, 'tables': tables, 'reports': reports}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ''
    text = str(value).strip()
    return re.sub(r'\s+', ' ', text)


def _match_first(text: str, rules: list[tuple[str, list[str]]]) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    for label, patterns in rules:
        for pattern in patterns:
            if re.search(pattern, lowered):
                return label
    return None


def _score_task(text: str) -> str:
    lowered = text.lower()
    scores = {key: 0 for key in TASK_LABEL_PATTERNS}
    for task, patterns in TASK_LABEL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                scores[task] += 1
    best_task = max(scores, key=scores.get)
    return best_task if scores[best_task] > 0 else 'unknown'


def _derive_confidence(row: dict[str, Any]) -> str:
    score = 0
    for key in ['canonical_task', 'condition', 'event_family']:
        if row.get(key) not in (None, '', 'unknown'):
            score += 1
    if row.get('source') == 'authors_repo':
        score += 1
    if score >= 4:
        return 'high'
    if score >= 2:
        return 'medium'
    return 'low'




def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == '.tsv':
        return pd.read_csv(path, sep='\t')
    return pd.read_csv(path)

def _read_audit_tables(config: EventMappingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    occ = _read_table(config.audit_output_root / 'tables' / config.event_occurrences_filename)
    defs = _read_table(config.audit_output_root / 'tables' / config.event_definitions_filename)
    return occ, defs


def _aggregate_unique_labels(occurrences: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not occurrences.empty:
        columns = list(occurrences.columns)
        label_column = None
        for candidate in ['trial_type', 'value', 'event_type', 'label', 'type']:
            if candidate in columns:
                label_column = candidate
                break
        if label_column:
            grouped = occurrences.groupby(label_column, dropna=False).size().reset_index(name='n_occurrences')
            grouped = grouped.rename(columns={label_column: 'source_label'})
            for rec in grouped.to_dict(orient='records'):
                rows.append({
                    'source_label': _normalize_text(rec['source_label']),
                    'source_description': '',
                    'n_occurrences': int(rec['n_occurrences']),
                    'source': 'bids_events_tsv',
                })
    if not definitions.empty:
        label_column = 'event_key' if 'event_key' in definitions.columns else definitions.columns[0] if len(definitions.columns) else None
        desc_column = None
        for candidate in ['description', 'Description', 'Levels_description']:
            if candidate in definitions.columns:
                desc_column = candidate
                break
        if label_column:
            for rec in definitions.to_dict(orient='records'):
                rows.append({
                    'source_label': _normalize_text(rec.get(label_column)),
                    'source_description': _normalize_text(rec.get(desc_column, '')),
                    'n_occurrences': 0,
                    'source': 'bids_events_json',
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['source_label', 'source_description', 'n_occurrences', 'source'])
    df = df[df['source_label'].astype(str).str.len() > 0].copy()
    def merge_desc(series: pd.Series) -> str:
        vals = [str(v).strip() for v in series if str(v).strip()]
        return ' | '.join(sorted(set(vals)))
    aggregated = (
        df.groupby('source_label', as_index=False)
          .agg(
              source_description=('source_description', merge_desc),
              n_occurrences=('n_occurrences', 'max'),
              sources=('source', lambda s: '|'.join(sorted(set(map(str, s)))))
          )
    )
    return aggregated.sort_values('source_label').reset_index(drop=True)


def _authors_reference_tables(config: EventMappingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.authors_repo_root.exists():
        return pd.DataFrame(columns=['file', 'variable', 'index', 'value']), pd.DataFrame(columns=['file', 'value'])
    return parse_matlab_arrays(config.authors_repo_root), extract_event_like_strings(config.authors_repo_root)


def build_event_mapping_table(config: EventMappingConfig) -> dict[str, pd.DataFrame]:
    occurrences, definitions = _read_audit_tables(config)
    unique_events = _aggregate_unique_labels(occurrences, definitions)
    authors_arrays, authors_strings = _authors_reference_tables(config)

    rows: list[dict[str, Any]] = []
    for rec in unique_events.to_dict(orient='records'):
        source_label = _normalize_text(rec['source_label'])
        source_description = _normalize_text(rec.get('source_description', ''))
        combined = ' | '.join(x for x in [source_label, source_description] if x)
        canonical_task = _score_task(combined)
        condition = _match_first(combined, CONDITION_RULES)
        role = _match_first(combined, ROLE_RULES)
        gait_event = _match_first(combined, GAIT_RULES)
        block_marker = _match_first(combined, BLOCK_MARKER_RULES)
        stimulus = _match_first(combined, STIMULUS_RULES)

        if gait_event:
            event_family = 'gait_event'
        elif stimulus in {'deviant', 'standard'}:
            event_family = 'stimulus'
        elif block_marker:
            event_family = 'block_marker'
        elif canonical_task == 'eyes_open_closed':
            event_family = 'task_marker'
        else:
            event_family = 'other'

        row = {
            'source_label': source_label,
            'source_description': source_description,
            'n_occurrences': rec.get('n_occurrences', 0),
            'sources': rec.get('sources', ''),
            'canonical_task': canonical_task,
            'condition': condition or 'unknown',
            'event_family': event_family,
            'block_marker': block_marker or 'unknown',
            'role': role or 'unknown',
            'gait_event': gait_event or 'unknown',
            'stimulus_class': stimulus or 'unknown',
            'source': 'bids',
        }
        row['confidence'] = _derive_confidence(row)
        rows.append(row)

    mapping_df = pd.DataFrame(rows)
    if mapping_df.empty:
        mapping_df = pd.DataFrame(columns=[
            'source_label', 'source_description', 'n_occurrences', 'sources', 'canonical_task',
            'condition', 'event_family', 'block_marker', 'role', 'gait_event', 'stimulus_class',
            'source', 'confidence'
        ])

    author_ref_rows: list[dict[str, Any]] = []
    for rec in authors_strings.to_dict(orient='records'):
        label = _normalize_text(rec['value'])
        combined = label
        row = {
            'source_label': label,
            'source_description': '',
            'n_occurrences': 0,
            'sources': rec.get('file', ''),
            'canonical_task': _score_task(combined),
            'condition': _match_first(combined, CONDITION_RULES) or 'unknown',
            'event_family': 'gait_event' if _match_first(combined, GAIT_RULES) else ('block_marker' if _match_first(combined, BLOCK_MARKER_RULES) else 'other'),
            'block_marker': _match_first(combined, BLOCK_MARKER_RULES) or 'unknown',
            'role': _match_first(combined, ROLE_RULES) or 'unknown',
            'gait_event': _match_first(combined, GAIT_RULES) or 'unknown',
            'stimulus_class': _match_first(combined, STIMULUS_RULES) or 'unknown',
            'source': 'authors_repo',
        }
        row['confidence'] = _derive_confidence(row)
        author_ref_rows.append(row)
    authors_ref_df = pd.DataFrame(author_ref_rows)
    if authors_ref_df.empty:
        authors_ref_df = pd.DataFrame(columns=mapping_df.columns)

    merged_view = pd.concat([mapping_df, authors_ref_df], ignore_index=True)
    merged_view = merged_view.drop_duplicates(subset=['source_label', 'source']).sort_values(['source_label', 'source']).reset_index(drop=True)

    unresolved = mapping_df[
        (mapping_df['confidence'] == 'low') |
        (mapping_df['canonical_task'] == 'unknown') |
        (mapping_df['condition'] == 'unknown')
    ].copy().sort_values(['confidence', 'source_label'])

    oddball_condition_dict = mapping_df[
        mapping_df['canonical_task'].eq('oddball') &
        mapping_df['condition'].ne('unknown')
    ][['source_label', 'condition', 'stimulus_class', 'block_marker']].drop_duplicates().sort_values('source_label').reset_index(drop=True)

    sync_gait_dict = mapping_df[
        mapping_df['canonical_task'].eq('synchronization')
    ][['source_label', 'condition', 'role', 'gait_event', 'event_family', 'block_marker']].drop_duplicates().sort_values('source_label').reset_index(drop=True)

    return {
        'unique_events': unique_events,
        'mapping': mapping_df,
        'authors_arrays': authors_arrays,
        'authors_strings': authors_strings,
        'authors_reference_mapping': authors_ref_df,
        'merged_view': merged_view,
        'unresolved': unresolved,
        'oddball_condition_dict': oddball_condition_dict,
        'sync_gait_dict': sync_gait_dict,
    }


def _jsonable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient='records'))


def write_event_mapping_outputs(config: EventMappingConfig, outputs: dict[str, pd.DataFrame]) -> dict[str, Path]:
    dirs = ensure_event_mapping_dirs(config)
    tables = dirs['tables']
    reports = dirs['reports']
    written: dict[str, Path] = {}

    for name, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            written[name] = save_dataframe(df, tables / f'{name}.csv')

    json_payload = {
        'oddball_condition_dict': _jsonable_records(outputs['oddball_condition_dict']),
        'sync_gait_dict': _jsonable_records(outputs['sync_gait_dict']),
    }
    json_path = reports / 'canonical_event_dictionaries.json'
    json_path.write_text(json.dumps(json_payload, indent=2), encoding='utf-8')
    written['canonical_event_dictionaries'] = json_path

    mapping = outputs['mapping']
    report_lines = [
        '# Event mapping report',
        '',
        f'- Total unique BIDS event labels mapped: {len(outputs["unique_events"])}',
        f'- Labels with direct mapping rows: {len(mapping)}',
        f'- Unresolved / low-confidence labels: {len(outputs["unresolved"])}',
        f'- Authors reference strings recovered from MATLAB repo: {len(outputs["authors_strings"])}',
        '',
        '## Task counts',
        '',
    ]
    if not mapping.empty:
        task_counts = mapping['canonical_task'].value_counts(dropna=False)
        for task, count in task_counts.items():
            report_lines.append(f'- {task}: {count}')
    report_lines.extend(['', '## Main reminders', '',
                         '- Treat oddball and synchronization tasks separately.',
                         '- Do not start ERP or TFR comparisons until unresolved labels have been checked against the BIDS sidecars.',
                         '- Use the authors MATLAB repo as a reference for synchronization labels and processing logic, not as a direct Python dependency.',
                         ''])
    report_path = write_text(reports / 'event_mapping_report.md', '\n'.join(report_lines))
    written['event_mapping_report'] = report_path
    return written


def run_full_event_mapping(config: EventMappingConfig) -> dict[str, str]:
    outputs = build_event_mapping_table(config)
    written = write_event_mapping_outputs(config, outputs)
    return {k: str(v) for k, v in written.items()}
