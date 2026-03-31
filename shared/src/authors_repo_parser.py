from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ARRAY_ASSIGNMENT_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{(?P<body>.*?)\};", re.DOTALL)
STRING_RE = re.compile(r"'([^'\n]{1,120})'")


@dataclass
class MatlabArrayRecord:
    file: str
    variable: str
    index: int
    value: str


def _extract_array_records(text: str, file_name: str) -> list[MatlabArrayRecord]:
    rows: list[MatlabArrayRecord] = []
    for match in ARRAY_ASSIGNMENT_RE.finditer(text):
        variable = match.group('name')
        values = STRING_RE.findall(match.group('body'))
        for idx, value in enumerate(values, start=1):
            rows.append(MatlabArrayRecord(file=file_name, variable=variable, index=idx, value=value))
    return rows


def parse_matlab_arrays(repo_root: str | Path) -> pd.DataFrame:
    repo_root = Path(repo_root)
    rows: list[dict[str, object]] = []
    for path in sorted(repo_root.rglob('*.m')):
        text = path.read_text(encoding='utf-8', errors='ignore')
        for record in _extract_array_records(text, path.relative_to(repo_root).as_posix()):
            rows.append({
                'file': record.file,
                'variable': record.variable,
                'index': record.index,
                'value': record.value,
            })
    return pd.DataFrame(rows, columns=['file', 'variable', 'index', 'value'])



def extract_event_like_strings(repo_root: str | Path) -> pd.DataFrame:
    repo_root = Path(repo_root)
    rows: list[dict[str, object]] = []
    keep_tokens = ['HS', 'TO', 'WN', 'WC', 'WS', 'Walking', 'Standing', 'walk ', 'Exp', 'Par', 'odd']
    for path in sorted(repo_root.rglob('*.m')):
        text = path.read_text(encoding='utf-8', errors='ignore')
        for value in STRING_RE.findall(text):
            if any(token.lower() in value.lower() for token in keep_tokens):
                rows.append({'file': path.relative_to(repo_root).as_posix(), 'value': value})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['file', 'value'])
    return df.drop_duplicates().sort_values(['value', 'file']).reset_index(drop=True)
