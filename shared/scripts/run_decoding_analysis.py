
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.src.decoding_analysis import DecodingConfig, run_decoding_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description='Run oddball condition decoding analysis.')
    parser.add_argument('--config', required=True, help='Path to YAML config file.')
    parser.add_argument('--subject', default=None, help='Optional subject filter, e.g. sub-001 or 001.')
    parser.add_argument('--max-subjects', type=int, default=None, help='Optional limit for debugging.')
    args = parser.parse_args()

    config = DecodingConfig.from_yaml(args.config, project_root=PROJECT_ROOT)
    outputs = run_decoding_analysis(config, subject=args.subject, max_subjects=args.max_subjects)
    summary = outputs['summary']
    print('Oddball decoding analysis finished.')
    print(f"pipeline_name: {summary['pipeline_name']}")
    print(f"n_runs_input: {summary['n_runs_input']}")
    print(f"n_subjects: {summary['n_subjects']}")
    print(f"n_subject_pair_rows: {summary['n_subject_pair_rows']}")
    print(f"n_stats_rows: {summary['n_stats_rows']}")
    print(f"figure_count: {summary['figure_count']}")
    print(f"summary_file: {outputs['summary_file']}")


if __name__ == '__main__':
    main()
