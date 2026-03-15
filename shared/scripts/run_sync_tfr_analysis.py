from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from shared.src.sync_tfr_analysis import SyncTFRConfig, run_sync_tfr_analysis


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run walking synchronization time-frequency analysis.')
    parser.add_argument('--config', required=True, help='Path to YAML config file.')
    parser.add_argument('--subject', help='Optional subject selector (e.g. 001 or sub-001).')
    parser.add_argument('--session', help='Optional session selector (e.g. 01 or ses-01).')
    parser.add_argument('--max-runs', type=int, help='Optional cap on number of runs.')
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = SyncTFRConfig.from_yaml(args.config, project_root=project_root)
    outputs = run_sync_tfr_analysis(config, subject=args.subject, session=args.session, max_runs=args.max_runs)
    print('Walking synchronization TFR analysis finished.')
    print(f"pipeline_name: {outputs['summary']['pipeline_name']}")
    print(f"n_runs_input: {outputs['summary']['n_runs_input']}")
    print(f"n_stride_rows: {outputs['summary']['n_stride_rows']}")
    print(f"n_run_level_cells: {outputs['summary']['n_run_level_cells']}")
    print(f"n_subject_session_rows: {outputs['summary']['n_subject_session_rows']}")
    print(f"n_subject_pooled_rows: {outputs['summary']['n_subject_pooled_rows']}")
    print(f"n_stats_rows: {outputs['summary']['n_stats_rows']}")
    print(f"summary_file: {outputs['summary_path']}")


if __name__ == '__main__':
    main()
