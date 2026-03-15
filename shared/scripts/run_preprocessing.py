from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.src.preprocessing import PreprocessingConfig, run_preprocessing


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run shared EEG preprocessing for one pipeline profile.')
    parser.add_argument('--config', required=True, help='Path to preprocessing YAML config.')
    parser.add_argument('--subject', default=None, help='Subject id, e.g. 001 or sub-001')
    parser.add_argument('--session', default=None, help='Session id, e.g. 01 or ses-01')
    parser.add_argument('--task', default=None, help='Task label from run inventory')
    parser.add_argument('--run', default=None, help='Run number')
    parser.add_argument('--max-runs', type=int, default=None, help='Optional limit after filtering')
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    config = PreprocessingConfig.from_yaml(args.config, project_root=PROJECT_ROOT)
    manifest = run_preprocessing(
        config,
        subject=args.subject,
        session=args.session,
        task=args.task,
        run=args.run,
        max_runs=args.max_runs,
    )
    print(f'Processed {len(manifest)} run(s) for pipeline: {config.pipeline_name}')
    if not manifest.empty and 'output_fif' in manifest.columns:
        print(manifest[['run_stem', 'output_fif']].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
