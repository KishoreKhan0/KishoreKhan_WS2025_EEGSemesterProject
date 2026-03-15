from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.src.erp_replication import OddballERPConfig, run_oddball_replication


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run oddball ERP replication on preprocessed FIF files.')
    parser.add_argument('--config', required=True, help='Path to oddball ERP YAML config.')
    parser.add_argument('--subject', default=None, help='Subject id, e.g. 001 or sub-001')
    parser.add_argument('--session', default=None, help='Session id, e.g. 01 or ses-01')
    parser.add_argument('--max-runs', type=int, default=None, help='Optional limit after filtering')
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    config = OddballERPConfig.from_yaml(args.config, project_root=PROJECT_ROOT)
    outputs = run_oddball_replication(config, subject=args.subject, session=args.session, max_runs=args.max_runs)
    run_manifest = outputs['run_manifest']
    print(f'Created {len(run_manifest)} run-level oddball evoked cells for pipeline: {config.pipeline_name}')
    if not run_manifest.empty:
        print(run_manifest[['run_stem', 'label', 'n_epochs', 'evoked_file']].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
