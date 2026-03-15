from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.src.sync_comparison import SyncComparisonConfig, run_sync_comparison  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description='Run walking synchronization comparison between authors and ours pipelines.')
    parser.add_argument('--config', required=True, help='Path to YAML config file.')
    args = parser.parse_args()

    config = SyncComparisonConfig.from_yaml(args.config, project_root=PROJECT_ROOT)
    outputs = run_sync_comparison(config)

    summary = outputs['summary']
    print('Walking sync comparison finished.')
    if hasattr(summary, 'to_string'):
        cols = [c for c in ['pipeline', 'n_stride_rows', 'n_run_level_cells', 'n_subject_pooled_rows', 'n_stats_rows'] if c in summary.columns]
        print(summary[cols].to_string(index=False))


if __name__ == '__main__':
    main()
