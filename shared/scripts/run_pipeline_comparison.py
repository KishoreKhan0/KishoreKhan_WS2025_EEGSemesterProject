from __future__ import annotations

import argparse
from pathlib import Path
import sys


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / 'configs').exists() and (candidate / 'shared').exists():
            return candidate
    raise RuntimeError('Could not locate project root from script location.')


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.src.pipeline_comparison import PipelineComparisonConfig, run_pipeline_comparison  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run side-by-side comparison between ours and authors outputs.')
    parser.add_argument('--config', required=True, help='Path to YAML config, relative to project root or absolute.')
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = PipelineComparisonConfig.from_yaml(config_path, project_root=PROJECT_ROOT)
    outputs = run_pipeline_comparison(config)
    print('Pipeline comparison finished.')
    print(outputs['oddball_summary'].to_string(index=False) if not outputs['oddball_summary'].empty else 'No oddball summary rows.')


if __name__ == '__main__':
    main()
