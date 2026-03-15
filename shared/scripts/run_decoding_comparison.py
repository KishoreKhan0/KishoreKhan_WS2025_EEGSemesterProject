
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / 'configs').exists() and (candidate / 'shared').exists():
            return candidate
    raise RuntimeError('Could not locate project root.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Run decoding comparison across authors and ours pipelines.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    project_root = find_project_root(Path.cwd())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from shared.src.decoding_comparison import DecodingComparisonConfig, run_decoding_comparison

    config = DecodingComparisonConfig.from_yaml(project_root / args.config, project_root=project_root)
    outputs = run_decoding_comparison(config)
    print('Decoding comparison finished.')
    print(outputs['pipeline_summary'].to_string(index=False))


if __name__ == '__main__':
    main()
