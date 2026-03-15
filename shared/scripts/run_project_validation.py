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
    parser = argparse.ArgumentParser(description='Run project-wide validation/freeze pass.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    project_root = find_project_root(Path.cwd())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from shared.src.project_validation import ProjectValidationConfig, run_project_validation

    cfg = ProjectValidationConfig.from_yaml(project_root / args.config, project_root=project_root)
    outputs = run_project_validation(cfg)
    checks = outputs['validation_checks']
    summary = outputs['validation_summary']
    print('Project validation finished.')
    print(summary.to_string(index=False))
    print('status_counts:')
    print(checks.groupby(['pipeline', 'status']).size().rename('n').reset_index().to_string(index=False))


if __name__ == '__main__':
    main()
