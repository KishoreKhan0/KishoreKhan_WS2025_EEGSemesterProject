from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.src.event_mapping import EventMappingConfig, run_full_event_mapping


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run shared event mapping for ds004033.')
    parser.add_argument('--config', type=str, default='configs/event_mapping.yaml', help='Path to event mapping YAML config.')
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = EventMappingConfig.from_yaml(args.config, project_root=PROJECT_ROOT)
    outputs = run_full_event_mapping(config)
    print('Event mapping finished. Files written:')
    for name, path in outputs.items():
        print(f' - {name}: {path}')


if __name__ == '__main__':
    main()
