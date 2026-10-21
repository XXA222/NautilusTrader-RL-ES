"""Require every workspace RLES package to load from installed wheels in this env."""
from __future__ import annotations

import importlib
from importlib.metadata import distribution
import json
from pathlib import Path
import sys


MODULES = (
    'rles_contracts', 'rles_data', 'rles_nautilus_adapter', 'rles_execution',
    'rles_risk', 'rles_gym_bridge', 'rles_rl_sb3', 'rles_es_evotorch',
    'rles_orchestrator', 'rles_compute_runtime', 'rles_research',
)


def main() -> None:
    prefix = Path(sys.prefix).resolve()
    records: dict[str, dict[str, object]] = {}
    for name in MODULES:
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise SystemExit(f'{name}: missing installed package path')
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(prefix):
            raise SystemExit(f'{name}: imported outside wheel environment: {path}')
        dist = distribution(name.replace('_', '-'))
        direct_url = dist.read_text('direct_url.json')
        if direct_url is None:
            raise SystemExit(f'{name}: local wheel provenance missing')
        origin = json.loads(direct_url)
        if (origin.get('dir_info', {}).get('editable')
            or not isinstance(origin.get('archive_info'), dict)
            or not origin.get('url', '').startswith('file:')
            or not origin.get('url', '').lower().endswith('.whl')):
            raise SystemExit(f'{name}: expected archive installation, got {origin}')
        entry_points = []
        for entry_point in sorted(dist.entry_points, key=lambda item: (item.group, item.name)):
            entry_module = importlib.import_module(entry_point.module)
            entry_path = (Path(entry_module.__file__).resolve()
                          if entry_module.__file__ is not None else None)
            if entry_path is None or not entry_path.is_relative_to(prefix):
                raise SystemExit(
                    f'{name}:{entry_point.group}:{entry_point.name}: '
                    f'entry point module imported outside wheel environment: {entry_path}'
                )
            entry_points.append({
                'group': entry_point.group,
                'name': entry_point.name,
                'module': entry_point.module,
                'module_path': str(entry_path),
            })
        records[name] = {
            'version': dist.version,
            'module_path': str(path),
            'entry_points': entry_points,
        }
    print(json.dumps({'python': sys.version.split()[0], 'prefix': str(prefix),
                      'installed_wheels': records}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
