#!/usr/bin/env python3
"""Export installed versions without recording credentials, environment values or index URLs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    packages = sorted({(dist.metadata.get('Name', ''), dist.version) for dist in metadata.distributions()
                       if dist.metadata.get('Name')}, key=lambda item: item[0].lower())
    report = {'schema_version': 1, 'captured_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': platform.python_version(), 'implementation': platform.python_implementation(),
              'system': platform.system(), 'machine': platform.machine(),
              'packages': [{'name': name, 'version': version} for name, version in packages],
              'scope': 'Observed package inventory, not a dependency solver lock or fresh-host compatibility guarantee',
              'secrets_environment_values_and_index_urls_recorded': False}
    probe = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                           capture_output=True, text=True) if __import__('shutil').which('nvidia-smi') else None
    report['gpu_driver_inventory'] = probe.stdout.strip().splitlines() if probe is not None and probe.returncode == 0 else []
    (args.output / 'environment_observed.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'packages_observed.txt').write_text(
        '# Observed versions only. Install by docs/installation_zh.md, not by blindly resolving this list.\n'
        + ''.join(f'{name}=={version}\n' for name, version in packages))
    print(f'Exported {len(packages)} package versions; no environment values, credentials or index URLs exported.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
