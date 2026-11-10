#!/usr/bin/env python3
"""Check a moved GeoMind checkout without starting Isaac Sim or robot actions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--assets-root', type=Path)
    parser.add_argument('--framework-manifest', type=Path, help='Override the pinned source manifest for an explicit audit')
    parser.add_argument('--simulation', action='store_true', help='Check the selected simulation interpreter and pinned framework.')
    parser.add_argument('--deep-assets', action='store_true', help='Hash every manifest file; may read tens of GB.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    errors: list[str] = []
    checks: list[dict] = []

    def check(name, ok, detail):
        checks.append({'name': name, 'passed': bool(ok), 'detail': detail})
        if not ok:
            errors.append(name)

    check('checkout', (root / 'pyproject.toml').is_file() and (root / 'geomind').is_dir(), str(root))
    for name in ('numpy', 'gymnasium'):
        try:
            value = metadata.version(name)
            check('distribution:' + name, True, value)
        except metadata.PackageNotFoundError:
            check('distribution:' + name, False, "Install -e '.[dev,analysis]' in the active interpreter")
    for name in ('geomind', 'geomind_lab', 'geomind_client', 'scripted_goal_policy'):
        try:
            spec = importlib.util.find_spec(name)
            origin = Path(spec.origin).resolve() if spec and spec.origin else None
            check('import-location:' + name, origin is not None and origin.is_relative_to(root), str(origin))
        except (ImportError, ValueError, AttributeError) as exc:
            check('import-location:' + name, False, type(exc).__name__)
    for relative in ('assets/annotations/annotated_tasks.json', 'assets/scene_profiles/assets_catalog_with_metadata.json',
                     'geomind/configs/current_task_mapping.json', 'geomind/configs/task_definitions'):
        path = root / relative
        check('checkout-resource:' + relative, path.exists(), relative)
    for name in ('git', 'ffmpeg', 'ffprobe'):
        check('executable:' + name, shutil.which(name) is not None, shutil.which(name))

    if args.simulation:
        check('simulation-python', sys.version_info[:2] == (3, 11), sys.version.split()[0])
        for name, prefix in (('isaacsim', '5.1.'), ('isaaclab', None), ('isaaclab-assets', None),
                             ('isaaclab-tasks', None), ('isaaclab-mimic', None), ('torch', '2.7.0')):
            try:
                value = metadata.version(name)
                check('simulation-distribution:' + name, prefix is None or value.startswith(prefix), value)
            except metadata.PackageNotFoundError:
                check('simulation-distribution:' + name, False, 'Not installed in this interpreter')
        framework = Path(os.environ.get('GEOMIND_ISAACLAB_ROOT', str(root / 'IsaacLab'))).resolve()
        check('framework-source', (framework / 'source/isaaclab').is_dir(), str(framework))
        for name in ('isaaclab', 'isaaclab-assets', 'isaaclab-tasks', 'isaaclab-mimic'):
            try:
                dist = metadata.distribution(name)
                raw = dist.read_text('direct_url.json')
                value = json.loads(raw) if raw else {}
                parsed = urlsplit(value.get('url', ''))
                source = Path(unquote(parsed.path)).resolve() if parsed.scheme == 'file' else None
                check('framework-editable:' + name,
                      source is not None and source.is_relative_to(framework)
                      and value.get('dir_info', {}).get('editable') is True,
                      str(source) if source is not None else 'Reinstall the pinned checkout in editable mode')
            except (metadata.PackageNotFoundError, ValueError, OSError):
                check('framework-editable:' + name, False, 'Missing or invalid editable source metadata')
        manifest_path = args.framework_manifest or root / 'migration/isaaclab/runtime_sources.json'
        check('framework-manifest', manifest_path.is_file(), str(manifest_path))
        check('framework-git-metadata', (framework / '.git').exists(), str(framework))
        if manifest_path.is_file() and (framework / '.git').exists():
            manifest = json.loads(manifest_path.read_text())
            proc = subprocess.run(['git', '-C', str(framework), 'rev-parse', 'HEAD'], capture_output=True, text=True)
            check('framework-commit', proc.returncode == 0 and proc.stdout.strip() == manifest['upstream_commit'],
                  proc.stdout.strip() if proc.returncode == 0 else 'Cannot read framework commit')
            for relative, expected in manifest.get('minimal_runtime_sources', {}).items():
                path = framework / relative
                matched = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
                check('framework-runtime:' + relative, matched, relative)

    assets = args.assets_root or (Path(os.environ['GEOMIND_ASSET_BUNDLE_ROOT'])
                                 if os.environ.get('GEOMIND_ASSET_BUNDLE_ROOT') else None)
    if assets is not None:
        assets = assets.resolve()
        for name in ('geomind_assets', 'grscenes_home', 'isaacsim_assets', 'material_cache', 'repo_assets'):
            check('asset-root:' + name, (assets / name).is_dir(), str(assets / name))
        manifest_path = assets / 'asset_bundle_manifest.json'
        check('asset-manifest', manifest_path.is_file(), str(manifest_path))
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            files = manifest.get('files', [])
            check('asset-manifest-nonempty', isinstance(files, list) and bool(files), len(files) if isinstance(files, list) else 0)
            missing = []
            mismatched = []
            for row in files if isinstance(files, list) else []:
                relative = row.get('path', row.get('relative_path', ''))
                path = (assets / relative).resolve()
                if not relative or not path.is_relative_to(assets) or not path.is_file():
                    missing.append(relative)
                    continue
                expected_size = row.get('size_bytes', row.get('bytes'))
                if expected_size is not None and path.stat().st_size != expected_size:
                    mismatched.append(relative)
                expected_hash = row.get('sha256')
                if args.deep_assets and expected_hash:
                    digest = hashlib.sha256()
                    with path.open('rb') as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b''):
                            digest.update(block)
                    if digest.hexdigest() != expected_hash:
                        mismatched.append(relative)
            check('asset-files-present', not missing, {'missing_count': len(missing), 'first_paths': missing[:20]})
            check('asset-files-integrity', not mismatched, {'deep_sha256': args.deep_assets, 'mismatch_count': len(mismatched), 'first_paths': mismatched[:20]})
    elif args.deep_assets:
        check('asset-root-configured', False, '--deep-assets requires --assets-root or GEOMIND_ASSET_BUNDLE_ROOT')

    report = {'schema_version': 1, 'passed': not errors, 'interpreter': sys.executable,
              'checks': checks, 'failed_checks': errors,
              'physics_started': False, 'robot_actions_executed': False,
              'scope': 'File, version and source-location preflight; real GPU/assets/action success still requires the documented bounded smoke test'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for row in checks:
        print(('PASS ' if row['passed'] else 'FAIL ') + row['name'] + ': ' + str(row['detail']))
    print(f"Preflight: {len(errors)} failure(s); no simulation or robot action was started.")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
