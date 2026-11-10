"""Build an initial scene from scene x object x layout, optionally settle in PhysX.

Example:
  python -m geomind.tools.assemble_scene --instruction \
    'put 4 blue cans in grid of 2*2 in scene 60' \
    --output outputs/assemblies/scene60_bluecan_grid2x2 --verify-physics
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from geomind.scene_assembly.assets import ROOT


def _python_environment(override=None, *, usd_only=False):
    # The selected environment is explicit and identical for all workers.
    # A CPU-only caller must name its IsaacSim5.1 interpreter; never silently
    # fall back to another old conda environment on the host.
    selected = override or os.environ.get('GEOMIND_ISAAC_PYTHON') or sys.executable
    python=Path(selected).expanduser()
    if not python.is_file():raise ValueError(f'Selected Isaac Python is unavailable: {python}')
    prefix=python.resolve().parents[1]
    usd_dirs=list(prefix.glob('lib/python*/site-packages/isaacsim/extscache/omni.usd.libs-*'))
    versions=list(prefix.glob('lib/python*/site-packages/isaacsim/VERSION'))
    if not usd_dirs or not any(path.read_text().strip().startswith('5.1.') for path in versions):
        raise ValueError('Selected interpreter needs IsaacSim5.1; set --isaac-python or GEOMIND_ISAAC_PYTHON to its Python3.11 environment')
    usd_dir=sorted(usd_dirs,key=lambda p:len(str(p)),reverse=True)[0]
    env=dict(os.environ);paths=[str(ROOT)]
    if usd_only:
        paths.append(str(usd_dir))
        env['LD_LIBRARY_PATH']=os.pathsep.join([str(prefix/'lib'),str(usd_dir/'bin'),env.get('LD_LIBRARY_PATH','')])
    env['PYTHONPATH']=os.pathsep.join(paths+[env.get('PYTHONPATH','')])
    env['PYTHONUNBUFFERED']='1'
    return str(python),env


def _run_worker(kind,output,python_override,timeout,token):
    python,env=_python_environment(python_override,usd_only=kind=='export')
    command=[python,'-m','geomind.tools.assemble_scene','--worker',kind,'--request',str(output/'request.json'),
             '--output',str(output),'--run-token',token]
    (output/f'{kind}_command.json').write_text(json.dumps({'argv':command,'timeout_s':timeout},indent=2))
    with (output/f'{kind}.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+timeout
        report_path=output/'physics_report.json' if kind=='physics' else None
        finished_at=None
        cleanup=False
        while process.poll() is None:
            if report_path and report_path.exists() and finished_at is None:
                try:
                    report=json.loads(report_path.read_text())
                    if report.get('run_token')==token:finished_at=time.monotonic()
                except (OSError,json.JSONDecodeError):pass
            if time.monotonic()>deadline or (finished_at and time.monotonic()-finished_at>8):
                cleanup=finished_at is not None
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                if not cleanup:raise TimeoutError(f'{kind} worker exceeded {timeout}s; see {output/f"{kind}.log"}')
                break
            time.sleep(.3)
        if process.returncode and not cleanup:
            raise RuntimeError(f'{kind} worker failed (exit {process.returncode}); see {output/f"{kind}.log"}')
        return {'exit_code':process.returncode,'bounded_cleanup_after_report':cleanup}


def main(argv=None):
    from geomind.scene_assembly.request import parse_request,validate_request
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--instruction')
    source.add_argument('--request',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--parse-only',action='store_true')
    parser.add_argument('--randomize',action='store_true',help='Enable existing structure-preserving XY randomization and bounded asset yaw')
    parser.add_argument('--seed',type=int,help='Override the reproducible layout/randomization seed')
    parser.add_argument('--verify-physics',action='store_true',help='Settle and measure this initial layout in local PhysX; includes preview and video')
    parser.add_argument('--timeout-s',type=float,default=420)
    parser.add_argument('--isaac-python')
    parser.add_argument('--worker',choices=['export','physics'],help=argparse.SUPPRESS)
    parser.add_argument('--run-token',default='',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    try:
        request=parse_request(args.instruction) if args.instruction else validate_request(json.loads(args.request.read_text()))
        if args.randomize:
            request['randomization']={**request.get('randomization',{}),'enabled':True}
        if args.seed is not None:request['seed']=args.seed
        request=validate_request(request)
        if args.parse_only:
            print(json.dumps(request,ensure_ascii=False,indent=2));return 0
        if args.output is None:raise ValueError('--output is required unless --parse-only is used')
        output=args.output.resolve()
        if args.worker=='export':
            from geomind.scene_assembly.usd import assemble
            manifest=assemble(request,output)
            print(json.dumps({'status':manifest['status'],'scene_usda':manifest['scene_usda']},ensure_ascii=False));return 0
        if args.worker=='physics':
            from geomind.scene_assembly.physics import verify
            report=verify(output,args.run_token)
            return 0 if report['passed'] else 1
        if not math.isfinite(args.timeout_s) or not 0<args.timeout_s<=420:
            raise ValueError('--timeout-s must be positive and at most 420 seconds per worker')
        # Validate all referenced local descriptors before creating a new output.
        from geomind.scene_assembly.assets import resolve_asset,resolve_scene
        resolve_asset(request['object']['asset']);resolve_scene(request['scene_id'])
        output.mkdir(parents=True,exist_ok=False)
        (output/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
        (output/'instruction.txt').write_text((args.instruction or json.dumps(request,ensure_ascii=False))+'\n')
        snapshot=output/'source_snapshot';snapshot.mkdir()
        import shutil
        for p in (ROOT/'geomind/scene_assembly').glob('*.py'):shutil.copyfile(p,snapshot/p.name)
        shutil.copyfile(Path(__file__),snapshot/'assemble_scene.py')
        if request.get('randomization',{}).get('enabled'):
            shutil.copyfile(ROOT/'geomind_lab/tasks/manager_based/manipulation/geomind/layout_randomizer.py',snapshot/'layout_randomizer.py')
        token=uuid.uuid4().hex
        print('Exporting scene, assets and layout...',flush=True)
        export=_run_worker('export',output,args.isaac_python,args.timeout_s,token)
        manifest=json.loads((output/'manifest.json').read_text());manifest['export_worker']=export
        manifest['instruction']=args.instruction or None
        if args.verify_physics:
            print('Checking actual resting objects in local PhysX...',flush=True)
            try:
                worker=_run_worker('physics',output,args.isaac_python,args.timeout_s,token)
            except Exception:
                # Keep the generated scene and explicitly retain its failed state.
                manifest['status']='assembled_physics_failed'
                if (output/'physics_report.json').exists():manifest['physics']=json.loads((output/'physics_report.json').read_text())
                (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
                raise
            physics=json.loads((output/'physics_report.json').read_text())
            if physics.get('run_token')!=token:raise ValueError('Physics report token mismatch')
            manifest['physics']=physics;manifest['physics_worker']=worker
            manifest['status']='assembled_physics_passed' if physics['passed'] else 'assembled_physics_failed'
            manifest['preview']=physics.get('preview');manifest['detail_preview']=physics.get('detail_preview');manifest['video']=physics.get('video')
        manifest['provenance']['implementation_snapshot_phase']='before_workers_started'
        manifest['provenance']['implementation_sha256']={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest()
            for p in snapshot.glob('*.py')}
        (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        if args.verify_physics and manifest['physics']['passed']:
            from geomind.scene_assembly.audit import audit
            try:
                manifest['offline_audit']=audit(output)
                if not manifest['offline_audit']['passed']:
                    raise ValueError('Offline measured-state audit failed')
            except Exception as exc:
                manifest['status']='assembled_audit_failed'
                manifest['audit_error']=str(exc)
                (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
                raise
            (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        from geomind.scene_assembly.gallery import publish_local
        gallery=publish_local(output)
        print(json.dumps({'status':manifest['status'],'scene_usda':manifest['scene_usda'],'manifest':str(output/'manifest.json'),
                          'preview':manifest.get('preview'),'gallery':gallery},ensure_ascii=False,indent=2))
        return 1 if args.verify_physics and not manifest['physics']['passed'] else 0
    except (ValueError,RuntimeError,TimeoutError,OSError) as exc:
        print(f'Assembly error: {exc}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
