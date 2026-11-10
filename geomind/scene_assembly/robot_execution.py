"""Run the existing native episode worker on already assembled rigid bodies."""
from __future__ import annotations
from geomind_lab.utils.portable_paths import configure_material_search_path as _portable_material_paths

import json
import os
from pathlib import Path
import subprocess

from .assets import ROOT


def _execute_isolated_controls(argv, output, destination, environment, timeout):
    """Keep each raw worker report, then audit their unchanged traces together."""
    from geomind.tools.run_episode_physics import verification_complete
    destination.mkdir()
    reports=[];codes=[];commands=[]
    for mode in ('positive','negative'):
        run_dir=destination/'runs'/mode
        command=list(argv)
        command[command.index('--output')+1]=str(run_dir)
        if mode=='negative':command.append('--negative-only')
        commands.append(command)
        with (output/f'robot_{mode}.log').open('w') as log:
            worker=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=timeout+40)
        path=run_dir/'report.json'
        if not path.exists():raise RuntimeError(f'{mode} worker produced no report')
        reports.append(json.loads(path.read_text()));codes.append(worker.returncode)
        # The independent auditor resolves evidence by trace stem. Links retain
        # original bytes and paths; neither raw worker report is overwritten.
        for kind in ('evidence','observations','videos'):
            (destination/kind).mkdir(exist_ok=True)
            for source in (run_dir/kind).glob('*'):
                (destination/kind/source.name).symlink_to(source)
    combined=dict(reports[0])
    cases=[case for report in reports for case in report.get('results',[])]
    combined.update(results=cases,run_token=None,case_mode='isolated_positive_and_negative',
        positive_cases=sum(not r['negative_control'] for r in cases),
        positive_successes=sum(r['success'] for r in cases if not r['negative_control']),
        negative_controls=sum(r['negative_control'] for r in cases),
        video_cases=sum(r.get('video') is not None for r in cases),
        physical_execution_verified=all(r.get('physical_execution_verified') for r in reports) and codes==[0,0]
            and verification_complete(cases,None,1,1),
        failure=[r.get('failure') for r in reports],shutdown='see_raw_worker_reports',
        raw_worker_reports=[str(destination/'runs'/mode/'report.json') for mode in ('positive','negative')],
        worker_returncodes=codes)
    (destination/'report.json').write_text(json.dumps(combined,indent=2))
    (output/'robot_command.json').write_text(json.dumps({'argv_by_case':commands,'timeout_s_per_worker':timeout,
        'isolation':'separate fresh IsaacLab/PhysX process per positive/negative case'},indent=2))
    return 0 if codes==[0,0] else 1


def _audit_in_sdk_environment(output, isaac_python=None, *, timeout_s=120):
    """Replay raw evidence in the selected SDK's CPU-only USD environment.

    The application interpreter need not expose pxr. Neither Kit nor a new
    physics simulation starts here; the existing offline audit algorithm runs
    unchanged in a bounded child process.
    """
    from geomind.tools.assemble_scene import _python_environment
    output=Path(output).resolve();destination=output/'robot_physics'
    command_record=None
    try:
        python,environment=_python_environment(isaac_python,usd_only=True)
        environment['GEOMIND_OFFLINE_AUDIT_USD_CHILD']='1'
        environment['CUDA_VISIBLE_DEVICES']=''
        argv=[python,'-m','geomind.tools.audit_assembled_robot','--run',str(output)]
        command_record={'argv':argv,'timeout_s':timeout_s,'usd_only':True,
                        'GPU_requested':False,'physical_reexecution':False}
        (output/'robot_audit_command.json').write_text(json.dumps(command_record,indent=2))
        with (output/'robot_audit.log').open('w') as log:
            worker=subprocess.run(argv,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=timeout_s)
        command_record['worker_returncode']=worker.returncode
        (output/'robot_audit_command.json').write_text(json.dumps(command_record,indent=2))
        if worker.returncode!=0:
            raise RuntimeError(f'Offline robot audit exited {worker.returncode}; see {output/"robot_audit.log"}')
        checked=json.loads((destination/'assembled_robot_audit.json').read_text())
        if not isinstance(checked,dict) or type(checked.get('passed')) is not bool:
            raise ValueError('Offline robot audit produced no boolean outcome')
        return checked
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.TimeoutExpired) as exc:
        checked={'passed':False,'failure':f'{type(exc).__name__}: {exc}',
                 'scope':'offline CPU audit unavailable or rejected; raw robot outcomes are preserved'}
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'assembled_robot_audit_failure.json').write_text(json.dumps(checked,indent=2))
        return checked


def execute_robot_stage(output, request, *, isolate_negative_controls=False, negative_placement_policy='initial_first', isaac_python=None):
    from .robot_task_config import prepare_robot_configs
    from geomind.tools.assemble_scene import _python_environment
    if negative_placement_policy not in {'initial_first','offsets_first'}:
        raise ValueError('Unknown negative placement candidate order')
    output=Path(output).resolve()
    prepared=prepare_robot_configs(output,request['robot'],request['execution'],output/'robot_configs')
    manifest=json.loads(Path(json.loads((output/'result.json').read_text())['assembly_manifest']).read_text())
    bounds=manifest['table']['bounds_xy_m'];height=manifest['table']['height_m']
    cx=(bounds[0]+bounds[1])*.5;cy=(bounds[2]+bounds[3])*.5
    python,environment=_python_environment(isaac_python)
    _portable_material_paths(environment)
    destination=output/'robot_physics'
    if destination.exists():raise ValueError('Refusing to overwrite robot physics evidence')
    config=request['execution']
    argv=[python,'-m','geomind.tools.run_episode_physics','--root',str(prepared['root']),
          '--episode',prepared['episode_id'],'--output',str(destination),'--timeout-s',str(config['timeout_s']),
          '--video','--video-width','768','--video-height','768',
          '--camera-eye',str(cx+1.15),str(cy-1.35),str(height+1.1),
          '--camera-look-at',str(cx),str(cy-.12),str(height+.18)]
    isolated=bool(isolate_negative_controls and config['negative_controls'])
    if negative_placement_policy!='initial_first':
        argv.extend(['--negative-placement-policy',negative_placement_policy])
    if config['negative_controls'] and not isolated:argv.append('--negative-controls')
    (output/'robot_command.json').write_text(json.dumps({'argv':argv,'timeout_s':config['timeout_s']},indent=2))
    print(f"Executing {request['robot']['embodiment']} / {config['action']} on the actual assembled objects...",flush=True)
    try:
        if isolated:
            worker_returncode=_execute_isolated_controls(argv,output,destination,environment,config['timeout_s'])
        else:
            with (output/'robot.log').open('w') as log:
                worker=subprocess.run(argv,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,
                                      timeout=config['timeout_s']+40)
                worker_returncode=worker.returncode
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Robot supervisor exceeded its bounded deadline; see robot logs') from exc
    report_path=destination/'report.json'
    if not report_path.exists():raise RuntimeError('Robot worker produced no report; see robot.log')
    report=json.loads(report_path.read_text())
    checked=None
    if report.get('results'):
        checked=_audit_in_sdk_environment(output,isaac_python=isaac_python)
    passed=bool(worker_returncode==0 and report.get('physical_execution_verified') and checked and checked['passed'])
    return {'executed':bool(report.get('results')),'passed':passed,'robot':request['robot'],
            'action':config['action'],'config_root':str(prepared['root']),
            'report_path':str(report_path),'report':report,'offline_audit':checked,
            'worker_returncode':worker_returncode,'model_evaluated':False,
            'case_isolation':'fresh_process' if isolated else 'shared_simulator'}
