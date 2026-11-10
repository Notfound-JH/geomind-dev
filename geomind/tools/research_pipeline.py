"""Reproducible research requests, using the existing native episode pipeline."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

from geomind.core.taxonomy import CANONICAL_TASK_FAMILIES
from geomind.episodes import EpisodeResolver, SemanticEpisode, Snapshot, solve
from geomind.episodes.resolver import _check, load_json
from geomind.tools.generate_spatial_suite import generate, verify, TASK_IDS
from geomind.tools.research_pairs import snapshot, write, public_observation, check_provenance

REPO = Path(__file__).resolve().parents[2]


def parse_request(raw):
    allowed = {'task_family', 'task_id', 'condition', 'augmentation', 'scene_id', 'layout',
               'seed', 'count', 'time_protocol', 'artifacts', 'backend', 'output', 'timeout_s', 'coverage', 'split', 'interaction_mode'}
    _check(isinstance(raw, dict) and not raw.keys() - allowed, 'Unknown request fields')
    r = dict(task_family=None, task_id=None, condition='Base', augmentation=None,
             scene_id='semantic_table', layout='spatial_generator_v1', seed=0, count=1,
             time_protocol=None, artifacts=['gt', 'observations', 'events', 'robot_trajectory'],
             backend='isaaclab', timeout_s=600, coverage='representative', split=None, interaction_mode=None)
    r.update(raw)
    _check('output' in r and isinstance(r['output'], str) and r['output'], 'output required')
    _check(r['task_family'] in CANONICAL_TASK_FAMILIES, 'Unknown task_family')
    family = CANONICAL_TASK_FAMILIES[r['task_family']]
    if r['interaction_mode'] is None:
        r['interaction_mode'] = 'manipulation'
    _check(r['interaction_mode'] in {'manipulation', 'diagnostic'}, 'Unknown interaction_mode')
    _check(r['coverage'] in {'representative', 'all'}, 'Unknown coverage')
    support = {t: t in TASK_IDS for t in family}
    r['family_coverage'] = support
    if r['coverage'] == 'all':
        _check(all(support.values()), f'Full family unsupported: {support}')
    _check(r['interaction_mode'] != 'manipulation' or (r['task_family'] == 'SceneUpdating' and r['task_id'] in (None, 'DetectSceneChange')),
           'Manipulation execution is not implemented in this pipeline: missing grasp, transport, '
           'release and post-release object-state evaluation. No task was generated or executed. '
           'ReportPoint/ReachAboveTarget is diagnostic only; do not substitute it for placement. '
           'Use interaction_mode=diagnostic only when coordinate/reach diagnostics are explicitly requested.')
    if r['task_id'] is None:
        preferred = 'ProjectOntoLine' if r['task_family'] == 'GeometricInference' else 'DetectSceneChange'
        _check(preferred in family and preferred in TASK_IDS, f'No native generator for family: {support}')
        r['task_id'] = preferred
        r['selection_reason'] = ('Existing displacement generator and PickLiftChangedObject executor' if r['interaction_mode'] == 'manipulation' else 'Existing native generator, independent GT check and PhysX ReachAboveTarget executor')
    else:
        _check(r['task_id'] in family and r['task_id'] in TASK_IDS, f'Unsupported task/family combination: {support}')
        r['selection_reason'] = 'explicit task_id'
    _check(r['scene_id'] in {'semantic_table', 'registered_scene_65'}, 'Unsupported scene; no fallback')
    _check(r['layout'] == 'spatial_generator_v1', 'Unsupported layout')
    _check(r['backend'] in {'semantic', 'isaaclab'}, 'Unsupported backend; no mock fallback')
    _check(r['condition'] in {'Base', 'Aug'}, 'Unknown condition')
    _check(type(r['seed']) is int and r['seed'] >= 0 and type(r['count']) is int and 1 <= r['count'] <= 4,
           'Require nonnegative seed and count 1..4')
    _check(type(r['timeout_s']) in (int, float) and math.isfinite(r['timeout_s']) and 1 <= r['timeout_s'] <= 1800,
           'timeout_s must be 1..1800')
    _check(isinstance(r['artifacts'], list) and set(r['artifacts']) <= {'gt','observations','events','robot_trajectory'}, 'Unknown artifacts')
    _check(not (r['backend'] == 'semantic' and 'robot_trajectory' in r['artifacts']), 'CPU cannot produce robot trajectory')
    protocol = 'observe_delay_respond_displacement' if r['task_id'] == 'DetectSceneChange' else 'static_query'
    _check(r['time_protocol'] in (None, protocol), 'Unsupported time protocol')
    r['time_protocol'] = protocol
    if r['condition'] == 'Aug':
        _check(r['task_id'] == 'ProjectOntoLine' and r['augmentation'] == 'translate_x_0.03m', 'Only explicit projection translation Aug supported')
    else:
        _check(r['augmentation'] is None, 'Base cannot specify augmentation')
    r['definition'] = 'native-v3 / docs/task_descriptions.md 2026-11-07; operational definitions remain draft'
    r['base_distribution'] = {'generator': 'generate_spatial_suite.generate', 'version': 1,
        'geometry': 'yaw U(-pi,pi), pitch U(-.4,.4), centre xy U(-.05,.05), z=.2; along U(-.16,.16); perpendicular offset .07 with nz=0',
        'displacement': '3 blocks x=-.15,0,.15; initial/destination y U(-.2,.2); one shift >=.12; threshold .03, margin .001; max 100 attempts',
        'scenes': ['semantic_table', 'registered_scene_65'], 'input': 'structured coordinates; no model evaluated'}
    r['augmentation_contract'] = None if r['condition'] == 'Base' else {
        'base': 'same scene, generator version and seed; paired Base configs saved', 'translation_m': [.03,0,0],
        'preserved': ['roles','infinite 3D line projection','answer type','relative geometry','input modality','time protocol'],
        'gt': 'independent construction translated; production GT recomputed after transform and physical settling',
        'status': 'engineering realization of allowed geometric-transform draft; no difficulty claim'}
    return r


def prepare(r, out):
    root = out / 'configs'
    manifest = generate(root, list(range(r['seed'], r['seed'] + r['count'])),
                        scene_id=65 if r['scene_id'] == 'registered_scene_65' else None)
    manifest['cases'] = [c for c in manifest['cases'] if c['task_id'] == r['task_id']]
    for case in manifest['cases']:
        ep = load_json(root / 'episodes' / (case['episode_id'] + '.json'))
        lp = root / 'layouts' / (ep['layout_id'] + '.json')
        layout = load_json(lp)
        if r['interaction_mode'] == 'manipulation':
            ep['physical_execution']['action'] = 'PickLiftChangedObject'
            write(root / 'episodes' / (case['episode_id'] + '.json'), ep)
        if r['condition'] == 'Aug':
            write(out / 'paired_base' / (case['episode_id'] + '.json'), {'episode': ep, 'layout': layout, 'expected_answer': case['expected_answer']})
            for obj in layout['objects']:
                obj['position_m'][0] += .03
            case['expected_answer']['position_m'][0] += .03
            write(lp, layout)
    write(root / 'suites/research_request.json', manifest)
    write(out / 'request.json', r)
    report = verify(root, manifest)
    for case in manifest['cases']:
        resolved = EpisodeResolver(root).resolve(case['episode_id'])
        env = SemanticEpisode(resolved)
        obs, _ = env.reset()
        stream = [public_observation(obs)]
        while not obs['allow_response']:
            obs, *_ = env.step()
            stream.append(public_observation(obs))
        write(out / 'semantic' / (case['episode_id'] + '.json'), {'observations': stream,
            'semantic_gt': env.result, 'evidence': env.export_evidence(), 'event_plan': resolved.protocol,
            'event_execution': 'CPU discrete state update; not physical motion'})
        env.close()
    return report


def audit_physics(root, output):
    """Offline re-computation from exported measured states, not physical re-execution."""
    import numpy as np
    report = load_json(output / 'report.json')
    if report.get('action') == 'PickLiftChangedObject':
        from geomind.tools.audit_changed_lift import audit
        return audit(root, output)
    results = []
    for case in report.get('results', []):
        trace_file = np.load(case['trace'], allow_pickle=False)
        trace = trace_file['data']
        tcp_index = list(trace_file['columns']).index('tcp_x_w')
        controls = np.load(case['control_trace'], allow_pickle=False)
        u = controls['data']; n = len(controls['joint_names']); limits = controls['joint_limits']
        _check(trace.ndim == 2 and len(trace) > 1 and trace.shape[1] == tcp_index+3+3*len(EpisodeResolver(root).resolve(case['episode_id']).layout['objects']), 'Invalid measured trace shape')
        _check(u.shape == (len(trace), 3*n+4), 'Control/measurement shape mismatch')
        _check(np.isfinite(trace).all() and np.isfinite(u).all(), 'Nonfinite trace')
        _check(np.all(np.diff(trace[:,0]) == 1), 'Nonmonotonic physical clock')
        _check(str(controls['instance_id']) == case['instance_id'], 'Mixed instance')
        resolved = EpisodeResolver(root).resolve(case['episode_id'])
        _check(str(controls['contract_hash']) == resolved.contract_hash, 'Mixed configuration')
        _check(np.all(u[:,:n] >= limits[:,0]-1e-5) and np.all(u[:,:n] <= limits[:,1]+1e-5), 'Action outside joint limits')
        measured_limits = bool(np.all(u[:,n:2*n] >= limits[:,0]-.005) and np.all(u[:,n:2*n] <= limits[:,1]+.005))
        _check(measured_limits, 'Measured joints outside limits (5mrad/5mm allowance)')
        p = case['physical']; target = np.array(p['target_tcp_w_m'])
        errors = np.linalg.norm(trace[:,tcp_index:tcp_index+3]-target, axis=1)
        angles = 2*np.arccos(np.clip(np.abs(u[:,-4:] @ np.array(case.get('tool_quaternion_wxyz',[0,1,0,0]))),0,1)) # quaternion wxyz dot [0,1,0,0]
        _check(abs(errors[-1] - p['position_error_m']) < 2e-6, 'Final error disagreement')
        dwell = 0
        for ok in ((errors <= p['position_tolerance_m']) & (angles <= p['orientation_tolerance_rad']) & (trace[:,2] == 4))[::-1]:
            if not ok: break
            dwell += 1
        offline_success = dwell >= p['required_dwell_steps']
        _check(offline_success == p['success'], 'Offline dwell disagrees')
        evidence = load_json(output / 'evidence' / (Path(case['trace']).stem + '.json'))
        _check(evidence['current']['instance_id'] == case['instance_id'], 'Mixed evidence instance')
        replay = solve(resolved, Snapshot.from_dict(evidence['current']), history={k: Snapshot.from_dict(v) for k,v in evidence['history'].items()})
        _check(replay['ground_truth'] == case['ground_truth'], 'GT replay mismatch')
        tick = evidence['physical_clock']['current_tick']
        row = trace[int(tick)-1]
        origin = np.array(resolved.scene['physics']['frame_origin_w_m'])
        for i,obj in enumerate(resolved.layout['objects']):
            _check(np.allclose(row[tcp_index+3+3*i:tcp_index+6+3*i]-origin, evidence['current']['positions_m'][obj['id']], atol=2e-7), 'Query state/trace mismatch')
        stream = load_json(output / 'observations' / (Path(case['trace']).stem + '.json'))['observations']
        for obs in stream:
            public_observation({k:v for k,v in obs.items() if k != 'robot'})
        _check(stream[-1]['allow_response'], 'Missing query observation')
        results.append({'episode_id':case['episode_id'], 'negative_control':case['negative_control'],
                        'rows':len(trace), 'dwell_recomputed':dwell, 'error_m':float(errors[-1]),
                        'joint_limits_checked':True, 'query_state_matched':True, 'gt_replayed':True})
    _check(results and report['physical_execution_verified'], 'Physical run incomplete')
    result = {'passed': True, 'mode':'offline recomputation; no physical replay', 'cases':results,
              'collision_status':'colliders active; no contact sensor/contact-free certification'}
    write(output / 'offline_audit.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--audit', type=Path, help='Existing run directory; offline only')
    parser.add_argument('--assembly-options', action='store_true', help='List actual scene/asset/layout choices and native task GT capabilities for an external agent')
    parser.add_argument('--assembled-request', type=Path, help='Assemble an agent-selected scene and solve task GT from measured states; task-action execution is separately unverified')
    parser.add_argument('--audit-assembled', type=Path, help='Offline replay of an assembled-task GT run, without re-executing physics')
    args = parser.parse_args(argv)
    _check(sum([bool(args.request),bool(args.audit),bool(args.assembly_options),bool(args.assembled_request),bool(args.audit_assembled)])==1,
           'Choose exactly one of --request, --audit, --assembly-options, --assembled-request, --audit-assembled')
    if args.audit_assembled:
        from geomind.scene_assembly.task_pipeline import audit_assembled_task
        try:
            print(json.dumps(audit_assembled_task(args.audit_assembled),ensure_ascii=False,indent=2));return 0
        except (ValueError,RuntimeError,OSError) as exc:
            print(f'Assembled task audit error: {exc}',file=sys.stderr);return 2
    if args.assembly_options:
        from geomind.scene_assembly.task_request import assembly_options
        print(json.dumps(assembly_options(),ensure_ascii=False,indent=2));return 0
    if args.assembled_request:
        from geomind.scene_assembly.task_pipeline import run_assembled_task
        try:
            assembled=run_assembled_task(load_json(args.assembled_request),command=[sys.executable,'-m','geomind.tools.research_pipeline',*(argv or sys.argv[1:])])
            return 1 if assembled['status']=='task_action_physics_failed' else 0
        except (ValueError,RuntimeError,TimeoutError,OSError) as exc:
            print(f'Assembled task generation error: {exc}',file=sys.stderr);return 2
    if args.audit:
        print(json.dumps(audit_physics(args.audit/'configs', args.audit/'physics'), indent=2)); return 0
    _check(args.request is not None, '--request required')
    raw = load_json(args.request)
    out = Path(raw['output'])
    out.mkdir(parents=True, exist_ok=False)
    write(out/'raw_request.json',raw)
    try:
        r = parse_request(raw)
        semantic = prepare(r,out)
        extra = ['geomind/tools/research_pipeline.py','geomind/tools/generate_spatial_suite.py',
                 'geomind/tools/run_episode_physics.py','docs/task_descriptions.md',
                 'geomind/core/envs/isaaclab_native_delegate.py', 'geomind/tools/audit_changed_lift.py']
        extra += [p.relative_to(REPO).as_posix() for p in (REPO/'geomind_lab/utils').glob('*.py')]
        extra += ['geomind_lab/tasks/manager_based/manipulation/geomind/geomind_scene_cfg.py']
        snapshot(out, [sys.executable,'-m','geomind.tools.research_pipeline',*(argv or sys.argv[1:])],
                 index_filename='request.json', extra_paths=extra)
        result = {'generated':True,'semantic_verified':semantic['passed'], 'physical_executed':False,
                  'physical_verified':False,'model_evaluated':False}
        write(out/'result.json',result)
        if r['backend'] == 'isaaclab':
            command = [sys.executable,'-m','geomind.tools.run_episode_physics','--root',str(out/'configs'),
                       '--suite','research_request','--output',str(out/'physics'),'--timeout-s',str(r['timeout_s']), '--negative-controls']
            if r['interaction_mode'] == 'manipulation':
                command.append('--video')
            write(out/'physical_command.json',command)
            with (out/'physics.log').open('w') as log:
                status = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
            physical = load_json(out/'physics/report.json')
            result.update(physical_executed=bool(physical.get('results')), physical_verified=False,
                          physical_returncode=status)
            write(out/'result.json',result)
            _check(status == 0, f'Physics failed: {physical.get("failure")}')
            audit_physics(out/'configs',out/'physics')
            result['physical_verified'] = True
        result['provenance_check'] = check_provenance(out)
        write(out/'result.json',result)
        print(json.dumps(result,indent=2)); return 0
    except Exception as exc:
        write(out/'failure.json',{'type':type(exc).__name__,'reason':str(exc)})
        if not (out/'result.json').exists():
            write(out/'result.json', {'generated': False, 'physical_executed': False,
                  'physical_verified': False, 'model_evaluated': False,
                  'status': 'request_rejected', 'reason': str(exc)})
        raise

if __name__ == '__main__':
    raise SystemExit(main())
