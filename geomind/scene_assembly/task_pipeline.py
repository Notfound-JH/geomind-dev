"""Bind agent-selected assembled scenes to the existing native GT solver.

This is the assemble-and-solve stage of task generation. Initial-layout physics
and task-action physics are separate results; the latter is never inferred from
a settling video. No model service or second GT registry is introduced here.
"""
from __future__ import annotations
from geomind_lab.utils.portable_paths import scene_atlas_root

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .assets import ROOT


def _write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def check_goal_geometry(manifest_path, task_spec, solved):
    """Check a placement's actual support and conservative static free space.

    This does not prove robot reachability, a grasp, or post-release stability.
    It reads geometry without opening another simulation or writing poses.
    """
    from pxr import Usd
    from .audit import measured_bounds
    from .usd import measure_table,supported_xy,_background_obstacles,_hits_obstacle
    manifest_path=Path(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    ground_truth=solved['ground_truth']
    if ground_truth['type']=='Object':
        return {'passed':True,'kind':'pick_identity','target_object':ground_truth['object_id'],
                'robot_reachability_verified':False,'task_action_physics_executed':False}
    cloud=np.load(manifest['asset_body_vertices'],allow_pickle=False)['points_m']
    with np.load(manifest_path.parent/'object_states.npz',allow_pickle=False) as trace:
        measured=trace['data'][-1]
        paths=list(map(str,trace['prim_paths']))
    geometry={}
    for path,row in zip(paths,measured):
        low,high=measured_bounds(cloud,row[:3],row[3:7])
        geometry[path.rsplit('/',1)[-1]]=(low,high)
    moving=task_spec['operation']['moving_object']
    low,high=geometry[moving];dimensions=high-low
    target=np.asarray(ground_truth['position_m'],dtype=float)
    stage=Usd.Stage.Open(manifest['scene_usda'])
    if stage is None:raise ValueError('Cannot reopen assembled USD for target support')
    table=measure_table(stage)
    if abs(table['height_m']-manifest['table']['height_m'])>1e-6:
        raise ValueError('Assembled table height changed before task generation')
    if abs(target[2]-(table['height_m']+dimensions[2]*.5))>.006:
        raise ValueError('Task GT has no supported standing height for the moving asset')
    if not supported_xy(table,target[:2],dimensions[:2]*.5):
        raise ValueError('Task GT footprint is not supported by the actual tabletop')
    obstacles=[o for o in _background_obstacles(stage,table['prim_path'])
               if not o['prim_path'].startswith('/World/AssembledObjects')]
    if _hits_obstacle(target[:2],dimensions,target[2]-dimensions[2]*.5,
                      target[2]+dimensions[2]*.5,obstacles):
        raise ValueError('Task GT intersects retained furniture')
    clearances={}
    for identity,(other_low,other_high) in geometry.items():
        if identity==moving:continue
        if other_high[2]<=target[2]-dimensions[2]*.5 or other_low[2]>=target[2]+dimensions[2]*.5:continue
        other_center=(other_low+other_high)*.5
        gaps=np.maximum(np.abs(target[:2]-other_center[:2])-(dimensions[:2]+other_high[:2]-other_low[:2])*.5,0)
        gap=float(np.linalg.norm(gaps));clearances[identity]=gap
        if gap<.005:raise ValueError(f'Task GT has less than 5mm conservative bbox gap to {identity}')
    return {'passed':True,'kind':'placement_target','reference_point':'GeometricCenter',
            'tabletop_support_samples':9,'target_center_w_m':target.tolist(),
            'moving_object':moving,'conservative_bbox_gaps_m':clearances,
            'retained_fixture_aabb_checks_passed':True,'robot_reachability_verified':False,
            'task_action_physics_executed':False,
            'limitations':'Static geometry only; not a grasp, reachability, full contact, or post-release stability proof.'}


def run_assembled_task(raw, *, command=None):
    from .task_request import validate_task_request
    from .task_binding import bind_and_solve
    from .task_gallery import publish_task_local
    from geomind.tools.research_pairs import snapshot,check_provenance,encoded,digest
    request=validate_task_request(raw)
    output=Path(request['output']).resolve()
    output.mkdir(parents=True,exist_ok=False)
    _write(output/'raw_request.json',raw);_write(output/'request.json',request)
    result={'stage':request['stage'],'status':'initializing',
            'selection':request['selection'],'initial_layout_physics_verified':False,
            'gt_verified':False,'task_action_physics_executed':False,
            'task_action_physics_verified':False,'model_evaluated':False}
    _write(output/'result.json',result)
    try:
        source_paths=[str(p.relative_to(ROOT)) for p in (ROOT/'geomind/scene_assembly').glob('*.py')]
        source_paths+=['geomind/tools/research_pipeline.py','geomind/tools/assemble_scene.py',
                       'geomind/tools/run_episode_physics.py','geomind/tools/audit_assembled_robot.py',
                       'geomind_lab/tasks/manager_based/manipulation/geomind/layout_randomizer.py']
        source_paths += [str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_assembled*.py')]
        source_paths += ['docs/robot_embodiments_zh.md','docs/assembled_tasks_zh.md']
        provenance=snapshot(output,command or [sys.executable,'-m','geomind.tools.research_pipeline'],
                            index_filename='request.json',extra_paths=source_paths)
        provenance['source_snapshot_phase']='before_assembly_and_gt'
        _write(output/'provenance.json',provenance)
        if 'assembly' in request:
            _write(output/'assembly_request.json',request['assembly'])
            argv=[sys.executable,'-m','geomind.tools.assemble_scene','--request',str(output/'assembly_request.json'),
                  '--output',str(output/'assembly'),'--verify-physics','--timeout-s',str(request['timeout_s'])]
            _write(output/'assembly_command.json',{'argv':argv,'timeout_s':2*request['timeout_s']+30})
            print('Assembling the selected scene and measuring actual settled objects...',flush=True)
            with (output/'assembly.log').open('w') as log:
                run=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                   timeout=2*request['timeout_s']+30)
            if run.returncode:raise ValueError(f'Assembly failed with exit {run.returncode}; see {output/"assembly.log"}')
            manifest_path=output/'assembly/manifest.json'
        else:
            manifest_path=Path(request['assembly_manifest']).resolve()
        manifest=json.loads(manifest_path.read_text())
        for source,expected_hash in manifest.get('provenance',{}).get('used_usd_layers_sha256',{}).items():
            if digest(Path(source).read_bytes())!=expected_hash:
                raise ValueError(f'Referenced assembly USD layer changed: {source}')
        result['assembly_manifest']=str(manifest_path)
        chosen_scene=manifest['request']['scene_id'];chosen_layout=manifest['request']['layout']['type']
        if chosen_scene not in request['selection']['considered_scene_ids'] or chosen_layout not in request['selection']['considered_layouts']:
            raise ValueError('Actual assembly disagrees with the recorded scene/layout selection candidates')
        print('Solving task GT from the measured query snapshot with the existing resolver...',flush=True)
        solved=bind_and_solve(manifest_path,request['task'],output)
        result.update(task=solved,initial_layout_physics_verified=True)
        geometry=check_goal_geometry(manifest_path,request['task'],solved)
        _write(output/'evaluator/goal_geometry.json',geometry)
        solved['target_support_and_clearance']=geometry
        solved['checks']['target_geometry_checked']=True
        solved['notes']='GT from measured settled geometry; static target geometry checked separately. Robot task-action physics is not executed.'
        _write(output/'task_result.json',solved)
        result.update(status='gt_solved_task_action_unverified',gt_verified=True,
                      task_goal_geometry=geometry,task_goal_geometry_verified=True)
        if request['stage']=='assemble_solve_execute':
            from .robot_execution import execute_robot_stage
            _write(output/'result.json',result)
            robot=execute_robot_stage(output,request)
            result.update(robot_execution=robot,task_action_physics_executed=robot['executed'],
                          task_action_physics_verified=robot['passed'],
                          status='task_action_physics_passed' if robot['passed'] else 'task_action_physics_failed')
            solved.update(action_physics_executed=robot['executed'],task_completed=robot['passed'],
                          status=result['status'])
            solved['notes']='GT and task-action outcomes are recorded separately; see robot_execution for actual runtime query and raw action evidence.'
            _write(output/'task_result.json',solved)
        # Preserve the pre-execution source snapshot, and bind newly generated
        # configs/evidence rather than recopying sources after execution.
        provenance['config_files']={p.relative_to(output).as_posix():digest(p.read_bytes())
                                    for p in sorted((output/'configs').rglob('*.json'))}
        for path in sorted((output/'robot_configs').rglob('*.json')):
            provenance['config_files'][path.relative_to(output).as_posix()]=digest(path.read_bytes())
        for name in ['raw_request.json','request.json','task_result.json','task_binding.json']:
            path=output/name
            if path.is_file():provenance['config_files'][name]=digest(path.read_bytes())
        for folder in ['evaluator','observations']:
            for path in sorted((output/folder).rglob('*.json')):
                provenance['config_files'][path.relative_to(output).as_posix()]=digest(path.read_bytes())
        provenance['configuration_id']=digest(encoded(provenance['config_files']))
        _write(output/'provenance.json',provenance)
        result['provenance_check']=check_provenance(output)
        if not result['provenance_check']['passed']:raise ValueError('Sources/configs changed during assembled-task generation')
        result['gallery']='http://127.0.0.1:8780/tasks.html' if (scene_atlas_root()/'dist').is_dir() else None
        _write(output/'result.json',result)
        published=publish_task_local(output)
        if published!=result['gallery']:raise ValueError('Local task gallery destination changed during publication')
        print(json.dumps({'status':result['status'],'scene_id':chosen_scene,'layout':chosen_layout,
                          'task_id':solved['task_id'],'gt_verified':True,
                          'task_action_physics_executed':result['task_action_physics_executed'],'output':str(output),
                          'gallery':result['gallery']},ensure_ascii=False,indent=2))
        return result
    except Exception as exc:
        result['status']='assembled_task_generation_failed'
        result['failure']={'type':type(exc).__name__,'reason':str(exc),
                           'category':'invalid_instance' if isinstance(exc,ValueError) else 'system_failure'}
        _write(output/'result.json',result);_write(output/'failure.json',result['failure'])
        raise


def audit_assembled_task(output):
    """Replay saved GT against its exact measured query; no physical execution."""
    from .task_binding import _load_measured_assembly,_independent_gt
    from geomind.episodes import EpisodeResolver,Snapshot,solve,evaluate_answer
    from geomind.tools.research_pairs import check_provenance
    output=Path(output).resolve()
    result=json.loads((output/'result.json').read_text())
    request=json.loads((output/'request.json').read_text())
    if result.get('status') not in {'gt_solved_task_action_unverified','task_action_physics_passed','task_action_physics_failed'} or not result.get('gt_verified'):
        raise ValueError('Assembled task did not finish its GT stage')
    provenance=check_provenance(output)
    if not provenance['passed']:raise ValueError('Source or configuration provenance changed')
    manifest,physical,_=_load_measured_assembly(result['assembly_manifest'])
    for source,expected_hash in manifest.get('provenance',{}).get('used_usd_layers_sha256',{}).items():
        import hashlib
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=expected_hash:
            raise ValueError(f'Referenced assembly USD layer changed: {source}')
    task=result['task']
    resolved=EpisodeResolver(output/'configs').resolve(task['episode_id'])
    query=Snapshot.from_dict(json.loads((output/'evaluator/query_snapshot.json').read_text()))
    by_prim={obj['prim_path']:obj for obj in physical}
    for obj in task['objects']:
        measured=by_prim[obj['prim_path']]['position_m']
        if not np.allclose(query.positions_m[obj['id']],measured,atol=1e-12,rtol=0):
            raise ValueError('Saved semantic query no longer matches the raw physical state')
    solved=solve(resolved,query)
    if solved['ground_truth']!=task['ground_truth']:raise ValueError('GT replay disagrees with saved task')
    oracle=_independent_gt(task['task_id'],resolved.episode['parameters'],query.positions_m)
    gt=solved['ground_truth']
    if gt['type']=='Point':
        agrees=np.allclose(oracle['position_m'],gt['position_m'],atol=1e-10,rtol=0)
    else:agrees=oracle==gt
    if not agrees:raise ValueError('Independent GT replay disagrees')
    if evaluate_answer(resolved,solved,task['checks']['semantic_wrong_answer'])['success']:
        raise ValueError('Saved wrong-answer control was accepted')
    geometry=check_goal_geometry(result['assembly_manifest'],request['task'],solved)
    public=json.loads(Path(task['public_observation_path']).read_text())
    forbidden={'ground_truth','parameters','operation','provenance','source','semantic_id_mapping'}
    if forbidden&public.keys():raise ValueError('Privileged task data found in public observation')
    public_objects={obj['object_id']:obj['position_m'] for obj in public['objects']}
    if public_objects!={key:list(value) for key,value in query.positions_m.items()}:
        raise ValueError('Public object observations disagree with measured query')
    report={'passed':True,'mode':'offline GT/geometry replay; no physical re-execution',
            'native_resolver_replayed':True,'independent_gt_agrees':True,
            'raw_state_and_semantic_query_match':True,'wrong_answer_rejected':True,
            'public_privileged_data_separated':True,'target_geometry':geometry,
            'task_action_physics_executed':False,'model_evaluated':False,
            'provenance':provenance}
    if result.get('stage')=='assemble_solve_execute':
        from geomind.tools.audit_assembled_robot import audit
        robot_audit=audit(output/'robot_configs',output/'robot_physics')
        report['robot_action_audit']=robot_audit
        report['task_action_physics_executed']=result['task_action_physics_executed']
    _write(output/'assembled_task_audit.json',report)
    return report
