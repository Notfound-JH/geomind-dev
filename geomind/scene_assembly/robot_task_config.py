"""Create an explicit robot contract bound to an already assembled USD scene.

Original GT configs and source physics evidence remain read-only. The native
loader opens the assembled USD and binds existing rigid bodies; no helper table,
new task props or second layout randomization is authored by this converter.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from geomind.episodes import EpisodeResolver
from geomind.episodes.physical_contract import physics_profile,execution_profile
from geomind.episodes.resolver import load_json

from .task_binding import _load_measured_assembly
from .task_request import validate_robot_execution


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def temporal_execution_timing(resolved, profile):
    """Materialize the authored/default motion horizon in integer PhysX ticks."""
    params = resolved.episode.get('parameters', {})
    temporal = params.get('temporal_plan', {})
    if not temporal.get('required'):
        return None
    movement = temporal.get('actual_movement', {})
    original = params['legacy_task_spec']['protocol'].get('intervention', {})
    explicit = 'duration_s' in original
    duration = original.get('duration_s', movement.get('duration_s'))
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Temporal task needs a positive materialized motion duration')
    waypoints = movement.get('waypoints', [])
    if not isinstance(waypoints, list) or len(waypoints) < 2 or resolved.protocol is None:
        raise ValueError('Temporal task needs waypoint commands and a native protocol')
    phases = resolved.protocol['phases'];events = phases[1].get('events', [])
    times = sorted({event['after_steps'] for event in events})
    if len(times) != len(waypoints)-1 or not times or times[0] <= 0:
        raise ValueError('Temporal event horizon differs from the authored waypoint sequence')
    if times != [times[0]*index for index in range(1, len(times)+1)]:
        raise ValueError('Temporal waypoint commands must use one explicit uniform event interval')
    horizon = times[-1]
    step_s = float(duration)/horizon
    ticks = step_s/profile['dt_s']
    if ticks < 1 or abs(ticks-round(ticks)) > 1e-6:
        raise ValueError('Temporal protocol step must be an integer number of physics ticks')
    ticks = int(round(ticks));actual_step_s = ticks*profile['dt_s']
    if abs(actual_step_s*horizon-duration) > 1e-9:
        raise ValueError('Temporal integer-tick horizon differs from the original motion duration')
    return {'duration_s': float(duration),
        'duration_source': 'explicit_legacy_intervention' if explicit else 'materialized_legacy_default',
        'waypoint_count': len(waypoints), 'event_interval_steps': times[0],
        'command_horizon_steps': horizon, 'protocol_step_s': actual_step_s,
        'physics_ticks_per_protocol_step': ticks, 'command_horizon_s': horizon*actual_step_s,
        'observe_duration_s': phases[0]['duration_steps']*actual_step_s,
        'delay_duration_s': phases[1]['duration_steps']*actual_step_s,
        'event_execution': 'external_force_pd',
        'measured_motion_duration_verified': False,
        'scope': 'planned command timing; actual body motion and query settling remain separately measured'}


def prepare_robot_configs(gt_run, robot, execution, destination):
    """Return a new native config root using explicit robot/action settings.

    ``gt_run/task_result.json`` must have been produced by the measured-state
    binding stage. Selection aliases and their original assembled prim paths
    are preserved. Runtime will settle again and freeze its own measured GT.
    """
    gt_run=Path(gt_run).resolve();destination=Path(destination).resolve()
    if destination.exists():raise ValueError('Refusing to overwrite existing robot task configs')
    task=load_json(gt_run/'task_result.json')
    if task.get('kind')!='assembled_task_binding' or task.get('status') not in {
            'gt_solved_task_action_unverified','gt_pending_runtime_task_action_unverified'}:
        raise ValueError('Robot execution requires a completed measured-state task binding')
    robot,execution=validate_robot_execution(robot,execution,task)
    manifest_path=Path(task['source']['manifest_path']).resolve()
    manifest,measured,_=_load_measured_assembly(manifest_path)
    source=Path(task['native_config_root']).resolve()
    if source!=gt_run/'configs':raise ValueError('Task configs must belong to the selected GT run')
    resolved=EpisodeResolver(source).resolve(task['episode_id'])
    if task.get('contract_hash')!=resolved.contract_hash:
        raise ValueError('Original semantic task contract changed')
    raw_by_path={obj['prim_path']:obj for obj in measured}
    aliases=[obj['id'] for obj in task['objects']]
    if aliases!=[obj['id'] for obj in resolved.layout['objects']]:
        raise ValueError('Semantic object order differs from the saved native contract')
    bindings=[];dimensions=[]
    for obj in task['objects']:
        actual=raw_by_path.get(obj['prim_path'])
        if actual is None or not np.allclose(actual['position_m'],obj['position_m'],atol=1e-12,rtol=0):
            raise ValueError('Semantic binding differs from original raw measured geometry')
        bindings.append({'object_id':obj['id'],'prim_path':obj['prim_path'],
                         'root_position_w_m':actual['rigid_origin_world_m'],
                         'root_orientation_wxyz':actual['orientation_wxyz']})
        dimensions.append(np.asarray(actual['bbox_max_world_m'])-np.asarray(actual['bbox_min_world_m']))
    profile=physics_profile();profile.pop('table_size_m')
    profile.update(scene_mode='AssembledScene',frame_origin_w_m=[0.,0.,0.],
        robot_base_w_m=robot['base_position_w_m'],object_size_m=np.max(dimensions,axis=0).tolist(),
        object_mass_kg=.1,settle_steps=120,settle_drop_height_m=0.,
        assembly_manifest_path=str(manifest_path),assembly_manifest_sha256=_digest(manifest_path),
        assembly_object_bindings=bindings,
        asset_body_vertices_path=manifest['asset_body_vertices'],
        asset_body_vertices_sha256=manifest['provenance']['asset_body_vertices_sha256'],
        tabletop_height_m=manifest['table']['height_m'])
    if robot['embodiment']=='franka_robotiq':
        # Resolve the closed-linkage contacts at twice the integration rate.
        # The action and event clocks preserve their duration in seconds.
        profile.update(dt_s=1/240,settle_steps=240)
    temporal_timing = temporal_execution_timing(resolved, profile)
    if temporal_timing is not None:
        profile['protocol_step_s'] = temporal_timing['protocol_step_s']
    scene=deepcopy(resolved.scene)
    scene.update(robot=robot['embodiment'],usd_path=manifest['scene_usda'],physics=profile)
    episode=deepcopy(resolved.episode)
    action=execution_profile()
    action.update({key:value for key,value in execution.items() if key not in {'negative_controls','timeout_s'}})
    if action['action']=='PickPlaceObject':action['moving_object']=task['operation']['moving_object']
    if robot['embodiment']=='piper_dual':action['active_arm']=robot['active_arm']
    episode['physical_execution']=action
    shutil.copytree(source,destination)
    _write(destination/'scenes'/f"{scene['scene_id']}.json",scene)
    _write(destination/'episodes'/f"{episode['episode_id']}.json",episode)
    # Direct physical runner manifests and ordinary semantic suites are both
    # provided; the source layout poses are never modified or re-jittered.
    suite={'suite_id':'assembled_robot','cases':[{'episode_id':episode['episode_id'],'task_id':episode['task_id']}],
           'episodes':[episode['episode_id']]}
    (destination/'suites').mkdir(exist_ok=True)
    _write(destination/'suites/assembled_robot.json',suite)
    converted=EpisodeResolver(destination).resolve(episode['episode_id'])
    evidence={'kind':'assembled_robot_execution_binding','source_gt_run':str(gt_run),
        'source_semantic_contract_hash':resolved.contract_hash,'robot_contract_hash':converted.contract_hash,
        'assembly_manifest':str(manifest_path),'assembly_manifest_sha256':_digest(manifest_path),
        'robot':robot,'execution':execution,'natural_instruction':task['instruction'],
        'semantic_identity_mapping':task['source']['identity_mapping'],
        'object_bindings':bindings,'objects_respawned':False,'layout_rerandomized':False,
        'reset_initialization_pose_writes_allowed':True,'motion_object_pose_writes_allowed':False,
        'runtime_gt_policy':'Recompute from measured assembly bodies after robot reset/settling, then freeze query GT.',
        'temporal_execution_timing': temporal_timing,
        'task_action_physics_executed':False}
    _write(destination/'robot_execution_binding.json',evidence)
    return {'root':str(destination),'episode_id':episode['episode_id'],'scene_id':scene['scene_id'],
            'suite_id':'assembled_robot','robot':robot['embodiment'],'action':action['action'],
            'contract_hash':converted.contract_hash,'source_contract_hash':resolved.contract_hash,
            'binding_path':str(destination/'robot_execution_binding.json')}
