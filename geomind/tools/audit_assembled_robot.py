"""Offline audit of contact actions using raw joints and original mesh geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from geomind.episodes import EpisodeResolver,Snapshot,solve


def _require(value,message):
    if not value:raise ValueError(message)


def _gt_equivalent(left,right):
    """Exact identity/schema, or a finite Point replay within one picometre."""
    if not isinstance(left,dict) or not isinstance(right,dict) or left.get('type')!=right.get('type'):return False
    if left.get('type')=='Object':
        return (set(left)==set(right)=={'type','object_id'} and isinstance(left['object_id'],str)
                and bool(left['object_id']) and left['object_id']==right['object_id'])
    kind=left.get('type')
    fields=({'type','position_m','frame'} if kind=='Point' else
            {'type','relation','target_object_id','reference_object_id','contract','position_m','frame','point_role'}
            if kind=='Relation' else None)
    if fields is None or set(left)!=set(right) or set(left)!=fields:return False
    if kind=='Relation':
        for key in fields-{'position_m','frame'}:
            if left[key]!=right[key]:return False
        if (not isinstance(left['contract'],dict) or not all(isinstance(left[key],str) and left[key]
            for key in ('target_object_id','reference_object_id','relation'))
                or left['point_role']!='representative_action_goal'):return False
    if not isinstance(left['frame'],str) or not left['frame'] or left['frame']!=right['frame']:return False
    for value in (left['position_m'],right['position_m']):
        if not isinstance(value,list) or len(value)!=3 or any(type(x) not in (int,float) for x in value):return False
        if not np.isfinite(value).all():return False
    return bool(np.all(np.abs(np.asarray(left['position_m'])-np.asarray(right['position_m']))<=1e-12))


def _place_sample_valid(*,relation,point_error,point_tolerance,speed,spin,drift,angle,fingers_open,table_valid):
    common=speed<=.01 and spin<=.1 and drift<=.005 and angle<=.12 and fingers_open
    # A relation's representative point is diagnostic, not its legal region.
    return bool(common and (relation['passed'] if relation is not None else point_error<=point_tolerance and table_valid))


def _command_sample_valid(*,error,speed,spin,drift,angle,fingers_open,bottom,support_height):
    return bool(error<=.01 and speed<=.01 and spin<=.1 and drift<=.005 and angle<=.12
                and fingers_open and abs(bottom-support_height)<=.006)


def _place_chain_complete(physical,raw_objects):
    """Require actual stage clock coverage and the recorded release/retreat chain."""
    stages={entry['stage']:entry for entry in physical['stages']}
    if len(stages)!=len(physical['stages']) or not set(range(2,11))<=set(stages):return False
    for label in range(2,11):
        entry=stages[label]
        rows=raw_objects[(raw_objects[:,0]>entry['start_tick'])&(raw_objects[:,0]<=entry['end_tick'])]
        if (not entry['reached'] or len(rows)==0 or len(rows)!=entry['steps']
                or not np.all(rows[:,1]==label)):return False
        if label not in {5,9}:
            if entry.get('tcp_error_m',float('inf'))>.003 or entry.get('orientation_error_rad',float('inf'))>.08:return False
    return bool(physical.get('release_command_tick')==stages[9]['start_tick']
                and physical.get('release_tick')==stages[9]['end_tick']==stages[10]['start_tick']
                and physical.get('retreat_end_tick')==stages[10]['end_tick']<=physical['hold_start_tick'])


def _trailing_dwell(samples):
    count=0
    for passed in reversed(samples):
        if not passed:break
        count+=1
    return count


def _valid_topology_negative(*,grasp,chain_complete,command_dwell,table_dwell,required,final_relation):
    return bool(grasp and chain_complete and command_dwell>=required and table_dwell>=required
                and final_relation is not None and final_relation['passed'] is False)


def _cloud_at(points,index):
    cloud=points if isinstance(points,np.ndarray) and points.ndim==2 else points[index]
    cloud=np.asarray(cloud,dtype=float)
    _require(cloud.ndim==2 and cloud.shape[1]==3 and len(cloud)>=4 and np.isfinite(cloud).all(),'Invalid original body geometry')
    return cloud


def _rotation(quaternion):
    quaternion=np.asarray(quaternion,dtype=float)
    _require(quaternion.shape==(4,) and np.isfinite(quaternion).all() and abs(np.linalg.norm(quaternion)-1)<1e-4,'Invalid raw object quaternion')
    w,x,y,z=quaternion/np.linalg.norm(quaternion)
    return np.asarray([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                       [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                       [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def _geometry(points,states):
    """Explicit quaternion matrix, independent of the action sample predicates."""
    centres=[];lows=[];highs=[]
    if not (isinstance(points,np.ndarray) and points.ndim==2):
        _require(len(points)==len(states),'Original body cloud count differs from raw state count')
    for index,state in enumerate(states):
        transformed=_cloud_at(points,index)@_rotation(state[3:7]).T+state[:3]
        low=transformed.min(axis=0);high=transformed.max(axis=0)
        centres.append((low+high)*.5);lows.append(low);highs.append(high)
    return np.asarray(centres),np.asarray(lows),np.asarray(highs)


def _load_body_geometry(profile,ids):
    """Independently bind archive entries to semantic IDs through original prims."""
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest_path=Path(profile['assembly_manifest_path'])
    _require(digest(manifest_path)==profile['assembly_manifest_sha256'],'Assembly manifest provenance changed')
    manifest=json.loads(manifest_path.read_text());provenance=manifest['provenance']
    points_path=Path(profile['asset_body_vertices_path'])
    _require(points_path.resolve()==Path(manifest['asset_body_vertices']).resolve(),'Body archive differs from assembly manifest')
    _require(digest(points_path)==profile['asset_body_vertices_sha256']==provenance['asset_body_vertices_sha256'],'Body point provenance changed')
    _require(digest(manifest['scene_usda'])==provenance['scene_usda_sha256'],'Assembly USD provenance changed')
    for path,expected in provenance.get('used_usd_layers_sha256',{}).items():
        _require(digest(path)==expected,f'Assembly source layer changed: {path}')
    source_state=manifest_path.parent/'object_states.npz'
    if 'object_states.npz' in manifest.get('physics',{}).get('sha256',{}):
        _require(digest(source_state)==manifest['physics']['sha256']['object_states.npz'],'Source settled state provenance changed')
    bindings=profile['assembly_object_bindings'];by_id={item['object_id']:item['prim_path'] for item in bindings}
    _require(len(by_id)==len(bindings)==len(ids) and set(by_id)==set(ids),'Body identity binding differs from raw trace')
    objects=manifest.get('objects',[{'prim_path':item['prim_path']} for item in bindings])
    by_prim={item['prim_path']:item.get('body_points_key','points_m') for item in objects}
    _require(len(by_prim)==len(objects)==len(ids) and set(by_prim)==set(by_id.values()),'Original geometry prim binding differs')
    clouds=[]
    with np.load(points_path,allow_pickle=False) as archive:
        for identity in ids:
            key=by_prim[by_id[identity]]
            _require(isinstance(key,str) and key in archive.files,'Missing original object geometry key')
            clouds.append(_cloud_at(np.asarray(archive[key],dtype=float),0))
    return manifest,clouds


def _finger_binding(robot_name,names):
    expected={'franka':('panda_finger_joint1','panda_finger_joint2'),
              'franka_robotiq':('finger_joint','right_outer_knuckle_joint')}
    _require(robot_name in expected,'Unsupported audited contact gripper')
    _require(len(set(names))==len(names) and all(name in names for name in expected[robot_name]),'Gripper measurement binding changed')
    return [names.index(name) for name in expected[robot_name]]


def _grasp_width_axis(case,physical,initial_low,initial_high,selected):
    """Derive the jaw's world cardinal axis from the recorded tool command.

    The jaw closes along hand-local Y. The only supported commands are the
    original downward hand pose and its 90-degree world-Z yaw, up to q sign.
    No production orientation planner or reported width is used to derive it.
    """
    value=case['tool_quaternion_wxyz']
    _require(isinstance(value,list) and len(value)==4 and all(type(v) in (int,float) for v in value),
             'Invalid recorded grasp tool quaternion')
    q=np.asarray(value,dtype=float);matrix=_rotation(q);q=q/np.linalg.norm(q)
    downward=np.asarray([0.,1.,0.,0.]);quarter_turn=np.asarray([0.,2**-.5,2**-.5,0.])
    same=lambda a,b:bool(np.allclose(a,b,atol=1e-6,rtol=0) or np.allclose(a,-b,atol=1e-6,rtol=0))
    _require(same(q,downward) or same(q,quarter_turn),'Grasp tool orientation is not supported yaw0/yaw90')
    jaw_axis=matrix[:,1]
    axis='x' if abs(jaw_axis[0])>1.-1e-6 else 'y'
    index=0 if axis=='x' else 1
    direction=np.asarray([1.,0.,0.] if axis=='x' else [0.,1.,0.])
    _require(np.allclose(np.abs(jaw_axis),direction,atol=1e-6,rtol=0),'Grasp jaw axis is not world cardinal')
    plan=physical.get('grasp_orientation_plan')
    if plan is None:
        _require(axis=='y','Legacy grasp without an orientation plan must retain the original Y-width convention')
    else:
        _require(isinstance(plan,dict) and type(plan.get('grip_width_axis')) is int
                 and plan['grip_width_axis']==index,'Reported grasp width axis differs from tool rotation')
        planned=np.asarray(plan.get('tool_quat_wxyz'),dtype=float)
        _require(planned.shape==(4,) and np.isfinite(planned).all() and abs(np.linalg.norm(planned)-1.)<1e-4
                 and same(planned/np.linalg.norm(planned),q),'Grasp plan quaternion differs from the actual tool command')
    width=float(initial_high[selected,index]-initial_low[selected,index])
    _require(np.isfinite(width) and width>0,'Invalid actual grasp-axis body width')
    if plan is not None:
        _require(type(plan.get('grip_width_m')) in (int,float) and np.isfinite(plan['grip_width_m'])
                 and abs(plan['grip_width_m']-width)<=1e-12,'Grasp plan width differs from raw initial body geometry')
    return axis,width


def _closing_control_width(physical,robot_name,world_width,cloud,identity,manifest,bindings):
    """Check new closing metadata against hash-bound original body geometry."""
    if robot_name!='franka_robotiq':
        return {'verified':False,'control_width_m':world_width,'policy':'panda_existing_measured_axis'}
    metadata=physical.get('closing_control_width')
    if metadata is None:
        return {'verified':False,'control_width_m':world_width,'policy':'legacy_measured_axis'}
    fields={'width_m','measured_world_axis_width_m','local_dimensions_m','asset_kind','policy'}
    _require(isinstance(metadata,dict) and set(metadata)==fields,'Invalid closing-control metadata schema')
    by_id={entry['object_id']:entry['prim_path'] for entry in bindings}
    source=[entry for entry in manifest['objects'] if entry['prim_path']==by_id[identity]]
    _require(len(source)==1,'Closing-control source prim is not unique')
    kind=source[0].get('kind')
    dimensions=np.asarray(cloud,dtype=float).max(axis=0)-np.asarray(cloud,dtype=float).min(axis=0)
    claimed=metadata['local_dimensions_m']
    _require(isinstance(claimed,list) and len(claimed)==3 and all(type(v) in (int,float) for v in claimed)
             and np.isfinite(claimed).all() and np.allclose(claimed,dimensions,atol=1e-12,rtol=0),
             'Closing-control local dimensions differ from original body cloud')
    _require(metadata['asset_kind']==kind,'Closing-control kind differs from the original manifest prim')
    square=kind=='primitive_cuboid' and dimensions[0]>.001 and abs(dimensions[0]-dimensions[1])<=1e-6
    width=min(world_width,float(min(dimensions[:2]))) if square else world_width
    policy='square_cuboid_local_face_for_closing' if square else 'measured_world_axis_width'
    _require(metadata['policy']==policy,'Closing-control policy differs from actual asset geometry')
    for key,expected in [('width_m',width),('measured_world_axis_width_m',world_width)]:
        _require(type(metadata[key]) in (int,float) and np.isfinite(metadata[key])
                 and abs(metadata[key]-expected)<=1e-12,f'Closing-control {key} differs from original geometry')
    angle=physical['finger_close_target_rad']
    _require(type(angle) in (int,float) and np.isfinite(angle) and 0.<=angle<=np.deg2rad(47.),
             'Invalid initial Robotiq close target angle')
    # Independent forward check of the monotonic USD linkage opening curve,
    # rather than importing the controller's inverse-gap implementation.
    inward=-.03716*(np.cos(angle)-1.)+.04343*np.sin(angle)
    gap=max(0.,.0870968475937844-2*inward)
    _require(abs(gap-max(.001,width-.003))<=1e-12,'Initial Robotiq close angle disagrees with control width')
    return {'verified':True,'control_width_m':width,'policy':policy,
            'local_dimensions_m':dimensions.tolist(),'source_prim_path':by_id[identity],
            'initial_close_gap_m':float(gap)}


def _finger_condition(robot_name,positions,*,closed,width_m):
    """Independent gripper-state check; does not import controller profiles."""
    q=np.asarray(positions,dtype=float)
    _require(q.shape==(2,) and np.isfinite(q).all(),'Invalid measured finger state')
    if robot_name=='franka':return bool(np.all(q<.038) if closed else np.all(q>.03))
    _require(robot_name=='franka_robotiq','Unsupported audited contact gripper')
    if not closed:return bool(np.all(q>=-.005) and np.all(q<=.04) and abs(q[0]-q[1])<=.04)
    # USD's mirrored outer joint uses gearing=-1, hence equal positive q.
    # Each pad's inward Y travel follows its independently measured joint.
    travel=-.03716*(np.cos(q)-1)+.04343*np.sin(q)
    gap=max(0.,.0870968475937844-float(travel.sum()))
    return bool(np.all(q>=.05) and np.all(q<=np.deg2rad(47)+.005)
                and abs(q[0]-q[1])<=.08 and gap<=float(width_m)+.006)


def _replay_query_gt(resolved,evidence,ticks,states,ids,points):
    """Validate current/history snapshots against raw measured clocks, then solve."""
    clock=evidence['physical_clock'];current=Snapshot.from_dict(evidence['current'])
    history={key:Snapshot.from_dict(value) for key,value in evidence.get('history',{}).items()}
    snapshots=[('Current',current,clock['current_tick'])]
    _require(set(history)==set(clock.get('history_ticks',{})),'History snapshot clocks are incomplete')
    snapshots.extend((key,value,clock['history_ticks'][key]) for key,value in history.items())
    for key,snapshot,tick in snapshots:
        _require(snapshot.instance_id==current.instance_id and snapshot.contract_hash==current.contract_hash
                 and snapshot.frame==current.frame,'Mixed query/history snapshot provenance')
        _require(key=='Current' or tick<=clock['current_tick'],'History comes after query')
        rows=np.flatnonzero(ticks==tick);_require(len(rows)==1,f'Missing measured state for snapshot: {key}')
        centres,_,_=_geometry(points,states[rows[0]])
        _require(set(snapshot.positions_m)==set(ids),'Snapshot body identities differ from measured trace')
        for index,identity in enumerate(ids):
            _require(np.allclose(centres[index],snapshot.positions_m[identity],atol=2e-7,rtol=0),f'Measured AABB centre differs from {key} evidence')
    result=solve(resolved,current,history=history)
    if resolved.episode['parameters'].get('legacy_task_spec',{}).get('protocol',{}).get('module_name')=='canonical_extension':
        from geomind.episodes.registered_tasks import check_extended_ground_truth
        independent=check_extended_ground_truth(resolved,current,result['ground_truth'],history=history)
        _require(independent['success'],'Independent canonical predicates reject replayed GT')
    return result


def _relation_context(resolved):
    params=resolved.episode['parameters'];spec=params.get('legacy_task_spec',{}).get('protocol',{}).get('gt_spec',{})
    resolver=spec.get('resolver')
    if resolver not in {'support_top_placement','containment_placement'}:return None
    key='support_object_id' if resolver=='support_top_placement' else 'container_reference_object_id'
    inverse={legacy:identity for identity,legacy in params['object_id_map'].items()}
    _require(spec.get(key) in inverse,'Missing measured topology reference binding')
    return spec,inverse[spec[key]]


def _relation_sample(contract,reference,moving,states,points,centres,low,high,*,released):
    """Recompute the legacy relation's geometry from actual poses and dimensions.

    Uses the saved relation thresholds. It does not infer containment or support
    from being on the scene's table, and does not call production predicates.
    """
    resolver=contract['resolver'];state=states[moving]
    speed=float(np.linalg.norm(state[7:10]));spin=float(np.linalg.norm(state[10:13]))
    stable=speed<=float(contract.get('max_linear_speed_after_settle',.05)) and spin<=float(contract.get('max_angular_speed_after_settle',.4))
    base=stable and (released or not contract.get('must_be_released',False))
    metrics={'resolver':resolver,'linear_speed_m_s':speed,'angular_speed_rad_s':spin,'released':bool(released)}
    if resolver=='support_top_placement':
        target_size=high[moving]-low[moving]
        overlap=np.maximum(0.,np.minimum(high[moving,:2],high[reference,:2])-np.maximum(low[moving,:2],low[reference,:2]))
        ratio=float(np.prod(overlap)/max(1e-8,float(np.prod(target_size[:2]))))
        gap=abs(float(low[moving,2]-high[reference,2]))
        above=bool(centres[moving,2]>high[reference,2]+.25*target_size[2])
        contact=gap<=float(contract.get('max_support_gap',.015)) and ratio>=float(contract.get('min_support_projection_overlap_ratio',.72))
        passed=base and (contact or not contract.get('contact_required',False)) and (above and contact or not contract.get('support_required',False))
        metrics.update(support_gap_m=gap,support_projection_overlap_ratio=ratio,object_above_support=above,
                       support_height_w_m=float(high[reference,2]))
    elif resolver=='containment_placement':
        ref_cloud=_cloud_at(points,reference);target_cloud=_cloud_at(points,moving)
        ref_center_local=(ref_cloud.min(axis=0)+ref_cloud.max(axis=0))*.5
        target_center_local=(target_cloud.min(axis=0)+target_cloud.max(axis=0))*.5
        ref_rotation=_rotation(states[reference,3:7]);target_rotation=_rotation(state[3:7])
        ref_center=states[reference,:3]+ref_rotation@ref_center_local
        target_center=state[:3]+target_rotation@target_center_local
        relative=ref_rotation.T@(target_center-ref_center)
        half=(target_cloud.max(axis=0)-target_cloud.min(axis=0))*.5
        projected=np.abs(ref_rotation.T@target_rotation)@half
        interior=np.asarray(contract.get('interior_half_extents_xy',[0.,0.]),dtype=float)
        room=interior-projected[:2]-float(contract.get('xy_inner_margin',0.))
        inside=bool(np.all(room>=0.) and np.all(np.abs(relative[:2])<=room+1e-6))
        gap=float(relative[2]-projected[2]-float(contract.get('interior_floor_top_offset',0.)))
        clearance=float(float(contract.get('opening_plane_z_offset',0.))-(relative[2]+projected[2]))
        passed=base and inside and gap<=float(contract.get('max_floor_gap',.02)) and gap>=-float(contract.get('max_floor_penetration',.004)) and clearance>=float(contract.get('min_top_clearance_below_opening',0.))
        floor_world=ref_center+ref_rotation@np.asarray([0.,0.,float(contract.get('interior_floor_top_offset',0.))])
        metrics.update(reference_local_center_m=relative.tolist(),reference_local_projected_half_extents_m=projected.tolist(),
                       xy_inside=inside,floor_gap_m=gap,top_clearance_m=clearance,support_height_w_m=float(floor_world[2]))
    else:raise ValueError('Unknown topology relation predicate')
    metrics['passed']=bool(passed)
    return metrics


def audit(root,output):
    from pxr import Usd
    from geomind.scene_assembly.usd import measure_table,supported_xy
    root=Path(root);output=Path(output)
    report=json.loads((output/'report.json').read_text());cases=[];hashes={}
    _require(report.get('results'),'Robot worker has no measured action cases')
    for case in report['results']:
        physical=case['physical'];action=physical['action']
        _require(action in {'PickLiftSelectedObject','PickPlaceObject'},'Not an assembled contact action')
        resolved=EpisodeResolver(root).resolve(case['episode_id'])
        profile=resolved.scene['physics']
        _require(profile['scene_mode']=='AssembledScene','Audit requires exact assembled-scene bindings')
        trace_path=Path(case['trace'])
        with np.load(trace_path,allow_pickle=False) as archive:trace=archive['data']
        object_path=trace_path.with_name(trace_path.stem+'_objects.npz')
        with np.load(object_path,allow_pickle=False) as archive:
            obj=archive['data'];ids=archive['object_ids'].tolist()
            object_dt=float(archive['dt_s'])
        manifest,points=_load_body_geometry(profile,ids)
        _require(Path(resolved.scene['usd_path']).resolve()==Path(manifest['scene_usda']).resolve(),'Audited USD differs from original assembly')
        with np.load(case['control_trace'],allow_pickle=False) as archive:
            controls=archive['data'];names=archive['joint_names'].tolist();limits=archive['joint_limits']
            _require(abs(float(archive['dt_s'])-object_dt)<1e-12
                     and abs(object_dt-profile['dt_s'])<1e-12,
                     'Measured/control/configured physical time steps disagree')
            _require(str(archive['instance_id'])==case['instance_id'],'Mixed robot instance')
            _require(str(archive['contract_hash'])==resolved.contract_hash,'Mixed robot contract')
        states=obj[:,2:].reshape(len(obj),len(ids),13);n=len(names)
        _require(len(trace)==len(obj)==len(controls) and len(trace)>1,'Incomplete trace lengths')
        _require(np.isfinite(states).all() and np.isfinite(controls).all(),'Nonfinite trace')
        _require(controls.shape==(len(obj),3*n+4) and limits.shape==(n,2),'Unexpected robot control schema')
        _require(len(set(ids))==len(ids),'Duplicate measured body identity')
        _require(np.allclose(np.linalg.norm(states[:,:,3:7],axis=2),1.,atol=1e-4,rtol=0),'Invalid measured quaternion trace')
        _require(np.array_equal(obj[:,0],trace[:,0]) and np.all(np.diff(obj[:,0])==1),'Physical clocks disagree')
        _require(np.all(controls[:,:n]>=limits[:,0]-1e-5) and np.all(controls[:,:n]<=limits[:,1]+1e-5),'Joint command outside limits')
        _require(np.all(controls[:,n:2*n]>=limits[:,0]-.005) and np.all(controls[:,n:2*n]<=limits[:,1]+.005),'Measured joint outside limits')
        evidence_path=output/'evidence'/f'{trace_path.stem}.json'
        evidence=json.loads(evidence_path.read_text());query_tick=evidence['physical_clock']['current_tick']
        _require(evidence['current']['instance_id']==case['instance_id'],'Query evidence belongs to a different robot instance')
        _require(abs(float(evidence['physical_clock']['dt_s'])-object_dt)<1e-12,'Query evidence physical clock changed')
        query_rows=np.flatnonzero(obj[:,0]==query_tick);_require(len(query_rows)==1,'Missing query state')
        initial=states[query_rows[0]];initial_centres,initial_low,initial_high=_geometry(points,initial)
        replay=_replay_query_gt(resolved,evidence,obj[:,0],states,ids,points)
        _require(_gt_equivalent(replay['ground_truth'],case['ground_truth']),'Runtime frozen GT replay disagrees')
        _require(_gt_equivalent(physical['frozen_ground_truth'],case['ground_truth']),'Action changed its frozen GT')
        target=np.asarray(physical['target_object_w_m']);selected=ids.index(physical['selected_object_id'])
        expected=ids.index(physical['expected_object_id']);moving=expected if action=='PickLiftSelectedObject' else selected
        if action=='PickLiftSelectedObject':
            _require(case['ground_truth']['object_id']==ids[expected],'Wrong lift GT identity')
            _require(np.allclose(target,initial_centres[expected],atol=2e-7),'Frozen lift target differs from the queried GT object')
        else:_require(np.allclose(target,case['ground_truth']['position_m'],atol=1e-12,rtol=0),'Placement GT target changed')
        robot_name=resolved.scene['robot'];fingers=_finger_binding(robot_name,names)
        _require(case.get('robot_name',robot_name)==robot_name,'Reported robot differs from the resolved embodiment')
        _require(physical['finger_joint_ids']==fingers and physical['finger_joint_names']==[names[i] for i in fingers],'Reported gripper measurement joints differ from raw trace')
        grip_axis,width=_grasp_width_axis(case,physical,initial_low,initial_high,selected)
        _require(abs(width-physical['measured_gripper_axis_width_m'])<1e-7,'Grasp width differs from original object geometry')
        closing_control=_closing_control_width(physical,robot_name,width,_cloud_at(points,selected),ids[selected],
                                                manifest,profile['assembly_object_bindings'])
        if robot_name=='franka_robotiq':
            metadata=physical['gripper_profile']
            _require(metadata['embodiment']==robot_name and metadata['joint_position_unit']=='rad','Robotiq gripper metadata has wrong embodiment or units')
            _require(metadata['command_joint_names']==['finger_joint'] and physical['command_finger_joint_names']==['finger_joint'],'Robotiq passive linkage was commanded as an independent finger')
        relation=_relation_context(resolved)
        relation_reference=ids.index(relation[1]) if relation is not None else None
        stage=Usd.Stage.Open(resolved.scene['usd_path']);table=measure_table(stage)
        hold=np.flatnonzero((obj[:,0]>physical['hold_start_tick'])&(obj[:,0]<=physical['action_end_tick']))
        _require(len(hold)>0 or not physical['success'] and physical['hold_start_tick']==physical['action_end_tick'],'Missing post-action hold samples')
        final_centres,_,_=_geometry(points,states[-1])
        good=[];command_good=[];command_table_good=[]
        command=np.asarray(physical['command_object_center_w_m'],dtype=float)
        _require(command.shape==(3,) and np.isfinite(command).all(),'Invalid commanded object centre')
        command_support=float(command[2]-(initial_high[selected,2]-initial_low[selected,2])*.5)
        last_command_error=float(np.linalg.norm(final_centres[moving]-command))
        last_error=float(np.linalg.norm(final_centres[moving]-target));last_height=float(final_centres[moving,2]-initial_centres[moving,2]);last_relation=None
        for row in hold:
            current=states[row];centres,low,high=_geometry(points,current)
            refs=[i for i in range(len(ids)) if i!=moving]
            drift=max((np.linalg.norm(centres[i]-initial_centres[i]) for i in refs),default=0.)
            angle=2*np.arccos(np.clip(np.abs(np.sum(current[refs,3:7]*initial[refs,3:7],axis=1)),0,1)).max() if refs else 0.
            speed=np.linalg.norm(current[moving,7:10]);spin=np.linalg.norm(current[moving,10:13])
            height=centres[moving,2]-initial_centres[moving,2]
            error=np.linalg.norm(centres[moving]-target)
            finger_positions=controls[row,n:2*n][fingers]
            finger_valid=_finger_condition(robot_name,finger_positions,closed=action=='PickLiftSelectedObject',width_m=width)
            if action=='PickLiftSelectedObject':
                arm_count=trace.shape[1]-6-3*len(ids)
                tcp=trace[row,3+arm_count:6+arm_count]
                valid=selected==expected and height>.06 and speed<=.01 and spin<=.1 and drift<=.005 and angle<=.12 and finger_valid and np.linalg.norm(centres[moving]-tcp)<=.08
            else:
                support=supported_xy(table,centres[moving,:2],(high[moving,:2]-low[moving,:2])*.5)
                if relation is None:
                    table_valid=abs(low[moving,2]-table['height_m'])<=.006 and support
                else:
                    last_relation=_relation_sample(relation[0],relation_reference,moving,current,points,centres,low,high,released=finger_valid)
                    # Report this diagnostic without narrowing the saved legal
                    # relation by a representative-point or support-height gate.
                    last_relation['support_height_error_m']=abs(float(low[moving,2])-last_relation['support_height_w_m'])
                    table_valid=False
                valid=_place_sample_valid(relation=last_relation,point_error=error,point_tolerance=physical['position_tolerance_m'],
                    speed=speed,spin=spin,drift=drift,angle=angle,fingers_open=finger_valid,table_valid=table_valid)
                last_command_error=float(np.linalg.norm(centres[moving]-command))
                command_valid=_command_sample_valid(error=last_command_error,speed=speed,spin=spin,drift=drift,angle=angle,
                    fingers_open=finger_valid,bottom=float(low[moving,2]),support_height=command_support)
                command_good.append(command_valid)
                command_table_good.append(bool(command_valid and abs(command_support-table['height_m'])<=1e-6
                    and abs(low[moving,2]-table['height_m'])<=.006 and support))
            good.append(bool(valid));last_error=float(error);last_height=float(height)
        dwell=_trailing_dwell(good)
        command_dwell=_trailing_dwell(command_good);command_table_dwell=_trailing_dwell(command_table_good)
        lift_rows=np.flatnonzero(obj[:,1]==6)
        lift=float(_geometry(points,states[lift_rows[-1]])[0][selected,2]-initial_centres[selected,2]) if len(lift_rows) else 0.
        grasp=bool(lift>.06)
        _require(grasp==physical['grasp_verified'],'Lift-stage end state disagrees with grasp verification')
        chain_complete=_place_chain_complete(physical,obj) if action=='PickPlaceObject' else None
        required=physical['required_dwell_steps']
        _require(required>=max(60,int(resolved.episode['physical_execution']['dwell_steps']))
                 and required*object_dt>=.5-1e-12,'Physical dwell shortened below the original half-second minimum')
        command_verified=bool(grasp and chain_complete and command_dwell>=required) if action=='PickPlaceObject' else None
        if action=='PickPlaceObject':
            if 'command_support_height_w_m' in physical:
                _require(abs(command_support-physical['command_support_height_w_m'])<=1e-12,'Command support height disagrees with raw query geometry')
            if 'command_position_error_m' in physical:
                _require(abs(last_command_error-physical['command_position_error_m'])<=1e-12,'Command position error disagrees with raw object state')
            if 'command_dwell_steps' in physical:
                _require(command_dwell==physical['command_dwell_steps'],'Command dwell disagrees with raw hold samples')
            if 'command_execution_verified' in physical:
                _require(command_verified==physical['command_execution_verified'],'Command execution outcome disagrees with raw action chain')
        negative_relation_valid=None
        if relation is not None:
            _require(action=='PickPlaceObject','Topology relation used by a non-placement action')
            if case['negative_control']:
                negative_relation_valid=_valid_topology_negative(grasp=grasp,chain_complete=chain_complete,
                    command_dwell=command_dwell,table_dwell=command_table_dwell,required=required,final_relation=last_relation)
                _require(negative_relation_valid,'Topology negative control did not execute a stable supported wrong command with false relation')
        point_negative_valid=None
        if action=='PickPlaceObject' and relation is None and case['negative_control']:
            point_negative_valid=bool(command_verified and command_table_dwell>=required
                                      and last_error>physical['position_tolerance_m'])
            _require(point_negative_valid,
                     'Point negative control did not execute a stable supported wrong command; motion failure is not semantic rejection')
        negative_pick_valid=None;negative_pick_command_audit=None
        if action=='PickLiftSelectedObject' and case['negative_control']:
            from geomind.tools.audit_negative_pick import evaluate_selected_wrong_pick
            command_physical=dict(physical)
            command_physical['_recorded_tool_quaternion_wxyz']=case['tool_quaternion_wxyz']
            arm_count=trace.shape[1]-6-3*len(ids)
            negative_pick_command_audit=evaluate_selected_wrong_pick(object_data=obj,object_ids=ids,
                body_points=points,dt_s=object_dt,query_tick=query_tick,controls=controls,joint_names=names,
                tcp_w=trace[:,3+arm_count:6+arm_count],trace_ticks=trace[:,0],trace_stages=trace[:,2],
                physical=command_physical,ground_truth=case['ground_truth'],robot_name=robot_name)
            negative_pick_valid=negative_pick_command_audit['command_execution_valid']
            _require(negative_pick_valid,'Wrong-object negative pick did not physically grasp and stably hold its selected body; motion failure is not semantic rejection')
            if 'command_execution_verified' in physical:
                _require(physical['command_execution_verified'] is True,'Reported selected-body command outcome differs from raw negative pick')
            if 'command_dwell_steps' in physical:
                _require(physical['command_dwell_steps']==negative_pick_command_audit['trailing_hold_steps'],
                         'Reported selected-body command dwell differs from raw negative pick')
        success=bool(all(stage['reached'] for stage in physical['stages']) and grasp and dwell>=physical['required_dwell_steps'])
        if relation is not None:success=bool(success and chain_complete)
        _require(success==physical['success'] and dwell==physical['dwell_steps'],
                 'Independent relation outcome or dwell disagrees; legacy centre-policy evidence needs rerun' if relation is not None
                 else 'Independent action outcome or dwell disagrees')
        for flag in ['object_pose_writes_after_reset','joint_state_writes_during_motion']:_require(physical[flag]==0,'Execution used state writes')
        _require(not physical['attachments_used'] and not physical['object_external_forces_used'],'Execution used attachment/external-force transport')
        obs_path=output/'observations'/f'{trace_path.stem}.json'
        observations=json.loads(obs_path.read_text())['observations']
        for observation in observations:
            _require(not {'ground_truth','provenance','parameters','source','semantic_id_mapping'}&observation.keys(),'Privileged data in robot observation')
            _require(all(set(item)<={'object_id','label','position_m'} for item in observation['objects']),'Internal binding in robot observation')
        for path in [trace_path,object_path,Path(case['control_trace']),evidence_path,obs_path]:
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        video=case.get('video')
        _require(isinstance(video,dict),'Missing actual robot video evidence')
        for key in ['observer','agent','frames']:
            path=Path(video[key]);actual=hashlib.sha256(path.read_bytes()).hexdigest()
            _require(actual==video['sha256'][key],f'Robot media artifact changed: {key}')
            hashes[str(path)]=actual
        frame_rows=[json.loads(line) for line in Path(video['frames']).read_text().splitlines()]
        _require(len(frame_rows)==video['frame_count'] and len(frame_rows)>1,'Video frame count disagrees with its physical metadata')
        ticks=[row['physics_tick'] for row in frame_rows]
        _require(all(b>a for a,b in zip(ticks,ticks[1:])),'Robot video clock is not monotonic')
        _require(ticks[0]>=obj[0,0] and ticks[-1]<=physical['action_end_tick']
                 and ticks[0]<=query_tick<=ticks[-1],'Robot video and measured query/action clocks disagree')
        if case.get('robot_asset_sha256'):
            _require(hashlib.sha256(Path(case['robot_asset']).read_bytes()).hexdigest()==case['robot_asset_sha256'],'Robot asset changed')
        cases.append({'episode_id':case['episode_id'],'negative_control':case['negative_control'],
                      'success':success,'dwell_steps_recomputed':dwell,'selected_lift_height_m':float(lift),
                      'expected_final_lift_height_m':last_height,'final_center_error_m':last_error,
                      'query_tick':query_tick,'raw_rows':len(trace),'joint_limits_checked':True,
                      'robot_name':robot_name,'gripper_measurement_joints':[names[i] for i in fingers],
                      'grasp_width_axis_recomputed':grip_axis,'grasp_axis_width_m_recomputed':width,
                      'grasp_tool_orientation_verified':True,
                      'closing_control_width_verification':closing_control,
                      'body_geometry_bound_by_prim':True,'history_snapshots_replayed':sorted(evidence.get('history',{})),
                      'relation_predicate_recomputed':last_relation,
                      'relation_representative_point_error_m':last_error if relation is not None else None,
                      'command_position_error_m':last_command_error if action=='PickPlaceObject' else None,
                      'command_support_height_w_m':command_support if action=='PickPlaceObject' else None,
                      'command_dwell_steps_recomputed':command_dwell if action=='PickPlaceObject' else None,
                      'command_table_dwell_steps_recomputed':command_table_dwell if action=='PickPlaceObject' else None,
                      'command_execution_verified_recomputed':command_verified,
                      'action_stage_clocks_and_release_chain_verified':chain_complete,
                      'topology_negative_control_valid':negative_relation_valid,
                      'point_negative_control_valid':point_negative_valid,
                      'negative_pick_command_valid':negative_pick_valid,
                      'negative_pick_command_audit':negative_pick_command_audit,
                      'runtime_gt_replayed':True,'public_privileged_data_separated':True})
    result={'passed':bool(cases),'mode':'independent offline replay; no physical re-execution',
            'cases':cases,'raw_sha256':hashes,'contact_free_certified':False,'model_evaluated':False}
    (output/'assembled_robot_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.run/'robot_configs',args.run/'robot_physics'),ensure_ascii=False,indent=2))
