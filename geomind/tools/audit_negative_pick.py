"""Independent execution validity for a deliberately wrong-object pick.

The original semantic score must reject the wrong identity. Separately, this
audit requires that the selected (wrong) body was really grasped and held. It
never changes or recomputes semantic GT and never advances physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from geomind.tools.audit_assembled_robot import (
    _finger_binding, _finger_condition, _geometry, _grasp_width_axis,
    _load_body_geometry, _trailing_dwell,
)


def require(condition,message):
    if not condition:raise ValueError(message)


def pick_chain_complete(physical,objects):
    entries=physical.get('stages',[]);stages={entry.get('stage'):entry for entry in entries}
    if len(stages)!=len(entries) or not set(range(2,7))<=set(stages):return False
    previous=physical.get('action_start_tick')
    for label in range(2,7):
        entry=stages[label]
        if not entry.get('reached') or entry.get('start_tick')!=previous:return False
        rows=objects[(objects[:,0]>entry['start_tick'])&(objects[:,0]<=entry['end_tick'])]
        if len(rows)==0 or len(rows)!=entry.get('steps') or not np.all(rows[:,1]==label):return False
        if label!=5 and (entry.get('tcp_error_m',float('inf'))>.003
                        or entry.get('orientation_error_rad',float('inf'))>.08):return False
        previous=entry['end_tick']
    return previous==physical.get('hold_start_tick')


def evaluate_selected_wrong_pick(*,object_data,object_ids,body_points,dt_s,query_tick,
                                 controls,joint_names,tcp_w,trace_ticks,trace_stages,
                                 physical,ground_truth,robot_name):
    """Recompute the selected-body lift and trailing hold from original arrays."""
    objects=np.asarray(object_data,dtype=float);control=np.asarray(controls,dtype=float)
    tcp=np.asarray(tcp_w,dtype=float);ids=[str(x) for x in object_ids];names=[str(x) for x in joint_names]
    require(len(ids)==len(set(ids)) and len(ids)>=2,'Invalid raw object identity order')
    require(objects.ndim==2 and objects.shape[1]==2+13*len(ids) and len(objects)>1,'Invalid object-state trace')
    require(control.ndim==2 and control.shape==(len(objects),3*len(names)+4),'Invalid raw measured control trace')
    require(tcp.shape==(len(objects),3) and np.isfinite(objects).all()
            and np.isfinite(control).all() and np.isfinite(tcp).all(),'Nonfinite or misaligned raw evidence')
    require(np.isfinite(dt_s) and dt_s>0,'Invalid measured timestep')
    ticks=objects[:,0]
    require(np.all(np.diff(ticks)==1) and np.array_equal(ticks,trace_ticks)
            and np.array_equal(objects[:,1],trace_stages),'Raw physics clocks or action labels disagree')
    require(ground_truth.get('type')=='Object' and ground_truth.get('object_id') in ids,'Expected frozen object identity GT')
    selected_id=physical.get('selected_object_id');expected_id=ground_truth['object_id']
    require(selected_id in ids and physical.get('expected_object_id')==expected_id,'Recorded selected/expected identity is invalid')
    selected=ids.index(selected_id);query_rows=np.flatnonzero(ticks==query_tick)
    require(len(query_rows)==1,'Missing exact measured query state')
    states=objects[:,2:].reshape(len(objects),len(ids),13)
    initial=states[query_rows[0]];initial_centres,initial_low,initial_high=_geometry(body_points,initial)
    finger_ids=_finger_binding(robot_name,names)
    require(physical.get('finger_joint_ids')==finger_ids
            and physical.get('finger_joint_names')==[names[i] for i in finger_ids],
            'Reported gripper joints disagree with measured control bindings')
    # Width follows the actual selected object's mesh and recorded jaw axis.
    case_for_width={'tool_quaternion_wxyz':physical['_recorded_tool_quaternion_wxyz']}
    _,width=_grasp_width_axis(case_for_width,physical,initial_low,initial_high,selected)
    hold_start=physical.get('hold_start_tick');action_end=physical.get('action_end_tick')
    require(type(hold_start) in (int,float) and type(action_end) in (int,float)
            and ticks[-1]==action_end and query_tick<=hold_start<action_end,'Invalid raw selected-body hold clock')
    hold=np.flatnonzero((ticks>hold_start)&(ticks<=action_end))
    minimum=math.ceil(.5/dt_s-1e-10)
    required=physical.get('required_dwell_steps')
    require(type(required) is int and required>=minimum,'Selected-body hold criterion shortened below half a second')
    refs=[i for i in range(len(ids)) if i!=selected]
    good=[];samples=[]
    for index in hold:
        current=states[index];centres,_,_=_geometry(body_points,current)
        height=float(centres[selected,2]-initial_centres[selected,2])
        speed=float(np.linalg.norm(current[selected,7:10]));spin=float(np.linalg.norm(current[selected,10:13]))
        drift=max((float(np.linalg.norm(centres[i]-initial_centres[i])) for i in refs),default=0.)
        if refs:
            before=initial[refs,3:7];now=current[refs,3:7]
            cosine=np.abs(np.sum(before*now,axis=1))/(np.linalg.norm(before,axis=1)*np.linalg.norm(now,axis=1))
            angle=float((2*np.arccos(np.clip(cosine,0.,1.))).max())
        else:angle=0.
        fingers=control[index,len(names):2*len(names)][finger_ids]
        closed=_finger_condition(robot_name,fingers,closed=True,width_m=width)
        distance=float(np.linalg.norm(centres[selected]-tcp[index]))
        valid=bool(height>.06 and speed<=.01 and spin<=.1 and drift<=.005
                   and angle<=.12 and closed and distance<=.08)
        good.append(valid);samples.append({'tick':int(ticks[index]),'lift_height_m':height,
            'linear_speed_m_s':speed,'angular_speed_rad_s':spin,'reference_drift_m':drift,
            'reference_orientation_error_rad':angle,'fingers_closed':bool(closed),
            'tcp_distance_m':distance,'valid':valid})
    dwell=_trailing_dwell(good)
    lift_rows=np.flatnonzero(objects[:,1]==6)
    selected_lift=float(_geometry(body_points,states[lift_rows[-1]])[0][selected,2]-initial_centres[selected,2]) if len(lift_rows) else 0.
    chain=pick_chain_complete(physical,objects)
    no_transport_intervention=(physical.get('object_pose_writes_after_reset')==0
        and physical.get('joint_state_writes_during_motion')==0
        and physical.get('attachments_used') is False
        and physical.get('object_external_forces_used') is False
        and physical.get('object_external_forces_during_robot_action') is False)
    checks={'selected_identity_is_wrong':selected_id!=expected_id,
            'grasp_stages_2_to_6_complete':chain,'reported_grasp_verified':physical.get('grasp_verified') is True,
            'selected_body_actual_lift_over_6cm':selected_lift>.06,
            'selected_body_trailing_hold_complete':dwell>=required,
            'no_pose_writes_attachment_or_external_transport':no_transport_intervention}
    tail=samples[-minimum:] if len(samples)>=minimum else samples
    return {'command_execution_valid':all(checks.values()),'selected_object_id':selected_id,
            'frozen_semantic_target_object_id':expected_id,'checks':checks,
            'selected_body_lift_at_stage6_end_m':selected_lift,'trailing_hold_steps':dwell,
            'trailing_hold_duration_s':dwell*dt_s,'required_hold_steps':required,
            'required_hold_duration_s':required*dt_s,'raw_hold_samples':len(hold),
            'hold_start_tick':hold_start,'action_end_tick':action_end,
            'final_half_second_maximums':{key:max((row[key] for row in tail),default=None)
                for key in ('linear_speed_m_s','angular_speed_rad_s','reference_drift_m',
                            'reference_orientation_error_rad','tcp_distance_m')},
            'final_selected_lift_height_m':samples[-1]['lift_height_m'] if samples else None,
            'final_fingers_closed':samples[-1]['fingers_closed'] if samples else False,
            'failed_checks':[key for key,value in checks.items() if not value],
            'scope':'Selected wrong-body actual grasp and hold, independently of semantic rejection; GT is read unchanged, never solved'}


def _inside(case,path):
    path=Path(path).resolve();require(path.is_relative_to(case) and path.is_file(),'Raw evidence path missing or outside selected case')
    return path


def audit_negative_picks(case,report=None):
    """Read original clocks, joints and hash-bound body geometry; no GPU use."""
    case=Path(case).resolve()
    if report is None:report=json.loads((case/'robot_physics/report.json').read_text())
    results=[];hashes={}
    for record in report.get('results',[]):
        if not record.get('negative_control') or record.get('physical',{}).get('action')!='PickLiftSelectedObject':continue
        item={'episode_id':record.get('episode_id'),'negative_control':True,
              'semantic_rejection_correct':record.get('semantic',{}).get('success') is False
                  and record.get('physical',{}).get('success') is False}
        try:
            identity=record['episode_id'];require(isinstance(identity,str) and identity.replace('_','').isalnum(),'Invalid episode identity')
            episode_path=_inside(case,case/'robot_configs/episodes'/f'{identity}.json')
            episode=json.loads(episode_path.read_text());scene_id=episode['scene_id']
            require(isinstance(scene_id,str) and scene_id.replace('_','').isalnum(),'Invalid scene identity')
            scene_path=_inside(case,case/'robot_configs/scenes'/f'{scene_id}.json')
            scene=json.loads(scene_path.read_text());profile=scene['physics']
            trace_path=_inside(case,record['trace']);objects_path=_inside(case,trace_path.with_name(trace_path.stem+'_objects.npz'))
            control_path=_inside(case,record['control_trace'])
            evidence_path=_inside(case,case/'robot_physics/evidence'/f'{trace_path.stem}.json')
            evidence=json.loads(evidence_path.read_text())
            require(evidence['current']['instance_id']==record['instance_id'],'Query evidence belongs to another instance')
            with np.load(objects_path,allow_pickle=False) as archive:
                objects=np.asarray(archive['data'],dtype=float);ids=archive['object_ids'].tolist();dt=float(archive['dt_s'])
            with np.load(trace_path,allow_pickle=False) as archive:
                trace=np.asarray(archive['data'],dtype=float);columns=archive['columns'].tolist()
            with np.load(control_path,allow_pickle=False) as archive:
                control=np.asarray(archive['data'],dtype=float);names=archive['joint_names'].tolist()
                require(abs(float(archive['dt_s'])-dt)<1e-12,'Control and object clocks disagree')
                require(str(archive['instance_id'])==record['instance_id'],'Control instance differs')
                require(str(archive['contract_hash'])==record['provenance']['contract_hash'],'Control contract differs')
            require(abs(float(evidence['physical_clock']['dt_s'])-dt)<1e-12,'Query physical timestep differs')
            require(len(columns)==len(set(columns)),'Duplicate trace columns')
            tcp=trace[:,[columns.index(name) for name in ('tcp_x_w','tcp_y_w','tcp_z_w')]]
            manifest,points=_load_body_geometry(profile,ids)
            physical=dict(record['physical']);physical['_recorded_tool_quaternion_wxyz']=record['tool_quaternion_wxyz']
            checked=evaluate_selected_wrong_pick(object_data=objects,object_ids=ids,body_points=points,dt_s=dt,
                query_tick=evidence['physical_clock']['current_tick'],controls=control,joint_names=names,
                tcp_w=tcp,trace_ticks=trace[:,columns.index('physics_step')],trace_stages=trace[:,columns.index('stage')],
                physical=physical,ground_truth=record['ground_truth'],robot_name=scene['robot'])
            item.update(checked)
            for path in (episode_path,scene_path,trace_path,objects_path,control_path,evidence_path):
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            item['actual_selected_command_valid']=bool(item['command_execution_valid'] and item['semantic_rejection_correct'])
        except (OSError,ValueError,KeyError,IndexError,TypeError) as exc:
            item.update(command_execution_valid=False,actual_selected_command_valid=False,failure=f'{type(exc).__name__}: {exc}')
        results.append(item)
    return {'passed':all(row.get('actual_selected_command_valid') is True for row in results),
            'applicable_negative_pick_cases':len(results),'cases':results,'raw_sha256':hashes,
            'scope':'Independent selected wrong-object command execution; separate from frozen-GT semantic rejection. Empty means not applicable to placement.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit_negative_picks(args.case)
    if args.output:args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
