"""Contact-driven Panda/Robotiq actions on the original assembled rigid bodies.

The controller writes joint targets only. The query GT is frozen before motion;
success uses measured object geometry, not end-effector arrival. Geometry and
sample predicates remain importable without Isaac Sim for CPU contract checks.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from geomind.scene_assembly.audit import measured_bounds
from geomind.scene_assembly.body_geometry import points_for
from geomind.episodes.gripper_profiles import gripper_profile


def finger_close_target_m(width_m):
    """Pinch below the measured half-width without commanding zero closure.

    The 1.5mm pad/compression allowance keeps contact preload while avoiding the
    full 20-30mm per-finger servo error caused by a zero target on a held can.
    It is a bounded Panda grasp heuristic, not a general force controller.
    """
    return gripper_profile('franka').close_target(width_m)


def grasp_tcp_height_offset_m(action, height_m):
    """Deepen only the held-object pick, bounded by measured object height.

    Short objects are not approached below their centre. The blue can permits
    the full 8mm downward offset, increasing finger coverage of its sidewall.
    Preserve the separately validated placement grasp unchanged.
    """
    return gripper_profile('franka').grasp_tcp_height_offset_m(action,height_m,width_m=.05)


def grasp_control_width(measured_axis_width_m, body_points, asset_kind):
    """Keep a rotated square block clamped if it aligns between the pads.

    The observed world AABB remains the scoring width. Only a square cuboid's
    closing command uses its local face width, because its world AABB shrinks
    as the free block aligns with the gripper. Other assets retain the observed
    width; a narrow face of a rectangular box is not a supported substitute.
    """
    measured=float(measured_axis_width_m)
    cloud=np.asarray(body_points,dtype=float)
    if not math.isfinite(measured) or measured<=0 or cloud.ndim!=2 or cloud.shape[1]!=3 or not np.isfinite(cloud).all():
        raise ValueError('Expected positive measured width and finite local body cloud')
    dimensions=np.ptp(cloud,axis=0)
    square=(asset_kind=='primitive_cuboid' and dimensions[0]>.001
            and abs(dimensions[0]-dimensions[1])<=1e-6)
    return {'width_m':min(measured,float(min(dimensions[:2]))) if square else measured,
            'measured_world_axis_width_m':measured,'local_dimensions_m':dimensions.tolist(),
            'asset_kind':asset_kind,
            'policy':'square_cuboid_local_face_for_closing' if square else 'measured_world_axis_width'}


def object_geometry(points, states):
    """Return world AABB centres/bounds from raw rigid states (wxyz, SI units)."""
    states=np.asarray(states,dtype=float)
    if states.ndim!=2 or states.shape[1]!=13 or not np.isfinite(states).all():
        raise ValueError('Expected one finite 13-value rigid state per assembled object')
    norms=np.linalg.norm(states[:,3:7],axis=1)
    if not np.allclose(norms,1,atol=1e-4):raise ValueError('Invalid measured object quaternion')
    if isinstance(points,(list,tuple)):
        shared=bool(len(points) and np.asarray(points[0]).ndim==1)
        if not shared and len(points)!=len(states):raise ValueError('Body cloud/state count mismatch')
    elif isinstance(points,np.ndarray) and points.ndim==3 and len(points)!=len(states):
        raise ValueError('Body cloud/state count mismatch')
    bounds=[measured_bounds(points_for(points,index),row[:3],row[3:7]) for index,row in enumerate(states)]
    low=np.asarray([pair[0] for pair in bounds]);high=np.asarray([pair[1] for pair in bounds])
    return (low+high)*.5,low,high


def sample_metrics(points, states, initial_centres, initial_quaternions, moving_index, table_height, target):
    """Measure an object and all preserved references without using the TCP."""
    states=np.asarray(states,dtype=float)
    centres,low,high=object_geometry(points,states)
    initial=np.asarray(initial_centres,dtype=float)
    initial_q=np.asarray(initial_quaternions,dtype=float)
    if initial.shape!=centres.shape or initial_q.shape!=(len(states),4):
        raise ValueError('Initial reference state shape mismatch')
    refs=[i for i in range(len(states)) if i!=moving_index]
    rotation_errors=2*np.arccos(np.clip(np.abs(np.sum(states[:,3:7]*initial_q,axis=1)),0,1))
    return {'center_w_m':centres[moving_index].tolist(),
            'bbox_min_w_m':low[moving_index].tolist(),'bbox_max_w_m':high[moving_index].tolist(),
            'lift_height_m':float(centres[moving_index,2]-initial[moving_index,2]),
            'position_error_m':float(np.linalg.norm(centres[moving_index]-np.asarray(target,dtype=float))),
            'linear_speed_m_s':float(np.linalg.norm(states[moving_index,7:10])),
            'angular_speed_rad_s':float(np.linalg.norm(states[moving_index,10:13])),
            'support_height_error_m':abs(float(low[moving_index,2])-float(table_height)),
            'reference_max_displacement_m':max((float(np.linalg.norm(centres[i]-initial[i])) for i in refs),default=0.),
            'reference_max_orientation_error_rad':max((float(rotation_errors[i]) for i in refs),default=0.)}


def lift_sample_valid(metrics, *, correct_object, fingers_closed, gripper_distance_m):
    return bool(correct_object and metrics['lift_height_m']>.06 and metrics['linear_speed_m_s']<=.01
                and metrics['angular_speed_rad_s']<=.1 and metrics['reference_max_displacement_m']<=.005
                and metrics['reference_max_orientation_error_rad']<=.12 and fingers_closed
                and gripper_distance_m<=.08)


def placement_sample_valid(metrics, *, fingers_open, support_samples_passed, position_tolerance_m=.01):
    return bool(metrics['position_error_m']<=position_tolerance_m and metrics['linear_speed_m_s']<=.01
                and metrics['angular_speed_rad_s']<=.1 and metrics['support_height_error_m']<=.006
                and metrics['reference_max_displacement_m']<=.005
                and metrics['reference_max_orientation_error_rad']<=.12 and fingers_open and support_samples_passed)


def post_lift_grip_policy(protocol, *, robot_name, asset_kind, action, initial_close_target):
    """Validate an explicit private position-hold ablation; default stays legacy.

    This selects a target policy only. It neither changes stiffness/materials
    nor claims that a grasp has physically passed its unchanged criteria.
    """
    if not isinstance(protocol, dict):
        raise ValueError('Private task protocol must be an object')
    policy=protocol.get('grip_hold_policy', 'mass_spring_estimate')
    if not isinstance(policy, str) or policy not in {'mass_spring_estimate', 'retain_initial_close_target'}:
        raise ValueError('Unknown private grip_hold_policy')
    if policy=='mass_spring_estimate':
        return {'policy':policy,'target':None,'explicit_opt_in':False,
                'controller':('existing post-lift measured mass/spring estimate' if robot_name=='franka_robotiq' else 'existing asset-specific close target')}
    if (protocol.get('module_name')!='canonical_extension' or robot_name!='franka_robotiq'
            or asset_kind!='primitive_cuboid' or action!='PickLiftSelectedObject'):
        raise ValueError('retain_initial_close_target is scoped to canonical Robotiq cuboid picks')
    if isinstance(initial_close_target, bool) or not isinstance(initial_close_target, (int,float)) or not math.isfinite(initial_close_target) or initial_close_target<=0:
        raise ValueError('Expected the original finite positive position close target')
    return {'policy':policy,'target':float(initial_close_target),'target_rad':float(initial_close_target),
            'explicit_opt_in':True,'controller':'retain the original grasp/lift position target',
            'command_unit':'rad','gripper_stiffness_changed':False,
            'force_model_applied':False,'direct_force_command':False,
            'physical_hold_verified':False}


def execute_assembled_action(env, answer):
    """Execute a bounded real pick/lift or pick/place in the current episode."""
    from isaaclab.utils.math import quat_apply,subtract_frame_transforms
    from geomind.scene_assembly.usd import supported_xy

    gripper=gripper_profile(env.robot_name)
    cfg=env.resolved.episode['physical_execution'];action=cfg['action']
    if action not in {'PickLiftSelectedObject','PickPlaceObject'}:raise ValueError('Unsupported assembled action')
    if env.targets:raise ValueError('Assembled static robot action cannot run with external-force event targets')
    torch=env.torch;points=env.assembled_body_points;table=env.assembled_table
    gt=deepcopy(env.result['ground_truth']);pick=action=='PickLiftSelectedObject'
    calculation=env.result['provenance']['calculation']
    relation=calculation.get('relation_contract')
    reference_index=(env.object_index[calculation['relation_reference_object']] if relation else None)
    expected_id=gt['object_id'] if pick else cfg['moving_object']
    selected_id=answer['object_id'] if pick else cfg['moving_object']
    if selected_id not in env.object_index or expected_id not in env.object_index:
        raise ValueError('Action references an unbound assembled object')
    selected=env.object_index[selected_id];expected=env.object_index[expected_id]
    fingers,finger_names=env.robot.find_joints(list(gripper.command_joint_names),preserve_order=True)
    measured_fingers,measured_names=env.robot.find_joints(list(gripper.measurement_joint_names),preserve_order=True)
    if tuple(finger_names)!=gripper.command_joint_names or tuple(measured_names)!=gripper.measurement_joint_names:
        raise ValueError('Gripper joint topology differs from the asset-specific contact profile')
    tick_ratio=(1/120)/env.dt
    budget=int(cfg['max_steps']);required=max(60,int(cfg['dwell_steps']),math.ceil(.5/env.dt))
    canonical_extension=(env.resolved.episode['parameters'].get('legacy_task_spec',{})
                         .get('protocol',{}).get('module_name')=='canonical_extension')
    # The new multi-stage fixtures can still be converging when the old
    # per-waypoint 6.67s cap expires, despite a large unused overall budget.
    # Permit up to 12.5s for each waypoint; total steps and worker deadline,
    # measured arrival tolerances and all grasp/placement predicates stay fixed.
    stage_step_limit=round((1500 if canonical_extension else 800)*tick_ratio)
    if budget<required+200:raise ValueError('Assembled action budget leaves no time for grasp and verification')
    position_tolerance=float(cfg['position_tolerance_m'])
    clearance=max(.10,float(cfg['approach_clearance_m']))
    initial_tick=env.physics_step_index;stages=[];joint_limits_ok=True;max_joint_limit_violation=0.

    def raw_states():
        return np.asarray([body.data.root_state_w[0].detach().cpu().numpy() for body in env.bodies],dtype=float)

    initial_states=raw_states();initial_centres,initial_low,initial_high=object_geometry(points,initial_states)
    initial_quaternions=initial_states[:,3:7].copy()
    size=initial_high[selected]-initial_low[selected]
    env.tool_quaternion=torch.tensor([[0.,1.,0.,0.]],dtype=env.robot.data.joint_pos.dtype,device=env.device)
    grasp_plan=None;width_axis=1
    if pick and env.robot_name=='franka_robotiq':
        from geomind.scene_assembly.grasp_orientation import plan_grasp_orientation
        grasp_plan=plan_grasp_orientation(initial_centres,initial_low,initial_high,selected,
            open_gripper_half_xy_m=[.015263,.076449],canonical_clearance_margin_m=.01)
        grasp_plan['footprint_model']={'source':'measured open Robotiq collision hull in panda_hand Z band [0.10,0.17]m',
            'scope':'conservative rectangular planning envelope; not a contact-free certificate'}
        width_axis=grasp_plan['grip_width_axis']
        env.tool_quaternion=torch.tensor([grasp_plan['tool_quat_wxyz']],dtype=env.robot.data.joint_pos.dtype,device=env.device)
    width=float(size[width_axis])
    if width>gripper.max_object_width_m or not .015<=size[2]<=.16:
        raise ValueError(f'Asset dimensions unsupported by this {env.robot_name} top grasp: {size.tolist()}')
    source_object=next(item for item in env.assembled_manifest['objects']
                       if item['prim_path']==env.assembled_bindings[selected]['prim_path'])
    closing_width=grasp_control_width(width,points_for(points,selected),source_object.get('kind'))
    control_width=closing_width['width_m'] if env.robot_name=='franka_robotiq' else width
    grip=gripper.close_target(control_width);open_grip=gripper.open_target
    canonical_contact_height=(env.robot_name=='franka_robotiq'
        and source_object.get('kind')=='primitive_cuboid'
        and env.resolved.episode['parameters'].get('legacy_task_spec',{}).get('protocol',{}).get('module_name')=='canonical_extension')
    grasp_height_offset=gripper.grasp_tcp_height_offset_m(action,float(size[2]),width_m=control_width,
        asset_kind=source_object.get('kind'),contact_angle_geometry=canonical_contact_height)
    hold_policy=post_lift_grip_policy(
        env.resolved.episode['parameters'].get('legacy_task_spec',{}).get('protocol',{}),
        robot_name=env.robot_name,asset_kind=source_object.get('kind'),action=action,initial_close_target=grip)
    frozen_target=initial_centres[expected].copy() if pick else np.asarray(gt['position_m'],dtype=float)
    command=initial_centres[selected].copy() if pick else np.asarray(answer['position_m'],dtype=float)
    if not pick and answer.get('frame')!=env.resolved.layout['frame']:raise ValueError('Placement command frame mismatch')
    target_tensor=torch.tensor([command],dtype=env.robot.data.joint_pos.dtype,device=env.device)

    def remaining():return budget-(env.physics_step_index-initial_tick)

    def tick(stage):
        nonlocal joint_limits_ok,max_joint_limit_violation
        if remaining()<=0:return False
        env._physics_tick(label=stage)
        actual=env.robot.data.joint_pos
        limits=env.robot.data.soft_joint_pos_limits
        violation=float(torch.maximum(limits[:,:,0]-actual,actual-limits[:,:,1]).clamp(min=0).max().item())
        max_joint_limit_violation=max(max_joint_limit_violation,violation)
        joint_limits_ok=joint_limits_ok and violation<=.005
        return True

    def move(goal,opening,stage,dwell_required=12):
        dwell_required=round(dwell_required*tick_ratio)
        env._joint_target[:,fingers]=opening
        hand_goal=goal-quat_apply(env.tool_quaternion,env.tcp_offset)
        root=env.robot.data.root_state_w[:,:7]
        pos,quat=subtract_frame_transforms(root[:,:3],root[:,3:7],hand_goal,env.tool_quaternion)
        env.controller.set_command(torch.cat((pos,quat),dim=-1))
        start=env.physics_step_index;dwell=0;error=float('inf');angle=float('inf')
        for _ in range(min(stage_step_limit,remaining())):
            hand=env.robot.data.body_state_w[:,env.hand_id,:7]
            pos,quat=subtract_frame_transforms(root[:,:3],root[:,3:7],hand[:,:3],hand[:,3:7])
            jacobian=env.robot.root_physx_view.get_jacobians()[:,env.hand_id-1][:,:,env.arm_ids]
            q=env.robot.data.joint_pos[:,env.arm_ids]
            desired=env.controller.compute(pos,quat,jacobian,q)
            desired=q+(desired-q).clamp(-cfg['joint_step_limit_rad'],cfg['joint_step_limit_rad'])
            limits=env.robot.data.soft_joint_pos_limits[:,env.arm_ids]
            env._joint_target[:,env.arm_ids]=torch.maximum(torch.minimum(desired,limits[:,:,1]),limits[:,:,0])
            if not tick(stage):break
            actual,ori=env._tcp_pose()
            error=float(torch.linalg.vector_norm(actual-goal).item())
            angle=2*math.acos(float((ori*env.tool_quaternion).sum().abs().clamp(0,1).item()))
            dwell=dwell+1 if error<=.003 and angle<=.08 else 0
            if dwell>=dwell_required:break
        stages.append({'stage':stage,'start_tick':start,'end_tick':env.physics_step_index,
                       'steps':env.physics_step_index-start,'tcp_error_m':error if math.isfinite(error) else None,
                       'orientation_error_rad':angle if math.isfinite(angle) else None,'reached':dwell>=dwell_required})
        return dwell>=dwell_required

    def finger_hold(opening,stage,steps=100):
        steps=round(steps*tick_ratio)
        start=env.physics_step_index;env._joint_target[:,fingers]=opening
        for _ in range(min(steps,remaining())):tick(stage)
        stages.append({'stage':stage,'start_tick':start,'end_tick':env.physics_step_index,
                       'steps':env.physics_step_index-start,'reached':env.physics_step_index-start==steps})
        return env.physics_step_index-start==steps

    start=torch.tensor([initial_centres[selected]],dtype=env.robot.data.joint_pos.dtype,device=env.device)
    high=start.clone();high[:,2]+=clearance
    tcp,_=env._tcp_pose();lift=tcp.clone();lift[:,2]=max(float(tcp[0,2]),float(high[0,2]))
    motion_ok=move(lift,open_grip,2) and move(high,open_grip,3)
    grasp=start.clone();grasp[:,2]+=grasp_height_offset
    if motion_ok:motion_ok=move(grasp,open_grip,4)
    grasp_start_tick=env.physics_step_index
    if motion_ok:motion_ok=finger_hold(grip,5) and move(high,grip,6)
    lifted=object_geometry(points,raw_states())[0][selected,2]-initial_centres[selected,2]
    grasp_verified=bool(lifted>.06)
    release_tick=None;release_command_tick=None;retreat_end_tick=None;transport_feedback=None

    if not pick and motion_ok and grasp_verified:
        centres,_,_=object_geometry(points,raw_states());tcp,_=env._tcp_pose()
        offset=torch.tensor([centres[selected]],dtype=tcp.dtype,device=env.device)-tcp
        above=target_tensor-offset;above[:,2]+=clearance
        motion_ok=move(above,grip,7)
        if motion_ok:
            centres,_,_=object_geometry(points,raw_states());tcp,_=env._tcp_pose()
            updated=torch.tensor([centres[selected]],dtype=tcp.dtype,device=env.device)-tcp
            transport_feedback={'tick':env.physics_step_index,'initial_object_to_tcp_offset_m':offset[0].detach().cpu().tolist(),
                                'measured_object_to_tcp_offset_m':updated[0].detach().cpu().tolist()}
            offset=updated
        place=target_tensor-offset;place[:,2]+=.002
        if transport_feedback is not None:transport_feedback['descent_tcp_goal_w_m']=place[0].detach().cpu().tolist()
        if motion_ok:motion_ok=move(place,grip,8)
        release_command_tick=env.physics_step_index
        opened=finger_hold(open_grip,9)
        release_tick=env.physics_step_index
        retreat=place.clone();retreat[:,2]+=clearance
        retreat_ok=move(retreat,open_grip,10) if opened else False
        retreat_end_tick=env.physics_step_index
        motion_ok=bool(motion_ok and opened and retreat_ok)
    elif not pick:
        # An unsuccessful grasp remains failed. Opening the fingers still makes
        # the retained failure safe to inspect without attaching the object.
        env._joint_target[:,fingers]=open_grip

    hold_start=env.physics_step_index;stable=0;samples=[];last=None;support_ok=False
    command_stable=0;command_error=float('inf')
    selected_command_stable=0;selected_command_samples=[];selected_command_last=None
    command_support_height=float(command[2]-.5*size[2]) if not pick else None
    hold_commands=[];hold_drive=None
    if pick and motion_ok and grasp_verified and hold_policy['policy']=='retain_initial_close_target':
        env._joint_target[:,fingers]=hold_policy['target']
        hold_commands.append({'tick':env.physics_step_index,'plan':deepcopy(hold_policy)})
    if pick and motion_ok and grasp_verified and env.robot_name=='franka_robotiq' and hold_policy['policy']=='mass_spring_estimate':
        aux_ids,aux_names=env.robot.find_joints(['left_inner_finger_joint','right_inner_finger_joint'],preserve_order=True)
        stiffness=torch.as_tensor(env.robot.root_physx_view.get_dof_stiffnesses(),device=env.device)
        master_stiffness=float(stiffness[0,fingers[0]].item())
        aux_stiffness=stiffness[0,aux_ids]
        mass=float(env.bodies[selected].data.default_mass[0].sum().item())
        hold_drive={'master_stiffness_nm_per_rad':master_stiffness,
                    'aux_joint_names':list(aux_names),'aux_stiffness_nm_per_rad':aux_stiffness.detach().cpu().tolist(),
                    'body_mass_kg':mass,'update_every_ticks':round(10*tick_ratio),
                    'control':'position targets using measured joint spring bias; no force sensor',
                    'activation':'after successful physical lift only'}
    for hold_index in range(min(max(round(120*tick_ratio),2*required),remaining())):
        if hold_drive is not None and hold_index%hold_drive['update_every_ticks']==0:
            measured_angle=float(env.robot.data.joint_pos[0,measured_fingers].mean().item())
            aux_error=(env.robot.data.joint_pos[0,aux_ids]-env._joint_target[0,aux_ids]).abs()
            aux_torque=float((aux_stiffness*aux_error).sum().item())
            plan=gripper.hold_target_plan(control_width,mass,measured_outer_angle_rad=measured_angle,
                                          drive_stiffness=master_stiffness,aux_opening_torque_nm=aux_torque)
            env._joint_target[:,fingers]=plan['target']
            hold_commands.append({'tick':env.physics_step_index,'plan':plan})
        if not tick(12 if pick else 11):break
        states=raw_states()
        # Negative pick control must be evaluated on the GT object, not the
        # object chosen by the wrong command.
        score_index=expected if pick else selected
        metrics=sample_metrics(points,states,initial_centres,initial_quaternions,score_index,table['height_m'],frozen_target)
        finger_positions=env.robot.data.joint_pos[0,measured_fingers].detach().cpu().tolist()
        finger_condition=(gripper.fingers_closed(finger_positions,width_m=width) if pick
                          else gripper.fingers_open(finger_positions))
        tcp,_=env._tcp_pose();gripper_distance=float(np.linalg.norm(np.asarray(metrics['center_w_m'])-tcp[0].detach().cpu().numpy()))
        if pick:
            valid=lift_sample_valid(metrics,correct_object=selected_id==expected_id,
                                    fingers_closed=finger_condition,gripper_distance_m=gripper_distance)
            # Preserve frozen-GT semantic scoring above. Independently verify
            # whether the selected command really lifted and held its body.
            selected_command_last=(metrics if selected==score_index else
                sample_metrics(points,states,initial_centres,initial_quaternions,selected,table['height_m'],command))
            selected_distance=float(np.linalg.norm(np.asarray(selected_command_last['center_w_m'])-tcp[0].detach().cpu().numpy()))
            selected_valid=lift_sample_valid(selected_command_last,correct_object=True,
                fingers_closed=finger_condition,gripper_distance_m=selected_distance)
            selected_command_stable=selected_command_stable+1 if selected_valid and joint_limits_ok else 0
            selected_command_samples.append([env.physics_step_index,selected_command_last['lift_height_m'],
                selected_command_last['linear_speed_m_s'],selected_command_last['angular_speed_rad_s'],
                selected_command_last['reference_max_displacement_m'],selected_command_last['reference_max_orientation_error_rad'],
                int(finger_condition),selected_distance,selected_command_stable])
        else:
            low=np.asarray(metrics['bbox_min_w_m']);high=np.asarray(metrics['bbox_max_w_m'])
            if relation:
                from .assembled_relations import measure_relation
                centres,bottoms,tops=object_geometry(points,states)
                relation_result=measure_relation(relation,reference_index,selected,points,states,centres,bottoms,tops,
                                                 released=finger_condition)
                support_ok=relation_result['passed']
                metrics['support_height_error_m']=abs(float(low[2])-relation_result['support_height_w_m'])
                valid=bool(support_ok and finger_condition and metrics['linear_speed_m_s']<=.01
                           and metrics['angular_speed_rad_s']<=.1 and metrics['reference_max_displacement_m']<=.005
                           and metrics['reference_max_orientation_error_rad']<=.12)
            else:
                support_ok=supported_xy(table,metrics['center_w_m'][:2],(high[:2]-low[:2])*.5)
                valid=placement_sample_valid(metrics,fingers_open=finger_condition,support_samples_passed=support_ok,
                                              position_tolerance_m=position_tolerance)
            command_error=float(np.linalg.norm(np.asarray(metrics['center_w_m'])-command))
            command_valid=(command_error<=position_tolerance and finger_condition
                           and metrics['linear_speed_m_s']<=.01 and metrics['angular_speed_rad_s']<=.1
                           and metrics['reference_max_displacement_m']<=.005
                           and metrics['reference_max_orientation_error_rad']<=.12
                           and abs(float(low[2])-command_support_height)<=.006 and joint_limits_ok)
            command_stable=command_stable+1 if command_valid else 0
        stable=stable+1 if valid and joint_limits_ok else 0
        last=metrics
        samples.append([env.physics_step_index,metrics['position_error_m'],metrics['lift_height_m'],
                        metrics['linear_speed_m_s'],metrics['angular_speed_rad_s'],metrics['support_height_error_m'],
                        metrics['reference_max_displacement_m'],metrics['reference_max_orientation_error_rad'],
                        int(finger_condition),gripper_distance,stable])
    if last is None:
        score_index=expected if pick else selected
        last=sample_metrics(points,raw_states(),initial_centres,initial_quaternions,score_index,table['height_m'],frozen_target)
    success=bool(motion_ok and grasp_verified and joint_limits_ok and stable>=required)
    report={'action':action,'success':success,'expected_object_id':expected_id,'selected_object_id':selected_id,
            'moving_object_id':selected_id,'initial_centers_w_m':initial_centres.tolist(),
            'initial_root_states_w':initial_states.tolist(),'initial_object_ids':list(env.ids),
            'frozen_ground_truth':gt,'target_object_w_m':frozen_target.tolist(),
            'command_object_center_w_m':command.tolist(),'final_object_w_m':last['center_w_m'],
            'bbox_min_w_m':last['bbox_min_w_m'],'bbox_max_w_m':last['bbox_max_w_m'],
            'geometry_reference':'World-axis-aligned mesh bbox center, not rigid root or mass centroid',
            'position_error_m':max(0.,.06-last['lift_height_m']) if pick else last['position_error_m'],
            'position_tolerance_m':position_tolerance,'orientation_tolerance_rad':cfg['orientation_tolerance_rad'],
            'grasp_verified':grasp_verified,'lift_height_m':float(lifted),'expected_lift_height_m':last['lift_height_m'],
            'dwell_steps':stable,'required_dwell_steps':required,'stages':stages,
            'motion_steps':env.physics_step_index-initial_tick,'max_steps':budget,
            'per_waypoint_step_limit':stage_step_limit,
            'action_start_tick':initial_tick,'action_end_tick':env.physics_step_index,
            'grasp_start_tick':grasp_start_tick,'hold_start_tick':hold_start,
            'release_command_tick':release_command_tick,'release_tick':release_tick,'retreat_end_tick':retreat_end_tick,
            'sample_columns':['tick','position_error_m','lift_height_m','linear_speed_m_s','angular_speed_rad_s',
                              'support_height_error_m','reference_max_displacement_m','reference_max_orientation_error_rad',
                              'fingers_condition','gripper_distance_m','stable_steps'],
            'hold_samples' if pick else 'post_release_samples':samples,
            'reference_max_displacement_m':last['reference_max_displacement_m'],
            'reference_max_orientation_error_rad':last['reference_max_orientation_error_rad'],
            'support_height_error_m':last['support_height_error_m'],'footprint_support_samples_passed':support_ok if not pick else None,
            'finger_joint_ids':list(measured_fingers),'finger_joint_names':list(measured_names),
            'command_finger_joint_ids':list(fingers),'command_finger_joint_names':list(finger_names),
            'gripper_profile':gripper.as_dict(),'gripper_open_target':open_grip,'gripper_close_target':grip,
            'finger_close_target_m':grip if gripper.joint_position_unit=='m' else None,
            'finger_close_target_rad':grip if gripper.joint_position_unit=='rad' else None,
            'measured_gripper_axis_width_m':width,'grasp_orientation_plan':grasp_plan,
            'closing_control_width':closing_width,
            'post_lift_grip_drive':hold_drive,'post_lift_grip_commands':hold_commands,
            'post_lift_grip_policy':deepcopy(hold_policy),
            'transport_offset_feedback':transport_feedback,
            'grasp_tcp_height_offset_m':grasp_height_offset,
            'grasp_height_geometry':'zero_preload_contact' if canonical_contact_height else 'legacy_preload_command',
            'joint_limits_verified':joint_limits_ok,'max_measured_joint_limit_violation':max_joint_limit_violation,
            'object_pose_writes_after_reset':0,'joint_state_writes_during_motion':0,
            'attachments_used':False,'object_external_forces_used':False,
            'object_external_forces_during_robot_action':False,'target_policy':'query_time_frozen',
            'termination_reason':'correct_object_held' if success and pick else 'stable_released_object' if success else 'assembled_action_failure',
            'collision_check':'Physical contact grasp; robot contact-free path is not certified'}
    if relation:
        report.update(relation_contract=relation,relation_reference_object=calculation['relation_reference_object'],
                      relation_result=locals().get('relation_result'))
    if pick:
        if selected_command_last is None:
            selected_command_last=sample_metrics(points,raw_states(),initial_centres,initial_quaternions,selected,table['height_m'],command)
        report.update(command_selected_object_id=selected_id,
            command_final_object_w_m=selected_command_last['center_w_m'],
            command_lift_height_m=selected_command_last['lift_height_m'],
            command_dwell_steps=selected_command_stable,
            command_execution_verified=bool(motion_ok and grasp_verified and joint_limits_ok and selected_command_stable>=required),
            command_hold_sample_columns=['tick','selected_lift_height_m','linear_speed_m_s','angular_speed_rad_s',
                'reference_max_displacement_m','reference_max_orientation_error_rad','fingers_closed','tcp_distance_m','stable_steps'],
            command_hold_samples=selected_command_samples,
            semantic_target_scoring_preserved=True)
    if not pick:
        report.update(command_support_height_w_m=command_support_height,
                      command_position_error_m=float(np.linalg.norm(np.asarray(last['center_w_m'])-command)),
                      command_dwell_steps=command_stable,
                      command_execution_verified=bool(motion_ok and grasp_verified and joint_limits_ok
                          and release_tick is not None and retreat_end_tick is not None and command_stable>=required))
    tcp,_=env._tcp_pose();report['actual_tcp_w_m']=tcp[0].detach().cpu().tolist()
    return report
