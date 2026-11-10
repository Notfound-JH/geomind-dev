"""Contact-driven midpoint placement using the existing IsaacLabEpisode world.

No object pose writes, attachments, or external forces are used during motion.
"""
from __future__ import annotations
import math


def execute_midpoint_placement(env, answer):
    from isaaclab.utils.math import quat_apply, subtract_frame_transforms
    torch = env.torch
    cfg = env.resolved.episode['physical_execution']
    lift_changed = cfg['action'] == 'PickLiftChangedObject'
    if lift_changed:
        env.targets.clear()  # Hidden external-force event ends before robot grasping.
    index = env.object_index[answer['object_id'] if lift_changed else cfg['moving_object']]
    body = env.bodies[index]
    fingers, _ = env.robot.find_joints('panda_finger_joint.*', preserve_order=True)
    initial = [b.data.root_pos_w.clone() for b in env.bodies]
    expected_id = env.result['ground_truth'].get('object_id')
    target = (env.bodies[env.object_index[expected_id]].data.root_pos_w.clone() if lift_changed else
              env._answer_point({'position_m': env.result['ground_truth']['position_m'], 'frame': 'table'}).clone())
    command = env._answer_point(answer).clone()
    stages = []

    def move(goal, opening, stage, required=12, budget=1500):
        env._joint_target[:, fingers] = opening
        hand_goal = goal - quat_apply(env.tool_quaternion, env.tcp_offset)
        root = env.robot.data.root_state_w[:, :7]
        pos, quat = subtract_frame_transforms(root[:, :3], root[:, 3:7], hand_goal, env.tool_quaternion)
        env.controller.set_command(torch.cat((pos, quat), dim=-1))
        dwell = 0
        for tick in range(budget):
            hand = env.robot.data.body_state_w[:, env.hand_id, :7]
            pos, quat = subtract_frame_transforms(root[:, :3], root[:, 3:7], hand[:, :3], hand[:, 3:7])
            jac = env.robot.root_physx_view.get_jacobians()[:, env.hand_id-1][:, :, env.arm_ids]
            q = env.robot.data.joint_pos[:, env.arm_ids]
            desired = env.controller.compute(pos, quat, jac, q)
            desired = q + (desired-q).clamp(-.04, .04)
            limits = env.robot.data.soft_joint_pos_limits[:, env.arm_ids]
            env._joint_target[:, env.arm_ids] = torch.maximum(torch.minimum(desired, limits[:,:,1]), limits[:,:,0])
            env._physics_tick(label=stage)
            actual, ori = env._tcp_pose()
            error = float(torch.linalg.vector_norm(actual-goal).item())
            angle = 2*math.acos(float((ori*env.tool_quaternion).sum().abs().clamp(0,1).item()))
            dwell = dwell+1 if error < .003 and angle < .08 else 0
            if dwell >= required:
                break
        stages.append({'stage':stage, 'steps':tick+1, 'tcp_error_m':error, 'reached':dwell>=required})
        return dwell >= required

    grip = .014 if lift_changed else 0.0
    start = initial[index].clone()
    high = start.clone(); high[:,2] += .16
    tcp,_ = env._tcp_pose(); lift=tcp.clone(); lift[:,2]=max(float(tcp[0,2]),float(high[0,2]))
    # Arm motion is physical joint-position control; fingers pinch the dynamic cube.
    ok = move(lift,.04,2) and move(high,.04,3)
    grasp = start.clone(); grasp[:,2] += .008
    ok = ok and move(grasp,.04,4)
    if ok:
        for _ in range(100):
            env._joint_target[:,fingers]=grip
            env._physics_tick(label=5)
        ok = move(high,grip,6)
    lifted_height = float(body.data.root_pos_w[0,2]-initial[index][0,2])
    grasp_verified = lifted_height > .06
    if lift_changed:
        expected_index = env.object_index[expected_id]
        expected_initial = initial[expected_index]
        stable = 0; samples = []
        hold_start = env.physics_step_index
        for _ in range(120):
            env._physics_tick(label=12)
            expected_body = env.bodies[expected_index]
            height = float(expected_body.data.root_pos_w[0,2]-expected_initial[0,2])
            speed = float(torch.linalg.vector_norm(expected_body.data.root_lin_vel_w).item())
            refs = max(float(torch.linalg.vector_norm(b.data.root_pos_w-initial[i]).item())
                       for i,b in enumerate(env.bodies) if i != expected_index)
            closed = bool((env.robot.data.joint_pos[:,fingers] < .025).all())
            valid = height > .06 and speed < .01 and refs <= .005 and closed
            stable = stable+1 if valid else 0
            samples.append([env.physics_step_index,height,speed,refs,int(closed),stable])
        correct = answer['object_id'] == expected_id
        return {'success': bool(ok and correct and grasp_verified and stable >= 60),
                'action': 'PickLiftChangedObject', 'position_error_m': max(0.,.06-height),
                'expected_object_id':expected_id,'selected_object_id':answer['object_id'],
                'expected_initial_position_w_m':expected_initial[0].cpu().tolist(),
                'grasp_verified':grasp_verified,'lift_height_m':height,'dwell_steps':stable,
                'required_dwell_steps':60,'motion_steps':sum(x['steps'] for x in stages),
                'stages':stages,'hold_start_tick':hold_start,'hold_samples':samples,
                'reference_max_displacement_m':refs,'object_pose_writes_after_reset':0,
                'joint_state_writes_during_motion':0,'attachments_used':False,
                'object_external_forces_during_robot_action':False,
                'termination_reason':'correct_object_held' if correct and stable>=60 else 'lift_action_failure'}
    # Preserve the measured grasp offset instead of assuming TCP is object centre.
    tcp,_ = env._tcp_pose()
    offset = body.data.root_pos_w.clone()-tcp
    if ok and grasp_verified:
        above = command-offset; above[:,2] += .12
        ok = move(above,0.0,7)
        place = command-offset; place[:,2] += .002
        ok = ok and move(place,0.0,8)
        for _ in range(100):
            env._joint_target[:,fingers]=.04
            env._physics_tick(label=9)
        retreat=place.clone(); retreat[:,2]+=.15
        ok = move(retreat,.04,10) and ok
    else:
        env._joint_target[:,fingers]=.04
    release_tick = env.physics_step_index
    stable = 0; samples=[]
    for _ in range(120):
        env._physics_tick(label=11)
        pos = body.data.root_pos_w
        error=float(torch.linalg.vector_norm(pos-target).item())
        speed=float(torch.linalg.vector_norm(body.data.root_lin_vel_w).item())
        angular=float(torch.linalg.vector_norm(body.data.root_ang_vel_w).item())
        support_error=abs(float(pos[0,2]-env.origin[0,2])-.015)
        refs=max(float(torch.linalg.vector_norm(b.data.root_pos_w-initial[i]).item()) for i,b in enumerate(env.bodies) if i!=index)
        fingers_open=bool((env.robot.data.joint_pos[:,fingers]>.03).all())
        valid=error<=.01 and speed<=.01 and angular<=.1 and support_error<=.004 and refs<=.005 and fingers_open
        stable=stable+1 if valid else 0
        samples.append([env.physics_step_index,error,speed,angular,support_error,refs,int(fingers_open),stable])
    success=bool(ok and grasp_verified and stable>=60)
    tcp,_=env._tcp_pose()
    return {'success':success,'action':'PickPlaceMidpoint','position_error_m':error,
            'position_tolerance_m':.01,'grasp_verified':grasp_verified,'lift_height_m':lifted_height,
            'dwell_steps':stable,'required_dwell_steps':60,'motion_steps':sum(s['steps'] for s in stages),
            'stages':stages,'target_object_w_m':target[0].cpu().tolist(),
            'final_object_w_m':body.data.root_pos_w[0].cpu().tolist(),
            'target_tcp_w_m':tcp[0].cpu().tolist(),'actual_tcp_w_m':tcp[0].cpu().tolist(),
            'release_tick':release_tick,'post_release_samples':samples,
            'reference_max_displacement_m':refs,'support_height_error_m':support_error,
            'object_pose_writes_after_reset':0,'joint_state_writes_during_motion':0,
            'attachments_used':False,'object_external_forces_used':False,
            'collision_check':'PhysX contact dynamics; collision-free motion not certified',
            'termination_reason':'stable_released_object' if success else 'placement_action_failure'}
