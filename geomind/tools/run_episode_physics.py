"""Execute native spatial episodes with Franka + real PhysX, always headless."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

from geomind.episodes import EpisodeResolver
from geomind.episodes.resolver import _check, load_json
from geomind.tools.migrate_episode_configs import write


def _negative_pick_object_id(choices, *, expected_object_id, object_id_map=None,
                             legacy_object_id=None):
    """Choose an eligible wrong body, optionally pinned by a private identity.

    object_id_map binds public aliases to evaluator-side legacy identities.
    A pinned body must remain unique, eligible and distinct from frozen GT;
    it must never silently fall back to an easier wrong candidate.
    """
    _check(bool(choices), 'No supported wrong-object candidate for a physical negative control')
    selected = choices[0]
    if legacy_object_id is not None:
        _check(isinstance(legacy_object_id, str) and bool(legacy_object_id),
               'Private negative control object identity must be a nonempty string')
        _check(isinstance(object_id_map, dict), 'Private negative control requires the object identity map')
        aliases = [alias for alias, legacy in object_id_map.items() if legacy == legacy_object_id]
        _check(len(aliases) == 1, 'Private negative control object must have exactly one alias')
        selected = aliases[0]
    _check(selected != expected_object_id and selected in choices,
           'Private negative control object must be an eligible non-GT candidate')
    return selected


def semantic_check(env, evidence: dict) -> dict:
    """Check the GT using independent geometric conditions / intervention identity."""
    result, params = env.result, env.resolved.episode["parameters"]
    positions = evidence["current"]["positions_m"]
    from geomind.episodes.resolver import is_registered_task, Snapshot, solve
    if is_registered_task(result['task_id']):
        if params.get('legacy_task_spec',{}).get('protocol',{}).get('module_name')=='canonical_extension':
            from geomind.episodes.registered_tasks import check_extended_ground_truth
            return check_extended_ground_truth(env.resolved,Snapshot.from_dict(evidence['current']),
                result['ground_truth'],history={key:Snapshot.from_dict(value) for key,value in evidence['history'].items()})
        # Re-run the original registered solver on recorded measurements, never
        # on planned event endpoints. The offline auditor checks raw trace rows.
        current=Snapshot.from_dict(evidence['current'])
        history={key:Snapshot.from_dict(value) for key,value in evidence['history'].items()}
        replay=solve(env.resolved,current,history=history)
        return {'success':replay['ground_truth']==result['ground_truth'],
                'validation_scope':'original registered solver replay from measured current/history',
                'calculation':replay['provenance']['calculation']}
    if result['task_id']=='SelectByOrder':
        sign=-1 if params['from']=='Right' else 1
        candidates=params['candidates']
        ranks={obj:1+sum(sign*positions[other][0]<sign*positions[obj][0] for other in candidates)
               for obj in candidates}
        winners=[obj for obj,rank in ranks.items() if rank==params['index']]
        return {'success':len(winners)==1 and winners[0]==result['ground_truth']['object_id'],
                'independent_ranks':ranks}
    if result["task_id"] == "InferMidpoint":
        a,b = [positions[o] for o in params["endpoints"]]
        q = result["ground_truth"]["position_m"]
        error = math.dist(q, [(x+y)/2 for x,y in zip(a,b)])
        return {"success": error < 1e-8, "midpoint_residual_m": error}
    if result["task_id"] == "DetectSceneChange":
        reference = evidence["history"][result["provenance"]["reference"]]["positions_m"]
        shifts = {o: math.dist(reference[o], positions[o]) for o in params["objects"]}
        events = env.resolved.protocol["phases"][1]["events"]
        planned = {e["object_id"] for e in events}
        measured = {o for o, shift in shifts.items() if shift > params["change_threshold_m"]}
        correct = len(measured) == 1 and measured == planned == {result["ground_truth"]["object_id"]}
        return {"success": correct, "planned_moved_objects": sorted(planned), "measured_displacements_m": shifts}
    point = positions[params["point"]]
    a, b = (positions[o] for o in params["line"])
    target = result["ground_truth"]["position_m"]
    projected = target if result["task_id"] == "ProjectOntoLine" else [(target[i] + point[i]) / 2 for i in range(3)]
    direction = [b[i] - a[i] for i in range(3)]
    delta = [projected[i] - a[i] for i in range(3)]
    norm = math.sqrt(sum(v * v for v in direction))
    cross = [delta[1] * direction[2] - delta[2] * direction[1],
             delta[2] * direction[0] - delta[0] * direction[2], delta[0] * direction[1] - delta[1] * direction[0]]
    on_line = math.sqrt(sum(v * v for v in cross)) / norm
    perpendicular = abs(sum((point[i] - projected[i]) * direction[i] for i in range(3))) / norm
    return {"success": on_line < 1e-8 and perpendicular < 1e-8,
            "line_residual_m": on_line, "orthogonality_residual_m": perpendicular}


def run_case(env, resolved, output: Path, *, negative: bool = False, negative_placement_policy='initial_first') -> dict:
    import numpy as np
    identity = resolved.episode["episode_id"] + ("_negative" if negative else "")
    obs, initial_info = env.reset(resolved=resolved)
    if env.recording_config is not None:
        env.start_recording(output / "videos" / identity)
    public_stream = [deepcopy(obs)]
    while not obs["allow_response"]:
        obs, *_ = env.step()
        public_stream.append(deepcopy(obs))
        if not obs["scene_visible"]:
            _check(obs["objects"] == [], "Physical memory observations leaked hidden scene")
    evidence = env.export_evidence()
    verified_gt = semantic_check(env, evidence)
    _check(verified_gt["success"], "Physical GT failed independent semantic validation")
    gt = env.result["ground_truth"]
    answer = {"object_id": gt["object_id"]} if gt["type"] == "Object" else {
        "position_m": list(gt["position_m"]), "frame": gt["frame"]}
    negative_plan=None
    if negative:
        if "object_id" in answer:
            choices=[o for o in env.ids if o != answer['object_id']]
            if env.is_assembled:
                from geomind.episodes.assembled_actions import object_geometry
                from geomind.episodes.gripper_profiles import gripper_profile
                raw=np.asarray([body.data.root_state_w[0].detach().cpu().numpy() for body in env.bodies])
                _,low,high=object_geometry(env.assembled_body_points,raw)
                source={obj['prim_path']:obj for obj in env.assembled_manifest['objects']}
                bindings={item['object_id']:item['prim_path'] for item in env.assembled_bindings}
                params=resolved.episode['parameters'];mapping=params.get('object_id_map',{})
                candidates=params.get('legacy_task_spec',{}).get('protocol',{}).get('candidate_object_ids')
                profile=gripper_profile(env.robot_name)
                choices=[o for o in choices if source[bindings[o]].get('dynamic',True)
                         and (not candidates or mapping[o] in candidates)
                         and high[env.object_index[o],1]-low[env.object_index[o],1]<=profile.max_object_width_m
                         and .015<=high[env.object_index[o],2]-low[env.object_index[o],2]<=.16]
            private_protocol = (resolved.episode['parameters'].get('legacy_task_spec',{}).get('protocol',{})
                                if env.is_assembled else {})
            answer["object_id"] = _negative_pick_object_id(
                choices, expected_object_id=gt['object_id'],
                object_id_map=resolved.episode['parameters'].get('object_id_map',{}) if env.is_assembled else None,
                legacy_object_id=private_protocol.get('negative_control_object_id'))
        else:
            if env.is_assembled:
                from geomind.scene_assembly.negative_controls import plan_wrong_placement_point
                raw=np.asarray([body.data.root_state_w[0].detach().cpu().numpy() for body in env.bodies])
                moving=env.object_index[resolved.episode['physical_execution']['moving_object']]
                negative_plan=plan_wrong_placement_point(gt,env.assembled_table,env.assembled_body_points,
                    raw,moving,resolved.scene['workspace'],candidate_order=negative_placement_policy)
                answer['position_m']=negative_plan['point_m']
            else:
                answer["position_m"][0] += 0.08
    query_result = deepcopy(env.result)
    _, reward, done, _, info = env.step(answer)
    video = env.finish_recording()
    _check(done, "Physical response did not terminate")
    path = output / "traces" / f"{identity}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=np.asarray(env.trace, dtype=np.float64),
        columns=np.array(["physics_step", "protocol_step", "stage"]
            + [env.robot.joint_names[i] for i in env.arm_ids] + ["tcp_x_w", "tcp_y_w", "tcp_z_w"]
            + [f"{identity}_{axis}_w" for identity in env.ids for axis in ("x", "y", "z")]))
    np.savez_compressed(output / "traces" / f"{identity}_objects.npz",
        data=np.asarray(env.object_state_trace), object_ids=np.array(env.ids), dt_s=env.dt)
    np.savez_compressed(output / "traces" / f"{identity}_controls.npz",
        data=np.asarray(env.control_trace, dtype=np.float64),
        joint_names=np.array(env.robot.joint_names),
        joint_limits=env.robot.data.soft_joint_pos_limits[0].cpu().numpy(),
        dt_s=env.dt,
        instance_id=env.instance_id,
        contract_hash=resolved.contract_hash)
    if env.robot_name == "piper_dual":
        np.savez_compressed(output / "traces" / f"{identity}_dual_controls.npz",
            data=np.asarray(env.dual_control_trace), arm_names=np.array(list(env.robots)),
            joint_names=np.array(env.robot.joint_names),
            joint_limits=np.stack([r.data.soft_joint_pos_limits[0].cpu().numpy() for r in env.robots.values()]),
            active_arm=env.active_arm, dt_s=env.dt)
    write(output / "observations" / f"{identity}.json", {"observations": public_stream})
    write(output / "events" / f"{identity}.json", {
        "planned_protocol": resolved.protocol, "measured_trace": str(path),
        "query_tick": evidence["physical_clock"]["current_tick"],
        "history_ticks": evidence["physical_clock"]["history_ticks"],
        "implementation": "external_force_pd" if resolved.protocol else "none"})
    write(output / "evidence" / f"{identity}.json", evidence)
    verified_response = (info["success"] if not negative else
                         not info["semantic"]["success"] and not info["physical"]["success"])
    if negative and env.is_assembled and info['physical'].get('action') in {'PickLiftSelectedObject','PickPlaceObject'}:
        # A rejected answer is only a valid manipulation control if the wrong
        # command itself was physically completed. Motion failure is separate.
        verified_response=bool(verified_response and info['physical'].get('command_execution_verified') is True)
    if negative and gt['type']=='Relation':
        physical=info['physical']
        from geomind.scene_assembly.usd import supported_xy
        low,high=np.asarray(physical['bbox_min_w_m']),np.asarray(physical['bbox_max_w_m'])
        # A legal relation outcome is never a wrong-answer control. Verify
        # that the robot actually executed the safe, unsupported-by-reference
        # command on the real table before accepting this rejection.
        verified_response=bool(not physical['success'] and (physical.get('relation_result') or {}).get('passed') is False
            and physical.get('command_execution_verified')
            and abs(physical['command_support_height_w_m']-env.assembled_table['height_m'])<=1e-6
            and supported_xy(env.assembled_table,physical['final_object_w_m'][:2],(high[:2]-low[:2])*.5))
    record = {"semantic_rejection_correct":bool(not info['semantic']['success'] and not info['physical']['success']) if negative else None,
        "negative_command_execution_valid":info['physical'].get('command_execution_verified') if negative and env.is_assembled else None,
        "episode_id": resolved.episode["episode_id"],
        "release_version": resolved.episode.get("release_version", "2.0"), "task_id": resolved.task["task_id"],
        "domain": resolved.task["domain"], "negative_control": negative, "success": info["success"],
        "expected_success": not negative, "verification_passed": bool(verified_response),
        "semantic": info["semantic"], "independent_gt_check": verified_gt, "physical": info["physical"],
        "negative_control_plan": negative_plan,
        "instruction": query_result["instruction"], "ground_truth": query_result["ground_truth"],
        "provenance": query_result["provenance"], "physics_steps": env.physics_step_index,
        "query_physical_clock": evidence["physical_clock"],
        "initial_object_positions_w_m": env.initial_positions_w, "reset": initial_info,
        "trace": str(path), "instance_id": env.instance_id,
        "control_trace": str(output / "traces" / f"{identity}_controls.npz"),
        "control_schema": {"columns": "joint targets, joint positions, joint velocities (joint_names order), TCP quaternion wxyz",
                           "action_space": "absolute joint position", "units": ("radians (arm and gripper)" if env.robot_name == 'franka_robotiq' else "radians; finger joints metres"),
                           "frame": "world", "frequency_hz": 1 / env.dt},
        "robot_asset": env.robot_asset, "active_arm": env.active_arm,
        "robot_asset_sha256": __import__('hashlib').sha256(Path(env.robot_asset).read_bytes()).hexdigest() if Path(env.robot_asset).is_file() else None,
        "arm_count": len(env.robots), "robot_name": env.robot_name,
        "tcp_offset_m": env.tcp_offset[0].cpu().tolist(),
        "tool_quaternion_wxyz": env.tool_quaternion[0].cpu().tolist(),
        "scene_provenance": env.scene_provenance,
        "object_assets": [{"asset": a, "descriptor": s} for a, s in zip(env.asset_order, env.asset_specs)],
        "robot_control": "Franka high-PD joint-position actuators; robot link gravity disabled",
        "robot_articulation_solver_configured": {
            "position_iterations": env.robot.cfg.spawn.articulation_props.solver_position_iteration_count,
            "velocity_iterations": env.robot.cfg.spawn.articulation_props.solver_velocity_iteration_count},
        "robot_solver_composed_schema": env.robot_solver_schema,
        "intervention": "external_force_pd" if resolved.protocol is not None else None,
        "video": video,
    }
    write(output / "cases" / f"{identity}.json", record)
    print(json.dumps({"episode": identity, "success": info["success"], "expected": not negative,
        "tcp_error_m": info["physical"]["position_error_m"], "motion_steps": info["physical"]["motion_steps"]}), flush=True)
    return record


def verification_complete(results, failure, expected_positive, expected_negative):
    """Reject missing controls as well as failed or incomplete positive runs."""
    positives = sum(not r['negative_control'] for r in results)
    negatives = sum(bool(r['negative_control']) for r in results)
    return bool(results and failure is None and positives == expected_positive
                and negatives == expected_negative and all(r['verification_passed'] for r in results))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--episode")
    group.add_argument("--suite", default="generated_spatial_suite")
    parser.add_argument("--output", type=Path, default=Path("outputs/headless_physics"))
    parser.add_argument("--device", default="cuda:0")
    control_mode = parser.add_mutually_exclusive_group()
    control_mode.add_argument("--negative-controls", action="store_true")
    control_mode.add_argument("--negative-only", action="store_true",
                              help="Run controls in a fresh simulator without preceding positive actions")
    parser.add_argument('--negative-placement-policy',choices=['initial_first','offsets_first'],default='initial_first',
                        help='Explicit fixed order for supported wrong placement candidates; all geometric gates remain unchanged')
    parser.add_argument("--video", action="store_true", help="Record RGB observer and visibility-filtered agent MP4s")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=[1.6, 1.3, 1.75])
    parser.add_argument("--camera-look-at", type=float, nargs=3, default=[0.4, 0.0, 1.05])
    parser.add_argument("--camera-focal-length", type=float, default=18.0)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-token", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        return supervise(args)
    resolver = EpisodeResolver(args.root)
    if args.episode:
        identities = [args.episode]
    else:
        manifest = load_json(args.root / "suites" / f"{args.suite}.json")
        identities = manifest.get("episodes") or [case["episode_id"] for case in manifest["cases"]]
    episodes = [resolver.resolve(identity) for identity in identities]
    _check(bool(episodes), "Empty physical suite")
    selected = {}
    if args.negative_controls or args.negative_only:
        for episode in episodes:
            selected.setdefault(episode.task['task_id'], episode)
    args.output.mkdir(parents=True, exist_ok=True)
    recording = {"fps": args.video_fps, "width": args.video_width, "height": args.video_height,
                 "eye_w_m": args.camera_eye, "look_at_w_m": args.camera_look_at,
                 "focal_length_mm": args.camera_focal_length} if args.video else None
    env, results, failure = None, [], None
    try:
        from geomind.episodes.isaaclab_physics import IsaacLabEpisode
        env = IsaacLabEpisode(episodes[0], device=args.device, headless=True, recording=recording)
        if not args.negative_only:
            for episode in episodes:
                results.append(run_case(env, episode, args.output))
        for episode in selected.values():
            results.append(run_case(env, episode, args.output, negative=True,negative_placement_policy=args.negative_placement_policy))
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        if env is not None and getattr(env,'object_state_trace',None):
            import numpy as np
            np.savez_compressed(args.output/'failure_objects.npz',data=np.asarray(env.object_state_trace,dtype=float),
                                object_ids=np.asarray(env.ids),dt_s=env.dt)
            write(args.output/'failure_state.json',{'failure':failure,
                'settling':getattr(env,'reset_settle_info',None),'physics_tick':env.physics_step_index,
                'scene_provenance':getattr(env,'scene_provenance',None),
                'object_state_trace':str(args.output/'failure_objects.npz'),
                'action_physics_verified':False})
    finally:
        positives = [r for r in results if not r["negative_control"]]
        success = verification_complete(results, failure, 0 if args.negative_only else len(episodes), len(selected))
        report = {"backend": "isaaclab_physx", "headless": True, "device": args.device, "policy": "oracle_gt",
            "isaac_sim_version": importlib.metadata.version("isaacsim"), "run_token": args.run_token,
            "physical_execution_verified": success, "positive_cases": len(positives),
            "positive_successes": sum(r["success"] for r in positives),
            "negative_controls": sum(r["negative_control"] for r in results), "failure": failure,
            "action": episodes[0].episode["physical_execution"]["action"], "shutdown": "pending", "results": results}
        report["recording"] = recording
        report["case_mode"] = "negative_only" if args.negative_only else "positive_and_negative" if args.negative_controls else "positive_only"
        report["video_cases"] = sum(r.get("video") is not None for r in results)
        # Flush all evidence before calling Kit shutdown, which can block on
        # this host. The parent owns a dedicated process group and a deadline.
        write(args.output / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2), flush=True)
        if env is not None:
            env.close()
        from geomind.core.envs.isaaclab_native_delegate import _IsaacLabAppRuntime
        app = _IsaacLabAppRuntime._simulation_app
        _IsaacLabAppRuntime._simulation_app = None
        _IsaacLabAppRuntime._launcher = None
        if app is not None:
            app.close(wait_for_replicator=False)
    return 0 if success else 1


def supervise(args) -> int:
    _check(math.isfinite(args.timeout_s) and args.timeout_s > 0, "timeout-s must be positive")
    token = uuid.uuid4().hex
    command = [sys.executable, "-m", "geomind.tools.run_episode_physics", *sys.argv[1:], "--worker", "--run-token", token]
    process = subprocess.Popen(command, start_new_session=True)
    deadline, cleanup_deadline = time.monotonic() + args.timeout_s, None
    report_path = args.output / "report.json"
    report = None
    bounded_cleanup = False
    try:
        while process.poll() is None:
            if report_path.is_file():
                try:
                    candidate = load_json(report_path)
                    if candidate.get("run_token") == token:
                        report = candidate
                        if cleanup_deadline is None:
                            cleanup_deadline = time.monotonic() + 10.0
                except ValueError:
                    pass  # Writer may still be flushing its fresh JSON record.
            if cleanup_deadline is not None and time.monotonic() >= cleanup_deadline:
                os.killpg(process.pid, signal.SIGKILL)
                bounded_cleanup = True
                break
            if time.monotonic() >= deadline:
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.1)
        process.wait()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    if report is None and report_path.is_file():
        candidate = load_json(report_path)
        if candidate.get("run_token") == token:
            report = candidate
    if report is None:
        write(report_path, {"backend": "isaaclab_physx", "headless": True, "run_token": token,
            "physical_execution_verified": False, "failure": "Worker failed or timed out before reporting",
            "worker_exit_code": process.returncode})
        print("Physics worker failed or timed out before completing its evidence report.", file=sys.stderr)
        return 1
    report["shutdown"] = "bounded_cleanup_after_report" if bounded_cleanup else "worker_exited"
    report["worker_exit_code"] = process.returncode
    write(report_path, report)
    print(json.dumps({"report": str(report_path), "physical_execution_verified": report["physical_execution_verified"],
                      "shutdown": report["shutdown"]}), flush=True)
    return 0 if report["physical_execution_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
