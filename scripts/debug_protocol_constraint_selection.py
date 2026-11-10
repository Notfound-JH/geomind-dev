#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import faulthandler
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass

from geomind_lab.bootstrap.isaaclab_path import ensure_isaaclab_source_on_path
from geomind_lab.mimic.envs.constraint_selection_logic import (
    CONSTRAINT_SELECTION_MODULE_NAME,
    build_constraint_selection_problem_from_object_positions,
)
from geomind_lab.mimic.utils.family_runner_registry import resolve_runner_family
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import (
    resolve_task_label,
    resolve_task_request,
)


ISAACLAB_SOURCE_DIR = ensure_isaaclab_source_on_path(REPO_ROOT)
for package_dir in (
    ISAACLAB_SOURCE_DIR / "isaaclab",
    ISAACLAB_SOURCE_DIR / "isaaclab_tasks",
    ISAACLAB_SOURCE_DIR / "isaaclab_mimic",
    ISAACLAB_SOURCE_DIR / "isaaclab_assets",
):
    package_dir_str = str(package_dir)
    if package_dir_str not in sys.path:
        sys.path.insert(0, package_dir_str)

if sys.platform != "win32":
    try:
        import pinocchio  # noqa: F401
    except Exception:
        pass

from isaaclab.app import AppLauncher


DEFAULT_DEBUG_SCENE_ID = 65
DEFAULT_DEBUG_LAYOUT_KEY = "select_visual_spatial_scene65_base"
DEFAULT_DEBUG_TASK_INDEX = 0
DEFAULT_RESET_SETTLE_STEPS = 40
GRIPPER_OPEN_COMMAND = 1.0
GRIPPER_CLOSE_COMMAND = -1.0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an oracle/debug rollout for annotation-backed constraint_selection grasp tasks "
            "using the standard GeoMind IK-relative scripted grasp primitive."
        )
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Preferred task selector. The script resolves the concrete annotation instance internally.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional task variant override. Falls back to the task profile default when omitted.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Optional scene alias override such as kitchen_table, lab_table, clutter_table, or scene_65.",
    )
    parser.add_argument("--scene-id", type=int, default=None, help="Optional scene filter or override.")
    parser.add_argument(
        "--layout-key",
        type=str,
        default=None,
        help="Optional annotation layout override. Usually you should pass --task-id instead.",
    )
    parser.add_argument("--task-index", type=int, default=None, help="Optional task index inside the selected layout.")
    parser.add_argument(
        "--object-mode",
        type=str,
        default=None,
        help="Optional object representation mode override: concrete or geometric.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized environments to spawn.")
    parser.add_argument(
        "--agent",
        type=str,
        choices=("oracle", "random"),
        default="oracle",
        help="Baseline selection policy for choosing the candidate object.",
    )
    parser.add_argument("--agent-seed", type=int, default=0, help="Base random seed for the random baseline.")
    parser.add_argument("--agent-episodes", type=int, default=1, help="Number of reset->select->grasp episodes to run.")
    parser.add_argument(
        "--reset-settle-steps",
        type=int,
        default=DEFAULT_RESET_SETTLE_STEPS,
        help="Number of zero-action settle frames after each reset.",
    )
    parser.add_argument(
        "--post-grasp-hold-steps",
        type=int,
        default=8,
        help="Number of hold frames after lift before reading final grasp success.",
    )
    parser.add_argument(
        "--summary-json-out",
        type=str,
        default=None,
        help="Optional path to write the structured task rollout summary JSON.",
    )
    parser.add_argument(
        "--trajectory-json-out",
        type=str,
        default=None,
        help="Optional path to write the per-step trajectory JSON.",
    )
    parser.add_argument(
        "--camera-rollout-video",
        action="store_true",
        default=False,
        help=(
            "When running headless, save per-camera RGB MP4 for the full episode-0 rollout. "
            "Requires OUTPUT_DIR. Or set HEADLESS_CAMERA_VIDEO=1. "
            "(Prefix camera-* avoids argparse clash with --headless.)"
        ),
    )
    parser.add_argument(
        "--camera-rollout-video-frames",
        type=int,
        default=0,
        help="Maximum frames per camera clip. Use 0 to capture the full rollout without truncation.",
    )
    parser.add_argument(
        "--camera-rollout-video-fps",
        type=float,
        default=60.0,
        help="FPS for rollout camera clips.",
    )
    parser.add_argument(
        "--disable-fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O instead.",
    )
    parser.add_argument(
        "--viewer-eye",
        type=float,
        nargs=3,
        default=(2.2, 2.2, 1.8),
        metavar=("X", "Y", "Z"),
        help="Viewer camera eye position.",
    )
    parser.add_argument(
        "--viewer-lookat",
        type=float,
        nargs=3,
        default=(0.3, 0.0, 0.8),
        metavar=("X", "Y", "Z"),
        help="Viewer camera look-at position.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def print_failure_context(stage: str, *, env_cfg=None) -> None:
    print("\n" + "=" * 72, file=sys.stderr, flush=True)
    print("GeoMind Constraint Selection Debug Failure", file=sys.stderr, flush=True)
    print("=" * 72, file=sys.stderr, flush=True)
    print(f"stage:          {stage}", file=sys.stderr, flush=True)

    if args_cli is not None:
        print(f"task_id:        {getattr(args_cli, 'task_id', None)}", file=sys.stderr, flush=True)
        print(f"variant:        {getattr(args_cli, 'variant', None)}", file=sys.stderr, flush=True)
        print(f"scene:          {getattr(args_cli, 'scene', None)}", file=sys.stderr, flush=True)
        print(f"scene_id:       {getattr(args_cli, 'scene_id', None)}", file=sys.stderr, flush=True)
        print(f"layout_key:     {getattr(args_cli, 'layout_key', None)}", file=sys.stderr, flush=True)
        print(f"task_index:     {getattr(args_cli, 'task_index', None)}", file=sys.stderr, flush=True)
        print(f"object_mode:    {getattr(args_cli, 'object_mode', None)}", file=sys.stderr, flush=True)
        print(f"agent:          {getattr(args_cli, 'agent', None)}", file=sys.stderr, flush=True)
        print(f"agent_episodes: {getattr(args_cli, 'agent_episodes', None)}", file=sys.stderr, flush=True)
        print(f"device:         {getattr(args_cli, 'device', None)}", file=sys.stderr, flush=True)
        print(f"headless:       {getattr(args_cli, 'headless', None)}", file=sys.stderr, flush=True)

    if env_cfg is not None:
        task_protocol = dict(getattr(env_cfg, "task_protocol", {}) or {})
        task_metadata = dict(getattr(env_cfg, "task_metadata", {}) or {})
        print(f"resolved_scene: {getattr(env_cfg, 'scene_id', None)}", file=sys.stderr, flush=True)
        print(f"resolved_scene_name:{getattr(env_cfg, 'scene_name', None)}", file=sys.stderr, flush=True)
        print(f"resolved_layout:{getattr(env_cfg, 'task_layout_key', None)}", file=sys.stderr, flush=True)
        print(f"resolved_index: {getattr(env_cfg, 'task_index', None)}", file=sys.stderr, flush=True)
        print(f"resolved_variant:{getattr(env_cfg, 'task_variant', None)}", file=sys.stderr, flush=True)
        print(f"resolved_object_mode:{getattr(env_cfg, 'object_mode', None)}", file=sys.stderr, flush=True)
        print(f"task_id:        {task_metadata.get('task_id', '')}", file=sys.stderr, flush=True)
        print(f"task_version:   {task_metadata.get('task_version', '')}", file=sys.stderr, flush=True)
        print(f"instruction:    {getattr(env_cfg, 'task_instruction', '')}", file=sys.stderr, flush=True)
        print(
            "task_protocol:  " + json.dumps(task_protocol, sort_keys=True, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )

    print("-" * 72, file=sys.stderr, flush=True)
    traceback.print_exc()
    print("=" * 72, file=sys.stderr, flush=True)


args_cli = build_argparser().parse_args()
setattr(args_cli, "enable_cameras", True)

simulation_app = None
try:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
except Exception:
    print_failure_context("app_launcher")
    raise SystemExit(1)


import torch
import isaaclab.utils.math as PoseUtils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from geomind_lab.mimic import GeoMindFrankaIKRelMimicEnv, GeoMindFrankaIKRelMimicEnvCfg
from geomind_lab.utils.headless_camera_video import RolloutCameraRecorder, create_rollout_camera_recorder
from geomind_lab.utils.headless_rollout_camera_hook import resolve_rollout_camera_video_dir
from geomind_lab.tasks.manager_based.manipulation.geomind import mdp as geomind_mdp
from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.common import get_asset_pose_w


DOWN_FACING_QUAT = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


@dataclass(frozen=True)
class EpisodeSelection:
    object_id: str
    object_index: int
    rationale: str


def _resolve_cli_task_request(
    *,
    task_id: str | None = None,
    variant: str | None = None,
    scene: str | None = None,
    scene_id: int | None = None,
    layout_key: str | None = None,
    task_index: int | None = None,
) -> object:
    return resolve_task_request(
        task_id=task_id,
        variant=variant,
        scene=scene,
        scene_id=scene_id,
        layout_key=layout_key or DEFAULT_DEBUG_LAYOUT_KEY,
        task_index=DEFAULT_DEBUG_TASK_INDEX if task_index is None else task_index,
        object_mode=args_cli.object_mode,
    )


def build_runtime_env_cfg(
    *,
    request=None,
):
    base_cfg_cls = GeoMindFrankaIKRelMimicEnvCfg
    resolved_request = request
    if resolved_request is None:
        resolved_request = _resolve_cli_task_request(
            task_id=args_cli.task_id,
            variant=args_cli.variant,
            scene=args_cli.scene,
            scene_id=args_cli.scene_id,
            layout_key=args_cli.layout_key,
            task_index=args_cli.task_index,
        )

    def __post_init__(self):
        self.task_id = str(resolved_request.task_id)
        self.task_variant = str(resolved_request.variant)
        self.scene_name = str(resolved_request.scene)
        self.object_mode = str(resolved_request.object_mode)
        self.scene_id = int(resolved_request.scene_id)
        self.task_layout_key = str(resolved_request.layout_key)
        self.task_index = int(resolved_request.task_index)
        self.scene.num_envs = args_cli.num_envs
        self.sim.device = args_cli.device
        self.sim.use_fabric = not args_cli.disable_fabric

        base_cfg_cls.__post_init__(self)

        self.viewer.eye = tuple(float(v) for v in args_cli.viewer_eye)
        self.viewer.lookat = tuple(float(v) for v in args_cli.viewer_lookat)

    runtime_cfg_cls = configclass(
        type(
            "RuntimeGeoMindConstraintSelectionDebugCfg",
            (base_cfg_cls,),
            {
                "__doc__": "Runtime GeoMind mimic cfg for constraint_selection oracle/debug rollouts.",
                "task_id": str(resolved_request.task_id),
                "task_variant": str(resolved_request.variant),
                "scene_name": str(resolved_request.scene),
                "object_mode": str(resolved_request.object_mode),
                "scene_id": int(resolved_request.scene_id),
                "task_layout_key": str(resolved_request.layout_key),
                "task_index": int(resolved_request.task_index),
                "__post_init__": __post_init__,
            },
        )
    )
    return runtime_cfg_cls()


def should_reset(terminated, truncated) -> bool:
    if terminated is not None and torch.any(terminated):
        return True
    if truncated is not None and torch.any(truncated):
        return True
    return False


def make_zero_actions(env: GeoMindFrankaIKRelMimicEnv) -> torch.Tensor:
    return torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)


def settle_env_after_reset(env: GeoMindFrankaIKRelMimicEnv, settle_steps: int) -> None:
    if settle_steps <= 0:
        return
    zero_actions = make_zero_actions(env)
    for _ in range(max(1, int(settle_steps))):
        _, _, terminated, truncated, _ = env.step(zero_actions)
        if should_reset(terminated, truncated):
            env.reset()


def _print_runtime_header(env_cfg, *, task_label: str) -> None:
    task_metadata = dict(getattr(env_cfg, "task_metadata", {}) or {})
    task_protocol = dict(getattr(env_cfg, "task_protocol", {}) or {})
    runner_family = os.environ.get("RUNNER_FAMILY") or resolve_runner_family(
        task_id=task_metadata.get("task_id"),
        family=task_protocol.get("family"),
        semantic_module=task_metadata.get("semantic_module"),
        skill_type=task_metadata.get("skill_type"),
    )
    print("=" * 72, flush=True)
    print("GeoMind Constraint Selection Debugger", flush=True)
    print("=" * 72, flush=True)
    print(f"task_id:        {task_metadata.get('task_id', '')}", flush=True)
    print(f"task_version:   {task_metadata.get('task_version', '')}", flush=True)
    print(f"phase:          {task_metadata.get('phase', '')}", flush=True)
    print(f"semantic_module:{task_metadata.get('semantic_module', '')}", flush=True)
    print(f"skill_type:     {task_metadata.get('skill_type', '')}", flush=True)
    print(f"runner_family:  {runner_family}", flush=True)
    print(f"runner_ready:   {bool(task_metadata.get('is_runnable', False))}", flush=True)
    print("preview_only:   True", flush=True)
    print(f"variant:        {getattr(env_cfg, 'task_variant', '')}", flush=True)
    print(f"scene:          {getattr(env_cfg, 'scene_name', '')}", flush=True)
    print(f"scene_id:       {env_cfg.scene_id}", flush=True)
    print(f"layout_key:     {env_cfg.task_layout_key}", flush=True)
    print(f"task_index:     {env_cfg.task_index}", flush=True)
    print(f"object_mode:    {getattr(env_cfg, 'object_mode', '')}", flush=True)
    print(f"task_label:     {task_label}", flush=True)
    print(f"instruction:    {getattr(env_cfg, 'task_instruction', '')}", flush=True)
    print(f"agent:          {args_cli.agent}", flush=True)
    print(f"agent_episodes: {args_cli.agent_episodes}", flush=True)
    print(f"reset_settle:   {args_cli.reset_settle_steps}", flush=True)
    print(f"post_grasp_hold:{args_cli.post_grasp_hold_steps}", flush=True)
    print(
        "task_protocol:  " + json.dumps(task_protocol, sort_keys=True, ensure_ascii=False),
        flush=True,
    )


def _write_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _compact_summary_payload(payload: object) -> object:
    if isinstance(payload, dict):
        compact: dict[str, object] = {}
        for key, value in payload.items():
            if key == "episodes":
                continue
            compact[str(key)] = _compact_summary_payload(value)
        return compact
    if isinstance(payload, list):
        return [_compact_summary_payload(value) for value in payload]
    return payload


def _camera_rollout_video_max_frames() -> int | None:
    value = int(getattr(args_cli, "camera_rollout_video_frames", 0))
    return None if value <= 0 else value


def _print_rollout_camera_video_result(paths: list[Path], video_dir: Path) -> None:
    if paths:
        print(f"headless_camera_video: wrote {len(paths)} full-rollout clip(s) under {video_dir}", flush=True)
        for path in paths:
            print(f"  {path}", flush=True)
    else:
        print(
            f"headless_camera_video: no clips written (no Camera sensors or empty RGB?) -> {video_dir}",
            flush=True,
        )


def _eef_name(env: GeoMindFrankaIKRelMimicEnv) -> str:
    return env._eef_name()


def _pose_from_position_and_quat(position: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    rotation = PoseUtils.matrix_from_quat(quat.unsqueeze(0))[0]
    return PoseUtils.make_pose(position, rotation)


def _action_from_target_pose(
    env: GeoMindFrankaIKRelMimicEnv,
    target_pose: torch.Tensor,
    gripper_action: float,
    *,
    env_id: int = 0,
) -> torch.Tensor:
    eef_name = _eef_name(env)
    action = env.target_eef_pose_to_action(
        target_eef_pose_dict={eef_name: target_pose},
        gripper_action_dict={
            eef_name: torch.tensor([gripper_action], device=env.device, dtype=torch.float32),
        },
        env_id=env_id,
    )
    if action.dim() == 1:
        action = action.unsqueeze(0)
    return action


def _current_asset_position(env: GeoMindFrankaIKRelMimicEnv, asset_name: str, *, env_id: int = 0) -> torch.Tensor:
    pos_w, _ = get_asset_pose_w(env, asset_name)
    return pos_w[env_id, :3].clone().detach()


def _current_asset_xy(env: GeoMindFrankaIKRelMimicEnv, asset_name: str, *, env_id: int = 0) -> tuple[float, float]:
    position = _current_asset_position(env, asset_name, env_id=env_id)
    return (float(position[0].item()), float(position[1].item()))


def _current_object_positions_xy_map(
    env: GeoMindFrankaIKRelMimicEnv,
    object_names: Sequence[str],
) -> dict[str, dict[str, list[float]]]:
    return {
        str(object_name): {"position": list(_current_asset_xy(env, str(object_name)))}
        for object_name in object_names
    }


def _current_eef_position(env: GeoMindFrankaIKRelMimicEnv, *, env_id: int = 0) -> torch.Tensor:
    pose = env.get_robot_eef_pose(_eef_name(env), env_ids=[env_id])[0]
    position, _ = PoseUtils.unmake_pose(pose)
    return position.clone().detach()


def _object_height(env: GeoMindFrankaIKRelMimicEnv, object_name: str) -> float:
    resolver = getattr(env.cfg, "_resolve_object_runtime_height", None)
    if callable(resolver):
        resolved = resolver(object_name)
        if resolved is not None:
            return float(resolved)
    return 0.07


def _is_runtime_primitive_sphere(env: GeoMindFrankaIKRelMimicEnv, object_name: str) -> bool:
    task_spec = getattr(env.cfg, "task_spec", None) or {}
    runtime_objects = task_spec.get("runtime_objects", []) if isinstance(task_spec, dict) else []
    if not isinstance(runtime_objects, list):
        return False
    for runtime_object in runtime_objects:
        if not isinstance(runtime_object, dict):
            continue
        if str(runtime_object.get("name", "")).strip() != object_name:
            continue
        return str(runtime_object.get("kind", "")).strip() == "primitive_sphere"
    return False


def _object_grasped(env: GeoMindFrankaIKRelMimicEnv, object_name: str, *, env_id: int = 0) -> bool:
    grasped = geomind_mdp.object_grasped(
        env,
        robot_cfg=SceneEntityCfg("robot"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg(object_name),
    )
    return bool(grasped[env_id].item())


def _gripper_joint_mean_position(env: GeoMindFrankaIKRelMimicEnv) -> float | None:
    robot = env.scene["robot"]
    if not hasattr(env.cfg, "gripper_joint_names"):
        return None
    joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    if len(joint_ids) == 0:
        return None
    return float(robot.data.joint_pos[0, joint_ids].mean().item())


def _trace_step(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    step_trace: list[dict[str, Any]],
    stage_name: str,
    stage_step: int,
    action: torch.Tensor,
    target_pose: torch.Tensor,
    gripper_action: float,
    anchor_object_id: str,
    candidate_object_ids: Sequence[str],
    selected_object_id: str,
    terminated,
    truncated,
) -> None:
    target_position, _ = PoseUtils.unmake_pose(target_pose)
    eef_position = _current_eef_position(env)
    selected_position = _current_asset_position(env, selected_object_id)
    anchor_position = _current_asset_position(env, anchor_object_id)
    step_trace.append(
        {
            "step_index": int(len(step_trace)),
            "stage": str(stage_name),
            "stage_step": int(stage_step),
            "action": [float(value) for value in action[0].detach().cpu().tolist()],
            "gripper_action": float(gripper_action),
            "target_pose_position": [float(value) for value in target_position.detach().cpu().tolist()],
            "eef_position": [float(value) for value in eef_position.detach().cpu().tolist()],
            "selected_object_id": str(selected_object_id),
            "selected_object_position": [float(value) for value in selected_position.detach().cpu().tolist()],
            "anchor_object_id": str(anchor_object_id),
            "anchor_object_position": [float(value) for value in anchor_position.detach().cpu().tolist()],
            "candidate_positions_xy": {
                str(object_id): [float(value) for value in _current_asset_xy(env, str(object_id))]
                for object_id in candidate_object_ids
            },
            "object_grasped": bool(_object_grasped(env, selected_object_id)),
            "terminated": bool(torch.any(terminated).item()) if isinstance(terminated, torch.Tensor) else bool(terminated),
            "truncated": bool(torch.any(truncated).item()) if isinstance(truncated, torch.Tensor) else bool(truncated),
        }
    )


def _step_with_tracked_target_xy(
    env: GeoMindFrankaIKRelMimicEnv,
    target_name: str,
    z: float,
    gripper_action: float,
    *,
    steps: int,
    stage_name: str,
    step_trace: list[dict[str, Any]],
    anchor_object_id: str,
    candidate_object_ids: Sequence[str],
    selected_object_id: str,
    fixed_xy: tuple[float, float] | None = None,
) -> bool:
    for stage_step in range(max(1, int(steps))):
        if fixed_xy is None:
            pos = _current_asset_position(env, target_name)
            target_x = float(pos[0].item())
            target_y = float(pos[1].item())
        else:
            target_x = float(fixed_xy[0])
            target_y = float(fixed_xy[1])
        target_pose = _pose_from_position_and_quat(
            torch.tensor([target_x, target_y, float(z)], device=env.device, dtype=torch.float32),
            DOWN_FACING_QUAT.to(device=env.device),
        )
        action = _action_from_target_pose(env, target_pose, gripper_action)
        _, _, terminated, truncated, _ = env.step(action)
        _trace_step(
            env,
            step_trace=step_trace,
            stage_name=stage_name,
            stage_step=stage_step,
            action=action,
            target_pose=target_pose,
            gripper_action=gripper_action,
            anchor_object_id=anchor_object_id,
            candidate_object_ids=candidate_object_ids,
            selected_object_id=selected_object_id,
            terminated=terminated,
            truncated=truncated,
        )
        if should_reset(terminated, truncated):
            return False
    return True


def _select_candidate(problem: dict[str, Any], *, agent_name: str, rng: random.Random) -> EpisodeSelection:
    candidate_object_ids = list(problem.get("candidate_object_ids", []))
    gt_object_id = str(problem.get("gt_object_id", ""))
    if agent_name == "oracle":
        return EpisodeSelection(
            object_id=gt_object_id,
            object_index=int(problem.get("gt_object_index", -1)),
            rationale="protocol_gt",
        )
    if not candidate_object_ids:
        return EpisodeSelection(object_id="", object_index=-1, rationale="no_candidates")
    object_index = rng.randrange(len(candidate_object_ids))
    return EpisodeSelection(
        object_id=str(candidate_object_ids[object_index]),
        object_index=int(object_index),
        rationale="uniform_random_candidate",
    )


def run_scripted_grasp_for_target(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    problem: dict[str, Any],
    target_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_position = _current_asset_position(env, target_name)
    object_height = _object_height(env, target_name)
    target_surface_z = float(target_position[2].item())
    approach_z = float(target_surface_z + max(0.12, 1.5 * object_height))
    if _is_runtime_primitive_sphere(env, target_name):
        grasp_z = float(target_surface_z - 0.15 * object_height)
    else:
        grasp_z = float(target_surface_z + max(0.0, 0.05 * object_height))
    lift_z = float(target_surface_z + max(0.18, 2.0 * object_height))
    step_trace: list[dict[str, Any]] = []

    completed = _step_with_tracked_target_xy(
        env,
        target_name,
        approach_z,
        GRIPPER_OPEN_COMMAND,
        steps=30,
        stage_name="approach",
        step_trace=step_trace,
        anchor_object_id=str(problem["anchor_object_id"]),
        candidate_object_ids=list(problem["candidate_object_ids"]),
        selected_object_id=target_name,
    )
    locked_target_xy: tuple[float, float] | None = None
    if completed:
        locked_target_xy = _current_asset_xy(env, target_name)

    completed = (
        completed
        and _step_with_tracked_target_xy(
            env,
            target_name,
            grasp_z,
            GRIPPER_OPEN_COMMAND,
            steps=20,
            stage_name="descend",
            step_trace=step_trace,
            anchor_object_id=str(problem["anchor_object_id"]),
            candidate_object_ids=list(problem["candidate_object_ids"]),
            selected_object_id=target_name,
            fixed_xy=locked_target_xy,
        )
        and _step_with_tracked_target_xy(
            env,
            target_name,
            grasp_z,
            GRIPPER_CLOSE_COMMAND,
            steps=28,
            stage_name="close",
            step_trace=step_trace,
            anchor_object_id=str(problem["anchor_object_id"]),
            candidate_object_ids=list(problem["candidate_object_ids"]),
            selected_object_id=target_name,
            fixed_xy=locked_target_xy,
        )
        and _step_with_tracked_target_xy(
            env,
            target_name,
            lift_z,
            GRIPPER_CLOSE_COMMAND,
            steps=24,
            stage_name="lift",
            step_trace=step_trace,
            anchor_object_id=str(problem["anchor_object_id"]),
            candidate_object_ids=list(problem["candidate_object_ids"]),
            selected_object_id=target_name,
            fixed_xy=locked_target_xy,
        )
    )
    if completed and int(args_cli.post_grasp_hold_steps) > 0:
        completed = _step_with_tracked_target_xy(
            env,
            target_name,
            lift_z,
            GRIPPER_CLOSE_COMMAND,
            steps=int(args_cli.post_grasp_hold_steps),
            stage_name="hold",
            step_trace=step_trace,
            anchor_object_id=str(problem["anchor_object_id"]),
            candidate_object_ids=list(problem["candidate_object_ids"]),
            selected_object_id=target_name,
            fixed_xy=locked_target_xy,
        )

    final_position = _current_asset_position(env, target_name)
    grasp_sensor_triggered = completed and _object_grasped(env, target_name)
    object_lifted = float(final_position[2].item()) > float(target_surface_z + max(0.08, 1.2 * object_height))
    grasp_success = completed and (grasp_sensor_triggered or object_lifted)
    eef_position = _current_eef_position(env)
    grasp_offset_xy = (
        float(final_position[0].item() - eef_position[0].item()),
        float(final_position[1].item() - eef_position[1].item()),
    )

    return (
        {
            "target_name": str(target_name),
            "target_height": float(object_height),
            "target_surface_z": float(target_surface_z),
            "approach_z": float(approach_z),
            "grasp_z": float(grasp_z),
            "lift_z": float(lift_z),
            "locked_target_xy": ([] if locked_target_xy is None else [float(locked_target_xy[0]), float(locked_target_xy[1])]),
            "execution_completed": bool(completed),
            "terminated_early": not bool(completed),
            "grasp_success": bool(grasp_success),
            "grasp_sensor_triggered": bool(grasp_sensor_triggered),
            "object_lifted": bool(object_lifted),
            "grasp_offset_xy": [float(grasp_offset_xy[0]), float(grasp_offset_xy[1])],
            "final_target_position": [float(value) for value in final_position.tolist()],
            "eef_position_final": [float(value) for value in eef_position.tolist()],
            "gripper_joint_mean_position": _gripper_joint_mean_position(env),
        },
        step_trace,
    )


def _build_episode_result(
    *,
    env: GeoMindFrankaIKRelMimicEnv,
    problem: dict[str, Any],
    selection: EpisodeSelection,
    grasp_execution: dict[str, Any],
    episode_index: int,
    task_label: str,
    trace_step_count: int,
) -> dict[str, Any]:
    selected_matches_gt = bool(selection.object_id) and str(selection.object_id) == str(problem["gt_object_id"])
    valid_episode = bool(selection.object_id) and bool(grasp_execution["execution_completed"])
    success = selected_matches_gt and bool(grasp_execution["grasp_success"])

    if not selection.object_id:
        failure_category = "invalid_selection"
    elif not bool(grasp_execution["execution_completed"]):
        failure_category = "execution_terminated"
    elif not selected_matches_gt:
        failure_category = "wrong_selection"
    elif not bool(grasp_execution["grasp_success"]):
        failure_category = "no_grasp"
    else:
        failure_category = "success"

    return {
        "episode_index": int(episode_index),
        "agent_name": str(args_cli.agent),
        "scene_id": int(getattr(env.cfg, "scene_id", -1)),
        "layout_key": str(getattr(env.cfg, "task_layout_key", "")),
        "task_index": int(getattr(env.cfg, "task_index", -1)),
        "task_label": str(task_label),
        "task_id": str(dict(getattr(env.cfg, "task_metadata", {}) or {}).get("task_id", "")),
        "task_version": str(dict(getattr(env.cfg, "task_metadata", {}) or {}).get("task_version", "")),
        "module_name": str(problem.get("module_name", CONSTRAINT_SELECTION_MODULE_NAME)),
        "variant": str(problem.get("variant", "")),
        "query_text": str(problem.get("query_text", "")),
        "anchor_object_id": str(problem["anchor_object_id"]),
        "anchor_xy": [float(value) for value in problem["anchor_xy"]],
        "candidate_object_ids": list(problem["candidate_object_ids"]),
        "candidate_semantics": dict(problem["candidate_semantics"]),
        "candidate_geometry": dict(problem["candidate_geometry"]),
        "target_filters": dict(problem["target_filters"]),
        "spatial_match_ids": list(problem["spatial_match_ids"]),
        "attribute_match_ids": list(problem["attribute_match_ids"]),
        "selected_object_id": str(selection.object_id),
        "selected_object_index": int(selection.object_index),
        "selection_rationale": str(selection.rationale),
        "gt_object_id": str(problem["gt_object_id"]),
        "gt_object_index": int(problem["gt_object_index"]),
        "selected_matches_gt": bool(selected_matches_gt),
        "grasp_execution_success": bool(grasp_execution["grasp_success"]),
        "grasp_sensor_triggered": bool(grasp_execution["grasp_sensor_triggered"]),
        "object_lifted": bool(grasp_execution["object_lifted"]),
        "valid_episode": bool(valid_episode),
        "success": bool(success),
        "failure_category": str(failure_category),
        "trace_step_count": int(trace_step_count),
        "grasp_execution": grasp_execution,
    }


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total > 0 else 0.0


def build_rollout_summary(
    *,
    env_cfg,
    task_label: str,
    episode_results: list[dict[str, Any]],
) -> dict[str, Any]:
    num_episodes = len(episode_results)
    selection_match_count = sum(bool(result.get("selected_matches_gt")) for result in episode_results)
    grasp_success_count = sum(bool(result.get("grasp_execution_success")) for result in episode_results)
    success_count = sum(bool(result.get("success")) for result in episode_results)
    valid_episode_count = sum(bool(result.get("valid_episode")) for result in episode_results)
    failure_breakdown: dict[str, int] = {}
    for result in episode_results:
        category = str(result.get("failure_category", ""))
        failure_breakdown[category] = failure_breakdown.get(category, 0) + 1

    task_metadata = dict(getattr(env_cfg, "task_metadata", {}) or {})
    task_protocol = dict(getattr(env_cfg, "task_protocol", {}) or {})
    return {
        "agent_name": str(args_cli.agent),
        "agent_seed": int(args_cli.agent_seed),
        "scene_id": int(getattr(env_cfg, "scene_id", -1)),
        "layout_key": str(getattr(env_cfg, "task_layout_key", "")),
        "task_index": int(getattr(env_cfg, "task_index", -1)),
        "task_label": str(task_label),
        "task_id": str(task_metadata.get("task_id", "")),
        "task_version": str(task_metadata.get("task_version", "")),
        "semantic_module": str(task_metadata.get("semantic_module", "")),
        "runner_family": resolve_runner_family(
            task_id=task_metadata.get("task_id"),
            family=task_protocol.get("family"),
            semantic_module=task_metadata.get("semantic_module"),
            skill_type=task_metadata.get("skill_type"),
        ),
        "benchmark_runnable": bool(task_metadata.get("is_runnable", False)),
        "preview_runnable": True,
        "instruction": str(getattr(env_cfg, "task_instruction", "")),
        "query_text": str(task_protocol.get("query_text", "")),
        "num_episodes": int(num_episodes),
        "episode_count": int(num_episodes),
        "selection_match_count": int(selection_match_count),
        "grasp_success_count": int(grasp_success_count),
        "success_count": int(success_count),
        "valid_episode_count": int(valid_episode_count),
        "selection_accuracy": _rate(selection_match_count, num_episodes),
        "grasp_success_rate": _rate(grasp_success_count, num_episodes),
        "success_rate": _rate(success_count, num_episodes),
        "valid_episode_rate": _rate(valid_episode_count, num_episodes),
        "failure_breakdown": dict(sorted(failure_breakdown.items())),
        "episodes": episode_results,
    }


def run_episode(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    rng: random.Random,
    episode_index: int,
    task_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env.reset()
    settle_env_after_reset(env, int(args_cli.reset_settle_steps))

    video_dir = None
    fps_roll = 30.0
    recorder: RolloutCameraRecorder | None = None
    if int(episode_index) == 0:
        video_dir = resolve_rollout_camera_video_dir(args_cli, task_label, env)
        if video_dir is not None:
            fps_roll = float(getattr(args_cli, "camera_rollout_video_fps", 30.0))
            recorder = create_rollout_camera_recorder(
                env,
                max_frames=_camera_rollout_video_max_frames(),
                sample_hz=fps_roll,
            ).install()
            recorder.capture_now()

    task_spec = dict(getattr(env.cfg, "task_spec", {}) or {})
    object_positions = _current_object_positions_xy_map(env, list(task_spec.get("object_positions", {}).keys()))
    problem = build_constraint_selection_problem_from_object_positions(task_spec, object_positions)
    selection = _select_candidate(problem, agent_name=args_cli.agent, rng=rng)

    try:
        if selection.object_id:
            grasp_execution, trajectory_steps = run_scripted_grasp_for_target(
                env,
                problem=problem,
                target_name=selection.object_id,
            )
        else:
            grasp_execution = {
                "target_name": "",
                "execution_completed": False,
                "terminated_early": False,
                "grasp_success": False,
                "grasp_sensor_triggered": False,
                "object_lifted": False,
                "final_target_position": [],
            }
            trajectory_steps = []
    finally:
        if recorder is not None:
            recorder.uninstall()

    if int(episode_index) == 0 and video_dir is not None and recorder is not None:
        paths = recorder.save_mp4s(video_dir, fps=fps_roll)
        _print_rollout_camera_video_result(paths, video_dir)

    episode_result = _build_episode_result(
        env=env,
        problem=problem,
        selection=selection,
        grasp_execution=grasp_execution,
        episode_index=episode_index,
        task_label=task_label,
        trace_step_count=len(trajectory_steps),
    )
    trajectory_payload = {
        "episode_index": int(episode_index),
        "scene_id": int(getattr(env.cfg, "scene_id", -1)),
        "layout_key": str(getattr(env.cfg, "task_layout_key", "")),
        "task_index": int(getattr(env.cfg, "task_index", -1)),
        "task_label": str(task_label),
        "agent_name": str(args_cli.agent),
        "selection": {
            "object_id": str(selection.object_id),
            "object_index": int(selection.object_index),
            "rationale": str(selection.rationale),
        },
        "problem": problem,
        "grasp_execution": grasp_execution,
        "steps": trajectory_steps,
    }
    return episode_result, trajectory_payload


def run_agent_rollout(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    task_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(int(args_cli.agent_seed))
    episode_results: list[dict[str, Any]] = []
    trajectory_episodes: list[dict[str, Any]] = []

    for episode_index in range(max(1, int(args_cli.agent_episodes))):
        episode_result, trajectory_payload = run_episode(
            env,
            rng=rng,
            episode_index=episode_index,
            task_label=task_label,
        )
        episode_results.append(episode_result)
        trajectory_episodes.append(trajectory_payload)
        print("episode_result:", flush=True)
        print(json.dumps(episode_result, indent=2, sort_keys=True, ensure_ascii=False), flush=True)

    summary = build_rollout_summary(env_cfg=env.cfg, task_label=task_label, episode_results=episode_results)
    trajectory_summary = {
        "task_id": str(summary.get("task_id", "")),
        "task_version": str(summary.get("task_version", "")),
        "scene_id": int(summary.get("scene_id", -1)),
        "layout_key": str(summary.get("layout_key", "")),
        "task_index": int(summary.get("task_index", -1)),
        "task_label": str(summary.get("task_label", "")),
        "agent_name": str(summary.get("agent_name", "")),
        "episodes": trajectory_episodes,
    }
    return summary, trajectory_summary


def main() -> None:
    env = None
    env_cfg = None
    stage = "build_runtime_env_cfg"

    try:
        resolved_request = _resolve_cli_task_request(
            task_id=args_cli.task_id,
            variant=args_cli.variant,
            scene=args_cli.scene,
            scene_id=args_cli.scene_id,
            layout_key=args_cli.layout_key,
            task_index=args_cli.task_index,
        )
        env_cfg = build_runtime_env_cfg(request=resolved_request)
        task_label = resolve_task_label(env_cfg.task_layout_key, env_cfg.task_index)
        _print_runtime_header(env_cfg, task_label=task_label)

        stage = "create_env"
        env = GeoMindFrankaIKRelMimicEnv(cfg=env_cfg)

        stage = "run_agent_rollout"
        with torch.inference_mode():
            rollout_summary, trajectory_summary = run_agent_rollout(
                env,
                task_label=task_label,
            )

        print("\n" + "=" * 72, flush=True)
        print("rollout_summary", flush=True)
        print("=" * 72, flush=True)
        print(json.dumps(_compact_summary_payload(rollout_summary), indent=2, sort_keys=True, ensure_ascii=False), flush=True)

        if args_cli.summary_json_out:
            _write_json(args_cli.summary_json_out, rollout_summary)
        if args_cli.trajectory_json_out:
            _write_json(args_cli.trajectory_json_out, trajectory_summary)
    except Exception:
        print_failure_context(stage, env_cfg=env_cfg)
        raise SystemExit(1)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()
