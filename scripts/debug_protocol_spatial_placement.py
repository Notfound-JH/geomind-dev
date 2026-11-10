#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import faulthandler
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass

from geomind_lab.bootstrap.isaaclab_path import ensure_isaaclab_source_on_path
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
DEFAULT_DEBUG_LAYOUT_KEY = "place_directional_scene65_base"
DEFAULT_DEBUG_TASK_INDEX = 0
DEFAULT_RESET_SETTLE_STEPS = 24
GRIPPER_OPEN_COMMAND = 1.0
GRIPPER_CLOSE_COMMAND = -1.0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an oracle/debug rollout for annotation-backed spatial placement tasks using "
            "the standard GeoMind IK-relative scripted grasp-and-place primitive."
        )
    )
    parser.add_argument("--task-id", type=str, default=None, help="Optional task id label for logging only.")
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
    parser.add_argument("--scene-id", type=int, default=None, help="Optional scene id override.")
    parser.add_argument(
        "--layout-key",
        type=str,
        default=None,
        help="Concrete annotation layout key to debug.",
    )
    parser.add_argument("--task-index", type=int, default=None, help="Task index inside the layout.")
    parser.add_argument(
        "--object-mode",
        type=str,
        default=None,
        help="Optional object representation mode override: concrete or geometric.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=("oracle", "random_gt"),
        default="oracle",
        help="Placement policy. random_gt keeps scripted motion but samples random legal GT targets.",
    )
    parser.add_argument("--agent-seed", type=int, default=0, help="Base random seed for random_gt.")
    parser.add_argument("--agent-episodes", type=int, default=1, help="Number of reset->grasp->place episodes to run.")
    parser.add_argument(
        "--reset-settle-steps",
        type=int,
        default=DEFAULT_RESET_SETTLE_STEPS,
        help="Zero-action settle frames after each reset before grasp begins.",
    )
    parser.add_argument(
        "--post-place-settle-steps",
        type=int,
        default=24,
        help="Hold/settle frames after release before reading final placement state.",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=120.0,
        help=(
            "Override episode time limit in seconds for the debug rollout runtime. "
            "Use a larger value to avoid timeout termination during scripted grasp/place."
        ),
    )
    parser.add_argument(
        "--allow-env-success-termination",
        action="store_true",
        default=False,
        help=(
            "Keep the environment success DoneTerm enabled. "
            "By default this debug runner disables success termination so release/settle checks can finish."
        ),
    )
    parser.add_argument(
        "--summary-json-out",
        type=str,
        default=None,
        help="Optional path to write the structured rollout summary JSON.",
    )
    parser.add_argument(
        "--trajectory-h5-out",
        type=str,
        default=None,
        help="Optional output H5 path (or directory for multiple episodes) for 10Hz trajectory data.",
    )
    parser.add_argument(
        "--trajectory-sample-hz",
        type=float,
        default=10.0,
        help="Target sampling frequency for --trajectory-h5-out.",
    )
    parser.add_argument(
        "--trajectory-segment-source",
        type=str,
        choices=("stage_v1", "event_kinematics_v2"),
        default=os.environ.get("TRAJECTORY_SEGMENT_SOURCE", "stage_v1"),
        help="Segmentation source for H5 labels. Defaults to stable stage_v1.",
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
        "--no-directional-zone-viz",
        action="store_true",
        default=False,
        help="Disable the green debug-line border for directional_region_placement success zones (GUI only).",
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
    print("GeoMind Spatial Placement Debug Failure", file=sys.stderr, flush=True)
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

from geomind_lab.data.trajectory import EpisodeCollector, export_episode_h5, resample_episode
from geomind_lab.mimic import GeoMindFrankaIKRelMimicEnv, GeoMindFrankaIKRelMimicEnvCfg
from geomind_lab.utils.headless_camera_video import RolloutCameraRecorder, create_rollout_camera_recorder
from geomind_lab.utils.headless_rollout_camera_hook import resolve_rollout_camera_video_dir
from geomind_lab.tasks.manager_based.manipulation.geomind import mdp as geomind_mdp
from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.common import get_asset_dimensions, get_asset_pose_w
from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.terminations import _object_stable


DOWN_FACING_QUAT = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


@dataclass(frozen=True)
class DirectionalPlacementProblem:
    task_label: str
    query_text: str
    target_object_id: str
    anchor_object_id: str
    anchor_xy: tuple[float, float]
    target_center_xy: tuple[float, float]
    target_half_extents: tuple[float, float]
    support_surface: str
    must_be_released: bool
    target_zone_bounds_xy: dict[str, float]
    xy_aabb_fully_inside_zone: bool = False
    target_xy_half_extent: tuple[float, float] = (0.0, 0.0)
    min_anchor_center_xy_distance: float | None = None
    settle_max_linear_speed: float | None = None
    settle_max_angular_speed: float | None = None


@dataclass(frozen=True)
class SupportTopPlacementProblem:
    task_label: str
    query_text: str
    target_object_id: str
    support_object_id: str
    support_center_xy: tuple[float, float]
    support_top_z: float
    support_size_xyz: tuple[float, float, float]
    target_size_xyz: tuple[float, float, float]
    contact_required: bool
    support_required: bool
    must_be_released: bool
    stable_after_settle_steps: int
    max_linear_speed_after_settle: float
    max_angular_speed_after_settle: float
    max_support_gap: float
    min_support_projection_overlap_ratio: float


@dataclass(frozen=True)
class ContainmentPlacementProblem:
    task_label: str
    query_text: str
    target_object_id: str
    container_reference_object_id: str
    container_center_xy: tuple[float, float]
    target_size_xyz: tuple[float, float, float]
    interior_half_extents_xy: tuple[float, float]
    interior_floor_top_z: float
    opening_plane_z: float
    must_be_released: bool
    stable_after_settle_steps: int
    max_linear_speed_after_settle: float
    max_angular_speed_after_settle: float
    max_floor_gap: float
    max_floor_penetration: float
    min_top_clearance_below_opening: float
    xy_inner_margin: float


def _resolve_cli_task_request():
    return resolve_task_request(
        task_id=args_cli.task_id,
        variant=args_cli.variant,
        scene=args_cli.scene,
        scene_id=args_cli.scene_id,
        layout_key=args_cli.layout_key or DEFAULT_DEBUG_LAYOUT_KEY,
        task_index=args_cli.task_index if args_cli.task_index is not None else DEFAULT_DEBUG_TASK_INDEX,
        object_mode=args_cli.object_mode,
    )


def build_runtime_env_cfg(*, request):
    base_cfg_cls = GeoMindFrankaIKRelMimicEnvCfg

    def __post_init__(self):
        self.task_id = str(request.task_id)
        self.task_variant = str(request.variant)
        self.scene_name = str(request.scene)
        self.object_mode = str(request.object_mode)
        self.scene_id = int(request.scene_id)
        self.task_layout_key = str(request.layout_key)
        self.task_index = int(request.task_index)
        self.scene.num_envs = 1
        self.sim.device = args_cli.device
        self.sim.use_fabric = not args_cli.disable_fabric

        base_cfg_cls.__post_init__(self)

        # Increase timeout budget for scripted pipelines (grasp + place + release + settle).
        requested_episode_length_s = float(getattr(args_cli, "episode_length_s", 120.0) or 120.0)
        self.episode_length_s = float(max(float(getattr(self, "episode_length_s", 30.0) or 30.0), requested_episode_length_s))

        # Prevent early environment success termination from interrupting scripted release/settle checks.
        if not bool(getattr(args_cli, "allow_env_success_termination", False)):
            if hasattr(self.terminations, "success"):
                delattr(self.terminations, "success")

        self.viewer.eye = tuple(float(v) for v in args_cli.viewer_eye)
        self.viewer.lookat = tuple(float(v) for v in args_cli.viewer_lookat)
        self.viz_directional_placement_zone = not bool(getattr(args_cli, "no_directional_zone_viz", False))

    runtime_cfg_cls = configclass(
        type(
            "RuntimeGeoMindSpatialPlacementDebugCfg",
            (base_cfg_cls,),
            {
                "__doc__": "Runtime GeoMind mimic cfg for spatial placement debug rollouts.",
                "task_id": str(request.task_id),
                "task_variant": str(request.variant),
                "scene_name": str(request.scene),
                "object_mode": str(request.object_mode),
                "scene_id": int(request.scene_id),
                "task_layout_key": str(request.layout_key),
                "task_index": int(request.task_index),
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
        family=task_metadata.get("family"),
        semantic_module=task_metadata.get("semantic_module"),
        skill_type=task_metadata.get("skill_type"),
    )
    print("=" * 72, flush=True)
    print("GeoMind Spatial Placement Debugger", flush=True)
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
    print(f"agent_seed:     {args_cli.agent_seed}", flush=True)
    print(f"agent_episodes: {args_cli.agent_episodes}", flush=True)
    print(f"reset_settle:   {args_cli.reset_settle_steps}", flush=True)
    print(f"post_place_settle:{args_cli.post_place_settle_steps}", flush=True)
    print(f"episode_length_s:{getattr(env_cfg, 'episode_length_s', '')}", flush=True)
    print(f"allow_env_success_termination:{bool(getattr(args_cli, 'allow_env_success_termination', False))}", flush=True)
    print(
        "task_protocol:  " + json.dumps(task_protocol, sort_keys=True, ensure_ascii=False),
        flush=True,
    )


def _write_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _trajectory_h5_path(episode_index: int) -> Path | None:
    raw = str(getattr(args_cli, "trajectory_h5_out", "") or "").strip()
    if not raw:
        return None
    configured = Path(raw).expanduser()
    if configured.suffix.lower() == ".h5":
        return configured if int(episode_index) == 0 else configured.with_name(
            f"{configured.stem}.episode_{int(episode_index):04d}.h5"
        )
    return configured / f"episode_{int(episode_index):04d}.h5"


def _env_int(name: str, default: int = 0) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw, 0)
    except ValueError:
        return int(default)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _create_episode_collector(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    episode_index: int,
) -> EpisodeCollector | None:
    if _trajectory_h5_path(episode_index) is None:
        return None
    task_spec = dict(getattr(env.cfg, "task_spec", {}) or {})
    trajectory_seed = int(args_cli.agent_seed) + int(episode_index)
    return EpisodeCollector(
        metadata={
            "task_id": str(dict(getattr(env.cfg, "task_metadata", {}) or {}).get("task_id", "")),
            "variant": str(getattr(env.cfg, "task_variant", "")),
            "layout_key": str(getattr(env.cfg, "task_layout_key", "")),
            "scene_id": int(getattr(env.cfg, "scene_id", -1)),
            "task_index": int(getattr(env.cfg, "task_index", -1)),
            "episode_index": int(episode_index),
            "layout_index": _env_int("LAYOUT_INDEX", 0),
            "rollout_index": int(episode_index),
            "trajectory_seed": trajectory_seed,
            "instruction": str(getattr(env.cfg, "task_instruction", "")),
            "layout_jitter_enabled": _env_flag("GEOMIND_LAYOUT_JITTER_ENABLE", False),
            "layout_jitter_seed": _env_int("GEOMIND_LAYOUT_JITTER_SEED", 0),
            "layout_shift_enabled": _env_flag("GEOMIND_LAYOUT_SHIFT_ENABLE", False),
            "layout_shift_seed": _env_int("GEOMIND_LAYOUT_SHIFT_SEED", 0),
            "layout_randomization_record": task_spec.get("applied_layout_randomization", {}),
            "initial_runtime_object_poses": task_spec.get("object_positions", {}),
            "trajectory_augmentation_enabled": False,
            "augmentation_record": {
                "trajectory_seed": trajectory_seed,
                "augmentation_enabled": False,
                "waypoint_order": [],
                "fallback_reason": "phase5_not_enabled",
            },
            "segmenter_version": str(args_cli.trajectory_segment_source),
            "segment_source": str(args_cli.trajectory_segment_source),
            "eef_action_quaternion_order": "xyzw",
            "timestamp_source": "episode_length_buf_times_step_dt",
        },
        sample_hz=float(args_cli.trajectory_sample_hz),
    )


def _export_episode_collector(collector: EpisodeCollector | None, *, episode_index: int) -> Path | None:
    h5_path = _trajectory_h5_path(episode_index)
    if collector is None or h5_path is None or not collector.records:
        return None
    sampled_episode = resample_episode(
        collector.as_dict(),
        target_hz=float(args_cli.trajectory_sample_hz),
        segment_source=str(args_cli.trajectory_segment_source),
    )
    return export_episode_h5(sampled_episode, h5_path)


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


def _step_with_target_pose(
    env: GeoMindFrankaIKRelMimicEnv,
    target_pose: torch.Tensor,
    gripper_action: float,
    *,
    steps: int,
    collector: EpisodeCollector | None = None,
    stage_name: str = "idle",
    target_name: str = "",
    require_reached: bool = False,
    xy_tolerance: float = 0.02,
    z_tolerance: float = 0.015,
    min_steps: int = 1,
) -> bool:
    target_position, _ = PoseUtils.unmake_pose(target_pose)
    reached = False
    for step_idx in range(max(1, int(steps))):
        action = _action_from_target_pose(env, target_pose, gripper_action)
        _, _, terminated, truncated, _ = env.step(action)
        _record_trajectory_step(
            env,
            collector=collector,
            stage_name=stage_name,
            action=action,
            target_pose=target_pose,
            gripper_action=gripper_action,
            target_name=target_name,
            terminated=terminated,
            truncated=truncated,
        )
        if should_reset(terminated, truncated):
            return False
        if require_reached and step_idx + 1 >= max(1, int(min_steps)):
            eef_position = _current_eef_position(env)
            xy_error = torch.linalg.vector_norm(eef_position[:2] - target_position[:2])
            z_error = torch.abs(eef_position[2] - target_position[2])
            if float(xy_error.item()) <= float(xy_tolerance) and float(z_error.item()) <= float(z_tolerance):
                reached = True
                break
    return bool(reached) if require_reached else True


def _current_object_position(env: GeoMindFrankaIKRelMimicEnv, object_name: str, *, env_id: int = 0) -> torch.Tensor:
    pos_w, _ = get_asset_pose_w(env, object_name)
    return pos_w[env_id, :3].clone().detach()


def _current_object_xy(env: GeoMindFrankaIKRelMimicEnv, object_name: str, *, env_id: int = 0) -> tuple[float, float]:
    position = _current_object_position(env, object_name, env_id=env_id)
    return (float(position[0].item()), float(position[1].item()))


def _current_object_linear_speed(env: GeoMindFrankaIKRelMimicEnv, object_name: str, *, env_id: int = 0) -> float | None:
    asset = env.scene[object_name]
    if not hasattr(asset, "data") or not hasattr(asset.data, "root_lin_vel_w"):
        return None
    velocity = asset.data.root_lin_vel_w[env_id, :3]
    return float(torch.linalg.vector_norm(velocity).item())


def _current_object_angular_speed(env: GeoMindFrankaIKRelMimicEnv, object_name: str, *, env_id: int = 0) -> float | None:
    asset = env.scene[object_name]
    if not hasattr(asset, "data") or not hasattr(asset.data, "root_ang_vel_w"):
        return None
    velocity = asset.data.root_ang_vel_w[env_id, :3]
    return float(torch.linalg.vector_norm(velocity).item())


def _current_eef_position(env: GeoMindFrankaIKRelMimicEnv, *, env_id: int = 0) -> torch.Tensor:
    pose = env.get_robot_eef_pose(_eef_name(env), env_ids=[env_id])[0]
    position, _ = PoseUtils.unmake_pose(pose)
    return position.clone().detach()


def _sim_time(env: GeoMindFrankaIKRelMimicEnv) -> tuple[float, int]:
    step_count = int(env.episode_length_buf[0].item())
    return float(step_count) * float(env.step_dt), step_count


def _pose_to_xyz_xyzw(pose: torch.Tensor) -> list[float]:
    position, rotation = PoseUtils.unmake_pose(pose)
    quat_wxyz = PoseUtils.quat_from_matrix(rotation.unsqueeze(0))[0]
    return [
        float(position[0].item()),
        float(position[1].item()),
        float(position[2].item()),
        float(quat_wxyz[1].item()),
        float(quat_wxyz[2].item()),
        float(quat_wxyz[3].item()),
        float(quat_wxyz[0].item()),
    ]


def _robot_joint_state(env: GeoMindFrankaIKRelMimicEnv) -> tuple[list[float], list[float], list[float]]:
    robot = env.scene["robot"]
    joint_pos = [float(value) for value in robot.data.joint_pos[0].detach().cpu().tolist()]
    joint_vel = [float(value) for value in robot.data.joint_vel[0].detach().cpu().tolist()]
    gripper_state: list[float] = []
    if hasattr(env.cfg, "gripper_joint_names"):
        joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        gripper_state = [float(robot.data.joint_pos[0, joint_id].item()) for joint_id in joint_ids]
    return joint_pos, joint_vel, gripper_state


def _record_trajectory_step(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    collector: EpisodeCollector | None,
    stage_name: str,
    action: torch.Tensor,
    target_pose: torch.Tensor,
    gripper_action: float,
    target_name: str,
    terminated,
    truncated,
) -> None:
    if collector is None:
        return
    timestamp_s, sim_step_index = _sim_time(env)
    joint_pos, joint_vel, gripper_state = _robot_joint_state(env)
    object_grasped = bool(target_name) and _object_grasped(env, target_name)
    collector.record_step(
        timestamp_s=timestamp_s,
        sim_step_index=sim_step_index,
        stage=stage_name,
        action_eef_abs=_pose_to_xyz_xyzw(target_pose),
        gripper_command=gripper_action,
        eef_pose_world=_pose_to_xyz_xyzw(env.get_robot_eef_pose(_eef_name(env), env_ids=[0])[0]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        gripper_state=gripper_state,
        object_grasped=object_grasped,
        terminated=bool(torch.any(terminated).item()) if isinstance(terminated, torch.Tensor) else bool(terminated),
        truncated=bool(torch.any(truncated).item()) if isinstance(truncated, torch.Tensor) else bool(truncated),
    )


def _current_object_positions_xy_map(env: GeoMindFrankaIKRelMimicEnv) -> dict[str, dict[str, list[float]]]:
    task_spec = dict(getattr(env.cfg, "task_spec", {}) or {})
    object_positions = dict(task_spec.get("object_positions", {}) or {})
    return {
        str(object_name): {"position": list(_current_object_xy(env, str(object_name)))}
        for object_name in object_positions.keys()
    }


def _object_height(env: GeoMindFrankaIKRelMimicEnv, object_name: str) -> float:
    resolver = getattr(env.cfg, "_resolve_object_runtime_height", None)
    if callable(resolver):
        resolved = resolver(object_name)
        if resolved is not None:
            return float(resolved)
    return 0.07


def _object_dimensions(env: GeoMindFrankaIKRelMimicEnv, object_name: str) -> tuple[float, float, float]:
    resolver = getattr(env.cfg, "_resolve_object_runtime_dimensions", None)
    if callable(resolver):
        resolved = resolver(object_name)
        if resolved is not None:
            return tuple(float(value) for value in resolved)
    height = _object_height(env, object_name)
    return (0.045, 0.045, float(height))


def _object_grasped(env: GeoMindFrankaIKRelMimicEnv, object_name: str, *, env_id: int = 0) -> bool:
    grasped = geomind_mdp.object_grasped(
        env,
        robot_cfg=SceneEntityCfg("robot"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg(object_name),
    )
    return bool(grasped[env_id].item())


def _resolve_spatial_variant(env: GeoMindFrankaIKRelMimicEnv) -> str:
    task_spec = dict(getattr(env.cfg, "task_spec", {}) or {})
    protocol = dict(task_spec.get("protocol", {}) or {})
    variant = str(protocol.get("variant", "")).strip()
    if variant:
        return variant
    gt_spec = dict(protocol.get("gt_spec", {}) or {})
    resolver = str(gt_spec.get("resolver", "")).strip()
    if resolver == "directional_region_placement":
        return "place_directional"
    if resolver == "support_top_placement":
        return "place_on_top"
    if resolver == "containment_placement":
        return "place_inside"
    return ""


def _build_directional_placement_problem(env: GeoMindFrankaIKRelMimicEnv, *, task_label: str) -> DirectionalPlacementProblem:
    task_spec = dict(getattr(env.cfg, "task_spec", {}) or {})
    protocol = dict(task_spec.get("protocol", {}) or {})
    if str(protocol.get("module_name", "")).strip() != "spatial_placement":
        raise ValueError("Expected protocol.module_name == 'spatial_placement'.")
    if str(protocol.get("variant", "")).strip() != "place_directional":
        raise ValueError("Expected protocol.variant == 'place_directional'.")

    gt_spec = dict(protocol.get("gt_spec", {}) or {})
    if str(gt_spec.get("resolver", "")).strip() != "directional_region_placement":
        raise ValueError("Expected gt_spec.resolver == 'directional_region_placement'.")
    if str(gt_spec.get("reference_frame", "")).strip() != "world":
        raise ValueError("Expected gt_spec.reference_frame == 'world'.")
    if str(gt_spec.get("anchor_mode", "")).strip() != "anchor_xy_only":
        raise ValueError("Expected gt_spec.anchor_mode == 'anchor_xy_only'.")
    if str(gt_spec.get("primary_axis", "")).strip() != "x" or str(gt_spec.get("primary_direction", "")).strip() != "positive":
        raise ValueError("Expected front direction to be encoded as world positive X.")

    target_object = dict(protocol.get("target_object", {}) or {})
    target_object_id = str(target_object.get("object_id", "")).strip()
    if not target_object_id:
        raise ValueError("Missing protocol.target_object.object_id.")

    anchor_object_id = str(gt_spec.get("anchor_object_id", "")).strip()
    if not anchor_object_id:
        anchor_object = dict(protocol.get("anchor_object", {}) or {})
        anchor_object_id = str(anchor_object.get("object_id", "")).strip()
    if not anchor_object_id:
        raise ValueError("Missing anchor object id for directional placement.")

    target_zone = dict(gt_spec.get("target_zone", {}) or {})
    center_offset = target_zone.get("center_offset_from_anchor", [])
    half_extents = target_zone.get("half_extents", [])
    if not (isinstance(center_offset, list) and len(center_offset) >= 2):
        raise ValueError("directional placement target_zone.center_offset_from_anchor must be length-2.")
    if not (isinstance(half_extents, list) and len(half_extents) >= 2):
        raise ValueError("directional placement target_zone.half_extents must be length-2.")

    runtime_object_positions = _current_object_positions_xy_map(env)
    anchor_position = runtime_object_positions.get(anchor_object_id, {}).get("position", [])
    if not (isinstance(anchor_position, list) and len(anchor_position) >= 2):
        raise ValueError(f"Unable to resolve anchor position for {anchor_object_id!r}.")

    anchor_xy = (float(anchor_position[0]), float(anchor_position[1]))
    target_center_xy = (
        float(anchor_xy[0]) + float(center_offset[0]),
        float(anchor_xy[1]) + float(center_offset[1]),
    )
    target_half_extents = (float(half_extents[0]), float(half_extents[1]))

    xy_aabb = bool(gt_spec.get("xy_aabb_fully_inside_zone", False))
    hx_obj = 0.0
    hy_obj = 0.0
    if xy_aabb:
        dims = get_asset_dimensions(env, target_object_id)
        if dims is not None:
            hx_obj = 0.5 * float(dims[0])
            hy_obj = 0.5 * float(dims[1])

    min_sep_raw = gt_spec.get("min_anchor_center_xy_distance", None)
    min_sep = float(min_sep_raw) if min_sep_raw is not None else None

    m_lin_raw = gt_spec.get("max_linear_speed_after_settle", None)
    m_ang_raw = gt_spec.get("max_angular_speed_after_settle", None)
    settle_lin = float(m_lin_raw) if m_lin_raw is not None else None
    settle_ang = float(m_ang_raw) if m_ang_raw is not None else None

    return DirectionalPlacementProblem(
        task_label=str(task_label),
        query_text=str(protocol.get("query_text", "")),
        target_object_id=target_object_id,
        anchor_object_id=anchor_object_id,
        anchor_xy=anchor_xy,
        target_center_xy=target_center_xy,
        target_half_extents=target_half_extents,
        support_surface=str(gt_spec.get("support_surface", "")),
        must_be_released=bool(gt_spec.get("must_be_released", False)),
        target_zone_bounds_xy={
            "x_min": float(target_center_xy[0] - target_half_extents[0]),
            "x_max": float(target_center_xy[0] + target_half_extents[0]),
            "y_min": float(target_center_xy[1] - target_half_extents[1]),
            "y_max": float(target_center_xy[1] + target_half_extents[1]),
        },
        xy_aabb_fully_inside_zone=xy_aabb,
        target_xy_half_extent=(float(hx_obj), float(hy_obj)),
        min_anchor_center_xy_distance=min_sep,
        settle_max_linear_speed=settle_lin,
        settle_max_angular_speed=settle_ang,
    )


def _build_support_top_placement_problem(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    task_label: str,
) -> SupportTopPlacementProblem:
    task_spec = dict(getattr(env.cfg, "task_spec", {}) or {})
    protocol = dict(task_spec.get("protocol", {}) or {})
    if str(protocol.get("module_name", "")).strip() != "spatial_placement":
        raise ValueError("Expected protocol.module_name == 'spatial_placement'.")
    if str(protocol.get("variant", "")).strip() != "place_on_top":
        raise ValueError("Expected protocol.variant == 'place_on_top'.")

    gt_spec = dict(protocol.get("gt_spec", {}) or {})
    if str(gt_spec.get("resolver", "")).strip() != "support_top_placement":
        raise ValueError("Expected gt_spec.resolver == 'support_top_placement'.")
    if str(gt_spec.get("reference_frame", "")).strip() != "world":
        raise ValueError("Expected gt_spec.reference_frame == 'world'.")

    target_object_id = str(gt_spec.get("target_object_id", "")).strip()
    if not target_object_id:
        target_object = dict(protocol.get("target_object", {}) or {})
        target_object_id = str(target_object.get("object_id", "")).strip()
    if not target_object_id:
        raise ValueError("Missing target object id for support-top placement.")

    support_object_id = str(gt_spec.get("support_object_id", "")).strip()
    if not support_object_id:
        support_object = dict(protocol.get("support_object", {}) or {})
        support_object_id = str(support_object.get("object_id", "")).strip()
    if not support_object_id:
        raise ValueError("Missing support object id for support-top placement.")

    support_position = _current_object_position(env, support_object_id)
    support_dimensions = _object_dimensions(env, support_object_id)
    target_dimensions = _object_dimensions(env, target_object_id)
    support_center_xy = (float(support_position[0].item()), float(support_position[1].item()))
    support_top_z = float(support_position[2].item()) + 0.5 * float(support_dimensions[2])

    return SupportTopPlacementProblem(
        task_label=str(task_label),
        query_text=str(protocol.get("query_text", "")),
        target_object_id=target_object_id,
        support_object_id=support_object_id,
        support_center_xy=support_center_xy,
        support_top_z=float(support_top_z),
        support_size_xyz=tuple(float(value) for value in support_dimensions),
        target_size_xyz=tuple(float(value) for value in target_dimensions),
        contact_required=bool(gt_spec.get("contact_required", False)),
        support_required=bool(gt_spec.get("support_required", False)),
        must_be_released=bool(gt_spec.get("must_be_released", False)),
        stable_after_settle_steps=max(0, int(gt_spec.get("stable_after_settle_steps", 0) or 0)),
        max_linear_speed_after_settle=float(gt_spec.get("max_linear_speed_after_settle", 0.05) or 0.05),
        max_angular_speed_after_settle=float(gt_spec.get("max_angular_speed_after_settle", 0.4) or 0.4),
        max_support_gap=float(gt_spec.get("max_support_gap", 0.015) or 0.015),
        min_support_projection_overlap_ratio=float(
            gt_spec.get("min_support_projection_overlap_ratio", 0.72) or 0.72
        ),
    )


def _build_containment_placement_problem(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    task_label: str,
) -> ContainmentPlacementProblem:
    task_spec = dict(getattr(env.cfg, "task_spec", {}) or {})
    protocol = dict(task_spec.get("protocol", {}) or {})
    if str(protocol.get("module_name", "")).strip() != "spatial_placement":
        raise ValueError("Expected protocol.module_name == 'spatial_placement'.")
    variant = str(protocol.get("variant", "")).strip()
    if variant not in {"place_inside", "place_inside_boundary_control"}:
        raise ValueError("Expected protocol.variant == 'place_inside' or 'place_inside_boundary_control'.")

    gt_spec = dict(protocol.get("gt_spec", {}) or {})
    if str(gt_spec.get("resolver", "")).strip() != "containment_placement":
        raise ValueError("Expected gt_spec.resolver == 'containment_placement'.")
    if str(gt_spec.get("reference_frame", "")).strip() != "world":
        raise ValueError("Expected gt_spec.reference_frame == 'world'.")

    target_object_id = str(gt_spec.get("target_object_id", "")).strip()
    if not target_object_id:
        target_object = dict(protocol.get("target_object", {}) or {})
        target_object_id = str(target_object.get("object_id", "")).strip()
    if not target_object_id:
        raise ValueError("Missing target object id for containment placement.")

    container_reference_object_id = str(gt_spec.get("container_reference_object_id", "")).strip()
    if not container_reference_object_id:
        raise ValueError("Missing container_reference_object_id for containment placement.")

    container_position = _current_object_position(env, container_reference_object_id)
    container_center_xy = (float(container_position[0].item()), float(container_position[1].item()))
    target_dimensions = _object_dimensions(env, target_object_id)

    interior_half_extents_xy = gt_spec.get("interior_half_extents_xy", [])
    if not (isinstance(interior_half_extents_xy, list) and len(interior_half_extents_xy) >= 2):
        raise ValueError("containment placement interior_half_extents_xy must be length-2.")

    interior_floor_top_offset = float(gt_spec.get("interior_floor_top_offset", 0.0) or 0.0)
    opening_plane_z_offset = float(gt_spec.get("opening_plane_z_offset", 0.0) or 0.0)

    return ContainmentPlacementProblem(
        task_label=str(task_label),
        query_text=str(protocol.get("query_text", "")),
        target_object_id=target_object_id,
        container_reference_object_id=container_reference_object_id,
        container_center_xy=container_center_xy,
        target_size_xyz=tuple(float(value) for value in target_dimensions),
        interior_half_extents_xy=(
            float(interior_half_extents_xy[0]),
            float(interior_half_extents_xy[1]),
        ),
        interior_floor_top_z=float(container_position[2].item()) + interior_floor_top_offset,
        opening_plane_z=float(container_position[2].item()) + opening_plane_z_offset,
        must_be_released=bool(gt_spec.get("must_be_released", False)),
        stable_after_settle_steps=max(0, int(gt_spec.get("stable_after_settle_steps", 0) or 0)),
        max_linear_speed_after_settle=float(gt_spec.get("max_linear_speed_after_settle", 0.05) or 0.05),
        max_angular_speed_after_settle=float(gt_spec.get("max_angular_speed_after_settle", 0.4) or 0.4),
        max_floor_gap=float(gt_spec.get("max_floor_gap", 0.02) or 0.02),
        max_floor_penetration=float(gt_spec.get("max_floor_penetration", 0.004) or 0.004),
        min_top_clearance_below_opening=float(gt_spec.get("min_top_clearance_below_opening", 0.0) or 0.0),
        xy_inner_margin=float(gt_spec.get("xy_inner_margin", 0.0) or 0.0),
    )


def run_scripted_grasp_for_target(
    env: GeoMindFrankaIKRelMimicEnv,
    target_name: str,
    *,
    collector: EpisodeCollector | None = None,
) -> dict[str, object]:
    target_position = _current_object_position(env, target_name)
    object_height = _object_height(env, target_name)
    target_surface_z = float(target_position[2].item())
    approach_z = float(target_surface_z + max(0.12, 1.5 * object_height))
    grasp_z = float(target_surface_z + max(0.0, 0.05 * object_height))
    lift_z = float(target_surface_z + max(0.18, 2.0 * object_height))

    approach_pose = _pose_from_position_and_quat(
        torch.tensor([target_position[0].item(), target_position[1].item(), approach_z], device=env.device, dtype=torch.float32),
        DOWN_FACING_QUAT.to(device=env.device),
    )

    approach_reached = _step_with_target_pose(
        env,
        approach_pose,
        GRIPPER_OPEN_COMMAND,
        steps=60,
        collector=collector,
        stage_name="approach",
        target_name=target_name,
        require_reached=True,
        xy_tolerance=0.03,
        z_tolerance=0.03,
        min_steps=12,
    )

    locked_target_position = _current_object_position(env, target_name) if approach_reached else target_position
    grasp_pose = _pose_from_position_and_quat(
        torch.tensor(
            [locked_target_position[0].item(), locked_target_position[1].item(), grasp_z],
            device=env.device,
            dtype=torch.float32,
        ),
        DOWN_FACING_QUAT.to(device=env.device),
    )
    lift_pose = _pose_from_position_and_quat(
        torch.tensor(
            [locked_target_position[0].item(), locked_target_position[1].item(), lift_z],
            device=env.device,
            dtype=torch.float32,
        ),
        DOWN_FACING_QUAT.to(device=env.device),
    )

    descend_reached = False
    completed = approach_reached
    if completed:
        descend_reached = _step_with_target_pose(
            env,
            grasp_pose,
            GRIPPER_OPEN_COMMAND,
            steps=80,
            collector=collector,
            stage_name="descend",
            target_name=target_name,
            require_reached=True,
            xy_tolerance=0.018,
            z_tolerance=0.015,
            min_steps=12,
        )
        completed = completed and descend_reached

    pre_close_eef_position = _current_eef_position(env)
    pre_close_target_position, _ = PoseUtils.unmake_pose(grasp_pose)
    pre_close_xy_error = torch.linalg.vector_norm(pre_close_eef_position[:2] - pre_close_target_position[:2])
    pre_close_z_error = torch.abs(pre_close_eef_position[2] - pre_close_target_position[2])

    completed = (
        completed
        and _step_with_target_pose(
            env, grasp_pose, GRIPPER_CLOSE_COMMAND, steps=28,
            collector=collector, stage_name="close", target_name=target_name,
        )
        and _step_with_target_pose(
            env, lift_pose, GRIPPER_CLOSE_COMMAND, steps=24,
            collector=collector, stage_name="lift", target_name=target_name,
        )
        and _step_with_target_pose(
            env, lift_pose, GRIPPER_CLOSE_COMMAND, steps=8,
            collector=collector, stage_name="hold", target_name=target_name,
        )
    )
    final_position = _current_object_position(env, target_name)
    grasp_sensor_triggered = completed and _object_grasped(env, target_name)
    object_lifted = float(final_position[2].item()) > float(target_surface_z + max(0.08, 1.2 * object_height))
    grasp_success = completed and (grasp_sensor_triggered or object_lifted)
    eef_position = _current_eef_position(env)
    grasp_offset_xy = (
        float(final_position[0].item() - eef_position[0].item()),
        float(final_position[1].item() - eef_position[1].item()),
    )

    return {
        "target_name": target_name,
        "target_height": float(object_height),
        "target_surface_z": float(target_surface_z),
        "approach_z": float(approach_z),
        "grasp_z": float(grasp_z),
        "lift_z": float(lift_z),
        "locked_target_xy": [float(locked_target_position[0].item()), float(locked_target_position[1].item())],
        "approach_reached": bool(approach_reached),
        "descend_reached": bool(descend_reached),
        "pre_close_eef_position": [float(value) for value in pre_close_eef_position.tolist()],
        "pre_close_target_position": [float(value) for value in pre_close_target_position.tolist()],
        "pre_close_xy_error": float(pre_close_xy_error.item()),
        "pre_close_z_error": float(pre_close_z_error.item()),
        "execution_completed": bool(completed),
        "terminated_early": not bool(completed),
        "grasp_success": bool(grasp_success),
        "grasp_sensor_triggered": bool(grasp_sensor_triggered),
        "object_lifted": bool(object_lifted),
        "grasp_offset_xy": [float(grasp_offset_xy[0]), float(grasp_offset_xy[1])],
        "final_target_position": [float(value) for value in final_position.tolist()],
        "eef_position_final": [float(value) for value in eef_position.tolist()],
    }


def run_scripted_place_for_target(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    target_name: str,
    target_pos_xy: tuple[float, float],
    target_surface_z: float,
    object_height: float,
    post_place_settle_steps: int,
    grasp_offset_xy: tuple[float, float] = (0.0, 0.0),
    collector: EpisodeCollector | None = None,
) -> dict[str, object]:
    target_x, target_y = (float(target_pos_xy[0]), float(target_pos_xy[1]))
    command_x = float(target_x - float(grasp_offset_xy[0]))
    command_y = float(target_y - float(grasp_offset_xy[1]))
    approach_z = float(target_surface_z + max(0.14, 1.5 * object_height))
    place_z = float(target_surface_z + max(0.02, 0.1 * object_height))
    retreat_z = float(target_surface_z + max(0.18, 2.0 * object_height))

    approach_pose = _pose_from_position_and_quat(
        torch.tensor([command_x, command_y, approach_z], device=env.device, dtype=torch.float32),
        DOWN_FACING_QUAT.to(device=env.device),
    )
    place_pose = _pose_from_position_and_quat(
        torch.tensor([command_x, command_y, place_z], device=env.device, dtype=torch.float32),
        DOWN_FACING_QUAT.to(device=env.device),
    )
    retreat_pose = _pose_from_position_and_quat(
        torch.tensor([command_x, command_y, retreat_z], device=env.device, dtype=torch.float32),
        DOWN_FACING_QUAT.to(device=env.device),
    )

    completed = (
        _step_with_target_pose(
            env, approach_pose, GRIPPER_CLOSE_COMMAND, steps=24,
            collector=collector, stage_name="transit_xy", target_name=target_name,
        )
        and _step_with_target_pose(
            env, place_pose, GRIPPER_CLOSE_COMMAND, steps=18,
            collector=collector, stage_name="descend_place", target_name=target_name,
        )
        and _step_with_target_pose(
            env, place_pose, GRIPPER_OPEN_COMMAND, steps=18,
            collector=collector, stage_name="open", target_name=target_name,
        )
        and _step_with_target_pose(
            env, retreat_pose, GRIPPER_OPEN_COMMAND, steps=18,
            collector=collector, stage_name="retreat", target_name=target_name,
        )
    )
    settle_applied = False
    if completed and int(post_place_settle_steps) > 0:
        settle_applied = _step_with_target_pose(
            env,
            retreat_pose,
            GRIPPER_OPEN_COMMAND,
            steps=int(post_place_settle_steps),
            collector=collector,
            stage_name="hold",
            target_name=target_name,
        )
        completed = completed and settle_applied

    final_position = _current_object_position(env, target_name)
    release_sensor_triggered = completed and (not _object_grasped(env, target_name))
    object_settled_near_surface = float(final_position[2].item()) <= float(target_surface_z + max(0.05, object_height))
    release_detected = completed and (release_sensor_triggered or object_settled_near_surface)
    return {
        "target_name": target_name,
        "target_pos_xy": [target_x, target_y],
        "command_pos_xy": [command_x, command_y],
        "approach_z": float(approach_z),
        "place_z": float(place_z),
        "retreat_z": float(retreat_z),
        "execution_completed": bool(completed),
        "terminated_early": not bool(completed),
        "release_detected": bool(release_detected),
        "release_sensor_triggered": bool(release_sensor_triggered),
        "object_settled_near_surface": bool(object_settled_near_surface),
        "settle_applied": bool(settle_applied),
        "pred_pos_xy": [float(final_position[0].item()), float(final_position[1].item())],
        "final_target_position": [float(value) for value in final_position.tolist()],
    }


def _zone_satisfied(
    *,
    pred_pos_xy: tuple[float, float],
    target_center_xy: tuple[float, float],
    target_half_extents: tuple[float, float],
) -> bool:
    dx = float(pred_pos_xy[0] - target_center_xy[0])
    dy = float(pred_pos_xy[1] - target_center_xy[1])
    return abs(dx) <= float(target_half_extents[0]) and abs(dy) <= float(target_half_extents[1])


def _directional_placement_success_for_script(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    problem: DirectionalPlacementProblem,
    pred_pos_xy: tuple[float, float],
) -> bool:
    """Match ``mdp.terminations._directional_region_placement_reached`` for directional layouts (single env debug)."""
    cx, cy = problem.target_center_xy
    hx, hy = problem.target_half_extents
    px, py = pred_pos_xy
    dx = px - cx
    dy = py - cy
    if problem.xy_aabb_fully_inside_zone:
        ox, oy = problem.target_xy_half_extent
        eff_hx = float(hx) - ox
        eff_hy = float(hy) - oy
        if eff_hx <= 1.0e-6 or eff_hy <= 1.0e-6:
            return False
        if abs(dx) > eff_hx or abs(dy) > eff_hy:
            return False
    else:
        if abs(dx) > float(hx) or abs(dy) > float(hy):
            return False

    if problem.min_anchor_center_xy_distance is not None:
        ax, ay = problem.anchor_xy
        dist = math.hypot(px - ax, py - ay)
        if dist < float(problem.min_anchor_center_xy_distance):
            return False

    if problem.settle_max_linear_speed is not None and problem.settle_max_angular_speed is not None:
        stable = _object_stable(
            env,
            problem.target_object_id,
            max_linear_speed=float(problem.settle_max_linear_speed),
            max_angular_speed=float(problem.settle_max_angular_speed),
        )
        if not bool(stable[0].item()):
            return False

    return True


def _projection_overlap_ratio(
    *,
    target_center_xy: tuple[float, float],
    target_size_xy: tuple[float, float],
    support_center_xy: tuple[float, float],
    support_size_xy: tuple[float, float],
) -> float:
    target_half_x = 0.5 * float(target_size_xy[0])
    target_half_y = 0.5 * float(target_size_xy[1])
    support_half_x = 0.5 * float(support_size_xy[0])
    support_half_y = 0.5 * float(support_size_xy[1])

    target_x_min = float(target_center_xy[0] - target_half_x)
    target_x_max = float(target_center_xy[0] + target_half_x)
    target_y_min = float(target_center_xy[1] - target_half_y)
    target_y_max = float(target_center_xy[1] + target_half_y)

    support_x_min = float(support_center_xy[0] - support_half_x)
    support_x_max = float(support_center_xy[0] + support_half_x)
    support_y_min = float(support_center_xy[1] - support_half_y)
    support_y_max = float(support_center_xy[1] + support_half_y)

    overlap_x = max(0.0, min(target_x_max, support_x_max) - max(target_x_min, support_x_min))
    overlap_y = max(0.0, min(target_y_max, support_y_max) - max(target_y_min, support_y_min))
    target_area = max(1.0e-8, float(target_size_xy[0]) * float(target_size_xy[1]))
    return float((overlap_x * overlap_y) / target_area)


def _evaluate_support_top_stack_satisfaction(
    *,
    placement_execution_success: bool,
    released_after_settle: bool,
    object_above_support: bool,
    object_stable_after_settle: bool,
    contact_required: bool,
    support_required: bool,
    must_be_released: bool,
    support_gap_abs: float,
    max_support_gap: float,
    support_projection_overlap_ratio: float,
    min_support_projection_overlap_ratio: float,
) -> dict[str, bool]:
    support_gap_within_limit = float(support_gap_abs) <= float(max_support_gap)
    support_overlap_sufficient = float(support_projection_overlap_ratio) >= float(min_support_projection_overlap_ratio)
    support_contact_established = bool(support_gap_within_limit and support_overlap_sufficient)
    required_release_satisfied = (not bool(must_be_released)) or bool(released_after_settle)
    required_contact_satisfied = (not bool(contact_required)) or bool(support_contact_established)
    required_support_satisfied = (not bool(support_required)) or bool(object_above_support and support_contact_established)
    stack_satisfied = bool(
        placement_execution_success
        and required_release_satisfied
        and required_contact_satisfied
        and required_support_satisfied
        and object_stable_after_settle
    )
    return {
        "support_gap_within_limit": bool(support_gap_within_limit),
        "support_overlap_sufficient": bool(support_overlap_sufficient),
        "support_contact_established": bool(support_contact_established),
        "required_release_satisfied": bool(required_release_satisfied),
        "required_contact_satisfied": bool(required_contact_satisfied),
        "required_support_satisfied": bool(required_support_satisfied),
        "stack_satisfied": bool(stack_satisfied),
    }


def _containment_local_metrics(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    target_object_id: str,
    container_reference_object_id: str,
    target_size_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    target_pos_w, target_quat_w = get_asset_pose_w(env, target_object_id)
    container_pos_w, container_quat_w = get_asset_pose_w(env, container_reference_object_id)

    target_rot = PoseUtils.matrix_from_quat(target_quat_w)[0]
    container_rot = PoseUtils.matrix_from_quat(container_quat_w)[0]
    relative_rot = torch.matmul(container_rot.transpose(0, 1), target_rot)
    target_half_size = torch.tensor(
        [0.5 * float(target_size_xyz[0]), 0.5 * float(target_size_xyz[1]), 0.5 * float(target_size_xyz[2])],
        device=env.device,
        dtype=torch.float32,
    )
    projected_half_extents = torch.matmul(torch.abs(relative_rot), target_half_size.unsqueeze(-1)).squeeze(-1)
    relative_center = torch.matmul(
        container_rot.transpose(0, 1),
        (target_pos_w[0, :3] - container_pos_w[0, :3]).unsqueeze(-1),
    ).squeeze(-1)

    return {
        "pred_center_w": [float(value) for value in target_pos_w[0, :3].tolist()],
        "relative_center": [float(value) for value in relative_center.tolist()],
        "projected_half_extents": [float(value) for value in projected_half_extents.tolist()],
    }


def run_directional_placement_episode(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    episode_index: int,
    task_label: str,
    rng: random.Random | None = None,
    recorder: RolloutCameraRecorder | None = None,
    collector: EpisodeCollector | None = None,
) -> dict[str, Any]:
    env.reset()
    settle_env_after_reset(env, args_cli.reset_settle_steps)
    if recorder is not None:
        recorder.capture_now()

    problem = _build_directional_placement_problem(env, task_label=task_label)
    target_pos_xy = tuple(float(value) for value in problem.target_center_xy)
    random_gt_used = False
    if str(args_cli.agent).strip().lower() == "random_gt":
        rng = rng or random.Random(int(args_cli.agent_seed))
        sampled_x = float(
            problem.target_center_xy[0] + rng.uniform(-float(problem.target_half_extents[0]), float(problem.target_half_extents[0]))
        )
        sampled_y = float(
            problem.target_center_xy[1] + rng.uniform(-float(problem.target_half_extents[1]), float(problem.target_half_extents[1]))
        )
        target_pos_xy = (sampled_x, sampled_y)
        random_gt_used = True

    grasp_summary = run_scripted_grasp_for_target(env, problem.target_object_id, collector=collector)
    if bool(grasp_summary["grasp_success"]):
        place_summary = run_scripted_place_for_target(
            env,
            target_name=problem.target_object_id,
            target_pos_xy=target_pos_xy,
            target_surface_z=float(grasp_summary["target_surface_z"]),
            object_height=float(grasp_summary["target_height"]),
            post_place_settle_steps=args_cli.post_place_settle_steps,
            grasp_offset_xy=tuple(float(value) for value in grasp_summary["grasp_offset_xy"]),
            collector=collector,
        )
    else:
        place_summary = {
            "target_name": problem.target_object_id,
            "target_pos_xy": [float(target_pos_xy[0]), float(target_pos_xy[1])],
            "execution_completed": False,
            "terminated_early": False,
            "release_detected": False,
            "settle_applied": False,
            "pred_pos_xy": list(_current_object_xy(env, problem.target_object_id)),
        }

    pred_pos_xy = tuple(float(value) for value in place_summary["pred_pos_xy"])
    placement_execution_success = bool(place_summary["execution_completed"]) and bool(place_summary["release_detected"])
    settle_applied = bool(place_summary["settle_applied"]) or int(args_cli.post_place_settle_steps) <= 0
    valid_episode = bool(grasp_summary["grasp_success"]) and placement_execution_success and settle_applied
    zone_satisfied = valid_episode and _directional_placement_success_for_script(
        env,
        problem=problem,
        pred_pos_xy=pred_pos_xy,
    )
    error_xy = (
        float(pred_pos_xy[0] - problem.target_center_xy[0]),
        float(pred_pos_xy[1] - problem.target_center_xy[1]),
    )
    l2_error = float(math.hypot(error_xy[0], error_xy[1]))

    result = {
        "agent_name": str(args_cli.agent),
        "episode_index": int(episode_index),
        "task_label": str(task_label),
        "scene_id": int(getattr(env.cfg, "scene_id", -1)),
        "layout_key": str(getattr(env.cfg, "task_layout_key", "")),
        "task_index": int(getattr(env.cfg, "task_index", -1)),
        "query_text": str(problem.query_text),
        "target_object_id": str(problem.target_object_id),
        "anchor_object_id": str(problem.anchor_object_id),
        "anchor_xy": [float(problem.anchor_xy[0]), float(problem.anchor_xy[1])],
        "target_center_xy": [float(problem.target_center_xy[0]), float(problem.target_center_xy[1])],
        "target_pos_xy_chosen": [float(target_pos_xy[0]), float(target_pos_xy[1])],
        "random_gt_used": bool(random_gt_used),
        "target_half_extents": [float(problem.target_half_extents[0]), float(problem.target_half_extents[1])],
        "target_zone_bounds_xy": dict(problem.target_zone_bounds_xy),
        "support_surface": str(problem.support_surface),
        "must_be_released": bool(problem.must_be_released),
        "pred_pos_xy": [float(pred_pos_xy[0]), float(pred_pos_xy[1])],
        "zone_error_xy": [float(error_xy[0]), float(error_xy[1])],
        "l2_error": float(l2_error),
        "valid_episode": bool(valid_episode),
        "zone_satisfied": bool(zone_satisfied),
        "grasp_execution_success": bool(grasp_summary["grasp_success"]),
        "placement_execution_success": bool(placement_execution_success),
        "release_detected": bool(place_summary["release_detected"]),
        "grasp_execution": grasp_summary,
        "placement_execution": place_summary,
    }

    print("\n" + "=" * 72, flush=True)
    print("episode_result", flush=True)
    print("=" * 72, flush=True)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return result


def run_support_top_placement_episode(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    episode_index: int,
    task_label: str,
    rng: random.Random | None = None,
    recorder: RolloutCameraRecorder | None = None,
    collector: EpisodeCollector | None = None,
) -> dict[str, Any]:
    env.reset()
    settle_env_after_reset(env, args_cli.reset_settle_steps)
    if recorder is not None:
        recorder.capture_now()

    problem = _build_support_top_placement_problem(env, task_label=task_label)
    effective_post_place_settle_steps = max(int(args_cli.post_place_settle_steps), int(problem.stable_after_settle_steps))

    target_pos_xy = tuple(float(v) for v in problem.support_center_xy)
    random_gt_used = False
    if str(args_cli.agent).strip().lower() == "random_gt":
        rng = rng or random.Random(int(args_cli.agent_seed))
        margin_x = max(0.0, 0.5 * float(problem.support_size_xyz[0]) - 0.35 * float(problem.target_size_xyz[0]))
        margin_y = max(0.0, 0.5 * float(problem.support_size_xyz[1]) - 0.35 * float(problem.target_size_xyz[1]))
        target_pos_xy = (
            float(problem.support_center_xy[0] + rng.uniform(-margin_x, margin_x)),
            float(problem.support_center_xy[1] + rng.uniform(-margin_y, margin_y)),
        )
        random_gt_used = True

    grasp_summary = run_scripted_grasp_for_target(env, problem.target_object_id, collector=collector)
    if bool(grasp_summary["grasp_success"]):
        place_summary = run_scripted_place_for_target(
            env,
            target_name=problem.target_object_id,
            target_pos_xy=target_pos_xy,
            target_surface_z=float(problem.support_top_z),
            object_height=float(problem.target_size_xyz[2]),
            post_place_settle_steps=effective_post_place_settle_steps,
            grasp_offset_xy=tuple(float(value) for value in grasp_summary["grasp_offset_xy"]),
            collector=collector,
        )
    else:
        place_summary = {
            "target_name": problem.target_object_id,
            "target_pos_xy": [float(problem.support_center_xy[0]), float(problem.support_center_xy[1])],
            "execution_completed": False,
            "terminated_early": False,
            "release_detected": False,
            "settle_applied": False,
            "pred_pos_xy": list(_current_object_xy(env, problem.target_object_id)),
        }

    final_position = _current_object_position(env, problem.target_object_id)
    pred_pos_xy = (float(final_position[0].item()), float(final_position[1].item()))
    pred_center_z = float(final_position[2].item())
    target_height = float(problem.target_size_xyz[2])
    expected_center_z = float(problem.support_top_z + 0.5 * target_height)
    target_bottom_z = float(pred_center_z - 0.5 * target_height)
    support_gap_abs = float(abs(target_bottom_z - problem.support_top_z))
    support_projection_overlap_ratio = _projection_overlap_ratio(
        target_center_xy=pred_pos_xy,
        target_size_xy=(float(problem.target_size_xyz[0]), float(problem.target_size_xyz[1])),
        support_center_xy=problem.support_center_xy,
        support_size_xy=(float(problem.support_size_xyz[0]), float(problem.support_size_xyz[1])),
    )
    object_above_support = bool(pred_center_z > float(problem.support_top_z + 0.25 * target_height))
    released_after_settle = not _object_grasped(env, problem.target_object_id)
    target_linear_speed_after_settle = _current_object_linear_speed(env, problem.target_object_id)
    target_angular_speed_after_settle = _current_object_angular_speed(env, problem.target_object_id)
    object_stable_after_settle = bool(
        target_linear_speed_after_settle is not None
        and target_angular_speed_after_settle is not None
        and target_linear_speed_after_settle <= float(problem.max_linear_speed_after_settle)
        and target_angular_speed_after_settle <= float(problem.max_angular_speed_after_settle)
    )
    placement_execution_success = bool(place_summary["execution_completed"]) and bool(place_summary["release_detected"])
    valid_episode = bool(grasp_summary["grasp_success"]) and bool(placement_execution_success)
    stack_eval = _evaluate_support_top_stack_satisfaction(
        placement_execution_success=placement_execution_success,
        released_after_settle=released_after_settle,
        object_above_support=object_above_support,
        object_stable_after_settle=object_stable_after_settle,
        contact_required=problem.contact_required,
        support_required=problem.support_required,
        must_be_released=problem.must_be_released,
        support_gap_abs=support_gap_abs,
        max_support_gap=problem.max_support_gap,
        support_projection_overlap_ratio=support_projection_overlap_ratio,
        min_support_projection_overlap_ratio=problem.min_support_projection_overlap_ratio,
    )
    stack_satisfied = bool(valid_episode and stack_eval["stack_satisfied"])

    result = {
        "agent_name": str(args_cli.agent),
        "episode_index": int(episode_index),
        "task_label": str(task_label),
        "scene_id": int(getattr(env.cfg, "scene_id", -1)),
        "layout_key": str(getattr(env.cfg, "task_layout_key", "")),
        "task_index": int(getattr(env.cfg, "task_index", -1)),
        "query_text": str(problem.query_text),
        "target_object_id": str(problem.target_object_id),
        "support_object_id": str(problem.support_object_id),
        "support_center_xy": [float(problem.support_center_xy[0]), float(problem.support_center_xy[1])],
        "target_pos_xy_chosen": [float(target_pos_xy[0]), float(target_pos_xy[1])],
        "random_gt_used": bool(random_gt_used),
        "support_size_xyz": [float(value) for value in problem.support_size_xyz],
        "target_size_xyz": [float(value) for value in problem.target_size_xyz],
        "support_top_z": float(problem.support_top_z),
        "pred_pos_xy": [float(pred_pos_xy[0]), float(pred_pos_xy[1])],
        "pred_center_z": float(pred_center_z),
        "expected_center_z": float(expected_center_z),
        "center_z_error": float(pred_center_z - expected_center_z),
        "support_gap_abs": float(support_gap_abs),
        "support_projection_overlap_ratio": float(support_projection_overlap_ratio),
        "contact_required": bool(problem.contact_required),
        "support_required": bool(problem.support_required),
        "must_be_released": bool(problem.must_be_released),
        "stable_after_settle_steps": int(problem.stable_after_settle_steps),
        "effective_post_place_settle_steps": int(effective_post_place_settle_steps),
        "max_support_gap": float(problem.max_support_gap),
        "min_support_projection_overlap_ratio": float(problem.min_support_projection_overlap_ratio),
        "max_linear_speed_after_settle": float(problem.max_linear_speed_after_settle),
        "max_angular_speed_after_settle": float(problem.max_angular_speed_after_settle),
        "target_linear_speed_after_settle": target_linear_speed_after_settle,
        "target_angular_speed_after_settle": target_angular_speed_after_settle,
        "valid_episode": bool(valid_episode),
        "object_above_support": bool(object_above_support),
        "support_gap_within_limit": bool(stack_eval["support_gap_within_limit"]),
        "support_overlap_sufficient": bool(stack_eval["support_overlap_sufficient"]),
        "support_contact_established": bool(stack_eval["support_contact_established"]),
        "required_release_satisfied": bool(stack_eval["required_release_satisfied"]),
        "required_contact_satisfied": bool(stack_eval["required_contact_satisfied"]),
        "required_support_satisfied": bool(stack_eval["required_support_satisfied"]),
        "object_stable_after_settle": bool(object_stable_after_settle),
        "placement_execution_success": bool(placement_execution_success),
        "released_after_settle": bool(released_after_settle),
        "stack_satisfied": bool(stack_satisfied),
        "grasp_execution_success": bool(grasp_summary["grasp_success"]),
        "release_detected": bool(place_summary["release_detected"]),
        "grasp_execution": grasp_summary,
        "placement_execution": place_summary,
    }

    print("\n" + "=" * 72, flush=True)
    print("episode_result", flush=True)
    print("=" * 72, flush=True)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return result


def run_containment_placement_episode(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    episode_index: int,
    task_label: str,
    rng: random.Random | None = None,
    recorder: RolloutCameraRecorder | None = None,
    collector: EpisodeCollector | None = None,
) -> dict[str, Any]:
    env.reset()
    settle_env_after_reset(env, args_cli.reset_settle_steps)
    if recorder is not None:
        recorder.capture_now()

    problem = _build_containment_placement_problem(env, task_label=task_label)
    effective_post_place_settle_steps = max(int(args_cli.post_place_settle_steps), int(problem.stable_after_settle_steps))

    target_pos_xy = tuple(float(v) for v in problem.container_center_xy)
    random_gt_used = False
    if str(args_cli.agent).strip().lower() == "random_gt":
        rng = rng or random.Random(int(args_cli.agent_seed))
        margin_x = max(0.0, float(problem.interior_half_extents_xy[0]) - 0.6 * float(problem.target_size_xyz[0]) - float(problem.xy_inner_margin))
        margin_y = max(0.0, float(problem.interior_half_extents_xy[1]) - 0.6 * float(problem.target_size_xyz[1]) - float(problem.xy_inner_margin))
        target_pos_xy = (
            float(problem.container_center_xy[0] + rng.uniform(-margin_x, margin_x)),
            float(problem.container_center_xy[1] + rng.uniform(-margin_y, margin_y)),
        )
        random_gt_used = True

    grasp_summary = run_scripted_grasp_for_target(env, problem.target_object_id, collector=collector)
    if bool(grasp_summary["grasp_success"]):
        place_summary = run_scripted_place_for_target(
            env,
            target_name=problem.target_object_id,
            target_pos_xy=target_pos_xy,
            target_surface_z=float(problem.interior_floor_top_z),
            object_height=float(problem.target_size_xyz[2]),
            post_place_settle_steps=effective_post_place_settle_steps,
            grasp_offset_xy=tuple(float(value) for value in grasp_summary["grasp_offset_xy"]),
            collector=collector,
        )
    else:
        place_summary = {
            "target_name": problem.target_object_id,
            "target_pos_xy": [float(problem.container_center_xy[0]), float(problem.container_center_xy[1])],
            "execution_completed": False,
            "terminated_early": False,
            "release_detected": False,
            "settle_applied": False,
            "pred_pos_xy": list(_current_object_xy(env, problem.target_object_id)),
        }

    metrics = _containment_local_metrics(
        env,
        target_object_id=problem.target_object_id,
        container_reference_object_id=problem.container_reference_object_id,
        target_size_xyz=problem.target_size_xyz,
    )
    relative_center = tuple(float(value) for value in metrics["relative_center"])
    projected_half_extents = tuple(float(value) for value in metrics["projected_half_extents"])
    pred_center_w = tuple(float(value) for value in metrics["pred_center_w"])

    container_reference_pos = _current_object_position(env, problem.container_reference_object_id)
    interior_floor_top_offset = float(problem.interior_floor_top_z - float(container_reference_pos[2].item()))
    opening_plane_z_offset = float(problem.opening_plane_z - float(container_reference_pos[2].item()))
    available_half_extents_xy = (
        float(problem.interior_half_extents_xy[0] - projected_half_extents[0] - problem.xy_inner_margin),
        float(problem.interior_half_extents_xy[1] - projected_half_extents[1] - problem.xy_inner_margin),
    )
    xy_inside = bool(
        available_half_extents_xy[0] >= 0.0
        and available_half_extents_xy[1] >= 0.0
        and abs(relative_center[0]) <= available_half_extents_xy[0] + 1.0e-6
        and abs(relative_center[1]) <= available_half_extents_xy[1] + 1.0e-6
    )
    target_bottom_local = float(relative_center[2] - projected_half_extents[2])
    target_top_local = float(relative_center[2] + projected_half_extents[2])
    floor_gap = float(target_bottom_local - interior_floor_top_offset)
    top_clearance = float(opening_plane_z_offset - target_top_local)
    near_floor = bool(floor_gap <= float(problem.max_floor_gap))
    not_penetrating_floor = bool(floor_gap >= -float(problem.max_floor_penetration))
    below_opening = bool(top_clearance >= float(problem.min_top_clearance_below_opening))
    released_after_settle = not _object_grasped(env, problem.target_object_id)
    target_linear_speed_after_settle = _current_object_linear_speed(env, problem.target_object_id)
    target_angular_speed_after_settle = _current_object_angular_speed(env, problem.target_object_id)
    object_stable_after_settle = bool(
        target_linear_speed_after_settle is not None
        and target_angular_speed_after_settle is not None
        and target_linear_speed_after_settle <= float(problem.max_linear_speed_after_settle)
        and target_angular_speed_after_settle <= float(problem.max_angular_speed_after_settle)
    )

    placement_execution_success = bool(place_summary["execution_completed"]) and bool(place_summary["release_detected"])
    valid_episode = bool(grasp_summary["grasp_success"]) and bool(placement_execution_success)
    containment_satisfied = bool(
        valid_episode
        and xy_inside
        and near_floor
        and not_penetrating_floor
        and below_opening
        and object_stable_after_settle
        and ((not problem.must_be_released) or released_after_settle)
    )

    result = {
        "agent_name": str(args_cli.agent),
        "episode_index": int(episode_index),
        "task_label": str(task_label),
        "scene_id": int(getattr(env.cfg, "scene_id", -1)),
        "layout_key": str(getattr(env.cfg, "task_layout_key", "")),
        "task_index": int(getattr(env.cfg, "task_index", -1)),
        "query_text": str(problem.query_text),
        "target_object_id": str(problem.target_object_id),
        "container_reference_object_id": str(problem.container_reference_object_id),
        "container_center_xy": [float(problem.container_center_xy[0]), float(problem.container_center_xy[1])],
        "target_pos_xy_chosen": [float(target_pos_xy[0]), float(target_pos_xy[1])],
        "random_gt_used": bool(random_gt_used),
        "target_size_xyz": [float(value) for value in problem.target_size_xyz],
        "interior_half_extents_xy": [float(value) for value in problem.interior_half_extents_xy],
        "interior_floor_top_z": float(problem.interior_floor_top_z),
        "opening_plane_z": float(problem.opening_plane_z),
        "pred_center_w": [float(value) for value in pred_center_w],
        "relative_center": [float(value) for value in relative_center],
        "projected_half_extents": [float(value) for value in projected_half_extents],
        "available_half_extents_xy": [float(value) for value in available_half_extents_xy],
        "target_bottom_local": float(target_bottom_local),
        "target_top_local": float(target_top_local),
        "floor_gap": float(floor_gap),
        "top_clearance": float(top_clearance),
        "must_be_released": bool(problem.must_be_released),
        "stable_after_settle_steps": int(problem.stable_after_settle_steps),
        "effective_post_place_settle_steps": int(effective_post_place_settle_steps),
        "max_floor_gap": float(problem.max_floor_gap),
        "max_floor_penetration": float(problem.max_floor_penetration),
        "min_top_clearance_below_opening": float(problem.min_top_clearance_below_opening),
        "xy_inner_margin": float(problem.xy_inner_margin),
        "max_linear_speed_after_settle": float(problem.max_linear_speed_after_settle),
        "max_angular_speed_after_settle": float(problem.max_angular_speed_after_settle),
        "target_linear_speed_after_settle": target_linear_speed_after_settle,
        "target_angular_speed_after_settle": target_angular_speed_after_settle,
        "valid_episode": bool(valid_episode),
        "xy_inside": bool(xy_inside),
        "near_floor": bool(near_floor),
        "not_penetrating_floor": bool(not_penetrating_floor),
        "below_opening": bool(below_opening),
        "released_after_settle": bool(released_after_settle),
        "object_stable_after_settle": bool(object_stable_after_settle),
        "placement_execution_success": bool(placement_execution_success),
        "containment_satisfied": bool(containment_satisfied),
        "grasp_execution_success": bool(grasp_summary["grasp_success"]),
        "release_detected": bool(place_summary["release_detected"]),
        "grasp_execution": grasp_summary,
        "placement_execution": place_summary,
    }

    print("\n" + "=" * 72, flush=True)
    print("episode_result", flush=True)
    print("=" * 72, flush=True)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return result


def build_task_summary(
    *,
    episode_results: list[dict[str, Any]],
    env_cfg,
    task_label: str,
) -> dict[str, Any]:
    total = max(1, len(episode_results))
    zone_successes = sum(1 for result in episode_results if bool(result.get("zone_satisfied", False)))
    stack_successes = sum(1 for result in episode_results if bool(result.get("stack_satisfied", False)))
    containment_successes = sum(1 for result in episode_results if bool(result.get("containment_satisfied", False)))
    valid_episodes = sum(1 for result in episode_results if bool(result.get("valid_episode", False)))
    grasp_successes = sum(1 for result in episode_results if bool(result.get("grasp_execution_success", False)))
    placement_successes = sum(1 for result in episode_results if bool(result.get("placement_execution_success", False)))
    l2_values = [float(result["l2_error"]) for result in episode_results if result.get("l2_error") is not None]
    support_overlap_values = [
        float(result["support_projection_overlap_ratio"])
        for result in episode_results
        if result.get("support_projection_overlap_ratio") is not None
    ]
    center_z_error_values = [
        float(abs(result["center_z_error"])) for result in episode_results if result.get("center_z_error") is not None
    ]
    floor_gap_values = [float(result["floor_gap"]) for result in episode_results if result.get("floor_gap") is not None]
    top_clearance_values = [
        float(result["top_clearance"]) for result in episode_results if result.get("top_clearance") is not None
    ]
    support_contact_successes = sum(
        1 for result in episode_results if bool(result.get("support_contact_established", False))
    )
    stable_successes = sum(1 for result in episode_results if bool(result.get("object_stable_after_settle", False)))
    if any("containment_satisfied" in result for result in episode_results):
        goal_metric = "containment_satisfied"
        goal_successes = containment_successes
    elif any("stack_satisfied" in result for result in episode_results):
        goal_metric = "stack_satisfied"
        goal_successes = stack_successes
    else:
        goal_metric = "zone_satisfied"
        goal_successes = zone_successes

    summary = {
        "task_id": str(getattr(env_cfg, "task_metadata", {}).get("task_id", "")),
        "task_version": str(getattr(env_cfg, "task_metadata", {}).get("task_version", "")),
        "scene_id": int(getattr(env_cfg, "scene_id", -1)),
        "layout_key": str(getattr(env_cfg, "task_layout_key", "")),
        "task_index": int(getattr(env_cfg, "task_index", -1)),
        "task_label": str(task_label),
        "agent_name": str(args_cli.agent),
        "agent_episodes": int(total),
        "goal_metric": str(goal_metric),
        "goal_satisfied_rate": float(goal_successes / total),
        "zone_satisfied_rate": float(zone_successes / total),
        "stack_satisfied_rate": float(stack_successes / total),
        "containment_satisfied_rate": float(containment_successes / total),
        "valid_episode_rate": float(valid_episodes / total),
        "grasp_success_rate": float(grasp_successes / total),
        "placement_execution_rate": float(placement_successes / total),
        "l2_error_mean": (float(sum(l2_values) / len(l2_values)) if l2_values else None),
        "support_contact_rate": float(support_contact_successes / total),
        "object_stable_after_settle_rate": float(stable_successes / total),
        "support_projection_overlap_ratio_mean": (
            float(sum(support_overlap_values) / len(support_overlap_values)) if support_overlap_values else None
        ),
        "center_z_error_abs_mean": (
            float(sum(center_z_error_values) / len(center_z_error_values)) if center_z_error_values else None
        ),
        "floor_gap_mean": (float(sum(floor_gap_values) / len(floor_gap_values)) if floor_gap_values else None),
        "top_clearance_mean": (
            float(sum(top_clearance_values) / len(top_clearance_values)) if top_clearance_values else None
        ),
        "episodes": episode_results,
    }
    return summary


def run_spatial_placement_episode(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    episode_index: int,
    task_label: str,
    rng: random.Random | None = None,
    recorder: RolloutCameraRecorder | None = None,
    collector: EpisodeCollector | None = None,
) -> dict[str, Any]:
    variant = _resolve_spatial_variant(env)
    if variant == "place_directional":
        return run_directional_placement_episode(
            env,
            episode_index=episode_index,
            task_label=task_label,
            rng=rng,
            recorder=recorder,
            collector=collector,
        )
    if variant == "place_on_top":
        return run_support_top_placement_episode(
            env,
            episode_index=episode_index,
            task_label=task_label,
            rng=rng,
            recorder=recorder,
            collector=collector,
        )
    if variant in {"place_inside", "place_inside_boundary_control"}:
        return run_containment_placement_episode(
            env,
            episode_index=episode_index,
            task_label=task_label,
            rng=rng,
            recorder=recorder,
            collector=collector,
        )
    raise ValueError(f"Unsupported spatial placement variant: {variant!r}")


def run_agent_rollout(env: GeoMindFrankaIKRelMimicEnv, *, num_episodes: int, task_label: str) -> dict[str, Any]:
    episode_results: list[dict[str, Any]] = []
    for episode_index in range(max(1, int(num_episodes))):
        episode_rng = random.Random(int(args_cli.agent_seed) + int(episode_index))
        collector = _create_episode_collector(env, episode_index=episode_index)
        video_dir = None
        fps_roll = 30.0
        recorder: RolloutCameraRecorder | None = None
        if episode_index == 0:
            video_dir = resolve_rollout_camera_video_dir(args_cli, task_label, env)
            if video_dir is not None:
                fps_roll = float(getattr(args_cli, "camera_rollout_video_fps", 30.0))
                recorder = create_rollout_camera_recorder(
                    env,
                    max_frames=_camera_rollout_video_max_frames(),
                    sample_hz=fps_roll,
                ).install()
        try:
            result = run_spatial_placement_episode(
                env,
                episode_index=episode_index,
                task_label=task_label,
                rng=episode_rng,
                recorder=recorder,
                collector=collector,
            )
        finally:
            if recorder is not None:
                recorder.uninstall()
        trajectory_h5 = _export_episode_collector(collector, episode_index=episode_index)
        result["trajectory_h5"] = None if trajectory_h5 is None else str(trajectory_h5)
        result["segment_source"] = str(args_cli.trajectory_segment_source)
        episode_results.append(result)
        if episode_index == 0 and video_dir is not None and recorder is not None:
            paths = recorder.save_mp4s(video_dir, fps=fps_roll)
            _print_rollout_camera_video_result(paths, video_dir)

    summary = build_task_summary(
        episode_results=episode_results,
        env_cfg=env.cfg,
        task_label=task_label,
    )
    print("\n" + "=" * 72, flush=True)
    print("task_rollout_summary", flush=True)
    print("=" * 72, flush=True)
    print(json.dumps(_compact_summary_payload(summary), indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    env = None
    env_cfg = None
    stage = "build_runtime_env_cfg"

    try:
        resolved_request = _resolve_cli_task_request()
        env_cfg = build_runtime_env_cfg(request=resolved_request)
        task_label = resolve_task_label(env_cfg.task_layout_key, env_cfg.task_index)
        _print_runtime_header(env_cfg, task_label=task_label)

        stage = "create_env"
        env = GeoMindFrankaIKRelMimicEnv(cfg=env_cfg)

        stage = "run_agent_rollout"
        with torch.inference_mode():
            task_summary = run_agent_rollout(
                env,
                num_episodes=args_cli.agent_episodes,
                task_label=task_label,
            )

        if args_cli.summary_json_out:
            _write_json(args_cli.summary_json_out, task_summary)
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
