#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import faulthandler
import json
import math
import os
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a metadata-aware live debugger for the legacy state-refresh / identity-tracking family backend."
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
        "--max-steps",
        type=int,
        default=0,
        help="Maximum environment steps before exit. Use 0 to run until the app window is closed.",
    )
    parser.add_argument(
        "--protocol-debug-interval",
        type=int,
        default=30,
        help="Print protocol snapshot every N env steps. Use 0 to disable periodic dumps.",
    )
    parser.add_argument(
        "--post-intervention-settle-steps",
        type=int,
        default=None,
        help="Override cfg.post_intervention_settle_steps for this live debug run.",
    )
    parser.add_argument(
        "--reset-stabilization-steps",
        type=int,
        default=None,
        help="Override cfg.reset_stabilization_steps for this live debug run.",
    )
    parser.add_argument(
        "--protocol-displacement-tol",
        type=float,
        default=None,
        help="Override cfg.protocol_displacement_tol for this live debug run.",
    )
    parser.add_argument(
        "--protocol-min-reorder-gap",
        type=float,
        default=None,
        help="Override cfg.protocol_min_reorder_gap for this live debug run.",
    )
    parser.add_argument(
        "--scripted-grasp-query",
        action="store_true",
        default=False,
        help="After reset, execute a minimal scripted grasp on the protocol GT object and print readout diagnostics.",
    )
    parser.add_argument(
        "--grasp-readout-window",
        type=int,
        default=12,
        help="Number of lift-hold frames to sample for grasped_object_indices diagnostics.",
    )
    parser.add_argument(
        "--post-release-settle-steps",
        type=int,
        default=30,
        help=(
            "After scripted grasp opens the gripper, run this many extra zero-action env.step() calls "
            "so blocks separate and fall under physics before the episode ends."
        ),
    )
    parser.add_argument(
        "--scripted-eef-speed-scale",
        type=float,
        default=0.5,
        help=(
            "Scale factor for scripted Cartesian EEF motion. "
            "Smaller values move the arm more slowly and automatically increase stage durations."
        ),
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=("oracle", "random"),
        default=None,
        help="Run a minimal state-refresh / identity-tracking target-selection loop with the selected baseline agent.",
    )
    parser.add_argument(
        "--agent-episodes",
        type=int,
        default=10,
        help="Number of reset->decide->grasp->evaluate episodes to run when --agent is enabled.",
    )
    parser.add_argument(
        "--agent-seed",
        type=int,
        default=0,
        help="Random seed for RandomAgent. Ignored by OracleAgent.",
    )
    parser.add_argument(
        "--task-seed",
        type=int,
        default=None,
        help="Seed for seeded task dynamics such as randomized left/right identity swaps.",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        choices=("full", "final_frame_only"),
        default=None,
        help="Override task_protocol.control_mode for live debug or agent rollout.",
    )
    parser.add_argument(
        "--eval-suite",
        type=str,
        choices=(
            "state_selection_core_minimal",
            "historical_identity_minimal",
            "object_position_update_minimal",
            "object_identity_tracking_minimal",
            "reorder_with_distractors_minimal",
            "semantic_protocol_minimal",
        ),
        default=None,
        help="Run a fixed minimal annotated task suite instead of a single layout/task pair.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        default=False,
        help="Suppress verbose per-step/per-episode debug dumps and keep summary-level output.",
    )
    parser.add_argument(
        "--summary-json-out",
        type=str,
        default=None,
        help="Optional path to write the final task or suite summary JSON.",
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
    parser.add_argument(
        "--viewer-warmup-steps",
        type=int,
        default=12,
        help=(
            "Extra zero-action steps right after reset in GUI mode, to ensure early-stage object dynamics are visible."
        ),
    )
    parser.add_argument(
        "--camera-rollout-video",
        action="store_true",
        default=False,
        help=(
            "When running headless, save per-camera RGB MP4 for the full first agent rollout. "
            "Requires OUTPUT_DIR in the environment (set by GeoMind rollout wrapper scripts). "
            "Can also enable via HEADLESS_CAMERA_VIDEO=1. "
            "(Named with camera-* prefix to avoid argparse ambiguity with --headless.)"
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
        help="FPS for rollout camera video clips.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def print_failure_context(stage: str, *, step_count: int | None = None, env_cfg=None) -> None:
    print("\n" + "=" * 72, file=sys.stderr, flush=True)
    print("GeoMind Task Debug Failure", file=sys.stderr, flush=True)
    print("=" * 72, file=sys.stderr, flush=True)
    print(f"stage:          {stage}", file=sys.stderr, flush=True)

    args_obj = globals().get("args_cli")
    if args_obj is not None:
        print(f"task_id:        {getattr(args_obj, 'task_id', None)}", file=sys.stderr, flush=True)
        print(f"variant:        {getattr(args_obj, 'variant', None)}", file=sys.stderr, flush=True)
        print(f"scene:          {getattr(args_obj, 'scene', None)}", file=sys.stderr, flush=True)
        print(f"scene_id:       {getattr(args_obj, 'scene_id', None)}", file=sys.stderr, flush=True)
        print(f"layout_key:     {getattr(args_obj, 'layout_key', None)}", file=sys.stderr, flush=True)
        print(f"task_index:     {getattr(args_obj, 'task_index', None)}", file=sys.stderr, flush=True)
        print(f"object_mode:    {getattr(args_obj, 'object_mode', None)}", file=sys.stderr, flush=True)
        print(f"eval_suite:     {getattr(args_obj, 'eval_suite', None)}", file=sys.stderr, flush=True)
        print(f"num_envs:       {getattr(args_obj, 'num_envs', None)}", file=sys.stderr, flush=True)
        print(
            f"settle_steps:   {getattr(args_obj, 'post_intervention_settle_steps', None)}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"reset_stabilize:{getattr(args_obj, 'reset_stabilization_steps', None)}",
            file=sys.stderr,
            flush=True,
        )
        print(f"task_seed:      {getattr(args_obj, 'task_seed', None)}", file=sys.stderr, flush=True)
        print(f"control_mode:   {getattr(args_obj, 'control_mode', None)}", file=sys.stderr, flush=True)
        print(f"device:         {getattr(args_obj, 'device', None)}", file=sys.stderr, flush=True)
        print(f"headless:       {getattr(args_obj, 'headless', None)}", file=sys.stderr, flush=True)

    if env_cfg is not None:
        _print_metadata_first_task_header(env_cfg, stream=sys.stderr)
        print(f"task_mode:      {getattr(env_cfg, 'task_mode', None)}", file=sys.stderr, flush=True)

    if step_count is not None:
        print(f"step_count:     {step_count}", file=sys.stderr, flush=True)

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
from isaaclab.utils import configclass

from geomind_lab.mimic import GeoMindFrankaIKRelMimicEnv, GeoMindFrankaIKRelMimicEnvCfg
from geomind_lab.mimic.simple_protocol_agents import (
    StateReindexingAgent,
    StateReindexingAgentInput,
    build_state_reindexing_agent,
)
from geomind_lab.mimic.state_reindexing_benchmark_utils import build_state_refresh_summary_metadata
from geomind_lab.mimic.utils.family_runner_registry import resolve_runner_family
from geomind_lab.utils.headless_camera_video import (
    RolloutCameraRecorder,
    create_rollout_camera_recorder,
)
from geomind_lab.utils.headless_rollout_camera_hook import resolve_rollout_camera_video_dir

from geomind_lab.mimic.utils.task_metadata_registry import (
    TASK_VERSION_BASE,
    TASK_VERSION_CONTROL,
    TASK_VERSION_STRESS,
    build_task_metadata,
    resolve_task_id,
    supported_task_versions,
)
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import (
    load_task,
    resolve_task_annotation,
    resolve_task_request,
)


DOWN_FACING_QUAT = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
GRIPPER_OPEN_COMMAND = 1.0
GRIPPER_CLOSE_COMMAND = -1.0
FULL_CONTROL_MODE = "full"
FINAL_FRAME_ONLY_CONTROL_MODE = "final_frame_only"
TRACK_MULTI_INTERVENTION_SCRIPT = "leftmost_to_middle_then_rightmost_to_left_of_leftmost"
TRACK_CLUTTER_INTERVENTION_SCRIPT = "middle_to_left_then_rightmost_to_left"


@dataclass(frozen=True)
class EvalTaskSpec:
    scene_id: int
    layout_key: str
    task_index: int
    label: str


MINIMAL_EVAL_SUITES: dict[str, tuple[EvalTaskSpec, ...]] = {
    "state_selection_core_minimal": (
        EvalTaskSpec(scene_id=65, layout_key="state_reindex_blocks_line_4", task_index=0, label="reindex_extreme_standard"),
        EvalTaskSpec(scene_id=10, layout_key="state_reindex_blocks_line_4_tight", task_index=0, label="reindex_extreme_stress"),
        EvalTaskSpec(scene_id=65, layout_key="state_reindex_insert_blocks_line_4", task_index=0, label="reindex_insert_standard"),
    ),
    "historical_identity_minimal": (
        EvalTaskSpec(scene_id=65, layout_key="historical_identity_blocks_line_3", task_index=0, label="track_single_standard"),
    ),
    "object_position_update_minimal": (
        EvalTaskSpec(scene_id=65, layout_key="state_reindex_blocks_line_4", task_index=0, label="state_reindex_runtime_standard"),
        EvalTaskSpec(scene_id=65, layout_key="state_reindex_insert_blocks_line_4", task_index=0, label="state_reindex_insert_standard"),
    ),
    "object_identity_tracking_minimal": (
        EvalTaskSpec(scene_id=65, layout_key="historical_identity_blocks_line_3", task_index=0, label="track_single_standard"),
        EvalTaskSpec(scene_id=65, layout_key="historical_identity_multistep_blocks_line_5", task_index=0, label="track_multi_standard"),
    ),
    "reorder_with_distractors_minimal": (
        EvalTaskSpec(scene_id=65, layout_key="state_reindex_red_subset_blocks_line_5", task_index=0, label="reindex_subset_standard"),
        EvalTaskSpec(scene_id=10, layout_key="state_reindex_red_subset_blocks_line_6_tight", task_index=0, label="reindex_subset_stress"),
    ),
    "semantic_protocol_minimal": (
        EvalTaskSpec(scene_id=65, layout_key="state_reindex_blocks_line_4", task_index=0, label="state_reindex_runtime_standard"),
        EvalTaskSpec(scene_id=65, layout_key="state_reindex_insert_blocks_line_4", task_index=0, label="state_reindex_insert_standard"),
        EvalTaskSpec(scene_id=65, layout_key="historical_identity_blocks_line_3", task_index=0, label="track_single_standard"),
        EvalTaskSpec(scene_id=65, layout_key="track_single_scene65_semantic_anchors", task_index=0, label="track_single_semantic"),
        EvalTaskSpec(scene_id=65, layout_key="historical_identity_multistep_blocks_line_5", task_index=0, label="track_multi_standard"),
        EvalTaskSpec(scene_id=65, layout_key="state_reindex_red_subset_blocks_line_5", task_index=0, label="reindex_subset_standard"),
        EvalTaskSpec(scene_id=10, layout_key="state_reindex_red_subset_blocks_line_6_tight", task_index=0, label="reindex_subset_stress"),
    ),
}

DEFAULT_DEBUG_SCENE_ID = 65
DEFAULT_DEBUG_LAYOUT_KEY = "state_reindex_blocks_line_4"
DEFAULT_DEBUG_TASK_INDEX = 0


def _resolve_cli_task_locator(
    *,
    task_id: str | None = None,
    variant: str | None = None,
    scene: str | None = None,
    scene_id: int | None = None,
    layout_key: str | None = None,
    task_index: int | None = None,
) -> object:
    effective_layout = layout_key
    effective_index = DEFAULT_DEBUG_TASK_INDEX if task_index is None else task_index
    if not task_id and effective_layout is None:
        effective_layout = DEFAULT_DEBUG_LAYOUT_KEY
    return resolve_task_request(
        task_id=task_id,
        variant=variant,
        scene=scene,
        scene_id=scene_id,
        layout_key=effective_layout,
        task_index=effective_index,
        object_mode=args_cli.object_mode,
    )


def _resolve_task_metadata_from_env_cfg(env_cfg, *, task_label: str | None = None) -> dict[str, Any]:
    metadata = dict(getattr(env_cfg, "task_metadata", {}) or {})
    if metadata:
        return metadata

    annotation = resolve_task_annotation(str(env_cfg.task_layout_key), int(env_cfg.task_index))
    protocol = dict(getattr(env_cfg, "task_protocol", {}) or {})
    raw_task_name = str(annotation.get("task_id") or task_label or "").strip()
    resolved_task_id = resolve_task_id(
        raw_task_id=raw_task_name,
        family=protocol.get("family"),
        variant=protocol.get("variant"),
        protocol=protocol,
    )
    return build_task_metadata(resolved_task_id) if resolved_task_id else {}


def _paired_control_companion_version(task_metadata: dict[str, Any]) -> str:
    paired_task_id = str(task_metadata.get("paired_control_id") or "")
    current_version = str(task_metadata.get("task_version") or "")
    paired_versions = tuple(supported_task_versions(paired_task_id))
    if current_version in paired_versions:
        return current_version
    if current_version == TASK_VERSION_BASE and TASK_VERSION_CONTROL in paired_versions:
        return TASK_VERSION_CONTROL
    if current_version == TASK_VERSION_CONTROL and TASK_VERSION_BASE in paired_versions:
        return TASK_VERSION_BASE
    if current_version == TASK_VERSION_STRESS and TASK_VERSION_STRESS in paired_versions:
        return TASK_VERSION_STRESS
    return paired_versions[0] if paired_versions else current_version


def _paired_control_metadata_warning(task_metadata: dict[str, Any]) -> str:
    paired_task_id = str(task_metadata.get("paired_control_id") or "")
    if not paired_task_id:
        return ""
    try:
        paired_metadata = build_task_metadata(
            paired_task_id,
            task_version=_paired_control_companion_version(task_metadata),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return f"unable to load paired control metadata for {paired_task_id}: {exc}"

    mismatches: list[str] = []
    for field_name in ("semantic_module", "evaluator_key", "enabled_probes", "phase"):
        if task_metadata.get(field_name) != paired_metadata.get(field_name):
            mismatches.append(field_name)
    if str(task_metadata.get("task_version") or "") == TASK_VERSION_STRESS and task_metadata.get("stress_factors") != paired_metadata.get("stress_factors"):
        mismatches.append("stress_factors")
    if not mismatches:
        return ""
    return (
        "paired control metadata mismatch with "
        f"{paired_task_id}:{paired_metadata.get('task_version', '')} on {', '.join(mismatches)}"
    )


def _print_metadata_first_task_header(env_cfg, *, task_label: str | None = None, stream=None) -> None:
    output_stream = stream or sys.stdout
    task_protocol = dict(getattr(env_cfg, "task_protocol", {}) or {})
    task_metadata = _resolve_task_metadata_from_env_cfg(env_cfg, task_label=task_label)
    runner_family = os.environ.get("RUNNER_FAMILY") or resolve_runner_family(
        task_id=task_metadata.get("task_id"),
        family=task_metadata.get("family"),
        semantic_module=task_metadata.get("semantic_module"),
        skill_type=task_metadata.get("skill_type"),
    )
    paired_control_id = str(task_metadata.get("paired_control_id") or "")
    task_version = str(task_metadata.get("task_version") or "")

    print(f"task_id:        {task_metadata.get('task_id', '')}", file=output_stream, flush=True)
    print(f"task_version:   {task_version}", file=output_stream, flush=True)
    print(f"phase:          {task_metadata.get('phase', '')}", file=output_stream, flush=True)
    print(f"semantic_module:{task_metadata.get('semantic_module', '')}", file=output_stream, flush=True)
    print(f"skill_type:     {task_metadata.get('skill_type', '')}", file=output_stream, flush=True)
    print(f"runner_family:  {runner_family}", file=output_stream, flush=True)
    if paired_control_id:
        print(f"paired_control_id:{paired_control_id}", file=output_stream, flush=True)
        if task_version == TASK_VERSION_CONTROL:
            print("task_role:      paired_control", file=output_stream, flush=True)
        else:
            print("task_role:      paired_primary", file=output_stream, flush=True)
    warning = _paired_control_metadata_warning(task_metadata)
    if warning:
        print(f"paired_warning: {warning}", file=output_stream, flush=True)
    print(f"variant:        {getattr(env_cfg, 'task_variant', '')}", file=output_stream, flush=True)
    print(f"scene:          {getattr(env_cfg, 'scene_name', '')}", file=output_stream, flush=True)
    print(f"scene_id:       {env_cfg.scene_id}", file=output_stream, flush=True)
    print(f"task_index:     {env_cfg.task_index}", file=output_stream, flush=True)
    print(f"object_mode:    {getattr(env_cfg, 'object_mode', '')}", file=output_stream, flush=True)
    print(f"instruction:    {getattr(env_cfg, 'task_instruction', '')}", file=output_stream, flush=True)
    legacy_family = str(task_metadata.get("family") or task_protocol.get("family", "")).strip()
    legacy_variant = str(task_protocol.get("variant", "")).strip()
    if legacy_family:
        print(f"legacy_family:  {legacy_family}", file=output_stream, flush=True)
    if legacy_variant:
        print(f"legacy_variant: {legacy_variant}", file=output_stream, flush=True)


def should_print_verbose_logs() -> bool:
    return not bool(getattr(args_cli, "summary_only", False))


_FALSE_ENV_VALUES = {"0", "false", "no", ""}


def _env_flag(name: str, default: bool = True) -> bool:
    fallback = "1" if default else "0"
    return os.environ.get(name, fallback).strip().lower() not in _FALSE_ENV_VALUES


_EXTREME_REINDEX_RIGID_BODY_PHYSICS: dict[str, float | int] = {
    "linear_damping": 1.2,
    "angular_damping": 4.0,
    "max_linear_velocity": 2.0,
    "max_angular_velocity": 20.0,
    "solver_position_iteration_count": 16,
    "solver_velocity_iteration_count": 4,
    "sleep_threshold": 0.002,
    "stabilization_threshold": 0.001,
}

_EXTREME_REINDEX_MATERIAL_PHYSICS: dict[str, float | str] = {
    "static_friction": 1.15,
    "dynamic_friction": 0.96,
    "restitution": 0.0,
    "friction_combine_mode": "max",
}

_EXTREME_REINDEX_COLOR_SEQUENCE: tuple[tuple[float, float, float], ...] = (
    (0.90, 0.25, 0.22),
    (0.25, 0.78, 0.30),
    (0.18, 0.42, 0.88),
    (0.92, 0.78, 0.18),
)


def _apply_extreme_runtime_rigid_and_material(obj: dict) -> None:
    """Enable movable rigid-body physics + friction without changing task semantics."""
    if not isinstance(obj, dict):
        return
    physics = obj.get("physics")
    if not isinstance(physics, dict):
        physics = {}
        obj["physics"] = physics
    rigid = physics.get("rigid_body")
    if not isinstance(rigid, dict):
        rigid = {}
        physics["rigid_body"] = rigid
    rigid["rigid_body_enabled"] = True
    rigid["kinematic_enabled"] = False
    for key, value in _EXTREME_REINDEX_RIGID_BODY_PHYSICS.items():
        rigid[key] = value
    collision = physics.get("collision")
    if not isinstance(collision, dict):
        collision = {}
        physics["collision"] = collision
    collision["collision_enabled"] = True
    collision.setdefault("contact_offset", 0.002)
    collision.setdefault("rest_offset", 0.0)
    if str(collision.get("type", "")).strip().lower() == "static_mesh":
        collision["type"] = "convex_decomposition"
    material = physics.get("material")
    if not isinstance(material, dict):
        material = {}
        physics["material"] = material
    for key, value in _EXTREME_REINDEX_MATERIAL_PHYSICS.items():
        material[key] = value


_NEIGHBOR_DEBUG_INSTRUCTION = "请拿起红色方块左边离它最近的物体"
_NEIGHBOR_DEBUG_RELAYOUT: dict[str, tuple[float, float]] = {
    "left_neighbor_block": (0.40, -0.058),
    "red_anchor_block": (0.40, -0.03),
    "right_neighbor_block": (0.40, 0.06),
    "left_front_distractor_block": (0.47, -0.10),
    "far_left_distractor_block": (0.50, -0.19),
}


def _patch_task_spec_runtime_neighbor_debug(task_spec: object) -> None:
    """Debug-only: prompt/GT、刚体/摩擦与桌面布局（内存中的 task_spec，不写回标注文件）。"""
    if not isinstance(task_spec, dict):
        return
    if task_spec.get("protocol", {}).get("gt_spec", {}).get("selection_resolver") == "structural_adjacency":
        return
    task_spec["instruction"] = _NEIGHBOR_DEBUG_INSTRUCTION
    protocol = dict(task_spec.get("protocol", {}) or {})
    protocol["query_text"] = _NEIGHBOR_DEBUG_INSTRUCTION
    gt_spec = dict(protocol.get("gt_spec", {}) or {})
    gt_spec["resolver"] = "nearest_planar_left_of_reference"
    gt_spec["reference_object_id"] = "red_anchor_block"
    gt_spec["order_axis"] = "y"
    protocol["gt_spec"] = gt_spec
    task_spec["protocol"] = protocol

    runtime = task_spec.get("runtime_objects")
    if not isinstance(runtime, list):
        runtime = []
    scene_adaptation = task_spec.get("scene_adaptation")
    if isinstance(scene_adaptation, dict):
        nested = scene_adaptation.get("runtime_objects")
        if isinstance(nested, list):
            runtime = nested
            task_spec["runtime_objects"] = nested
    if not isinstance(runtime, list) or not runtime:
        return
    for obj in runtime:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", "")).strip()
        if name in _NEIGHBOR_DEBUG_RELAYOUT:
            x, y = _NEIGHBOR_DEBUG_RELAYOUT[name]
            pose = obj.get("pose")
            if not isinstance(pose, dict):
                pose = {}
                obj["pose"] = pose
            pose["xy"] = [float(x), float(y)]
            pose.setdefault("yaw", 0.0)
        _apply_extreme_runtime_rigid_and_material(obj)

    object_positions: dict[str, dict[str, list[float]]] = {}
    for obj in runtime:
        if not isinstance(obj, dict):
            continue
        object_name = str(obj.get("name", "")).strip()
        pose = obj.get("pose", {})
        xy = pose.get("xy", []) if isinstance(pose, dict) else []
        if not object_name or not isinstance(xy, list) or len(xy) < 2:
            continue
        object_positions[object_name] = {"position": [float(xy[0]), float(xy[1])]}
    if object_positions:
        task_spec["object_positions"] = object_positions


def _patch_task_spec_runtime_insert_objects(
    task_spec: object,
    *,
    preserve_visual_colors: bool = False,
) -> None:
    """Patch runtime objects in-memory only (do not touch annotation files)."""
    if not isinstance(task_spec, dict):
        return
    runtime = task_spec.get("runtime_objects")
    if not isinstance(runtime, list):
        runtime = []
    scene_adaptation = task_spec.get("scene_adaptation")
    if isinstance(scene_adaptation, dict):
        nested = scene_adaptation.get("runtime_objects")
        if isinstance(nested, list):
            runtime = nested
            task_spec["runtime_objects"] = nested
    if not isinstance(runtime, list):
        return
    for obj_idx, obj in enumerate(runtime):
        if not isinstance(obj, dict):
            continue
        _apply_extreme_runtime_rigid_and_material(obj)
        # Align visual material parameters with extreme template. For subset tasks,
        # keep authored semantic colors (e.g., red subset) unless explicitly overridden.
        if preserve_visual_colors:
            raw_color = obj.get("visual_color", [0.7, 0.7, 0.7])
            if isinstance(raw_color, list) and len(raw_color) >= 3:
                rgb = (float(raw_color[0]), float(raw_color[1]), float(raw_color[2]))
            else:
                rgb = _EXTREME_REINDEX_COLOR_SEQUENCE[obj_idx % len(_EXTREME_REINDEX_COLOR_SEQUENCE)]
                obj["visual_color"] = [float(rgb[0]), float(rgb[1]), float(rgb[2])]
        else:
            rgb = _EXTREME_REINDEX_COLOR_SEQUENCE[obj_idx % len(_EXTREME_REINDEX_COLOR_SEQUENCE)]
            obj["visual_color"] = [float(rgb[0]), float(rgb[1]), float(rgb[2])]
        visual_material = obj.get("visual_material")
        if not isinstance(visual_material, dict):
            visual_material = {}
            obj["visual_material"] = visual_material
        visual_material["fallback_diffuse_color"] = [float(rgb[0]), float(rgb[1]), float(rgb[2])]
        visual_material["fallback_roughness"] = 0.58


def _usd_set_gprim_display_color_recursive(prim, rgb: tuple[float, float, float]) -> None:
    try:
        from pxr import Gf, UsdGeom
    except ImportError:
        return
    if prim.IsA(UsdGeom.Gprim):
        gprim = UsdGeom.Gprim(prim)
        attr = gprim.GetDisplayColorAttr()
        vec = Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        if attr:
            attr.Set([vec])
        else:
            gprim.CreateDisplayColorAttr([vec])
    for child in prim.GetChildren():
        _usd_set_gprim_display_color_recursive(child, rgb)


def _usd_set_bound_preview_surface_diffuse_recursive(prim, rgb: tuple[float, float, float]) -> None:
    """Set bound PreviewSurface diffuse color when material binding exists.

    In Isaac Sim scenes, bound material usually overrides ``displayColor``.
    """
    try:
        from pxr import Gf, Sdf, UsdShade
    except ImportError:
        return

    vec = Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))

    def _set_shader_input(shader_prim) -> bool:
        if not shader_prim or not shader_prim.IsValid():
            return False
        shader = UsdShade.Shader(shader_prim)
        shader_id_attr = shader.GetIdAttr()
        shader_id = str(shader_id_attr.Get() or "").lower() if shader_id_attr else ""
        if "previewsurface" not in shader_id:
            return False
        diffuse = shader.GetInput("diffuseColor")
        if not diffuse:
            diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        diffuse.Set(vec)
        return True

    binding_api = UsdShade.MaterialBindingAPI(prim)
    bound = binding_api.ComputeBoundMaterial()
    material = bound[0] if isinstance(bound, tuple) else bound
    if material and material.GetPrim().IsValid():
        # direct surface output shader
        surface_out = material.GetSurfaceOutput()
        if surface_out:
            source = surface_out.GetConnectedSource()
            if source:
                source_shader = source[0]
                if hasattr(source_shader, "GetPrim"):
                    _set_shader_input(source_shader.GetPrim())
        # fallback scan child shader prims
        for child in material.GetPrim().GetChildren():
            _set_shader_input(child)

    for child in prim.GetChildren():
        _usd_set_bound_preview_surface_diffuse_recursive(child, rgb)


def _maybe_apply_insert_target_visual_colors(env: GeoMindFrankaIKRelMimicEnv) -> None:
    if not bool(getattr(env.cfg, "debug_insert_visual_colors", False)):
        return
    if not (
        bool(getattr(env.cfg, "debug_smooth_planar_insert", False))
        or bool(getattr(env.cfg, "debug_smooth_planar_neighbor", False))
    ):
        return
    snapshot = env.get_state_reindexing_snapshot(env_ids=[0])
    if str(snapshot.get("intervention_script_name", "")).strip() != "rightmost_to_between_first_and_second":
        return
    gt_i = int(_first_snapshot_value(snapshot, "gt_object_indices", -1))
    moved_i = int(_first_snapshot_value(snapshot, "moved_object_indices", -1))
    if gt_i < 0 and moved_i < 0:
        return
    try:
        from isaaclab.sim.utils import find_matching_prims
    except ImportError:
        return
    names = list(snapshot.get("object_names", []))
    env_ns = getattr(env.scene, "env_ns", "/World/envs")
    stage = getattr(env.sim, "stage", None)
    if stage is None:
        return
    for idx, name in enumerate(names):
        if idx == gt_i:
            rgb = (0.15, 0.92, 0.22)
        elif idx == moved_i:
            rgb = (0.96, 0.42, 0.12)
        else:
            continue
        try:
            prims = find_matching_prims(f"{env_ns}/env_0/{name}", stage=stage)
        except Exception:
            continue
        for prim in prims:
            _usd_set_bound_preview_surface_diffuse_recursive(prim, rgb)
            _usd_set_gprim_display_color_recursive(prim, rgb)


def build_runtime_env_cfg(
    *,
    request=None,
    control_mode: str | None = None,
):
    base_cfg_cls = GeoMindFrankaIKRelMimicEnvCfg
    resolved_request = request
    if resolved_request is None:
        resolved_request = _resolve_cli_task_locator(
            task_id=args_cli.task_id,
            variant=args_cli.variant,
            scene=args_cli.scene,
            scene_id=args_cli.scene_id,
            layout_key=args_cli.layout_key,
            task_index=args_cli.task_index,
        )
    resolved_control_mode = args_cli.control_mode if control_mode is None else control_mode

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

        tid = str(resolved_request.task_id or "")
        neighbor_dbg = _env_flag("GEOMIND_NEIGHBOR_DEBUG", default=True)
        insert_dbg = _env_flag("GEOMIND_INSERT_DEBUG", default=True)
        subset_dbg = _env_flag("GEOMIND_SUBSET_DEBUG", default=True)
        insert_runtime_prepatched = False

        if tid == "reindex_insert" and insert_dbg:
            if not isinstance(getattr(self, "task_spec", None), dict):
                loaded_task = load_task(
                    task_id=resolved_request.task_id,
                    variant=resolved_request.variant,
                    scene=resolved_request.scene,
                    scene_id=resolved_request.scene_id,
                    layout_key=resolved_request.layout_key,
                    task_index=resolved_request.task_index,
                    object_mode=resolved_request.object_mode,
                )
                self.task_spec = loaded_task.task_spec
            _patch_task_spec_runtime_insert_objects(
                getattr(self, "task_spec", None),
                preserve_visual_colors=False,
            )
            insert_runtime_prepatched = True
        if tid == "reindex_neighbor" and neighbor_dbg:
            _patch_task_spec_runtime_neighbor_debug(getattr(self, "task_spec", None))

        base_cfg_cls.__post_init__(self)

        if tid == "reindex_neighbor" and neighbor_dbg:
            self.debug_smooth_planar_neighbor = True
            self.debug_insert_visual_colors = _env_flag("GEOMIND_NEIGHBOR_COLOR_TAGS", default=False)
        if (tid == "reindex_insert" and insert_dbg) or (tid == "reindex_subset" and subset_dbg):
            # Debug-only runtime behavior for insert/subset task; GT/prompt stay untouched.
            self.debug_smooth_planar_insert = True
            color_tags_key = "GEOMIND_INSERT_COLOR_TAGS" if tid == "reindex_insert" else "GEOMIND_SUBSET_COLOR_TAGS"
            self.debug_insert_visual_colors = _env_flag(color_tags_key, default=False)
            if not insert_runtime_prepatched:
                _patch_task_spec_runtime_insert_objects(
                    getattr(self, "task_spec", None),
                    preserve_visual_colors=(tid == "reindex_subset"),
                )

        if args_cli.post_intervention_settle_steps is not None:
            self.post_intervention_settle_steps = int(args_cli.post_intervention_settle_steps)
        if args_cli.reset_stabilization_steps is not None:
            self.reset_stabilization_steps = int(args_cli.reset_stabilization_steps)
        if args_cli.task_seed is not None:
            self.seed = int(args_cli.task_seed)
        if args_cli.protocol_displacement_tol is not None:
            self.protocol_displacement_tol = float(args_cli.protocol_displacement_tol)
        if args_cli.protocol_min_reorder_gap is not None:
            self.protocol_min_reorder_gap = float(args_cli.protocol_min_reorder_gap)
        if resolved_control_mode is not None:
            task_protocol = dict(getattr(self, "task_protocol", {}) or {})
            task_protocol["control_mode"] = str(resolved_control_mode)
            self.task_protocol = task_protocol
        relocation_motion_mode = str(os.environ.get("GEOMIND_RELOCATION_MOTION_MODE", "") or "").strip()
        if relocation_motion_mode:
            task_protocol = dict(getattr(self, "task_protocol", {}) or {})
            intervention = dict(task_protocol.get("intervention", {}) or {})
            intervention["motion_mode"] = relocation_motion_mode
            task_protocol["intervention"] = intervention
            self.task_protocol = task_protocol

        self.viewer.eye = tuple(float(v) for v in args_cli.viewer_eye)
        self.viewer.lookat = tuple(float(v) for v in args_cli.viewer_lookat)

    runtime_cfg_cls = configclass(
        type(
            "RuntimeGeoMindStateReindexDebugCfg",
            (base_cfg_cls,),
            {
                "__doc__": "Runtime GeoMind mimic cfg for metadata-first state-refresh backend live debugging.",
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


def resolve_eval_suite(eval_suite_name: str) -> tuple[EvalTaskSpec, ...]:
    suite = MINIMAL_EVAL_SUITES.get(str(eval_suite_name))
    if suite is None:
        raise ValueError(f"Unsupported eval suite: {eval_suite_name!r}")
    return suite


def should_reset(terminated, truncated) -> bool:
    if terminated is not None and torch.any(terminated):
        return True
    if truncated is not None and torch.any(truncated):
        return True
    return False


def make_zero_actions(env: GeoMindFrankaIKRelMimicEnv) -> torch.Tensor:
    return torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)


def settle_env_after_reset(env: GeoMindFrankaIKRelMimicEnv) -> None:
    settle_steps = max(int(getattr(env.cfg, "reset_stabilization_steps", 0)), 0)
    if settle_steps <= 0:
        return
    zero_actions = make_zero_actions(env)
    for _ in range(settle_steps):
        _, _, terminated, truncated, _ = env.step(zero_actions)
        if should_reset(terminated, truncated):
            env.reset()


def print_protocol_state(stage: str, env: GeoMindFrankaIKRelMimicEnv) -> None:
    return


def _build_agent_observation(env: GeoMindFrankaIKRelMimicEnv, *, env_id: int = 0) -> dict[str, object]:
    policy_obs = dict(env.obs_buf.get("policy", {}))
    observation: dict[str, object] = {}
    for key, value in policy_obs.items():
        if isinstance(value, torch.Tensor):
            observation[str(key)] = value[env_id].clone().detach()
        else:
            observation[str(key)] = value
    return observation


def _first_snapshot_value(snapshot: dict[str, object], key: str, default=None):
    value = snapshot.get(key, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


def _resolve_target_info(
    snapshot: dict[str, object],
    *,
    target_index: int,
    uuid_key: str,
) -> dict[str, object] | None:
    object_names = list(snapshot.get("object_names", []))
    if not (0 <= int(target_index) < len(object_names)):
        return None

    target_uuid = _first_snapshot_value(snapshot, uuid_key, None)
    return {
        "target_index": int(target_index),
        "target_name": str(object_names[int(target_index)]),
        "target_uuid": target_uuid,
    }


def _eef_name(env: GeoMindFrankaIKRelMimicEnv) -> str:
    return env._eef_name()


def _pose_from_position_and_quat(position: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    rotation = PoseUtils.matrix_from_quat(quat.unsqueeze(0))[0]
    return PoseUtils.make_pose(position, rotation)


def _scripted_eef_speed_scale() -> float:
    try:
        resolved = float(getattr(args_cli, "scripted_eef_speed_scale", 0.5))
    except (TypeError, ValueError):
        resolved = 0.5
    return min(max(resolved, 0.05), 1.0)


def _scripted_stage_steps(base_steps: int) -> int:
    speed_scale = _scripted_eef_speed_scale()
    return max(int(base_steps), int(round(float(base_steps) / speed_scale)))


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
    action_dim = int(action.shape[-1])
    motion_dim = min(6, action_dim - 1) if action_dim >= 7 else action_dim
    if motion_dim > 0:
        action[..., :motion_dim] *= _scripted_eef_speed_scale()
    return action


def _current_object_position(env: GeoMindFrankaIKRelMimicEnv, object_name: str, *, env_id: int = 0) -> torch.Tensor:
    return env.scene[object_name].data.root_pos_w[env_id, :3].clone().detach()


def _current_eef_position(env: GeoMindFrankaIKRelMimicEnv, *, env_id: int = 0) -> torch.Tensor:
    pose = env.get_robot_eef_pose(_eef_name(env), env_ids=[env_id])[0]
    position, _ = PoseUtils.unmake_pose(pose)
    return position.clone().detach()


def _current_eef_pose(env: GeoMindFrankaIKRelMimicEnv, *, env_id: int = 0) -> torch.Tensor:
    return env.get_robot_eef_pose(_eef_name(env), env_ids=[env_id])[0].clone().detach()


def _object_height(env: GeoMindFrankaIKRelMimicEnv, object_name: str) -> float:
    resolver = getattr(env.cfg, "_resolve_object_runtime_height", None)
    if callable(resolver):
        resolved = resolver(object_name)
        if resolved is not None:
            return float(resolved)
    return 0.07


def _gripper_joint_mean_position(env: GeoMindFrankaIKRelMimicEnv) -> float | None:
    robot = env.scene["robot"]
    if not hasattr(env.cfg, "gripper_joint_names"):
        return None
    joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    if len(joint_ids) == 0:
        return None
    return float(robot.data.joint_pos[0, joint_ids].mean().item())


def _step_with_target_pose(
    env: GeoMindFrankaIKRelMimicEnv,
    target_pose: torch.Tensor,
    gripper_action: float,
    *,
    steps: int,
    stage_name: str,
) -> tuple[bool, dict]:
    return _step_with_target_pose_provider(
        env,
        lambda: target_pose,
        gripper_action,
        steps=steps,
        stage_name=stage_name,
    )


def _step_with_target_pose_provider(
    env: GeoMindFrankaIKRelMimicEnv,
    target_pose_provider,
    gripper_action: float,
    *,
    steps: int,
    stage_name: str,
    marker_target_name: str | None = None,
) -> tuple[bool, dict]:
    terminated = None
    truncated = None
    extras = {}
    for _ in range(steps):
        if marker_target_name:
            _draw_track_target_marker(
                env,
                target_name=str(marker_target_name),
                radius=max(0.02, 0.9 * float(_object_height(env, str(marker_target_name)))),
            )
        target_pose = target_pose_provider()
        action = _action_from_target_pose(env, target_pose, gripper_action)
        _, _, terminated, truncated, extras = env.step(action)
        if should_reset(terminated, truncated):
            print_protocol_state(f"{stage_name}_pre_auto_reset", env)
            return False, extras
    return True, extras


def _select_scripted_grasp_target(env: GeoMindFrankaIKRelMimicEnv) -> dict[str, object]:
    snapshot = env.get_state_reindexing_snapshot()
    gt_indices = list(snapshot.get("gt_object_indices", []))
    if not gt_indices:
        raise RuntimeError("Protocol snapshot does not contain gt_object_indices.")
    target_info = _resolve_target_info(snapshot, target_index=int(gt_indices[0]), uuid_key="gt_uuid")
    if target_info is None:
        raise RuntimeError(f"Protocol GT is unresolved: gt_object_indices={gt_indices}")
    return target_info


def _show_track_target_marker_enabled() -> bool:
    raw_value = str(os.environ.get("SHOW_TRACK_TARGET_MARKER", "1")).strip().lower()
    return raw_value not in {"0", "false", "off", "no"}


def _draw_track_target_marker(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    target_name: str,
    radius: float,
) -> None:
    if float(radius) <= 0.0:
        return
    sim = getattr(env, "sim", None)
    if sim is None or not sim.has_gui():
        return
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        try:
            from isaacsim.util.debug_draw import _debug_draw
        except ImportError:
            return
    try:
        target_position = _current_object_position(env, str(target_name))
        cx = float(target_position[0].item())
        cy = float(target_position[1].item())
        draw_z = float(target_position[2].item()) + 0.015
        segment_count = 40
        starts: list[tuple[float, float, float]] = []
        ends: list[tuple[float, float, float]] = []
        for idx in range(segment_count):
            theta0 = 2.0 * math.pi * float(idx) / float(segment_count)
            theta1 = 2.0 * math.pi * float(idx + 1) / float(segment_count)
            starts.append((cx + float(radius) * math.cos(theta0), cy + float(radius) * math.sin(theta0), draw_z))
            ends.append((cx + float(radius) * math.cos(theta1), cy + float(radius) * math.sin(theta1), draw_z))
        cross = max(0.01, float(radius) * 0.35)
        starts.extend([(cx - cross, cy, draw_z), (cx, cy - cross, draw_z)])
        ends.extend([(cx + cross, cy, draw_z), (cx, cy + cross, draw_z)])
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()
        colors = [(0.15, 0.95, 0.25, 1.0)] * len(starts)
        sizes = [4.0] * len(starts)
        draw.draw_lines(starts, ends, colors, sizes)
    except Exception:
        return


def _maybe_draw_track_multi_target_marker(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    snapshot: dict[str, object],
) -> None:
    if not _show_track_target_marker_enabled():
        return
    if str(snapshot.get("intervention_script_name", "")).strip() != TRACK_MULTI_INTERVENTION_SCRIPT:
        return
    gt_object_index = int(_first_snapshot_value(snapshot, "gt_object_indices", -1))
    if gt_object_index < 0:
        return
    target_info = _resolve_target_info(snapshot, target_index=gt_object_index, uuid_key="gt_uuid")
    if target_info is None:
        return
    target_name = str(target_info.get("target_name", "")).strip()
    if not target_name:
        return
    radius = max(0.02, 0.9 * float(_object_height(env, target_name)))
    _draw_track_target_marker(env, target_name=target_name, radius=radius)


def _is_track_multi_intervention_snapshot(snapshot: dict[str, object]) -> bool:
    script_name = str(snapshot.get("intervention_script_name", "")).strip()
    return script_name in {TRACK_MULTI_INTERVENTION_SCRIPT, TRACK_CLUTTER_INTERVENTION_SCRIPT}


def _wait_dynamic_intervention_ready(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    max_wait_steps: int,
) -> dict[str, object]:
    """Wait until dynamic scripted object motion completes before agent selects/grasps."""
    snapshot = env.get_state_reindexing_snapshot(env_ids=[0])
    for _ in range(max(0, int(max_wait_steps))):
        dynamic_enabled = bool(_first_snapshot_value(snapshot, "dynamic_motion_enabled", False))
        dynamic_completed = bool(_first_snapshot_value(snapshot, "dynamic_motion_completed", True))
        if (not dynamic_enabled) or dynamic_completed:
            return snapshot
        _, _, terminated, truncated, _ = env.step(make_zero_actions(env))
        if should_reset(terminated, truncated):
            break
        snapshot = env.get_state_reindexing_snapshot(env_ids=[0])
    dynamic_enabled = bool(_first_snapshot_value(snapshot, "dynamic_motion_enabled", False))
    dynamic_completed = bool(_first_snapshot_value(snapshot, "dynamic_motion_completed", True))
    if dynamic_enabled and not dynamic_completed:
        elapsed = _first_snapshot_value(snapshot, "dynamic_motion_elapsed_s", None)
        duration = _first_snapshot_value(snapshot, "dynamic_motion_duration_s", None)
        print(
            "[state_reindex_wait] dynamic intervention did not complete before timeout: "
            f"max_wait_steps={int(max_wait_steps)} elapsed_s={elapsed} duration_s={duration}",
            flush=True,
        )
    return snapshot


def _wait_track_multi_intervention_ready(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    max_wait_steps: int,
) -> dict[str, object]:
    snapshot = _wait_dynamic_intervention_ready(env, max_wait_steps=max_wait_steps)
    if not _is_track_multi_intervention_snapshot(snapshot):
        return snapshot
    for _ in range(max(0, int(max_wait_steps))):
        _maybe_draw_track_multi_target_marker(env, snapshot=snapshot)
        intervention_success = bool(_first_snapshot_value(snapshot, "intervention_success", False))
        intervention_failure = bool(_first_snapshot_value(snapshot, "intervention_failure", False))
        if intervention_success and not intervention_failure:
            return snapshot
        _, _, terminated, truncated, _ = env.step(make_zero_actions(env))
        if should_reset(terminated, truncated):
            break
        snapshot = env.get_state_reindexing_snapshot(env_ids=[0])
    return snapshot


def _is_insert_debug_motion_snapshot(snapshot: dict[str, object]) -> bool:
    return str(snapshot.get("intervention_script_name", "")).strip() == "rightmost_to_between_first_and_second"


def _preposition_arm_during_insert_motion(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    max_wait_steps: int,
) -> dict[str, object]:
    """Move the arm while insert objects are still sliding; keep final selection/grasp unchanged."""
    snapshot = env.get_state_reindexing_snapshot(env_ids=[0])
    if not _is_insert_debug_motion_snapshot(snapshot):
        return _wait_track_multi_intervention_ready(env, max_wait_steps=max_wait_steps)
    if not bool(_first_snapshot_value(snapshot, "dynamic_motion_enabled", False)):
        return snapshot
    if bool(_first_snapshot_value(snapshot, "dynamic_motion_completed", True)):
        return snapshot

    target_index = int(_first_snapshot_value(snapshot, "moved_object_indices", -1))
    target_info = _resolve_target_info(snapshot, target_index=target_index, uuid_key="gt_uuid")
    if target_info is None:
        return _wait_track_multi_intervention_ready(env, max_wait_steps=max_wait_steps)

    target_name = str(target_info["target_name"])
    object_height = _object_height(env, target_name)
    down_facing_quat = DOWN_FACING_QUAT.to(device=env.device)
    approach_clearance = max(0.16, 2.2 * float(object_height))

    for _ in range(max(0, int(max_wait_steps))):
        dynamic_enabled = bool(_first_snapshot_value(snapshot, "dynamic_motion_enabled", False))
        dynamic_completed = bool(_first_snapshot_value(snapshot, "dynamic_motion_completed", True))
        if (not dynamic_enabled) or dynamic_completed:
            return snapshot

        target_position = _current_object_position(env, target_name)
        target_pose = _pose_from_position_and_quat(
            torch.tensor(
                [
                    float(target_position[0].item()),
                    float(target_position[1].item()),
                    float(target_position[2].item()) + approach_clearance,
                ],
                device=env.device,
                dtype=torch.float32,
            ),
            down_facing_quat,
        )
        action = _action_from_target_pose(env, target_pose, GRIPPER_OPEN_COMMAND)
        _, _, terminated, truncated, _ = env.step(action)
        if should_reset(terminated, truncated):
            break
        snapshot = env.get_state_reindexing_snapshot(env_ids=[0])

    return _wait_track_multi_intervention_ready(env, max_wait_steps=max_wait_steps)


def _insert_debug_motion_ready_after_reset(env: GeoMindFrankaIKRelMimicEnv) -> bool:
    snapshot = env.get_state_reindexing_snapshot(env_ids=[0])
    return _is_insert_debug_motion_snapshot(snapshot) and bool(
        _first_snapshot_value(snapshot, "dynamic_motion_enabled", False)
    )


def _majority_vote(values: list[int]) -> int:
    if not values:
        return -1
    return Counter(values).most_common(1)[0][0]


def _collect_grasp_readout_diagnostics(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    target_name: str,
    hold_pose: torch.Tensor,
    hold_steps: int,
) -> dict[str, object]:
    grasped_history: list[int] = []
    success_history: list[bool] = []
    eef_distance_history: list[float] = []
    subtask_signal_history: list[dict[str, bool]] = []

    for _ in range(hold_steps):
        action = _action_from_target_pose(env, hold_pose, GRIPPER_CLOSE_COMMAND)
        _, _, terminated, truncated, _ = env.step(action)
        if should_reset(terminated, truncated):
            break

        snapshot = env.get_state_reindexing_snapshot()
        grasped_history.append(int(list(snapshot.get("grasped_object_indices", [-1]))[0]))
        success_history.append(bool(list(snapshot.get("success", [False]))[0]))
        object_position = _current_object_position(env, target_name)
        eef_position = _current_eef_position(env)
        eef_distance_history.append(float(torch.linalg.vector_norm(object_position - eef_position).item()))
        subtask_signals = env.get_subtask_term_signals(env_ids=[0])
        subtask_signal_history.append(
            {str(name): bool(value.reshape(-1)[0].item()) for name, value in subtask_signals.items()}
        )

    final_grasped_index = grasped_history[-1] if grasped_history else -1
    return {
        "hold_steps": hold_steps,
        "grasped_object_indices_history": grasped_history,
        "grasped_object_indices_vote": _majority_vote(grasped_history),
        "grasped_object_indices_final": final_grasped_index,
        "success_history": success_history,
        "eef_target_distance_history": eef_distance_history,
        "eef_target_distance_final": eef_distance_history[-1] if eef_distance_history else None,
        "gripper_joint_mean_position": _gripper_joint_mean_position(env),
        "subtask_signal_history_tail": subtask_signal_history[-3:],
    }


def run_scripted_grasp_for_target(
    env: GeoMindFrankaIKRelMimicEnv,
    target_info: dict[str, object],
    *,
    draw_track_marker: bool = False,
    hold_after_grasp: bool = False,
) -> dict[str, object]:
    target_index = int(target_info["target_index"])
    target_name = str(target_info["target_name"])
    target_uuid = target_info["target_uuid"]
    target_position = _current_object_position(env, target_name)
    object_height = _object_height(env, target_name)
    target_surface_z = float(target_position[2].item())
    approach_z = float(target_surface_z + max(0.12, 1.5 * object_height))
    grasp_z = float(target_surface_z + max(0.0, 0.05 * object_height))
    lift_z = float(target_surface_z + max(0.18, 2.0 * object_height))
    approach_steps = _scripted_stage_steps(30)
    descend_steps = _scripted_stage_steps(20)
    close_steps = _scripted_stage_steps(28)
    lift_steps = _scripted_stage_steps(24)
    release_steps = _scripted_stage_steps(18)
    execution_summary = {
        "target_index": target_index,
        "target_name": target_name,
        "target_uuid": target_uuid,
        "target_position": [float(v) for v in target_position.tolist()],
        "target_height": object_height,
        "approach_z": approach_z,
        "grasp_z": grasp_z,
        "lift_z": lift_z,
        "scripted_eef_speed_scale": _scripted_eef_speed_scale(),
        "dynamic_follow_target": True,
        "execution_completed": False,
        "terminated_early": False,
        "grasp_readout_diagnostics": {},
    }

    if should_print_verbose_logs():
        print("\n" + "=" * 72, flush=True)
        print("scripted_grasp_query", flush=True)
        print("=" * 72, flush=True)
        print(
            json.dumps(execution_summary, indent=2, sort_keys=True),
            flush=True,
        )

    down_facing_quat = DOWN_FACING_QUAT.to(device=env.device)

    def _follow_pose(z_world: float):
        current_target_position = _current_object_position(env, target_name)
        return _pose_from_position_and_quat(
            torch.tensor(
                [current_target_position[0].item(), current_target_position[1].item(), z_world],
                device=env.device,
                dtype=torch.float32,
            ),
            down_facing_quat,
        )

    ok, _ = _step_with_target_pose_provider(
        env,
        lambda: _follow_pose(approach_z),
        GRIPPER_OPEN_COMMAND,
        steps=approach_steps,
        stage_name="approach",
        marker_target_name=target_name if draw_track_marker else None,
    )
    print_protocol_state("after_scripted_approach", env)
    if not ok:
        execution_summary["terminated_early"] = True
        return execution_summary

    ok, _ = _step_with_target_pose_provider(
        env,
        lambda: _follow_pose(grasp_z),
        GRIPPER_OPEN_COMMAND,
        steps=descend_steps,
        stage_name="descend",
        marker_target_name=target_name if draw_track_marker else None,
    )
    print_protocol_state("after_scripted_descend", env)
    if not ok:
        execution_summary["terminated_early"] = True
        return execution_summary

    ok, _ = _step_with_target_pose_provider(
        env,
        lambda: _follow_pose(grasp_z),
        GRIPPER_CLOSE_COMMAND,
        steps=close_steps,
        stage_name="close",
        marker_target_name=target_name if draw_track_marker else None,
    )
    print_protocol_state("after_scripted_close", env)
    if not ok:
        execution_summary["terminated_early"] = True
        return execution_summary

    lift_pose = _follow_pose(lift_z)
    ok, _ = _step_with_target_pose_provider(
        env,
        lambda: _follow_pose(lift_z),
        GRIPPER_CLOSE_COMMAND,
        steps=lift_steps,
        stage_name="lift",
        marker_target_name=target_name if draw_track_marker else None,
    )
    print_protocol_state("after_scripted_lift", env)
    if not ok:
        execution_summary["terminated_early"] = True
        return execution_summary

    diagnostics = _collect_grasp_readout_diagnostics(
        env,
        target_name=target_name,
        hold_pose=lift_pose,
        hold_steps=max(1, int(args_cli.grasp_readout_window)),
    )
    execution_summary["grasp_readout_diagnostics"] = diagnostics
    if should_print_verbose_logs():
        print("grasp_readout_diagnostics:", flush=True)
        print(json.dumps(diagnostics, indent=2, sort_keys=True), flush=True)
    print_protocol_state("after_scripted_grasp_readout_window", env)
    execution_summary["hold_after_grasp"] = bool(hold_after_grasp)

    if hold_after_grasp:
        grasp_vote = int(diagnostics.get("grasped_object_indices_vote", -1))
        if grasp_vote >= 0 and grasp_vote == target_index:
            execution_summary["execution_completed"] = True
            execution_summary["release_skipped"] = True
            execution_summary["post_release_settle_steps"] = 0
            return execution_summary

    # 保持当前末端位姿松爪，并多跑若干全量物理步，避免夹爪张开后仍黏在指尖、等下一次逻辑才下落
    ok_release, _ = _step_with_target_pose_provider(
        env,
        lambda: _current_eef_pose(env),
        GRIPPER_OPEN_COMMAND,
        steps=release_steps,
        stage_name="release",
        marker_target_name=target_name if draw_track_marker else None,
    )
    print_protocol_state("after_scripted_release", env)
    if not ok_release:
        execution_summary["terminated_early"] = True
        return execution_summary

    release_settle = max(0, int(args_cli.post_release_settle_steps))
    for _ in range(release_settle):
        _, _, terminated, truncated, _ = env.step(make_zero_actions(env))
        if should_reset(terminated, truncated):
            print_protocol_state("release_settle_pre_auto_reset", env)
            execution_summary["terminated_early"] = True
            return execution_summary

    print_protocol_state("after_scripted_post_release_settle", env)
    execution_summary["execution_completed"] = True
    execution_summary["post_release_settle_steps"] = release_settle
    return execution_summary


def run_scripted_grasp_query(env: GeoMindFrankaIKRelMimicEnv) -> dict[str, object]:
    return run_scripted_grasp_for_target(env, _select_scripted_grasp_target(env))


def _build_agent_input(env: GeoMindFrankaIKRelMimicEnv) -> StateReindexingAgentInput:
    oracle_snapshot = env.get_state_reindexing_snapshot()
    agent_snapshot = env.get_agent_protocol_snapshot()
    query_text = str(agent_snapshot.get("query_text", "")) or str(getattr(env.cfg, "task_instruction", ""))
    return StateReindexingAgentInput(
        observation=_build_agent_observation(env),
        query_text=query_text,
        protocol_snapshot=agent_snapshot,
        oracle_snapshot=oracle_snapshot,
        control_mode=str(agent_snapshot.get("control_mode", FULL_CONTROL_MODE)),
    )


def _classify_agent_episode_result(result: dict[str, object]) -> str:
    if not bool(result["intervention_success"]):
        return "intervention_failed"
    if not bool(result["valid_episode"]):
        return "episode_invalid"
    if not bool(result["selected_matches_gt"]):
        return "wrong_selection"
    if not bool(result["grasp_detected"]):
        return "no_grasp"
    if not bool(result["success"]):
        return "grasped_wrong_object"
    return "success"


def _build_agent_episode_result(
    *,
    episode_index: int,
    agent_name: str,
    env: GeoMindFrankaIKRelMimicEnv,
    final_snapshot: dict[str, object],
    selected_object_index: int,
    grasp_execution: dict[str, object],
) -> dict[str, object]:
    gt_object_index = int(_first_snapshot_value(final_snapshot, "gt_object_indices", -1))
    grasped_object_index = int(_first_snapshot_value(final_snapshot, "grasped_object_indices", -1))
    result = {
        "episode_index": episode_index,
        "agent_name": agent_name,
        "scene_id": int(getattr(env.cfg, "scene_id", -1)),
        "layout_key": str(getattr(env.cfg, "task_layout_key", "")),
        "task_index": int(getattr(env.cfg, "task_index", -1)),
        "track_name": str(final_snapshot.get("track_name", "")),
        "control_mode": str(final_snapshot.get("control_mode", FULL_CONTROL_MODE)),
        "gt_resolver": str(final_snapshot.get("gt_resolver", "")),
        "target_rank": str(final_snapshot.get("target_rank", "")),
        "target_subset_object_ids": list(final_snapshot.get("target_subset_object_ids", [])),
        "query_text": str(final_snapshot.get("query_text", "")),
        "selected_object_index": selected_object_index,
        "selected_uuid": _first_snapshot_value(final_snapshot, "selected_uuid", None),
        "gt_object_indices": gt_object_index,
        "gt_uuid": _first_snapshot_value(final_snapshot, "gt_uuid", None),
        "grasped_object_indices": grasped_object_index,
        "grasped_uuid": _first_snapshot_value(final_snapshot, "grasped_uuid", None),
        "success": bool(_first_snapshot_value(final_snapshot, "success", False)),
        "valid_episode": bool(_first_snapshot_value(final_snapshot, "valid_episode", False)),
        "intervention_success": bool(_first_snapshot_value(final_snapshot, "intervention_success", False)),
        "accidental_displacement": bool(_first_snapshot_value(final_snapshot, "accidental_displacement", False)),
        "max_non_target_planar_displacement": float(
            _first_snapshot_value(final_snapshot, "max_non_target_planar_displacement", 0.0)
        ),
        "selected_matches_gt": selected_object_index >= 0 and selected_object_index == gt_object_index,
        "grasp_detected": grasped_object_index >= 0,
        "grasp_execution": grasp_execution,
    }
    result["failure_category"] = _classify_agent_episode_result(result)
    return result


def run_agent_episode(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    agent: StateReindexingAgent,
    episode_index: int,
) -> dict[str, object]:
    insert_debug_motion = _insert_debug_motion_ready_after_reset(env)
    if not insert_debug_motion:
        settle_env_after_reset(env)
    if not insert_debug_motion and not bool(getattr(args_cli, "headless", False)):
        warmup_steps = max(0, int(getattr(args_cli, "viewer_warmup_steps", 12) or 0))
        if warmup_steps > 0:
            zero_actions = make_zero_actions(env)
            for _ in range(warmup_steps):
                _, _, terminated, truncated, _ = env.step(zero_actions)
                if should_reset(terminated, truncated):
                    env.reset()
                    settle_env_after_reset(env)
                    break

    reset_snapshot = _preposition_arm_during_insert_motion(
        env,
        max_wait_steps=max(20, int(getattr(args_cli, "post_intervention_settle_steps", 60) or 60)),
    )
    _maybe_apply_insert_target_visual_colors(env)
    _maybe_draw_track_multi_target_marker(env, snapshot=reset_snapshot)
    agent_input = _build_agent_input(env)
    agent_output = agent(agent_input)
    selected_object_index = int(agent_output.selected_object_index)
    env.set_state_reindexing_selected_object_indices(selected_object_index, env_ids=[0])
    selection_snapshot = env.get_state_reindexing_snapshot(env_ids=[0])

    if should_print_verbose_logs():
        print("\n" + "=" * 72, flush=True)
        print("agent_decision", flush=True)
        print("=" * 72, flush=True)
        print(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "agent_name": agent.name,
                    "track_name": str(reset_snapshot.get("track_name", "")),
                    "control_mode": str(reset_snapshot.get("control_mode", FULL_CONTROL_MODE)),
                    "query_text": str(reset_snapshot.get("query_text", "")),
                    "selected_object_index": selected_object_index,
                    "selected_uuid": _first_snapshot_value(selection_snapshot, "selected_uuid", None),
                    "gt_object_indices": int(_first_snapshot_value(reset_snapshot, "gt_object_indices", -1)),
                    "gt_uuid": _first_snapshot_value(reset_snapshot, "gt_uuid", None),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    print_protocol_state(f"agent_episode_{episode_index}_after_selection", env)

    target_info = _resolve_target_info(
        selection_snapshot,
        target_index=selected_object_index,
        uuid_key="selected_uuid",
    )
    if target_info is None:
        grasp_execution = {
            "target_index": selected_object_index,
            "execution_completed": False,
            "terminated_early": False,
            "grasp_readout_diagnostics": {},
            "skip_reason": "invalid_selection",
        }
    else:
        draw_track_marker = _is_track_multi_intervention_snapshot(selection_snapshot)
        if draw_track_marker:
            _maybe_draw_track_multi_target_marker(env, snapshot=selection_snapshot)
        track_name = str(selection_snapshot.get("track_name", "")).strip().lower()
        hold_after_grasp = (
            str(agent.name).strip().lower() == "oracle"
            and ("tracking" in track_name or "reindex" in track_name)
        )
        grasp_execution = run_scripted_grasp_for_target(
            env,
            target_info,
            draw_track_marker=draw_track_marker,
            hold_after_grasp=hold_after_grasp,
        )

    final_snapshot = env.get_state_reindexing_snapshot(env_ids=[0])
    return _build_agent_episode_result(
        episode_index=episode_index,
        agent_name=agent.name,
        env=env,
        final_snapshot=final_snapshot,
        selected_object_index=selected_object_index,
        grasp_execution=grasp_execution,
    )


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total > 0 else 0.0


def _count_failure_category(episode_results: list[dict[str, object]], category: str) -> int:
    return sum(str(result.get("failure_category")) == category for result in episode_results)


def _task_total_object_count(env: GeoMindFrankaIKRelMimicEnv) -> int | None:
    task_spec = getattr(env.cfg, "task_spec", {})
    if not isinstance(task_spec, dict):
        return None
    object_positions = task_spec.get("object_positions", {})
    if not isinstance(object_positions, dict):
        return None
    return len(object_positions)


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


def write_summary_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_compact_summary_payload(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _camera_rollout_video_max_frames() -> int | None:
    value = int(getattr(args_cli, "camera_rollout_video_frames", 0))
    return None if value <= 0 else value


def build_rollout_summary(
    *,
    agent_name: str,
    episode_results: list[dict[str, object]],
    scene_id: int,
    layout_key: str,
    task_index: int,
    track_name: str,
    control_mode: str,
    task_label: str | None = None,
    motion_mode: str | None = None,
) -> dict[str, object]:
    num_episodes = len(episode_results)
    valid_episode_count = sum(bool(result["valid_episode"]) for result in episode_results)
    success_count = sum(bool(result["success"]) for result in episode_results)
    selected_match_count = sum(bool(result["selected_matches_gt"]) for result in episode_results)
    grasp_detected_count = sum(bool(result["grasp_detected"]) for result in episode_results)
    intervention_success_count = sum(bool(result["intervention_success"]) for result in episode_results)
    accidental_displacement_count = sum(bool(result.get("accidental_displacement")) for result in episode_results)
    failure_counts = Counter(str(result["failure_category"]) for result in episode_results)

    wrong_selection_count = _count_failure_category(episode_results, "wrong_selection")
    no_grasp_count = _count_failure_category(episode_results, "no_grasp")
    intervention_failed_count = _count_failure_category(episode_results, "intervention_failed")
    episode_invalid_count = _count_failure_category(episode_results, "episode_invalid")
    grasped_wrong_object_count = _count_failure_category(episode_results, "grasped_wrong_object")

    return {
        "agent_name": agent_name,
        "scene_id": scene_id,
        "layout_key": layout_key,
        "task_index": task_index,
        "task_label": task_label,
        "track_name": track_name,
        "control_mode": control_mode,
        "motion_mode": motion_mode,
        "num_episodes": num_episodes,
        "episode_count": num_episodes,
        "valid_episode_count": valid_episode_count,
        "success_count": success_count,
        "selection_match_count": selected_match_count,
        "grasp_detected_count": grasp_detected_count,
        "intervention_success_count": intervention_success_count,
        "accidental_displacement_count": accidental_displacement_count,
        "success_rate": _rate(success_count, num_episodes),
        "valid_rate": _rate(valid_episode_count, num_episodes),
        "valid_episode_rate": _rate(valid_episode_count, num_episodes),
        "selection_accuracy": _rate(selected_match_count, num_episodes),
        "grasp_detection_rate": _rate(grasp_detected_count, num_episodes),
        "intervention_success_rate": _rate(intervention_success_count, num_episodes),
        "accidental_displacement_rate": _rate(accidental_displacement_count, num_episodes),
        "wrong_selection_count": wrong_selection_count,
        "wrong_selection_rate": _rate(wrong_selection_count, num_episodes),
        "no_grasp_count": no_grasp_count,
        "no_grasp_rate": _rate(no_grasp_count, num_episodes),
        "intervention_failed_count": intervention_failed_count,
        "intervention_failed_rate": _rate(intervention_failed_count, num_episodes),
        "episode_invalid_count": episode_invalid_count,
        "episode_invalid_rate": _rate(episode_invalid_count, num_episodes),
        "grasped_wrong_object_count": grasped_wrong_object_count,
        "grasped_wrong_object_rate": _rate(grasped_wrong_object_count, num_episodes),
        "failure_breakdown": dict(sorted(failure_counts.items())),
        "episodes": episode_results,
    }


def run_agent_rollout(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    agent: StateReindexingAgent,
    num_episodes: int,
    task_label: str | None = None,
) -> dict[str, object]:
    episode_results: list[dict[str, object]] = []
    for episode_index in range(max(1, int(num_episodes))):
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
            env.reset()
            if recorder is not None:
                recorder.capture_now()
            settle_env_after_reset(env)
            print_protocol_state(f"agent_episode_{episode_index}_after_reset", env)
            result = run_agent_episode(env, agent=agent, episode_index=episode_index)
        finally:
            if recorder is not None:
                recorder.uninstall()
        episode_results.append(result)
        if episode_index == 0 and video_dir is not None and recorder is not None:
            paths = recorder.save_mp4s(video_dir, fps=fps_roll)
            _print_rollout_camera_video_result(paths, video_dir)
        if should_print_verbose_logs():
            print("agent_episode_result:", flush=True)
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    protocol_snapshot = env.get_state_reindexing_snapshot(env_ids=[0])
    summary = build_rollout_summary(
        agent_name=agent.name,
        episode_results=episode_results,
        scene_id=int(getattr(env.cfg, "scene_id", -1)),
        layout_key=str(getattr(env.cfg, "task_layout_key", "")),
        task_index=int(getattr(env.cfg, "task_index", -1)),
        track_name=str(protocol_snapshot.get("track_name", "")),
        control_mode=str(protocol_snapshot.get("control_mode", FULL_CONTROL_MODE)),
        task_label=task_label,
        motion_mode=str(protocol_snapshot.get("motion_mode") or "") or None,
    )
    summary.update(
        build_state_refresh_summary_metadata(
            protocol=dict(getattr(env.cfg, "task_protocol", {}) or {}),
            total_object_count=_task_total_object_count(env),
            task_name=task_label
            or f"{getattr(env.cfg, 'task_layout_key', 'unknown_layout')}__task_{int(getattr(env.cfg, 'task_index', -1))}",
        )
    )
    print("\n" + "=" * 72, flush=True)
    print("task_rollout_summary", flush=True)
    print("=" * 72, flush=True)
    summary_to_print = _compact_summary_payload(summary) if not should_print_verbose_logs() else summary
    print(json.dumps(summary_to_print, indent=2, sort_keys=True), flush=True)
    return summary


def run_viewer_idle_loop(env: GeoMindFrankaIKRelMimicEnv) -> int:
    """Step with zero actions until the window closes or ``max_steps`` is reached."""
    step_count = 0
    while simulation_app.is_running():
        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break

        with torch.inference_mode():
            actions = make_zero_actions(env)
            _, _, terminated, truncated, _ = env.step(actions)
            step_count += 1

            if args_cli.protocol_debug_interval > 0 and step_count % args_cli.protocol_debug_interval == 0:
                print_protocol_state(f"step_{step_count}", env)

            if should_reset(terminated, truncated):
                print_protocol_state(f"pre_auto_reset_{step_count}", env)
                env.reset()
                settle_env_after_reset(env)
                print_protocol_state(f"post_auto_reset_{step_count}", env)

    return step_count


def print_runtime_header(
    env_cfg,
    *,
    task_label: str | None = None,
    eval_suite: str | None = None,
) -> None:
    if not should_print_verbose_logs():
        print("=" * 72, flush=True)
        print("GeoMind Task Debugger", flush=True)
        print("=" * 72, flush=True)
        _print_metadata_first_task_header(env_cfg, task_label=task_label)
        if eval_suite is not None:
            print(f"eval_suite:     {eval_suite}", flush=True)
        print(f"agent:          {args_cli.agent}", flush=True)
        print(f"agent_episodes: {args_cli.agent_episodes}", flush=True)
        print(f"settle_steps:   {env_cfg.post_intervention_settle_steps}", flush=True)
        print(f"debug_insert:   {bool(getattr(env_cfg, 'debug_smooth_planar_insert', False))}", flush=True)
        print(f"control_mode:   {dict(getattr(env_cfg, 'task_protocol', {}) or {}).get('control_mode', FULL_CONTROL_MODE)}", flush=True)
        print(f"track_name:     {dict(getattr(env_cfg, 'task_protocol', {}) or {}).get('track_name', '')}", flush=True)
        print(f"gt_resolver:    {dict(dict(getattr(env_cfg, 'task_protocol', {}) or {}).get('gt_spec', {}) or {}).get('resolver', '')}", flush=True)
        print(
            f"target_subset:  {dict(dict(getattr(env_cfg, 'task_protocol', {}) or {}).get('target_subset_spec', {}) or {}).get('object_ids', [])}",
            flush=True,
        )
        return

    print("=" * 72, flush=True)
    print("GeoMind Task Debugger", flush=True)
    print("=" * 72, flush=True)
    _print_metadata_first_task_header(env_cfg, task_label=task_label)
    if eval_suite is not None:
        print(f"eval_suite:     {eval_suite}", flush=True)
    if task_label is not None:
        print(f"legacy_task_label:{task_label}", flush=True)
    print(f"layout_key:     {env_cfg.task_layout_key}", flush=True)
    print(f"task_mode:      {env_cfg.task_mode}", flush=True)
    print(f"num_envs:       {env_cfg.scene.num_envs}", flush=True)
    print(f"device:         {env_cfg.sim.device}", flush=True)
    print(f"use_fabric:     {env_cfg.sim.use_fabric}", flush=True)
    print(f"viewer_eye:     {tuple(env_cfg.viewer.eye)}", flush=True)
    print(f"viewer_lookat:  {tuple(env_cfg.viewer.lookat)}", flush=True)
    print(f"task_seed:      {env_cfg.seed}", flush=True)
    print(f"settle_steps:   {env_cfg.post_intervention_settle_steps}", flush=True)
    print(f"reset_stabilize:{env_cfg.reset_stabilization_steps}", flush=True)
    print(f"debug_insert:   {bool(getattr(env_cfg, 'debug_smooth_planar_insert', False))}", flush=True)
    print(f"debug_colors:   {bool(getattr(env_cfg, 'debug_insert_visual_colors', False))}", flush=True)
    print(f"disp_tol:       {env_cfg.protocol_displacement_tol}", flush=True)
    print(f"reorder_gap:    {env_cfg.protocol_min_reorder_gap}", flush=True)
    print(f"scripted_grasp: {args_cli.scripted_grasp_query}", flush=True)
    print(f"agent:          {args_cli.agent}", flush=True)
    print(f"agent_episodes: {args_cli.agent_episodes}", flush=True)
    print(f"agent_seed:     {args_cli.agent_seed}", flush=True)
    print(f"control_mode:   {dict(getattr(env_cfg, 'task_protocol', {}) or {}).get('control_mode', FULL_CONTROL_MODE)}", flush=True)
    print(
        f"gt_resolver:    {dict(dict(getattr(env_cfg, 'task_protocol', {}) or {}).get('gt_spec', {}) or {}).get('resolver', '')}",
        flush=True,
    )
    print(
        f"target_subset:  {dict(dict(getattr(env_cfg, 'task_protocol', {}) or {}).get('target_subset_spec', {}) or {}).get('object_ids', [])}",
        flush=True,
    )
    print(
        "task_protocol:  "
        + json.dumps(getattr(env_cfg, "task_protocol", {}), sort_keys=True),
        flush=True,
    )


def run_agent_eval_suite(
    *,
    agent: StateReindexingAgent,
    num_episodes: int,
    eval_suite: str,
) -> dict[str, object]:
    task_specs = resolve_eval_suite(eval_suite)
    task_summaries: list[dict[str, object]] = []
    all_episode_results: list[dict[str, object]] = []

    for task_spec in task_specs:
        env = None
        env_cfg = build_runtime_env_cfg(
            request=resolve_task_request(
                scene_id=task_spec.scene_id,
                layout_key=task_spec.layout_key,
                task_index=task_spec.task_index,
                object_mode=args_cli.object_mode,
            )
        )
        try:
            if int(env_cfg.scene.num_envs) != 1:
                raise ValueError("--agent currently requires --num-envs=1 because the scripted grasp helper is single-env.")

            print_runtime_header(env_cfg, task_label=task_spec.label, eval_suite=eval_suite)
            env = GeoMindFrankaIKRelMimicEnv(cfg=env_cfg)
            if should_print_verbose_logs():
                print(f"action_space:   {env.action_space}", flush=True)
                print(f"observation:    {env.observation_space}", flush=True)
                print("=" * 72, flush=True)

            with torch.inference_mode():
                task_summary = run_agent_rollout(
                    env,
                    agent=agent,
                    num_episodes=num_episodes,
                    task_label=task_spec.label,
                )
        finally:
            if env is not None:
                env.close()

        task_summaries.append(task_summary)
        all_episode_results.extend(list(task_summary.get("episodes", [])))

    track_names = sorted({str(task_summary.get("track_name", "")) for task_summary in task_summaries})
    control_modes = sorted({str(task_summary.get("control_mode", FULL_CONTROL_MODE)) for task_summary in task_summaries})
    suite_summary = build_rollout_summary(
        agent_name=agent.name,
        episode_results=all_episode_results,
        scene_id=-1,
        layout_key=eval_suite,
        task_index=-1,
        track_name=track_names[0] if len(track_names) == 1 else "mixed",
        control_mode=control_modes[0] if len(control_modes) == 1 else "mixed",
        task_label=eval_suite,
    )
    suite_summary["eval_suite"] = eval_suite
    suite_summary["task_count"] = len(task_summaries)
    suite_summary["task_summaries"] = task_summaries

    print("\n" + "=" * 72, flush=True)
    print("suite_rollout_summary", flush=True)
    print("=" * 72, flush=True)
    suite_summary_to_print = _compact_summary_payload(suite_summary) if not should_print_verbose_logs() else suite_summary
    print(json.dumps(suite_summary_to_print, indent=2, sort_keys=True), flush=True)
    return suite_summary


def main() -> None:
    env = None
    env_cfg = None
    step_count = 0
    stage = "build_runtime_env_cfg"

    try:
        if args_cli.eval_suite is not None and args_cli.agent is None:
            raise ValueError("--eval-suite requires --agent because the suite runner is rollout-only.")

        if args_cli.eval_suite is not None:
            stage = "agent_eval_suite"
            agent = build_state_reindexing_agent(args_cli.agent, seed=args_cli.agent_seed)
            with torch.inference_mode():
                suite_summary = run_agent_eval_suite(
                    agent=agent,
                    num_episodes=args_cli.agent_episodes,
                    eval_suite=args_cli.eval_suite,
                )
            if args_cli.summary_json_out:
                write_summary_json(args_cli.summary_json_out, suite_summary)
            return

        resolved_request = _resolve_cli_task_locator(
            task_id=args_cli.task_id,
            variant=args_cli.variant,
            scene=args_cli.scene,
            scene_id=args_cli.scene_id,
            layout_key=args_cli.layout_key,
            task_index=args_cli.task_index,
        )
        env_cfg = build_runtime_env_cfg(request=resolved_request)
        if args_cli.agent is not None and int(env_cfg.scene.num_envs) != 1:
            raise ValueError("--agent currently requires --num-envs=1 because the scripted grasp helper is single-env.")

        task_label = resolve_task_annotation(resolved_request.layout_key, resolved_request.task_index).get("task_id") or None
        print_runtime_header(env_cfg, task_label=task_label)

        stage = "create_env"
        env = GeoMindFrankaIKRelMimicEnv(cfg=env_cfg)
        if should_print_verbose_logs():
            print(f"action_space:   {env.action_space}", flush=True)
            print(f"observation:    {env.observation_space}", flush=True)
            print("=" * 72, flush=True)

        if args_cli.agent is not None:
            stage = "agent_rollout"
            agent = build_state_reindexing_agent(args_cli.agent, seed=args_cli.agent_seed)
            with torch.inference_mode():
                task_summary = run_agent_rollout(env, agent=agent, num_episodes=args_cli.agent_episodes)
            if args_cli.summary_json_out:
                write_summary_json(args_cli.summary_json_out, task_summary)
            if not bool(getattr(args_cli, "headless", False)) and os.environ.get(
                "VIEWER_IDLE_AFTER_AGENT", "1"
            ).strip().lower() not in {"0", "false", "no"}:
                print(
                    "\n[viewer] Agent 回合已结束；窗口将保持打开（零动作），关闭窗口或按停止即可退出。\n"
                    "[viewer] Rollout finished; window stays open on zero-torque hold. Close the app to exit.\n",
                    flush=True,
                )
                stage = "viewer_idle"
                with torch.inference_mode():
                    step_count = run_viewer_idle_loop(env)
            return

        stage = "env_reset"
        with torch.inference_mode():
            env.reset()
            settle_env_after_reset(env)
            _wait_dynamic_intervention_ready(
                env,
                max_wait_steps=max(20, int(getattr(args_cli, "post_intervention_settle_steps", 60) or 60)),
            )
            _maybe_apply_insert_target_visual_colors(env)
        print_protocol_state("after_reset", env)
        if args_cli.scripted_grasp_query:
            stage = "scripted_grasp_query"
            with torch.inference_mode():
                run_scripted_grasp_query(env)

        stage = "viewer_idle"
        with torch.inference_mode():
            step_count = run_viewer_idle_loop(env)

    except Exception:
        print_failure_context(stage, step_count=step_count, env_cfg=env_cfg)
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
