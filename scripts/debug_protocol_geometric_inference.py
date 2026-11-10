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
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass

from geomind_lab.bootstrap.isaaclab_path import ensure_isaaclab_source_on_path
from geomind_lab.mimic.envs.geometric_inference_logic import (
    DEFAULT_EPSILON_POS,
    DEFAULT_TAU_LINE,
    DEFAULT_TAU_TREND,
    DEFAULT_TAU_YAW,
    EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME,
    EXTENSION_RULE_TYPE,
    GEOMETRIC_INFERENCE_MODULE_NAME,
    IMPLICIT_TREND_FITTING_TRACK_NAME,
    MIDPOINT_RULE_TYPE,
    TREND_CONTINUATION_SUBTASK_NAME,
    WEAK_DESTRUCTION_CONTROL_SUBTASK_NAME,
    build_geometric_inference_debug_result,
    build_geometric_inference_problem_from_object_positions,
    build_geometric_inference_suite_summary,
    build_geometric_inference_task_summary,
    normalize_rule_type,
)
from geomind_lab.mimic.geometric_inference_agents import (
    GeometricInferenceAgent,
    GeometricInferenceAgentInput,
    build_geometric_inference_agent,
)
from geomind_lab.mimic.utils.family_runner_registry import resolve_runner_family
from geomind_lab.mimic.utils.task_metadata_registry import (
    TASK_VERSION_BASE,
    TASK_VERSION_CONTROL,
    TASK_VERSION_STRESS,
    build_task_metadata,
    resolve_task_id,
    supported_task_versions,
)
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import (
    load_annotated_tasks,
    resolve_task_annotation,
    resolve_task_label,
    resolve_task_payload,
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


@dataclass(frozen=True)
class EvalTaskSpec:
    scene_id: int
    layout_key: str
    task_index: int
    label: str


DEFAULT_DEBUG_LAYOUT_KEY = "geometric_inference_extension_line_3"
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
    resolved_task_id = str(task_id or "").strip()
    resolved_layout_key = str(layout_key or "").strip()
    # Keep debug default layout only for pure layout-driven ad-hoc runs.
    # When task_id is provided (e.g. run_suite / debug_isometric_extension.sh),
    # let resolve_task_request choose profile layout from task_id + variant + scene.
    if not resolved_layout_key and not resolved_task_id:
        resolved_layout_key = DEFAULT_DEBUG_LAYOUT_KEY
    return resolve_task_request(
        task_id=resolved_task_id or None,
        variant=variant,
        scene=scene,
        scene_id=scene_id,
        layout_key=resolved_layout_key or None,
        task_index=DEFAULT_DEBUG_TASK_INDEX if task_index is None else task_index,
        object_mode=args_cli.object_mode if "args_cli" in globals() else None,
    )


def _resolve_task_metadata_from_env_cfg(env_cfg, *, task_label: str | None = None) -> dict[str, Any]:
    metadata = dict(getattr(env_cfg, "task_metadata", {}) or {})
    if metadata:
        return metadata

    annotation = _resolve_task_annotation(str(env_cfg.task_layout_key), int(env_cfg.task_index))
    task_protocol = dict(getattr(env_cfg, "task_protocol", {}) or {})
    raw_task_name = str(annotation.get("task_id") or task_label or "").strip()
    resolved_task_id = resolve_task_id(
        raw_task_id=raw_task_name,
        family=task_protocol.get("family"),
        variant=task_protocol.get("variant"),
        protocol=task_protocol,
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
    legacy_family = str(task_metadata.get("family") or task_protocol.get("family", task_protocol.get("track_name", ""))).strip()
    legacy_variant = str(task_protocol.get("variant", task_protocol.get("subtask_name", ""))).strip()
    if legacy_family:
        print(f"legacy_family:  {legacy_family}", file=output_stream, flush=True)
    if legacy_variant:
        print(f"legacy_variant: {legacy_variant}", file=output_stream, flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a metadata-aware real-placement debugger for the legacy geometric structure-inference family backend."
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
    parser.add_argument(
        "--scene-id",
        type=int,
        default=None,
        help="Optional scene id override. Defaults to the scene_id from the annotated layout.",
    )
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
    parser.add_argument(
        "--eval-suite",
        type=str,
        default=None,
        help=(
            "Optional multi-task eval suite. Supports: geometric_inference_primary (alias: structure_primary), "
            "explicit_minimal, explicit_geometric_minimal, implicit_trend_minimal."
        ),
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=(
            "oracle",
            "random",
            "explicit_midpoint_oracle",
            "explicit_extension_oracle",
            "explicit_random",
            "implicit_trend_oracle",
            "implicit_trend_random",
        ),
        default="oracle",
        help="Baseline agent that proposes target_pos_xy and, when needed, target_yaw.",
    )
    parser.add_argument("--agent-seed", type=int, default=0, help="Base random seed for RandomPositionAgent.")
    parser.add_argument(
        "--agent-episodes",
        type=int,
        default=1,
        help="Number of real rollout episodes per task.",
    )
    parser.add_argument(
        "--epsilon-pos",
        type=float,
        default=DEFAULT_EPSILON_POS,
        help="Position success threshold for the minimal readout.",
    )
    parser.add_argument(
        "--tau-line",
        type=float,
        default=DEFAULT_TAU_LINE,
        help="Legacy alias for the explicit orthogonal-distance threshold.",
    )
    parser.add_argument(
        "--tau-orthogonal",
        type=float,
        default=None,
        help="Explicit-rule orthogonal-distance threshold. Falls back to --tau-line when omitted.",
    )
    parser.add_argument(
        "--tau-trend",
        type=float,
        default=DEFAULT_TAU_TREND,
        help="Trend-axis orthogonal-distance threshold for implicit trend fitting readout.",
    )
    parser.add_argument(
        "--tau-yaw",
        type=float,
        default=DEFAULT_TAU_YAW,
        help="Yaw-alignment threshold for shape-completion skill/stress readout.",
    )
    parser.add_argument(
        "--post-place-settle-steps",
        type=int,
        default=24,
        help="Number of hold/settle frames after release before reading the terminal object pose.",
    )
    parser.add_argument(
        "--summary-json-out",
        type=str,
        default=None,
        help="Optional path to write the structured task or suite summary JSON.",
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
    print("GeoMind Task Debug Failure", file=sys.stderr, flush=True)
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
        print(f"eval_suite:     {getattr(args_cli, 'eval_suite', None)}", file=sys.stderr, flush=True)
        print(f"agent:          {getattr(args_cli, 'agent', None)}", file=sys.stderr, flush=True)
        print(f"agent_episodes: {getattr(args_cli, 'agent_episodes', None)}", file=sys.stderr, flush=True)
        print(
            f"settle_steps:   {getattr(args_cli, 'post_place_settle_steps', None)}",
            file=sys.stderr,
            flush=True,
        )
        print(f"device:         {getattr(args_cli, 'device', None)}", file=sys.stderr, flush=True)
        print(f"headless:       {getattr(args_cli, 'headless', None)}", file=sys.stderr, flush=True)

    if env_cfg is not None:
        _print_metadata_first_task_header(env_cfg, stream=sys.stderr)
        print(
            "task_protocol:  " + json.dumps(getattr(env_cfg, "task_protocol", {}), ensure_ascii=False, sort_keys=True),
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

from geomind_lab.data.trajectory import (
    EpisodeCollector,
    build_trajectory_metadata_contract,
    export_episode_h5,
    resample_episode,
)
from geomind_lab.mimic import GeoMindFrankaIKRelMimicEnv, GeoMindFrankaIKRelMimicEnvCfg
from geomind_lab.utils.headless_camera_video import RolloutCameraRecorder, create_rollout_camera_recorder
from geomind_lab.utils.headless_rollout_camera_hook import resolve_rollout_camera_video_dir
from geomind_lab.tasks.manager_based.manipulation.geomind import mdp as geomind_mdp
from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.common import get_asset_pose_w


DOWN_FACING_QUAT = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
GRIPPER_OPEN_COMMAND = 1.0
GRIPPER_CLOSE_COMMAND = -1.0


_EPISODE_SIM_STEP_COUNTER = 0


def _resolve_task_payload(layout_key: str, task_index: int) -> dict[str, Any]:
    return dict(resolve_task_payload(layout_key, task_index))


def _resolve_task_annotation(layout_key: str, task_index: int) -> dict[str, Any]:
    return dict(resolve_task_annotation(layout_key, task_index))


def _resolve_task_label(layout_key: str, task_index: int) -> str:
    return resolve_task_label(layout_key, task_index)


def _is_explicit_geometric_protocol(protocol: Any, *, allowed_rule_types: set[str] | None = None) -> bool:
    if not isinstance(protocol, dict):
        return False
    gt_spec = protocol.get("gt_spec", {})
    if not isinstance(gt_spec, dict):
        return False
    try:
        rule_type = normalize_rule_type(gt_spec.get("rule_type", ""))
    except ValueError:
        return False
    if allowed_rule_types is not None and rule_type not in allowed_rule_types:
        return False
    return (
        str(protocol.get("module_name", "")) == GEOMETRIC_INFERENCE_MODULE_NAME
        and str(protocol.get("track_name", "")) == EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME
        and str(protocol.get("query_family", "")) == "explicit_rule_query"
    )


def _is_implicit_trend_protocol(protocol: Any, *, allowed_subtask_names: set[str] | None = None) -> bool:
    if not isinstance(protocol, dict):
        return False
    subtask_name = str(protocol.get("subtask_name", "")).strip()
    if allowed_subtask_names is not None and subtask_name not in allowed_subtask_names:
        return False
    return (
        str(protocol.get("module_name", "")) == GEOMETRIC_INFERENCE_MODULE_NAME
        and str(protocol.get("track_name", "")) == IMPLICIT_TREND_FITTING_TRACK_NAME
        and str(protocol.get("query_family", "")) == "implicit_trend_query"
    )


def _tau_orthogonal_threshold() -> float:
    if args_cli.tau_orthogonal is not None:
        return float(args_cli.tau_orthogonal)
    return float(args_cli.tau_line)


def _shrink_explicit_minimal_suite(task_specs: Sequence[EvalTaskSpec]) -> tuple[EvalTaskSpec, ...]:
    if not task_specs:
        return tuple()

    preferred_labels = (
        "perfect_midpoint",
        "isometric_extension",
    )
    label_to_spec = {
        str(resolve_task_id(task_spec.label) or str(task_spec.label)): task_spec
        for task_spec in task_specs
    }
    chosen: list[EvalTaskSpec] = []
    seen_rule_types: set[str] = set()

    def maybe_append(task_spec: EvalTaskSpec) -> None:
        protocol = _resolve_task_annotation(task_spec.layout_key, task_spec.task_index).get("protocol", {})
        if not isinstance(protocol, dict):
            return
        gt_spec = protocol.get("gt_spec", {})
        if not isinstance(gt_spec, dict):
            return
        try:
            rule_type = normalize_rule_type(gt_spec.get("rule_type", ""))
        except ValueError:
            return
        if rule_type in seen_rule_types:
            return
        chosen.append(task_spec)
        seen_rule_types.add(rule_type)

    for label in preferred_labels:
        task_spec = label_to_spec.get(label)
        if task_spec is not None:
            maybe_append(task_spec)

    for task_spec in task_specs:
        if len(chosen) >= 2:
            break
        maybe_append(task_spec)

    return tuple(chosen)


def resolve_eval_suite(eval_suite_name: str) -> tuple[EvalTaskSpec, ...]:
    normalized = str(eval_suite_name).strip()
    if normalized in {"structure_primary", "geometric_inference_primary"}:
        allowed_rule_types = {EXTENSION_RULE_TYPE}
        protocol_matcher = lambda protocol: _is_explicit_geometric_protocol(  # noqa: E731
            protocol,
            allowed_rule_types=allowed_rule_types,
        )
    elif normalized == "explicit_minimal":
        allowed_rule_types = {MIDPOINT_RULE_TYPE, EXTENSION_RULE_TYPE}
        protocol_matcher = lambda protocol: _is_explicit_geometric_protocol(  # noqa: E731
            protocol,
            allowed_rule_types=allowed_rule_types,
        )
    elif normalized == "explicit_geometric_minimal":
        allowed_rule_types = {MIDPOINT_RULE_TYPE, EXTENSION_RULE_TYPE}
        protocol_matcher = lambda protocol: _is_explicit_geometric_protocol(  # noqa: E731
            protocol,
            allowed_rule_types=allowed_rule_types,
        )
    elif normalized == "implicit_trend_minimal":
        allowed_subtask_names = {
            TREND_CONTINUATION_SUBTASK_NAME,
            WEAK_DESTRUCTION_CONTROL_SUBTASK_NAME,
        }
        protocol_matcher = lambda protocol: _is_implicit_trend_protocol(  # noqa: E731
            protocol,
            allowed_subtask_names=allowed_subtask_names,
        )
    else:
        raise ValueError(f"Unsupported eval suite: {eval_suite_name!r}")

    payload = load_annotated_tasks()
    task_specs: list[EvalTaskSpec] = []
    for layout_key, layout in sorted(payload.items()):
        if not isinstance(layout, dict):
            continue
        scene_id = int(layout.get("scene_id", 1))
        tasks = layout.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        for task_index, task_payload in enumerate(tasks):
            if not isinstance(task_payload, dict):
                continue
            annotation = task_payload.get("annotation", {})
            if not isinstance(annotation, dict):
                continue
            protocol = annotation.get("protocol", {})
            if not protocol_matcher(protocol):
                continue
            task_specs.append(
                EvalTaskSpec(
                    scene_id=scene_id,
                    layout_key=str(layout_key),
                    task_index=int(task_index),
                    label=_resolve_task_label(str(layout_key), int(task_index)),
                )
            )

    if not task_specs:
        raise ValueError(
            f"Eval suite {eval_suite_name!r} resolved to zero annotated geometric structure-inference backend tasks."
        )
    if normalized == "explicit_minimal":
        task_specs = list(_shrink_explicit_minimal_suite(task_specs))
        if not task_specs:
            raise ValueError(f"Eval suite {eval_suite_name!r} resolved to zero minimal explicit-rule tasks.")
    return tuple(task_specs)


def _compact_summary_payload(payload: object) -> object:
    if isinstance(payload, dict):
        compact: dict[str, object] = {}
        for key, value in payload.items():
            if key == "episodes":
                if isinstance(value, list):
                    compact[key] = [
                        {
                            field: episode.get(field)
                            for field in (
                                "episode_index",
                                "trajectory_seed",
                                "trajectory_h5",
                                "success",
                                "valid_episode",
                            )
                        }
                        for episode in value
                        if isinstance(episode, dict)
                    ]
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


def write_summary_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_compact_summary_payload(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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


def _trajectory_seed(episode_index: int, fallback_seed: int) -> int:
    raw = str(os.environ.get("TRAJECTORY_SEEDS", "")).strip()
    if raw:
        try:
            seeds = [int(item.strip(), 0) for item in raw.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError("TRAJECTORY_SEEDS must contain comma-separated integers.") from exc
        if int(episode_index) >= len(seeds):
            raise ValueError(
                f"TRAJECTORY_SEEDS has {len(seeds)} values but episode {int(episode_index)} was requested."
            )
        return seeds[int(episode_index)]
    return int(fallback_seed) + int(episode_index)


def print_episode_result(result: dict[str, Any]) -> None:
    print("\n" + "=" * 72, flush=True)
    print("episode_result", flush=True)
    print("=" * 72, flush=True)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False), flush=True)

def build_runtime_env_cfg(*, request=None):
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

    def __post_init__(self):
        self.task_id = str(resolved_request.task_id)
        self.task_variant = str(resolved_request.variant)
        self.scene_name = str(resolved_request.scene)
        self.object_mode = str(resolved_request.object_mode)
        self.scene_id = int(resolved_request.scene_id)
        self.task_layout_key = str(resolved_request.layout_key)
        self.task_index = int(resolved_request.task_index)
        self.scene.num_envs = 1
        self.sim.device = args_cli.device
        self.sim.use_fabric = not args_cli.disable_fabric

        base_cfg_cls.__post_init__(self)

        self.viewer.eye = tuple(float(v) for v in args_cli.viewer_eye)
        self.viewer.lookat = tuple(float(v) for v in args_cli.viewer_lookat)

    runtime_cfg_cls = configclass(
        type(
            "RuntimeGeoMindGeometricInferenceDebugCfg",
            (base_cfg_cls,),
            {
                "__doc__": "Runtime GeoMind mimic cfg for metadata-first geometric structure-inference backend debugging.",
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


def restore_geometric_runtime_object_states(env: GeoMindFrankaIKRelMimicEnv) -> None:
    task_spec = dict(getattr(env.cfg, "task_spec", {}) or {})
    runtime_objects = task_spec.get("runtime_objects", [])
    if not isinstance(runtime_objects, list):
        return
    task_protocol = dict(getattr(env.cfg, "task_protocol", {}) or {})
    target_object = task_protocol.get("target_object", {})
    preloaded_target_name = ""
    if isinstance(target_object, dict) and bool(target_object.get("preloaded", False)):
        preloaded_target_name = str(target_object.get("object_id", "")).strip()
    env_ids = torch.arange(int(getattr(env, "num_envs", 1)), device=env.device, dtype=torch.long)
    for runtime_object in runtime_objects:
        if not isinstance(runtime_object, dict):
            continue
        object_name = str(runtime_object.get("name", "")).strip()
        if not object_name or object_name not in env.scene.keys():
            continue
        if object_name == preloaded_target_name:
            continue
        asset = env.scene[object_name]
        root_state = getattr(getattr(asset, "data", None), "default_root_state", None)
        if root_state is None or not hasattr(asset, "write_root_state_to_sim"):
            continue
        state = root_state[env_ids].clone()
        state[:, 7:13] = 0.0
        asset.write_root_state_to_sim(state, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.scene.update(dt=float(env.physics_dt))


def settle_env_after_reset(env: GeoMindFrankaIKRelMimicEnv, settle_steps: int) -> None:
    zero_actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    for _ in range(max(0, int(settle_steps))):
        _, _, terminated, truncated, _ = env.step(zero_actions)
        if should_reset(terminated, truncated):
            env.reset(seed=_env_int("ROBOT_RESET_SEED", 20260831))


def print_runtime_header(
    env_cfg,
    *,
    task_label: str | None = None,
    eval_suite: str | None = None,
    suite_name: str | None = None,
) -> None:
    task_protocol = dict(getattr(env_cfg, "task_protocol", {}) or {})
    print("=" * 72, flush=True)
    print("GeoMind Task Debugger", flush=True)
    print("=" * 72, flush=True)
    _print_metadata_first_task_header(env_cfg, task_label=task_label)
    if eval_suite is not None:
        print(f"eval_suite:     {eval_suite}", flush=True)
    if suite_name is not None:
        print(f"suite_name:     {suite_name}", flush=True)
    if task_label is not None:
        print(f"legacy_task_label:{task_label}", flush=True)
    print(f"layout_key:     {env_cfg.task_layout_key}", flush=True)
    print(f"agent:          {args_cli.agent}", flush=True)
    print(f"agent_episodes: {args_cli.agent_episodes}", flush=True)
    print(f"agent_seed:     {args_cli.agent_seed}", flush=True)
    print(f"settle_steps:   {args_cli.post_place_settle_steps}", flush=True)
    print(f"tau_yaw:        {args_cli.tau_yaw}", flush=True)
    print(f"track_name:     {task_protocol.get('track_name', '')}", flush=True)
    print(
        "task_protocol:  " + json.dumps(task_protocol, indent=None, sort_keys=True, ensure_ascii=False),
        flush=True,
    )


def _derived_agent_seed(task_ordinal: int) -> int:
    return int(args_cli.agent_seed) + int(task_ordinal) * 100003


def _eef_name(env: GeoMindFrankaIKRelMimicEnv) -> str:
    return env._eef_name()


def _pose_from_position_and_quat(position: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    rotation = PoseUtils.matrix_from_quat(quat.unsqueeze(0))[0]
    return PoseUtils.make_pose(position, rotation)


def _pose_from_position_and_yaw(position: torch.Tensor, yaw_rad: float) -> torch.Tensor:
    down_rotation = PoseUtils.matrix_from_quat(DOWN_FACING_QUAT.to(device=position.device).unsqueeze(0))[0]
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    yaw_rotation = torch.tensor(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=position.device,
        dtype=position.dtype,
    )
    return PoseUtils.make_pose(position, yaw_rotation @ down_rotation)


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
    global _EPISODE_SIM_STEP_COUNTER
    target_position, _ = PoseUtils.unmake_pose(target_pose)
    reached = False
    for step_idx in range(max(1, int(steps))):
        action = _action_from_target_pose(env, target_pose, gripper_action)
        _, _, terminated, truncated, _ = env.step(action)
        _EPISODE_SIM_STEP_COUNTER += 1
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


def _current_object_position(
    env: GeoMindFrankaIKRelMimicEnv,
    object_name: str,
    *,
    env_id: int = 0,
) -> torch.Tensor:
    pos_w, _ = get_asset_pose_w(env, object_name)
    return pos_w[env_id, :3].clone().detach()


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> float:
    w = float(quat_wxyz[0].item())
    x = float(quat_wxyz[1].item())
    y = float(quat_wxyz[2].item())
    z = float(quat_wxyz[3].item())
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _current_object_yaw(
    env: GeoMindFrankaIKRelMimicEnv,
    object_name: str,
    *,
    env_id: int = 0,
) -> float:
    _, quat_w = get_asset_pose_w(env, object_name)
    return _yaw_from_quat_wxyz(quat_w[env_id].clone().detach())


def _current_eef_position(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    env_id: int = 0,
) -> torch.Tensor:
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
        float(position[0].item()), float(position[1].item()), float(position[2].item()),
        float(quat_wxyz[1].item()), float(quat_wxyz[2].item()),
        float(quat_wxyz[3].item()), float(quat_wxyz[0].item()),
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
        object_grasped=bool(target_name) and _object_grasped(env, target_name),
        terminated=bool(torch.any(terminated).item()) if isinstance(terminated, torch.Tensor) else bool(terminated),
        truncated=bool(torch.any(truncated).item()) if isinstance(truncated, torch.Tensor) else bool(truncated),
    )


def _run_seeded_place_waypoints(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    target_name: str,
    command_xy: tuple[float, float],
    safe_z: float,
    target_yaw: float | None,
    trajectory_seed: int,
    collector: EpisodeCollector | None,
) -> tuple[bool, dict[str, Any]]:
    requested = _env_flag("TRAJECTORY_AUGMENTATION_ENABLE", False)
    record: dict[str, Any] = {
        "trajectory_seed": int(trajectory_seed),
        "requested": bool(requested),
        "augmentation_enabled": False,
        "algorithm": "geometric_free_space_waypoints_v1",
        "executed_waypoints": [],
        "fallback_reason": "phase5_not_enabled" if not requested else "",
    }
    if not requested:
        return True, record

    max_offset = max(0.036, float(os.environ.get("TRAJECTORY_AUGMENTATION_MAX_LATERAL_OFFSET", "0.06")))
    rng = random.Random(int(trajectory_seed))
    angle = rng.uniform(-math.pi, math.pi)
    radius = rng.uniform(0.035, max_offset)
    offset_x = radius * math.cos(angle)
    offset_y = radius * math.sin(angle)
    waypoints = (
        (command_xy[0] + offset_x, command_xy[1] + offset_y, safe_z + 0.04),
        (command_xy[0] + 0.45 * offset_x, command_xy[1] + 0.45 * offset_y, safe_z + 0.02),
    )
    orientation_arg: torch.Tensor | float = DOWN_FACING_QUAT.to(device=env.device)
    orientation_builder = _pose_from_position_and_quat
    if target_yaw is not None:
        orientation_arg = float(target_yaw)
        orientation_builder = _pose_from_position_and_yaw

    completed = True
    for point in waypoints:
        waypoint_pose = orientation_builder(
            torch.tensor(point, device=env.device, dtype=torch.float32),
            orientation_arg,
        )
        completed = completed and _step_with_target_pose(
            env,
            waypoint_pose,
            GRIPPER_CLOSE_COMMAND,
            steps=45,
            collector=collector,
            stage_name="transit_xy",
            target_name=target_name,
            require_reached=True,
            xy_tolerance=0.035,
            z_tolerance=0.035,
            min_steps=8,
        )
        if not completed:
            break
    record.update(
        {
            "augmentation_enabled": bool(completed),
            "executed_waypoints": [list(point) for point in waypoints] if completed else [],
            "max_lateral_deviation_m": float(radius) if completed else 0.0,
            "fallback_reason": "" if completed else "waypoint_execution_failed",
        }
    )
    return completed, record


def _current_object_xy(
    env: GeoMindFrankaIKRelMimicEnv,
    object_name: str,
    *,
    env_id: int = 0,
) -> tuple[float, float]:
    position = _current_object_position(env, object_name, env_id=env_id)
    return (float(position[0].item()), float(position[1].item()))


def _current_object_positions_xy_map(env: GeoMindFrankaIKRelMimicEnv) -> dict[str, dict[str, list[float]]]:
    task_spec = dict(getattr(env.cfg, "task_spec", {}) or {})
    object_positions = dict(task_spec.get("object_positions", {}) or {})
    return {
        str(object_name): {"position": list(_current_object_xy(env, str(object_name)))}
        for object_name in object_positions.keys()
    }


def _draw_midpoint_gt_success_zone(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    gt_pos_xy: tuple[float, float],
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

    task_spec = dict(getattr(env.cfg, "task_spec", {}) or {})
    object_positions = dict(task_spec.get("object_positions", {}) or {})
    z_samples: list[float] = []
    for payload in object_positions.values():
        if not isinstance(payload, dict):
            continue
        position = payload.get("position")
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            try:
                z_samples.append(float(position[2]))
            except Exception:
                continue
    draw_z = (sum(z_samples) / float(len(z_samples)) if z_samples else 0.95) + 0.012

    cx, cy = float(gt_pos_xy[0]), float(gt_pos_xy[1])
    segment_count = 48
    starts: list[tuple[float, float, float]] = []
    ends: list[tuple[float, float, float]] = []
    for idx in range(segment_count):
        theta0 = 2.0 * math.pi * float(idx) / float(segment_count)
        theta1 = 2.0 * math.pi * float(idx + 1) / float(segment_count)
        p0 = (cx + float(radius) * math.cos(theta0), cy + float(radius) * math.sin(theta0), draw_z)
        p1 = (cx + float(radius) * math.cos(theta1), cy + float(radius) * math.sin(theta1), draw_z)
        starts.append(p0)
        ends.append(p1)

    cross = max(0.01, float(radius) * 0.25)
    starts.extend(
        [
            (cx - cross, cy, draw_z),
            (cx, cy - cross, draw_z),
        ]
    )
    ends.extend(
        [
            (cx + cross, cy, draw_z),
            (cx, cy + cross, draw_z),
        ]
    )

    try:
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()
        colors = [(0.15, 0.95, 0.25, 1.0)] * len(starts)
        sizes = [4.0] * len(starts)
        draw.draw_lines(starts, ends, colors, sizes)
    except Exception:
        return


def _show_gt_zone_enabled() -> bool:
    raw_value = str(os.environ.get("SHOW_GT_ZONE", "1")).strip().lower()
    return raw_value not in {"0", "false", "off", "no"}


def _maybe_draw_gt_success_zone_for_task(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    problem: dict[str, Any],
) -> None:
    if not _show_gt_zone_enabled():
        return
    if str(problem.get("subtask_name", "")).strip() not in {"perfect_midpoint", "trend_continuation", "isometric_extension", "point_symmetry", "weak_destruction_control", "square_completion"}:
        return
    gt_pos_xy = problem.get("gt_pos_xy")
    if not isinstance(gt_pos_xy, (list, tuple)) or len(gt_pos_xy) < 2:
        return
    _draw_midpoint_gt_success_zone(
        env,
        gt_pos_xy=(float(gt_pos_xy[0]), float(gt_pos_xy[1])),
        radius=float(args_cli.epsilon_pos),
    )


def _object_height(env: GeoMindFrankaIKRelMimicEnv, object_name: str) -> float:
    resolver = getattr(env.cfg, "_resolve_object_runtime_height", None)
    if callable(resolver):
        resolved = resolver(object_name)
        if resolved is not None:
            return float(resolved)
    return 0.07


def _object_grasped(
    env: GeoMindFrankaIKRelMimicEnv,
    object_name: str,
    *,
    env_id: int = 0,
) -> bool:
    grasped = geomind_mdp.object_grasped(
        env,
        robot_cfg=SceneEntityCfg("robot"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg(object_name),
    )
    return bool(grasped[env_id].item())


def _protocol_target_is_preloaded(
    env: GeoMindFrankaIKRelMimicEnv,
    target_name: str,
) -> bool:
    task_protocol = dict(getattr(env.cfg, "task_protocol", {}) or {})
    target_object = task_protocol.get("target_object", {})
    if not isinstance(target_object, dict):
        return False
    object_id = str(target_object.get("object_id", "")).strip()
    return object_id == str(target_name).strip() and bool(target_object.get("preloaded", False))


def _protocol_declares_target_preloaded(env: GeoMindFrankaIKRelMimicEnv) -> bool:
    task_protocol = dict(getattr(env.cfg, "task_protocol", {}) or {})
    target_object = task_protocol.get("target_object", {})
    return isinstance(target_object, dict) and "preloaded" in target_object


def _infer_target_surface_z_from_problem(
    env: GeoMindFrankaIKRelMimicEnv,
    problem: dict[str, Any],
    target_name: str,
) -> float:
    anchor_object_ids = problem.get("anchor_object_ids", {})
    if isinstance(anchor_object_ids, dict):
        anchor_z_values: list[float] = []
        for object_id in anchor_object_ids.values():
            object_name = str(object_id).strip()
            if not object_name:
                continue
            try:
                anchor_position = _current_object_position(env, object_name)
                anchor_z_values.append(float(anchor_position[2].item()))
            except Exception:
                continue
        if anchor_z_values:
            return float(sum(anchor_z_values) / float(len(anchor_z_values)))
    target_position = _current_object_position(env, target_name)
    object_height = _object_height(env, target_name)
    return float(target_position[2].item() - max(0.08, 1.2 * object_height))


def _build_preloaded_grasp_summary(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    target_name: str,
    problem: dict[str, Any],
) -> dict[str, object]:
    target_position = _current_object_position(env, target_name)
    eef_position = _current_eef_position(env)
    object_height = _object_height(env, target_name)
    target_surface_z = _infer_target_surface_z_from_problem(env, problem, target_name)
    grasp_sensor_triggered = _object_grasped(env, target_name)
    grasp_offset_xy = (
        float(target_position[0].item() - eef_position[0].item()),
        float(target_position[1].item() - eef_position[1].item()),
    )
    return {
        "target_name": target_name,
        "target_height": float(object_height),
        "target_surface_z": float(target_surface_z),
        "approach_z": None,
        "grasp_z": None,
        "lift_z": None,
        "execution_completed": True,
        "terminated_early": False,
        "grasp_success": True,
        "grasp_sensor_triggered": bool(grasp_sensor_triggered),
        "object_lifted": True,
        "grasp_offset_xy": [float(grasp_offset_xy[0]), float(grasp_offset_xy[1])],
        "final_target_position": [float(value) for value in target_position.tolist()],
        "grasp_bypass_reason": "target_preloaded_on_gripper",
    }


def _release_preloaded_target_lock(env: GeoMindFrankaIKRelMimicEnv, target_name: str) -> None:
    task_protocol = getattr(env.cfg, "task_protocol", {})
    if not isinstance(task_protocol, dict):
        return
    target_object = task_protocol.get("target_object", {})
    if not isinstance(target_object, dict):
        return
    if str(target_object.get("object_id", "")).strip() != str(target_name).strip():
        return
    setattr(env, "_geomind_debug_preloaded_target_name", str(target_name).strip())
    target_object["preloaded"] = False


def _restore_preloaded_target_declaration(env: GeoMindFrankaIKRelMimicEnv) -> None:
    target_name = str(getattr(env, "_geomind_debug_preloaded_target_name", "")).strip()
    if not target_name:
        return
    task_protocol = getattr(env.cfg, "task_protocol", {})
    target_object = task_protocol.get("target_object", {}) if isinstance(task_protocol, dict) else {}
    if isinstance(target_object, dict) and str(target_object.get("object_id", "")).strip() == target_name:
        target_object["preloaded"] = True


def run_scripted_grasp_for_target(
    env: GeoMindFrankaIKRelMimicEnv,
    target_name: str,
    *,
    precise_mode: bool = False,
    collector: EpisodeCollector | None = None,
) -> dict[str, object]:
    target_position = _current_object_position(env, target_name)
    object_height = _object_height(env, target_name)
    target_surface_z = float(target_position[2].item())
    approach_z = float(target_surface_z + max(0.18, 1.5 * object_height))
    grasp_z = float(target_surface_z + max(0.0, 0.05 * object_height))
    lift_z = float(target_surface_z + max(0.18, 2.0 * object_height))

    approach_pose = _pose_from_position_and_quat(
        torch.tensor(
            [target_position[0].item(), target_position[1].item(), approach_z],
            device=env.device,
            dtype=torch.float32,
        ),
        DOWN_FACING_QUAT.to(device=env.device),
    )
    grasp_pose = _pose_from_position_and_quat(
        torch.tensor(
            [target_position[0].item(), target_position[1].item(), grasp_z],
            device=env.device,
            dtype=torch.float32,
        ),
        DOWN_FACING_QUAT.to(device=env.device),
    )
    lift_pose = _pose_from_position_and_quat(
        torch.tensor(
            [target_position[0].item(), target_position[1].item(), lift_z],
            device=env.device,
            dtype=torch.float32,
        ),
        DOWN_FACING_QUAT.to(device=env.device),
    )

    approach_reached = True
    descend_reached = True
    locked_target_position = target_position
    if precise_mode:
        precise_grasp_z = float(target_surface_z - max(0.006, 0.15 * object_height))
        grasp_z = precise_grasp_z
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
        if approach_reached:
            locked_target_position = _current_object_position(env, target_name)
            grasp_pose = _pose_from_position_and_quat(
                torch.tensor(
                    [locked_target_position[0].item(), locked_target_position[1].item(), precise_grasp_z],
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
        completed = (
            approach_reached
            and descend_reached
            and _step_with_target_pose(env, grasp_pose, GRIPPER_CLOSE_COMMAND, steps=36, collector=collector, stage_name="close", target_name=target_name)
            and _step_with_target_pose(env, lift_pose, GRIPPER_CLOSE_COMMAND, steps=24, collector=collector, stage_name="lift", target_name=target_name)
            and _step_with_target_pose(env, lift_pose, GRIPPER_CLOSE_COMMAND, steps=8, collector=collector, stage_name="hold", target_name=target_name)
        )
    else:
        completed = (
            _step_with_target_pose(env, approach_pose, GRIPPER_OPEN_COMMAND, steps=30, collector=collector, stage_name="approach", target_name=target_name)
            and _step_with_target_pose(env, grasp_pose, GRIPPER_OPEN_COMMAND, steps=20, collector=collector, stage_name="descend", target_name=target_name)
            and _step_with_target_pose(env, grasp_pose, GRIPPER_CLOSE_COMMAND, steps=28, collector=collector, stage_name="close", target_name=target_name)
            and _step_with_target_pose(env, lift_pose, GRIPPER_CLOSE_COMMAND, steps=24, collector=collector, stage_name="lift", target_name=target_name)
            and _step_with_target_pose(env, lift_pose, GRIPPER_CLOSE_COMMAND, steps=8, collector=collector, stage_name="hold", target_name=target_name)
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
        "execution_completed": bool(completed),
        "terminated_early": not bool(completed),
        "approach_reached": bool(approach_reached),
        "descend_reached": bool(descend_reached),
        "locked_target_xy": [float(locked_target_position[0].item()), float(locked_target_position[1].item())],
        "grasp_success": bool(grasp_success),
        "grasp_sensor_triggered": bool(grasp_sensor_triggered),
        "object_lifted": bool(object_lifted),
        "grasp_offset_xy": [float(grasp_offset_xy[0]), float(grasp_offset_xy[1])],
        "final_target_position": [float(value) for value in final_position.tolist()],
    }


def run_scripted_place_for_target(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    target_name: str,
    target_pos_xy: tuple[float, float],
    target_yaw: float | None,
    target_surface_z: float,
    object_height: float,
    post_place_settle_steps: int,
    grasp_offset_xy: tuple[float, float] = (0.0, 0.0),
    collector: EpisodeCollector | None = None,
    trajectory_seed: int = 0,
    preloaded_target: bool = False,
) -> dict[str, object]:
    target_x, target_y = (float(target_pos_xy[0]), float(target_pos_xy[1]))
    command_x = float(target_x - float(grasp_offset_xy[0]))
    command_y = float(target_y - float(grasp_offset_xy[1]))
    approach_z = float(target_surface_z + max(0.14, 1.5 * object_height))
    place_z = float(target_surface_z + max(0.02, 0.1 * object_height))
    retreat_z = float(target_surface_z + max(0.18, 2.0 * object_height))

    orientation_builder = _pose_from_position_and_quat
    orientation_arg = DOWN_FACING_QUAT.to(device=env.device)
    if target_yaw is not None:
        orientation_builder = _pose_from_position_and_yaw
        orientation_arg = float(target_yaw)

    approach_pose = orientation_builder(
        torch.tensor([command_x, command_y, approach_z], device=env.device, dtype=torch.float32),
        orientation_arg,
    )
    place_pose = orientation_builder(
        torch.tensor([command_x, command_y, place_z], device=env.device, dtype=torch.float32),
        orientation_arg,
    )
    retreat_pose = orientation_builder(
        torch.tensor([command_x, command_y, retreat_z], device=env.device, dtype=torch.float32),
        orientation_arg,
    )

    waypoint_completed, approach_augmentation = _run_seeded_place_waypoints(
        env,
        target_name=target_name,
        command_xy=(command_x, command_y),
        safe_z=retreat_z,
        target_yaw=target_yaw,
        trajectory_seed=trajectory_seed,
        collector=collector,
    )
    completed = waypoint_completed
    completed = completed and _step_with_target_pose(
        env, approach_pose, GRIPPER_CLOSE_COMMAND, steps=24,
        collector=collector, stage_name="transit_xy", target_name=target_name,
    )
    completed = completed and _step_with_target_pose(
        env, place_pose, GRIPPER_CLOSE_COMMAND, steps=18,
        collector=collector, stage_name="descend_place", target_name=target_name,
    )
    if completed and preloaded_target:
        _release_preloaded_target_lock(env, target_name)
    completed = completed and _step_with_target_pose(
        env, place_pose, GRIPPER_OPEN_COMMAND, steps=18,
        collector=collector, stage_name="open", target_name=target_name,
    )
    completed = completed and _step_with_target_pose(
        env, retreat_pose, GRIPPER_OPEN_COMMAND, steps=18,
        collector=collector, stage_name="retreat", target_name=target_name,
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
    pred_yaw = _current_object_yaw(env, target_name)
    return {
        "target_name": target_name,
        "target_pos_xy": [target_x, target_y],
        "target_yaw": None if target_yaw is None else float(target_yaw),
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
        "pred_yaw": float(pred_yaw),
        "final_target_position": [float(value) for value in final_position.tolist()],
        "approach_augmentation": approach_augmentation,
    }


def run_geometric_inference_episode(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    agent: GeometricInferenceAgent,
    episode_index: int,
    task_label: str,
    agent_seed: int | None = None,
    recorder: RolloutCameraRecorder | None = None,
) -> dict[str, Any]:
    global _EPISODE_SIM_STEP_COUNTER
    _EPISODE_SIM_STEP_COUNTER = 0
    _restore_preloaded_target_declaration(env)
    env.reset(seed=_env_int("ROBOT_RESET_SEED", 20260831))
    restore_geometric_runtime_object_states(env)
    settle_env_after_reset(env, _env_int("RESET_SETTLE_STEPS", 24))
    restore_geometric_runtime_object_states(env)
    if recorder is not None:
        recorder.capture_now()

    task_spec = dict(getattr(env.cfg, "task_spec", {}) or {})
    trajectory_seed = _trajectory_seed(
        episode_index,
        int(agent_seed) if agent_seed is not None else int(args_cli.agent_seed),
    )
    h5_path = _trajectory_h5_path(episode_index)
    collector: EpisodeCollector | None = None
    if h5_path is not None:
        collector = EpisodeCollector(
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
                "augmentation_record": {"trajectory_seed": trajectory_seed, "augmentation_enabled": False, "waypoint_order": [], "fallback_reason": "phase5_not_enabled"},
                "segmenter_version": str(args_cli.trajectory_segment_source),
                "segment_source": str(args_cli.trajectory_segment_source),
                "eef_action_quaternion_order": "xyzw",
                "timestamp_source": "episode_length_buf_times_step_dt",
                **build_trajectory_metadata_contract(
                    env_cfg=env.cfg,
                    task_spec=task_spec,
                    sample_hz=float(args_cli.trajectory_sample_hz),
                    initial_object_poses=dict(task_spec.get("object_positions", {}) or {}),
                ),
            },
            sample_hz=float(args_cli.trajectory_sample_hz),
        )
    runtime_object_positions = _current_object_positions_xy_map(env)
    problem = build_geometric_inference_problem_from_object_positions(task_spec, runtime_object_positions)
    _maybe_draw_gt_success_zone_for_task(env, problem=problem)
    target_object_id = str(problem.get("target_object_id", ""))
    if not target_object_id:
        raise ValueError("Geometric inference task is missing protocol.target_object.object_id.")

    agent_input = GeometricInferenceAgentInput(
        query_text=str(problem["query_text"]),
        anchor_positions_xy=dict(problem["anchor_positions_xy"]),
        protocol_snapshot=problem,
        gt_pos_xy=tuple(problem["gt_pos_xy"]),
        gt_yaw_rad=(
            float(problem["gt_yaw_rad"])
            if problem.get("gt_yaw_rad") is not None
            else None
        ),
        workspace_bounds_xy=tuple(problem["workspace_bounds_xy"]),
    )
    agent_output = agent(agent_input)
    target_pos_xy = tuple(float(value) for value in agent_output.target_pos_xy)
    target_yaw = None if agent_output.target_yaw is None else float(agent_output.target_yaw)

    preloaded_target = _protocol_target_is_preloaded(env, target_object_id)
    if not _protocol_declares_target_preloaded(env):
        preloaded_target = preloaded_target or _object_grasped(env, target_object_id)
    if preloaded_target:
        grasp_summary = _build_preloaded_grasp_summary(
            env,
            target_name=target_object_id,
            problem=problem,
        )
    else:
        precise_grasp_mode = (
            str(problem.get("subtask_name", "")).strip() in {"perfect_midpoint", "trend_continuation", "isometric_extension", "point_symmetry", "weak_destruction_control", "square_completion"}
            or str(task_label).strip().startswith("perfect_midpoint")
            or str(task_label).strip().startswith("trend_continuation")
            or str(task_label).strip().startswith("isometric_extension")
            or str(task_label).strip().startswith("point_symmetry")
            or str(task_label).strip().startswith("weak_destruction_control")
            or str(task_label).strip().startswith("square_completion")
        )
        grasp_summary = run_scripted_grasp_for_target(
            env,
            target_object_id,
            precise_mode=precise_grasp_mode,
            collector=collector,
        )
    if bool(grasp_summary["grasp_success"]):
        place_summary = run_scripted_place_for_target(
            env,
            target_name=target_object_id,
            target_pos_xy=target_pos_xy,
            target_yaw=target_yaw,
            target_surface_z=float(grasp_summary["target_surface_z"]),
            object_height=float(grasp_summary["target_height"]),
            post_place_settle_steps=args_cli.post_place_settle_steps,
            grasp_offset_xy=tuple(float(value) for value in grasp_summary["grasp_offset_xy"]),
            collector=collector,
            trajectory_seed=trajectory_seed,
            preloaded_target=preloaded_target,
        )
    else:
        place_summary = {
            "target_name": target_object_id,
            "target_pos_xy": [float(target_pos_xy[0]), float(target_pos_xy[1])],
            "target_yaw": target_yaw,
            "execution_completed": False,
            "terminated_early": False,
            "release_detected": False,
            "settle_applied": False,
            "pred_pos_xy": list(_current_object_xy(env, target_object_id)),
            "pred_yaw": float(_current_object_yaw(env, target_object_id)),
        }

    pred_pos_xy = tuple(float(value) for value in place_summary["pred_pos_xy"])
    pred_yaw = float(place_summary["pred_yaw"])
    placement_execution_success = bool(place_summary["execution_completed"]) and bool(place_summary["release_detected"])
    settle_applied = bool(place_summary["settle_applied"]) or int(args_cli.post_place_settle_steps) <= 0
    valid_episode = bool(grasp_summary["grasp_success"]) and placement_execution_success and settle_applied

    result = build_geometric_inference_debug_result(
        problem,
        target_pos_xy=target_pos_xy,
        pred_pos_xy=pred_pos_xy,
        epsilon_pos=args_cli.epsilon_pos,
        tau_line=_tau_orthogonal_threshold(),
        tau_trend=args_cli.tau_trend,
        target_yaw=target_yaw,
        pred_yaw=pred_yaw,
        tau_yaw=args_cli.tau_yaw,
        valid_episode=valid_episode,
        placement_execution_success=placement_execution_success,
        settle_applied=settle_applied,
    )
    result["agent_name"] = str(agent.name)
    result["agent_seed"] = int(agent_seed) if agent_seed is not None else None
    result["trajectory_seed"] = int(trajectory_seed)
    result["episode_index"] = int(episode_index)
    result["task_label"] = str(task_label)
    result["scene_id"] = int(getattr(env.cfg, "scene_id", -1))
    result["layout_key"] = str(getattr(env.cfg, "task_layout_key", ""))
    result["task_index"] = int(getattr(env.cfg, "task_index", -1))
    result["rule_type"] = str(problem.get("rule_type", ""))
    result["target_object_id"] = target_object_id
    result["grasp_execution_success"] = bool(grasp_summary["grasp_success"])
    result["release_detected"] = bool(place_summary["release_detected"])
    result["grasp_execution"] = grasp_summary
    result["placement_execution"] = place_summary
    result["trace_step_count"] = int(_EPISODE_SIM_STEP_COUNTER)
    if collector is not None and h5_path is not None and collector.records:
        augmentation_record = dict(place_summary.get("approach_augmentation", {}) or {})
        collector.metadata["trajectory_augmentation_requested"] = bool(augmentation_record.get("requested", False))
        collector.metadata["trajectory_augmentation_enabled"] = bool(augmentation_record.get("augmentation_enabled", False))
        collector.metadata["augmentation_record"] = augmentation_record
        collector.metadata["augmented_atomic_ids"] = [1] if bool(augmentation_record.get("augmentation_enabled", False)) else []
        sampled_episode = resample_episode(
            collector.as_dict(),
            target_hz=float(args_cli.trajectory_sample_hz),
            segment_source=str(args_cli.trajectory_segment_source),
        )
        export_episode_h5(sampled_episode, h5_path)
        result["trajectory_h5"] = str(h5_path)
    else:
        result["trajectory_h5"] = None
    result["segment_source"] = str(args_cli.trajectory_segment_source)
    print_episode_result(result)
    return result


def run_agent_rollout(
    env: GeoMindFrankaIKRelMimicEnv,
    *,
    num_episodes: int,
    task_label: str,
    task_ordinal: int = 0,
) -> dict[str, Any]:
    agent_seed = _derived_agent_seed(task_ordinal)
    agent = build_geometric_inference_agent(args_cli.agent, seed=agent_seed)
    episode_results: list[dict[str, Any]] = []

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
            result = run_geometric_inference_episode(
                env,
                agent=agent,
                episode_index=episode_index,
                task_label=task_label,
                agent_seed=agent_seed,
                recorder=recorder,
            )
        finally:
            if recorder is not None:
                recorder.uninstall()
        episode_results.append(result)
        if episode_index == 0 and video_dir is not None and recorder is not None:
            paths = recorder.save_mp4s(video_dir, fps=fps_roll)
            _print_rollout_camera_video_result(paths, video_dir)

    summary = build_geometric_inference_task_summary(
        agent_name=agent.name,
        episode_results=episode_results,
        scene_id=int(getattr(env.cfg, "scene_id", -1)),
        layout_key=str(getattr(env.cfg, "task_layout_key", "")),
        task_index=int(getattr(env.cfg, "task_index", -1)),
        task_label=task_label,
    )
    summary["agent_seed"] = int(agent_seed)
    summary["trace_step_count"] = int(sum(int(item.get("trace_step_count", 0) or 0) for item in episode_results))

    print("\n" + "=" * 72, flush=True)
    print("task_rollout_summary", flush=True)
    print("=" * 72, flush=True)
    print(json.dumps(_compact_summary_payload(summary), indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return summary


def run_agent_eval_suite(
    *,
    num_episodes: int,
    eval_suite: str,
) -> dict[str, Any]:
    task_specs = resolve_eval_suite(eval_suite)
    suite_name = f"{eval_suite}_{args_cli.agent}"
    task_summaries: list[dict[str, Any]] = []

    for task_ordinal, task_spec in enumerate(task_specs):
        env = None
        env_cfg = build_runtime_env_cfg(
            request=resolve_task_request(
                scene_id=task_spec.scene_id,
                layout_key=task_spec.layout_key,
                task_index=task_spec.task_index,
                object_mode=args_cli.object_mode,
            ),
        )
        try:
            print_runtime_header(
                env_cfg,
                task_label=task_spec.label,
                eval_suite=eval_suite,
                suite_name=suite_name,
            )
            env = GeoMindFrankaIKRelMimicEnv(cfg=env_cfg)
            with torch.inference_mode():
                task_summary = run_agent_rollout(
                    env,
                    num_episodes=num_episodes,
                    task_label=task_spec.label,
                    task_ordinal=task_ordinal,
                )
            task_summary["suite_name"] = suite_name
            task_summaries.append(task_summary)
        finally:
            if env is not None:
                env.close()

    suite_summary = build_geometric_inference_suite_summary(
        suite_name=suite_name,
        agent_name=args_cli.agent,
        task_summaries=task_summaries,
    )
    suite_summary["eval_suite"] = str(eval_suite)

    print("\n" + "=" * 72, flush=True)
    print("suite_rollout_summary", flush=True)
    print("=" * 72, flush=True)
    print(json.dumps(_compact_summary_payload(suite_summary), indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return suite_summary


def main() -> None:
    env = None
    env_cfg = None
    stage = "build_runtime_env_cfg"

    try:
        if args_cli.eval_suite is not None:
            if args_cli.trajectory_h5_out:
                raise ValueError("--trajectory-h5-out is supported only for a single geometric task, not --eval-suite.")
            stage = "run_agent_eval_suite"
            with torch.inference_mode():
                suite_summary = run_agent_eval_suite(
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
        task_label = _resolve_task_label(resolved_request.layout_key, resolved_request.task_index)
        env_cfg = build_runtime_env_cfg(request=resolved_request)
        print_runtime_header(env_cfg, task_label=task_label)

        stage = "create_env"
        env = GeoMindFrankaIKRelMimicEnv(cfg=env_cfg)

        stage = "run_agent_rollout"
        with torch.inference_mode():
            task_summary = run_agent_rollout(
                env,
                num_episodes=args_cli.agent_episodes,
                task_label=task_label,
                task_ordinal=0,
            )

        if args_cli.summary_json_out:
            write_summary_json(args_cli.summary_json_out, task_summary)
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
