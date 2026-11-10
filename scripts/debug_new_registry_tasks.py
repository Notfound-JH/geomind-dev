#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import faulthandler
import json
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

from geomind_lab.mimic.utils.family_runner_registry import resolve_runner_family
from geomind_lab.mimic.utils.task_metadata_registry import (
    TASK_VERSION_BASE,
    build_task_metadata,
    resolve_task_id,
    resolve_task_version,
    supported_task_versions,
)
from geomind_lab.utils import asset_utils, scene_utils


SCENE_CATEGORIES = ("function", "object", "container", "decoration")
DEFAULT_SCENE_ID = 10
DEFAULT_TASK_IDS = (
    "reindex_extreme",
    "reindex_insert",
    "track_single",
    "track_multi",
)
DEFAULT_GOAL_TOLERANCE = 0.05
DEFAULT_RESET_SETTLE_STEPS = 40
DEFAULT_INTERVENTION_SETTLE_STEPS = 18
DEFAULT_TARGET_SETTLE_STEPS = 24


@dataclass(frozen=True)
class SimpleSceneProfile:
    scene_id: int
    usd_path: str
    object_names: tuple[str, ...]
    object_positions_xy: dict[str, tuple[float, float]]
    basket_name: str | None
    basket_xy: tuple[float, float] | None
    table_name: str | None


@dataclass(frozen=True)
class InterventionStep:
    object_name: str
    target_xy: tuple[float, float]
    reason: str


@dataclass(frozen=True)
class TaskPlan:
    requested_task_id: str
    resolved_task_id: str
    task_version: str
    scene_id: int
    supported: bool
    unsupported_reason: str
    instruction: str
    runner_family: str
    family: str
    semantic_module: str
    skill_type: str
    enabled_probes: tuple[str, ...]
    notes: str
    initial_order: tuple[str, ...]
    planned_order_after_interventions: tuple[str, ...]
    target_object_name: str | None
    goal_xy: tuple[float, float] | None
    goal_tolerance: float
    interventions: tuple[InterventionStep, ...]
    planned_object_positions_xy: dict[str, tuple[float, float]]


def _as_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _coerce_tuple(value: Any, length: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)) and len(value) >= length:
        return tuple(float(item) for item in value[:length])
    return default


def _node_to_xform(node: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    attrs = node.get("attrs", {}) if isinstance(node, dict) else {}
    return {
        "pos": _coerce_tuple(attrs.get("translate"), 3, (0.0, 0.0, 0.0)),
        "rot": _coerce_tuple(attrs.get("orient"), 4, (0.0, 0.0, 0.0, 1.0)),
        "scale": _coerce_tuple(attrs.get("scale"), 3, (1.0, 1.0, 1.0)),
    }


def _load_scene_structure(nodes: list[scene_utils.PrimNode]) -> dict[str, dict[str, dict[str, Any]]]:
    xform_tree = scene_utils.extract_xform_tree(nodes)
    category_nodes: dict[str, dict[str, Any]] = {}

    def walk(node: dict[str, Any]) -> None:
        name = str(node.get("name", ""))
        if name in SCENE_CATEGORIES and name not in category_nodes:
            category_nodes[name] = node
        for child in node.get("children", []):
            if isinstance(child, dict):
                walk(child)

    for node in xform_tree:
        walk(node)

    structured: dict[str, dict[str, dict[str, Any]]] = {}
    for category in SCENE_CATEGORIES:
        children = category_nodes.get(category, {}).get("children", [])
        structured[category] = {
            str(child.get("name", "")): child for child in children if isinstance(child, dict) and str(child.get("name", ""))
        }
    return structured


def _select_asset_name(structure: dict[str, dict[str, dict[str, Any]]], category: str, prefix: str) -> str | None:
    matches = [name for name in structure.get(category, {}) if name.startswith(prefix)]
    return matches[0] if matches else None


def resolve_scene_profile(scene_id: int) -> SimpleSceneProfile:
    assets_root = asset_utils.resolve_assets_root()
    stage_dir = assets_root / "stage" / str(scene_id)
    new_path = stage_dir / f"{scene_id}_new.usda"
    scene_path = new_path if new_path.is_file() else stage_dir / f"{scene_id}.usda"
    nodes = scene_utils.parse_usda_scene(scene_path)
    structure = _load_scene_structure(nodes)

    object_positions_xy: dict[str, tuple[float, float]] = {}
    for name, node in structure.get("object", {}).items():
        pos = _node_to_xform(node)["pos"]
        object_positions_xy[name] = (float(pos[0]), float(pos[1]))

    basket_name = _select_asset_name(structure, "container", "basket")
    basket_xy = None
    if basket_name is not None:
        basket_pos = _node_to_xform(structure["container"][basket_name])["pos"]
        basket_xy = (float(basket_pos[0]), float(basket_pos[1]))

    return SimpleSceneProfile(
        scene_id=int(scene_id),
        usd_path=str(scene_path),
        object_names=tuple(structure.get("object", {}).keys()),
        object_positions_xy=object_positions_xy,
        basket_name=basket_name,
        basket_xy=basket_xy,
        table_name=_select_asset_name(structure, "function", "table"),
    )


def sort_objects_by_xy(scene_profile: SimpleSceneProfile, positions_xy: dict[str, tuple[float, float]] | None = None) -> tuple[str, ...]:
    resolved_positions = scene_profile.object_positions_xy if positions_xy is None else positions_xy
    return tuple(
        sorted(
            resolved_positions.keys(),
            key=lambda name: (
                float(resolved_positions[name][0]),
                float(resolved_positions[name][1]),
                str(name),
            ),
        )
    )


def _workspace_points(scene_profile: SimpleSceneProfile, initial_order: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    positions = scene_profile.object_positions_xy
    xs = [float(positions[name][0]) for name in initial_order]
    ys = [float(positions[name][1]) for name in initial_order]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    basket_x = float(scene_profile.basket_xy[0]) if scene_profile.basket_xy is not None else float(sum(xs) / len(xs))
    basket_y = float(scene_profile.basket_xy[1]) if scene_profile.basket_xy is not None else float(min_y - 0.25)
    second_x = float(positions[initial_order[1]][0]) if len(initial_order) >= 2 else min_x + 0.04

    return {
        "insert_xy": ((min_x + second_x) * 0.5, min_y - 0.10),
        "left_stage_xy": (min_x - 0.02, min_y - 0.18),
        "right_stage_xy": (max_x + 0.04, min_y - 0.10),
        "delivery_xy": (basket_x, basket_y + 0.13),
    }


def _replace_position(
    positions_xy: dict[str, tuple[float, float]],
    object_name: str,
    new_xy: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    updated = dict(positions_xy)
    updated[object_name] = (float(new_xy[0]), float(new_xy[1]))
    return updated


def _unsupported_plan(
    *,
    requested_task_id: str,
    resolved_task_id: str,
    task_version: str,
    scene_id: int,
    metadata: dict[str, Any],
    reason: str,
    initial_order: tuple[str, ...],
) -> TaskPlan:
    return TaskPlan(
        requested_task_id=requested_task_id,
        resolved_task_id=resolved_task_id,
        task_version=task_version,
        scene_id=scene_id,
        supported=False,
        unsupported_reason=reason,
        instruction=_as_str(metadata.get("instruction")),
        runner_family=resolve_runner_family(
            task_id=metadata.get("task_id"),
            family=metadata.get("family"),
            semantic_module=metadata.get("semantic_module"),
            skill_type=metadata.get("skill_type"),
        ),
        family=_as_str(metadata.get("family")),
        semantic_module=_as_str(metadata.get("semantic_module")),
        skill_type=_as_str(metadata.get("skill_type")),
        enabled_probes=tuple(str(item) for item in metadata.get("enabled_probes", [])),
        notes=_as_str(metadata.get("notes")),
        initial_order=initial_order,
        planned_order_after_interventions=initial_order,
        target_object_name=None,
        goal_xy=None,
        goal_tolerance=DEFAULT_GOAL_TOLERANCE,
        interventions=tuple(),
        planned_object_positions_xy={},
    )


def build_task_plan(
    *,
    scene_profile: SimpleSceneProfile,
    requested_task_id: str,
    task_version: str,
) -> TaskPlan:
    resolved_task_id = resolve_task_id(requested_task_id)
    if not resolved_task_id:
        raise ValueError(f"Unknown task_id: {requested_task_id!r}")

    resolved_version = resolve_task_version(resolved_task_id, explicit_task_version=task_version)
    metadata = build_task_metadata(resolved_task_id, task_version=resolved_version)
    initial_order = sort_objects_by_xy(scene_profile)

    if len(initial_order) < 5:
        return _unsupported_plan(
            requested_task_id=requested_task_id,
            resolved_task_id=resolved_task_id,
            task_version=resolved_version,
            scene_id=scene_profile.scene_id,
            metadata=metadata,
            reason="Scene must expose at least 5 movable objects for the current scene-10 smoke planners.",
            initial_order=initial_order,
        )
    if scene_profile.basket_xy is None:
        return _unsupported_plan(
            requested_task_id=requested_task_id,
            resolved_task_id=resolved_task_id,
            task_version=resolved_version,
            scene_id=scene_profile.scene_id,
            metadata=metadata,
            reason="Scene is missing a basket/container anchor for the delivery goal.",
            initial_order=initial_order,
        )
    if scene_profile.scene_id != DEFAULT_SCENE_ID:
        return _unsupported_plan(
            requested_task_id=requested_task_id,
            resolved_task_id=resolved_task_id,
            task_version=resolved_version,
            scene_id=scene_profile.scene_id,
            metadata=metadata,
            reason=f"Current planner is tuned for scene_id={DEFAULT_SCENE_ID} only.",
            initial_order=initial_order,
        )

    points = _workspace_points(scene_profile, initial_order)
    planned_positions = dict(scene_profile.object_positions_xy)
    interventions: list[InterventionStep] = []
    target_name: str | None = None
    planner_note = ""

    if resolved_task_id == "reindex_extreme":
        target_name = initial_order[0]
        planner_note = "Use the current leftmost object directly and deliver it to the basket-front staging zone."
    elif resolved_task_id == "reindex_insert":
        moved_object = initial_order[-1]
        insert_xy = points["insert_xy"]
        interventions.append(
            InterventionStep(
                object_name=moved_object,
                target_xy=insert_xy,
                reason="Move the original rightmost object into the second slot before resolving the current order.",
            )
        )
        planned_positions = _replace_position(planned_positions, moved_object, insert_xy)
        target_name = sort_objects_by_xy(scene_profile, planned_positions)[1]
        planner_note = "Single insertion refresh: after the staged move, pick the current second-left object."
    elif resolved_task_id == "track_single":
        target_name = initial_order[len(initial_order) // 2]
        move_xy = points["right_stage_xy"]
        interventions.append(
            InterventionStep(
                object_name=initial_order[0],
                target_xy=move_xy,
                reason="Displace a non-target object so the final order changes while the queried identity remains the original middle object.",
            )
        )
        planned_positions = _replace_position(planned_positions, initial_order[0], move_xy)
        planner_note = "Single-step identity tracking: remember the initial middle object across one staged reorder."
    elif resolved_task_id == "track_multi":
        target_name = initial_order[0]
        right_move_xy = points["right_stage_xy"]
        left_move_xy = points["left_stage_xy"]
        interventions.extend(
            (
                InterventionStep(
                    object_name=initial_order[2],
                    target_xy=right_move_xy,
                    reason="First shuffle step: move an interior object to the far right.",
                ),
                InterventionStep(
                    object_name=initial_order[-1],
                    target_xy=left_move_xy,
                    reason="Second shuffle step: move the original rightmost object to the far left.",
                ),
            )
        )
        planned_positions = _replace_position(planned_positions, initial_order[2], right_move_xy)
        planned_positions = _replace_position(planned_positions, initial_order[-1], left_move_xy)
        planner_note = "Two-step identity tracking: keep the original leftmost identity through two deterministic reorder moves."
    else:
        unsupported_reasons = {
            "reindex_subset": "Scene 10 has no subset-defining visual attribute split for the red-only subset query.",
            "reindex_neighbor": "Scene 10 lacks the required typed reference object for the red-anchor neighbor query.",
            "track_clutter": "Annotated runnable instances live under clutter_robust_identity_blocks_line_3 / _tight; the scene-10 five-object smoke planner is not wired for this task yet.",
        }
        return _unsupported_plan(
            requested_task_id=requested_task_id,
            resolved_task_id=resolved_task_id,
            task_version=resolved_version,
            scene_id=scene_profile.scene_id,
            metadata=metadata,
            reason=unsupported_reasons.get(
                resolved_task_id,
                f"Task {resolved_task_id!r} is not part of the scene-10 smoke subset yet.",
            ),
            initial_order=initial_order,
        )

    planned_order = sort_objects_by_xy(scene_profile, planned_positions)
    runner_family = resolve_runner_family(
        task_id=metadata.get("task_id"),
        family=metadata.get("family"),
        semantic_module=metadata.get("semantic_module"),
        skill_type=metadata.get("skill_type"),
    )
    notes = _as_str(metadata.get("notes"))
    if planner_note:
        notes = f"{notes} {planner_note}".strip()

    return TaskPlan(
        requested_task_id=requested_task_id,
        resolved_task_id=resolved_task_id,
        task_version=resolved_version,
        scene_id=scene_profile.scene_id,
        supported=True,
        unsupported_reason="",
        instruction=_as_str(metadata.get("instruction")),
        runner_family=runner_family,
        family=_as_str(metadata.get("family")),
        semantic_module=_as_str(metadata.get("semantic_module")),
        skill_type=_as_str(metadata.get("skill_type")),
        enabled_probes=tuple(str(item) for item in metadata.get("enabled_probes", [])),
        notes=notes,
        initial_order=initial_order,
        planned_order_after_interventions=planned_order,
        target_object_name=target_name,
        goal_xy=tuple(float(value) for value in points["delivery_xy"]),
        goal_tolerance=DEFAULT_GOAL_TOLERANCE,
        interventions=tuple(interventions),
        planned_object_positions_xy=planned_positions,
    )


def _goal_sequence_for_plan(plan: TaskPlan) -> list[dict[str, Any]]:
    if plan.target_object_name is None or plan.goal_xy is None:
        return []
    return [
        {
            "step": 1,
            "goals": [
                {
                    "object_id": plan.target_object_name,
                    "xy": [float(plan.goal_xy[0]), float(plan.goal_xy[1])],
                    "tolerance": {"pos_r": float(plan.goal_tolerance)},
                }
            ],
        }
    ]


def build_manual_task_spec(scene_profile: SimpleSceneProfile, plan: TaskPlan) -> dict[str, Any]:
    object_positions = {
        name: {"position": [float(xy[0]), float(xy[1])]}
        for name, xy in scene_profile.object_positions_xy.items()
    }
    metadata = build_task_metadata(plan.resolved_task_id, task_version=plan.task_version)
    protocol = {
        "module_name": "registry_manual_smoke",
        "query_text": plan.instruction,
        "task_id": plan.resolved_task_id,
        "task_version": plan.task_version,
        "scene_id": plan.scene_id,
        "resolved_target_object": plan.target_object_name,
        "initial_order": list(plan.initial_order),
        "planned_order_after_interventions": list(plan.planned_order_after_interventions),
        "goal_xy": list(plan.goal_xy) if plan.goal_xy is not None else [],
        "interventions": [
            {
                "object_name": step.object_name,
                "target_xy": [float(step.target_xy[0]), float(step.target_xy[1])],
                "reason": step.reason,
            }
            for step in plan.interventions
        ],
    }
    return {
        "task_name": plan.resolved_task_id,
        "instruction": plan.instruction,
        "goal_sequence": _goal_sequence_for_plan(plan),
        "object_positions": object_positions,
        "protocol": protocol,
        "task_metadata": metadata,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(inner) for inner in value]
    return value


def plan_to_summary(plan: TaskPlan) -> dict[str, Any]:
    payload = asdict(plan)
    return _json_ready(payload)


def print_plan(plan: TaskPlan) -> None:
    print("=" * 72, flush=True)
    print("New Registry Task Plan", flush=True)
    print("=" * 72, flush=True)
    print(f"requested_task_id: {plan.requested_task_id}", flush=True)
    print(f"resolved_task_id:  {plan.resolved_task_id}", flush=True)
    print(f"task_version:      {plan.task_version}", flush=True)
    print(f"scene_id:          {plan.scene_id}", flush=True)
    print(f"runner_family:     {plan.runner_family}", flush=True)
    print(f"family:            {plan.family}", flush=True)
    print(f"skill_type:        {plan.skill_type}", flush=True)
    print(f"supported:         {plan.supported}", flush=True)
    if not plan.supported:
        print(f"unsupported_reason:{plan.unsupported_reason}", flush=True)
    print(f"instruction:       {plan.instruction}", flush=True)
    if plan.target_object_name:
        print(f"target_object:     {plan.target_object_name}", flush=True)
    if plan.goal_xy is not None:
        print(f"goal_xy:           {list(plan.goal_xy)}", flush=True)
    print(f"initial_order:     {list(plan.initial_order)}", flush=True)
    print(f"planned_order:     {list(plan.planned_order_after_interventions)}", flush=True)
    if plan.interventions:
        print("interventions:", flush=True)
        for step in plan.interventions:
            print(
                f"  - {step.object_name} -> {[float(step.target_xy[0]), float(step.target_xy[1])]} :: {step.reason}",
                flush=True,
            )


def write_summary_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_base_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Metadata-first scene-10 smoke runner for the new canonical GeoMind tasks. "
            "This script does not use the legacy debug_protocol_* entrypoints."
        ),
        conflict_handler="resolve",
    )
    parser.add_argument("task_ids", nargs="*", default=list(DEFAULT_TASK_IDS), help="Canonical task_id list to test.")
    parser.add_argument("--scene-id", type=int, default=DEFAULT_SCENE_ID, help="Scene id to load. Default: 10.")
    parser.add_argument(
        "--task-version",
        type=str,
        default=TASK_VERSION_BASE,
        help="Task version to test. Default: base.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False, help="Only build plans; do not launch IsaacLab.")
    parser.add_argument(
        "--summary-json-out",
        type=str,
        default=None,
        help="Optional path to write the aggregated summary JSON.",
    )
    parser.add_argument(
        "--disable-fabric",
        action="store_true",
        default=False,
        help="Disable fabric in the IsaacLab runtime.",
    )
    parser.add_argument(
        "--reset-settle-steps",
        type=int,
        default=DEFAULT_RESET_SETTLE_STEPS,
        help="Zero-action settle frames right after env.reset().",
    )
    parser.add_argument(
        "--intervention-settle-steps",
        type=int,
        default=DEFAULT_INTERVENTION_SETTLE_STEPS,
        help="Hold frames after each staged intervention place.",
    )
    parser.add_argument(
        "--target-settle-steps",
        type=int,
        default=DEFAULT_TARGET_SETTLE_STEPS,
        help="Hold frames after the final target placement.",
    )
    parser.add_argument(
        "--viewer-eye",
        type=float,
        nargs=3,
        default=(2.2, 2.2, 1.8),
        metavar=("X", "Y", "Z"),
        help="Viewer eye position for non-headless runs.",
    )
    parser.add_argument(
        "--viewer-lookat",
        type=float,
        nargs=3,
        default=(0.3, 0.0, 0.8),
        metavar=("X", "Y", "Z"),
        help="Viewer look-at target for non-headless runs.",
    )
    return parser


def _extract_passthrough_flag(unknown_args: list[str], flag_name: str) -> bool:
    return any(str(item).strip() == flag_name for item in unknown_args)


def _extract_passthrough_value(unknown_args: list[str], flag_name: str, default: str) -> str:
    for index, item in enumerate(unknown_args):
        token = str(item).strip()
        if token == flag_name and index + 1 < len(unknown_args):
            candidate = str(unknown_args[index + 1]).strip()
            if candidate:
                return candidate
        prefix = f"{flag_name}="
        if token.startswith(prefix):
            candidate = token[len(prefix):].strip()
            if candidate:
                return candidate
    return default


def _strip_passthrough_args(argv: list[str]) -> list[str]:
    boolean_flags = {
        "--headless",
        "--enable_cameras",
        "--xr",
        "--verbose",
        "--info",
    }
    value_flags = {
        "--device",
        "--livestream",
        "--experience",
        "--kit_args",
    }
    filtered: list[str] = []
    index = 0
    while index < len(argv):
        token = str(argv[index]).strip()
        if token in boolean_flags:
            index += 1
            continue
        matched_value_flag = False
        for flag_name in value_flags:
            if token == flag_name:
                index += 2
                matched_value_flag = True
                break
            if token.startswith(f"{flag_name}="):
                index += 1
                matched_value_flag = True
                break
        if matched_value_flag:
            continue
        filtered.append(argv[index])
        index += 1
    return filtered


def parse_args() -> tuple[argparse.Namespace, Any]:
    raw_argv = list(sys.argv[1:])
    base_parser = build_base_argparser()
    if "--dry-run" in raw_argv:
        filtered_argv = _strip_passthrough_args(raw_argv)
        args = base_parser.parse_args(filtered_argv)
        setattr(args, "headless", _extract_passthrough_flag(raw_argv, "--headless"))
        setattr(args, "device", _extract_passthrough_value(raw_argv, "--device", "cuda:0"))
        setattr(args, "enable_cameras", _extract_passthrough_flag(raw_argv, "--enable_cameras"))
        return args, None

    from geomind_lab.bootstrap.isaaclab_path import ensure_isaaclab_source_on_path

    isaaclab_source_dir = ensure_isaaclab_source_on_path(REPO_ROOT)
    for package_dir in (
        isaaclab_source_dir / "isaaclab",
        isaaclab_source_dir / "isaaclab_tasks",
        isaaclab_source_dir / "isaaclab_mimic",
        isaaclab_source_dir / "isaaclab_assets",
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

    full_parser = build_base_argparser()
    AppLauncher.add_app_launcher_args(full_parser)
    args = full_parser.parse_args()
    setattr(args, "enable_cameras", True)
    return args, AppLauncher


def should_reset(terminated, truncated) -> bool:
    if terminated is not None and bool(terminated.any()):
        return True
    if truncated is not None and bool(truncated.any()):
        return True
    return False


def print_failure_context(stage: str, *, args_obj=None, current_task: str | None = None) -> None:
    print("\n" + "=" * 72, file=sys.stderr, flush=True)
    print("New Registry Task Smoke Failure", file=sys.stderr, flush=True)
    print("=" * 72, file=sys.stderr, flush=True)
    print(f"stage:          {stage}", file=sys.stderr, flush=True)
    if current_task is not None:
        print(f"task_id:        {current_task}", file=sys.stderr, flush=True)
    if args_obj is not None:
        print(f"scene_id:       {getattr(args_obj, 'scene_id', None)}", file=sys.stderr, flush=True)
        print(f"task_version:   {getattr(args_obj, 'task_version', None)}", file=sys.stderr, flush=True)
        print(f"headless:       {getattr(args_obj, 'headless', None)}", file=sys.stderr, flush=True)
        print(f"device:         {getattr(args_obj, 'device', None)}", file=sys.stderr, flush=True)
    print("-" * 72, file=sys.stderr, flush=True)
    traceback.print_exc()
    print("=" * 72, file=sys.stderr, flush=True)


def run_real_tasks(args, app_launcher_cls, scene_profile: SimpleSceneProfile, plans: list[TaskPlan]) -> list[dict[str, Any]]:
    simulation_app = None
    env = None
    current_task = None
    stage = "app_launcher"

    try:
        app_launcher = app_launcher_cls(args)
        simulation_app = app_launcher.app

        import torch
        import isaaclab.utils.math as PoseUtils
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.utils import configclass

        from geomind_lab.mimic import GeoMindFrankaIKRelMimicEnv, GeoMindFrankaIKRelMimicEnvCfg
        from geomind_lab.tasks.manager_based.manipulation.geomind import mdp as geomind_mdp
        from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.common import get_asset_pose_w

        down_facing_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        gripper_open_command = 1.0
        gripper_close_command = -1.0

        def settle_env_after_reset(env_obj, settle_steps: int) -> None:
            zero_actions = torch.zeros(env_obj.action_space.shape, device=env_obj.device, dtype=torch.float32)
            for _ in range(max(0, int(settle_steps))):
                _, _, terminated, truncated, _ = env_obj.step(zero_actions)
                if should_reset(terminated, truncated):
                    env_obj.reset()

        def build_runtime_env_cfg(plan: TaskPlan):
            base_cfg_cls = GeoMindFrankaIKRelMimicEnvCfg
            manual_task_spec = build_manual_task_spec(scene_profile, plan)

            def __post_init__(self):
                self.scene_id = scene_profile.scene_id
                self.task_layout_key = f"manual_registry_smoke::{plan.resolved_task_id}"
                self.task_index = 0
                self.scene.num_envs = 1
                self.sim.device = args.device
                self.sim.use_fabric = not args.disable_fabric
                self.task_spec = manual_task_spec

                base_cfg_cls.__post_init__(self)

                self.viewer.eye = tuple(float(v) for v in args.viewer_eye)
                self.viewer.lookat = tuple(float(v) for v in args.viewer_lookat)

            runtime_cfg_cls = configclass(
                type(
                    "RuntimeGeoMindNewRegistryTaskCfg",
                    (base_cfg_cls,),
                    {
                        "__doc__": "Runtime GeoMind mimic cfg for manual new-registry task smoke tests.",
                        "scene_id": scene_profile.scene_id,
                        "task_layout_key": f"manual_registry_smoke::{plan.resolved_task_id}",
                        "task_index": 0,
                        "__post_init__": __post_init__,
                    },
                )
            )
            return runtime_cfg_cls()

        def _eef_name(env_obj) -> str:
            return env_obj._eef_name()

        def _pose_from_position_and_quat(position: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
            rotation = PoseUtils.matrix_from_quat(quat.unsqueeze(0))[0]
            return PoseUtils.make_pose(position, rotation)

        def _action_from_target_pose(env_obj, target_pose: torch.Tensor, gripper_action: float, *, env_id: int = 0):
            eef_name = _eef_name(env_obj)
            action = env_obj.target_eef_pose_to_action(
                target_eef_pose_dict={eef_name: target_pose},
                gripper_action_dict={
                    eef_name: torch.tensor([gripper_action], device=env_obj.device, dtype=torch.float32),
                },
                env_id=env_id,
            )
            if action.dim() == 1:
                action = action.unsqueeze(0)
            return action

        def _step_with_target_pose(env_obj, target_pose: torch.Tensor, gripper_action: float, *, steps: int) -> bool:
            for _ in range(max(1, int(steps))):
                action = _action_from_target_pose(env_obj, target_pose, gripper_action)
                _, _, terminated, truncated, _ = env_obj.step(action)
                if should_reset(terminated, truncated):
                    return False
            return True

        def _current_object_position(env_obj, object_name: str, *, env_id: int = 0) -> torch.Tensor:
            pos_w, _ = get_asset_pose_w(env_obj, object_name)
            return pos_w[env_id, :3].clone().detach()

        def _current_object_xy(env_obj, object_name: str, *, env_id: int = 0) -> tuple[float, float]:
            position = _current_object_position(env_obj, object_name, env_id=env_id)
            return (float(position[0].item()), float(position[1].item()))

        def _current_order(env_obj) -> tuple[str, ...]:
            return tuple(
                sorted(
                    scene_profile.object_names,
                    key=lambda name: (
                        float(_current_object_xy(env_obj, name)[0]),
                        float(_current_object_xy(env_obj, name)[1]),
                        str(name),
                    ),
                )
            )

        def _current_eef_position(env_obj, *, env_id: int = 0) -> torch.Tensor:
            pose = env_obj.get_robot_eef_pose(_eef_name(env_obj), env_ids=[env_id])[0]
            position, _ = PoseUtils.unmake_pose(pose)
            return position.clone().detach()

        def _object_height(env_obj, object_name: str) -> float:
            resolver = getattr(env_obj.cfg, "_resolve_object_runtime_height", None)
            if callable(resolver):
                resolved = resolver(object_name)
                if resolved is not None:
                    return float(resolved)
            return 0.07

        def _object_grasped(env_obj, object_name: str, *, env_id: int = 0) -> bool:
            grasped = geomind_mdp.object_grasped(
                env_obj,
                robot_cfg=SceneEntityCfg("robot"),
                ee_frame_cfg=SceneEntityCfg("ee_frame"),
                object_cfg=SceneEntityCfg(object_name),
            )
            return bool(grasped[env_id].item())

        def run_scripted_grasp_for_target(env_obj, target_name: str) -> dict[str, Any]:
            target_position = _current_object_position(env_obj, target_name)
            object_height = _object_height(env_obj, target_name)
            target_surface_z = float(target_position[2].item())
            approach_z = float(target_surface_z + max(0.12, 1.5 * object_height))
            grasp_z = float(target_surface_z + max(0.0, 0.05 * object_height))
            lift_z = float(target_surface_z + max(0.18, 2.0 * object_height))

            approach_pose = _pose_from_position_and_quat(
                torch.tensor(
                    [target_position[0].item(), target_position[1].item(), approach_z],
                    device=env_obj.device,
                    dtype=torch.float32,
                ),
                down_facing_quat.to(device=env_obj.device),
            )
            grasp_pose = _pose_from_position_and_quat(
                torch.tensor(
                    [target_position[0].item(), target_position[1].item(), grasp_z],
                    device=env_obj.device,
                    dtype=torch.float32,
                ),
                down_facing_quat.to(device=env_obj.device),
            )
            lift_pose = _pose_from_position_and_quat(
                torch.tensor(
                    [target_position[0].item(), target_position[1].item(), lift_z],
                    device=env_obj.device,
                    dtype=torch.float32,
                ),
                down_facing_quat.to(device=env_obj.device),
            )

            completed = (
                _step_with_target_pose(env_obj, approach_pose, gripper_open_command, steps=30)
                and _step_with_target_pose(env_obj, grasp_pose, gripper_open_command, steps=20)
                and _step_with_target_pose(env_obj, grasp_pose, gripper_close_command, steps=28)
                and _step_with_target_pose(env_obj, lift_pose, gripper_close_command, steps=24)
                and _step_with_target_pose(env_obj, lift_pose, gripper_close_command, steps=8)
            )
            final_position = _current_object_position(env_obj, target_name)
            grasp_sensor_triggered = completed and _object_grasped(env_obj, target_name)
            object_lifted = float(final_position[2].item()) > float(target_surface_z + max(0.08, 1.2 * object_height))
            grasp_success = completed and (grasp_sensor_triggered or object_lifted)
            eef_position = _current_eef_position(env_obj)
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
                "grasp_success": bool(grasp_success),
                "grasp_sensor_triggered": bool(grasp_sensor_triggered),
                "object_lifted": bool(object_lifted),
                "grasp_offset_xy": [float(grasp_offset_xy[0]), float(grasp_offset_xy[1])],
                "final_target_position": [float(value) for value in final_position.tolist()],
            }

        def run_scripted_place_for_target(
            env_obj,
            *,
            target_name: str,
            target_pos_xy: tuple[float, float],
            target_surface_z: float,
            object_height: float,
            post_place_settle_steps: int,
            grasp_offset_xy: tuple[float, float] = (0.0, 0.0),
        ) -> dict[str, Any]:
            target_x, target_y = (float(target_pos_xy[0]), float(target_pos_xy[1]))
            command_x = float(target_x - float(grasp_offset_xy[0]))
            command_y = float(target_y - float(grasp_offset_xy[1]))
            approach_z = float(target_surface_z + max(0.14, 1.5 * object_height))
            place_z = float(target_surface_z + max(0.02, 0.1 * object_height))
            retreat_z = float(target_surface_z + max(0.18, 2.0 * object_height))

            approach_pose = _pose_from_position_and_quat(
                torch.tensor([command_x, command_y, approach_z], device=env_obj.device, dtype=torch.float32),
                down_facing_quat.to(device=env_obj.device),
            )
            place_pose = _pose_from_position_and_quat(
                torch.tensor([command_x, command_y, place_z], device=env_obj.device, dtype=torch.float32),
                down_facing_quat.to(device=env_obj.device),
            )
            retreat_pose = _pose_from_position_and_quat(
                torch.tensor([command_x, command_y, retreat_z], device=env_obj.device, dtype=torch.float32),
                down_facing_quat.to(device=env_obj.device),
            )

            completed = (
                _step_with_target_pose(env_obj, approach_pose, gripper_close_command, steps=24)
                and _step_with_target_pose(env_obj, place_pose, gripper_close_command, steps=18)
                and _step_with_target_pose(env_obj, place_pose, gripper_open_command, steps=18)
                and _step_with_target_pose(env_obj, retreat_pose, gripper_open_command, steps=18)
            )
            settle_applied = False
            if completed and int(post_place_settle_steps) > 0:
                settle_applied = _step_with_target_pose(
                    env_obj,
                    retreat_pose,
                    gripper_open_command,
                    steps=int(post_place_settle_steps),
                )
                completed = completed and settle_applied

            final_position = _current_object_position(env_obj, target_name)
            release_sensor_triggered = completed and (not _object_grasped(env_obj, target_name))
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

        task_results: list[dict[str, Any]] = []
        for plan in plans:
            current_task = plan.resolved_task_id
            if not plan.supported:
                task_results.append({"task_id": plan.resolved_task_id, "supported": False, "reason": plan.unsupported_reason})
                continue

            stage = f"build_env_cfg::{plan.resolved_task_id}"
            env_cfg = build_runtime_env_cfg(plan)
            env = GeoMindFrankaIKRelMimicEnv(cfg=env_cfg)
            print_plan(plan)
            print(f"layout_key:         {env_cfg.task_layout_key}", flush=True)
            print(f"scene_usd:          {scene_profile.usd_path}", flush=True)

            stage = f"run_task::{plan.resolved_task_id}"
            with torch.inference_mode():
                env.reset()
                settle_env_after_reset(env, args.reset_settle_steps)

                intervention_results: list[dict[str, Any]] = []
                for step in plan.interventions:
                    grasp_summary = run_scripted_grasp_for_target(env, step.object_name)
                    if bool(grasp_summary["grasp_success"]):
                        place_summary = run_scripted_place_for_target(
                            env,
                            target_name=step.object_name,
                            target_pos_xy=step.target_xy,
                            target_surface_z=float(grasp_summary["target_surface_z"]),
                            object_height=float(grasp_summary["target_height"]),
                            post_place_settle_steps=args.intervention_settle_steps,
                            grasp_offset_xy=tuple(float(value) for value in grasp_summary["grasp_offset_xy"]),
                        )
                    else:
                        place_summary = {
                            "target_name": step.object_name,
                            "target_pos_xy": [float(step.target_xy[0]), float(step.target_xy[1])],
                            "execution_completed": False,
                            "release_detected": False,
                            "pred_pos_xy": list(_current_object_xy(env, step.object_name)),
                        }
                    achieved_xy = tuple(float(value) for value in place_summary.get("pred_pos_xy", (0.0, 0.0)))
                    l2_error = (
                        (achieved_xy[0] - float(step.target_xy[0])) ** 2 + (achieved_xy[1] - float(step.target_xy[1])) ** 2
                    ) ** 0.5
                    intervention_results.append(
                        {
                            "object_name": step.object_name,
                            "reason": step.reason,
                            "planned_target_xy": [float(step.target_xy[0]), float(step.target_xy[1])],
                            "achieved_target_xy": [float(achieved_xy[0]), float(achieved_xy[1])],
                            "l2_error": float(l2_error),
                            "success": bool(grasp_summary["grasp_success"]) and bool(place_summary["release_detected"]),
                            "grasp": grasp_summary,
                            "place": place_summary,
                        }
                    )

                runtime_order_after_interventions = list(_current_order(env))
                planned_order_matches_runtime = runtime_order_after_interventions == list(plan.planned_order_after_interventions)

                target_grasp = run_scripted_grasp_for_target(env, str(plan.target_object_name))
                if bool(target_grasp["grasp_success"]) and plan.goal_xy is not None:
                    target_place = run_scripted_place_for_target(
                        env,
                        target_name=str(plan.target_object_name),
                        target_pos_xy=plan.goal_xy,
                        target_surface_z=float(target_grasp["target_surface_z"]),
                        object_height=float(target_grasp["target_height"]),
                        post_place_settle_steps=args.target_settle_steps,
                        grasp_offset_xy=tuple(float(value) for value in target_grasp["grasp_offset_xy"]),
                    )
                else:
                    target_place = {
                        "target_name": str(plan.target_object_name),
                        "target_pos_xy": list(plan.goal_xy) if plan.goal_xy is not None else [],
                        "execution_completed": False,
                        "release_detected": False,
                        "pred_pos_xy": list(_current_object_xy(env, str(plan.target_object_name))),
                    }

                final_xy = tuple(float(value) for value in target_place.get("pred_pos_xy", (0.0, 0.0)))
                goal_xy = plan.goal_xy if plan.goal_xy is not None else (0.0, 0.0)
                goal_l2_error = ((final_xy[0] - goal_xy[0]) ** 2 + (final_xy[1] - goal_xy[1]) ** 2) ** 0.5
                final_success = bool(target_grasp["grasp_success"]) and bool(target_place["release_detected"]) and goal_l2_error <= float(plan.goal_tolerance)

                result = {
                    "task_id": plan.resolved_task_id,
                    "task_version": plan.task_version,
                    "scene_id": scene_profile.scene_id,
                    "supported": True,
                    "instruction": plan.instruction,
                    "initial_order": list(plan.initial_order),
                    "planned_order_after_interventions": list(plan.planned_order_after_interventions),
                    "runtime_order_after_interventions": runtime_order_after_interventions,
                    "planned_order_matches_runtime": bool(planned_order_matches_runtime),
                    "target_object_name": plan.target_object_name,
                    "goal_xy": list(plan.goal_xy) if plan.goal_xy is not None else [],
                    "goal_tolerance": float(plan.goal_tolerance),
                    "interventions": intervention_results,
                    "target_grasp": target_grasp,
                    "target_place": target_place,
                    "goal_l2_error": float(goal_l2_error),
                    "success": bool(final_success),
                }
                task_results.append(result)
                print(json.dumps(_json_ready(result), indent=2, sort_keys=True, ensure_ascii=False), flush=True)

            env.close()
            env = None

        return task_results
    except Exception:
        print_failure_context(stage, args_obj=args, current_task=current_task)
        raise
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


def main() -> int:
    args, app_launcher_cls = parse_args()
    resolved_scene_profile = resolve_scene_profile(int(args.scene_id))

    plans: list[TaskPlan] = []
    for requested_task_id in tuple(args.task_ids) or DEFAULT_TASK_IDS:
        normalized_requested = _as_str(requested_task_id)
        if not normalized_requested:
            continue
        try:
            plan = build_task_plan(
                scene_profile=resolved_scene_profile,
                requested_task_id=normalized_requested,
                task_version=args.task_version,
            )
        except Exception as exc:
            resolved_task = resolve_task_id(normalized_requested) or normalized_requested
            if resolved_task and tuple(supported_task_versions(resolved_task)):
                try:
                    metadata = build_task_metadata(resolved_task, task_version=args.task_version)
                    normalized_version = resolve_task_version(resolved_task, explicit_task_version=args.task_version)
                except Exception:
                    metadata = {}
                    normalized_version = _as_str(args.task_version)
            else:
                metadata = {}
                normalized_version = _as_str(args.task_version)
            plan = _unsupported_plan(
                requested_task_id=normalized_requested,
                resolved_task_id=resolved_task,
                task_version=normalized_version,
                scene_id=resolved_scene_profile.scene_id,
                metadata=metadata,
                reason=str(exc),
                initial_order=sort_objects_by_xy(resolved_scene_profile),
            )
        plans.append(plan)

    aggregate_summary = {
        "scene_id": resolved_scene_profile.scene_id,
        "scene_usd": resolved_scene_profile.usd_path,
        "dry_run": bool(args.dry_run),
        "task_version": _as_str(args.task_version),
        "scene_objects": list(resolved_scene_profile.object_names),
        "basket_name": resolved_scene_profile.basket_name,
        "basket_xy": list(resolved_scene_profile.basket_xy) if resolved_scene_profile.basket_xy is not None else [],
        "plans": [plan_to_summary(plan) for plan in plans],
        "results": [],
    }

    for plan in plans:
        print_plan(plan)

    supported_plans = [plan for plan in plans if plan.supported]
    if args.dry_run:
        aggregate_summary["results"] = [plan_to_summary(plan) for plan in plans]
    elif supported_plans:
        aggregate_summary["results"] = run_real_tasks(args, app_launcher_cls, resolved_scene_profile, supported_plans)

    if args.summary_json_out:
        write_summary_json(args.summary_json_out, aggregate_summary)
        print(f"summary_json={args.summary_json_out}", flush=True)

    if not supported_plans:
        print("No supported scene-10 task plan was resolved.", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
