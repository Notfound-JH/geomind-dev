#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.utils import asset_utils, scene_utils
from geomind_lab.mimic.envs.constraint_selection_logic import resolve_visual_spatial_constraint_problem


SCENE_CATEGORIES = ("function", "object", "container", "decoration")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_annotated_tasks_path() -> Path:
    candidates = (
        repo_root() / "assets" / "annotations" / "annotated_tasks.json",
        repo_root() / "assets" / "tasks" / "annotated_tasks.json",
        repo_root() / "assets" / "tasks" / "annotated_tasks_1.json",
        repo_root() / "annotated_tasks_1.json",
        Path("/home/elm/projects/geomind/annotated_tasks_1.json"),
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def resolve_scene_usd_path(scene_id: int, assets_root: Path) -> tuple[Path, str]:
    stage_dir = assets_root / "stage" / str(scene_id)
    new_path = stage_dir / f"{scene_id}_new.usda"
    if new_path.is_file():
        return new_path, new_path.name
    legacy_path = stage_dir / f"{scene_id}.usda"
    return legacy_path, legacy_path.name


def load_annotated_tasks(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_prim_nodes(nodes: list[scene_utils.PrimNode]):
    for node in nodes:
        yield node
        yield from _walk_prim_nodes(node.children)


def _collect_category_children(xform_tree: list[dict[str, Any]]) -> dict[str, list[str]]:
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

    return {
        category: [str(child.get("name")) for child in category_nodes.get(category, {}).get("children", [])]
        for category in SCENE_CATEGORIES
    }


def _first_with_prefix(names: list[str], prefix: str) -> str | None:
    for name in names:
        if name.startswith(prefix):
            return name
    return None


def inspect_scene(scene_path: Path) -> dict[str, Any]:
    nodes = scene_utils.parse_usda_scene(scene_path)
    xform_tree = scene_utils.extract_xform_tree(nodes)
    categories = _collect_category_children(xform_tree)
    all_prim_names = sorted({node.name for node in _walk_prim_nodes(nodes)})
    wall_like = [name for name in all_prim_names if "wall" in name.lower()]

    function_names = categories["function"]
    container_names = categories["container"]
    decoration_names = categories["decoration"]

    return {
        "categories": categories,
        "all_prim_names": all_prim_names,
        "wall_like_prims": wall_like,
        "table_name": _first_with_prefix(function_names, "table"),
        "ground_name": _first_with_prefix(function_names, "wood_ground"),
        "basket_name": _first_with_prefix(container_names, "basket"),
        "lamp_name": _first_with_prefix(decoration_names, "lamp"),
        "chair_name": _first_with_prefix(decoration_names, "chair"),
        "blanket_name": _first_with_prefix(decoration_names, "blanket"),
    }


def find_layouts_for_scene(scene_id: int, payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    matches: list[tuple[str, dict[str, Any]]] = []
    for layout_key, entry in payload.items():
        if isinstance(entry, dict) and entry.get("scene_id") == scene_id:
            matches.append((layout_key, entry))
    return matches


def _sort_by_suffix(names: list[str]) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        match = re.search(r"_(\d+)$", name)
        suffix = int(match.group(1)) if match else 0
        return (suffix, name)

    return sorted(names, key=key)


def build_object_mapping(annotated_names: list[str], scene_names: list[str]) -> dict[str, str]:
    if not annotated_names or not scene_names:
        return {}
    ordered_annotated = _sort_by_suffix(annotated_names)
    ordered_scene = _sort_by_suffix(scene_names)
    return {src: dst for src, dst in zip(ordered_annotated, ordered_scene)}


def summarize_container(scene_structure: dict[str, Any]) -> str:
    container = scene_structure.get("container", {})
    shelf = scene_structure.get("shelf", {})
    basket = int(container.get("basket", 0)) if isinstance(container, dict) else 0
    plate = int(container.get("plate", 0)) if isinstance(container, dict) else 0
    bowl = int(container.get("bowl", 0)) if isinstance(container, dict) else 0
    shelf_count = int(shelf.get("shelf", 0)) if isinstance(shelf, dict) else 0
    return f"basket={basket}, plate={plate}, bowl={bowl}, shelf={shelf_count}"


def summarize_goals(goal_sequence: Any, object_mapping: dict[str, str]) -> list[str]:
    lines: list[str] = []
    if not isinstance(goal_sequence, list):
        return lines

    for step in goal_sequence:
        if not isinstance(step, dict):
            continue
        step_id = step.get("step", "?")
        goals = step.get("goals", [])
        if not isinstance(goals, list):
            continue

        goal_parts: list[str] = []
        for goal in goals:
            if not isinstance(goal, dict):
                continue
            raw_object_id = str(goal.get("object_id", ""))
            resolved_object_id = object_mapping.get(raw_object_id, raw_object_id)
            xy = goal.get("xy")
            xy_text = ""
            if isinstance(xy, list) and len(xy) >= 2:
                xy_text = f" @ ({xy[0]}, {xy[1]})"
            if raw_object_id and resolved_object_id and raw_object_id != resolved_object_id:
                goal_parts.append(f"{raw_object_id}->{resolved_object_id}{xy_text}")
            elif resolved_object_id:
                goal_parts.append(f"{resolved_object_id}{xy_text}")
        if goal_parts:
            lines.append(f"step {step_id}: " + ", ".join(goal_parts))
    return lines


def _runtime_object_pose_map(annotation: dict[str, Any]) -> dict[str, list[float]]:
    scene_adaptation = annotation.get("scene_adaptation", {})
    if not isinstance(scene_adaptation, dict):
        return {}
    runtime_objects = scene_adaptation.get("runtime_objects", [])
    if not isinstance(runtime_objects, list):
        return {}

    pose_map: dict[str, list[float]] = {}
    for runtime_object in runtime_objects:
        if not isinstance(runtime_object, dict):
            continue
        object_name = str(runtime_object.get("name", "")).strip()
        pose = runtime_object.get("pose", {})
        xy = pose.get("xy", []) if isinstance(pose, dict) else []
        if object_name and isinstance(xy, list) and len(xy) >= 2:
            pose_map[object_name] = [float(xy[0]), float(xy[1])]
    return pose_map


def _runtime_object_spec_map(annotation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scene_adaptation = annotation.get("scene_adaptation", {})
    if not isinstance(scene_adaptation, dict):
        return {}
    runtime_objects = scene_adaptation.get("runtime_objects", [])
    if not isinstance(runtime_objects, list):
        return {}

    return {
        str(runtime_object.get("name", "")).strip(): runtime_object
        for runtime_object in runtime_objects
        if isinstance(runtime_object, dict) and str(runtime_object.get("name", "")).strip()
    }


def _runtime_object_dimensions(annotation: dict[str, Any], object_name: str) -> list[float] | None:
    runtime_object = _runtime_object_spec_map(annotation).get(str(object_name).strip())
    if not isinstance(runtime_object, dict):
        return None

    for key in ("size", "dimensions"):
        value = runtime_object.get(key)
        if isinstance(value, list) and len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]

    radius = runtime_object.get("radius")
    height = runtime_object.get("height")
    if radius is not None and height is not None:
        return [2.0 * float(radius), 2.0 * float(radius), float(height)]
    return None


def _resolve_support_top_center(annotation: dict[str, Any], support_object_id: str) -> dict[str, Any]:
    support_xy = _runtime_object_pose_map(annotation).get(support_object_id)
    resolved = {"support_object_id": support_object_id}
    if isinstance(support_xy, list) and len(support_xy) >= 2:
        resolved["xy"] = [float(support_xy[0]), float(support_xy[1])]

    support_dims = _runtime_object_dimensions(annotation, support_object_id)
    if isinstance(support_dims, list) and len(support_dims) >= 3:
        resolved["support_size"] = [float(value) for value in support_dims[:3]]
        resolved["top_z_offset"] = float(support_dims[2]) * 0.5
    return resolved


def _resolve_containment_center(annotation: dict[str, Any], container_reference_object_id: str) -> dict[str, Any]:
    reference_xy = _runtime_object_pose_map(annotation).get(container_reference_object_id)
    resolved = {"container_reference_object_id": container_reference_object_id}
    if isinstance(reference_xy, list) and len(reference_xy) >= 2:
        resolved["xy"] = [float(reference_xy[0]), float(reference_xy[1])]

    protocol = annotation.get("protocol", {})
    gt_spec = protocol.get("gt_spec", {}) if isinstance(protocol, dict) else {}
    if isinstance(gt_spec, dict):
        interior_half_extents_xy = gt_spec.get("interior_half_extents_xy")
        if isinstance(interior_half_extents_xy, list) and len(interior_half_extents_xy) >= 2:
            resolved["interior_half_extents_xy"] = [
                float(interior_half_extents_xy[0]),
                float(interior_half_extents_xy[1]),
            ]
        opening_plane_z_offset = gt_spec.get("opening_plane_z_offset")
        if opening_plane_z_offset is not None:
            resolved["opening_plane_z_offset"] = float(opening_plane_z_offset)
    return resolved


def _resolve_preview_proxy_goal(annotation: dict[str, Any]) -> dict[str, Any]:
    scene_adaptation = annotation.get("scene_adaptation", {})
    if not isinstance(scene_adaptation, dict):
        return {}
    preview_proxy_goal = scene_adaptation.get("preview_proxy_goal", {})
    if not isinstance(preview_proxy_goal, dict) or not preview_proxy_goal:
        return {}

    resolved = dict(preview_proxy_goal)
    xy = resolved.get("xy")
    if isinstance(xy, list) and len(xy) >= 2:
        resolved["xy"] = [float(xy[0]), float(xy[1])]
        return resolved

    resolved_type = str(resolved.get("type", "")).strip()
    if resolved_type == "support_top_center":
        support_object_id = str(resolved.get("support_object_id", "")).strip()
        if not support_object_id:
            protocol = annotation.get("protocol", {})
            if isinstance(protocol, dict):
                gt_spec = protocol.get("gt_spec", {})
                if isinstance(gt_spec, dict):
                    support_object_id = str(gt_spec.get("support_object_id", "")).strip()
        if not support_object_id:
            return resolved
        derived = _resolve_support_top_center(annotation, support_object_id)
        resolved.update(derived)
        resolved["derived_from"] = "protocol.gt_spec.support_object_id"
        return resolved

    if resolved_type == "containment_center":
        container_reference_object_id = str(resolved.get("container_reference_object_id", "")).strip()
        if not container_reference_object_id:
            protocol = annotation.get("protocol", {})
            if isinstance(protocol, dict):
                gt_spec = protocol.get("gt_spec", {})
                if isinstance(gt_spec, dict):
                    container_reference_object_id = str(gt_spec.get("container_reference_object_id", "")).strip()
        if not container_reference_object_id:
            return resolved
        derived = _resolve_containment_center(annotation, container_reference_object_id)
        resolved.update(derived)
        resolved["derived_from"] = "protocol.gt_spec.container_reference_object_id"
        return resolved

    if resolved_type != "zone_center_xy":
        return resolved

    protocol = annotation.get("protocol", {})
    if not isinstance(protocol, dict):
        return resolved
    gt_spec = protocol.get("gt_spec", {})
    if not isinstance(gt_spec, dict):
        return resolved
    target_zone = gt_spec.get("target_zone", {})
    if not isinstance(target_zone, dict):
        return resolved
    center_offset = target_zone.get("center_offset_from_anchor", [])
    if not isinstance(center_offset, list) or len(center_offset) < 2:
        return resolved

    anchor_object_id = str(resolved.get("anchor_object_id") or gt_spec.get("anchor_object_id", "")).strip()
    if not anchor_object_id:
        return resolved

    anchor_xy = _runtime_object_pose_map(annotation).get(anchor_object_id)
    if not isinstance(anchor_xy, list) or len(anchor_xy) < 2:
        return resolved

    resolved["xy"] = [
        float(anchor_xy[0]) + float(center_offset[0]),
        float(anchor_xy[1]) + float(center_offset[1]),
    ]
    resolved["derived_from"] = "protocol.gt_spec.target_zone.center_offset_from_anchor"
    return resolved


def summarize_preview_proxy_goal(annotation: dict[str, Any], object_mapping: dict[str, str]) -> str | None:
    resolved = _resolve_preview_proxy_goal(annotation)
    if not resolved:
        return None

    raw_object_id = str(resolved.get("object_id", "")).strip()
    if not raw_object_id:
        return None
    resolved_object_id = object_mapping.get(raw_object_id, raw_object_id)
    support_object_id = str(resolved.get("support_object_id", "")).strip()
    resolved_support_object_id = object_mapping.get(support_object_id, support_object_id)
    xy = resolved.get("xy")
    if not isinstance(xy, list) or len(xy) < 2:
        if resolved_support_object_id:
            return f"{resolved_object_id} on {resolved_support_object_id} preview_proxy"
        return f"{resolved_object_id} preview_proxy"

    tolerance = resolved.get("tolerance", {})
    pos_r = tolerance.get("pos_r") if isinstance(tolerance, dict) else None
    z_abs = tolerance.get("z_abs") if isinstance(tolerance, dict) else None
    tolerance_parts = []
    if pos_r is not None:
        tolerance_parts.append(f"r={pos_r}")
    if z_abs is not None:
        tolerance_parts.append(f"z={z_abs}")
    tolerance_text = f", {', '.join(tolerance_parts)}" if tolerance_parts else ""
    support_size = resolved.get("support_size")
    support_size_text = ""
    if isinstance(support_size, list) and len(support_size) >= 2:
        support_size_text = f", support_size={support_size[:2]}"
    interior_half_extents_xy = resolved.get("interior_half_extents_xy")
    containment_size_text = ""
    if isinstance(interior_half_extents_xy, list) and len(interior_half_extents_xy) >= 2:
        containment_size_text = f", cavity_half_extents={interior_half_extents_xy[:2]}"
    if resolved_support_object_id:
        return (
            f"{resolved_object_id} on {resolved_support_object_id} @ ({xy[0]}, {xy[1]})"
            f"{support_size_text}{tolerance_text}"
        )
    return f"{resolved_object_id} @ ({xy[0]}, {xy[1]}){containment_size_text}{tolerance_text}"


def summarize_gt_spec(annotation: dict[str, Any]) -> str | None:
    protocol = annotation.get("protocol", {})
    if not isinstance(protocol, dict):
        return None
    gt_spec = protocol.get("gt_spec", {})
    if not isinstance(gt_spec, dict):
        return None

    resolver = str(gt_spec.get("resolver", "")).strip()
    if resolver == "directional_region_placement":
        target_zone = gt_spec.get("target_zone", {})
        if not isinstance(target_zone, dict):
            target_zone = {}
        center_offset = target_zone.get("center_offset_from_anchor", [])
        half_extents = target_zone.get("half_extents", [])
        return (
            "resolver=directional_region_placement, "
            f"frame={gt_spec.get('reference_frame')}, "
            f"front={gt_spec.get('primary_direction')} {gt_spec.get('primary_axis')}, "
            f"center_offset={center_offset}, "
            f"half_extents={half_extents}"
        )

    if resolver == "support_top_placement":
        return (
            "resolver=support_top_placement, "
            f"frame={gt_spec.get('reference_frame')}, "
            f"support={gt_spec.get('support_object_id')}, "
            f"target={gt_spec.get('target_object_id')}, "
            f"contact={gt_spec.get('contact_required')}, "
            f"support_required={gt_spec.get('support_required')}, "
            f"settle_steps={gt_spec.get('stable_after_settle_steps')}, "
            f"max_lin={gt_spec.get('max_linear_speed_after_settle')}, "
            f"max_ang={gt_spec.get('max_angular_speed_after_settle')}"
        )

    if resolver == "containment_placement":
        return (
            "resolver=containment_placement, "
            f"frame={gt_spec.get('reference_frame')}, "
            f"container_ref={gt_spec.get('container_reference_object_id')}, "
            f"target={gt_spec.get('target_object_id')}, "
            f"inner_half_extents={gt_spec.get('interior_half_extents_xy')}, "
            f"opening_z_offset={gt_spec.get('opening_plane_z_offset')}, "
            f"floor_gap={gt_spec.get('max_floor_gap')}"
        )

    if resolver == "visual_spatial_constraint_selection":
        anchor_object = protocol.get("anchor_object", {})
        anchor_object_id = anchor_object.get("object_id") if isinstance(anchor_object, dict) else None
        return (
            "resolver=visual_spatial_constraint_selection, "
            f"frame={gt_spec.get('reference_frame')}, "
            f"anchor={anchor_object_id}, "
            f"axis={gt_spec.get('axis')}, "
            f"direction={gt_spec.get('direction')}, "
            f"min_offset={gt_spec.get('min_primary_offset')}, "
            f"max_secondary={gt_spec.get('max_abs_secondary_offset')}, "
            f"filters={gt_spec.get('target_filters')}"
        )

    if resolver:
        return f"resolver={resolver}"
    return None


def summarize_constraint_selection_resolution(annotation: dict[str, Any]) -> str | None:
    protocol = annotation.get("protocol", {})
    if not isinstance(protocol, dict):
        return None
    gt_spec = protocol.get("gt_spec", {})
    if not isinstance(gt_spec, dict):
        return None
    if str(gt_spec.get("resolver", "")).strip() != "visual_spatial_constraint_selection":
        return None

    runtime_object_pose_map = _runtime_object_pose_map(annotation)
    if not runtime_object_pose_map:
        return "runtime_resolution_error=no_runtime_object_pose_map"

    object_positions = {
        object_name: {"position": [float(xy[0]), float(xy[1])]}
        for object_name, xy in runtime_object_pose_map.items()
        if isinstance(xy, list) and len(xy) >= 2
    }

    try:
        problem = resolve_visual_spatial_constraint_problem(protocol, object_positions)
    except Exception as exc:
        return f"runtime_resolution_error={exc}"

    return (
        f"anchor={problem.get('anchor_object_id')}, "
        f"gt={problem.get('gt_object_id')}, "
        f"spatial={problem.get('spatial_match_ids')}, "
        f"attributes={problem.get('attribute_match_ids')}, "
        f"accepted={problem.get('valid_candidate_ids')}"
    )


def print_scene_summary(
    scene_id: int,
    scene_path: Path,
    variant: str,
    annotations_path: Path,
    scene_info: dict[str, Any],
    *,
    layout_key: str | None = None,
) -> None:
    categories = scene_info["categories"]

    print("=" * 72)
    print("GeoMind Scene Inspection")
    print("=" * 72)
    print(f"scene_id:         {scene_id}")
    print(f"scene_usd:        {scene_path}")
    print(f"selected_variant: {variant}")
    if layout_key:
        print(f"selected_layout:  {layout_key}")
    print(f"assets_root:      {asset_utils.resolve_assets_root()}")
    print(f"annotated_tasks:  {annotations_path}")

    print("\nScene Info")
    print(f"  function:       {categories['function']}")
    print(f"  object:         {categories['object']}")
    print(f"  container:      {categories['container']}")
    print(f"  decoration:     {categories['decoration']}")
    print(f"  wall_like:      {scene_info['wall_like_prims']}")
    print(f"  table:          {scene_info['table_name']}")
    print(f"  ground:         {scene_info['ground_name']}")
    print(f"  basket:         {scene_info['basket_name']}")
    print(f"  lamp/chair:     {scene_info['lamp_name']} / {scene_info['chair_name']}")
    print(f"  blanket:        {scene_info['blanket_name']}")


def print_layouts(scene_info: dict[str, Any], matching_layouts: list[tuple[str, dict[str, Any]]]) -> None:
    scene_object_names = scene_info["categories"]["object"]

    print("\nMatching Tasks")
    if not matching_layouts:
        print("  No annotated layout matches this scene_id.")
        return

    for layout_key, entry in matching_layouts:
        scene_structure = entry.get("scene_structure", {})
        tasks = entry.get("tasks", [])
        annotated_positions = scene_structure.get("object_positions", {}) if isinstance(scene_structure, dict) else {}
        annotated_object_names = list(annotated_positions.keys()) if isinstance(annotated_positions, dict) else []
        has_runtime_injection = False
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                annotation = task.get("annotation", {})
                if not isinstance(annotation, dict):
                    continue
                scene_adaptation = annotation.get("scene_adaptation", {})
                runtime_objects = scene_adaptation.get("runtime_objects", []) if isinstance(scene_adaptation, dict) else []
                if isinstance(runtime_objects, list) and runtime_objects:
                    has_runtime_injection = True
                    break
        object_mapping = {} if has_runtime_injection else build_object_mapping(annotated_object_names, scene_object_names)

        print(f"\n  layout_key:     {layout_key}")
        print(f"  layout_type:    {scene_structure.get('layout') if isinstance(scene_structure, dict) else None}")
        print(f"  annotated_objs: {annotated_object_names}")
        print(f"  scene_objects:  {scene_object_names}")
        print(f"  object_mapping: {object_mapping}")
        if isinstance(scene_structure, dict):
            print(f"  object_asset:   {scene_structure.get('object_asset')}")
            print(f"  container_cfg:  {summarize_container(scene_structure)}")
        print(f"  task_count:     {len(tasks) if isinstance(tasks, list) else 0}")

        if not isinstance(tasks, list) or not tasks:
            continue

        for task_index, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            annotation = task.get("annotation", {})
            if not isinstance(annotation, dict):
                annotation = {}

            raw_targets = [str(item) for item in annotation.get("targets", []) if isinstance(item, str)]
            resolved_targets = [object_mapping.get(name, name) for name in raw_targets]
            goal_lines = summarize_goals(annotation.get("goal_sequence", []), object_mapping)
            preview_proxy_line = summarize_preview_proxy_goal(annotation, object_mapping)
            gt_summary = summarize_gt_spec(annotation)
            runtime_resolution = summarize_constraint_selection_resolution(annotation)
            task_id = annotation.get("task_id")

            print(f"    [{task_index}] {task.get('task_category', 'unknown')}")
            print(f"         instruction: {task.get('instruction', '')}")
            print(f"         task_id: {task_id}")
            print(f"         targets: {resolved_targets} (raw: {raw_targets})")
            if gt_summary:
                print(f"         gt_spec: {gt_summary}")
            if runtime_resolution:
                print(f"         resolved_gt: {runtime_resolution}")
            if preview_proxy_line:
                print(f"         preview_proxy: {preview_proxy_line}")
            if goal_lines:
                print(f"         goals: {goal_lines}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a GeoMind scene by scene_id and list the matching annotated tasks."
    )
    parser.add_argument("scene_id", type=int, help="Scene id under <GEOMIND_ASSETS_ROOT>/stage/<scene_id>/")
    parser.add_argument("--layout-key", type=str, default=None, help="Optional layout_key filter within the scene.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    scene_id = args.scene_id
    layout_key_filter = str(args.layout_key).strip() if args.layout_key else ""
    assets_root = Path(asset_utils.resolve_assets_root())
    scene_path, selected_variant = resolve_scene_usd_path(scene_id, assets_root)
    annotations_path = resolve_annotated_tasks_path()
    annotated_tasks = load_annotated_tasks(annotations_path)
    matching_layouts = find_layouts_for_scene(scene_id, annotated_tasks)
    if layout_key_filter:
        matching_layouts = [
            (layout_key, entry)
            for layout_key, entry in matching_layouts
            if str(layout_key) == layout_key_filter
        ]

    scene_info: dict[str, Any] | None = None
    if scene_path.is_file():
        scene_info = inspect_scene(scene_path)
    else:
        print("=" * 72)
        print("GeoMind Scene Inspection")
        print("=" * 72)
        print(f"scene_id:         {scene_id}")
        print(f"scene_usd:        {scene_path}")
        print(f"selected_variant: {selected_variant}")
        print(f"assets_root:      {assets_root}")
        print(f"annotated_tasks:  {annotations_path}")
        print("\nScene Info")
        print("  Scene USDA file does not exist.")

    if scene_info is not None:
        print_scene_summary(
            scene_id,
            scene_path,
            selected_variant,
            annotations_path,
            scene_info,
            layout_key=(layout_key_filter or None),
        )
        print_layouts(scene_info, matching_layouts)
    else:
        print("\nMatching Tasks")
        if not matching_layouts:
            print("  No annotated layout matches this scene_id.")
        else:
            for layout_key, entry in matching_layouts:
                task_count = len(entry.get("tasks", [])) if isinstance(entry.get("tasks"), list) else 0
                print(f"  {layout_key}: {task_count} task(s)")

    if scene_info is None and not matching_layouts:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
