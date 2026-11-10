#!/usr/bin/env python3
"""Regenerate compact v3 VLA registrations and isolated legacy protocol layouts."""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from geomind.core.schema import normalize_task_config, upgrade_task_config
TASKS_DIR = REPO / "geomind" / "configs" / "tasks"
OUT_DIR = REPO / "geomind" / "configs" / "vla_tasks"
LEGACY_SRC = REPO / "geomind" / "configs" / "legacy" / "layouts" / "standard.json"
VLA_LAYOUTS = REPO / "geomind" / "configs" / "legacy" / "layouts" / "vla.json"
ANNOTATED_SRC = REPO / "assets" / "annotations" / "annotated_tasks.json"
VLA_VISIBLE_INTERVENTION_DURATION_S = 0.35
VLA_REINDEX_EXTREME_INTERVENTION_DURATION_S = 0.8
VLA_REINDEX_NEIGHBOR_INTERVENTION_DURATION_S = 2.5
VLA_TRACK_CLUTTER_INTERVENTION_DURATION_S = 1.0
VLA_REINDEX_SUBSET_INTERVENTION_DURATION_S = 1.0
VLA_DYNAMIC_CANONICALS = {
    "track_single",
    "track_multi",
    "track_clutter",
    "reindex_extreme",
    "reindex_insert",
    "reindex_neighbor",
    "reindex_subset",
}


def _deepcopy_json(obj):
    return copy.deepcopy(obj)


def _vla_task_id(task_id: str) -> str:
    """Append Vla to compact task names, e.g. TaskSem_v0 -> TaskSemVla_v0."""
    m = re.match(r"^(.+)_v(\d+)$", task_id)
    if not m:
        return f"{task_id}Vla"
    if m.group(1).endswith("Vla"):
        return task_id
    return f"{m.group(1)}Vla_v{m.group(2)}"


def _vla_instruction(canonical: str, original_instruction: str) -> str:
    """Short English instructions aligned with VLA move-to / place-held semantics."""
    # Placement / geometric: emphasize held cube + single placement (layout may add pre_attach).
    place_style = {
        "place_directional": "You already hold the wooden cube. Place it in the legal region in front of the blue plate.",
        "place_on_top": "You already hold the cube. Release it onto the top surface of the red wooden block so it rests stably.",
        "place_inside": "You already hold the cube. Put it fully inside the blue box.",
        "square_completion": "You already hold the fourth cube. Place it to complete a square with the three anchor cubes.",
        "trend_continuation": "You already hold the next cube. Place it to continue the visible row/column pattern.",
        "perfect_midpoint": "You already hold the cube. Place it midway between the red and blue reference cubes.",
        "point_symmetry": "You already hold the cube. Place it symmetrically to the blue cube about the red cube.",
        "isometric_extension": "You already hold the cube. Place it on the red–blue line extension at equal spacing.",
    }
    move_style = {
        "track_single": "Move the end effector near the cube that started in the middle.",
        "track_multi": "Move the end effector near the cube that started at the far left.",
        "track_clutter": "Move the end effector near the cube that started in the middle after clutter settles.",
        "grasp_anchor_direction": "Move the end effector near the cube to the right of the blue bowl.",
        "grasp_anchor_distance": "Move the end effector near the sphere closest to the red cylinder.",
        "reindex_extreme": "Move the end effector near the leftmost cube after reordering.",
        "reindex_insert": "Move the end effector near the second-from-left cube.",
        "reindex_neighbor": "Move the end effector near the cube immediately left of the red cube.",
        "reindex_subset": "Move the end effector near the middle cube among the red cubes only.",
        "ordinal_insert_static": "Move the end effector near the second cube from the left.",
        "grid_rowcol_grounding": "Move the end effector near the cube at row 1, column 2.",
        "select_exclusion_unique": "Move the end effector near the red cube closest to the blue plate that is not the leftmost.",
        "select_visual_spatial": "Move the end effector near the wood-grain cylinder to the right of the red block.",
    }
    if canonical in place_style:
        return place_style[canonical]
    if canonical in move_style:
        return move_style[canonical]
    return original_instruction


def _layout_runtime_objects(blob: dict, task_index: int) -> list[dict]:
    ann = blob["tasks"][task_index]["annotation"]
    scene_adaptation = ann.get("scene_adaptation", {})
    runtime_objects = scene_adaptation.get("runtime_objects", []) if isinstance(scene_adaptation, dict) else []
    return [runtime_object for runtime_object in runtime_objects if isinstance(runtime_object, dict)]


def _runtime_object_dimensions_xy(runtime_object: dict) -> tuple[float, float] | None:
    kind = str(runtime_object.get("kind", "")).strip()
    if kind == "primitive_cuboid":
        size = runtime_object.get("size", [])
        if isinstance(size, list) and len(size) >= 2:
            return (abs(float(size[0])), abs(float(size[1])))
    if kind == "primitive_sphere":
        radius = abs(float(runtime_object.get("radius", 0.0) or 0.0))
        if radius > 0.0:
            diameter = 2.0 * radius
            return (diameter, diameter)
    if kind in {"primitive_cylinder", "primitive_cone"}:
        radius = abs(float(runtime_object.get("radius", 0.0) or 0.0))
        if radius > 0.0:
            diameter = 2.0 * radius
            return (diameter, diameter)
    return None


def _move_to_success_radius_for_target(blob: dict, task_index: int, target_object_id: str) -> float:
    for runtime_object in _layout_runtime_objects(blob, task_index):
        if str(runtime_object.get("name", "")).strip() != target_object_id:
            continue
        dims_xy = _runtime_object_dimensions_xy(runtime_object)
        if dims_xy is None:
            break
        object_size = max(float(dims_xy[0]), float(dims_xy[1]))
        return round(max(0.055, min(0.065, 1.2 * object_size)), 6)
    return 0.06


def _layout_track_single_vla(legacy: dict) -> dict:
    src_key = "historical_identity_blocks_line_3"
    blob = _deepcopy_json(legacy[src_key])
    t = blob["tasks"][0]
    t["instruction"] = _vla_instruction("track_single", "")
    ann = t["annotation"]
    # Keep protocol intervention visible inside short VLA smoke rollouts.
    ann["protocol"]["query_text"] = t["instruction"]
    ann["protocol"]["gt_spec"] = {
        "resolver": "move_to_object_proximity",
        "target_object_id": "identity_block_1",
        "success_radius": _move_to_success_radius_for_target(blob, 0, "identity_block_1"),
        "xy_only": True,
        # Blocks are colinear along world Y (same X); intervention motion ranks/sorts on this axis.
        "order_axis": "y",
    }
    ann["protocol"]["target_object"] = {"object_id": "identity_block_1"}
    interv = dict(ann["protocol"].get("intervention") or {})
    interv["duration_s"] = VLA_VISIBLE_INTERVENTION_DURATION_S
    interv["instant_swap"] = False
    interv["vla_visible_motion"] = True
    ann["protocol"]["intervention"] = interv
    return blob


def _layout_anchor_direction_vla(legacy: dict) -> dict:
    src_key = "grasp_anchor_direction_scene65_base"
    blob = _deepcopy_json(legacy[src_key])
    t = blob["tasks"][0]
    t["instruction"] = _vla_instruction("grasp_anchor_direction", "")
    ann = t["annotation"]
    ann["protocol"]["query_text"] = t["instruction"]
    ann["protocol"]["gt_spec"] = {
        "resolver": "move_to_object_proximity",
        "target_object_id": "block_right",
        "success_radius": _move_to_success_radius_for_target(blob, 0, "block_right"),
        "xy_only": True,
    }
    ann["protocol"]["target_object"] = {"object_id": "block_right"}
    return blob


def _layout_anchor_distance_vla(legacy: dict) -> dict:
    src_key = "grasp_anchor_distance_scene10_base"
    blob = _deepcopy_json(legacy[src_key])
    t = blob["tasks"][0]
    t["instruction"] = _vla_instruction("grasp_anchor_distance", "")
    ann = t["annotation"]
    ann["protocol"]["query_text"] = t["instruction"]
    ann["protocol"]["gt_spec"] = {
        "resolver": "move_to_object_proximity",
        "target_object_id": "ball_nearest",
        "success_radius": _move_to_success_radius_for_target(blob, 0, "ball_nearest"),
        "xy_only": True,
    }
    ann["protocol"]["target_object"] = {"object_id": "ball_nearest"}
    return blob


def _layout_state_reindex_extreme_vla(legacy: dict) -> dict:
    src_key = "state_reindex_blocks_line_4"
    blob = _deepcopy_json(legacy[src_key])
    t = blob["tasks"][0]
    t["instruction"] = _vla_instruction("reindex_extreme", "")
    ann = t["annotation"]
    interv = dict(ann["protocol"].get("intervention") or {})
    interv["duration_s"] = VLA_REINDEX_EXTREME_INTERVENTION_DURATION_S
    interv["instant_swap"] = False
    interv["vla_visible_motion"] = True
    ann["protocol"]["intervention"] = interv
    ann["protocol"]["query_text"] = t["instruction"]
    ann["protocol"]["gt_spec"] = {
        "resolver": "move_to_object_proximity",
        "target_object_id": "reindex_block_1",
        "success_radius": _move_to_success_radius_for_target(blob, 0, "reindex_block_1"),
        "xy_only": True,
    }
    ann["protocol"]["target_object"] = {"object_id": "reindex_block_1"}
    return blob


def _layout_place_directional_vla(legacy: dict, src_key: str, canonical_instruction: str) -> dict:
    blob = _deepcopy_json(legacy[src_key])
    t = blob["tasks"][0]
    t["instruction"] = canonical_instruction
    ann = t["annotation"]
    ann["protocol"]["query_text"] = t["instruction"]
    proto = ann["protocol"]
    tgt = proto.get("target_object", {})
    if isinstance(tgt, dict):
        oid = str(tgt.get("object_id", "movable_cube"))
    else:
        oid = "movable_cube"
    proto["target_object"] = {"object_id": oid, "carrier": "gripper", "preloaded": True}
    gt = proto.get("gt_spec", {})
    if isinstance(gt, dict) and isinstance(gt.get("target_zone"), dict):
        gt["resolver"] = "directional_region_placement"
        gt["must_be_released"] = True
    return blob


def _runtime_xy_by_name(blob: dict, task_index: int) -> dict[str, tuple[float, float]]:
    ann = blob["tasks"][task_index]["annotation"]
    scene_adaptation = ann.get("scene_adaptation", {})
    runtime_objects = scene_adaptation.get("runtime_objects", []) if isinstance(scene_adaptation, dict) else []
    xy_by_name: dict[str, tuple[float, float]] = {}
    for runtime_object in runtime_objects:
        if not isinstance(runtime_object, dict):
            continue
        name = str(runtime_object.get("name", "")).strip()
        pose = runtime_object.get("pose", {})
        xy = pose.get("xy", []) if isinstance(pose, dict) else []
        if name and isinstance(xy, list) and len(xy) >= 2:
            xy_by_name[name] = (float(xy[0]), float(xy[1]))
    object_positions = (blob.get("scene_structure", {}) or {}).get("object_positions", {})
    if isinstance(object_positions, dict):
        for name, payload in object_positions.items():
            pos = payload.get("position", []) if isinstance(payload, dict) else []
            if str(name).strip() and isinstance(pos, list) and len(pos) >= 2:
                xy_by_name.setdefault(str(name), (float(pos[0]), float(pos[1])))
    return xy_by_name


def _protocol_anchor_xy(protocol: dict, xy_by_name: dict[str, tuple[float, float]], anchor_id: str) -> tuple[float, float]:
    for anchor in protocol.get("anchors", []):
        if not isinstance(anchor, dict):
            continue
        if str(anchor.get("anchor_id", "")).strip() != anchor_id:
            continue
        object_id = str(anchor.get("object_id", "")).strip()
        if object_id in xy_by_name:
            return xy_by_name[object_id]
    raise KeyError(f"Missing geometric anchor {anchor_id!r}.")


def _geometric_goal_xy(blob: dict, task_index: int, canonical: str) -> tuple[float, float]:
    ann = blob["tasks"][task_index]["annotation"]
    protocol = ann["protocol"]
    xy_by_name = _runtime_xy_by_name(blob, task_index)
    if canonical == "perfect_midpoint":
        ax, ay = _protocol_anchor_xy(protocol, xy_by_name, "A")
        bx, by = _protocol_anchor_xy(protocol, xy_by_name, "B")
        return (0.5 * (ax + bx), 0.5 * (ay + by))
    if canonical == "isometric_extension":
        ax, ay = _protocol_anchor_xy(protocol, xy_by_name, "A")
        bx, by = _protocol_anchor_xy(protocol, xy_by_name, "B")
        return (bx + (bx - ax), by + (by - ay))
    if canonical == "point_symmetry":
        cx, cy = _protocol_anchor_xy(protocol, xy_by_name, "A")
        rx, ry = _protocol_anchor_xy(protocol, xy_by_name, "B")
        return (2.0 * cx - rx, 2.0 * cy - ry)
    if canonical == "trend_continuation":
        anchors = [
            xy_by_name[str(anchor.get("object_id", "")).strip()]
            for anchor in protocol.get("anchors", [])
            if isinstance(anchor, dict) and str(anchor.get("object_id", "")).strip() in xy_by_name
        ]
        if len(anchors) < 2:
            raise KeyError("Trend continuation requires at least two resolved anchors.")
        dx = sum(anchors[i + 1][0] - anchors[i][0] for i in range(len(anchors) - 1)) / float(len(anchors) - 1)
        dy = sum(anchors[i + 1][1] - anchors[i][1] for i in range(len(anchors) - 1)) / float(len(anchors) - 1)
        return (anchors[-1][0] + dx, anchors[-1][1] + dy)
    if canonical == "square_completion":
        ax, ay = _protocol_anchor_xy(protocol, xy_by_name, "A")
        bx, by = _protocol_anchor_xy(protocol, xy_by_name, "B")
        cx, cy = _protocol_anchor_xy(protocol, xy_by_name, "C")
        return (bx + cx - ax, by + cy - ay)
    raise ValueError(f"Unsupported geometric VLA canonical task: {canonical!r}")


def _unit_vector(dx: float, dy: float) -> tuple[float, float] | None:
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1.0e-9:
        return None
    return (dx / norm, dy / norm)


def _geometric_goal_axis(blob: dict, task_index: int, canonical: str) -> tuple[float, float] | None:
    """Unit direction along which ``success_tolerance_l2`` is allowed to be spent.

    ``success_tolerance_orthogonal`` applies perpendicular to it. ``square_completion`` is
    driven by two independent edge directions, so there is no single orthogonal axis and it
    stays isotropic.
    """
    ann = blob["tasks"][task_index]["annotation"]
    protocol = ann["protocol"]
    xy_by_name = _runtime_xy_by_name(blob, task_index)
    if canonical in {"perfect_midpoint", "isometric_extension"}:
        ax, ay = _protocol_anchor_xy(protocol, xy_by_name, "A")
        bx, by = _protocol_anchor_xy(protocol, xy_by_name, "B")
        return _unit_vector(bx - ax, by - ay)
    if canonical == "point_symmetry":
        cx, cy = _protocol_anchor_xy(protocol, xy_by_name, "A")
        rx, ry = _protocol_anchor_xy(protocol, xy_by_name, "B")
        return _unit_vector(cx - rx, cy - ry)
    if canonical == "trend_continuation":
        anchors = [
            xy_by_name[str(anchor.get("object_id", "")).strip()]
            for anchor in protocol.get("anchors", [])
            if isinstance(anchor, dict) and str(anchor.get("object_id", "")).strip() in xy_by_name
        ]
        if len(anchors) < 2:
            return None
        return _unit_vector(anchors[-1][0] - anchors[0][0], anchors[-1][1] - anchors[0][1])
    return None


def _geometric_goal_tolerance(blob: dict, task_index: int, canonical: str) -> dict:
    protocol = blob["tasks"][task_index]["annotation"]["protocol"]
    tolerance = float(protocol.get("success_tolerance_l2", 0.05) or 0.05)
    payload = {"pos_r": round(tolerance, 6)}
    try:
        orthogonal = float(protocol.get("success_tolerance_orthogonal") or 0.0)
    except (TypeError, ValueError):
        orthogonal = 0.0
    axis = _geometric_goal_axis(blob, task_index, canonical)
    if axis is not None and 0.0 < orthogonal < tolerance:
        payload["orthogonal_r"] = round(orthogonal, 6)
        payload["axis"] = [round(axis[0], 6), round(axis[1], 6)]
    return payload


def _patch_geometric_goal_sequence(blob: dict, task_index: int, canonical: str) -> None:
    ann = blob["tasks"][task_index]["annotation"]
    protocol = ann["protocol"]
    target_object = protocol.get("target_object", {})
    target_object_id = str(target_object.get("object_id", "")).strip() if isinstance(target_object, dict) else ""
    if not target_object_id:
        raise KeyError(f"Missing target_object.object_id for {canonical}.")
    goal_xy = _geometric_goal_xy(blob, task_index, canonical)
    goal = {
        "object_id": target_object_id,
        "xy": [round(float(goal_xy[0]), 6), round(float(goal_xy[1]), 6)],
        "tolerance": _geometric_goal_tolerance(blob, task_index, canonical),
    }
    ann["goal_sequence"] = [{"step": 1, "goals": [goal]}]
    scene_adaptation = ann.setdefault("scene_adaptation", {})
    if isinstance(scene_adaptation, dict):
        scene_adaptation["preview_proxy_goal"] = {
            "object_id": target_object_id,
            "xy": list(goal["xy"]),
            "tolerance": dict(goal["tolerance"]),
        }


def _layout_square_completion_vla(legacy: dict) -> dict:
    src_key = "geometric_inference_square_completion_corner_3"
    blob = _deepcopy_json(legacy[src_key])
    t = blob["tasks"][0]
    t["instruction"] = _vla_instruction("square_completion", "")
    ann = t["annotation"]
    ann["protocol"]["query_text"] = t["instruction"]
    to = ann["protocol"].get("target_object")
    if isinstance(to, dict) and to.get("object_id"):
        to["preloaded"] = True
        to["carrier"] = "gripper"
    _patch_geometric_goal_sequence(blob, 0, "square_completion")
    return blob


def _patch_move_to_proximity_task(
    blob: dict,
    *,
    task_index: int,
    target_object_id: str,
    canonical: str,
) -> None:
    t = blob["tasks"][task_index]
    t["instruction"] = _vla_instruction(canonical, t.get("instruction", ""))
    ann = t["annotation"]
    # Tracking/reindex layouts need scripted/timed interventions so objects move while the policy acts.
    if canonical not in VLA_DYNAMIC_CANONICALS:
        ann["protocol"].pop("intervention", None)
    else:
        interv = dict(ann["protocol"].get("intervention") or {})
        interv["duration_s"] = (
            VLA_REINDEX_SUBSET_INTERVENTION_DURATION_S
            if canonical == "reindex_subset"
            else VLA_TRACK_CLUTTER_INTERVENTION_DURATION_S
            if canonical == "track_clutter"
            else VLA_REINDEX_EXTREME_INTERVENTION_DURATION_S
            if canonical == "reindex_extreme"
            else VLA_REINDEX_NEIGHBOR_INTERVENTION_DURATION_S
            if canonical == "reindex_neighbor"
            else VLA_VISIBLE_INTERVENTION_DURATION_S
        )
        interv["instant_swap"] = False
        interv["vla_visible_motion"] = True
        ann["protocol"]["intervention"] = interv
    ann["protocol"]["query_text"] = t["instruction"]
    gt_spec: dict = {
        "resolver": "move_to_object_proximity",
        "target_object_id": target_object_id,
        "success_radius": _move_to_success_radius_for_target(blob, task_index, target_object_id),
        "xy_only": True,
    }
    if canonical in {"track_single", "track_multi", "track_clutter"}:
        gt_spec["order_axis"] = "y"
    ann["protocol"]["gt_spec"] = gt_spec
    ann["protocol"]["target_object"] = {"object_id": target_object_id}


def _patch_target_object_preloaded(blob: dict, task_index: int, object_id: str) -> None:
    ann = blob["tasks"][task_index]["annotation"]
    proto = ann["protocol"]
    proto["target_object"] = {
        "object_id": object_id,
        "carrier": "gripper",
        "preloaded": True,
    }


def _patch_geometric_instruction_preloaded(blob: dict, task_index: int, canonical: str) -> None:
    t = blob["tasks"][task_index]
    t["instruction"] = _vla_instruction(canonical, t.get("instruction", ""))
    ann = t["annotation"]
    p = ann["protocol"]
    p["query_text"] = t["instruction"]
    to = p.get("target_object")
    if isinstance(to, dict) and to.get("object_id"):
        to["preloaded"] = True
        to["carrier"] = "gripper"
    _patch_geometric_goal_sequence(blob, task_index, canonical)


def build_vla_layouts_from_annotated(annot: dict) -> dict:
    """VLA layouts sourced from assets/annotations/annotated_tasks.json (IsaacLab truth source)."""
    out: dict[str, dict] = {}

    def put(src_key: str, dst_key: str, fn) -> None:
        if src_key not in annot:
            raise KeyError(f"Missing annotated layout {src_key!r}")
        blob = _deepcopy_json(annot[src_key])
        fn(blob)
        out[dst_key] = blob

    put(
        "historical_identity_multistep_blocks_line_5",
        "historical_identity_multistep_blocks_line_5_vla",
        lambda b: _patch_move_to_proximity_task(b, task_index=0, target_object_id="identity_block_0", canonical="track_multi"),
    )
    put(
        "clutter_robust_identity_blocks_line_3",
        "clutter_robust_identity_blocks_line_3_vla",
        lambda b: _patch_move_to_proximity_task(b, task_index=0, target_object_id="identity_block_1", canonical="track_clutter"),
    )
    put(
        "state_reindex_insert_blocks_line_4",
        "state_reindex_insert_blocks_line_4_vla",
        lambda b: _patch_move_to_proximity_task(b, task_index=0, target_object_id="insert_block_3", canonical="reindex_insert"),
    )
    put(
        "state_reindex_neighbor_red_left_blocks_5",
        "state_reindex_neighbor_red_left_blocks_5_vla",
        lambda b: _patch_move_to_proximity_task(b, task_index=0, target_object_id="right_neighbor_block", canonical="reindex_neighbor"),
    )
    put(
        "state_reindex_red_subset_blocks_line_5",
        "state_reindex_red_subset_blocks_line_5_vla",
        lambda b: _patch_move_to_proximity_task(b, task_index=0, target_object_id="red_subset_block_2", canonical="reindex_subset"),
    )
    put(
        "grid_rowcol_grounding_grid_2x3_base",
        "grid_rowcol_grounding_grid_2x3_base_vla",
        lambda b: _patch_move_to_proximity_task(b, task_index=0, target_object_id="grid_cell_1", canonical="grid_rowcol_grounding"),
    )
    put(
        "ordinal_insert_static_blocks_base",
        "ordinal_insert_static_blocks_base_vla",
        lambda b: _patch_move_to_proximity_task(b, task_index=0, target_object_id="ordinal_static_1", canonical="ordinal_insert_static"),
    )
    put(
        "select_exclusion_unique_scene65_base",
        "select_exclusion_unique_scene65_base_vla",
        lambda b: _patch_move_to_proximity_task(b, task_index=0, target_object_id="red_block_target", canonical="select_exclusion_unique"),
    )
    put(
        "select_visual_spatial_scene65_base",
        "select_visual_spatial_scene65_base_vla",
        lambda b: _patch_move_to_proximity_task(b, task_index=0, target_object_id="wood_cylinder_target", canonical="select_visual_spatial"),
    )

    def _place_on_top_vla(b: dict) -> None:
        t = b["tasks"][0]
        t["instruction"] = _vla_instruction("place_on_top", t.get("instruction", ""))
        ann = t["annotation"]
        ann["protocol"]["query_text"] = t["instruction"]
        _patch_target_object_preloaded(b, 0, "movable_cube")

    put("place_on_top_scene65_skill", "place_on_top_scene65_skill_vla", _place_on_top_vla)

    def _place_inside_vla(b: dict) -> None:
        t = b["tasks"][0]
        t["instruction"] = _vla_instruction("place_inside", t.get("instruction", ""))
        ann = t["annotation"]
        ann["protocol"]["query_text"] = t["instruction"]
        _patch_target_object_preloaded(b, 0, "movable_cube")

    put("place_inside_scene65_skill", "place_inside_scene65_skill_vla", _place_inside_vla)

    for src, canon in (
        ("geometric_inference_perfect_midpoint_line_2", "perfect_midpoint"),
        ("geometric_inference_point_symmetry_line_2", "point_symmetry"),
        ("geometric_inference_isometric_extension_line_2", "isometric_extension"),
        ("geometric_inference_implicit_trend_line_3", "trend_continuation"),
    ):
        dst = f"{src}_vla"

        def _geom_closure(c: str):
            def _apply(b: dict) -> None:
                _patch_geometric_instruction_preloaded(b, 0, c)

            return _apply

        put(src, dst, _geom_closure(canon))

    return out


def build_vla_legacy_layouts(legacy: dict) -> dict:
    out = {}
    out["historical_identity_blocks_line_3_vla"] = _layout_track_single_vla(legacy)
    out["grasp_anchor_direction_scene65_base_vla"] = _layout_anchor_direction_vla(legacy)
    out["grasp_anchor_distance_scene10_base_vla"] = _layout_anchor_distance_vla(legacy)
    out["state_reindex_blocks_line_4_vla"] = _layout_state_reindex_extreme_vla(legacy)
    out["geometric_inference_square_completion_corner_3_vla"] = _layout_square_completion_vla(legacy)
    out["place_directional_scene65_base_vla"] = _layout_place_directional_vla(
        legacy,
        "place_directional_scene65_base",
        _vla_instruction("place_directional", ""),
    )
    out["place_directional_scene65_semantic_vla"] = _layout_place_directional_vla(
        legacy,
        "place_directional_scene65_semantic",
        _vla_instruction("place_directional", ""),
    )
    out["place_directional_scene65_visual_vla"] = _layout_place_directional_vla(
        legacy,
        "place_directional_scene65_visual",
        _vla_instruction("place_directional", ""),
    )
    return out


CANONICAL_TO_VLA_LAYOUT_WHEN_EMPTY = {
    "track_multi": "historical_identity_multistep_blocks_line_5_vla",
    "track_clutter": "clutter_robust_identity_blocks_line_3_vla",
    "place_on_top": "place_on_top_scene65_skill_vla",
    "place_inside": "place_inside_scene65_skill_vla",
    "reindex_insert": "state_reindex_insert_blocks_line_4_vla",
    "grid_rowcol_grounding": "grid_rowcol_grounding_grid_2x3_base_vla",
    "ordinal_insert_static": "ordinal_insert_static_blocks_base_vla",
    "reindex_subset": "state_reindex_red_subset_blocks_line_5_vla",
    "perfect_midpoint": "geometric_inference_perfect_midpoint_line_2_vla",
    "point_symmetry": "geometric_inference_point_symmetry_line_2_vla",
    "isometric_extension": "geometric_inference_isometric_extension_line_2_vla",
    "trend_continuation": "geometric_inference_implicit_trend_line_3_vla",
    "select_exclusion_unique": "select_exclusion_unique_scene65_base_vla",
    "select_visual_spatial": "select_visual_spatial_scene65_base_vla",
    "reindex_neighbor": "state_reindex_neighbor_red_left_blocks_5_vla",
}


def _map_layout_key_for_vla(original_key: str, canonical_task_id: str) -> str:
    """Map base layout keys to vla_layouts keys; empty keys use canonical_task_id."""
    mapping = {
        "historical_identity_blocks_line_3": "historical_identity_blocks_line_3_vla",
        "track_single_scene65_semantic_anchors": "historical_identity_blocks_line_3_vla",
        "historical_identity_blocks_line_3_visual_occlusion": "historical_identity_blocks_line_3_vla",
        "grasp_anchor_direction_scene65_base": "grasp_anchor_direction_scene65_base_vla",
        "grasp_anchor_distance_scene10_base": "grasp_anchor_distance_scene10_base_vla",
        "state_reindex_blocks_line_4": "state_reindex_blocks_line_4_vla",
        "geometric_inference_square_completion_corner_3": "geometric_inference_square_completion_corner_3_vla",
        "place_directional_scene65_base": "place_directional_scene65_base_vla",
        "place_directional_scene65_semantic": "place_directional_scene65_semantic_vla",
        "place_directional_scene65_visual": "place_directional_scene65_visual_vla",
    }
    key = str(original_key or "").strip()
    if key:
        return mapping.get(key, key)
    canon = str(canonical_task_id or "").strip()
    return CANONICAL_TO_VLA_LAYOUT_WHEN_EMPTY.get(canon, "")


def emit_task_json(src_path: Path, legacy: dict) -> None:
    definition = json.loads(src_path.read_text(encoding="utf-8"))
    if definition.get("status") in {"Deprecated", "RelationVariant"} or definition.get("runtime", definition).get("canonical_task_id") == "place_directional":
        return
    payload = normalize_task_config(definition)
    if payload.get("implementation_status") == "placeholder" or payload.get('backend')=='assembled_native':
        # Native assembled recipes execute actual pick/place through their own
        # existing PhysX runner. They cannot be turned into an old manager/VLA
        # environment by copying legacy metadata or changing task_version.
        return
    old_id = str(payload.get("task_id", ""))
    if old_id == "legacy_layouts.json" or not old_id:
        return
    canon = str(payload.get("canonical_task_id", ""))
    instr = str(payload.get("instruction", ""))
    new_instr = _vla_instruction(canon, instr) if canon else instr
    if payload.get("public_task_id") in {"SelectByOrder", "UpdateObjectOrder", "SelectByAdjacency", "TrackObjectOrigin"} and instr.startswith("Pick up"):
        new_instr = instr.replace("Pick up", "Move the end effector near", 1)

    payload["task_id"] = _vla_task_id(old_id)
    payload["registration_id"] = payload["task_id"]
    payload["interface"] = "Vla"
    payload["variant"] = "vla"
    payload["task_version"] = "vla"
    payload["instruction"] = new_instr
    payload.pop("evaluator_key", None)
    payload.pop("skill_type", None)
    olayout = str(payload.get("annotation_layout_key", ""))
    payload["annotation_layout_key"] = _map_layout_key_for_vla(olayout, canon)
    tags = list(payload.get("tags", []))
    if "vla" not in [str(x).lower() for x in tags]:
        tags.append("vla")
    payload["tags"] = tags
    # Public base aliases must point to the corresponding VLA version here;
    # copying them verbatim would shadow the base task after prefix removal.
    aliases = [
        _vla_task_id(str(alias))
        if re.fullmatch(r"[A-Z][A-Za-z0-9_]*_v\d+", str(alias))
        else str(alias)
        for alias in payload.get("aliases", [])
    ]
    existing_path = OUT_DIR / f"{payload['task_id']}.json"
    if existing_path.is_file():
        aliases.extend(json.loads(existing_path.read_text(encoding="utf-8")).get("aliases", []))
    aliases = list(dict.fromkeys(
        _vla_task_id(str(alias))
        if re.fullmatch(r"[A-Z][A-Za-z0-9_]*_v\d+", str(alias))
        else str(alias)
        for alias in aliases
    ))
    alias_seed = payload["task_id"].replace("_v0", "")
    if alias_seed not in aliases:
        aliases.append(f"{canon}:vla" if canon else alias_seed)
    payload["aliases"] = aliases

    out_path = OUT_DIR / f"{payload['task_id']}.json"
    for key in ("public_task_id", "task_family", "condition", "layout"):
        payload.pop(key, None)
    output_definition = upgrade_task_config(
        payload, family=definition.get("family") if "runtime" in definition else None,
        condition=definition.get("condition"), layout=definition.get("layout"),
    )
    output_definition["parameters"] = dict(payload.get("parameters", {}))
    out_path.write_text(json.dumps(output_definition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    legacy = json.loads(LEGACY_SRC.read_text(encoding="utf-8"))
    vla_legacy = build_vla_legacy_layouts(legacy)
    if ANNOTATED_SRC.is_file():
        annot = json.loads(ANNOTATED_SRC.read_text(encoding="utf-8"))
        vla_legacy.update(build_vla_layouts_from_annotated(annot))
    VLA_LAYOUTS.parent.mkdir(parents=True, exist_ok=True)
    VLA_LAYOUTS.write_text(
        json.dumps(vla_legacy, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for path in sorted(TASKS_DIR.glob("*.json")):
        if path.name == "legacy_layouts.json":
            continue
        emit_task_json(path, legacy)
    # The legacy emitter builds protocol records; publish compact episodes as
    # the active configs rather than reintroducing runtime fields there.
    from geomind.tools.migrate_episode_configs import migrate
    migrate(REPO / "geomind" / "configs", replace_records=True)
    print(f"Wrote {OUT_DIR} ({len(vla_legacy)} layout keys + task JSONs)")


if __name__ == "__main__":
    main()
