from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


CONSTRAINT_SELECTION_MODULE_NAME = "constraint_selection"
VISUAL_SPATIAL_CONSTRAINT_SELECTION_RESOLVER = "visual_spatial_constraint_selection"
EXCLUSION_NEAREST_CONSTRAINT_SELECTION_RESOLVER = "exclusion_nearest_constraint_selection"
DEFAULT_MIN_PRIMARY_OFFSET = 0.06
DEFAULT_MAX_ABS_SECONDARY_OFFSET = 0.03
DEFAULT_RANK_TIE_EPSILON = 1.0e-4
DEFAULT_DISTANCE_TIE_EPSILON = 1.0e-4


def _coerce_xy(value: Any, *, field_name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise ValueError(f"{field_name} must provide at least two numeric values.")
    return (float(value[0]), float(value[1]))


def _extract_object_xy(object_positions: Mapping[str, Any], object_id: str) -> tuple[float, float]:
    payload = object_positions.get(object_id)
    if not isinstance(payload, Mapping):
        raise KeyError(f"Object '{object_id}' is missing from object positions.")
    return _coerce_xy(payload.get("position"), field_name=f"object_positions[{object_id!r}].position")


def _candidate_object_ids(protocol: Mapping[str, Any]) -> list[str]:
    raw_ids = protocol.get("candidate_object_ids", [])
    if not isinstance(raw_ids, list):
        return []
    return [str(object_id) for object_id in raw_ids if isinstance(object_id, str) and str(object_id).strip()]


def is_constraint_selection_protocol(protocol: Any) -> bool:
    return isinstance(protocol, Mapping) and str(protocol.get("module_name", "")).strip() == CONSTRAINT_SELECTION_MODULE_NAME


def _normalize_axis(axis: Any) -> str:
    normalized = str(axis).strip().lower()
    if normalized not in {"x", "y"}:
        raise ValueError(f"Unsupported constraint selection axis: {axis!r}")
    return normalized


def _normalize_direction(direction: Any) -> str:
    normalized = str(direction).strip().lower()
    alias_map = {
        "positive": "positive",
        "+": "positive",
        "negative": "negative",
        "-": "negative",
    }
    resolved = alias_map.get(normalized)
    if resolved is None:
        raise ValueError(f"Unsupported constraint selection direction: {direction!r}")
    return resolved


def _normalize_extreme_rank(rank: Any) -> str:
    normalized = str(rank).strip().lower()
    alias_map = {
        "minimum": "minimum",
        "min": "minimum",
        "leftmost": "minimum",
        "smallest": "minimum",
        "maximum": "maximum",
        "max": "maximum",
        "rightmost": "maximum",
        "largest": "maximum",
    }
    resolved = alias_map.get(normalized)
    if resolved is None:
        raise ValueError(f"Unsupported exclusion rank: {rank!r}")
    return resolved


def _normalize_metric(metric: Any) -> str:
    normalized = str(metric).strip().lower() or "planar_euclidean"
    if normalized != "planar_euclidean":
        raise ValueError(f"Unsupported constraint selection metric: {metric!r}")
    return normalized


def _normalize_distance_axes(gt_spec: Mapping[str, Any]) -> tuple[str, ...]:
    raw_distance_axes = gt_spec.get("distance_axes", ("x", "y"))
    if not isinstance(raw_distance_axes, Sequence) or isinstance(raw_distance_axes, (str, bytes)):
        raw_distance_axes = ("x", "y")
    distance_axes = tuple(dict.fromkeys(_normalize_axis(axis) for axis in raw_distance_axes))
    if set(distance_axes) != {"x", "y"}:
        raise ValueError(
            "constraint selection currently supports planar Euclidean distance on the world X/Y axes only."
        )
    return distance_axes


def _normalize_semantics_mapping(protocol: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    raw_mapping = protocol.get("object_semantics", {})
    if not isinstance(raw_mapping, Mapping):
        return {}

    resolved: dict[str, dict[str, str]] = {}
    for object_id, payload in raw_mapping.items():
        if not isinstance(payload, Mapping):
            continue
        object_id_text = str(object_id).strip()
        if not object_id_text:
            continue
        semantics: dict[str, str] = {}
        for key, value in payload.items():
            key_text = str(key).strip()
            value_text = str(value).strip()
            if key_text and value_text:
                semantics[key_text] = value_text
        resolved[object_id_text] = semantics
    return resolved


def _normalize_target_filters(protocol: Mapping[str, Any], gt_spec: Mapping[str, Any]) -> dict[str, str]:
    raw_filters = gt_spec.get("target_filters", protocol.get("target_filters", {}))
    if not isinstance(raw_filters, Mapping):
        return {}
    resolved: dict[str, str] = {}
    for key, value in raw_filters.items():
        key_text = str(key).strip()
        value_text = str(value).strip()
        if key_text and value_text:
            resolved[key_text] = value_text
    return resolved


def _attribute_match(
    semantics: Mapping[str, str],
    target_filters: Mapping[str, str],
) -> bool:
    for key, expected_value in target_filters.items():
        if str(semantics.get(key, "")).strip() != str(expected_value).strip():
            return False
    return True


def _extract_anchor_and_candidate_xy(
    protocol: Mapping[str, Any],
    object_positions: Mapping[str, Any],
) -> tuple[str, tuple[float, float], list[str], dict[str, tuple[float, float]], Mapping[str, Any]]:
    if not is_constraint_selection_protocol(protocol):
        raise ValueError("task protocol is missing or is not a constraint_selection protocol.")

    anchor_object = protocol.get("anchor_object", {})
    if not isinstance(anchor_object, Mapping):
        raise ValueError("constraint_selection protocol is missing anchor_object.")
    anchor_object_id = str(anchor_object.get("object_id", "")).strip()
    if not anchor_object_id:
        raise ValueError("constraint_selection protocol is missing anchor_object.object_id.")

    candidate_object_ids = _candidate_object_ids(protocol)
    if not candidate_object_ids:
        raise ValueError("constraint_selection protocol is missing candidate_object_ids.")

    gt_spec = protocol.get("gt_spec", {})
    if not isinstance(gt_spec, Mapping):
        gt_spec = {}

    anchor_xy = _extract_object_xy(object_positions, anchor_object_id)
    candidate_positions_xy = {
        object_id: _extract_object_xy(object_positions, object_id)
        for object_id in candidate_object_ids
    }
    return anchor_object_id, anchor_xy, candidate_object_ids, candidate_positions_xy, gt_spec


def resolve_visual_spatial_constraint_problem(
    protocol: Mapping[str, Any],
    object_positions: Mapping[str, Any],
) -> dict[str, Any]:
    (
        anchor_object_id,
        anchor_xy,
        candidate_object_ids,
        candidate_positions_xy,
        gt_spec,
    ) = _extract_anchor_and_candidate_xy(protocol, object_positions)

    resolver = str(gt_spec.get("resolver", "")).strip()
    if resolver and resolver != VISUAL_SPATIAL_CONSTRAINT_SELECTION_RESOLVER:
        raise ValueError(f"Unsupported constraint selection resolver: {resolver!r}")

    axis = _normalize_axis(gt_spec.get("axis", "y"))
    direction = _normalize_direction(gt_spec.get("direction", "negative"))
    min_primary_offset = float(gt_spec.get("min_primary_offset", DEFAULT_MIN_PRIMARY_OFFSET))
    max_abs_secondary_offset = float(gt_spec.get("max_abs_secondary_offset", DEFAULT_MAX_ABS_SECONDARY_OFFSET))
    expected_target = str(gt_spec.get("expected_target", "")).strip()
    target_filters = _normalize_target_filters(protocol, gt_spec)
    if not target_filters:
        raise ValueError("constraint_selection protocol is missing gt_spec.target_filters.")

    if expected_target and expected_target not in candidate_object_ids:
        raise ValueError(f"Expected target {expected_target!r} is not listed in candidate_object_ids.")

    object_semantics = _normalize_semantics_mapping(protocol)

    candidate_geometry: dict[str, dict[str, float | bool]] = {}
    candidate_semantics: dict[str, dict[str, str]] = {}
    spatial_match_ids: list[str] = []
    attribute_match_ids: list[str] = []
    valid_candidate_ids: list[str] = []
    anchor_x, anchor_y = anchor_xy

    for object_id, position_xy in candidate_positions_xy.items():
        semantics = dict(object_semantics.get(object_id, {}))
        candidate_semantics[object_id] = semantics

        candidate_x, candidate_y = position_xy
        dx = float(candidate_x - anchor_x)
        dy = float(candidate_y - anchor_y)
        if axis == "x":
            primary_delta = dx
            secondary_delta = dy
        else:
            primary_delta = dy
            secondary_delta = dx
        signed_primary_offset = primary_delta if direction == "positive" else -primary_delta
        spatial_match = signed_primary_offset >= min_primary_offset and abs(secondary_delta) <= max_abs_secondary_offset
        attribute_match = _attribute_match(semantics, target_filters)
        accepted = bool(spatial_match and attribute_match)

        candidate_geometry[object_id] = {
            "dx": dx,
            "dy": dy,
            "primary_delta": primary_delta,
            "secondary_delta": secondary_delta,
            "signed_primary_offset": signed_primary_offset,
            "spatial_match": bool(spatial_match),
            "attribute_match": bool(attribute_match),
            "accepted": bool(accepted),
        }
        if spatial_match:
            spatial_match_ids.append(object_id)
        if attribute_match:
            attribute_match_ids.append(object_id)
        if accepted:
            valid_candidate_ids.append(object_id)

    if len(valid_candidate_ids) != 1:
        raise ValueError(
            "constraint_selection runtime geometry/semantics is ambiguous: "
            f"expected exactly one accepted candidate, got {valid_candidate_ids!r}"
        )

    gt_object_id = valid_candidate_ids[0]
    if expected_target and gt_object_id != expected_target:
        raise ValueError(
            "constraint_selection runtime resolution disagrees with protocol.gt_spec.expected_target: "
            f"{gt_object_id!r} != {expected_target!r}"
        )

    return {
        "module_name": CONSTRAINT_SELECTION_MODULE_NAME,
        "query_text": str(protocol.get("query_text", "")),
        "variant": str(protocol.get("variant", "")),
        "task_version": str(protocol.get("task_version", "")),
        "anchor_object_id": anchor_object_id,
        "anchor_xy": anchor_xy,
        "candidate_object_ids": candidate_object_ids,
        "candidate_positions_xy": candidate_positions_xy,
        "candidate_semantics": candidate_semantics,
        "candidate_geometry": candidate_geometry,
        "gt_object_id": gt_object_id,
        "gt_object_index": int(candidate_object_ids.index(gt_object_id)),
        "valid_candidate_ids": valid_candidate_ids,
        "spatial_match_ids": spatial_match_ids,
        "attribute_match_ids": attribute_match_ids,
        "resolver": resolver or VISUAL_SPATIAL_CONSTRAINT_SELECTION_RESOLVER,
        "reference_frame": str(gt_spec.get("reference_frame", "world")),
        "axis": axis,
        "direction": direction,
        "min_primary_offset": min_primary_offset,
        "max_abs_secondary_offset": max_abs_secondary_offset,
        "target_filters": target_filters,
        "expected_target": expected_target or gt_object_id,
    }


def resolve_exclusion_nearest_constraint_problem(
    protocol: Mapping[str, Any],
    object_positions: Mapping[str, Any],
) -> dict[str, Any]:
    (
        anchor_object_id,
        anchor_xy,
        candidate_object_ids,
        candidate_positions_xy,
        gt_spec,
    ) = _extract_anchor_and_candidate_xy(protocol, object_positions)

    resolver = str(gt_spec.get("resolver", "")).strip()
    if resolver and resolver != EXCLUSION_NEAREST_CONSTRAINT_SELECTION_RESOLVER:
        raise ValueError(f"Unsupported constraint selection resolver: {resolver!r}")

    reference_frame = str(gt_spec.get("reference_frame", "world"))
    target_filters = _normalize_target_filters(protocol, gt_spec)
    if not target_filters:
        raise ValueError("constraint_selection protocol is missing gt_spec.target_filters.")

    exclusion_axis = _normalize_axis(gt_spec.get("exclusion_axis", gt_spec.get("axis", "x")))
    exclusion_rank = _normalize_extreme_rank(gt_spec.get("exclusion_rank", "minimum"))
    metric = _normalize_metric(gt_spec.get("metric", "planar_euclidean"))
    distance_axes = _normalize_distance_axes(gt_spec)
    rank_tie_epsilon = abs(float(gt_spec.get("rank_tie_epsilon", DEFAULT_RANK_TIE_EPSILON)))
    distance_tie_epsilon = abs(float(gt_spec.get("distance_tie_epsilon", DEFAULT_DISTANCE_TIE_EPSILON)))
    expected_target = str(gt_spec.get("expected_target", "")).strip()
    if expected_target and expected_target not in candidate_object_ids:
        raise ValueError(f"Expected target {expected_target!r} is not listed in candidate_object_ids.")

    object_semantics = _normalize_semantics_mapping(protocol)
    anchor_x, anchor_y = anchor_xy
    candidate_semantics: dict[str, dict[str, str]] = {}
    candidate_geometry: dict[str, dict[str, float | bool | int | str]] = {}
    attribute_match_ids: list[str] = []

    for object_id, position_xy in candidate_positions_xy.items():
        semantics = dict(object_semantics.get(object_id, {}))
        candidate_semantics[object_id] = semantics

        candidate_x, candidate_y = position_xy
        dx = float(candidate_x - anchor_x)
        dy = float(candidate_y - anchor_y)
        axis_value = float(candidate_x if exclusion_axis == "x" else candidate_y)
        distance_xy = float((dx * dx + dy * dy) ** 0.5)
        attribute_match = _attribute_match(semantics, target_filters)
        if attribute_match:
            attribute_match_ids.append(object_id)

        candidate_geometry[object_id] = {
            "dx": dx,
            "dy": dy,
            "axis_value": axis_value,
            "distance_xy": distance_xy,
            "attribute_match": bool(attribute_match),
            "excluded_by_rank": False,
            "spatial_match": False,
            "accepted": False,
            "extreme_rank": -1,
            "distance_rank": -1,
            "exclusion_axis": exclusion_axis,
            "exclusion_rank": exclusion_rank,
        }

    if len(attribute_match_ids) < 2:
        raise ValueError(
            "constraint_selection runtime geometry/semantics is insufficient for exclusion selection: "
            f"expected at least two attribute-matched candidates, got {attribute_match_ids!r}"
        )

    ranked_by_extreme = sorted(
        (
            float(candidate_geometry[object_id]["axis_value"]),
            object_id,
        )
        for object_id in attribute_match_ids
    )
    if exclusion_rank == "maximum":
        ranked_by_extreme = list(reversed(ranked_by_extreme))

    extreme_value = ranked_by_extreme[0][0]
    excluded_candidate_ids = [
        object_id
        for axis_value, object_id in ranked_by_extreme
        if abs(axis_value - extreme_value) <= rank_tie_epsilon
    ]
    if len(excluded_candidate_ids) != 1:
        raise ValueError(
            "constraint_selection runtime geometry/semantics is ambiguous: "
            "expected exactly one excluded candidate from the exclusion rank, "
            f"got {excluded_candidate_ids!r}"
        )

    for rank_index, (_, object_id) in enumerate(ranked_by_extreme):
        candidate_geometry[object_id]["extreme_rank"] = int(rank_index)
    excluded_candidate_id = excluded_candidate_ids[0]
    candidate_geometry[excluded_candidate_id]["excluded_by_rank"] = True

    post_exclusion_candidate_ids = [object_id for object_id in attribute_match_ids if object_id != excluded_candidate_id]
    if not post_exclusion_candidate_ids:
        raise ValueError(
            "constraint_selection runtime geometry/semantics is ambiguous: "
            "no attribute-matched candidates remain after exclusion."
        )

    ranked_by_distance = sorted(
        (
            float(candidate_geometry[object_id]["distance_xy"]),
            object_id,
        )
        for object_id in post_exclusion_candidate_ids
    )
    nearest_distance = ranked_by_distance[0][0]
    nearest_candidate_ids = [
        object_id
        for distance_xy, object_id in ranked_by_distance
        if abs(distance_xy - nearest_distance) <= distance_tie_epsilon
    ]
    if len(nearest_candidate_ids) != 1:
        raise ValueError(
            "constraint_selection runtime geometry/semantics is ambiguous: "
            "expected exactly one nearest candidate after exclusion, "
            f"got {nearest_candidate_ids!r}"
        )

    for rank_index, (_, object_id) in enumerate(ranked_by_distance):
        candidate_geometry[object_id]["distance_rank"] = int(rank_index)
        candidate_geometry[object_id]["spatial_match"] = True

    gt_object_id = nearest_candidate_ids[0]
    candidate_geometry[gt_object_id]["accepted"] = True
    if expected_target and gt_object_id != expected_target:
        ranked_distances = ", ".join(f"{object_id}:{distance_xy:.6f}" for distance_xy, object_id in ranked_by_distance)
        raise ValueError(
            "constraint_selection runtime geometry/semantics disagrees with protocol.gt_spec.expected_target: "
            f"{gt_object_id!r} != {expected_target!r}; ranked_distances=[{ranked_distances}]"
        )

    return {
        "module_name": CONSTRAINT_SELECTION_MODULE_NAME,
        "query_text": str(protocol.get("query_text", "")),
        "variant": str(protocol.get("variant", "")),
        "task_version": str(protocol.get("task_version", "")),
        "anchor_object_id": anchor_object_id,
        "anchor_xy": anchor_xy,
        "candidate_object_ids": candidate_object_ids,
        "candidate_positions_xy": candidate_positions_xy,
        "candidate_semantics": candidate_semantics,
        "candidate_geometry": candidate_geometry,
        "gt_object_id": gt_object_id,
        "gt_object_index": int(candidate_object_ids.index(gt_object_id)),
        "valid_candidate_ids": [gt_object_id],
        "spatial_match_ids": post_exclusion_candidate_ids,
        "attribute_match_ids": attribute_match_ids,
        "excluded_candidate_ids": excluded_candidate_ids,
        "distance_ranked_candidate_ids": [object_id for _, object_id in ranked_by_distance],
        "resolver": resolver or EXCLUSION_NEAREST_CONSTRAINT_SELECTION_RESOLVER,
        "reference_frame": reference_frame,
        "exclusion_axis": exclusion_axis,
        "exclusion_rank": exclusion_rank,
        "metric": metric,
        "distance_axes": list(distance_axes),
        "rank_tie_epsilon": rank_tie_epsilon,
        "distance_tie_epsilon": distance_tie_epsilon,
        "target_filters": target_filters,
        "expected_target": expected_target or gt_object_id,
    }


def build_constraint_selection_problem_from_object_positions(
    task_spec: Mapping[str, Any],
    object_positions: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = task_spec.get("protocol", {})
    if not isinstance(protocol, Mapping):
        raise ValueError("task_spec.protocol is missing or invalid.")

    gt_spec = protocol.get("gt_spec", {})
    if not isinstance(gt_spec, Mapping):
        gt_spec = {}
    resolver = str(gt_spec.get("resolver", "")).strip() or VISUAL_SPATIAL_CONSTRAINT_SELECTION_RESOLVER

    if resolver == VISUAL_SPATIAL_CONSTRAINT_SELECTION_RESOLVER:
        problem = resolve_visual_spatial_constraint_problem(protocol, object_positions)
    elif resolver == EXCLUSION_NEAREST_CONSTRAINT_SELECTION_RESOLVER:
        problem = resolve_exclusion_nearest_constraint_problem(protocol, object_positions)
    else:
        raise ValueError(f"Unsupported constraint selection resolver: {resolver!r}")
    problem["instruction"] = str(task_spec.get("instruction", "") or protocol.get("query_text", ""))
    return problem


__all__ = [
    "CONSTRAINT_SELECTION_MODULE_NAME",
    "EXCLUSION_NEAREST_CONSTRAINT_SELECTION_RESOLVER",
    "VISUAL_SPATIAL_CONSTRAINT_SELECTION_RESOLVER",
    "build_constraint_selection_problem_from_object_positions",
    "is_constraint_selection_protocol",
    "resolve_exclusion_nearest_constraint_problem",
    "resolve_visual_spatial_constraint_problem",
]
