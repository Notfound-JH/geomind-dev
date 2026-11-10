from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ANCHOR_REFERENCE_MODULE_NAME = "anchor_reference"
DIRECTIONAL_ANCHOR_SELECTION_RESOLVER = "directional_anchor_selection"
NEAREST_ANCHOR_SELECTION_RESOLVER = "nearest_anchor_selection"
DEFAULT_MIN_PRIMARY_OFFSET = 0.06
DEFAULT_MAX_ABS_SECONDARY_OFFSET = 0.03
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


def is_anchor_reference_protocol(protocol: Any) -> bool:
    return isinstance(protocol, Mapping) and str(protocol.get("module_name", "")).strip() == ANCHOR_REFERENCE_MODULE_NAME


def _normalize_axis(axis: Any) -> str:
    normalized = str(axis).strip().lower()
    if normalized not in {"x", "y"}:
        raise ValueError(f"Unsupported anchor reference axis: {axis!r}")
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
        raise ValueError(f"Unsupported anchor reference direction: {direction!r}")
    return resolved


def _extract_anchor_and_candidate_xy(
    protocol: Mapping[str, Any],
    object_positions: Mapping[str, Any],
) -> tuple[str, tuple[float, float], list[str], dict[str, tuple[float, float]], Mapping[str, Any]]:
    if not is_anchor_reference_protocol(protocol):
        raise ValueError("task protocol is missing or is not an anchor_reference protocol.")

    anchor_object = protocol.get("anchor_object", {})
    if not isinstance(anchor_object, Mapping):
        raise ValueError("anchor_reference protocol is missing anchor_object.")
    anchor_object_id = str(anchor_object.get("object_id", "")).strip()
    if not anchor_object_id:
        raise ValueError("anchor_reference protocol is missing anchor_object.object_id.")

    candidate_object_ids = _candidate_object_ids(protocol)
    if not candidate_object_ids:
        raise ValueError("anchor_reference protocol is missing candidate_object_ids.")

    gt_spec = protocol.get("gt_spec", {})
    if not isinstance(gt_spec, Mapping):
        gt_spec = {}

    anchor_xy = _extract_object_xy(object_positions, anchor_object_id)
    candidate_positions_xy = {
        object_id: _extract_object_xy(object_positions, object_id)
        for object_id in candidate_object_ids
    }
    return anchor_object_id, anchor_xy, candidate_object_ids, candidate_positions_xy, gt_spec


def resolve_directional_anchor_problem(
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
    if resolver and resolver != DIRECTIONAL_ANCHOR_SELECTION_RESOLVER:
        raise ValueError(f"Unsupported anchor reference resolver: {resolver!r}")

    axis = _normalize_axis(gt_spec.get("axis", "x"))
    direction = _normalize_direction(gt_spec.get("direction", "positive"))
    min_primary_offset = float(
        gt_spec.get(
            "min_primary_offset",
            gt_spec.get("min_dx", DEFAULT_MIN_PRIMARY_OFFSET),
        )
    )
    max_abs_secondary_offset = float(
        gt_spec.get(
            "max_abs_secondary_offset",
            gt_spec.get("max_abs_dy", DEFAULT_MAX_ABS_SECONDARY_OFFSET),
        )
    )
    expected_target = str(gt_spec.get("expected_target", "")).strip()
    if expected_target and expected_target not in candidate_positions_xy:
        raise ValueError(f"Expected target {expected_target!r} is not listed in candidate_object_ids.")

    candidate_geometry: dict[str, dict[str, float | bool]] = {}
    valid_candidate_ids: list[str] = []
    anchor_x, anchor_y = anchor_xy
    for object_id, position_xy in candidate_positions_xy.items():
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
        accepted = signed_primary_offset >= min_primary_offset and abs(secondary_delta) <= max_abs_secondary_offset
        candidate_geometry[object_id] = {
            "dx": dx,
            "dy": dy,
            "primary_delta": primary_delta,
            "secondary_delta": secondary_delta,
            "signed_primary_offset": signed_primary_offset,
            "accepted": bool(accepted),
        }
        if accepted:
            valid_candidate_ids.append(object_id)

    if len(valid_candidate_ids) != 1:
        raise ValueError(
            "anchor_reference runtime geometry is ambiguous: "
            f"expected exactly one accepted candidate, got {valid_candidate_ids!r}"
        )

    gt_object_id = valid_candidate_ids[0]
    if expected_target and gt_object_id != expected_target:
        raise ValueError(
            "anchor_reference runtime geometry disagrees with protocol.gt_spec.expected_target: "
            f"{gt_object_id!r} != {expected_target!r}"
        )

    return {
        "module_name": ANCHOR_REFERENCE_MODULE_NAME,
        "query_text": str(protocol.get("query_text", "")),
        "variant": str(protocol.get("variant", "")),
        "task_version": str(protocol.get("task_version", "")),
        "anchor_object_id": anchor_object_id,
        "anchor_xy": anchor_xy,
        "candidate_object_ids": candidate_object_ids,
        "candidate_positions_xy": candidate_positions_xy,
        "candidate_geometry": candidate_geometry,
        "gt_object_id": gt_object_id,
        "gt_object_index": int(candidate_object_ids.index(gt_object_id)),
        "valid_candidate_ids": valid_candidate_ids,
        "resolver": resolver or DIRECTIONAL_ANCHOR_SELECTION_RESOLVER,
        "reference_frame": str(gt_spec.get("reference_frame", "world")),
        "axis": axis,
        "direction": direction,
        "min_primary_offset": min_primary_offset,
        "max_abs_secondary_offset": max_abs_secondary_offset,
        "expected_target": expected_target or gt_object_id,
    }


def resolve_nearest_anchor_problem(
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
    if resolver and resolver != NEAREST_ANCHOR_SELECTION_RESOLVER:
        raise ValueError(f"Unsupported anchor reference resolver: {resolver!r}")

    reference_frame = str(gt_spec.get("reference_frame", "world"))
    metric = str(gt_spec.get("metric", "planar_euclidean")).strip().lower() or "planar_euclidean"
    if metric != "planar_euclidean":
        raise ValueError(f"Unsupported anchor reference metric: {metric!r}")

    raw_distance_axes = gt_spec.get("distance_axes", ("x", "y"))
    if not isinstance(raw_distance_axes, Sequence) or isinstance(raw_distance_axes, (str, bytes)):
        raw_distance_axes = ("x", "y")
    distance_axes = tuple(dict.fromkeys(_normalize_axis(axis) for axis in raw_distance_axes))
    if set(distance_axes) != {"x", "y"}:
        raise ValueError(
            "nearest anchor selection currently supports planar Euclidean distance on the world X/Y axes only."
        )

    expected_target = str(gt_spec.get("expected_target", "")).strip()
    if expected_target and expected_target not in candidate_positions_xy:
        raise ValueError(f"Expected target {expected_target!r} is not listed in candidate_object_ids.")

    tie_epsilon = abs(float(gt_spec.get("tie_epsilon", DEFAULT_DISTANCE_TIE_EPSILON)))
    anchor_x, anchor_y = anchor_xy
    candidate_geometry: dict[str, dict[str, float | bool | int]] = {}
    ranked_candidates: list[tuple[float, str]] = []
    for object_id, position_xy in candidate_positions_xy.items():
        candidate_x, candidate_y = position_xy
        dx = float(candidate_x - anchor_x)
        dy = float(candidate_y - anchor_y)
        distance_xy = float((dx * dx + dy * dy) ** 0.5)
        ranked_candidates.append((distance_xy, object_id))
        candidate_geometry[object_id] = {
            "dx": dx,
            "dy": dy,
            "distance_xy": distance_xy,
            "accepted": False,
            "rank": -1,
        }

    ranked_candidates.sort(key=lambda item: (item[0], item[1]))
    if not ranked_candidates:
        raise ValueError("anchor_reference runtime geometry is missing ranked candidates.")

    nearest_distance = ranked_candidates[0][0]
    nearest_ids = [object_id for distance_xy, object_id in ranked_candidates if abs(distance_xy - nearest_distance) <= tie_epsilon]
    if len(nearest_ids) != 1:
        raise ValueError(
            "anchor_reference runtime geometry is ambiguous: "
            f"expected exactly one nearest candidate within tie_epsilon={tie_epsilon:.6g}, got {nearest_ids!r}"
        )

    for rank_index, (_, object_id) in enumerate(ranked_candidates):
        candidate_geometry[object_id]["rank"] = int(rank_index)
    gt_object_id = nearest_ids[0]
    candidate_geometry[gt_object_id]["accepted"] = True

    if expected_target and gt_object_id != expected_target:
        ranked_distances = ", ".join(f"{object_id}:{distance_xy:.6f}" for distance_xy, object_id in ranked_candidates)
        raise ValueError(
            "anchor_reference runtime geometry disagrees with protocol.gt_spec.expected_target: "
            f"{gt_object_id!r} != {expected_target!r}; ranked_distances=[{ranked_distances}]"
        )

    second_nearest_distance = ranked_candidates[1][0] if len(ranked_candidates) > 1 else None
    distance_gap_to_second = (
        None if second_nearest_distance is None else float(second_nearest_distance - nearest_distance)
    )

    return {
        "module_name": ANCHOR_REFERENCE_MODULE_NAME,
        "query_text": str(protocol.get("query_text", "")),
        "variant": str(protocol.get("variant", "")),
        "task_version": str(protocol.get("task_version", "")),
        "anchor_object_id": anchor_object_id,
        "anchor_xy": anchor_xy,
        "candidate_object_ids": candidate_object_ids,
        "candidate_positions_xy": candidate_positions_xy,
        "candidate_geometry": candidate_geometry,
        "gt_object_id": gt_object_id,
        "gt_object_index": int(candidate_object_ids.index(gt_object_id)),
        "valid_candidate_ids": [gt_object_id],
        "resolver": resolver or NEAREST_ANCHOR_SELECTION_RESOLVER,
        "reference_frame": reference_frame,
        "metric": metric,
        "distance_axes": list(distance_axes),
        "tie_epsilon": tie_epsilon,
        "nearest_distance": float(nearest_distance),
        "second_nearest_distance": second_nearest_distance,
        "distance_gap_to_second": distance_gap_to_second,
        "expected_target": expected_target or gt_object_id,
    }


def build_anchor_reference_problem_from_object_positions(
    task_spec: Mapping[str, Any],
    object_positions: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = task_spec.get("protocol", {})
    if not isinstance(protocol, Mapping):
        raise ValueError("task_spec.protocol is missing or invalid.")

    gt_spec = protocol.get("gt_spec", {})
    if not isinstance(gt_spec, Mapping):
        gt_spec = {}
    resolver = str(gt_spec.get("resolver", "")).strip() or DIRECTIONAL_ANCHOR_SELECTION_RESOLVER

    if resolver == DIRECTIONAL_ANCHOR_SELECTION_RESOLVER:
        problem = resolve_directional_anchor_problem(protocol, object_positions)
    elif resolver == NEAREST_ANCHOR_SELECTION_RESOLVER:
        problem = resolve_nearest_anchor_problem(protocol, object_positions)
    else:
        raise ValueError(f"Unsupported anchor reference resolver: {resolver!r}")
    problem["instruction"] = str(task_spec.get("instruction", "") or protocol.get("query_text", ""))
    return problem


__all__ = [
    "ANCHOR_REFERENCE_MODULE_NAME",
    "DEFAULT_DISTANCE_TIE_EPSILON",
    "DIRECTIONAL_ANCHOR_SELECTION_RESOLVER",
    "DEFAULT_MAX_ABS_SECONDARY_OFFSET",
    "DEFAULT_MIN_PRIMARY_OFFSET",
    "NEAREST_ANCHOR_SELECTION_RESOLVER",
    "build_anchor_reference_problem_from_object_positions",
    "is_anchor_reference_protocol",
    "resolve_directional_anchor_problem",
    "resolve_nearest_anchor_problem",
]
