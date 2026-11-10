from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


PERFECT_MIDPOINT_TASK_ID = "perfect_midpoint"
ISOMETRIC_EXTENSION_TASK_ID = "isometric_extension"
POINT_SYMMETRY_TASK_ID = "point_symmetry"
TREND_CONTINUATION_TASK_ID = "trend_continuation"
WEAK_DESTRUCTION_CONTROL_TASK_ID = "weak_destruction_control"
SQUARE_COMPLETION_TASK_ID = "square_completion"
PAIRED_CONTROL_TASK_IDS = {
    TREND_CONTINUATION_TASK_ID: WEAK_DESTRUCTION_CONTROL_TASK_ID,
    WEAK_DESTRUCTION_CONTROL_TASK_ID: TREND_CONTINUATION_TASK_ID,
}

GEOMETRIC_INFERENCE_MODULE_NAME = "geometric_inference"
EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME = "explicit_rule_extrapolation"
IMPLICIT_TREND_FITTING_TRACK_NAME = "implicit_trend_fitting"
SHAPE_COMPLETION_TRACK_NAME = "shape_completion"
MIDPOINT_RULE_TYPE = "midpoint_placement"
EXTENSION_RULE_TYPE = "extension_placement"
PERFECT_MIDPOINT_VARIANT = "perfect_midpoint"
ISOMETRIC_EXTENSION_VARIANT = "isometric_extension"
POINT_SYMMETRY_RULE_TYPE = "point_symmetry"
POINT_SYMMETRY_VARIANT = "point_symmetry"
SQUARE_COMPLETION_RULE_TYPE = "square_completion"
PCA_TREND_RULE_TYPE = "pca_trend_continuation"
TREND_CONTINUATION_SUBTASK_NAME = "trend_continuation"
WEAK_DESTRUCTION_CONTROL_SUBTASK_NAME = "weak_destruction_control"

DEFAULT_EPSILON_POS = 0.05
DEFAULT_TAU_LINE = 0.02
DEFAULT_TAU_TREND = 0.03
DEFAULT_TAU_YAW = 0.35
DEFAULT_WORKSPACE_PADDING = 0.15
DEFAULT_MIN_WORKSPACE_SPAN = 0.3
DEFAULT_WORKSPACE_BOUNDS_XY = ((0.15, -0.35), (0.65, 0.35))


def resolve_geometric_task_id(protocol: Mapping[str, Any] | None) -> str:
    if not isinstance(protocol, Mapping):
        return PERFECT_MIDPOINT_TASK_ID

    track_name = str(protocol.get("track_name", "")).strip()
    variant = str(protocol.get("variant", "")).strip().lower()
    gt_spec = protocol.get("gt_spec", {})
    if not isinstance(gt_spec, Mapping):
        gt_spec = {}
    rule_type = str(gt_spec.get("rule_type", "")).strip().lower()

    if track_name == IMPLICIT_TREND_FITTING_TRACK_NAME:
        if variant == WEAK_DESTRUCTION_CONTROL_TASK_ID or str(protocol.get("control_mode", "")).strip().lower() == "weak_destruction":
            return WEAK_DESTRUCTION_CONTROL_TASK_ID
        return TREND_CONTINUATION_TASK_ID

    if variant in {POINT_SYMMETRY_TASK_ID, SQUARE_COMPLETION_TASK_ID}:
        return variant
    if rule_type == POINT_SYMMETRY_TASK_ID:
        return POINT_SYMMETRY_TASK_ID
    if rule_type == SQUARE_COMPLETION_TASK_ID:
        return SQUARE_COMPLETION_TASK_ID
    if rule_type in {EXTENSION_RULE_TYPE, ISOMETRIC_EXTENSION_VARIANT} or variant == ISOMETRIC_EXTENSION_TASK_ID:
        return ISOMETRIC_EXTENSION_TASK_ID
    return PERFECT_MIDPOINT_TASK_ID


def _coerce_xy(value: Any, *, field_name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise ValueError(f"{field_name} must provide at least two numeric values.")
    return (float(value[0]), float(value[1]))


def _extract_object_xy(object_positions: Mapping[str, Any], object_id: str) -> tuple[float, float]:
    payload = object_positions.get(object_id)
    if not isinstance(payload, Mapping):
        raise KeyError(f"Object '{object_id}' is missing from task_spec['object_positions'].")
    return _coerce_xy(payload.get("position"), field_name=f"object_positions[{object_id!r}].position")


def _resolve_anchor_entries(
    protocol: Mapping[str, Any] | None,
    *,
    minimum_count: int,
    task_name: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(protocol, Mapping):
        raise ValueError("task protocol is missing or invalid.")

    anchors = protocol.get("anchors", [])
    if not isinstance(anchors, list) or len(anchors) < minimum_count:
        raise ValueError(f"{task_name} require at least {minimum_count} anchor references.")

    resolved: list[tuple[str, str]] = []
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, Mapping):
            continue
        object_id = anchor.get("object_id")
        if not isinstance(object_id, str) or not object_id.strip():
            continue
        anchor_id = anchor.get("anchor_id", anchor.get("role", chr(ord("A") + index)))
        resolved.append((str(anchor_id), str(object_id)))

    if len(resolved) < minimum_count:
        raise ValueError(f"Unable to resolve at least {minimum_count} anchor object ids from task protocol.")
    return tuple(resolved)


def _extract_anchor_positions_xy(
    object_positions: Mapping[str, Any],
    anchor_entries: Sequence[tuple[str, str]],
) -> dict[str, tuple[float, float]]:
    return {
        str(anchor_id): _extract_object_xy(object_positions, str(object_id))
        for anchor_id, object_id in anchor_entries
    }


def normalize_explicit_variant(value: Any) -> str:
    normalized = str(value).strip().lower()
    alias_map = {
        "midpoint": PERFECT_MIDPOINT_VARIANT,
        MIDPOINT_RULE_TYPE: PERFECT_MIDPOINT_VARIANT,
        PERFECT_MIDPOINT_VARIANT: PERFECT_MIDPOINT_VARIANT,
        "extension": ISOMETRIC_EXTENSION_VARIANT,
        EXTENSION_RULE_TYPE: ISOMETRIC_EXTENSION_VARIANT,
        ISOMETRIC_EXTENSION_VARIANT: ISOMETRIC_EXTENSION_VARIANT,
        POINT_SYMMETRY_RULE_TYPE: POINT_SYMMETRY_VARIANT,
        POINT_SYMMETRY_VARIANT: POINT_SYMMETRY_VARIANT,
    }
    resolved = alias_map.get(normalized)
    if resolved is None:
        raise ValueError(f"Unsupported geometric inference explicit variant: {value!r}")
    return resolved


def normalize_rule_type(rule_type: Any) -> str:
    variant = normalize_explicit_variant(rule_type)
    if variant == PERFECT_MIDPOINT_VARIANT:
        return MIDPOINT_RULE_TYPE
    if variant == ISOMETRIC_EXTENSION_VARIANT:
        return EXTENSION_RULE_TYPE
    if variant == POINT_SYMMETRY_VARIANT:
        return POINT_SYMMETRY_RULE_TYPE
    raise ValueError(f"Unsupported geometric inference rule: {rule_type!r}")


def resolve_geometric_anchor_object_ids(protocol: Mapping[str, Any] | None) -> dict[str, str]:
    anchor_entries = _resolve_anchor_entries(
        protocol,
        minimum_count=2,
        task_name="geometric_inference explicit-rule tasks",
    )

    resolved = {anchor_id: object_id for anchor_id, object_id in anchor_entries}
    ordered_object_ids = [object_id for _, object_id in anchor_entries]
    if "A" not in resolved and ordered_object_ids:
        resolved["A"] = ordered_object_ids[0]
    if "B" not in resolved and len(ordered_object_ids) >= 2:
        resolved["B"] = ordered_object_ids[1]
    if "A" not in resolved or "B" not in resolved:
        raise ValueError("Unable to resolve anchor A/B object ids from task protocol.")
    return {"A": resolved["A"], "B": resolved["B"]}


def resolve_extension_anchor_object_ids(protocol: Mapping[str, Any] | None) -> dict[str, str]:
    return resolve_geometric_anchor_object_ids(protocol)


def resolve_trend_anchor_object_ids(protocol: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    return _resolve_anchor_entries(
        protocol,
        minimum_count=3,
        task_name="implicit trend fitting tasks",
    )


def resolve_square_anchor_object_ids(protocol: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    return _resolve_anchor_entries(
        protocol,
        minimum_count=3,
        task_name="shape completion tasks",
    )


def resolve_target_object_id(protocol: Mapping[str, Any] | None) -> str:
    if not isinstance(protocol, Mapping):
        return ""
    target_object = protocol.get("target_object", {})
    if not isinstance(target_object, Mapping):
        return ""
    object_id = target_object.get("object_id", "")
    return str(object_id) if isinstance(object_id, str) else ""


def solve_midpoint_gt(anchor_a_xy: Sequence[float], anchor_b_xy: Sequence[float]) -> tuple[float, float]:
    ax, ay = _coerce_xy(anchor_a_xy, field_name="anchor_a_xy")
    bx, by = _coerce_xy(anchor_b_xy, field_name="anchor_b_xy")
    return (0.5 * (ax + bx), 0.5 * (ay + by))


def solve_extension_gt(anchor_a_xy: Sequence[float], anchor_b_xy: Sequence[float]) -> tuple[float, float]:
    ax, ay = _coerce_xy(anchor_a_xy, field_name="anchor_a_xy")
    bx, by = _coerce_xy(anchor_b_xy, field_name="anchor_b_xy")
    return (bx + (bx - ax), by + (by - ay))


def solve_point_symmetry_gt(center_xy: Sequence[float], reference_xy: Sequence[float]) -> tuple[float, float]:
    cx, cy = _coerce_xy(center_xy, field_name="center_xy")
    rx, ry = _coerce_xy(reference_xy, field_name="reference_xy")
    return (2.0 * cx - rx, 2.0 * cy - ry)


def solve_square_completion_gt(
    shared_corner_xy: Sequence[float],
    adjacent_corner_b_xy: Sequence[float],
    adjacent_corner_c_xy: Sequence[float],
) -> dict[str, Any]:
    ax, ay = _coerce_xy(shared_corner_xy, field_name="shared_corner_xy")
    bx, by = _coerce_xy(adjacent_corner_b_xy, field_name="adjacent_corner_b_xy")
    cx, cy = _coerce_xy(adjacent_corner_c_xy, field_name="adjacent_corner_c_xy")

    gt_pos_xy = (bx + cx - ax, by + cy - ay)
    center_xy = (0.5 * (ax + gt_pos_xy[0]), 0.5 * (ay + gt_pos_xy[1]))

    edge_ab = math.hypot(bx - ax, by - ay)
    edge_ac = math.hypot(cx - ax, cy - ay)
    denom = edge_ab * edge_ac
    if denom <= 1.0e-8:
        shared_corner_cosine = 1.0
    else:
        shared_corner_cosine = abs(((bx - ax) * (cx - ax) + (by - ay) * (cy - ay)) / denom)

    gt_yaw_rad = math.atan2(center_xy[1] - gt_pos_xy[1], center_xy[0] - gt_pos_xy[0])
    return {
        "gt_pos_xy": (float(gt_pos_xy[0]), float(gt_pos_xy[1])),
        "center_xy": (float(center_xy[0]), float(center_xy[1])),
        "edge_length_shared_b": float(edge_ab),
        "edge_length_shared_c": float(edge_ac),
        "shared_corner_cosine": float(shared_corner_cosine),
        "gt_yaw_rad": float(gt_yaw_rad),
    }


def solve_explicit_rule_gt(
    rule_type: str,
    anchor_a_xy: Sequence[float],
    anchor_b_xy: Sequence[float],
) -> tuple[float, float]:
    normalized_rule_type = normalize_rule_type(rule_type)
    if normalized_rule_type == MIDPOINT_RULE_TYPE:
        return solve_midpoint_gt(anchor_a_xy, anchor_b_xy)
    if normalized_rule_type == EXTENSION_RULE_TYPE:
        return solve_extension_gt(anchor_a_xy, anchor_b_xy)
    if normalized_rule_type == POINT_SYMMETRY_RULE_TYPE:
        return solve_point_symmetry_gt(anchor_a_xy, anchor_b_xy)
    raise ValueError(f"Unsupported geometric inference rule: {rule_type!r}")


def solve_trend_continuation_gt(anchor_positions_xy: Sequence[Sequence[float]]) -> dict[str, Any]:
    if len(anchor_positions_xy) < 3:
        raise ValueError("trend continuation requires at least three anchor positions.")

    points = [_coerce_xy(point_xy, field_name=f"anchor_positions_xy[{index}]") for index, point_xy in enumerate(anchor_positions_xy)]
    num_points = len(points)
    centroid_x = sum(point[0] for point in points) / float(num_points)
    centroid_y = sum(point[1] for point in points) / float(num_points)

    centered_points = [(point[0] - centroid_x, point[1] - centroid_y) for point in points]
    cov_xx = sum(point[0] * point[0] for point in centered_points) / float(num_points)
    cov_xy = sum(point[0] * point[1] for point in centered_points) / float(num_points)
    cov_yy = sum(point[1] * point[1] for point in centered_points) / float(num_points)

    if abs(cov_xy) <= 1.0e-10:
        principal_axis = (1.0, 0.0) if cov_xx >= cov_yy else (0.0, 1.0)
    else:
        trace = cov_xx + cov_yy
        delta = math.sqrt(max((cov_xx - cov_yy) ** 2 + 4.0 * cov_xy * cov_xy, 0.0))
        largest_eigenvalue = 0.5 * (trace + delta)
        axis_x = cov_xy
        axis_y = largest_eigenvalue - cov_xx
        axis_norm = math.hypot(axis_x, axis_y)
        if axis_norm <= 1.0e-10:
            principal_axis = (1.0, 0.0) if cov_xx >= cov_yy else (0.0, 1.0)
        else:
            principal_axis = (axis_x / axis_norm, axis_y / axis_norm)

    order_direction = (points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    if (order_direction[0] * principal_axis[0] + order_direction[1] * principal_axis[1]) < 0.0:
        principal_axis = (-principal_axis[0], -principal_axis[1])

    projections = [
        centered_point[0] * principal_axis[0] + centered_point[1] * principal_axis[1]
        for centered_point in centered_points
    ]
    min_projection = min(projections)
    max_projection = max(projections)
    projected_span = max_projection - min_projection
    if projected_span <= 1.0e-8:
        raise ValueError("trend continuation anchors are degenerate along the PCA principal axis.")

    projected_step = projected_span / float(num_points - 1)
    terminal_anchor_index = max(range(num_points), key=lambda index: (projections[index], index))
    terminal_anchor_xy = points[terminal_anchor_index]
    gt_pos_xy = (
        terminal_anchor_xy[0] + projected_step * principal_axis[0],
        terminal_anchor_xy[1] + projected_step * principal_axis[1],
    )

    return {
        "gt_pos_xy": gt_pos_xy,
        "principal_axis": principal_axis,
        "projected_step": float(projected_step),
        "terminal_anchor_index": int(terminal_anchor_index),
        "terminal_anchor_xy": terminal_anchor_xy,
        "centroid_xy": (float(centroid_x), float(centroid_y)),
        "projections": tuple(float(value) for value in projections),
    }


def point_to_line_distance_xy(
    point_xy: Sequence[float],
    anchor_a_xy: Sequence[float],
    anchor_b_xy: Sequence[float],
) -> float:
    px, py = _coerce_xy(point_xy, field_name="point_xy")
    ax, ay = _coerce_xy(anchor_a_xy, field_name="anchor_a_xy")
    bx, by = _coerce_xy(anchor_b_xy, field_name="anchor_b_xy")
    abx = bx - ax
    aby = by - ay
    ab_norm = math.hypot(abx, aby)
    if ab_norm <= 1.0e-8:
        return math.hypot(px - bx, py - by)
    return abs(abx * (ay - py) - (ax - px) * aby) / ab_norm


def projection_ratio_xy(
    point_xy: Sequence[float],
    anchor_a_xy: Sequence[float],
    anchor_b_xy: Sequence[float],
) -> float:
    px, py = _coerce_xy(point_xy, field_name="point_xy")
    ax, ay = _coerce_xy(anchor_a_xy, field_name="anchor_a_xy")
    bx, by = _coerce_xy(anchor_b_xy, field_name="anchor_b_xy")
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1.0e-8:
        return 0.0
    apx = px - ax
    apy = py - ay
    return float((apx * abx + apy * aby) / denom)


def l2_distance_xy(
    point_a_xy: Sequence[float],
    point_b_xy: Sequence[float],
) -> float:
    ax, ay = _coerce_xy(point_a_xy, field_name="point_a_xy")
    bx, by = _coerce_xy(point_b_xy, field_name="point_b_xy")
    return float(math.hypot(ax - bx, ay - by))


def wrap_angle_pi(angle_rad: float) -> float:
    wrapped = (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped <= -math.pi:
        wrapped += 2.0 * math.pi
    return float(wrapped)


def yaw_error_mod_pi(pred_yaw_rad: float, gt_yaw_rad: float) -> float:
    delta = abs(wrap_angle_pi(float(pred_yaw_rad) - float(gt_yaw_rad)))
    return float(min(delta, abs(math.pi - delta)))


def direction_alignment_xy(
    point_xy: Sequence[float],
    anchor_a_xy: Sequence[float],
    anchor_b_xy: Sequence[float],
) -> float:
    px, py = _coerce_xy(point_xy, field_name="point_xy")
    ax, ay = _coerce_xy(anchor_a_xy, field_name="anchor_a_xy")
    bx, by = _coerce_xy(anchor_b_xy, field_name="anchor_b_xy")
    return float((px - bx) * (bx - ax) + (py - by) * (by - ay))


def _finalize_rule_readout(
    *,
    l2_error: float,
    orthogonal_distance: float,
    rule_satisfied: bool,
    epsilon_pos: float,
    tau_metric_name: str,
    tau_metric_value: float,
    extra_metrics: Mapping[str, Any] | None = None,
    additional_success_conditions: Sequence[bool] | None = None,
) -> dict[str, Any]:
    success = float(l2_error) < float(epsilon_pos) and bool(rule_satisfied)
    if additional_success_conditions is not None:
        success = success and all(bool(value) for value in additional_success_conditions)
    result = {
        "l2_error": float(l2_error),
        "orthogonal_distance": float(orthogonal_distance),
        "rule_satisfied": bool(rule_satisfied),
        "success": bool(success),
        "epsilon_pos": float(epsilon_pos),
        str(tau_metric_name): float(tau_metric_value),
    }
    if tau_metric_name == "tau_orthogonal":
        result["tau_line"] = float(tau_metric_value)
    if extra_metrics:
        result.update(dict(extra_metrics))
    return result


def _build_common_debug_result(
    problem: Mapping[str, Any],
    *,
    pred_pos_xy: Sequence[float],
    readout: Mapping[str, Any],
    valid_episode: bool,
) -> dict[str, Any]:
    track_name = str(problem.get("track_name", ""))
    anchor_positions_xy = problem.get("anchor_positions_xy", {})
    if not isinstance(anchor_positions_xy, Mapping):
        raise ValueError("problem['anchor_positions_xy'] is missing or invalid.")

    serialized_anchor_positions_xy = {
        str(anchor_id): [float(value) for value in _coerce_xy(anchor_xy, field_name=f"anchor_positions_xy.{anchor_id}")]
        for anchor_id, anchor_xy in anchor_positions_xy.items()
    }

    result = {
        "module_name": str(problem.get("module_name", "")),
        "track_name": track_name,
        "family": str(problem.get("family", track_name)),
        "variant": str(problem.get("variant", problem.get("subtask_name", ""))),
        "query_family": str(problem.get("query_family", "")),
        "query_text": str(problem.get("query_text", "")),
        "subtask_name": str(problem.get("subtask_name", "")),
        "rule_type": str(problem.get("rule_type", "")),
        "anchor_positions_xy": serialized_anchor_positions_xy,
        "gt_pos_xy": [float(value) for value in _coerce_xy(problem.get("gt_pos_xy", ()), field_name="gt_pos_xy")],
        "pred_pos_xy": [float(value) for value in _coerce_xy(pred_pos_xy, field_name="pred_pos_xy")],
        "l2_error": float(readout["l2_error"]),
        "orthogonal_distance": float(readout["orthogonal_distance"]),
        "rule_satisfied": bool(readout["rule_satisfied"]),
        "success": bool(valid_episode) and bool(readout["success"]),
        "valid_episode": bool(valid_episode),
    }
    control_mode = str(problem.get("control_mode", ""))
    noise_level = str(problem.get("noise_level", ""))
    if control_mode:
        result["control_mode"] = control_mode
    if noise_level:
        result["noise_level"] = noise_level
    return result


def evaluate_trend_readout(
    *,
    axis_origin_xy: Sequence[float],
    principal_axis: Sequence[float],
    gt_pos_xy: Sequence[float],
    pred_pos_xy: Sequence[float],
    epsilon_pos: float = DEFAULT_EPSILON_POS,
    tau_trend: float = DEFAULT_TAU_TREND,
) -> dict[str, Any]:
    origin_x, origin_y = _coerce_xy(axis_origin_xy, field_name="axis_origin_xy")
    axis_x, axis_y = _coerce_xy(principal_axis, field_name="principal_axis")
    axis_norm = math.hypot(axis_x, axis_y)
    if axis_norm <= 1.0e-8:
        raise ValueError("principal_axis must be non-zero.")
    unit_axis = (axis_x / axis_norm, axis_y / axis_norm)

    l2_error = l2_distance_xy(gt_pos_xy, pred_pos_xy)
    orthogonal_distance = point_to_line_distance_xy(
        pred_pos_xy,
        (origin_x, origin_y),
        (origin_x + unit_axis[0], origin_y + unit_axis[1]),
    )
    trend_consistent = orthogonal_distance < float(tau_trend)
    return _finalize_rule_readout(
        l2_error=l2_error,
        orthogonal_distance=orthogonal_distance,
        rule_satisfied=trend_consistent,
        epsilon_pos=epsilon_pos,
        tau_metric_name="tau_trend",
        tau_metric_value=tau_trend,
        extra_metrics={
            "trend_consistent": bool(trend_consistent),
        },
    )


def evaluate_midpoint_readout(
    *,
    anchor_a_xy: Sequence[float],
    anchor_b_xy: Sequence[float],
    gt_pos_xy: Sequence[float],
    pred_pos_xy: Sequence[float],
    epsilon_pos: float = DEFAULT_EPSILON_POS,
    tau_line: float = DEFAULT_TAU_LINE,
) -> dict[str, Any]:
    l2_error = l2_distance_xy(gt_pos_xy, pred_pos_xy)
    orthogonal_distance = point_to_line_distance_xy(pred_pos_xy, anchor_a_xy, anchor_b_xy)
    between_score = projection_ratio_xy(pred_pos_xy, anchor_a_xy, anchor_b_xy)
    rule_satisfied = orthogonal_distance < float(tau_line) and 0.0 <= between_score <= 1.0
    return _finalize_rule_readout(
        l2_error=l2_error,
        orthogonal_distance=orthogonal_distance,
        rule_satisfied=rule_satisfied,
        epsilon_pos=epsilon_pos,
        tau_metric_name="tau_orthogonal",
        tau_metric_value=tau_line,
        extra_metrics={
            "between_score": float(between_score),
        },
    )


def evaluate_extension_readout(
    *,
    anchor_a_xy: Sequence[float],
    anchor_b_xy: Sequence[float],
    gt_pos_xy: Sequence[float],
    pred_pos_xy: Sequence[float],
    epsilon_pos: float = DEFAULT_EPSILON_POS,
    tau_line: float = DEFAULT_TAU_LINE,
) -> dict[str, Any]:
    l2_error = l2_distance_xy(gt_pos_xy, pred_pos_xy)
    orthogonal_distance = point_to_line_distance_xy(pred_pos_xy, anchor_a_xy, anchor_b_xy)
    direction_correct = direction_alignment_xy(pred_pos_xy, anchor_a_xy, anchor_b_xy) > 0.0
    rule_satisfied = orthogonal_distance < float(tau_line) and direction_correct
    return _finalize_rule_readout(
        l2_error=l2_error,
        orthogonal_distance=orthogonal_distance,
        rule_satisfied=rule_satisfied,
        epsilon_pos=epsilon_pos,
        tau_metric_name="tau_orthogonal",
        tau_metric_value=tau_line,
        extra_metrics={
            "direction_correct": bool(direction_correct),
        },
    )


def evaluate_point_symmetry_readout(
    *,
    center_xy: Sequence[float],
    reference_xy: Sequence[float],
    gt_pos_xy: Sequence[float],
    pred_pos_xy: Sequence[float],
    epsilon_pos: float = DEFAULT_EPSILON_POS,
    tau_line: float = DEFAULT_TAU_LINE,
) -> dict[str, Any]:
    l2_error = l2_distance_xy(gt_pos_xy, pred_pos_xy)
    orthogonal_distance = point_to_line_distance_xy(pred_pos_xy, center_xy, reference_xy)
    mirrored_distance_error = abs(
        l2_distance_xy(pred_pos_xy, center_xy) - l2_distance_xy(reference_xy, center_xy)
    )
    opposite_side_score = projection_ratio_xy(pred_pos_xy, center_xy, reference_xy)
    opposite_side = opposite_side_score <= 0.0
    rule_satisfied = (
        orthogonal_distance < float(tau_line)
        and opposite_side
        and mirrored_distance_error < float(epsilon_pos)
    )
    return _finalize_rule_readout(
        l2_error=l2_error,
        orthogonal_distance=orthogonal_distance,
        rule_satisfied=rule_satisfied,
        epsilon_pos=epsilon_pos,
        tau_metric_name="tau_orthogonal",
        tau_metric_value=tau_line,
        extra_metrics={
            "mirrored_distance_error": float(mirrored_distance_error),
            "opposite_side": bool(opposite_side),
            "opposite_side_score": float(opposite_side_score),
        },
    )


def evaluate_square_completion_readout(
    *,
    shared_corner_xy: Sequence[float],
    adjacent_corner_b_xy: Sequence[float],
    adjacent_corner_c_xy: Sequence[float],
    gt_pos_xy: Sequence[float],
    pred_pos_xy: Sequence[float],
    epsilon_pos: float = DEFAULT_EPSILON_POS,
    tau_line: float = DEFAULT_TAU_LINE,
    gt_yaw_rad: float | None = None,
    pred_yaw_rad: float | None = None,
    tau_yaw: float = DEFAULT_TAU_YAW,
) -> dict[str, Any]:
    ax, ay = _coerce_xy(shared_corner_xy, field_name="shared_corner_xy")
    bx, by = _coerce_xy(adjacent_corner_b_xy, field_name="adjacent_corner_b_xy")
    cx, cy = _coerce_xy(adjacent_corner_c_xy, field_name="adjacent_corner_c_xy")
    px, py = _coerce_xy(pred_pos_xy, field_name="pred_pos_xy")

    l2_error = l2_distance_xy(gt_pos_xy, pred_pos_xy)
    midpoint_target = (0.5 * (bx + cx), 0.5 * (by + cy))
    midpoint_pred = (0.5 * (ax + px), 0.5 * (ay + py))
    diagonal_midpoint_error = l2_distance_xy(midpoint_pred, midpoint_target)

    edge_ab = math.hypot(bx - ax, by - ay)
    edge_ac = math.hypot(cx - ax, cy - ay)
    edge_pb = math.hypot(px - bx, py - by)
    edge_pc = math.hypot(px - cx, py - cy)
    edge_length_error = 0.5 * (abs(edge_pb - edge_ac) + abs(edge_pc - edge_ab))

    pred_vec_b = (bx - px, by - py)
    pred_vec_c = (cx - px, cy - py)
    pred_vec_b_norm = math.hypot(pred_vec_b[0], pred_vec_b[1])
    pred_vec_c_norm = math.hypot(pred_vec_c[0], pred_vec_c[1])
    if pred_vec_b_norm <= 1.0e-8 or pred_vec_c_norm <= 1.0e-8:
        square_corner_cosine = 1.0
    else:
        square_corner_cosine = abs(
            (pred_vec_b[0] * pred_vec_c[0] + pred_vec_b[1] * pred_vec_c[1])
            / (pred_vec_b_norm * pred_vec_c_norm)
        )

    yaw_satisfied = True
    yaw_error = None
    if gt_yaw_rad is not None and pred_yaw_rad is not None:
        yaw_error = yaw_error_mod_pi(float(pred_yaw_rad), float(gt_yaw_rad))
        yaw_satisfied = yaw_error < float(tau_yaw)

    rule_satisfied = (
        diagonal_midpoint_error < float(tau_line)
        and edge_length_error < float(epsilon_pos)
        and square_corner_cosine < 0.25
    )
    readout = _finalize_rule_readout(
        l2_error=l2_error,
        orthogonal_distance=diagonal_midpoint_error,
        rule_satisfied=rule_satisfied,
        epsilon_pos=epsilon_pos,
        tau_metric_name="tau_orthogonal",
        tau_metric_value=tau_line,
        extra_metrics={
            "diagonal_midpoint_error": float(diagonal_midpoint_error),
            "edge_length_error": float(edge_length_error),
            "square_corner_cosine": float(square_corner_cosine),
        },
        additional_success_conditions=(yaw_satisfied,),
    )
    if yaw_error is not None:
        readout["yaw_error"] = float(yaw_error)
        readout["yaw_satisfied"] = bool(yaw_satisfied)
        readout["tau_yaw"] = float(tau_yaw)
    return readout


def evaluate_explicit_rule_readout(
    *,
    rule_type: str,
    anchor_a_xy: Sequence[float],
    anchor_b_xy: Sequence[float],
    gt_pos_xy: Sequence[float],
    pred_pos_xy: Sequence[float],
    epsilon_pos: float = DEFAULT_EPSILON_POS,
    tau_line: float = DEFAULT_TAU_LINE,
) -> dict[str, Any]:
    normalized_rule_type = normalize_rule_type(rule_type)
    if normalized_rule_type == MIDPOINT_RULE_TYPE:
        return evaluate_midpoint_readout(
            anchor_a_xy=anchor_a_xy,
            anchor_b_xy=anchor_b_xy,
            gt_pos_xy=gt_pos_xy,
            pred_pos_xy=pred_pos_xy,
            epsilon_pos=epsilon_pos,
            tau_line=tau_line,
        )
    if normalized_rule_type == EXTENSION_RULE_TYPE:
        return evaluate_extension_readout(
            anchor_a_xy=anchor_a_xy,
            anchor_b_xy=anchor_b_xy,
            gt_pos_xy=gt_pos_xy,
            pred_pos_xy=pred_pos_xy,
            epsilon_pos=epsilon_pos,
            tau_line=tau_line,
        )
    if normalized_rule_type == POINT_SYMMETRY_RULE_TYPE:
        return evaluate_point_symmetry_readout(
            center_xy=anchor_a_xy,
            reference_xy=anchor_b_xy,
            gt_pos_xy=gt_pos_xy,
            pred_pos_xy=pred_pos_xy,
            epsilon_pos=epsilon_pos,
            tau_line=tau_line,
        )
    raise ValueError(f"Unsupported geometric inference rule: {rule_type!r}")


def build_workspace_bounds_xy_from_points(
    points_xy: Sequence[Sequence[float]],
    *,
    padding: float = DEFAULT_WORKSPACE_PADDING,
    min_span: float = DEFAULT_MIN_WORKSPACE_SPAN,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not points_xy:
        return DEFAULT_WORKSPACE_BOUNDS_XY

    xs = [float(point[0]) for point in points_xy]
    ys = [float(point[1]) for point in points_xy]
    min_x = min(xs) - float(padding)
    max_x = max(xs) + float(padding)
    min_y = min(ys) - float(padding)
    max_y = max(ys) + float(padding)

    if (max_x - min_x) < float(min_span):
        center_x = 0.5 * (min_x + max_x)
        half_span_x = 0.5 * float(min_span)
        min_x = center_x - half_span_x
        max_x = center_x + half_span_x
    if (max_y - min_y) < float(min_span):
        center_y = 0.5 * (min_y + max_y)
        half_span_y = 0.5 * float(min_span)
        min_y = center_y - half_span_y
        max_y = center_y + half_span_y

    return ((float(min_x), float(min_y)), (float(max_x), float(max_y)))


def build_workspace_bounds_xy(
    object_positions: Mapping[str, Any] | None,
    *,
    padding: float = DEFAULT_WORKSPACE_PADDING,
    min_span: float = DEFAULT_MIN_WORKSPACE_SPAN,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not isinstance(object_positions, Mapping):
        return DEFAULT_WORKSPACE_BOUNDS_XY

    points_xy: list[tuple[float, float]] = []
    for payload in object_positions.values():
        if not isinstance(payload, Mapping):
            continue
        position = payload.get("position")
        if isinstance(position, Sequence) and not isinstance(position, (str, bytes)) and len(position) >= 2:
            points_xy.append((float(position[0]), float(position[1])))
    return build_workspace_bounds_xy_from_points(points_xy, padding=padding, min_span=min_span)


def _build_explicit_rule_problem_from_object_positions(
    task_spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    object_positions: Mapping[str, Any],
) -> dict[str, Any]:
    gt_spec = protocol.get("gt_spec", {})
    raw_rule_type = str(gt_spec.get("rule_type", "")) if isinstance(gt_spec, Mapping) else ""
    raw_variant = str(protocol.get("variant", protocol.get("subtask_name", raw_rule_type)))
    variant = normalize_explicit_variant(raw_variant or raw_rule_type)
    rule_type = normalize_rule_type(raw_rule_type or raw_variant)
    anchor_object_ids = resolve_geometric_anchor_object_ids(protocol)
    anchor_positions_xy = _extract_anchor_positions_xy(object_positions, tuple(anchor_object_ids.items()))
    gt_pos_xy = solve_explicit_rule_gt(rule_type, anchor_positions_xy["A"], anchor_positions_xy["B"])

    return {
        "module_name": GEOMETRIC_INFERENCE_MODULE_NAME,
        "track_name": EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME,
        "family": str(protocol.get("family", EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME)),
        "variant": variant,
        "query_family": str(protocol.get("query_family", "")),
        "query_text": str(protocol.get("query_text", task_spec.get("instruction", ""))),
        "subtask_name": str(protocol.get("subtask_name", variant)),
        "rule_type": rule_type,
        "anchor_object_ids": anchor_object_ids,
        "anchor_positions_xy": anchor_positions_xy,
        "target_object_id": resolve_target_object_id(protocol),
        "gt_pos_xy": gt_pos_xy,
        "control_mode": str(protocol.get("control_mode", "deterministic_rule")),
        "noise_level": str(protocol.get("noise_level", "zero_noise")),
        "workspace_bounds_xy": build_workspace_bounds_xy(object_positions),
    }


def _build_implicit_trend_problem_from_object_positions(
    task_spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    object_positions: Mapping[str, Any],
) -> dict[str, Any]:
    anchor_entries = resolve_trend_anchor_object_ids(protocol)
    anchor_object_ids = {anchor_id: object_id for anchor_id, object_id in anchor_entries}
    anchor_positions_xy = _extract_anchor_positions_xy(object_positions, anchor_entries)
    solved_gt = solve_trend_continuation_gt(
        [anchor_positions_xy[anchor_id] for anchor_id, _ in anchor_entries]
    )
    terminal_anchor_id, terminal_object_id = anchor_entries[int(solved_gt["terminal_anchor_index"])]
    gt_spec = protocol.get("gt_spec", {})
    rule_type = (
        str(gt_spec.get("rule_type", PCA_TREND_RULE_TYPE))
        if isinstance(gt_spec, Mapping)
        else PCA_TREND_RULE_TYPE
    )

    return {
        "module_name": GEOMETRIC_INFERENCE_MODULE_NAME,
        "track_name": IMPLICIT_TREND_FITTING_TRACK_NAME,
        "family": str(protocol.get("family", IMPLICIT_TREND_FITTING_TRACK_NAME)),
        "variant": str(protocol.get("variant", protocol.get("subtask_name", TREND_CONTINUATION_SUBTASK_NAME))),
        "query_family": str(protocol.get("query_family", "")),
        "query_text": str(protocol.get("query_text", task_spec.get("instruction", ""))),
        "subtask_name": str(protocol.get("subtask_name", TREND_CONTINUATION_SUBTASK_NAME)),
        "rule_type": rule_type,
        "anchor_object_ids": anchor_object_ids,
        "anchor_positions_xy": anchor_positions_xy,
        "anchor_order": [anchor_id for anchor_id, _ in anchor_entries],
        "target_object_id": resolve_target_object_id(protocol),
        "gt_pos_xy": solved_gt["gt_pos_xy"],
        "principal_axis": solved_gt["principal_axis"],
        "projected_step": float(solved_gt["projected_step"]),
        "centroid_xy": solved_gt["centroid_xy"],
        "terminal_anchor": {
            "anchor_id": str(terminal_anchor_id),
            "object_id": str(terminal_object_id),
            "xy": solved_gt["terminal_anchor_xy"],
        },
        "control_mode": str(protocol.get("control_mode", "")),
        "noise_level": str(protocol.get("noise_level", "")),
        "workspace_bounds_xy": build_workspace_bounds_xy(object_positions),
    }


def _build_square_completion_problem_from_object_positions(
    task_spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    object_positions: Mapping[str, Any],
) -> dict[str, Any]:
    anchor_entries = resolve_square_anchor_object_ids(protocol)
    anchor_object_ids = {anchor_id: object_id for anchor_id, object_id in anchor_entries}
    anchor_positions_xy = _extract_anchor_positions_xy(object_positions, anchor_entries)
    shared_anchor_id, _ = anchor_entries[0]
    adjacent_anchor_b_id, _ = anchor_entries[1]
    adjacent_anchor_c_id, _ = anchor_entries[2]
    solved_gt = solve_square_completion_gt(
        anchor_positions_xy[shared_anchor_id],
        anchor_positions_xy[adjacent_anchor_b_id],
        anchor_positions_xy[adjacent_anchor_c_id],
    )

    gt_spec = protocol.get("gt_spec", {})
    gt_spec_mapping = gt_spec if isinstance(gt_spec, Mapping) else {}
    yaw_required = bool(
        gt_spec_mapping.get("require_principal_axis_to_center", False)
        or str(gt_spec_mapping.get("yaw_alignment_mode", "")).strip()
    )

    result = {
        "module_name": GEOMETRIC_INFERENCE_MODULE_NAME,
        "track_name": SHAPE_COMPLETION_TRACK_NAME,
        "family": str(protocol.get("family", SHAPE_COMPLETION_TRACK_NAME)),
        "variant": str(protocol.get("variant", protocol.get("subtask_name", SQUARE_COMPLETION_TASK_ID))),
        "query_family": str(protocol.get("query_family", "")),
        "query_text": str(protocol.get("query_text", task_spec.get("instruction", ""))),
        "subtask_name": str(protocol.get("subtask_name", SQUARE_COMPLETION_TASK_ID)),
        "rule_type": SQUARE_COMPLETION_RULE_TYPE,
        "anchor_object_ids": anchor_object_ids,
        "anchor_positions_xy": anchor_positions_xy,
        "anchor_order": [anchor_id for anchor_id, _ in anchor_entries],
        "target_object_id": resolve_target_object_id(protocol),
        "gt_pos_xy": solved_gt["gt_pos_xy"],
        "square_center_xy": solved_gt["center_xy"],
        "edge_length_shared_b": float(solved_gt["edge_length_shared_b"]),
        "edge_length_shared_c": float(solved_gt["edge_length_shared_c"]),
        "shared_corner_cosine": float(solved_gt["shared_corner_cosine"]),
        "control_mode": str(protocol.get("control_mode", "deterministic_completion")),
        "noise_level": str(protocol.get("noise_level", "zero_noise")),
        "workspace_bounds_xy": build_workspace_bounds_xy(object_positions),
    }
    if yaw_required:
        result["gt_yaw_rad"] = float(solved_gt["gt_yaw_rad"])
        result["yaw_alignment_mode"] = str(gt_spec_mapping.get("yaw_alignment_mode", "principal_axis_to_center"))
        result["yaw_period_rad"] = float(gt_spec_mapping.get("yaw_period_rad", math.pi))
    return result


def _build_geometric_inference_problem_from_object_positions(
    task_spec: Mapping[str, Any],
    object_positions: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = task_spec.get("protocol", {})
    if not isinstance(protocol, Mapping):
        raise ValueError("task_spec does not contain a valid protocol payload.")

    module_name = str(protocol.get("module_name", ""))
    track_name = str(protocol.get("track_name", ""))
    if module_name != GEOMETRIC_INFERENCE_MODULE_NAME:
        raise ValueError(f"Unsupported protocol module: {module_name!r}")
    if track_name == EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME:
        return _build_explicit_rule_problem_from_object_positions(task_spec, protocol, object_positions)
    if track_name == IMPLICIT_TREND_FITTING_TRACK_NAME:
        return _build_implicit_trend_problem_from_object_positions(task_spec, protocol, object_positions)
    if track_name == SHAPE_COMPLETION_TRACK_NAME:
        return _build_square_completion_problem_from_object_positions(task_spec, protocol, object_positions)
    raise ValueError(f"Unsupported protocol track: {track_name!r}")


def build_geometric_inference_problem(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    object_positions = task_spec.get("object_positions", {})
    if not isinstance(object_positions, Mapping):
        raise ValueError("task_spec['object_positions'] is missing or invalid.")
    return _build_geometric_inference_problem_from_object_positions(task_spec, object_positions)


def build_geometric_inference_problem_from_object_positions(
    task_spec: Mapping[str, Any],
    object_positions: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(object_positions, Mapping):
        raise ValueError("object_positions is missing or invalid.")
    return _build_geometric_inference_problem_from_object_positions(task_spec, object_positions)


def build_geometric_inference_debug_result(
    problem: Mapping[str, Any],
    *,
    target_pos_xy: Sequence[float] | None = None,
    pred_pos_xy: Sequence[float],
    epsilon_pos: float = DEFAULT_EPSILON_POS,
    tau_line: float = DEFAULT_TAU_LINE,
    tau_trend: float = DEFAULT_TAU_TREND,
    target_yaw: float | None = None,
    pred_yaw: float | None = None,
    tau_yaw: float = DEFAULT_TAU_YAW,
    valid_episode: bool = True,
    placement_execution_success: bool | None = None,
    settle_applied: bool | None = None,
) -> dict[str, Any]:
    track_name = str(problem.get("track_name", ""))
    anchor_positions_xy = problem.get("anchor_positions_xy", {})
    if not isinstance(anchor_positions_xy, Mapping):
        raise ValueError("problem['anchor_positions_xy'] is missing or invalid.")

    if track_name == EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME:
        anchor_a_xy = anchor_positions_xy.get("A")
        anchor_b_xy = anchor_positions_xy.get("B")
        if anchor_a_xy is None or anchor_b_xy is None:
            raise ValueError("problem is missing anchor A/B positions.")

        rule_type = normalize_rule_type(problem.get("rule_type", ""))
        readout = evaluate_explicit_rule_readout(
            rule_type=rule_type,
            anchor_a_xy=anchor_a_xy,
            anchor_b_xy=anchor_b_xy,
            gt_pos_xy=problem.get("gt_pos_xy", ()),
            pred_pos_xy=pred_pos_xy,
            epsilon_pos=epsilon_pos,
            tau_line=tau_line,
        )
        result = _build_common_debug_result(
            problem,
            pred_pos_xy=pred_pos_xy,
            readout=readout,
            valid_episode=valid_episode,
        )
        anchor_object_ids = problem.get("anchor_object_ids", {})
        if isinstance(anchor_object_ids, Mapping):
            result["anchor_a"] = {
                "anchor_id": "A",
                "object_id": str(anchor_object_ids.get("A", "")),
                "xy": [float(value) for value in _coerce_xy(anchor_a_xy, field_name="anchor_a_xy")],
            }
            result["anchor_b"] = {
                "anchor_id": "B",
                "object_id": str(anchor_object_ids.get("B", "")),
                "xy": [float(value) for value in _coerce_xy(anchor_b_xy, field_name="anchor_b_xy")],
            }
    elif track_name == IMPLICIT_TREND_FITTING_TRACK_NAME:
        readout = evaluate_trend_readout(
            axis_origin_xy=problem.get("centroid_xy", ()),
            principal_axis=problem.get("principal_axis", ()),
            gt_pos_xy=problem.get("gt_pos_xy", ()),
            pred_pos_xy=pred_pos_xy,
            epsilon_pos=epsilon_pos,
            tau_trend=tau_trend,
        )
        result = _build_common_debug_result(
            problem,
            pred_pos_xy=pred_pos_xy,
            readout=readout,
            valid_episode=valid_episode,
        )
        result["principal_axis"] = [
            float(value) for value in _coerce_xy(problem.get("principal_axis", ()), field_name="principal_axis")
        ]
        result["projected_step"] = float(problem.get("projected_step", 0.0))
        terminal_anchor = problem.get("terminal_anchor", {})
        if isinstance(terminal_anchor, Mapping):
            terminal_anchor_xy = terminal_anchor.get("xy")
            result["terminal_anchor"] = {
                "anchor_id": str(terminal_anchor.get("anchor_id", "")),
                "object_id": str(terminal_anchor.get("object_id", "")),
                "xy": [float(value) for value in _coerce_xy(terminal_anchor_xy, field_name="terminal_anchor.xy")],
            }
    elif track_name == SHAPE_COMPLETION_TRACK_NAME:
        anchor_order = list(problem.get("anchor_order", ()))
        if len(anchor_order) < 3:
            raise ValueError("square completion problem is missing anchor_order with at least three anchors.")
        readout = evaluate_square_completion_readout(
            shared_corner_xy=anchor_positions_xy[anchor_order[0]],
            adjacent_corner_b_xy=anchor_positions_xy[anchor_order[1]],
            adjacent_corner_c_xy=anchor_positions_xy[anchor_order[2]],
            gt_pos_xy=problem.get("gt_pos_xy", ()),
            pred_pos_xy=pred_pos_xy,
            epsilon_pos=epsilon_pos,
            tau_line=tau_line,
            gt_yaw_rad=problem.get("gt_yaw_rad"),
            pred_yaw_rad=pred_yaw,
            tau_yaw=tau_yaw,
        )
        result = _build_common_debug_result(
            problem,
            pred_pos_xy=pred_pos_xy,
            readout=readout,
            valid_episode=valid_episode,
        )
        result["anchor_order"] = [str(value) for value in anchor_order]
        result["square_center_xy"] = [
            float(value)
            for value in _coerce_xy(problem.get("square_center_xy", ()), field_name="square_center_xy")
        ]
        if "gt_yaw_rad" in problem:
            result["gt_yaw_rad"] = float(problem["gt_yaw_rad"])
        if pred_yaw is not None:
            result["pred_yaw_rad"] = float(pred_yaw)
        if target_yaw is not None:
            result["target_yaw_rad"] = float(target_yaw)
    else:
        raise ValueError(f"Unsupported protocol track: {track_name!r}")

    if target_pos_xy is not None:
        result["target_pos_xy"] = [float(value) for value in _coerce_xy(target_pos_xy, field_name="target_pos_xy")]
    if "trend_consistent" in readout:
        result["trend_consistent"] = bool(readout["trend_consistent"])
    if "direction_correct" in readout:
        result["direction_correct"] = bool(readout["direction_correct"])
    if "between_score" in readout:
        result["between_score"] = float(readout["between_score"])
    if "mirrored_distance_error" in readout:
        result["mirrored_distance_error"] = float(readout["mirrored_distance_error"])
    if "opposite_side" in readout:
        result["opposite_side"] = bool(readout["opposite_side"])
    if "opposite_side_score" in readout:
        result["opposite_side_score"] = float(readout["opposite_side_score"])
    if "diagonal_midpoint_error" in readout:
        result["diagonal_midpoint_error"] = float(readout["diagonal_midpoint_error"])
    if "edge_length_error" in readout:
        result["edge_length_error"] = float(readout["edge_length_error"])
    if "square_corner_cosine" in readout:
        result["square_corner_cosine"] = float(readout["square_corner_cosine"])
    if "yaw_error" in readout:
        result["yaw_error"] = float(readout["yaw_error"])
    if "yaw_satisfied" in readout:
        result["yaw_satisfied"] = bool(readout["yaw_satisfied"])
    if "tau_yaw" in readout:
        result["tau_yaw"] = float(readout["tau_yaw"])
    if placement_execution_success is not None:
        result["placement_execution_success"] = bool(placement_execution_success)
    if settle_applied is not None:
        result["settle_applied"] = bool(settle_applied)
    return result


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total > 0 else 0.0


def _mean_and_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return (None, None)

    resolved = [float(value) for value in values]
    mean = sum(resolved) / float(len(resolved))
    variance = sum((value - mean) ** 2 for value in resolved) / float(len(resolved))
    return (float(mean), float(math.sqrt(variance)))


def build_geometric_inference_task_summary(
    *,
    agent_name: str,
    episode_results: Sequence[Mapping[str, Any]],
    scene_id: int,
    layout_key: str,
    task_index: int,
    task_label: str | None = None,
) -> dict[str, Any]:
    episodes = [dict(result) for result in episode_results]
    num_episodes = len(episodes)
    valid_episodes = [episode for episode in episodes if bool(episode.get("valid_episode"))]

    valid_episode_count = len(valid_episodes)
    success_count = sum(bool(episode.get("success")) for episode in episodes)
    rule_satisfied_count = sum(bool(episode.get("rule_satisfied")) for episode in episodes)
    trend_consistent_count = sum(bool(episode.get("trend_consistent")) for episode in episodes)
    grasp_execution_success_count = sum(bool(episode.get("grasp_execution_success")) for episode in episodes)
    placement_execution_success_count = sum(bool(episode.get("placement_execution_success")) for episode in episodes)

    l2_error_mean, l2_error_std = _mean_and_std(
        [float(episode.get("l2_error", 0.0)) for episode in valid_episodes]
    )
    orthogonal_distance_mean, orthogonal_distance_std = _mean_and_std(
        [float(episode.get("orthogonal_distance", 0.0)) for episode in valid_episodes]
    )
    yaw_error_mean, yaw_error_std = _mean_and_std(
        [float(episode.get("yaw_error", 0.0)) for episode in valid_episodes if "yaw_error" in episode]
    )

    track_names = sorted({str(episode.get("track_name", "")) for episode in episodes if episode.get("track_name")})
    rule_types = sorted({str(episode.get("rule_type", "")) for episode in episodes if episode.get("rule_type")})
    yaw_satisfied_count = sum(bool(episode.get("yaw_satisfied")) for episode in episodes if "yaw_satisfied" in episode)
    yaw_metric_episode_count = sum(1 for episode in episodes if "yaw_satisfied" in episode)

    return {
        "agent_name": str(agent_name),
        "scene_id": int(scene_id),
        "layout_key": str(layout_key),
        "task_index": int(task_index),
        "task_label": str(task_label) if task_label is not None else None,
        "track_name": track_names[0] if len(track_names) == 1 else "mixed",
        "rule_type": rule_types[0] if len(rule_types) == 1 else "mixed",
        "num_episodes": num_episodes,
        "valid_episode_count": valid_episode_count,
        "valid_episode_rate": _rate(valid_episode_count, num_episodes),
        "success_count": success_count,
        "success_rate": _rate(success_count, num_episodes),
        "metric_episode_count": valid_episode_count,
        "l2_error_mean": l2_error_mean,
        "l2_error_std": l2_error_std,
        "orthogonal_distance_mean": orthogonal_distance_mean,
        "orthogonal_distance_std": orthogonal_distance_std,
        "yaw_error_mean": yaw_error_mean,
        "yaw_error_std": yaw_error_std,
        "rule_satisfied_count": rule_satisfied_count,
        "rule_satisfied_rate": _rate(rule_satisfied_count, num_episodes),
        "yaw_satisfied_count": yaw_satisfied_count,
        "yaw_satisfied_rate": _rate(yaw_satisfied_count, yaw_metric_episode_count),
        "trend_consistent_count": trend_consistent_count,
        "trend_consistent_rate": _rate(trend_consistent_count, num_episodes),
        "grasp_execution_success_count": grasp_execution_success_count,
        "grasp_execution_success_rate": _rate(grasp_execution_success_count, num_episodes),
        "placement_execution_success_count": placement_execution_success_count,
        "placement_execution_success_rate": _rate(placement_execution_success_count, num_episodes),
        "episodes": episodes,
    }


def build_geometric_inference_suite_summary(
    *,
    suite_name: str,
    agent_name: str,
    task_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    task_results = [dict(summary) for summary in task_summaries]
    all_episode_results: list[dict[str, Any]] = []
    for task_summary in task_results:
        episodes = task_summary.get("episodes", [])
        if isinstance(episodes, list):
            all_episode_results.extend(dict(episode) for episode in episodes if isinstance(episode, Mapping))

    valid_episodes = [episode for episode in all_episode_results if bool(episode.get("valid_episode"))]
    valid_episode_count = len(valid_episodes)
    total_episodes = len(all_episode_results)
    success_count = sum(bool(episode.get("success")) for episode in all_episode_results)
    rule_satisfied_count = sum(bool(episode.get("rule_satisfied")) for episode in all_episode_results)
    trend_consistent_count = sum(bool(episode.get("trend_consistent")) for episode in all_episode_results)
    grasp_execution_success_count = sum(bool(episode.get("grasp_execution_success")) for episode in all_episode_results)
    placement_execution_success_count = sum(
        bool(episode.get("placement_execution_success")) for episode in all_episode_results
    )

    l2_error_mean, l2_error_std = _mean_and_std(
        [float(episode.get("l2_error", 0.0)) for episode in valid_episodes]
    )
    orthogonal_distance_mean, orthogonal_distance_std = _mean_and_std(
        [float(episode.get("orthogonal_distance", 0.0)) for episode in valid_episodes]
    )
    yaw_error_mean, yaw_error_std = _mean_and_std(
        [float(episode.get("yaw_error", 0.0)) for episode in valid_episodes if "yaw_error" in episode]
    )
    yaw_satisfied_count = sum(bool(episode.get("yaw_satisfied")) for episode in all_episode_results if "yaw_satisfied" in episode)
    yaw_metric_episode_count = sum(1 for episode in all_episode_results if "yaw_satisfied" in episode)

    return {
        "suite_name": str(suite_name),
        "agent_name": str(agent_name),
        "num_tasks": len(task_results),
        "total_episodes": total_episodes,
        "valid_episode_count": valid_episode_count,
        "valid_episode_rate": _rate(valid_episode_count, total_episodes),
        "success_count": success_count,
        "success_rate": _rate(success_count, total_episodes),
        "metric_episode_count": valid_episode_count,
        "l2_error_mean": l2_error_mean,
        "l2_error_std": l2_error_std,
        "orthogonal_distance_mean": orthogonal_distance_mean,
        "orthogonal_distance_std": orthogonal_distance_std,
        "yaw_error_mean": yaw_error_mean,
        "yaw_error_std": yaw_error_std,
        "rule_satisfied_count": rule_satisfied_count,
        "rule_satisfied_rate": _rate(rule_satisfied_count, total_episodes),
        "yaw_satisfied_count": yaw_satisfied_count,
        "yaw_satisfied_rate": _rate(yaw_satisfied_count, yaw_metric_episode_count),
        "trend_consistent_count": trend_consistent_count,
        "trend_consistent_rate": _rate(trend_consistent_count, total_episodes),
        "grasp_execution_success_count": grasp_execution_success_count,
        "grasp_execution_success_rate": _rate(grasp_execution_success_count, total_episodes),
        "placement_execution_success_count": placement_execution_success_count,
        "placement_execution_success_rate": _rate(placement_execution_success_count, total_episodes),
        "task_results": task_results,
    }


__all__ = [
    "DEFAULT_EPSILON_POS",
    "DEFAULT_TAU_LINE",
    "DEFAULT_TAU_TREND",
    "DEFAULT_TAU_YAW",
    "EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME",
    "EXTENSION_RULE_TYPE",
    "GEOMETRIC_INFERENCE_MODULE_NAME",
    "IMPLICIT_TREND_FITTING_TRACK_NAME",
    "ISOMETRIC_EXTENSION_VARIANT",
    "MIDPOINT_RULE_TYPE",
    "PCA_TREND_RULE_TYPE",
    "PERFECT_MIDPOINT_VARIANT",
    "POINT_SYMMETRY_RULE_TYPE",
    "POINT_SYMMETRY_VARIANT",
    "SHAPE_COMPLETION_TRACK_NAME",
    "SQUARE_COMPLETION_RULE_TYPE",
    "TREND_CONTINUATION_SUBTASK_NAME",
    "WEAK_DESTRUCTION_CONTROL_SUBTASK_NAME",
    "build_geometric_inference_debug_result",
    "build_geometric_inference_problem",
    "build_geometric_inference_problem_from_object_positions",
    "build_geometric_inference_suite_summary",
    "build_geometric_inference_task_summary",
    "build_workspace_bounds_xy",
    "build_workspace_bounds_xy_from_points",
    "evaluate_explicit_rule_readout",
    "evaluate_extension_readout",
    "evaluate_midpoint_readout",
    "evaluate_point_symmetry_readout",
    "evaluate_square_completion_readout",
    "evaluate_trend_readout",
    "l2_distance_xy",
    "normalize_explicit_variant",
    "normalize_rule_type",
    "direction_alignment_xy",
    "point_to_line_distance_xy",
    "projection_ratio_xy",
    "resolve_square_anchor_object_ids",
    "resolve_extension_anchor_object_ids",
    "resolve_geometric_anchor_object_ids",
    "resolve_trend_anchor_object_ids",
    "resolve_target_object_id",
    "solve_explicit_rule_gt",
    "solve_extension_gt",
    "solve_midpoint_gt",
    "solve_point_symmetry_gt",
    "solve_square_completion_gt",
    "solve_trend_continuation_gt",
    "wrap_angle_pi",
    "yaw_error_mod_pi",
]
