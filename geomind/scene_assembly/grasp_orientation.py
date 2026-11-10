"""Two-yaw grasp direction screening from measured boxes, not collision proof.

The caller supplies a measured conservative horizontal envelope of the open
gripper. No jaw/finger size is guessed here. Only XY projections are compared;
actual contact, joint limits, reference preservation and stable holding remain
runtime acceptance requirements.
"""
from __future__ import annotations

import math
import numpy as np


def _boxes(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.ndim != 2 or array.shape[1] != 3 or len(array) < 1:
        raise ValueError(f"{name} must be a nonempty numeric Nx3 array")
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return array


def _half_xy(value):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.shape != (2,):
        raise ValueError("Open-gripper footprint must contain two numeric XY half extents")
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError("Open-gripper footprint half extents must be finite and positive")
    return array


def _signed_xy_clearance(center, half, low, high):
    """Signed separation of two axis-aligned rectangles in XY.

    Positive values are the shortest Euclidean distance between the rectangles.
    Negative values are the smallest axis translation separating an overlap.
    A zero value is contact, which is conservatively classified as blocked.
    """
    footprint_low, footprint_high = center-half, center+half
    axis_gaps = np.maximum(low-footprint_high, footprint_low-high)
    outside = float(np.linalg.norm(np.maximum(axis_gaps, 0.)))
    inside = min(float(axis_gaps.max()), 0.)
    return outside+inside, axis_gaps


def plan_grasp_orientation(centres_w_m, bbox_min_w_m, bbox_max_w_m, selected_index,
                           *, open_gripper_half_xy_m, footprint_shape="rectangle",
                           canonical_clearance_margin_m=None):
    """Choose yaw 0/90 by maximum minimum signed reference-box clearance.

    ``open_gripper_half_xy_m`` is the caller's conservative hand-centered open
    gripper envelope at yaw zero. Rotating yaw 90 swaps its horizontal axes.
    Baseline jaws close along world Y; yaw 90 jaws close along world X. Inputs
    must be the measured world AABB centres and bounds in matching object order.
    """
    if footprint_shape != "rectangle":
        raise ValueError("Only a measured rectangular open-gripper footprint is supported")
    if canonical_clearance_margin_m is not None:
        if (type(canonical_clearance_margin_m) not in (int,float)
                or not math.isfinite(canonical_clearance_margin_m) or canonical_clearance_margin_m<0):
            raise ValueError('Canonical clearance margin must be finite and nonnegative')
    centres = _boxes(centres_w_m, "centres_w_m")
    low = _boxes(bbox_min_w_m, "bbox_min_w_m")
    high = _boxes(bbox_max_w_m, "bbox_max_w_m")
    if centres.shape != low.shape or low.shape != high.shape:
        raise ValueError("Measured centre and bounding-box array shapes differ")
    if np.any(high <= low):
        raise ValueError("Measured bounding boxes must have positive dimensions on all axes")
    if not np.allclose(centres, (low+high)*.5, atol=1e-7, rtol=0):
        raise ValueError("Measured centres differ from the supplied world AABB midpoints")
    if type(selected_index) is not int or not 0 <= selected_index < len(centres):
        raise ValueError("selected_index must identify one measured object")
    half_zero = _half_xy(open_gripper_half_xy_m)
    references = [index for index in range(len(centres)) if index != selected_index]
    candidates = []
    for yaw_degrees, width_axis in ((0, 1), (90, 0)):
        yaw = math.radians(yaw_degrees)
        half = half_zero.copy() if yaw_degrees == 0 else half_zero[::-1].copy()
        clearances = []
        for index in references:
            clearance, axis_gaps = _signed_xy_clearance(centres[selected_index, :2], half,
                                                       low[index, :2], high[index, :2])
            clearances.append({"reference_index": index, "signed_xy_clearance_m": clearance,
                "xy_axis_gaps_m": axis_gaps.tolist(), "blocked": bool(clearance <= 0.)})
        minimum = min((item["signed_xy_clearance_m"] for item in clearances), default=None)
        candidates.append({"yaw_degrees": yaw_degrees, "yaw_rad": yaw,
            "tool_quat_wxyz": [0., math.cos(yaw*.5), math.sin(yaw*.5), 0.],
            "grip_width_axis": width_axis,
            "grip_width_m": float(high[selected_index, width_axis]-low[selected_index, width_axis]),
            "open_gripper_half_xy_m": half.tolist(),
            "minimum_reference_clearance_m": minimum,
            "reference_clearances": clearances,
            "blocked_reference_indices": [item["reference_index"] for item in clearances if item["blocked"]]})
    # Ordered candidates retain yaw zero for an exact tie (including no refs).
    winner = max(candidates, key=lambda item: math.inf if item["minimum_reference_clearance_m"] is None
                 else item["minimum_reference_clearance_m"])
    policy="maximum minimum signed XY reference clearance; exact ties prefer yaw 0"
    if canonical_clearance_margin_m is not None:
        canonical=candidates[0]
        gap=canonical['minimum_reference_clearance_m']
        if not canonical['blocked_reference_indices'] and (gap is None or gap>=canonical_clearance_margin_m):
            winner=canonical
        policy='prefer canonical yaw 0 when its measured reference clearance meets the explicit margin; otherwise maximum minimum clearance'
    return {"kind": "measured_xy_grasp_orientation_screen", "selected_index": selected_index,
        **{key: winner[key] for key in ("yaw_degrees", "yaw_rad", "tool_quat_wxyz", "grip_width_axis",
            "grip_width_m", "minimum_reference_clearance_m", "blocked_reference_indices")},
        "selected_has_positive_reference_clearance": not bool(winner["blocked_reference_indices"]),
        "all_candidates_blocked": all(bool(item["blocked_reference_indices"]) for item in candidates),
        "input_open_gripper_half_xy_m": half_zero.tolist(), "footprint_shape": footprint_shape,
        "candidate_orientations": candidates,
        "selection_policy":policy,"canonical_clearance_margin_m":canonical_clearance_margin_m,
        "limitations": [
            "XY AABB projection only; vertical separation is not used to dismiss references.",
            "The caller must supply an asset-measured conservative open-gripper footprint; none is inferred here.",
            "Only yaw 0 and 90 are screened; jaw opening capacity and IK are not certified.",
            "No approach/transport path, full robot self collision or contact-free certificate.",
            "All measured references are retained, even when both candidates are blocked; runtime acceptance remains strict.",
        ]}
