from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


TRACK_SINGLE_PAIRED_ARC = "track_single_paired_arc"
TRACK_MULTI_STAGGERED_WAYPOINT = "track_multi_staggered_waypoint"
REINDEX_INSERT_SHALLOW_ARC = "reindex_insert_shallow_arc"
REINDEX_NEIGHBOR_BEZIER_LANES = "reindex_neighbor_bezier_lanes"
REINDEX_SUBSET_TWO_BLOCK_SWAP = "reindex_subset_two_block_swap"

SUPPORTED_POST_ACTION_RELOCATION_MOTION_MODES = frozenset(
    {
        TRACK_SINGLE_PAIRED_ARC,
        TRACK_MULTI_STAGGERED_WAYPOINT,
        REINDEX_INSERT_SHALLOW_ARC,
        REINDEX_NEIGHBOR_BEZIER_LANES,
        REINDEX_SUBSET_TWO_BLOCK_SWAP,
    }
)


def resolve_post_action_relocation_motion_mode(
    intervention: dict[str, Any] | None,
    *,
    default: str,
) -> str:
    """Resolve a portable motion function name without changing task/GT semantics."""
    raw_mode = intervention.get("motion_mode") if isinstance(intervention, dict) else None
    normalized = str(raw_mode or default).strip().lower()
    if normalized not in SUPPORTED_POST_ACTION_RELOCATION_MOTION_MODES:
        supported = ", ".join(sorted(SUPPORTED_POST_ACTION_RELOCATION_MOTION_MODES))
        raise ValueError(f"Unsupported post-action relocation motion_mode {normalized!r}; expected one of: {supported}")
    return normalized


def quintic_smoothstep(progress: torch.Tensor) -> torch.Tensor:
    progress = torch.clamp(progress, 0.0, 1.0)
    return progress * progress * progress * (progress * (progress * 6.0 - 15.0) + 10.0)


def piecewise_two_segment_via_waypoint(
    start_xy: torch.Tensor,
    target_xy: torch.Tensor,
    waypoint_xy: torch.Tensor,
    progress: torch.Tensor,
) -> torch.Tensor:
    first_leg = waypoint_xy - start_xy
    second_leg = target_xy - waypoint_xy
    first_length = torch.linalg.norm(first_leg, dim=-1).clamp(min=1e-5)
    second_length = torch.linalg.norm(second_leg, dim=-1).clamp(min=1e-5)
    distance = progress * (first_length + second_length)
    first_xy = start_xy + first_leg * (distance / first_length).clamp(0.0, 1.0).unsqueeze(-1)
    second_alpha = ((distance - first_length) / second_length).clamp(0.0, 1.0).unsqueeze(-1)
    second_xy = waypoint_xy + second_leg * second_alpha
    return torch.where(distance.unsqueeze(-1) < first_length.unsqueeze(-1), first_xy, second_xy)


def cubic_bezier_xy(
    start_xy: torch.Tensor,
    control_start_xy: torch.Tensor,
    control_end_xy: torch.Tensor,
    target_xy: torch.Tensor,
    progress: torch.Tensor,
) -> torch.Tensor:
    t = torch.clamp(progress, 0.0, 1.0).unsqueeze(-1)
    one_minus_t = 1.0 - t
    return (
        (one_minus_t * one_minus_t * one_minus_t) * start_xy
        + (3.0 * one_minus_t * one_minus_t * t) * control_start_xy
        + (3.0 * one_minus_t * t * t) * control_end_xy
        + (t * t * t) * target_xy
    )


def track_single_paired_arc_xy(
    start_xy: torch.Tensor,
    target_xy: torch.Tensor,
    motion_object_indices: torch.Tensor,
    progress: torch.Tensor,
    *,
    object_names: Sequence[str] = (),
) -> torch.Tensor:
    """Track-single baseline: exchange each active pair on opposite semicircles."""
    del object_names
    eased = 0.5 - 0.5 * torch.cos(torch.pi * torch.clamp(progress, 0.0, 1.0))
    interpolated = start_xy + (target_xy - start_xy) * eased.view(-1, 1, 1)
    num_slots = int(motion_object_indices.shape[1])
    two_pi = start_xy.new_full((), 2.0 * float(torch.pi))
    for pair_start in range(0, num_slots - 1, 2):
        pair_end = pair_start + 1
        clockwise = (pair_start // 2) % 2 == 0
        pair_active = (motion_object_indices[:, pair_start] >= 0) & (motion_object_indices[:, pair_end] >= 0)
        if not torch.any(pair_active):
            continue
        midpoint = 0.5 * (start_xy[:, pair_start, :] + start_xy[:, pair_end, :])
        radius = torch.linalg.norm(start_xy[:, pair_start, :] - midpoint, dim=-1)
        arc_ok = pair_active & (radius >= 1e-4)
        for slot in (pair_start, pair_end):
            theta_start = torch.atan2(start_xy[:, slot, 1] - midpoint[:, 1], start_xy[:, slot, 0] - midpoint[:, 0])
            theta_target = torch.atan2(target_xy[:, slot, 1] - midpoint[:, 1], target_xy[:, slot, 0] - midpoint[:, 0])
            if clockwise:
                delta = (theta_start - theta_target) % two_pi
                delta = torch.where(delta < 1e-5, torch.full_like(delta, two_pi), delta)
                theta = theta_start - eased * delta
            else:
                delta = (theta_target - theta_start) % two_pi
                delta = torch.where(delta < 1e-5, torch.full_like(delta, two_pi), delta)
                theta = theta_start + eased * delta
            arc_xy = torch.stack(
                (midpoint[:, 0] + radius * torch.cos(theta), midpoint[:, 1] + radius * torch.sin(theta)),
                dim=-1,
            )
            interpolated[:, slot, :] = torch.where(arc_ok.unsqueeze(-1), arc_xy, interpolated[:, slot, :])
    return interpolated


def reindex_insert_shallow_arc_xy(
    start_xy: torch.Tensor,
    target_xy: torch.Tensor,
    motion_object_indices: torch.Tensor,
    progress: torch.Tensor,
    *,
    object_names: Sequence[str] = (),
) -> torch.Tensor:
    """Single-mover tabletop arc derived from track-single's side-stepping path."""
    del object_names
    interpolated = start_xy + (target_xy - start_xy) * progress.view(-1, 1, 1)
    eased = quintic_smoothstep(progress)
    x_floor = start_xy.new_tensor(0.20)
    for slot in range(int(motion_object_indices.shape[1])):
        active = motion_object_indices[:, slot] >= 0
        if not torch.any(active):
            continue
        start = start_xy[:, slot, :]
        target = target_xy[:, slot, :]
        waypoint = torch.stack(
            (
                torch.maximum(torch.minimum(start[:, 0], target[:, 0]) - 0.055, x_floor),
                0.5 * (start[:, 1] + target[:, 1]),
            ),
            dim=-1,
        )
        routed = piecewise_two_segment_via_waypoint(start, target, waypoint, eased)
        interpolated[:, slot, :] = torch.where(active.unsqueeze(-1), routed, interpolated[:, slot, :])
    return interpolated


def _enforce_min_pairwise_separation_xy(
    xy_row: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    min_separation: float,
) -> torch.Tensor:
    result = xy_row.clone()
    for _ in range(3):
        for first in range(int(result.shape[0]) - 1):
            if not bool(active_mask[first].item()):
                continue
            for second in range(first + 1, int(result.shape[0])):
                if not bool(active_mask[second].item()):
                    continue
                delta = result[first] - result[second]
                distance = float(torch.linalg.norm(delta).item())
                if distance >= min_separation:
                    continue
                if distance < 1e-6:
                    delta = result.new_tensor([1.0, 0.0]) if (first + second) % 2 == 0 else result.new_tensor([0.0, 1.0])
                    distance = 1.0
                correction = 0.5 * (min_separation - distance) * delta / max(distance, 1e-6)
                result[first] += correction
                result[second] -= correction
    return result


def track_multi_staggered_waypoint_xy(
    start_xy: torch.Tensor,
    target_xy: torch.Tensor,
    motion_object_indices: torch.Tensor,
    progress: torch.Tensor,
    *,
    object_names: Sequence[str] = (),
) -> torch.Tensor:
    """Three-mover cycle with staggered, separated tabletop lanes."""
    del object_names
    eased = 0.5 - 0.5 * torch.cos(torch.pi * torch.clamp(progress, 0.0, 1.0))
    interpolated = start_xy + (target_xy - start_xy) * eased.view(-1, 1, 1)
    lane_offsets_x = start_xy.new_tensor([0.28, 0.24, 0.20, 0.16])
    lane_offsets_y = start_xy.new_tensor([0.14, -0.14, 0.10, -0.10])
    phase_start = start_xy.new_tensor([0.20, 0.35, 0.00, 0.05])
    phase_end = start_xy.new_tensor([0.90, 1.00, 0.70, 0.75])
    for slot in range(int(motion_object_indices.shape[1])):
        active = motion_object_indices[:, slot] >= 0
        if not torch.any(active):
            continue
        lane = min(slot, int(lane_offsets_x.shape[0]) - 1)
        start = start_xy[:, slot, :]
        target = target_xy[:, slot, :]
        slot_progress = torch.clamp(
            (eased - phase_start[lane]) / torch.clamp(phase_end[lane] - phase_start[lane], min=1e-4),
            0.0,
            1.0,
        )
        waypoint = torch.stack(
            (
                torch.maximum(torch.minimum(start[:, 0], target[:, 0]) - lane_offsets_x[lane], start_xy.new_tensor(0.18)),
                torch.clamp(0.5 * (start[:, 1] + target[:, 1]) + lane_offsets_y[lane], -0.30, 0.30),
            ),
            dim=-1,
        )
        routed = piecewise_two_segment_via_waypoint(start, target, waypoint, slot_progress)
        interpolated[:, slot, :] = torch.where(active.unsqueeze(-1), routed, interpolated[:, slot, :])
    return interpolated


def _bezier_lane_swap_xy(
    start_xy: torch.Tensor,
    target_xy: torch.Tensor,
    motion_object_indices: torch.Tensor,
    progress: torch.Tensor,
    *,
    object_names: Sequence[str],
    subset_mode: bool,
) -> torch.Tensor:
    cosine_eased = 0.5 - 0.5 * torch.cos(torch.pi * torch.clamp(progress, 0.0, 1.0))
    interpolated = start_xy + (target_xy - start_xy) * cosine_eased.view(-1, 1, 1)
    eased = quintic_smoothstep(progress)
    lane_count = max(int(motion_object_indices.shape[1]), 4)
    if subset_mode:
        lane_offsets_x = torch.linspace(0.30, 0.16, lane_count, device=start_xy.device, dtype=start_xy.dtype)
        lane_y_abs = 0.18
        min_separation = 0.064
        phase_start = torch.zeros(lane_count, device=start_xy.device, dtype=start_xy.dtype)
        phase_end = torch.ones(lane_count, device=start_xy.device, dtype=start_xy.dtype)
    else:
        lane_offsets_x = torch.linspace(0.14, 0.08, lane_count, device=start_xy.device, dtype=start_xy.dtype)
        lane_y_abs = 0.075
        min_separation = 0.072
        phase_start = torch.linspace(0.00, 0.18, lane_count, device=start_xy.device, dtype=start_xy.dtype)
        phase_end = torch.clamp(phase_start + 0.72, max=1.0)
    lane_offsets_y = torch.zeros(lane_count, device=start_xy.device, dtype=start_xy.dtype)
    lane_offsets_y[0::2] = torch.linspace(lane_y_abs, 0.04, (lane_count + 1) // 2, device=start_xy.device, dtype=start_xy.dtype)
    lane_offsets_y[1::2] = -torch.linspace(lane_y_abs, 0.04, lane_count // 2, device=start_xy.device, dtype=start_xy.dtype)
    right_red_index = object_names.index("red_subset_block_2") if "red_subset_block_2" in object_names else -1
    middle_red_index = object_names.index("red_subset_block_1") if "red_subset_block_1" in object_names else -1

    for slot in range(min(int(motion_object_indices.shape[1]), lane_count)):
        active = motion_object_indices[:, slot] >= 0
        start = start_xy[:, slot, :]
        target = target_xy[:, slot, :]
        moving = active & (torch.linalg.vector_norm(target - start, dim=-1) > 1e-3)
        if not torch.any(moving):
            continue
        middle_direct = moving & (motion_object_indices[:, slot] == middle_red_index) if subset_mode else torch.zeros_like(moving)
        routed_mask = moving & ~middle_direct
        if not torch.any(routed_mask):
            continue
        slot_progress = quintic_smoothstep(
            torch.clamp((eased - phase_start[slot]) / torch.clamp(phase_end[slot] - phase_start[slot], min=1e-4), 0.0, 1.0)
        )
        right_back = subset_mode and right_red_index >= 0 and torch.any(
            routed_mask & (motion_object_indices[:, slot] == right_red_index)
        )
        if right_back:
            back_x = torch.clamp(torch.minimum(start[:, 0], target[:, 0]) - 0.16, min=0.16, max=0.70)
            control_start = torch.stack((back_x, torch.clamp(start[:, 1], -0.34, 0.34)), dim=-1)
            control_end = torch.stack((back_x, torch.clamp(target[:, 1], -0.34, 0.34)), dim=-1)
        else:
            lane_x = torch.maximum(torch.minimum(start[:, 0], target[:, 0]) - lane_offsets_x[slot], start_xy.new_tensor(0.16))
            control_start = torch.stack((lane_x, torch.clamp(start[:, 1] + lane_offsets_y[slot], -0.34, 0.34)), dim=-1)
            control_end = torch.stack((lane_x, torch.clamp(target[:, 1] + lane_offsets_y[slot], -0.34, 0.34)), dim=-1)
        routed = cubic_bezier_xy(start, control_start, control_end, target, slot_progress)
        interpolated[:, slot, :] = torch.where(routed_mask.unsqueeze(-1), routed, interpolated[:, slot, :])

    for row in range(int(interpolated.shape[0])):
        active = motion_object_indices[row] >= 0
        if int(active.sum().item()) >= 2:
            interpolated[row] = _enforce_min_pairwise_separation_xy(
                interpolated[row],
                active,
                min_separation=min_separation,
            )
    return interpolated


def reindex_neighbor_bezier_lanes_xy(
    start_xy: torch.Tensor,
    target_xy: torch.Tensor,
    motion_object_indices: torch.Tensor,
    progress: torch.Tensor,
    *,
    object_names: Sequence[str] = (),
) -> torch.Tensor:
    return _bezier_lane_swap_xy(
        start_xy,
        target_xy,
        motion_object_indices,
        progress,
        object_names=object_names,
        subset_mode=False,
    )


def reindex_subset_two_block_swap_xy(
    start_xy: torch.Tensor,
    target_xy: torch.Tensor,
    motion_object_indices: torch.Tensor,
    progress: torch.Tensor,
    *,
    object_names: Sequence[str] = (),
) -> torch.Tensor:
    return _bezier_lane_swap_xy(
        start_xy,
        target_xy,
        motion_object_indices,
        progress,
        object_names=object_names,
        subset_mode=True,
    )


MotionFunction = Callable[..., torch.Tensor]

MOTION_FUNCTIONS: dict[str, MotionFunction] = {
    TRACK_SINGLE_PAIRED_ARC: track_single_paired_arc_xy,
    TRACK_MULTI_STAGGERED_WAYPOINT: track_multi_staggered_waypoint_xy,
    REINDEX_INSERT_SHALLOW_ARC: reindex_insert_shallow_arc_xy,
    REINDEX_NEIGHBOR_BEZIER_LANES: reindex_neighbor_bezier_lanes_xy,
    REINDEX_SUBSET_TWO_BLOCK_SWAP: reindex_subset_two_block_swap_xy,
}


def get_post_action_relocation_motion_function(motion_mode: str) -> MotionFunction:
    try:
        return MOTION_FUNCTIONS[motion_mode]
    except KeyError as exc:
        raise ValueError(f"Motion mode {motion_mode!r} is not registered") from exc