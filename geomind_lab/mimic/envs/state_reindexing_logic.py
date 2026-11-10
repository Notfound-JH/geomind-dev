from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from .task_contract_logic import structural_adjacency_indices


REINDEX_EXTREME_TASK_ID = "reindex_extreme"
SWAP_ADJACENT_PAIRS_OVER_4_SCRIPT_NAME = "swap_adjacent_pairs_over_4"
REINDEX_INSERT_TASK_ID = "reindex_insert"
REINDEX_SUBSET_TASK_ID = "reindex_subset"
ORDINAL_INSERT_STATIC_TASK_ID = "ordinal_insert_static"
GRID_ROWCOL_GROUNDING_TASK_ID = "grid_rowcol_grounding"
REINDEX_NEIGHBOR_TASK_ID = "reindex_neighbor"
TRACK_SINGLE_TASK_ID = "track_single"
TRACK_MULTI_TASK_ID = "track_multi"
TRACK_CLUTTER_TASK_ID = "track_clutter"

DEFAULT_STATE_REINDEXING_SCRIPT_NAME = "leftmost_to_right_of_rightmost"
DEFAULT_STATE_REINDEXING_TARGET_RANK = "leftmost"
DEFAULT_POST_ACTION_RELOCATION_CONTROL_MODE = "full"
DEFAULT_POST_ACTION_RELOCATION_ORDER_AXIS = "x"
FINAL_FRAME_ONLY_CONTROL_MODE = "final_frame_only"

DEFAULT_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME = "middle_to_left_then_rightmost_to_left"
DEFAULT_HISTORICAL_IDENTITY_TRACKING_TARGET_RANK = "middle"
SINGLE_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME = "middle_to_left_of_leftmost"
MULTI_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME = "leftmost_to_middle_then_rightmost_to_left_of_leftmost"
DYNAMIC_SWAP_FIRST_SECOND_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME = "swap_first_and_second_over_2s"
SEEDED_NEIGHBOR_SWAP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME = "swap_middle_with_seeded_neighbor_over_2s"
# Same swap plan and GT as ``swap_middle_with_seeded_neighbor_over_2s``; sim motion uses a straight path for the
# middle object (dynamic slot 1) while the neighbor (slot 0) keeps the default paired arc trajectory.
SEEDED_NEIGHBOR_SWAP_MIDDLE_DIRECT_OVER_2S_SCRIPT_NAME = "swap_middle_with_seeded_neighbor_middle_direct_over_2s"

SEEDED_NEIGHBOR_SWAP_SCRIPT_NAMES: frozenset[str] = frozenset(
    {
        SEEDED_NEIGHBOR_SWAP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
        SEEDED_NEIGHBOR_SWAP_MIDDLE_DIRECT_OVER_2S_SCRIPT_NAME,
    }
)

STATIC_SCENE_NOOP_SCRIPT_NAME = "static_scene_noop"
GT_RESOLVER_RANK_INITIAL_LAYOUT = "rank_initial_layout"
GT_RESOLVER_GRID_ROWCOL_SLOT = "grid_rowcol_slot"
GT_RESOLVER_NEAREST_PLANAR_LEFT_OF_REFERENCE = "nearest_planar_left_of_reference"

SUPPORTED_STATE_REINDEXING_SCRIPT_NAMES = frozenset(
    {
        "leftmost_to_right_of_rightmost",
        "rightmost_to_left_of_leftmost",
        "rightmost_to_between_first_and_second",
        SWAP_ADJACENT_PAIRS_OVER_4_SCRIPT_NAME,
        STATIC_SCENE_NOOP_SCRIPT_NAME,
    }
)
SUPPORTED_POST_ACTION_RELOCATION_CONTROL_MODES = frozenset(
    {
        DEFAULT_POST_ACTION_RELOCATION_CONTROL_MODE,
        FINAL_FRAME_ONLY_CONTROL_MODE,
    }
)
SUPPORTED_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAMES = frozenset(
    {
        DEFAULT_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
        SINGLE_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
        MULTI_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
        DYNAMIC_SWAP_FIRST_SECOND_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
        SEEDED_NEIGHBOR_SWAP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
        SEEDED_NEIGHBOR_SWAP_MIDDLE_DIRECT_OVER_2S_SCRIPT_NAME,
    }
)
SUPPORTED_POST_ACTION_RELOCATION_ORDER_AXES = frozenset({"x", "y"})
SUPPORTED_TARGET_RANK_ALIASES = frozenset(
    {
        "leftmost",
        "first",
        "frontmost",
        "rightmost",
        "last",
        "middle",
        "center",
        "second_from_start",
    }
)
SUPPORTED_STATE_REINDEXING_TARGET_RANKS = SUPPORTED_TARGET_RANK_ALIASES | frozenset(
    {
        "index_0",
        "index_1",
        "index_2",
        "index_3",
        "index_4",
    }
)
SUPPORTED_HISTORICAL_IDENTITY_TRACKING_TARGET_RANKS = SUPPORTED_STATE_REINDEXING_TARGET_RANKS


def resolve_post_action_relocation_task_id(protocol: dict[str, Any] | None) -> str:
    if not isinstance(protocol, dict):
        return REINDEX_EXTREME_TASK_ID

    track_name = str(protocol.get("track_name", "")).strip()
    if track_name in {"historical_identity_tracking", "object_identity_tracking"}:
        script_name = resolve_state_reindexing_script_name(protocol)
        if script_name == MULTI_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME:
            return TRACK_MULTI_TASK_ID
        return TRACK_SINGLE_TASK_ID

    subset_spec = protocol.get("target_subset_spec", {})
    if track_name == "reorder_with_distractors" or isinstance(subset_spec, dict):
        return REINDEX_SUBSET_TASK_ID

    script_name = resolve_state_reindexing_script_name(protocol)
    if script_name == STATIC_SCENE_NOOP_SCRIPT_NAME:
        gt_spec = protocol.get("gt_spec", {})
        if isinstance(gt_spec, dict) and str(gt_spec.get("resolver", "")).strip() == GT_RESOLVER_GRID_ROWCOL_SLOT:
            return GRID_ROWCOL_GROUNDING_TASK_ID
        return ORDINAL_INSERT_STATIC_TASK_ID
    if script_name == "rightmost_to_between_first_and_second":
        return REINDEX_INSERT_TASK_ID
    return REINDEX_EXTREME_TASK_ID


def resolve_state_reindexing_script_name(protocol: dict[str, Any] | None) -> str:
    if not isinstance(protocol, dict):
        return DEFAULT_STATE_REINDEXING_SCRIPT_NAME
    intervention = protocol.get("intervention", {})
    if not isinstance(intervention, dict):
        return DEFAULT_STATE_REINDEXING_SCRIPT_NAME
    script_name = intervention.get("script_name")
    if not isinstance(script_name, str) or not script_name.strip():
        return DEFAULT_STATE_REINDEXING_SCRIPT_NAME
    return script_name.strip()


def resolve_state_reindexing_target_rank(protocol: dict[str, Any] | None) -> str:
    if not isinstance(protocol, dict):
        return DEFAULT_STATE_REINDEXING_TARGET_RANK
    gt_spec = protocol.get("gt_spec", {})
    if not isinstance(gt_spec, dict):
        return DEFAULT_STATE_REINDEXING_TARGET_RANK
    target_rank = gt_spec.get("target_rank")
    if isinstance(target_rank, int):
        return f"index_{int(target_rank)}"
    if not isinstance(target_rank, str) or not target_rank.strip():
        return DEFAULT_STATE_REINDEXING_TARGET_RANK
    return target_rank.strip()


def normalize_post_action_relocation_control_mode(control_mode: Any) -> str:
    if not isinstance(control_mode, str):
        return DEFAULT_POST_ACTION_RELOCATION_CONTROL_MODE
    normalized = control_mode.strip().lower()
    if normalized not in SUPPORTED_POST_ACTION_RELOCATION_CONTROL_MODES:
        return DEFAULT_POST_ACTION_RELOCATION_CONTROL_MODE
    return normalized


def resolve_post_action_relocation_control_mode(protocol: dict[str, Any] | None) -> str:
    if not isinstance(protocol, dict):
        return DEFAULT_POST_ACTION_RELOCATION_CONTROL_MODE
    return normalize_post_action_relocation_control_mode(protocol.get("control_mode"))


def normalize_post_action_relocation_order_axis(order_axis: Any) -> str:
    if not isinstance(order_axis, str):
        return DEFAULT_POST_ACTION_RELOCATION_ORDER_AXIS
    normalized = order_axis.strip().lower()
    if normalized not in SUPPORTED_POST_ACTION_RELOCATION_ORDER_AXES:
        return DEFAULT_POST_ACTION_RELOCATION_ORDER_AXIS
    return normalized


def resolve_post_action_relocation_order_axis(protocol: dict[str, Any] | None) -> str:
    if not isinstance(protocol, dict):
        return DEFAULT_POST_ACTION_RELOCATION_ORDER_AXIS
    gt_spec = protocol.get("gt_spec", {})
    if not isinstance(gt_spec, dict):
        return DEFAULT_POST_ACTION_RELOCATION_ORDER_AXIS
    return normalize_post_action_relocation_order_axis(gt_spec.get("order_axis"))


def _order_axis_index(order_axis: str) -> int:
    return 1 if normalize_post_action_relocation_order_axis(order_axis) == "y" else 0


def _normalize_active_object_mask(
    reference_xy: torch.Tensor,
    active_object_mask: torch.Tensor | None,
) -> torch.Tensor:
    num_envs = int(reference_xy.shape[0])
    num_objects = int(reference_xy.shape[1]) if reference_xy.ndim >= 2 else 0
    if active_object_mask is None:
        return torch.ones((num_envs, num_objects), dtype=torch.bool, device=reference_xy.device)

    mask = active_object_mask.to(device=reference_xy.device, dtype=torch.bool)
    if mask.ndim == 1:
        if int(mask.shape[0]) != num_objects:
            return torch.ones((num_envs, num_objects), dtype=torch.bool, device=reference_xy.device)
        return mask.unsqueeze(0).expand(num_envs, -1)
    if mask.ndim == 2 and tuple(mask.shape[:2]) == (num_envs, num_objects):
        return mask
    return torch.ones((num_envs, num_objects), dtype=torch.bool, device=reference_xy.device)


def _build_rank_template(num_envs: int, num_objects: int, device: torch.device) -> torch.Tensor:
    return torch.full((num_envs, num_objects), -1, dtype=torch.long, device=device)


def _gather_rank_indices(rank_indices: torch.Tensor, rank_positions: torch.Tensor) -> torch.Tensor:
    num_envs = int(rank_indices.shape[0])
    result = torch.full((num_envs,), -1, dtype=torch.long, device=rank_indices.device)
    if rank_indices.ndim != 2 or rank_indices.shape[1] == 0:
        return result

    valid_mask = (rank_positions >= 0) & (rank_positions < int(rank_indices.shape[1]))
    if not torch.any(valid_mask):
        return result

    batch_index = torch.arange(num_envs, device=rank_indices.device)
    result[valid_mask] = rank_indices[batch_index[valid_mask], rank_positions[valid_mask]]
    return result


def _active_order_summary(
    xy: torch.Tensor,
    *,
    order_axis: str,
    active_object_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    device = xy.device
    num_envs = int(xy.shape[0])
    num_objects = int(xy.shape[1]) if xy.ndim >= 2 else 0
    axis_index = _order_axis_index(order_axis)
    active_mask = _normalize_active_object_mask(xy, active_object_mask)

    rank_indices = _build_rank_template(num_envs, num_objects, device)
    active_counts = active_mask.sum(dim=1)
    strictly_increasing = torch.ones((num_envs,), dtype=torch.bool, device=device)
    mean_gap = torch.zeros((num_envs,), dtype=xy.dtype, device=device)

    if num_objects == 0:
        return {
            "rank_indices": rank_indices,
            "active_counts": active_counts,
            "strictly_increasing": strictly_increasing,
            "mean_gap": mean_gap,
            "active_object_mask": active_mask,
        }

    masked_axis = xy[..., axis_index].masked_fill(~active_mask, torch.finfo(xy.dtype).max)
    sort_indices = torch.argsort(masked_axis, dim=1)
    rank_positions = torch.arange(num_objects, device=device).unsqueeze(0)
    rank_valid_mask = rank_positions < active_counts.unsqueeze(1)
    rank_indices = torch.where(rank_valid_mask, sort_indices, torch.full_like(sort_indices, -1))

    if num_objects > 1:
        sorted_axis = torch.gather(xy[..., axis_index], 1, sort_indices)
        gaps = sorted_axis[:, 1:] - sorted_axis[:, :-1]
        gap_valid_mask = rank_positions[:, :-1] < (active_counts - 1).clamp_min(0).unsqueeze(1)
        if gaps.numel() > 0:
            gap_is_positive = gaps > 1.0e-4
            gap_checks = torch.where(gap_valid_mask, gap_is_positive, torch.ones_like(gap_is_positive))
            strictly_increasing = gap_checks.all(dim=1)
            gap_sums = torch.where(gap_valid_mask, gaps, torch.zeros_like(gaps)).sum(dim=1)
            gap_counts = gap_valid_mask.sum(dim=1).clamp_min(1)
            mean_gap = gap_sums / gap_counts.to(dtype=xy.dtype)

    return {
        "rank_indices": rank_indices,
        "active_counts": active_counts,
        "strictly_increasing": strictly_increasing,
        "mean_gap": mean_gap,
        "active_object_mask": active_mask,
    }


def _resolve_rank_descriptor_supported(target_rank: Any) -> bool:
    if isinstance(target_rank, int):
        return int(target_rank) >= 0
    if not isinstance(target_rank, str):
        return False
    normalized = target_rank.strip().lower()
    if normalized in SUPPORTED_TARGET_RANK_ALIASES:
        return True
    if normalized.startswith("index_"):
        try:
            return int(normalized.split("_", 1)[1]) >= 0
        except (TypeError, ValueError):
            return False
    if normalized.startswith("from_end_"):
        try:
            return int(normalized.rsplit("_", 1)[1]) > 0
        except ValueError:
            return False
    return False


def _resolve_target_rank_position(target_rank: Any, active_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = active_counts.device
    num_envs = int(active_counts.shape[0])
    positions = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    supported = torch.zeros((num_envs,), dtype=torch.bool, device=device)

    if isinstance(target_rank, int):
        rank_index = int(target_rank)
        if rank_index >= 0:
            positions.fill_(rank_index)
            supported = active_counts > rank_index
        return positions, supported

    if not isinstance(target_rank, str):
        return positions, supported

    normalized = target_rank.strip().lower()
    if normalized in {"leftmost", "first", "frontmost"}:
        positions.zero_()
        supported = active_counts > 0
        return positions, supported
    if normalized in {"rightmost", "last"}:
        positions = active_counts - 1
        supported = active_counts > 0
        return positions, supported
    if normalized in {"middle", "center"}:
        positions = active_counts // 2
        supported = (active_counts > 0) & (active_counts % 2 == 1)
        return positions, supported
    if normalized == "second_from_start":
        positions.fill_(1)
        supported = active_counts > 1
        return positions, supported
    if normalized.startswith("index_"):
        try:
            rank_index = int(normalized.split("_", 1)[1])
        except (TypeError, ValueError):
            return positions, supported
        if rank_index >= 0:
            positions.fill_(rank_index)
            supported = active_counts > rank_index
    if normalized.startswith("from_end_"):
        try:
            index = int(normalized.rsplit("_", 1)[1])
        except ValueError:
            return positions, supported
        positions = active_counts - index
        supported = (index > 0) & (positions >= 0)
    return positions, supported


def _masked_planar_displacement(
    initial_xy: torch.Tensor,
    post_xy: torch.Tensor,
    *,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    device = initial_xy.device
    num_envs = int(initial_xy.shape[0])
    num_objects = int(initial_xy.shape[1]) if initial_xy.ndim >= 2 else 0
    if num_objects == 0:
        return torch.zeros((num_envs,), dtype=initial_xy.dtype, device=device)

    displacement = torch.linalg.vector_norm(post_xy - initial_xy, dim=-1)
    masked = displacement.masked_fill(~active_mask, 0.0)
    return masked.max(dim=1).values


def _expected_post_rank_indices_after_adjacent_pair_swaps(
    initial_rank_indices: torch.Tensor,
    active_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """四块沿 order 轴排序时：rank0↔rank1、rank2↔rank3 互换后，各 rank 上应为的原物体下标。"""
    device = initial_rank_indices.device
    num_objects = int(initial_rank_indices.shape[1]) if initial_rank_indices.ndim >= 2 else 0
    supported = (active_counts == 4) & (num_objects >= 4)
    expected = initial_rank_indices.clone()
    if num_objects < 4:
        return expected, torch.zeros_like(supported)
    r0 = initial_rank_indices[:, 0]
    r1 = initial_rank_indices[:, 1]
    r2 = initial_rank_indices[:, 2]
    r3 = initial_rank_indices[:, 3]
    expected[:, 0] = torch.where(supported, r1, expected[:, 0])
    expected[:, 1] = torch.where(supported, r0, expected[:, 1])
    expected[:, 2] = torch.where(supported, r3, expected[:, 2])
    expected[:, 3] = torch.where(supported, r2, expected[:, 3])
    return expected, supported


def _expected_terminal_rank_position(script_name: str, active_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = active_counts.device
    positions = torch.full((int(active_counts.shape[0]),), -1, dtype=torch.long, device=device)
    supported = torch.zeros_like(positions, dtype=torch.bool)

    if script_name == "leftmost_to_right_of_rightmost":
        positions = active_counts - 1
        supported = active_counts >= 3
    elif script_name == "rightmost_to_left_of_leftmost":
        positions.zero_()
        supported = active_counts >= 3
    elif script_name == "rightmost_to_between_first_and_second":
        positions.fill_(1)
        supported = active_counts >= 3
    return positions, supported


def _expected_historical_rank_indices(
    initial_rank_indices: torch.Tensor,
    active_counts: torch.Tensor,
    *,
    script_name: str,
    swap_with_right_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = initial_rank_indices.device
    num_envs, num_objects = tuple(initial_rank_indices.shape[:2])
    expected = _build_rank_template(num_envs, num_objects, device)
    supported = torch.zeros((num_envs,), dtype=torch.bool, device=device)

    if num_objects == 0:
        return expected, supported

    if script_name in {
        SINGLE_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
        DYNAMIC_SWAP_FIRST_SECOND_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
    }:
        supported = active_counts == 3
        if torch.any(supported):
            expected[supported, 0] = initial_rank_indices[supported, 1]
            expected[supported, 1] = initial_rank_indices[supported, 0]
            expected[supported, 2] = initial_rank_indices[supported, 2]
        return expected, supported

    if script_name in SEEDED_NEIGHBOR_SWAP_SCRIPT_NAMES:
        supported = active_counts == 3
        if torch.any(supported):
            normalized_swap_with_right_mask = torch.zeros((num_envs,), dtype=torch.bool, device=device)
            if isinstance(swap_with_right_mask, torch.Tensor) and tuple(swap_with_right_mask.shape[:1]) == (num_envs,):
                normalized_swap_with_right_mask = swap_with_right_mask.to(device=device, dtype=torch.bool)
            swap_left_mask = supported & ~normalized_swap_with_right_mask
            swap_right_mask = supported & normalized_swap_with_right_mask
            if torch.any(swap_left_mask):
                expected[swap_left_mask, 0] = initial_rank_indices[swap_left_mask, 1]
                expected[swap_left_mask, 1] = initial_rank_indices[swap_left_mask, 0]
                expected[swap_left_mask, 2] = initial_rank_indices[swap_left_mask, 2]
            if torch.any(swap_right_mask):
                expected[swap_right_mask, 0] = initial_rank_indices[swap_right_mask, 0]
                expected[swap_right_mask, 1] = initial_rank_indices[swap_right_mask, 2]
                expected[swap_right_mask, 2] = initial_rank_indices[swap_right_mask, 1]
        return expected, supported

    if script_name == DEFAULT_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME:
        supported = active_counts == 3
        if torch.any(supported):
            expected[supported, 0] = initial_rank_indices[supported, 2]
            expected[supported, 1] = initial_rank_indices[supported, 1]
            expected[supported, 2] = initial_rank_indices[supported, 0]
        return expected, supported

    if script_name == MULTI_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME:
        supported = active_counts == 5
        if torch.any(supported):
            # Three-object cycle while keeping rank-1 and rank-3 fixed:
            # r0->r4, r4->r2, r2->r0.
            expected[supported, 0] = initial_rank_indices[supported, 2]
            expected[supported, 1] = initial_rank_indices[supported, 1]
            expected[supported, 2] = initial_rank_indices[supported, 4]
            expected[supported, 3] = initial_rank_indices[supported, 3]
            expected[supported, 4] = initial_rank_indices[supported, 0]
        return expected, supported

    return expected, supported


def _axis_index_for_name(axis_name: str) -> int:
    return 1 if normalize_post_action_relocation_order_axis(axis_name) == "y" else 0


def _rank_rows_match(
    actual_rank_indices: torch.Tensor,
    expected_rank_indices: torch.Tensor,
    active_counts: torch.Tensor,
) -> torch.Tensor:
    num_objects = int(actual_rank_indices.shape[1]) if actual_rank_indices.ndim == 2 else 0
    if num_objects == 0:
        return torch.ones((int(actual_rank_indices.shape[0]),), dtype=torch.bool, device=actual_rank_indices.device)

    rank_positions = torch.arange(num_objects, device=actual_rank_indices.device).unsqueeze(0)
    valid_mask = rank_positions < active_counts.unsqueeze(1)
    matches = torch.where(valid_mask, actual_rank_indices == expected_rank_indices, torch.ones_like(valid_mask))
    return matches.all(dim=1)


def _grid_slot_assignment_for_env(
    xy_row: torch.Tensor,
    *,
    num_rows: int,
    num_cols: int,
    row_axis: str,
    col_axis: str,
    quant_step: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
    """Assign each object a (row,col) in 1..num_rows / 1..num_cols; return None if invalid."""
    device = xy_row.device
    dtype = xy_row.dtype
    num_objects = int(xy_row.shape[0])
    if num_rows <= 0 or num_cols <= 0 or num_objects != num_rows * num_cols:
        return None, None, False

    ri = _axis_index_for_name(row_axis)
    ci = _axis_index_for_name(col_axis)
    if ri == ci:
        return None, None, False

    xs = xy_row[:, ri]
    ys = xy_row[:, ci]
    step = max(float(quant_step), 1.0e-6)
    row_key = (xs / step).round() * step
    unique_rows = torch.unique(row_key, sorted=True)
    if int(unique_rows.numel()) != num_rows:
        return None, None, False

    row_ids = torch.searchsorted(unique_rows, row_key, right=False)
    row_ids = torch.clamp(row_ids, max=num_rows - 1)

    col_index = torch.empty((num_objects,), dtype=torch.long, device=device)
    for r in range(num_rows):
        mask = row_ids == r
        count = int(mask.sum().item())
        if count != num_cols:
            return None, None, False
        sub_y = ys[mask]
        local = torch.argsort(sub_y, dim=0)
        global_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        ordered = global_idx[local]
        col_index[ordered] = torch.arange(num_cols, device=device, dtype=torch.long)

    row_1based = row_ids.to(dtype=torch.long) + 1
    col_1based = col_index.to(dtype=torch.long) + 1
    return row_1based, col_1based, True


def _grid_min_center_separation(xy_row: torch.Tensor) -> float:
    if int(xy_row.shape[0]) < 2:
        return float("inf")
    delta = xy_row.unsqueeze(1) - xy_row.unsqueeze(0)
    dist = torch.linalg.vector_norm(delta, dim=-1)
    mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
    if not torch.any(mask):
        return float("inf")
    return float(dist[mask].min().item())


def build_state_reindexing_intervention_plan(
    initial_xy: torch.Tensor,
    *,
    min_reorder_gap: float,
    script_name: str,
    order_axis: str = DEFAULT_POST_ACTION_RELOCATION_ORDER_AXIS,
    active_object_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | bool | str]:
    device = initial_xy.device
    num_envs = int(initial_xy.shape[0])
    num_objects = int(initial_xy.shape[1]) if initial_xy.ndim >= 2 else 0
    axis_index = _order_axis_index(order_axis)
    other_axis_index = 1 - axis_index

    summary = _active_order_summary(initial_xy, order_axis=order_axis, active_object_mask=active_object_mask)
    initial_rank_indices = summary["rank_indices"]
    active_counts = summary["active_counts"]
    normalized_active_object_mask = summary["active_object_mask"]
    moved_object_indices = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    anchor_object_indices = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    target_xy = torch.zeros((num_envs, 2), dtype=initial_xy.dtype, device=device)
    reorder_gap = torch.clamp(summary["mean_gap"], min=float(min_reorder_gap))
    expected_terminal_rank_position = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    planned_moved_mask = torch.zeros((num_envs, num_objects), dtype=torch.bool, device=device)

    script_is_supported = script_name in SUPPORTED_STATE_REINDEXING_SCRIPT_NAMES
    if num_objects == 0:
        return {
            "initial_rank_indices": initial_rank_indices,
            "moved_object_indices": moved_object_indices,
            "anchor_object_indices": anchor_object_indices,
            "target_xy": target_xy,
            "reorder_gap": reorder_gap,
            "expected_terminal_rank_position": expected_terminal_rank_position,
            "script_name": script_name,
            "script_is_supported": script_is_supported,
            "order_axis": normalize_post_action_relocation_order_axis(order_axis),
            "planned_moved_mask": planned_moved_mask,
            "active_object_mask": normalized_active_object_mask,
        }

    if script_name == STATIC_SCENE_NOOP_SCRIPT_NAME:
        return {
            "initial_rank_indices": initial_rank_indices,
            "moved_object_indices": torch.full((num_envs,), -1, dtype=torch.long, device=device),
            "anchor_object_indices": torch.full((num_envs,), -1, dtype=torch.long, device=device),
            "target_xy": torch.zeros((num_envs, 2), dtype=initial_xy.dtype, device=device),
            "reorder_gap": reorder_gap,
            "expected_terminal_rank_position": torch.full((num_envs,), -1, dtype=torch.long, device=device),
            "script_name": script_name,
            "script_is_supported": True,
            "order_axis": normalize_post_action_relocation_order_axis(order_axis),
            "planned_moved_mask": planned_moved_mask,
            "active_object_mask": normalized_active_object_mask,
        }

    batch_index = torch.arange(num_envs, device=device)
    leftmost_positions = torch.zeros_like(active_counts)
    rightmost_positions = active_counts - 1
    leftmost_indices = _gather_rank_indices(initial_rank_indices, leftmost_positions)
    rightmost_indices = _gather_rank_indices(initial_rank_indices, rightmost_positions)

    if script_is_supported and script_name == "leftmost_to_right_of_rightmost":
        valid_mask = active_counts >= 3
        moved_object_indices = torch.where(valid_mask, leftmost_indices, moved_object_indices)
        anchor_object_indices = torch.where(valid_mask, rightmost_indices, anchor_object_indices)
        target_xy[valid_mask, axis_index] = (
            initial_xy[batch_index[valid_mask], rightmost_indices[valid_mask], axis_index] + reorder_gap[valid_mask]
        )
        target_xy[valid_mask, other_axis_index] = initial_xy[
            batch_index[valid_mask], rightmost_indices[valid_mask], other_axis_index
        ]
        expected_terminal_rank_position[valid_mask] = active_counts[valid_mask] - 1

    elif script_is_supported and script_name == "rightmost_to_left_of_leftmost":
        valid_mask = active_counts >= 3
        moved_object_indices = torch.where(valid_mask, rightmost_indices, moved_object_indices)
        anchor_object_indices = torch.where(valid_mask, leftmost_indices, anchor_object_indices)
        target_xy[valid_mask, axis_index] = (
            initial_xy[batch_index[valid_mask], leftmost_indices[valid_mask], axis_index] - reorder_gap[valid_mask]
        )
        target_xy[valid_mask, other_axis_index] = initial_xy[
            batch_index[valid_mask], leftmost_indices[valid_mask], other_axis_index
        ]
        expected_terminal_rank_position[valid_mask] = 0

    elif script_is_supported and script_name == "rightmost_to_between_first_and_second":
        valid_mask = active_counts >= 3
        second_positions = torch.ones_like(active_counts)
        first_indices = _gather_rank_indices(initial_rank_indices, torch.zeros_like(active_counts))
        second_indices = _gather_rank_indices(initial_rank_indices, second_positions)
        moved_object_indices = torch.where(valid_mask, rightmost_indices, moved_object_indices)
        anchor_object_indices = torch.where(valid_mask, first_indices, anchor_object_indices)
        target_xy[valid_mask, axis_index] = 0.5 * (
            initial_xy[batch_index[valid_mask], first_indices[valid_mask], axis_index]
            + initial_xy[batch_index[valid_mask], second_indices[valid_mask], axis_index]
        )
        target_xy[valid_mask, other_axis_index] = 0.5 * (
            initial_xy[batch_index[valid_mask], first_indices[valid_mask], other_axis_index]
            + initial_xy[batch_index[valid_mask], second_indices[valid_mask], other_axis_index]
        )
        expected_terminal_rank_position[valid_mask] = 1

    elif script_is_supported and script_name == SWAP_ADJACENT_PAIRS_OVER_4_SCRIPT_NAME:
        valid_mask = (active_counts == 4) & (num_objects >= 4)
        if num_objects >= 4:
            r0 = initial_rank_indices[:, 0]
            r1 = initial_rank_indices[:, 1]
            r2 = initial_rank_indices[:, 2]
            r3 = initial_rank_indices[:, 3]
            if torch.any(valid_mask):
                vb = batch_index[valid_mask]
                planned_moved_mask[vb, r0[valid_mask]] = True
                planned_moved_mask[vb, r1[valid_mask]] = True
                planned_moved_mask[vb, r2[valid_mask]] = True
                planned_moved_mask[vb, r3[valid_mask]] = True
        moved_object_indices.fill_(-1)

    valid_moved_mask = moved_object_indices >= 0
    if torch.any(valid_moved_mask):
        planned_moved_mask[batch_index[valid_moved_mask], moved_object_indices[valid_moved_mask]] = True

    out: dict[str, torch.Tensor | bool | str] = {
        "initial_rank_indices": initial_rank_indices,
        "moved_object_indices": moved_object_indices,
        "anchor_object_indices": anchor_object_indices,
        "target_xy": target_xy,
        "reorder_gap": reorder_gap,
        "expected_terminal_rank_position": expected_terminal_rank_position,
        "script_name": script_name,
        "script_is_supported": script_is_supported,
        "order_axis": normalize_post_action_relocation_order_axis(order_axis),
        "planned_moved_mask": planned_moved_mask,
        "active_object_mask": normalized_active_object_mask,
    }
    if script_is_supported and script_name == SWAP_ADJACENT_PAIRS_OVER_4_SCRIPT_NAME:
        valid_mask = (active_counts == 4) & (num_objects >= 4)
        per_object_target_xy = initial_xy.clone()
        if num_objects >= 4 and torch.any(valid_mask):
            r0 = initial_rank_indices[:, 0]
            r1 = initial_rank_indices[:, 1]
            r2 = initial_rank_indices[:, 2]
            r3 = initial_rank_indices[:, 3]
            vb = batch_index[valid_mask]
            per_object_target_xy[vb, r0[valid_mask], :] = initial_xy[vb, r1[valid_mask], :]
            per_object_target_xy[vb, r1[valid_mask], :] = initial_xy[vb, r0[valid_mask], :]
            per_object_target_xy[vb, r2[valid_mask], :] = initial_xy[vb, r3[valid_mask], :]
            per_object_target_xy[vb, r3[valid_mask], :] = initial_xy[vb, r2[valid_mask], :]
        out["per_object_target_xy"] = per_object_target_xy
    return out


def build_historical_identity_tracking_intervention_plan(
    initial_xy: torch.Tensor,
    *,
    min_reorder_gap: float,
    script_name: str,
    order_axis: str = DEFAULT_POST_ACTION_RELOCATION_ORDER_AXIS,
    active_object_mask: torch.Tensor | None = None,
    swap_with_right_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | bool | str]:
    device = initial_xy.device
    num_envs = int(initial_xy.shape[0])
    num_objects = int(initial_xy.shape[1]) if initial_xy.ndim >= 2 else 0
    axis_index = _order_axis_index(order_axis)
    other_axis_index = 1 - axis_index

    summary = _active_order_summary(initial_xy, order_axis=order_axis, active_object_mask=active_object_mask)
    initial_rank_indices = summary["rank_indices"]
    active_counts = summary["active_counts"]
    normalized_active_object_mask = summary["active_object_mask"]
    moved_object_indices = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    planned_moved_mask = torch.zeros((num_envs, num_objects), dtype=torch.bool, device=device)
    # Enough columns for every runtime object (e.g. track_clutter: clutter + identity steps in one tensor).
    step_slots = max(4, num_objects)
    step_object_indices = torch.full((num_envs, step_slots), -1, dtype=torch.long, device=device)
    step_target_xy = torch.zeros((num_envs, step_slots, 2), dtype=initial_xy.dtype, device=device)
    reorder_gap = torch.clamp(summary["mean_gap"], min=float(min_reorder_gap))
    normalized_swap_with_right_mask = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    if isinstance(swap_with_right_mask, torch.Tensor) and tuple(swap_with_right_mask.shape[:1]) == (num_envs,):
        normalized_swap_with_right_mask = swap_with_right_mask.to(device=device, dtype=torch.bool)

    script_is_supported = script_name in SUPPORTED_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAMES
    if num_objects == 0:
        return {
            "initial_rank_indices": initial_rank_indices,
            "moved_object_indices": moved_object_indices,
            "planned_moved_mask": planned_moved_mask,
            "step_object_indices": step_object_indices,
            "step_target_xy": step_target_xy,
            "reorder_gap": reorder_gap,
            "script_name": script_name,
            "script_is_supported": script_is_supported,
            "order_axis": normalize_post_action_relocation_order_axis(order_axis),
            "active_object_mask": normalized_active_object_mask,
            "swap_with_right_mask": normalized_swap_with_right_mask,
        }

    batch_index = torch.arange(num_envs, device=device)

    if script_is_supported and script_name == SINGLE_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME:
        valid_mask = active_counts == 3
        leftmost_indices = _gather_rank_indices(initial_rank_indices, torch.zeros_like(active_counts))
        middle_indices = _gather_rank_indices(initial_rank_indices, torch.ones_like(active_counts))
        moved_object_indices = torch.where(valid_mask, middle_indices, moved_object_indices)
        step_object_indices[valid_mask, 0] = middle_indices[valid_mask]
        step_target_xy[valid_mask, 0, axis_index] = (
            initial_xy[batch_index[valid_mask], leftmost_indices[valid_mask], axis_index] - reorder_gap[valid_mask]
        )
        step_target_xy[valid_mask, 0, other_axis_index] = initial_xy[
            batch_index[valid_mask], leftmost_indices[valid_mask], other_axis_index
        ]
    elif script_is_supported and script_name == DYNAMIC_SWAP_FIRST_SECOND_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME:
        valid_mask = active_counts == 3
        leftmost_indices = _gather_rank_indices(initial_rank_indices, torch.zeros_like(active_counts))
        middle_indices = _gather_rank_indices(initial_rank_indices, torch.ones_like(active_counts))
        moved_object_indices = torch.where(valid_mask, middle_indices, moved_object_indices)
        step_object_indices[valid_mask, 0] = leftmost_indices[valid_mask]
        step_target_xy[valid_mask, 0, axis_index] = initial_xy[
            batch_index[valid_mask], middle_indices[valid_mask], axis_index
        ]
        step_target_xy[valid_mask, 0, other_axis_index] = initial_xy[
            batch_index[valid_mask], middle_indices[valid_mask], other_axis_index
        ]
        step_object_indices[valid_mask, 1] = middle_indices[valid_mask]
        step_target_xy[valid_mask, 1, axis_index] = initial_xy[
            batch_index[valid_mask], leftmost_indices[valid_mask], axis_index
        ]
        step_target_xy[valid_mask, 1, other_axis_index] = initial_xy[
            batch_index[valid_mask], leftmost_indices[valid_mask], other_axis_index
        ]
    elif script_is_supported and script_name in SEEDED_NEIGHBOR_SWAP_SCRIPT_NAMES:
        valid_mask = active_counts == 3
        leftmost_indices = _gather_rank_indices(initial_rank_indices, torch.zeros_like(active_counts))
        middle_indices = _gather_rank_indices(initial_rank_indices, torch.ones_like(active_counts))
        rightmost_indices = _gather_rank_indices(initial_rank_indices, active_counts - 1)
        effective_swap_with_right_mask = valid_mask & normalized_swap_with_right_mask
        partner_indices = torch.where(effective_swap_with_right_mask, rightmost_indices, leftmost_indices)
        moved_object_indices = torch.where(valid_mask, middle_indices, moved_object_indices)
        step_object_indices[valid_mask, 0] = partner_indices[valid_mask]
        step_target_xy[valid_mask, 0, axis_index] = initial_xy[
            batch_index[valid_mask], middle_indices[valid_mask], axis_index
        ]
        step_target_xy[valid_mask, 0, other_axis_index] = initial_xy[
            batch_index[valid_mask], middle_indices[valid_mask], other_axis_index
        ]
        step_object_indices[valid_mask, 1] = middle_indices[valid_mask]
        step_target_xy[valid_mask, 1, axis_index] = initial_xy[
            batch_index[valid_mask], partner_indices[valid_mask], axis_index
        ]
        step_target_xy[valid_mask, 1, other_axis_index] = initial_xy[
            batch_index[valid_mask], partner_indices[valid_mask], other_axis_index
        ]
    elif script_is_supported and script_name == DEFAULT_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME:
        valid_mask = active_counts == 3
        leftmost_indices = _gather_rank_indices(initial_rank_indices, torch.zeros_like(active_counts))
        middle_indices = _gather_rank_indices(initial_rank_indices, torch.ones_like(active_counts))
        rightmost_positions = active_counts - 1
        rightmost_indices = _gather_rank_indices(initial_rank_indices, rightmost_positions)
        moved_object_indices = torch.where(valid_mask, middle_indices, moved_object_indices)
        step_object_indices[valid_mask, 0] = middle_indices[valid_mask]
        step_target_xy[valid_mask, 0, axis_index] = (
            initial_xy[batch_index[valid_mask], leftmost_indices[valid_mask], axis_index] - reorder_gap[valid_mask]
        )
        step_target_xy[valid_mask, 0, other_axis_index] = initial_xy[
            batch_index[valid_mask], leftmost_indices[valid_mask], other_axis_index
        ]
        step_object_indices[valid_mask, 1] = rightmost_indices[valid_mask]
        step_target_xy[valid_mask, 1, axis_index] = (
            initial_xy[batch_index[valid_mask], leftmost_indices[valid_mask], axis_index] - 2.0 * reorder_gap[valid_mask]
        )
        step_target_xy[valid_mask, 1, other_axis_index] = initial_xy[
            batch_index[valid_mask], leftmost_indices[valid_mask], other_axis_index
        ]
        # Clutter-robust variants include extra non-target objects in the scene (num_objects > 3):
        # move the third identity object as well to increase desktop scrambling while preserving GT rank semantics.
        clutter_valid_mask = valid_mask & (num_objects > 3)
        if torch.any(clutter_valid_mask):
            step_object_indices[clutter_valid_mask, 2] = leftmost_indices[clutter_valid_mask]
            step_target_xy[clutter_valid_mask, 2, axis_index] = initial_xy[
                batch_index[clutter_valid_mask], middle_indices[clutter_valid_mask], axis_index
            ]
            step_target_xy[clutter_valid_mask, 2, other_axis_index] = initial_xy[
                batch_index[clutter_valid_mask], middle_indices[clutter_valid_mask], other_axis_index
            ]
    elif script_is_supported and script_name == MULTI_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME:
        valid_mask = active_counts == 5
        leftmost_indices = _gather_rank_indices(initial_rank_indices, torch.zeros_like(active_counts))
        middle_left_indices = _gather_rank_indices(initial_rank_indices, torch.full_like(active_counts, 2))
        rightmost_positions = active_counts - 1
        rightmost_indices = _gather_rank_indices(initial_rank_indices, rightmost_positions)
        moved_object_indices = torch.where(valid_mask, leftmost_indices, moved_object_indices)
        step_object_indices[valid_mask, 0] = leftmost_indices[valid_mask]
        step_target_xy[valid_mask, 0, axis_index] = initial_xy[
            batch_index[valid_mask], rightmost_indices[valid_mask], axis_index
        ]
        step_target_xy[valid_mask, 0, other_axis_index] = initial_xy[
            batch_index[valid_mask], rightmost_indices[valid_mask], other_axis_index
        ]
        step_object_indices[valid_mask, 1] = rightmost_indices[valid_mask]
        step_target_xy[valid_mask, 1, axis_index] = initial_xy[
            batch_index[valid_mask], middle_left_indices[valid_mask], axis_index
        ]
        step_target_xy[valid_mask, 1, other_axis_index] = initial_xy[
            batch_index[valid_mask], middle_left_indices[valid_mask], other_axis_index
        ]
        step_object_indices[valid_mask, 2] = middle_left_indices[valid_mask]
        step_target_xy[valid_mask, 2, axis_index] = initial_xy[
            batch_index[valid_mask], leftmost_indices[valid_mask], axis_index
        ]
        step_target_xy[valid_mask, 2, other_axis_index] = initial_xy[
            batch_index[valid_mask], leftmost_indices[valid_mask], other_axis_index
        ]

    for step_index in range(int(step_object_indices.shape[1])):
        step_indices = step_object_indices[:, step_index]
        valid_step_mask = step_indices >= 0
        if torch.any(valid_step_mask):
            planned_moved_mask[batch_index[valid_step_mask], step_indices[valid_step_mask]] = True

    return {
        "initial_rank_indices": initial_rank_indices,
        "moved_object_indices": moved_object_indices,
        "planned_moved_mask": planned_moved_mask,
        "step_object_indices": step_object_indices,
        "step_target_xy": step_target_xy,
        "reorder_gap": reorder_gap,
        "script_name": script_name,
        "script_is_supported": script_is_supported,
        "order_axis": normalize_post_action_relocation_order_axis(order_axis),
        "active_object_mask": normalized_active_object_mask,
        "swap_with_right_mask": normalized_swap_with_right_mask,
    }


def _compute_nearest_planar_left_gt_indices(
    post_xy: torch.Tensor,
    *,
    ref_idx: int,
    order_axis: str,
    intervention_success: torch.Tensor,
    left_eps: float = 1e-4,
    tie_eps: float = 5e-4,
) -> torch.Tensor:
    """Pick the active object strictly ``left`` of the reference on ``order_axis`` with smallest planar distance.

    ``left`` means smaller coordinate on the order axis (same convention as rank ordering).
    """
    device = post_xy.device
    num_envs = int(post_xy.shape[0])
    num_objects = int(post_xy.shape[1]) if post_xy.ndim >= 2 else 0
    out = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    if ref_idx < 0 or num_objects == 0:
        return out
    axis_i = _order_axis_index(order_axis)
    ref_xy = post_xy[:, ref_idx, :2]
    ref_coord = post_xy[:, ref_idx, axis_i]
    for e in range(num_envs):
        if not bool(intervention_success[e].item()):
            continue
        best_j = -1
        best_d = float("inf")
        ties = 0
        r_c = float(ref_coord[e].item())
        r_vec = ref_xy[e]
        for j in range(num_objects):
            if j == ref_idx:
                continue
            c = float(post_xy[e, j, axis_i].item())
            if c >= r_c - left_eps:
                continue
            d = float(torch.norm(post_xy[e, j, :2] - r_vec).item())
            if d < best_d - tie_eps:
                best_d = d
                best_j = j
                ties = 1
            elif abs(d - best_d) <= tie_eps:
                ties += 1
        if best_j >= 0 and ties == 1:
            out[e] = best_j
    return out


def evaluate_state_reindexing_terminal_state(
    *,
    initial_xy: torch.Tensor,
    post_xy: torch.Tensor,
    initial_rank_indices: torch.Tensor,
    moved_object_indices: torch.Tensor,
    displacement_tol: float,
    script_name: str,
    target_rank: str,
    order_axis: str = DEFAULT_POST_ACTION_RELOCATION_ORDER_AXIS,
    active_object_mask: torch.Tensor | None = None,
    gt_spec: dict[str, Any] | None = None,
    planned_moved_mask: torch.Tensor | None = None,
    object_names: Sequence[str] | None = None,
) -> dict[str, torch.Tensor | bool | str]:
    device = post_xy.device
    num_envs = int(post_xy.shape[0])
    num_objects = int(post_xy.shape[1]) if post_xy.ndim >= 2 else 0

    summary_initial = _active_order_summary(initial_xy, order_axis=order_axis, active_object_mask=active_object_mask)
    summary_post = _active_order_summary(post_xy, order_axis=order_axis, active_object_mask=active_object_mask)
    active_counts = summary_post["active_counts"]
    normalized_active_object_mask = summary_post["active_object_mask"]

    post_rank_indices = summary_post["rank_indices"]
    gt_object_indices = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    intervention_success = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    intervention_failure = torch.ones((num_envs,), dtype=torch.bool, device=device)
    gt_resolution_failure = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    valid_episode = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    max_non_target_planar_displacement = torch.zeros((num_envs,), dtype=post_xy.dtype, device=device)
    max_distractor_planar_displacement = torch.zeros((num_envs,), dtype=post_xy.dtype, device=device)
    accidental_displacement = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    invalid_due_to_confound = torch.zeros((num_envs,), dtype=torch.bool, device=device)

    script_is_supported = script_name in SUPPORTED_STATE_REINDEXING_SCRIPT_NAMES
    target_rank_is_supported = _resolve_rank_descriptor_supported(target_rank)
    if num_objects == 0:
        return {
            "post_rank_indices": post_rank_indices,
            "gt_object_indices": gt_object_indices,
            "intervention_success": intervention_success,
            "intervention_failure": intervention_failure,
            "gt_resolution_failure": gt_resolution_failure,
            "valid_episode": valid_episode,
            "max_non_target_planar_displacement": max_non_target_planar_displacement,
            "max_distractor_planar_displacement": max_distractor_planar_displacement,
            "accidental_displacement": accidental_displacement,
            "invalid_due_to_confound": invalid_due_to_confound,
            "script_name": script_name,
            "script_is_supported": script_is_supported,
            "target_rank": target_rank,
            "target_rank_is_supported": target_rank_is_supported,
            "order_axis": normalize_post_action_relocation_order_axis(order_axis),
            "active_object_mask": normalized_active_object_mask,
        }

    spec_dict: dict[str, Any] = gt_spec if isinstance(gt_spec, dict) else {}
    gt_resolver = str(spec_dict.get("selection_resolver", spec_dict.get("resolver", ""))).strip()

    if script_name == STATIC_SCENE_NOOP_SCRIPT_NAME and gt_resolver == GT_RESOLVER_RANK_INITIAL_LAYOUT:
        max_non_target_planar_displacement = _masked_planar_displacement(
            initial_xy,
            post_xy,
            active_mask=normalized_active_object_mask,
        )
        max_distractor_planar_displacement = _masked_planar_displacement(
            initial_xy,
            post_xy,
            active_mask=~normalized_active_object_mask,
        )
        accidental_displacement = max_non_target_planar_displacement > float(displacement_tol)
        order_unchanged = _rank_rows_match(post_rank_indices, initial_rank_indices, active_counts)
        intervention_success = (
            bool(script_is_supported)
            & summary_initial["strictly_increasing"]
            & summary_post["strictly_increasing"]
            & order_unchanged
        )
        intervention_failure = ~intervention_success
        target_rank_positions, target_rank_row_supported = _resolve_target_rank_position(target_rank, active_counts)
        resolved_gt_indices = _gather_rank_indices(post_rank_indices, target_rank_positions)
        gt_object_indices = torch.where(
            intervention_success & target_rank_row_supported,
            resolved_gt_indices,
            torch.full_like(resolved_gt_indices, -1),
        )
        gt_resolution_failure = intervention_success & (gt_object_indices < 0)
        invalid_due_to_confound = intervention_success & accidental_displacement
        valid_episode = intervention_success & ~accidental_displacement & ~gt_resolution_failure
        return {
            "post_rank_indices": post_rank_indices,
            "gt_object_indices": gt_object_indices,
            "intervention_success": intervention_success,
            "intervention_failure": intervention_failure,
            "gt_resolution_failure": gt_resolution_failure,
            "valid_episode": valid_episode,
            "max_non_target_planar_displacement": max_non_target_planar_displacement,
            "max_distractor_planar_displacement": max_distractor_planar_displacement,
            "accidental_displacement": accidental_displacement,
            "invalid_due_to_confound": invalid_due_to_confound,
            "script_name": script_name,
            "script_is_supported": script_is_supported,
            "target_rank": target_rank,
            "target_rank_is_supported": target_rank_is_supported,
            "order_axis": normalize_post_action_relocation_order_axis(order_axis),
            "active_object_mask": normalized_active_object_mask,
        }

    if script_name == STATIC_SCENE_NOOP_SCRIPT_NAME and gt_resolver == GT_RESOLVER_GRID_ROWCOL_SLOT:
        max_non_target_planar_displacement = _masked_planar_displacement(
            initial_xy,
            post_xy,
            active_mask=normalized_active_object_mask,
        )
        max_distractor_planar_displacement = _masked_planar_displacement(
            initial_xy,
            post_xy,
            active_mask=~normalized_active_object_mask,
        )
        accidental_displacement = max_non_target_planar_displacement > float(displacement_tol)

        num_rows = int(spec_dict.get("grid_num_rows", 2))
        num_cols = int(spec_dict.get("grid_num_cols", 3))
        target_row = int(spec_dict.get("target_row", 1))
        target_col = int(spec_dict.get("target_col", 2))
        row_axis = str(spec_dict.get("row_axis", "x"))
        col_axis = str(spec_dict.get("col_axis", "y"))
        quant_step = float(spec_dict.get("grid_quant_step", 0.028))
        min_center_sep = float(spec_dict.get("min_center_separation", 0.024))

        for env_index in range(num_envs):
            mask_e = normalized_active_object_mask[env_index]
            idx = torch.nonzero(mask_e, as_tuple=False).squeeze(-1)
            if int(idx.numel()) != num_rows * num_cols:
                continue
            xy_post_sel = post_xy[env_index, idx]
            xy_init_sel = initial_xy[env_index, idx]
            if torch.linalg.vector_norm(xy_post_sel - xy_init_sel, dim=-1).max() > float(displacement_tol):
                continue
            row_1based, col_1based, assign_ok = _grid_slot_assignment_for_env(
                xy_post_sel,
                num_rows=num_rows,
                num_cols=num_cols,
                row_axis=row_axis,
                col_axis=col_axis,
                quant_step=quant_step,
            )
            if not assign_ok or row_1based is None or col_1based is None:
                continue
            matches = (row_1based == target_row) & (col_1based == target_col)
            if int(matches.sum().item()) != 1:
                continue
            if _grid_min_center_separation(xy_post_sel) + 1.0e-9 < min_center_sep:
                continue
            pick = int(torch.nonzero(matches, as_tuple=False)[0].item())
            gt_object_indices[env_index] = int(idx[pick].item())
            intervention_success[env_index] = True

        intervention_failure = ~intervention_success
        gt_resolution_failure = intervention_success & (gt_object_indices < 0)
        invalid_due_to_confound = intervention_success & accidental_displacement
        valid_episode = intervention_success & ~accidental_displacement & ~gt_resolution_failure
        return {
            "post_rank_indices": post_rank_indices,
            "gt_object_indices": gt_object_indices,
            "intervention_success": intervention_success,
            "intervention_failure": intervention_failure,
            "gt_resolution_failure": gt_resolution_failure,
            "valid_episode": valid_episode,
            "max_non_target_planar_displacement": max_non_target_planar_displacement,
            "max_distractor_planar_displacement": max_distractor_planar_displacement,
            "accidental_displacement": accidental_displacement,
            "invalid_due_to_confound": invalid_due_to_confound,
            "script_name": script_name,
            "script_is_supported": script_is_supported,
            "target_rank": target_rank,
            "target_rank_is_supported": target_rank_is_supported,
            "order_axis": normalize_post_action_relocation_order_axis(order_axis),
            "active_object_mask": normalized_active_object_mask,
        }

    if script_name == SWAP_ADJACENT_PAIRS_OVER_4_SCRIPT_NAME:
        expected_post_rank, swap_supported = _expected_post_rank_indices_after_adjacent_pair_swaps(
            initial_rank_indices, active_counts
        )
        reached_swap = _rank_rows_match(post_rank_indices, expected_post_rank, active_counts)
        intervention_success = (
            bool(script_is_supported)
            & summary_initial["strictly_increasing"]
            & summary_post["strictly_increasing"]
            & swap_supported
            & reached_swap
        )
        intervention_failure = ~intervention_success
        batch_index = torch.arange(num_envs, device=device)
        if planned_moved_mask is not None:
            pm = planned_moved_mask.to(device=device, dtype=torch.bool)
            active_non_target_mask = normalized_active_object_mask & ~pm
        else:
            mover_mask = torch.zeros_like(normalized_active_object_mask)
            if torch.any(swap_supported):
                vb = batch_index[swap_supported]
                for k in range(4):
                    oi = initial_rank_indices[swap_supported, k]
                    mover_mask[vb, oi] = True
            active_non_target_mask = normalized_active_object_mask & ~mover_mask
        max_non_target_planar_displacement = _masked_planar_displacement(
            initial_xy,
            post_xy,
            active_mask=active_non_target_mask,
        )
        max_distractor_planar_displacement = _masked_planar_displacement(
            initial_xy,
            post_xy,
            active_mask=~normalized_active_object_mask,
        )
        accidental_displacement = max_non_target_planar_displacement > float(displacement_tol)
        target_rank_positions, target_rank_row_supported = _resolve_target_rank_position(target_rank, active_counts)
        resolved_gt_indices = _gather_rank_indices(expected_post_rank, target_rank_positions)
        gt_object_indices = torch.where(
            target_rank_row_supported,
            resolved_gt_indices,
            torch.full_like(resolved_gt_indices, -1),
        )
        gt_resolution_failure = intervention_success & (gt_object_indices < 0)
        invalid_due_to_confound = intervention_success & accidental_displacement
        valid_episode = intervention_success & ~accidental_displacement & ~gt_resolution_failure
        return {
            "post_rank_indices": post_rank_indices,
            "gt_object_indices": gt_object_indices,
            "intervention_success": intervention_success,
            "intervention_failure": intervention_failure,
            "gt_resolution_failure": gt_resolution_failure,
            "valid_episode": valid_episode,
            "max_non_target_planar_displacement": max_non_target_planar_displacement,
            "max_distractor_planar_displacement": max_distractor_planar_displacement,
            "accidental_displacement": accidental_displacement,
            "invalid_due_to_confound": invalid_due_to_confound,
            "script_name": script_name,
            "script_is_supported": script_is_supported,
            "target_rank": target_rank,
            "target_rank_is_supported": target_rank_is_supported,
            "order_axis": normalize_post_action_relocation_order_axis(order_axis),
            "active_object_mask": normalized_active_object_mask,
        }

    expected_terminal_rank_position, script_rank_supported = _expected_terminal_rank_position(script_name, active_counts)
    reached_expected_object_indices = _gather_rank_indices(post_rank_indices, expected_terminal_rank_position)
    valid_moved_mask = moved_object_indices >= 0
    moved_reached_expected_rank = valid_moved_mask & (reached_expected_object_indices == moved_object_indices)
    if script_name == SWAP_ADJACENT_PAIRS_OVER_4_SCRIPT_NAME:
        expected_post_rank_indices, pair_swap_supported = _expected_post_rank_indices_after_adjacent_pair_swaps(
            initial_rank_indices,
            active_counts,
        )
        reached_pair_swap_order = _rank_rows_match(post_rank_indices, expected_post_rank_indices, active_counts)
        max_non_target_planar_displacement = _masked_planar_displacement(
            initial_xy,
            post_xy,
            active_mask=torch.zeros_like(normalized_active_object_mask, dtype=torch.bool, device=device),
        )
        max_distractor_planar_displacement = _masked_planar_displacement(
            initial_xy,
            post_xy,
            active_mask=~normalized_active_object_mask,
        )
        accidental_displacement = max_non_target_planar_displacement > float(displacement_tol)
        intervention_success = (
            bool(script_is_supported)
            & summary_initial["strictly_increasing"]
            & summary_post["strictly_increasing"]
            & pair_swap_supported
            & reached_pair_swap_order
        )
        intervention_failure = ~intervention_success
        target_rank_positions, target_rank_row_supported = _resolve_target_rank_position(target_rank, active_counts)
        resolved_gt_indices = _gather_rank_indices(expected_post_rank_indices, target_rank_positions)
        gt_object_indices = torch.where(
            target_rank_row_supported,
            resolved_gt_indices,
            torch.full_like(resolved_gt_indices, -1),
        )
        gt_resolution_failure = intervention_success & (gt_object_indices < 0)
        invalid_due_to_confound = intervention_success & accidental_displacement
        valid_episode = intervention_success & ~accidental_displacement & ~gt_resolution_failure
        return {
            "post_rank_indices": post_rank_indices,
            "gt_object_indices": gt_object_indices,
            "intervention_success": intervention_success,
            "intervention_failure": intervention_failure,
            "gt_resolution_failure": gt_resolution_failure,
            "valid_episode": valid_episode,
            "max_non_target_planar_displacement": max_non_target_planar_displacement,
            "max_distractor_planar_displacement": max_distractor_planar_displacement,
            "accidental_displacement": accidental_displacement,
            "invalid_due_to_confound": invalid_due_to_confound,
            "script_name": script_name,
            "script_is_supported": script_is_supported,
            "target_rank": target_rank,
            "target_rank_is_supported": target_rank_is_supported,
            "order_axis": normalize_post_action_relocation_order_axis(order_axis),
            "active_object_mask": normalized_active_object_mask,
        }

    active_non_target_mask = normalized_active_object_mask.clone()
    batch_index = torch.arange(num_envs, device=device)
    if torch.any(valid_moved_mask):
        active_non_target_mask[batch_index[valid_moved_mask], moved_object_indices[valid_moved_mask]] = False
    if planned_moved_mask is not None:
        pm = planned_moved_mask.to(device=device, dtype=torch.bool)
        if tuple(pm.shape[:2]) == tuple(normalized_active_object_mask.shape[:2]):
            active_non_target_mask = normalized_active_object_mask & ~pm
    max_non_target_planar_displacement = _masked_planar_displacement(
        initial_xy,
        post_xy,
        active_mask=active_non_target_mask,
    )
    max_distractor_planar_displacement = _masked_planar_displacement(
        initial_xy,
        post_xy,
        active_mask=~normalized_active_object_mask,
    )
    accidental_displacement = max_non_target_planar_displacement > float(displacement_tol)

    intervention_success = (
        bool(script_is_supported)
        & summary_initial["strictly_increasing"]
        & summary_post["strictly_increasing"]
        & script_rank_supported
        & valid_moved_mask
        & moved_reached_expected_rank
    )
    intervention_failure = ~intervention_success

    if gt_resolver == "structural_adjacency":
        gt_object_indices = structural_adjacency_indices(post_xy, spec_dict, list(object_names or []))
        gt_object_indices = torch.where(intervention_success, gt_object_indices, torch.full_like(gt_object_indices, -1))
    elif gt_resolver == GT_RESOLVER_NEAREST_PLANAR_LEFT_OF_REFERENCE:
        ref_name = str(spec_dict.get("reference_object_id", "red_anchor_block")).strip()
        names_list = list(object_names) if object_names is not None else []
        ref_idx = names_list.index(ref_name) if ref_name in names_list else -1
        gt_object_indices = _compute_nearest_planar_left_gt_indices(
            post_xy,
            ref_idx=ref_idx,
            order_axis=order_axis,
            intervention_success=intervention_success,
        )
    else:
        target_rank_positions, target_rank_row_supported = _resolve_target_rank_position(target_rank, active_counts)
        resolved_gt_indices = _gather_rank_indices(post_rank_indices, target_rank_positions)
        gt_object_indices = torch.where(
            intervention_success & target_rank_row_supported,
            resolved_gt_indices,
            torch.full_like(resolved_gt_indices, -1),
        )

    gt_resolution_failure = intervention_success & (gt_object_indices < 0)
    invalid_due_to_confound = intervention_success & accidental_displacement
    valid_episode = intervention_success & ~accidental_displacement & ~gt_resolution_failure

    return {
        "post_rank_indices": post_rank_indices,
        "gt_object_indices": gt_object_indices,
        "intervention_success": intervention_success,
        "intervention_failure": intervention_failure,
        "gt_resolution_failure": gt_resolution_failure,
        "valid_episode": valid_episode,
        "max_non_target_planar_displacement": max_non_target_planar_displacement,
        "max_distractor_planar_displacement": max_distractor_planar_displacement,
        "accidental_displacement": accidental_displacement,
        "invalid_due_to_confound": invalid_due_to_confound,
        "script_name": script_name,
        "script_is_supported": script_is_supported,
        "target_rank": target_rank,
        "target_rank_is_supported": target_rank_is_supported,
        "order_axis": normalize_post_action_relocation_order_axis(order_axis),
        "active_object_mask": normalized_active_object_mask,
    }


def evaluate_historical_identity_tracking_terminal_state(
    *,
    initial_xy: torch.Tensor,
    post_xy: torch.Tensor,
    initial_rank_indices: torch.Tensor,
    planned_moved_mask: torch.Tensor,
    displacement_tol: float,
    script_name: str,
    target_rank: str,
    order_axis: str = DEFAULT_POST_ACTION_RELOCATION_ORDER_AXIS,
    active_object_mask: torch.Tensor | None = None,
    swap_with_right_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | bool | str]:
    device = post_xy.device
    num_envs = int(post_xy.shape[0])
    num_objects = int(post_xy.shape[1]) if post_xy.ndim >= 2 else 0

    summary_initial = _active_order_summary(initial_xy, order_axis=order_axis, active_object_mask=active_object_mask)
    summary_post = _active_order_summary(post_xy, order_axis=order_axis, active_object_mask=active_object_mask)
    active_counts = summary_post["active_counts"]
    normalized_active_object_mask = summary_post["active_object_mask"]

    post_rank_indices = summary_post["rank_indices"]
    gt_object_indices = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    intervention_success = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    intervention_failure = torch.ones((num_envs,), dtype=torch.bool, device=device)
    gt_resolution_failure = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    valid_episode = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    max_non_target_planar_displacement = torch.zeros((num_envs,), dtype=post_xy.dtype, device=device)
    max_distractor_planar_displacement = torch.zeros((num_envs,), dtype=post_xy.dtype, device=device)
    accidental_displacement = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    invalid_due_to_confound = torch.zeros((num_envs,), dtype=torch.bool, device=device)

    script_is_supported = script_name in SUPPORTED_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAMES
    target_rank_is_supported = _resolve_rank_descriptor_supported(target_rank)
    if num_objects == 0:
        return {
            "post_rank_indices": post_rank_indices,
            "gt_object_indices": gt_object_indices,
            "intervention_success": intervention_success,
            "intervention_failure": intervention_failure,
            "gt_resolution_failure": gt_resolution_failure,
            "valid_episode": valid_episode,
            "max_non_target_planar_displacement": max_non_target_planar_displacement,
            "max_distractor_planar_displacement": max_distractor_planar_displacement,
            "accidental_displacement": accidental_displacement,
            "invalid_due_to_confound": invalid_due_to_confound,
            "script_name": script_name,
            "script_is_supported": script_is_supported,
            "target_rank": target_rank,
            "target_rank_is_supported": target_rank_is_supported,
            "order_axis": normalize_post_action_relocation_order_axis(order_axis),
            "active_object_mask": normalized_active_object_mask,
        }

    expected_post_rank_indices, expected_order_supported = _expected_historical_rank_indices(
        initial_rank_indices,
        active_counts,
        script_name=script_name,
        swap_with_right_mask=swap_with_right_mask,
    )
    reached_expected_rank_order = _rank_rows_match(post_rank_indices, expected_post_rank_indices, active_counts)

    active_non_target_mask = normalized_active_object_mask & ~planned_moved_mask.to(
        device=device,
        dtype=torch.bool,
    )
    max_non_target_planar_displacement = _masked_planar_displacement(
        initial_xy,
        post_xy,
        active_mask=active_non_target_mask,
    )
    max_distractor_planar_displacement = _masked_planar_displacement(
        initial_xy,
        post_xy,
        active_mask=~normalized_active_object_mask,
    )
    accidental_displacement = max_non_target_planar_displacement > float(displacement_tol)

    intervention_success = (
        bool(script_is_supported)
        & summary_initial["strictly_increasing"]
        & summary_post["strictly_increasing"]
        & expected_order_supported
        & reached_expected_rank_order
    )
    intervention_failure = ~intervention_success

    target_rank_positions, target_rank_row_supported = _resolve_target_rank_position(target_rank, active_counts)
    resolved_gt_indices = _gather_rank_indices(initial_rank_indices, target_rank_positions)
    gt_object_indices = torch.where(
        target_rank_row_supported,
        resolved_gt_indices,
        torch.full_like(resolved_gt_indices, -1),
    )
    gt_resolution_failure = target_rank_row_supported & (gt_object_indices < 0)
    invalid_due_to_confound = intervention_success & accidental_displacement
    valid_episode = intervention_success & ~accidental_displacement & ~gt_resolution_failure

    return {
        "post_rank_indices": post_rank_indices,
        "gt_object_indices": gt_object_indices,
        "intervention_success": intervention_success,
        "intervention_failure": intervention_failure,
        "gt_resolution_failure": gt_resolution_failure,
        "valid_episode": valid_episode,
        "max_non_target_planar_displacement": max_non_target_planar_displacement,
        "max_distractor_planar_displacement": max_distractor_planar_displacement,
        "accidental_displacement": accidental_displacement,
        "invalid_due_to_confound": invalid_due_to_confound,
        "script_name": script_name,
        "script_is_supported": script_is_supported,
        "target_rank": target_rank,
        "target_rank_is_supported": target_rank_is_supported,
        "order_axis": normalize_post_action_relocation_order_axis(order_axis),
        "active_object_mask": normalized_active_object_mask,
    }


def evaluate_state_reindexing_readout(
    *,
    gt_object_indices: torch.Tensor,
    grasped_object_indices: torch.Tensor,
    valid_episode: torch.Tensor,
) -> dict[str, torch.Tensor]:
    success = (grasped_object_indices >= 0) & (grasped_object_indices == gt_object_indices)
    grasp_readout_failure = valid_episode & (gt_object_indices >= 0) & (grasped_object_indices < 0)
    valid_episode_task_failed = valid_episode & ~success & (grasped_object_indices >= 0)
    task_failed = valid_episode & ~success
    return {
        "success": success,
        "grasp_readout_failure": grasp_readout_failure,
        "valid_episode_task_failed": valid_episode_task_failed,
        "task_failed": task_failed,
    }


def build_state_reindexing_status_labels(
    *,
    intervention_failure: torch.Tensor,
    gt_resolution_failure: torch.Tensor,
    accidental_displacement: torch.Tensor,
    grasp_readout_failure: torch.Tensor,
    valid_episode_task_failed: torch.Tensor,
    success: torch.Tensor,
) -> list[str]:
    labels: list[str] = []
    for index in range(intervention_failure.shape[0]):
        if bool(intervention_failure[index].item()):
            labels.append("intervention_failed")
        elif bool(gt_resolution_failure[index].item()):
            labels.append("gt_resolution_failed")
        elif bool(accidental_displacement[index].item()):
            labels.append("invalid_due_to_confound")
        elif bool(grasp_readout_failure[index].item()):
            labels.append("grasp_readout_failed")
        elif bool(valid_episode_task_failed[index].item()):
            labels.append("valid_episode_but_task_failed")
        elif bool(success[index].item()):
            labels.append("success")
        else:
            labels.append("unknown")
    return labels
