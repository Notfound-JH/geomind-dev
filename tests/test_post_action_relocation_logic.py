from __future__ import annotations

import torch

from geomind_lab.mimic.envs.post_action_relocation_motion import (
    MOTION_FUNCTIONS,
    REINDEX_INSERT_SHALLOW_ARC,
    TRACK_SINGLE_PAIRED_ARC,
    get_post_action_relocation_motion_function,
    resolve_post_action_relocation_motion_mode,
)
from geomind_lab.mimic.envs.state_reindexing_logic import (
    MULTI_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
    SINGLE_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
    SWAP_ADJACENT_PAIRS_OVER_4_SCRIPT_NAME,
    build_historical_identity_tracking_intervention_plan,
    build_state_reindexing_intervention_plan,
    evaluate_historical_identity_tracking_terminal_state,
    evaluate_state_reindexing_terminal_state,
)


def _xy(*ys: float) -> torch.Tensor:
    rows = [[[0.0, float(value)] for value in ys]]
    return torch.tensor(rows, dtype=torch.float32)


def test_motion_mode_selects_portable_track_single_function() -> None:
    mode = resolve_post_action_relocation_motion_mode(
        {"motion_mode": "track_single_paired_arc"},
        default=REINDEX_INSERT_SHALLOW_ARC,
    )
    assert mode == TRACK_SINGLE_PAIRED_ARC
    assert get_post_action_relocation_motion_function(mode).__name__ == "track_single_paired_arc_xy"


def test_track_single_arc_preserves_endpoints_and_curves_midpoint() -> None:
    start_xy = _xy(-0.1, 0.1)
    target_xy = _xy(0.1, -0.1)
    object_indices = torch.tensor([[0, 1]], dtype=torch.long)
    motion_fn = get_post_action_relocation_motion_function(TRACK_SINGLE_PAIRED_ARC)

    at_start = motion_fn(start_xy, target_xy, object_indices, torch.tensor([0.0]))
    at_middle = motion_fn(start_xy, target_xy, object_indices, torch.tensor([0.5]))
    at_end = motion_fn(start_xy, target_xy, object_indices, torch.tensor([1.0]))

    assert torch.allclose(at_start, start_xy, atol=1e-6)
    assert torch.allclose(at_end, target_xy, atol=1e-6)
    assert abs(float(at_middle[0, 0, 0].item())) > 0.05


def test_registered_motion_functions_preserve_start_and_target_endpoints() -> None:
    start_xy = _xy(-0.1, 0.1, 0.2, 0.3)
    target_xy = _xy(0.1, -0.1, 0.3, 0.2)
    object_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    object_names = ("red_subset_block_2", "red_subset_block_1", "other_0", "other_1")

    for motion_function in MOTION_FUNCTIONS.values():
        at_start = motion_function(
            start_xy,
            target_xy,
            object_indices,
            torch.tensor([0.0]),
            object_names=object_names,
        )
        at_end = motion_function(
            start_xy,
            target_xy,
            object_indices,
            torch.tensor([1.0]),
            object_names=object_names,
        )
        assert torch.allclose(at_start, start_xy, atol=1e-6), motion_function.__name__
        assert torch.allclose(at_end, target_xy, atol=1e-6), motion_function.__name__


def test_swap_adjacent_pairs_over_4_gt_is_post_swap_leftmost() -> None:
    initial_xy = _xy(0.0, 1.0, 2.0, 3.0)
    plan = build_state_reindexing_intervention_plan(
        initial_xy,
        min_reorder_gap=0.5,
        script_name=SWAP_ADJACENT_PAIRS_OVER_4_SCRIPT_NAME,
        order_axis="y",
    )
    assert bool(plan["script_is_supported"]) is True
    post_xy = plan["per_object_target_xy"].clone()
    terminal_state = evaluate_state_reindexing_terminal_state(
        initial_xy=initial_xy,
        post_xy=post_xy,
        initial_rank_indices=plan["initial_rank_indices"],
        moved_object_indices=plan["moved_object_indices"],
        displacement_tol=0.05,
        script_name=SWAP_ADJACENT_PAIRS_OVER_4_SCRIPT_NAME,
        target_rank="leftmost",
        order_axis="y",
        planned_moved_mask=plan["planned_moved_mask"],
    )
    assert bool(terminal_state["intervention_success"][0].item()) is True
    assert bool(terminal_state["valid_episode"][0].item()) is True
    # 交换后沿 y 最左为原先第二个物体（下标 1）
    assert int(terminal_state["gt_object_indices"][0].item()) == 1


def test_object_position_update_insert_second_rank() -> None:
    initial_xy = _xy(0.0, 1.0, 2.0, 3.0)
    plan = build_state_reindexing_intervention_plan(
        initial_xy,
        min_reorder_gap=0.5,
        script_name="rightmost_to_between_first_and_second",
        order_axis="y",
    )

    assert bool(plan["script_is_supported"]) is True
    assert int(plan["moved_object_indices"][0].item()) == 3

    post_xy = initial_xy.clone()
    post_xy[0, 3, 1] = 0.5
    terminal_state = evaluate_state_reindexing_terminal_state(
        initial_xy=initial_xy,
        post_xy=post_xy,
        initial_rank_indices=plan["initial_rank_indices"],
        moved_object_indices=plan["moved_object_indices"],
        displacement_tol=0.05,
        script_name="rightmost_to_between_first_and_second",
        target_rank="index_1",
        order_axis="y",
    )

    assert bool(terminal_state["intervention_success"][0].item()) is True
    assert bool(terminal_state["valid_episode"][0].item()) is True
    assert int(terminal_state["gt_object_indices"][0].item()) == 3


def test_object_identity_tracking_single_step_uses_initial_middle() -> None:
    initial_xy = _xy(0.0, 1.0, 2.0)
    plan = build_historical_identity_tracking_intervention_plan(
        initial_xy,
        min_reorder_gap=0.5,
        script_name=SINGLE_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
        order_axis="y",
    )

    assert bool(plan["script_is_supported"]) is True
    assert int(plan["moved_object_indices"][0].item()) == 1

    post_xy = initial_xy.clone()
    post_xy[0, 1, 1] = -1.0
    terminal_state = evaluate_historical_identity_tracking_terminal_state(
        initial_xy=initial_xy,
        post_xy=post_xy,
        initial_rank_indices=plan["initial_rank_indices"],
        planned_moved_mask=plan["planned_moved_mask"],
        displacement_tol=0.05,
        script_name=SINGLE_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
        target_rank="middle",
        order_axis="y",
    )

    assert bool(terminal_state["intervention_success"][0].item()) is True
    assert bool(terminal_state["valid_episode"][0].item()) is True
    assert int(terminal_state["gt_object_indices"][0].item()) == 1


def test_object_identity_tracking_multistep_uses_initial_leftmost() -> None:
    initial_xy = _xy(0.0, 1.0, 2.0, 3.0, 4.0)
    plan = build_historical_identity_tracking_intervention_plan(
        initial_xy,
        min_reorder_gap=0.5,
        script_name=MULTI_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
        order_axis="y",
    )

    assert bool(plan["script_is_supported"]) is True
    assert int(plan["moved_object_indices"][0].item()) == 0
    assert int((plan["step_object_indices"][0] >= 0).sum().item()) == 3

    post_xy = initial_xy.clone()
    post_xy[0, 0, 1] = 4.0
    post_xy[0, 4, 1] = 2.0
    post_xy[0, 2, 1] = 0.0
    terminal_state = evaluate_historical_identity_tracking_terminal_state(
        initial_xy=initial_xy,
        post_xy=post_xy,
        initial_rank_indices=plan["initial_rank_indices"],
        planned_moved_mask=plan["planned_moved_mask"],
        displacement_tol=0.05,
        script_name=MULTI_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
        target_rank="leftmost",
        order_axis="y",
    )

    assert bool(terminal_state["intervention_success"][0].item()) is True
    assert bool(terminal_state["valid_episode"][0].item()) is True
    assert int(terminal_state["gt_object_indices"][0].item()) == 0


def test_reorder_with_distractors_uses_subset_only() -> None:
    initial_xy = _xy(0.0, 1.0, 2.0, 0.1, 1.1, 2.1)
    active_mask = torch.tensor([[True, True, True, False, False, False]], dtype=torch.bool)
    plan = build_state_reindexing_intervention_plan(
        initial_xy,
        min_reorder_gap=0.5,
        script_name="leftmost_to_right_of_rightmost",
        order_axis="y",
        active_object_mask=active_mask,
    )

    assert bool(plan["script_is_supported"]) is True
    assert int(plan["moved_object_indices"][0].item()) == 0

    post_xy = initial_xy.clone()
    post_xy[0, 0, 1] = 3.0
    post_xy[0, 4, 1] = 1.7
    terminal_state = evaluate_state_reindexing_terminal_state(
        initial_xy=initial_xy,
        post_xy=post_xy,
        initial_rank_indices=plan["initial_rank_indices"],
        moved_object_indices=plan["moved_object_indices"],
        displacement_tol=0.05,
        script_name="leftmost_to_right_of_rightmost",
        target_rank="leftmost",
        order_axis="y",
        active_object_mask=active_mask,
    )

    assert bool(terminal_state["intervention_success"][0].item()) is True
    assert bool(terminal_state["valid_episode"][0].item()) is True
    assert int(terminal_state["gt_object_indices"][0].item()) == 1
    assert float(terminal_state["max_non_target_planar_displacement"][0].item()) == 0.0
    assert float(terminal_state["max_distractor_planar_displacement"][0].item()) > 0.0


def test_reorder_with_distractors_middle_after_subset_insertion() -> None:
    initial_xy = _xy(-0.14, -0.03, 0.11, -0.09, 0.035)
    active_mask = torch.tensor([[True, True, True, False, False]], dtype=torch.bool)
    plan = build_state_reindexing_intervention_plan(
        initial_xy,
        min_reorder_gap=0.08,
        script_name="rightmost_to_between_first_and_second",
        order_axis="y",
        active_object_mask=active_mask,
    )

    assert bool(plan["script_is_supported"]) is True
    assert int(plan["moved_object_indices"][0].item()) == 2

    post_xy = initial_xy.clone()
    post_xy[0, 2, 1] = -0.085
    terminal_state = evaluate_state_reindexing_terminal_state(
        initial_xy=initial_xy,
        post_xy=post_xy,
        initial_rank_indices=plan["initial_rank_indices"],
        moved_object_indices=plan["moved_object_indices"],
        displacement_tol=0.05,
        script_name="rightmost_to_between_first_and_second",
        target_rank="middle",
        order_axis="y",
        active_object_mask=active_mask,
    )

    assert bool(terminal_state["intervention_success"][0].item()) is True
    assert bool(terminal_state["valid_episode"][0].item()) is True
    assert int(terminal_state["gt_object_indices"][0].item()) == 2
