from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from geomind_lab.mimic.envs.state_reindexing_logic import (
    GT_RESOLVER_GRID_ROWCOL_SLOT,
    GT_RESOLVER_RANK_INITIAL_LAYOUT,
    STATIC_SCENE_NOOP_SCRIPT_NAME,
    build_state_reindexing_intervention_plan,
    build_state_reindexing_status_labels,
    evaluate_state_reindexing_readout,
    evaluate_state_reindexing_terminal_state,
)
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import (
    load_task_spec,
    resolve_task_instruction,
    resolve_task_protocol,
)


def _dummy_scene_profile_for_layout(layout_key: str) -> SimpleNamespace:
    payload = json.loads((REPO_ROOT / "assets" / "annotations" / "annotated_tasks.json").read_text(encoding="utf-8"))
    layout = payload[layout_key]
    scene_structure = layout["scene_structure"]
    object_positions = dict(scene_structure.get("object_positions", {}))
    object_names = tuple(object_positions.keys())
    xforms = {
        name: {
            "pos": (float(position["position"][0]), float(position["position"][1]), 0.75),
            "rot": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        }
        for name, position in object_positions.items()
    }
    return SimpleNamespace(
        object_names=object_names,
        xforms=xforms,
        basket_name=None,
    )


class TestStateReindexingProtocol(unittest.TestCase):
    def test_protocol_passthrough_from_annotation_to_cfg_helper(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_blocks_line_4")
        task_spec = load_task_spec("state_reindex_blocks_line_4", 0, scene_profile)
        task_protocol = resolve_task_protocol(task_spec)

        self.assertEqual(task_protocol["module_name"], "post_action_relocation")
        self.assertEqual(task_protocol["track_name"], "state_reindexing")
        self.assertEqual(task_protocol["task_version"], "base")
        self.assertEqual(task_protocol["intervention"]["script_name"], "swap_adjacent_pairs_over_4")
        self.assertEqual(task_protocol["gt_spec"]["order_axis"], "y")
        self.assertEqual(task_protocol["gt_spec"]["target_rank"], "leftmost")
        self.assertEqual(resolve_task_instruction(task_spec, task_protocol), "拿起当前最左侧的方块。")

    def test_stress_protocol_version_passthrough_from_annotation(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_blocks_line_4_tight")
        task_spec = load_task_spec("state_reindex_blocks_line_4_tight", 0, scene_profile)
        task_protocol = resolve_task_protocol(task_spec)

        self.assertEqual(task_protocol["module_name"], "post_action_relocation")
        self.assertEqual(task_protocol["track_name"], "state_reindexing")
        self.assertEqual(task_protocol["task_version"], "stress")
        self.assertEqual(task_protocol["intervention"]["script_name"], "swap_adjacent_pairs_over_4")
        self.assertEqual(task_protocol["gt_spec"]["order_axis"], "y")

    def test_insert_protocol_passthrough_from_annotation(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_insert_blocks_line_4")
        task_spec = load_task_spec("state_reindex_insert_blocks_line_4", 0, scene_profile)
        task_protocol = resolve_task_protocol(task_spec)

        self.assertEqual(task_protocol["module_name"], "post_action_relocation")
        self.assertEqual(task_protocol["track_name"], "state_reindexing")
        self.assertEqual(task_protocol["task_version"], "base")
        self.assertEqual(task_protocol["intervention"]["script_name"], "rightmost_to_between_first_and_second")
        self.assertEqual(task_protocol["gt_spec"]["order_axis"], "y")
        self.assertEqual(task_protocol["gt_spec"]["target_rank"], "index_1")
        self.assertEqual(resolve_task_instruction(task_spec, task_protocol), "拿起当前左起第二个方块。")

    def test_insert_stress_protocol_passthrough_from_annotation(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_insert_blocks_line_4_tight")
        task_spec = load_task_spec("state_reindex_insert_blocks_line_4_tight", 0, scene_profile)
        task_protocol = resolve_task_protocol(task_spec)

        self.assertEqual(task_protocol["module_name"], "post_action_relocation")
        self.assertEqual(task_protocol["track_name"], "state_reindexing")
        self.assertEqual(task_protocol["task_version"], "stress")
        self.assertEqual(task_protocol["intervention"]["script_name"], "rightmost_to_between_first_and_second")
        self.assertEqual(task_protocol["gt_spec"]["order_axis"], "y")
        self.assertEqual(task_protocol["gt_spec"]["target_rank"], "index_1")

    def test_subset_protocol_passthrough_from_annotation(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_red_subset_blocks_line_5")
        task_spec = load_task_spec("state_reindex_red_subset_blocks_line_5", 0, scene_profile)
        task_protocol = resolve_task_protocol(task_spec)

        self.assertEqual(task_protocol["module_name"], "post_action_relocation")
        self.assertEqual(task_protocol["track_name"], "reorder_with_distractors")
        self.assertEqual(task_protocol["task_version"], "base")
        self.assertEqual(task_protocol["intervention"]["script_name"], "rightmost_to_between_first_and_second")
        self.assertEqual(task_protocol["gt_spec"]["order_axis"], "y")
        self.assertEqual(task_protocol["gt_spec"]["target_rank"], "middle")
        self.assertEqual(
            task_protocol["target_subset_spec"]["object_ids"],
            ["red_subset_block_0", "red_subset_block_1", "red_subset_block_2"],
        )
        self.assertEqual(resolve_task_instruction(task_spec, task_protocol), "只看红色方块，拿起其中位于中间的那个。")

    def test_subset_stress_protocol_passthrough_from_annotation(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_red_subset_blocks_line_6_tight")
        task_spec = load_task_spec("state_reindex_red_subset_blocks_line_6_tight", 0, scene_profile)
        task_protocol = resolve_task_protocol(task_spec)

        self.assertEqual(task_protocol["module_name"], "post_action_relocation")
        self.assertEqual(task_protocol["track_name"], "reorder_with_distractors")
        self.assertEqual(task_protocol["task_version"], "stress")
        self.assertEqual(task_protocol["intervention"]["script_name"], "rightmost_to_between_first_and_second")
        self.assertEqual(task_protocol["gt_spec"]["order_axis"], "y")
        self.assertEqual(task_protocol["gt_spec"]["target_rank"], "middle")

    def test_ordinal_insert_static_protocol_passthrough_from_annotation(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("ordinal_insert_static_blocks_base")
        task_spec = load_task_spec("ordinal_insert_static_blocks_base", 0, scene_profile)
        task_protocol = resolve_task_protocol(task_spec)

        self.assertEqual(task_protocol["intervention"]["script_name"], STATIC_SCENE_NOOP_SCRIPT_NAME)
        self.assertEqual(task_protocol["gt_spec"]["resolver"], GT_RESOLVER_RANK_INITIAL_LAYOUT)
        self.assertEqual(task_protocol["gt_spec"]["target_rank"], "index_1")
        self.assertEqual(resolve_task_instruction(task_spec, task_protocol), "拿起左侧第二个方块。")

    def test_grid_rowcol_protocol_passthrough_from_annotation(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("grid_rowcol_grounding_grid_2x3_base")
        task_spec = load_task_spec("grid_rowcol_grounding_grid_2x3_base", 0, scene_profile)
        task_protocol = resolve_task_protocol(task_spec)

        self.assertEqual(task_protocol["intervention"]["script_name"], STATIC_SCENE_NOOP_SCRIPT_NAME)
        self.assertEqual(task_protocol["gt_spec"]["resolver"], GT_RESOLVER_GRID_ROWCOL_SLOT)
        self.assertEqual(int(task_protocol["gt_spec"]["target_row"]), 1)
        self.assertEqual(int(task_protocol["gt_spec"]["target_col"]), 2)

    def test_static_rank_initial_layout_terminal_eval_matches_index_1(self) -> None:
        initial_xy = torch.tensor(
            [
                [
                    [0.43, -0.18],
                    [0.43, -0.06],
                    [0.43, 0.05],
                    [0.43, 0.16],
                    [0.43, 0.27],
                ]
            ],
            dtype=torch.float32,
        )
        post_xy = initial_xy.clone()
        plan = build_state_reindexing_intervention_plan(
            initial_xy,
            min_reorder_gap=0.08,
            script_name=STATIC_SCENE_NOOP_SCRIPT_NAME,
            order_axis="y",
        )
        moved = torch.tensor([-1], dtype=torch.long)
        out = evaluate_state_reindexing_terminal_state(
            initial_xy=initial_xy,
            post_xy=post_xy,
            initial_rank_indices=plan["initial_rank_indices"],
            moved_object_indices=moved,
            displacement_tol=0.02,
            script_name=STATIC_SCENE_NOOP_SCRIPT_NAME,
            target_rank="index_1",
            order_axis="y",
            gt_spec={"resolver": GT_RESOLVER_RANK_INITIAL_LAYOUT},
        )
        self.assertTrue(bool(out["intervention_success"].all().item()))
        self.assertEqual(int(out["gt_object_indices"][0].item()), 1)

    def test_grid_rowcol_slot_terminal_eval_matches_first_row_second_col(self) -> None:
        initial_xy = torch.tensor(
            [
                [
                    [0.37, -0.15],
                    [0.37, -0.03],
                    [0.37, 0.09],
                    [0.5, -0.15],
                    [0.5, -0.03],
                    [0.5, 0.09],
                ]
            ],
            dtype=torch.float32,
        )
        post_xy = initial_xy.clone()
        plan = build_state_reindexing_intervention_plan(
            initial_xy,
            min_reorder_gap=0.08,
            script_name=STATIC_SCENE_NOOP_SCRIPT_NAME,
            order_axis="y",
        )
        gt_spec = {
            "resolver": GT_RESOLVER_GRID_ROWCOL_SLOT,
            "grid_num_rows": 2,
            "grid_num_cols": 3,
            "target_row": 1,
            "target_col": 2,
            "row_axis": "x",
            "col_axis": "y",
            "grid_quant_step": 0.028,
            "min_center_separation": 0.024,
        }
        out = evaluate_state_reindexing_terminal_state(
            initial_xy=initial_xy,
            post_xy=post_xy,
            initial_rank_indices=plan["initial_rank_indices"],
            moved_object_indices=torch.tensor([-1], dtype=torch.long),
            displacement_tol=0.02,
            script_name=STATIC_SCENE_NOOP_SCRIPT_NAME,
            target_rank="index_0",
            order_axis="y",
            gt_spec=gt_spec,
        )
        self.assertTrue(bool(out["intervention_success"].all().item()))
        self.assertEqual(int(out["gt_object_indices"][0].item()), 1)

    def test_leftmost_intervention_updates_post_rank_and_gt(self) -> None:
        initial_xy = torch.tensor(
            [[[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]],
            dtype=torch.float32,
        )
        plan = build_state_reindexing_intervention_plan(
            initial_xy,
            min_reorder_gap=0.08,
            script_name="leftmost_to_right_of_rightmost",
        )
        post_xy = initial_xy.clone()
        post_xy[0, int(plan["moved_object_indices"][0].item())] = plan["target_xy"][0]

        terminal = evaluate_state_reindexing_terminal_state(
            initial_xy=initial_xy,
            post_xy=post_xy,
            initial_rank_indices=plan["initial_rank_indices"],
            moved_object_indices=plan["moved_object_indices"],
            displacement_tol=0.02,
            script_name="leftmost_to_right_of_rightmost",
            target_rank="leftmost",
        )

        self.assertEqual(plan["moved_object_indices"].tolist(), [0])
        self.assertEqual(terminal["post_rank_indices"].tolist(), [[1, 2, 3, 0]])
        self.assertEqual(terminal["gt_object_indices"].tolist(), [1])
        self.assertEqual(terminal["intervention_success"].tolist(), [True])
        self.assertEqual(terminal["valid_episode"].tolist(), [True])
        self.assertEqual(terminal["accidental_displacement"].tolist(), [False])

    def test_rightmost_intervention_and_rightmost_gt(self) -> None:
        initial_xy = torch.tensor(
            [[[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]],
            dtype=torch.float32,
        )
        plan = build_state_reindexing_intervention_plan(
            initial_xy,
            min_reorder_gap=0.08,
            script_name="rightmost_to_left_of_leftmost",
        )
        post_xy = initial_xy.clone()
        post_xy[0, int(plan["moved_object_indices"][0].item())] = plan["target_xy"][0]

        terminal = evaluate_state_reindexing_terminal_state(
            initial_xy=initial_xy,
            post_xy=post_xy,
            initial_rank_indices=plan["initial_rank_indices"],
            moved_object_indices=plan["moved_object_indices"],
            displacement_tol=0.02,
            script_name="rightmost_to_left_of_leftmost",
            target_rank="rightmost",
        )

        self.assertEqual(plan["moved_object_indices"].tolist(), [3])
        self.assertEqual(terminal["post_rank_indices"].tolist(), [[3, 0, 1, 2]])
        self.assertEqual(terminal["gt_object_indices"].tolist(), [2])
        self.assertEqual(terminal["intervention_success"].tolist(), [True])
        self.assertEqual(terminal["valid_episode"].tolist(), [True])

    def test_accidental_displacement_invalidates_episode(self) -> None:
        initial_xy = torch.tensor(
            [[[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]],
            dtype=torch.float32,
        )
        plan = build_state_reindexing_intervention_plan(
            initial_xy,
            min_reorder_gap=0.08,
            script_name="leftmost_to_right_of_rightmost",
        )
        post_xy = initial_xy.clone()
        post_xy[0, int(plan["moved_object_indices"][0].item())] = plan["target_xy"][0]
        post_xy[0, 2, 1] += 0.03

        terminal = evaluate_state_reindexing_terminal_state(
            initial_xy=initial_xy,
            post_xy=post_xy,
            initial_rank_indices=plan["initial_rank_indices"],
            moved_object_indices=plan["moved_object_indices"],
            displacement_tol=0.02,
            script_name="leftmost_to_right_of_rightmost",
            target_rank="leftmost",
        )

        self.assertEqual(terminal["intervention_success"].tolist(), [True])
        self.assertEqual(terminal["accidental_displacement"].tolist(), [True])
        self.assertEqual(terminal["valid_episode"].tolist(), [False])

    def test_gt_resolution_failure_for_unsupported_target_rank(self) -> None:
        initial_xy = torch.tensor(
            [[[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]],
            dtype=torch.float32,
        )
        plan = build_state_reindexing_intervention_plan(
            initial_xy,
            min_reorder_gap=0.08,
            script_name="leftmost_to_right_of_rightmost",
        )
        post_xy = initial_xy.clone()
        post_xy[0, int(plan["moved_object_indices"][0].item())] = plan["target_xy"][0]

        terminal = evaluate_state_reindexing_terminal_state(
            initial_xy=initial_xy,
            post_xy=post_xy,
            initial_rank_indices=plan["initial_rank_indices"],
            moved_object_indices=plan["moved_object_indices"],
            displacement_tol=0.02,
            script_name="leftmost_to_right_of_rightmost",
            target_rank="middle",
        )

        self.assertEqual(terminal["intervention_success"].tolist(), [True])
        self.assertEqual(terminal["gt_resolution_failure"].tolist(), [True])
        self.assertEqual(terminal["valid_episode"].tolist(), [False])

    def test_readout_failure_categories_and_status_labels(self) -> None:
        readout = evaluate_state_reindexing_readout(
            gt_object_indices=torch.tensor([1, 1, 1], dtype=torch.long),
            grasped_object_indices=torch.tensor([-1, 2, 1], dtype=torch.long),
            valid_episode=torch.tensor([True, True, True], dtype=torch.bool),
        )

        self.assertEqual(readout["success"].tolist(), [False, False, True])
        self.assertEqual(readout["grasp_readout_failure"].tolist(), [True, False, False])
        self.assertEqual(readout["valid_episode_task_failed"].tolist(), [False, True, False])

        labels = build_state_reindexing_status_labels(
            intervention_failure=torch.tensor([True, False, False, False, False], dtype=torch.bool),
            gt_resolution_failure=torch.tensor([False, True, False, False, False], dtype=torch.bool),
            accidental_displacement=torch.tensor([False, False, True, False, False], dtype=torch.bool),
            grasp_readout_failure=torch.tensor([False, False, False, True, False], dtype=torch.bool),
            valid_episode_task_failed=torch.tensor([False, False, False, False, True], dtype=torch.bool),
            success=torch.tensor([False, False, False, False, False], dtype=torch.bool),
        )
        self.assertEqual(
            labels,
            [
                "intervention_failed",
                "gt_resolution_failed",
                "invalid_due_to_confound",
                "grasp_readout_failed",
                "valid_episode_but_task_failed",
            ],
        )


if __name__ == "__main__":
    unittest.main()
