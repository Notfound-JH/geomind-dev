from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from geomind_lab.mimic.envs.geometric_inference_logic import (
    EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME,
    EXTENSION_RULE_TYPE,
    GEOMETRIC_INFERENCE_MODULE_NAME,
    SHAPE_COMPLETION_TRACK_NAME,
    build_geometric_inference_debug_result,
    build_geometric_inference_problem,
    build_geometric_inference_suite_summary,
    build_geometric_inference_task_summary,
    evaluate_extension_readout,
    evaluate_point_symmetry_readout,
    evaluate_square_completion_readout,
    normalize_rule_type,
    solve_extension_gt,
    solve_point_symmetry_gt,
    solve_square_completion_gt,
)
from geomind_lab.mimic.geometric_inference_agents import (
    GeometricInferenceAgentInput,
    OraclePositionAgent,
    RandomPositionAgent,
)
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import (
    load_annotated_tasks,
    load_task_spec,
    resolve_task_protocol,
)


def _dummy_scene_profile_for_layout(
    layout_key: str,
    *,
    object_names: tuple[str, ...] | None = None,
) -> SimpleNamespace:
    payload = load_annotated_tasks()
    layout = payload[layout_key]
    object_items = list(layout["scene_structure"]["object_positions"].items())
    resolved_object_names = object_names or tuple(name for name, _ in object_items)

    xforms = {}
    for target_name, (_, position_payload) in zip(resolved_object_names, object_items):
        position = position_payload["position"]
        xforms[target_name] = {
            "pos": (float(position[0]), float(position[1]), 0.75),
            "rot": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        }

    return SimpleNamespace(
        object_names=tuple(resolved_object_names),
        xforms=xforms,
        basket_name=None,
    )


class TestGeometricInferenceProtocol(unittest.TestCase):
    def test_primary_extension_annotations_cover_multiple_samples(self) -> None:
        payload = load_annotated_tasks()
        matched_tasks: list[tuple[str, int]] = []
        for layout_key, layout in payload.items():
            if not isinstance(layout, dict):
                continue
            tasks = layout.get("tasks", [])
            if not isinstance(tasks, list):
                continue
            for task_index, task in enumerate(tasks):
                if not isinstance(task, dict):
                    continue
                annotation = task.get("annotation", {})
                protocol = annotation.get("protocol", {}) if isinstance(annotation, dict) else {}
                if not isinstance(protocol, dict):
                    continue
                gt_spec = protocol.get("gt_spec", {})
                if (
                    str(protocol.get("module_name", "")) == GEOMETRIC_INFERENCE_MODULE_NAME
                    and str(protocol.get("track_name", "")) == EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME
                    and str(protocol.get("query_family", "")) == "explicit_rule_query"
                    and isinstance(gt_spec, dict)
                    and normalize_rule_type(gt_spec.get("rule_type", "")) == EXTENSION_RULE_TYPE
                ):
                    matched_tasks.append((str(layout_key), int(task_index)))

        expected_layouts = {
            "geometric_inference_extension_line_3",
            "geometric_inference_extension_vertical_line_3",
            "geometric_inference_extension_grid_2_2_top_row",
            "geometric_inference_extension_grid_2_3_left_column",
        }
        self.assertGreaterEqual(len(matched_tasks), 4)
        self.assertTrue(expected_layouts.issubset({layout_key for layout_key, _ in matched_tasks}))

    def test_protocol_passthrough_and_problem_building(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("geometric_inference_extension_line_3")
        task_spec = load_task_spec("geometric_inference_extension_line_3", 0, scene_profile)
        task_protocol = resolve_task_protocol(task_spec)

        self.assertEqual(task_protocol["module_name"], GEOMETRIC_INFERENCE_MODULE_NAME)
        self.assertEqual(task_protocol["track_name"], EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME)
        self.assertEqual(task_protocol["query_family"], "explicit_rule_query")
        self.assertEqual(task_protocol["gt_spec"]["rule_type"], EXTENSION_RULE_TYPE)
        self.assertEqual(task_protocol["anchors"][0]["object_id"], "low_can_1")
        self.assertEqual(task_protocol["anchors"][1]["object_id"], "low_can_3")

        problem = build_geometric_inference_problem(task_spec)
        expected_gt = solve_extension_gt(
            problem["anchor_positions_xy"]["A"],
            problem["anchor_positions_xy"]["B"],
        )

        self.assertEqual(problem["module_name"], GEOMETRIC_INFERENCE_MODULE_NAME)
        self.assertEqual(problem["track_name"], EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME)
        self.assertEqual(problem["anchor_object_ids"], {"A": "low_can_1", "B": "low_can_3"})
        self.assertEqual(problem["target_object_id"], "low_can")
        self.assertEqual(problem["gt_pos_xy"], expected_gt)

    def test_protocol_object_refs_are_remapped_with_scene_names(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout(
            "geometric_inference_extension_line_3",
            object_names=("demo_obj", "demo_obj_1", "demo_obj_2", "demo_obj_3"),
        )
        task_spec = load_task_spec("geometric_inference_extension_line_3", 0, scene_profile)
        task_protocol = resolve_task_protocol(task_spec)

        self.assertEqual(set(task_spec["object_positions"].keys()), {"demo_obj", "demo_obj_1", "demo_obj_2", "demo_obj_3"})
        self.assertEqual(task_protocol["anchors"][0]["object_id"], "demo_obj_1")
        self.assertEqual(task_protocol["anchors"][1]["object_id"], "demo_obj_3")
        self.assertEqual(task_protocol["target_object"]["object_id"], "demo_obj")

        problem = build_geometric_inference_problem(task_spec)
        self.assertEqual(problem["anchor_object_ids"], {"A": "demo_obj_1", "B": "demo_obj_3"})
        self.assertEqual(problem["target_object_id"], "demo_obj")

    def test_new_extension_layouts_build_runtime_problem(self) -> None:
        for layout_key in (
            "geometric_inference_extension_vertical_line_3",
            "geometric_inference_extension_grid_2_2_top_row",
            "geometric_inference_extension_grid_2_3_left_column",
        ):
            with self.subTest(layout_key=layout_key):
                scene_profile = _dummy_scene_profile_for_layout(layout_key)
                task_spec = load_task_spec(layout_key, 0, scene_profile)
                task_protocol = resolve_task_protocol(task_spec)
                problem = build_geometric_inference_problem(task_spec)

                self.assertEqual(task_protocol["module_name"], GEOMETRIC_INFERENCE_MODULE_NAME)
                self.assertEqual(task_protocol["track_name"], EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME)
                self.assertEqual(task_protocol["query_family"], "explicit_rule_query")
                self.assertEqual(normalize_rule_type(task_protocol["gt_spec"]["rule_type"]), EXTENSION_RULE_TYPE)
                self.assertIn("A", problem["anchor_positions_xy"])
                self.assertIn("B", problem["anchor_positions_xy"])
                self.assertIsInstance(problem["target_object_id"], str)
                self.assertEqual(
                    tuple(problem["gt_pos_xy"]),
                    solve_extension_gt(
                        problem["anchor_positions_xy"]["A"],
                        problem["anchor_positions_xy"]["B"],
                    ),
                )

    def test_point_symmetry_layouts_build_runtime_problem(self) -> None:
        for layout_key in (
            "geometric_inference_point_symmetry_line_2",
            "geometric_inference_point_symmetry_tight_obstacle",
        ):
            with self.subTest(layout_key=layout_key):
                scene_profile = _dummy_scene_profile_for_layout(layout_key)
                task_spec = load_task_spec(layout_key, 0, scene_profile)
                task_protocol = resolve_task_protocol(task_spec)
                problem = build_geometric_inference_problem(task_spec)

                self.assertEqual(task_protocol["module_name"], GEOMETRIC_INFERENCE_MODULE_NAME)
                self.assertEqual(task_protocol["track_name"], EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME)
                self.assertEqual(task_protocol["query_family"], "explicit_rule_query")
                self.assertEqual(task_protocol["gt_spec"]["rule_type"], "point_symmetry")
                self.assertEqual(task_protocol["anchors"][0]["object_id"], "red_center_block")
                self.assertEqual(task_protocol["anchors"][1]["object_id"], "blue_reference_block")
                self.assertEqual(problem["anchor_object_ids"], {"A": "red_center_block", "B": "blue_reference_block"})
                self.assertEqual(problem["target_object_id"], "held_target_block")
                self.assertEqual(
                    tuple(problem["gt_pos_xy"]),
                    solve_point_symmetry_gt(
                        problem["anchor_positions_xy"]["A"],
                        problem["anchor_positions_xy"]["B"],
                    ),
                )

    def test_square_completion_layouts_build_runtime_problem(self) -> None:
        for layout_key, task_version in (
            ("geometric_inference_square_completion_corner_3", "base"),
            ("geometric_inference_square_completion_corner_3_skill", "skill"),
            ("geometric_inference_square_completion_tight_obstacle", "stress"),
        ):
            with self.subTest(layout_key=layout_key):
                scene_profile = _dummy_scene_profile_for_layout(layout_key)
                task_spec = load_task_spec(layout_key, 0, scene_profile)
                task_protocol = resolve_task_protocol(task_spec)
                problem = build_geometric_inference_problem(task_spec)

                self.assertEqual(task_protocol["module_name"], GEOMETRIC_INFERENCE_MODULE_NAME)
                self.assertEqual(task_protocol["track_name"], SHAPE_COMPLETION_TRACK_NAME)
                self.assertEqual(task_protocol["query_family"], "shape_completion_query")
                self.assertEqual(task_protocol["task_version"], task_version)
                self.assertEqual(task_protocol["gt_spec"]["rule_type"], "square_completion")
                self.assertEqual(problem["target_object_id"], "held_target_block")
                self.assertEqual(
                    tuple(problem["gt_pos_xy"]),
                    tuple(
                        solve_square_completion_gt(
                            problem["anchor_positions_xy"]["A"],
                            problem["anchor_positions_xy"]["B"],
                            problem["anchor_positions_xy"]["C"],
                        )["gt_pos_xy"]
                    ),
                )
                if task_version == "base":
                    self.assertNotIn("gt_yaw_rad", problem)
                else:
                    self.assertIn("gt_yaw_rad", problem)

    def test_extension_readout_and_baseline_agents(self) -> None:
        readout = evaluate_extension_readout(
            anchor_a_xy=(0.0, 0.0),
            anchor_b_xy=(0.1, 0.0),
            gt_pos_xy=(0.2, 0.0),
            pred_pos_xy=(0.19, 0.01),
        )
        self.assertAlmostEqual(readout["l2_error"], math.hypot(0.01, 0.01), places=6)
        self.assertAlmostEqual(readout["orthogonal_distance"], 0.01, places=6)
        self.assertTrue(readout["direction_correct"])
        self.assertTrue(readout["rule_satisfied"])
        self.assertTrue(readout["success"])

        scene_profile = _dummy_scene_profile_for_layout("geometric_inference_extension_line_3")
        task_spec = load_task_spec("geometric_inference_extension_line_3", 0, scene_profile)
        problem = build_geometric_inference_problem(task_spec)

        agent_input = GeometricInferenceAgentInput(
            query_text=str(problem["query_text"]),
            anchor_positions_xy=dict(problem["anchor_positions_xy"]),
            protocol_snapshot=problem,
            gt_pos_xy=tuple(problem["gt_pos_xy"]),
            workspace_bounds_xy=tuple(problem["workspace_bounds_xy"]),
        )

        oracle_output = OraclePositionAgent()(agent_input)
        oracle_result = build_geometric_inference_debug_result(
            problem,
            target_pos_xy=oracle_output.target_pos_xy,
            pred_pos_xy=oracle_output.target_pos_xy,
            placement_execution_success=True,
            settle_applied=True,
        )
        self.assertEqual(tuple(oracle_output.target_pos_xy), tuple(problem["gt_pos_xy"]))
        self.assertTrue(oracle_result["success"])
        self.assertEqual(tuple(oracle_result["target_pos_xy"]), tuple(problem["gt_pos_xy"]))
        self.assertTrue(oracle_result["valid_episode"])

        random_output = RandomPositionAgent(seed=7)(agent_input)
        (min_x, min_y), (max_x, max_y) = problem["workspace_bounds_xy"]
        self.assertGreaterEqual(random_output.target_pos_xy[0], min_x)
        self.assertLessEqual(random_output.target_pos_xy[0], max_x)
        self.assertGreaterEqual(random_output.target_pos_xy[1], min_y)
        self.assertLessEqual(random_output.target_pos_xy[1], max_y)

    def test_point_symmetry_readout(self) -> None:
        readout = evaluate_point_symmetry_readout(
            center_xy=(0.1, 0.0),
            reference_xy=(0.2, 0.1),
            gt_pos_xy=(0.0, -0.1),
            pred_pos_xy=(0.01, -0.1),
        )
        self.assertAlmostEqual(readout["l2_error"], 0.01, places=6)
        self.assertTrue(readout["opposite_side"])
        self.assertTrue(readout["rule_satisfied"])
        self.assertTrue(readout["success"])

    def test_square_completion_readout_with_yaw(self) -> None:
        solved = solve_square_completion_gt((0.34, -0.02), (0.34, 0.12), (0.48, -0.02))
        readout = evaluate_square_completion_readout(
            shared_corner_xy=(0.34, -0.02),
            adjacent_corner_b_xy=(0.34, 0.12),
            adjacent_corner_c_xy=(0.48, -0.02),
            gt_pos_xy=solved["gt_pos_xy"],
            pred_pos_xy=(0.48, 0.12),
            gt_yaw_rad=float(solved["gt_yaw_rad"]),
            pred_yaw_rad=float(solved["gt_yaw_rad"]) + 0.05,
            tau_yaw=0.2,
        )
        self.assertAlmostEqual(readout["l2_error"], 0.0, places=6)
        self.assertLess(readout["yaw_error"], 0.2)
        self.assertTrue(readout["yaw_satisfied"])
        self.assertTrue(readout["success"])

    def test_debug_result_includes_execution_fields(self) -> None:
        payload = build_geometric_inference_debug_result(
            {
                "module_name": GEOMETRIC_INFERENCE_MODULE_NAME,
                "track_name": EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME,
                "query_text": "把物体放在 A 和 B 连线的延长线上，间距相同。",
                "rule_type": EXTENSION_RULE_TYPE,
                "anchor_positions_xy": {"A": (0.0, 0.0), "B": (0.1, 0.0)},
                "gt_pos_xy": (0.2, 0.0),
            },
            target_pos_xy=(0.2, 0.0),
            pred_pos_xy=(0.21, 0.0),
            valid_episode=True,
            placement_execution_success=True,
            settle_applied=True,
        )

        self.assertEqual(payload["module_name"], GEOMETRIC_INFERENCE_MODULE_NAME)
        self.assertEqual(payload["track_name"], EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME)
        self.assertIn("query_text", payload)
        self.assertIn("anchor_positions_xy", payload)
        self.assertIn("gt_pos_xy", payload)
        self.assertIn("target_pos_xy", payload)
        self.assertIn("pred_pos_xy", payload)
        self.assertIn("l2_error", payload)
        self.assertIn("orthogonal_distance", payload)
        self.assertIn("direction_correct", payload)
        self.assertIn("rule_satisfied", payload)
        self.assertIn("valid_episode", payload)
        self.assertIn("placement_execution_success", payload)
        self.assertIn("settle_applied", payload)
        self.assertTrue(payload["success"])

    def test_task_and_suite_summary_metrics(self) -> None:
        episode_results = [
            {
                "track_name": EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME,
                "rule_type": EXTENSION_RULE_TYPE,
                "valid_episode": True,
                "success": True,
                "l2_error": 0.01,
                "orthogonal_distance": 0.005,
                "rule_satisfied": True,
                "grasp_execution_success": True,
                "placement_execution_success": True,
            },
            {
                "track_name": EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME,
                "rule_type": EXTENSION_RULE_TYPE,
                "valid_episode": True,
                "success": False,
                "l2_error": 0.03,
                "orthogonal_distance": 0.015,
                "rule_satisfied": True,
                "grasp_execution_success": True,
                "placement_execution_success": True,
            },
            {
                "track_name": EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME,
                "rule_type": EXTENSION_RULE_TYPE,
                "valid_episode": False,
                "success": False,
                "l2_error": 0.2,
                "orthogonal_distance": 0.2,
                "rule_satisfied": False,
                "grasp_execution_success": False,
                "placement_execution_success": False,
            },
        ]

        task_summary = build_geometric_inference_task_summary(
            agent_name="oracle",
            episode_results=episode_results,
            scene_id=4,
            layout_key="geometric_inference_extension_line_3",
            task_index=0,
            task_label="geometric_inference_extension_primary",
        )
        self.assertEqual(task_summary["num_episodes"], 3)
        self.assertEqual(task_summary["valid_episode_count"], 2)
        self.assertAlmostEqual(task_summary["valid_episode_rate"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(task_summary["success_rate"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(task_summary["l2_error_mean"], 0.02, places=6)
        self.assertAlmostEqual(task_summary["orthogonal_distance_mean"], 0.01, places=6)
        self.assertAlmostEqual(task_summary["rule_satisfied_rate"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(task_summary["grasp_execution_success_rate"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(task_summary["placement_execution_success_rate"], 2.0 / 3.0, places=6)

        suite_summary = build_geometric_inference_suite_summary(
            suite_name="geometric_inference_primary_oracle",
            agent_name="oracle",
            task_summaries=[task_summary, task_summary],
        )
        self.assertEqual(suite_summary["suite_name"], "geometric_inference_primary_oracle")
        self.assertEqual(suite_summary["agent_name"], "oracle")
        self.assertEqual(suite_summary["num_tasks"], 2)
        self.assertEqual(suite_summary["total_episodes"], 6)
        self.assertAlmostEqual(suite_summary["valid_episode_rate"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(suite_summary["success_rate"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(suite_summary["l2_error_mean"], 0.02, places=6)
        self.assertAlmostEqual(suite_summary["orthogonal_distance_mean"], 0.01, places=6)
        self.assertAlmostEqual(suite_summary["rule_satisfied_rate"], 2.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
