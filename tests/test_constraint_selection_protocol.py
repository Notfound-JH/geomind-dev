from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from geomind_lab.mimic.envs.constraint_selection_logic import (  # noqa: E402
    EXCLUSION_NEAREST_CONSTRAINT_SELECTION_RESOLVER,
    VISUAL_SPATIAL_CONSTRAINT_SELECTION_RESOLVER,
    build_constraint_selection_problem_from_object_positions,
    resolve_exclusion_nearest_constraint_problem,
    resolve_visual_spatial_constraint_problem,
)
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import (  # noqa: E402
    load_annotated_tasks,
    load_task_spec,
    resolve_task_protocol,
)


def _dummy_scene_profile_for_layout(layout_key: str) -> SimpleNamespace:
    payload = load_annotated_tasks()
    layout = payload[layout_key]
    object_positions = dict(layout["scene_structure"]["object_positions"])
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


class TestConstraintSelectionProtocol(unittest.TestCase):
    def test_select_visual_spatial_annotations_resolve_unique_gt(self) -> None:
        for layout_key, expected_target in (
            ("select_visual_spatial_scene65_base", "wood_cylinder_target"),
            ("select_visual_spatial_scene17_stress", "wood_cylinder_target"),
            ("select_visual_spatial_control_scene65_control", "wood_cylinder_target"),
            ("select_visual_spatial_control_scene17_stress", "wood_cylinder_target"),
        ):
            with self.subTest(layout_key=layout_key):
                scene_profile = _dummy_scene_profile_for_layout(layout_key)
                task_spec = load_task_spec(layout_key, 0, scene_profile)
                task_protocol = resolve_task_protocol(task_spec)
                gt_spec = dict(task_protocol.get("gt_spec", {}))

                self.assertEqual(gt_spec["resolver"], VISUAL_SPATIAL_CONSTRAINT_SELECTION_RESOLVER)
                self.assertEqual(gt_spec["axis"], "y")
                self.assertEqual(gt_spec["direction"], "negative")
                self.assertEqual(gt_spec["target_filters"], {"material": "wood_grain", "shape": "cylinder"})
                self.assertEqual(gt_spec["expected_target"], expected_target)

                problem = build_constraint_selection_problem_from_object_positions(
                    task_spec,
                    task_spec["object_positions"],
                )
                self.assertEqual(problem["gt_object_id"], expected_target)
                self.assertEqual(problem["resolver"], VISUAL_SPATIAL_CONSTRAINT_SELECTION_RESOLVER)
                self.assertEqual(problem["target_filters"], {"material": "wood_grain", "shape": "cylinder"})
                self.assertEqual(problem["valid_candidate_ids"], [expected_target])

    def test_select_exclusion_unique_annotations_resolve_unique_gt(self) -> None:
        for layout_key, expected_target in (
            ("select_exclusion_unique_scene65_base", "red_block_target"),
            ("select_exclusion_unique_scene17_stress", "red_block_target"),
        ):
            with self.subTest(layout_key=layout_key):
                scene_profile = _dummy_scene_profile_for_layout(layout_key)
                task_spec = load_task_spec(layout_key, 0, scene_profile)
                task_protocol = resolve_task_protocol(task_spec)
                gt_spec = dict(task_protocol.get("gt_spec", {}))

                self.assertEqual(gt_spec["resolver"], EXCLUSION_NEAREST_CONSTRAINT_SELECTION_RESOLVER)
                self.assertEqual(gt_spec["exclusion_axis"], "x")
                self.assertEqual(gt_spec["exclusion_rank"], "minimum")
                self.assertEqual(gt_spec["metric"], "planar_euclidean")
                self.assertEqual(gt_spec["distance_axes"], ["x", "y"])
                self.assertEqual(gt_spec["target_filters"], {"color": "red", "shape": "cuboid"})
                self.assertEqual(gt_spec["expected_target"], expected_target)

                problem = build_constraint_selection_problem_from_object_positions(
                    task_spec,
                    task_spec["object_positions"],
                )
                self.assertEqual(problem["gt_object_id"], expected_target)
                self.assertEqual(problem["resolver"], EXCLUSION_NEAREST_CONSTRAINT_SELECTION_RESOLVER)
                self.assertEqual(problem["target_filters"], {"color": "red", "shape": "cuboid"})
                self.assertEqual(problem["excluded_candidate_ids"], ["red_block_leftmost"])
                self.assertEqual(problem["valid_candidate_ids"], [expected_target])
                self.assertIn(expected_target, problem["spatial_match_ids"])

    def test_select_exclusion_unique_red_cuboids_keep_clearance_from_blue_plate(self) -> None:
        min_clearance = 0.16
        for layout_key in (
            "select_exclusion_unique_scene65_base",
            "select_exclusion_unique_scene17_stress",
        ):
            with self.subTest(layout_key=layout_key):
                payload = load_annotated_tasks()
                layout = payload[layout_key]
                task = layout["tasks"][0]
                protocol = dict((task.get("annotation") or {}).get("protocol") or {})
                semantics = dict(protocol.get("object_semantics") or {})
                object_positions = dict((layout.get("scene_structure") or {}).get("object_positions") or {})
                anchor_xy = object_positions["blue_plate"]["position"]
                for object_id in protocol.get("candidate_object_ids", []):
                    object_semantics = dict(semantics.get(object_id) or {})
                    if object_semantics.get("shape") != "cuboid":
                        continue
                    x, y = object_positions[object_id]["position"]
                    clearance = math.hypot(float(x) - float(anchor_xy[0]), float(y) - float(anchor_xy[1]))
                    self.assertGreaterEqual(
                        clearance,
                        min_clearance,
                        msg=f"{layout_key}:{object_id} is too close to blue_plate ({clearance:.4f} < {min_clearance:.4f})",
                    )

    def test_visual_spatial_constraint_selection_rejects_ambiguous_runtime_layout(self) -> None:
        protocol = {
            "module_name": "constraint_selection",
            "variant": "select_visual_spatial",
            "anchor_object": {"object_id": "anchor"},
            "candidate_object_ids": ["target_a", "target_b", "distractor"],
            "object_semantics": {
                "target_a": {"material": "wood_grain", "shape": "cylinder"},
                "target_b": {"material": "wood_grain", "shape": "cylinder"},
                "distractor": {"material": "painted", "shape": "cuboid"},
            },
            "gt_spec": {
                "resolver": VISUAL_SPATIAL_CONSTRAINT_SELECTION_RESOLVER,
                "reference_frame": "world",
                "axis": "y",
                "direction": "negative",
                "min_primary_offset": 0.06,
                "max_abs_secondary_offset": 0.03,
                "target_filters": {"material": "wood_grain", "shape": "cylinder"},
                "expected_target": "target_a",
            },
        }
        object_positions = {
            "anchor": {"position": [0.0, 0.0]},
            "target_a": {"position": [0.0, -0.09]},
            "target_b": {"position": [0.01, -0.10]},
            "distractor": {"position": [0.0, 0.09]},
        }

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_visual_spatial_constraint_problem(protocol, object_positions)

    def test_exclusion_nearest_constraint_selection_rejects_ambiguous_post_exclusion_distance(self) -> None:
        protocol = {
            "module_name": "constraint_selection",
            "variant": "select_exclusion_unique",
            "anchor_object": {"object_id": "blue_plate"},
            "candidate_object_ids": ["red_leftmost", "red_target_a", "red_target_b", "orange_block"],
            "object_semantics": {
                "red_leftmost": {"color": "red", "shape": "cuboid"},
                "red_target_a": {"color": "red", "shape": "cuboid"},
                "red_target_b": {"color": "red", "shape": "cuboid"},
                "orange_block": {"color": "orange", "shape": "cuboid"},
            },
            "gt_spec": {
                "resolver": EXCLUSION_NEAREST_CONSTRAINT_SELECTION_RESOLVER,
                "reference_frame": "world",
                "exclusion_axis": "x",
                "exclusion_rank": "minimum",
                "rank_tie_epsilon": 1.0e-4,
                "metric": "planar_euclidean",
                "distance_axes": ["x", "y"],
                "distance_tie_epsilon": 1.0e-4,
                "target_filters": {"color": "red", "shape": "cuboid"},
                "expected_target": "red_target_a",
            },
        }
        object_positions = {
            "blue_plate": {"position": [0.0, 0.0]},
            "red_leftmost": {"position": [-0.2, 0.0]},
            "red_target_a": {"position": [0.12, 0.0]},
            "red_target_b": {"position": [0.0, 0.12]},
            "orange_block": {"position": [0.08, 0.0]},
        }

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_exclusion_nearest_constraint_problem(protocol, object_positions)

    def test_exclusion_nearest_constraint_selection_rejects_non_unique_exclusion_rank(self) -> None:
        protocol = {
            "module_name": "constraint_selection",
            "variant": "select_exclusion_unique",
            "anchor_object": {"object_id": "blue_plate"},
            "candidate_object_ids": ["red_leftmost_a", "red_leftmost_b", "red_target"],
            "object_semantics": {
                "red_leftmost_a": {"color": "red", "shape": "cuboid"},
                "red_leftmost_b": {"color": "red", "shape": "cuboid"},
                "red_target": {"color": "red", "shape": "cuboid"},
            },
            "gt_spec": {
                "resolver": EXCLUSION_NEAREST_CONSTRAINT_SELECTION_RESOLVER,
                "reference_frame": "world",
                "exclusion_axis": "x",
                "exclusion_rank": "minimum",
                "rank_tie_epsilon": 1.0e-4,
                "metric": "planar_euclidean",
                "distance_axes": ["x", "y"],
                "distance_tie_epsilon": 1.0e-4,
                "target_filters": {"color": "red", "shape": "cuboid"},
                "expected_target": "red_target",
            },
        }
        object_positions = {
            "blue_plate": {"position": [0.0, 0.0]},
            "red_leftmost_a": {"position": [-0.2, 0.0]},
            "red_leftmost_b": {"position": [-0.2, 0.08]},
            "red_target": {"position": [0.1, 0.0]},
        }

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_exclusion_nearest_constraint_problem(protocol, object_positions)


if __name__ == "__main__":
    unittest.main()
