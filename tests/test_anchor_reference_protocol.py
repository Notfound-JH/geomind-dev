from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from geomind_lab.mimic.envs.anchor_reference_logic import (  # noqa: E402
    DEFAULT_DISTANCE_TIE_EPSILON,
    NEAREST_ANCHOR_SELECTION_RESOLVER,
    build_anchor_reference_problem_from_object_positions,
    resolve_nearest_anchor_problem,
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


class TestAnchorReferenceProtocol(unittest.TestCase):
    def test_distance_annotations_use_pure_nearest_semantics(self) -> None:
        for layout_key in (
            "grasp_anchor_distance_scene10_base",
            "grasp_anchor_distance_scene10_stress",
        ):
            with self.subTest(layout_key=layout_key):
                scene_profile = _dummy_scene_profile_for_layout(layout_key)
                task_spec = load_task_spec(layout_key, 0, scene_profile)
                task_protocol = resolve_task_protocol(task_spec)
                gt_spec = dict(task_protocol.get("gt_spec", {}))

                self.assertEqual(gt_spec["resolver"], NEAREST_ANCHOR_SELECTION_RESOLVER)
                self.assertEqual(gt_spec["metric"], "planar_euclidean")
                self.assertEqual(gt_spec["distance_axes"], ["x", "y"])
                self.assertEqual(gt_spec["expected_target"], "ball_nearest")
                self.assertNotIn("min_gap_to_second", gt_spec)

                problem = build_anchor_reference_problem_from_object_positions(
                    task_spec,
                    task_spec["object_positions"],
                )
                self.assertEqual(problem["gt_object_id"], "ball_nearest")
                self.assertEqual(problem["resolver"], NEAREST_ANCHOR_SELECTION_RESOLVER)
                self.assertEqual(problem["metric"], "planar_euclidean")
                self.assertEqual(problem["distance_axes"], ["x", "y"])

    def test_tie_epsilon_only_handles_numerical_ambiguity(self) -> None:
        protocol = {
            "module_name": "anchor_reference",
            "variant": "grasp_anchor_distance",
            "anchor_object": {"object_id": "anchor"},
            "candidate_object_ids": ["ball_nearest", "ball_second"],
            "gt_spec": {
                "resolver": NEAREST_ANCHOR_SELECTION_RESOLVER,
                "reference_frame": "world",
                "metric": "planar_euclidean",
                "distance_axes": ["x", "y"],
                "expected_target": "ball_nearest",
                "tie_epsilon": DEFAULT_DISTANCE_TIE_EPSILON,
            },
        }

        unique_positions = {
            "anchor": {"position": [0.0, 0.0]},
            "ball_nearest": {"position": [0.1, 0.0]},
            "ball_second": {"position": [0.1002, 0.0]},
        }
        problem = resolve_nearest_anchor_problem(protocol, unique_positions)
        self.assertEqual(problem["gt_object_id"], "ball_nearest")

        ambiguous_positions = {
            "anchor": {"position": [0.0, 0.0]},
            "ball_nearest": {"position": [0.1, 0.0]},
            "ball_second": {"position": [0.10005, 0.0]},
        }
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_nearest_anchor_problem(protocol, ambiguous_positions)


if __name__ == "__main__":
    unittest.main()
