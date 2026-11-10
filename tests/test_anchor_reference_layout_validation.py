from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ANNOTATIONS_PATH = REPO_ROOT / "assets" / "annotations" / "annotated_tasks.json"


def _load_payload() -> dict[str, Any]:
    return json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))


def _annotation(layout_key: str) -> dict[str, Any]:
    payload = _load_payload()
    return dict(payload[layout_key]["tasks"][0]["annotation"])


def _runtime_xy_map(annotation: dict[str, Any]) -> dict[str, tuple[float, float]]:
    runtime_objects = annotation["scene_adaptation"]["runtime_objects"]
    xy_by_name: dict[str, tuple[float, float]] = {}
    for runtime_object in runtime_objects:
        pose = runtime_object.get("pose", {})
        xy = pose.get("xy", [])
        xy_by_name[str(runtime_object["name"])] = (float(xy[0]), float(xy[1]))
    return xy_by_name


def _distance_summary(layout_key: str) -> dict[str, float | str]:
    annotation = _annotation(layout_key)
    protocol = annotation["protocol"]
    xy_by_name = _runtime_xy_map(annotation)
    anchor_xy = xy_by_name[str(protocol["anchor_object"]["object_id"])]

    distances = []
    for object_id in protocol["candidate_object_ids"]:
        object_xy = xy_by_name[str(object_id)]
        distances.append(
            (
                math.hypot(object_xy[0] - anchor_xy[0], object_xy[1] - anchor_xy[1]),
                str(object_id),
            )
        )
    distances.sort(key=lambda item: (item[0], item[1]))
    return {
        "nearest_id": distances[0][1],
        "nearest_distance": distances[0][0],
        "second_distance": distances[1][0],
        "gap_to_second": distances[1][0] - distances[0][0],
        "constraint": float(annotation["scene_adaptation"]["layout_constraints"]["min_gap_to_second"]),
    }


class TestAnchorReferenceLayoutValidation(unittest.TestCase):
    def test_layout_gap_constraints_live_outside_gt_spec(self) -> None:
        for layout_key in (
            "grasp_anchor_distance_scene10_base",
            "grasp_anchor_distance_scene10_stress",
        ):
            with self.subTest(layout_key=layout_key):
                annotation = _annotation(layout_key)
                self.assertIn("layout_constraints", annotation["scene_adaptation"])
                self.assertIn("min_gap_to_second", annotation["scene_adaptation"]["layout_constraints"])
                self.assertNotIn("min_gap_to_second", annotation["protocol"]["gt_spec"])

    def test_base_and_stress_share_semantics_but_differ_in_layout_gap(self) -> None:
        base_annotation = _annotation("grasp_anchor_distance_scene10_base")
        stress_annotation = _annotation("grasp_anchor_distance_scene10_stress")
        for field_name in ("resolver", "reference_frame", "metric", "distance_axes", "expected_target"):
            self.assertEqual(
                base_annotation["protocol"]["gt_spec"][field_name],
                stress_annotation["protocol"]["gt_spec"][field_name],
            )

        base_summary = _distance_summary("grasp_anchor_distance_scene10_base")
        stress_summary = _distance_summary("grasp_anchor_distance_scene10_stress")

        self.assertEqual(base_summary["nearest_id"], "ball_nearest")
        self.assertEqual(stress_summary["nearest_id"], "ball_nearest")
        self.assertGreater(float(base_summary["gap_to_second"]), float(stress_summary["gap_to_second"]))
        self.assertGreaterEqual(float(base_summary["gap_to_second"]), float(base_summary["constraint"]))
        self.assertGreaterEqual(float(stress_summary["gap_to_second"]), float(stress_summary["constraint"]))


if __name__ == "__main__":
    unittest.main()
