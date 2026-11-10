from __future__ import annotations

import math
import os
import unittest
from unittest.mock import patch

from geomind_lab.tasks.manager_based.manipulation.geomind.layout_randomizer import (
    _resolve_strategy,
    randomize_runtime_object_layout,
)


def _object(name: str, xy: tuple[float, float], *, role: str = "target", size: tuple[float, float, float] = (0.05, 0.05, 0.05)) -> dict:
    return {"name": name, "kind": "primitive_cuboid", "role": role, "size": list(size), "pose": {"xy": list(xy)}}


class TestLayoutRandomizer(unittest.TestCase):
    def setUp(self) -> None:
        self.environ = patch.dict(
            os.environ,
            {
                "GEOMIND_LAYOUT_JITTER_ENABLE": "1",
                "GEOMIND_LAYOUT_JITTER_SEED": "123",
                "GEOMIND_LAYOUT_SHIFT_ENABLE": "0",
                "GEOMIND_LAYOUT_BOUNDS": "0.15,0.68,-0.32,0.34",
                "GEOMIND_LAYOUT_MIN_CLEARANCE": "0.025",
            },
            clear=False,
        )
        self.environ.start()

    def tearDown(self) -> None:
        self.environ.stop()

    def test_only_simple_spatial_variants_use_workspace_sampler(self) -> None:
        for variant, resolver in (
            ("place_directional", "directional_region_placement"),
            ("place_inside", "containment_placement"),
            ("place_on_top", "support_top_placement"),
        ):
            self.assertEqual(
                _resolve_strategy({"resolver": resolver}, {"family": "spatial_placement", "variant": variant}),
                "spatial_free",
            )
        self.assertEqual(
            _resolve_strategy({"resolver": "directional_region_placement"}, {"family": "other", "variant": "place_directional"}),
            "free",
        )

    def test_container_parts_stay_rigid_and_groups_do_not_overlap(self) -> None:
        objects = [
            _object("blue_box_floor", (0.43, 0.04), role="container_floor", size=(0.17, 0.15, 0.012)),
            _object("blue_box_wall_left", (0.351, 0.04), role="container_wall", size=(0.012, 0.15, 0.07)),
            _object("blue_box_wall_right", (0.509, 0.04), role="container_wall", size=(0.012, 0.15, 0.07)),
            _object("movable_cube", (0.24, -0.11)),
        ]
        before_offset = [
            objects[index]["pose"]["xy"][axis] - objects[0]["pose"]["xy"][axis]
            for index in (1, 2)
            for axis in (0, 1)
        ]
        protocol = {
            "family": "spatial_placement",
            "variant": "place_inside",
            "gt_spec": {
                "resolver": "containment_placement",
                "container_floor_object_id": "blue_box_floor",
                "container_wall_object_ids": ["blue_box_wall_left", "blue_box_wall_right"],
            },
        }
        record = randomize_runtime_object_layout(objects, protocol=protocol)
        self.assertEqual(record["strategy"], "spatial_free")
        after_offset = [
            objects[index]["pose"]["xy"][axis] - objects[0]["pose"]["xy"][axis]
            for index in (1, 2)
            for axis in (0, 1)
        ]
        self.assertEqual(before_offset, after_offset)
        floor = objects[0]["pose"]["xy"]
        cube = objects[3]["pose"]["xy"]
        self.assertGreaterEqual(math.dist(floor, cube), 0.11)
        for obj in (objects[0], objects[3]):
            x, y = obj["pose"]["xy"]
            self.assertGreaterEqual(x, 0.15)
            self.assertLessEqual(x, 0.68)
            self.assertGreaterEqual(y, -0.32)
            self.assertLessEqual(y, 0.34)

    def test_directional_target_zone_stays_inside_workspace(self) -> None:
        with patch.dict(os.environ, {"GEOMIND_LAYOUT_SHIFT_ENABLE": "1", "GEOMIND_LAYOUT_SHIFT_SEED": "456"}):
            objects = [
                _object("blue_plate", (0.39, 0.02), role="anchor", size=(0.252, 0.252, 0.02)),
                _object("movable_cube", (0.24, -0.11)),
            ]
            protocol = {
                "family": "spatial_placement",
                "variant": "place_directional",
                "gt_spec": {
                    "resolver": "move_to_object_proximity",
                    "anchor_object_id": "blue_plate",
                    "target_zone": {"center_offset_from_anchor": [0.26, 0.0], "half_extents": [0.126, 0.126]},
                },
            }
            randomize_runtime_object_layout(objects, protocol=protocol)
        anchor_x, anchor_y = objects[0]["pose"]["xy"]
        target_x, target_y = anchor_x + 0.26, anchor_y
        self.assertGreaterEqual(target_x - 0.126, 0.15)
        self.assertLessEqual(target_x + 0.126, 0.68)
        self.assertGreaterEqual(target_y - 0.126, -0.32)
        self.assertLessEqual(target_y + 0.126, 0.34)


if __name__ == "__main__":
    unittest.main()