"""Scene assembly's parser and geometric contracts run without USD or a GPU."""
from copy import deepcopy
import math
import unittest

from geomind.scene_assembly.layouts import LayoutError, generate_positions
from geomind.scene_assembly.request import ASSET_IDS, RequestError, normalize_asset, parse_request, validate_request


class TestAssemblyRequest(unittest.TestCase):
    def request(self):
        return {"schema_version": 1, "scene_id": 60, "object": {"asset": "blue_can", "count": 4}, "layout": {"type": "grid", "rows": 2, "cols": 2}, "seed": 0}

    def test_original_markdown_multiplication_and_standard_shapes(self):
        for shape in ("2* *2", "2**2", "2 * 2", "2x2", "2×2", r"2* \* 2", r"2\*2"):
            with self.subTest(shape=shape):
                self.assertEqual(parse_request(f"put 4 blue cans in grid of {shape} in scene 60."), self.request())

    def test_chinese_and_english_aliases(self):
        for text in ("在场景60中，将4个蓝色罐摆成2×2网格。", "在场景60中摆放4个蓝色罐，按2行2列网格排列。", "将4个蓝色罐按2x2网格摆放在场景60中", "arrange 4 blue_can in 2x2 grid in scene 60"):
            with self.subTest(text=text):
                self.assertEqual(parse_request(text), self.request())
        self.assertEqual(normalize_asset("red cans"), "red_can")
        self.assertEqual(normalize_asset("low cans"), "low_can")
        self.assertEqual(normalize_asset("蓝色罐"), "blue_can")
        self.assertEqual(len(ASSET_IDS), 35)
        for asset in ASSET_IDS:
            self.assertEqual(normalize_asset(asset), asset)

    def test_layout_forms_and_seed(self):
        examples = (("line", 4, "line"), ("ring", 4, "ring"), ("random", 4, "random"), ("t-shaped", 5, "t_shape"))
        for phrase, count, expected in examples:
            with self.subTest(phrase=phrase):
                result = parse_request(f"place {count} red cans in {phrase} in scene 1 with seed 19")
                self.assertEqual(result["layout"]["type"], expected)
                self.assertEqual(result["seed"], 19)
        self.assertEqual(parse_request("put 4 blue cans in line along x in scene 102")["layout"]["axis"], "x")

    def test_unknown_text_and_count_conflicts_are_not_silently_ignored(self):
        bad = (
            "put 4 blue cans in grid of 3x2 in scene 60",
            "put 6 blue cans in grid in scene 60",
            "put 4 blue cans in grid of 2x2x1 in scene 60",
            r"put 4 blue cans in grid of 2\x2 in scene 60",
            r"put 4 blue\ cans in grid of 2x2 in scene 60",
            r"put 4 blue cans in grid of 2* \* 2 in scene 60 then move the table",
            "put 4 blue cans in grid of 2* *2 in scene 60 then move the table",
            "put 4 blue cans in grid of 2x2 in scene 60 and 2 red cans",
            "put 4 blue cans in unsupported in scene 60",
            "put 4 cups in line in scene 60",
            "put 4 unidentified cans in line in scene 60",
            "put 4 blue cans in t_shape in scene 60",
            "put 4 blue cans in line in scene 103",
            "delete all objects in scene 60",
        )
        for text in bad:
            with self.subTest(text=text), self.assertRaises(RequestError):
                parse_request(text)

    def test_implicit_square_and_optional_defaults(self):
        req = self.request()
        req.pop("schema_version")
        req.pop("seed")
        req["layout"] = {"type": "grid"}
        result = validate_request(req)
        self.assertEqual(result, self.request())
        self.assertEqual(req["layout"], {"type": "grid"})

    def test_randomization_is_explicit_with_bounded_defaults(self):
        original = self.request()
        self.assertNotIn("randomization", validate_request(original))
        original["randomization"] = {}
        expected = {"enabled": True, "jitter_xy_m": 0.02, "shift_max_m": 0.15, "yaw_deg": 7.5}
        self.assertEqual(validate_request(original)["randomization"], expected)
        self.assertEqual(original["randomization"], {})
        original["randomization"] = {"enabled": False, "jitter_xy_m": 0, "shift_max_m": 0.3, "yaw_deg": 180}
        self.assertEqual(validate_request(original)["randomization"], {"enabled": False, "jitter_xy_m": 0.0, "shift_max_m": 0.3, "yaw_deg": 180.0})

    def test_randomization_instruction_suffix_preserves_old_seed_grammar(self):
        instruction = "put 4 blue cans in grid of 2x2 in scene 60"
        expected = self.request()
        expected["randomization"] = {"enabled": True, "jitter_xy_m": 0.02, "shift_max_m": 0.15, "yaw_deg": 7.5}
        self.assertEqual(parse_request(instruction + " with randomization."), expected)
        expected["seed"] = 7
        self.assertEqual(parse_request(instruction + " with randomization and seed 7"), expected)
        legacy = parse_request(instruction + " with seed 7")
        self.assertEqual(legacy["seed"], 7)
        self.assertNotIn("randomization", legacy)
        for suffix in (" with randomization and seed -1", " with randomization and seed true", " with randomization then move table", " with randomization and seed 7 with seed 8", " with randomization and unsupported"):
            with self.subTest(suffix=suffix), self.assertRaises(RequestError):
                parse_request(instruction + suffix)

    def test_randomization_schema_rejects_unknown_and_invalid_values(self):
        bad = [None, True, [], {"surprise": 1}, {"enabled": 1}, {"enabled": "false"}]
        for field, maximum in (("jitter_xy_m", 0.05), ("shift_max_m", 0.3), ("yaw_deg", 180)):
            bad.extend({field: value} for value in (True, "0.01", float("nan"), float("inf"), -0.001, maximum + 0.001))
        for randomization in bad:
            req = self.request()
            req["randomization"] = randomization
            with self.subTest(randomization=randomization), self.assertRaises(RequestError):
                validate_request(req)

    def test_strict_nested_schema_and_types(self):
        changes = (
            ((), "scene_id", True), ((), "schema_version", True), ((), "seed", False),
            ((), "scene_id", 0), ((), "scene_id", 103), ((), "schema_version", 2),
            ((), "seed", -1), ((), "surprise", 1),
            (("object",), "count", True), (("object",), "count", 0),
            (("object",), "count", 65), (("object",), "count", 4.0),
            (("object",), "asset", "blue cans"), (("object",), "unknown", 1),
            (("layout",), "rows", True), (("layout",), "cols", 3),
            (("layout",), "rows", 0), (("layout",), "spacing_m", float("nan")),
            (("layout",), "spacing_m", True), (("layout",), "spacing_m", 0),
            (("layout",), "axis", "x"), (("layout",), "radius_m", 1),
            (("layout",), "unknown", 1),
        )
        for path, field, value in changes:
            req = deepcopy(self.request())
            destination = req
            for key in path:
                destination = destination[key]
            destination[field] = value
            with self.subTest(path=path, field=field, value=value), self.assertRaises(RequestError):
                validate_request(req)
        for layout in ({"type": "grid", "rows": 2}, {"type": "line", "axis": "z"}, {"type": "random", "rows": 2}):
            req = self.request()
            req["layout"] = layout
            with self.subTest(layout=layout), self.assertRaises(RequestError):
                validate_request(req)


class TestAssemblyLayouts(unittest.TestCase):
    bounds = (-0.5, 0.5, -0.4, 0.4)
    footprint = (0.08, 0.06)

    def generate(self, layout, count, **kwargs):
        return generate_positions(layout, count, kwargs.pop("bounds_xy", self.bounds), kwargs.pop("footprint_xy", self.footprint), **kwargs)

    def assert_safe(self, points, bounds=None, footprint=None, clearance=0.025, margin=0.03):
        xmin, xmax, ymin, ymax = bounds or self.bounds
        sx, sy = footprint or self.footprint
        for x, y in points:
            self.assertGreaterEqual(x - sx / 2 + 1e-12, xmin + margin)
            self.assertLessEqual(x + sx / 2, xmax - margin + 1e-12)
            self.assertGreaterEqual(y - sy / 2 + 1e-12, ymin + margin)
            self.assertLessEqual(y + sy / 2, ymax - margin + 1e-12)
        for i, first in enumerate(points):
            for second in points[i + 1:]:
                self.assertTrue(abs(first[0] - second[0]) + 1e-12 >= sx + clearance or abs(first[1] - second[1]) + 1e-12 >= sy + clearance)

    def test_rectangular_grid_rows_are_world_x_columns_world_y(self):
        points = self.generate({"type": "grid", "rows": 2, "cols": 3, "spacing_m": 0.12}, 6)
        self.assertEqual(len(points), 6)
        self.assertAlmostEqual(points[3][0] - points[0][0], 0.12)
        self.assertAlmostEqual(points[1][1] - points[0][1], 0.12)
        self.assertEqual(points[0][0], points[1][0])
        self.assert_safe(points)

    def test_line_uses_y_by_default_and_accepts_x(self):
        yline = self.generate({"type": "line"}, 4)
        xline = self.generate({"type": "line", "axis": "x"}, 4)
        self.assertTrue(all(x == 0 for x, _ in yline))
        self.assertTrue(all(y == 0 for _, y in xline))
        self.assert_safe(yline)
        self.assert_safe(xline)

    def test_t_has_five_objects_and_ring_is_conservative(self):
        tshape = self.generate({"type": "t_shape"}, 5)
        self.assertEqual(len(tshape), 5)
        self.assert_safe(tshape)
        ring = self.generate({"type": "ring"}, 8)
        radii = [math.hypot(x, y) for x, y in ring]
        self.assertAlmostEqual(min(radii), max(radii))
        self.assert_safe(ring)
        diameter = math.hypot(*self.footprint) + 0.025
        self.assertGreaterEqual(math.dist(ring[0], ring[1]) + 1e-12, diameter)

    def test_random_seed_reproducibility_gap_and_full_count(self):
        spec = {"type": "random", "spacing_m": 0.14}
        first = self.generate(spec, 15, seed=41)
        self.assertEqual(first, self.generate(spec, 15, seed=41))
        self.assertNotEqual(first, self.generate(spec, 15, seed=42))
        self.assertEqual(len(first), 15)
        self.assert_safe(first)
        for i, point in enumerate(first):
            for other in first[i + 1:]:
                self.assertGreaterEqual(math.dist(point, other) + 1e-12, 0.14)

    def test_edge_margin_uses_full_asset_footprint(self):
        exact_bounds = (2.0, 2.14, -1.0, -0.88)
        point = self.generate({"type": "grid"}, 1, bounds_xy=exact_bounds)[0]
        self.assertAlmostEqual(point[0], 2.07)
        self.assertAlmostEqual(point[1], -0.94)
        self.assert_safe([point], bounds=exact_bounds)
        with self.assertRaises(LayoutError):
            self.generate({"type": "line"}, 1, bounds_xy=(0, 0.139, 0, 1))

    def test_too_small_spacing_and_overfull_layouts_raise(self):
        cases = (
            ({"type": "grid", "rows": 2, "cols": 2, "spacing_m": 0.08}, 4),
            ({"type": "line", "spacing_m": 0.06}, 3),
            ({"type": "ring", "spacing_m": 0.1}, 3),
            ({"type": "ring", "radius_m": 0.01}, 4),
            ({"type": "random", "spacing_m": 0.08}, 4),
            ({"type": "line", "spacing_m": 0.4}, 4),
            ({"type": "ring", "radius_m": 0.5}, 4),
            ({"type": "grid", "rows": 8, "cols": 8}, 64),
        )
        for spec, count in cases:
            with self.subTest(spec=spec, count=count), self.assertRaises(LayoutError):
                self.generate(spec, count)
        with self.assertRaisesRegex(LayoutError, "cannot fit the support area"):
            self.generate({"type": "random"}, 64, bounds_xy=(0, 0.3, 0, 0.3))

    def test_invalid_geometry_is_rejected(self):
        for kwargs in ({"footprint_xy": (0, 0.1)}, {"footprint_xy": (True, 0.1)}, {"bounds_xy": (1, 0, 0, 1)}, {"bounds_xy": (0, math.inf, 0, 1)}, {"clearance_m": -0.1}, {"edge_margin_m": True}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(LayoutError):
                self.generate({"type": "line"}, 2, **kwargs)


if __name__ == "__main__":
    unittest.main()
