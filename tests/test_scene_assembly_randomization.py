from __future__ import annotations

import math
import os
import unittest
from unittest.mock import patch

from geomind.scene_assembly.layouts import LayoutError
from geomind.scene_assembly.randomization import randomize_positions


class SceneAssemblyRandomizationTests(unittest.TestCase):
    def setUp(self):
        self.positions = [[0.20, -0.06], [0.20, 0.06], [0.32, -0.06], [0.32, 0.06]]
        self.kwargs = dict(layout={"type": "grid", "rows": 2, "cols": 2}, footprint_xy=(0.05, 0.05), bounds_xy=(0.0, 0.7, -0.4, 0.4), seed=7, config={"enabled": True, "jitter_xy_m": 0.02, "shift_max_m": 0.15, "yaw_deg": 7.5})

    def test_grid_calls_existing_runtime_planner_and_seed_replays(self):
        from geomind.scene_assembly import randomization as module
        backend = module.randomize_runtime_object_layout
        with patch.object(module, "randomize_runtime_object_layout", wraps=backend) as call:
            positions, record = randomize_positions(self.positions, **self.kwargs)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.kwargs["protocol"]["gt_spec"]["resolver"], "grid_rowcol_slot")
        self.assertEqual(record["strategy"], "grid")
        self.assertEqual((positions, record), randomize_positions(self.positions, **self.kwargs))
        self.assertNotEqual(positions, self.positions)
        changed, _ = randomize_positions(self.positions, **(self.kwargs | {"seed": 8}))
        self.assertNotEqual(changed, positions)

    def test_grid_preserves_row_and_column_labels_and_measured_bounds(self):
        for seed in range(30):
            positions, record = randomize_positions(self.positions, **(self.kwargs | {"seed": seed}))
            self.assertEqual(record["cell_indices"], [[0, 0], [0, 1], [1, 0], [1, 1]])
            # Within-row/column jitter stays small compared with every row/col gap.
            self.assertLess(max(positions[i][0] for i in (0, 1)), min(positions[i][0] for i in (2, 3)))
            self.assertLess(max(positions[i][1] for i in (0, 2)), min(positions[i][1] for i in (1, 3)))
            self.assertLessEqual(abs(positions[0][0] - positions[1][0]), 0.01 + 1e-9)
            self.assertLessEqual(abs(positions[0][1] - positions[2][1]), 0.01 + 1e-9)
            for x, y in positions:
                self.assertTrue(0.025 <= x <= 0.675 and -0.375 <= y <= 0.375)

    def test_shift_changes_do_not_change_grid_relative_geometry(self):
        shifted, _ = randomize_positions(self.positions, **self.kwargs)
        unshifted, _ = randomize_positions(self.positions, **(self.kwargs | {"config": self.kwargs["config"] | {"shift_max_m": 0}}))
        for i in range(1, 4):
            for axis in range(2):
                self.assertAlmostEqual(shifted[i][axis] - shifted[0][axis], unshifted[i][axis] - unshifted[0][axis])

    def test_explicit_settings_ignore_and_preserve_process_environment(self):
        expected = randomize_positions(self.positions, **self.kwargs)
        with patch.dict(os.environ, {"GEOMIND_LAYOUT_JITTER_ENABLE": "0", "GEOMIND_LAYOUT_SHIFT_ENABLE": "0", "GEOMIND_LAYOUT_BOUNDS": "-100,100,-100,100", "GEOMIND_LAYOUT_JITTER_SEED": "999"}):
            before = dict(os.environ)
            self.assertEqual(expected, randomize_positions(self.positions, **self.kwargs))
            self.assertEqual(before, dict(os.environ))

    def test_other_structures_are_translated_rigidly(self):
        for kind in ("line", "ring", "t_shape", "random"):
            positions, record = randomize_positions(self.positions, **(self.kwargs | {"layout": {"type": kind}}))
            self.assertEqual(record["strategy"], "shift_only")
            self.assertFalse(record["jitter_enabled"])
            self.assertEqual(record["local_jitter_policy"], "disabled_to_preserve_structure")
            for i in range(1, 4):
                self.assertAlmostEqual(math.dist(positions[i], positions[0]), math.dist(self.positions[i], self.positions[0]))

    def test_disabled_is_copy_and_fallback_is_error(self):
        positions, record = randomize_positions(self.positions, **(self.kwargs | {"config": {"enabled": False}}))
        self.assertEqual(positions, self.positions)
        self.assertIsNot(positions[0], self.positions[0])
        self.assertIsNone(record)
        with patch("geomind.scene_assembly.randomization.randomize_runtime_object_layout", return_value={"strategy": "frozen"}):
            with self.assertRaisesRegex(LayoutError, "silent fallback"):
                randomize_positions(self.positions, **self.kwargs)

    def test_invalid_bounds_and_mismatched_grid_are_rejected(self):
        with self.assertRaises(LayoutError):
            randomize_positions(self.positions, **(self.kwargs | {"bounds_xy": (0.2, 0.3, -0.4, 0.4)}))
        with self.assertRaises(LayoutError):
            randomize_positions(self.positions, **(self.kwargs | {"layout": {"type": "grid", "rows": 3, "cols": 2}}))
        with self.assertRaisesRegex(LayoutError, "row-major"):
            randomize_positions([self.positions[i] for i in (2, 3, 0, 1)], **self.kwargs)
        with self.assertRaises(LayoutError):
            randomize_positions(self.positions, **(self.kwargs | {"config": {"enabled": True, "jitter_xy_m": float("nan")}}))


if __name__ == "__main__":
    unittest.main()
