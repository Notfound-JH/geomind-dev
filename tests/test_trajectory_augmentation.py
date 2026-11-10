from __future__ import annotations

import unittest

from geomind_lab.data.trajectory.augmentation import (
    IncrementPlanConfig,
    WaypointPlan,
    choose_valid_waypoint_candidate,
    increment_plan_metrics,
    sample_increment_candidates,
)


class TestTrajectoryAugmentation(unittest.TestCase):
    def setUp(self) -> None:
        self.config = IncrementPlanConfig(
            segment_count=6,
            candidate_count=16,
            reverse_probability=1.0,
            reverse_ratio=0.30,
            max_reverse_absolute_m=0.08,
            max_segment_length_m=0.20,
            max_segment_ratio=1.0,
        )

    def test_increment_candidates_are_seeded_and_preserve_endpoints(self) -> None:
        start, end = (0.1, -0.1, 0.3), (0.4, 0.15, 0.45)
        candidates = sample_increment_candidates(start, end, seed=17, config=self.config)
        self.assertEqual(candidates, sample_increment_candidates(start, end, seed=17, config=self.config))
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(candidate.waypoints[0], start)
            self.assertEqual(candidate.waypoints[-1], end)
            self.assertEqual(len(candidate.increment_vectors), self.config.segment_count)
            for axis in range(3):
                self.assertAlmostEqual(sum(vector[axis] for vector in candidate.increment_vectors), end[axis] - start[axis], places=8)
            self.assertLessEqual(increment_plan_metrics(candidate).endpoint_error_m, 1e-8)

    def test_reverse_increments_respect_configured_absolute_limit(self) -> None:
        candidates = sample_increment_candidates((0.0, 0.0, 0.3), (0.3, 0.0, 0.3), seed=3, config=self.config)
        self.assertTrue(candidates)
        self.assertTrue(any(any(vector[0] < 0.0 for vector in candidate.increment_vectors) for candidate in candidates))
        for candidate in candidates:
            for vector in candidate.increment_vectors:
                self.assertGreaterEqual(vector[0], -self.config.max_reverse_absolute_m - 1e-9)

    def test_increment_signs_stay_within_small_reverse_bound(self) -> None:
        candidates = sample_increment_candidates((0.0, 0.0, 0.3), (0.3, -0.3, 0.3), seed=3, config=self.config)
        self.assertTrue(candidates)
        for candidate in candidates:
            for vector in candidate.increment_vectors:
                self.assertGreater(vector[0], -0.05)
                self.assertLess(vector[1], 0.05)

    def test_permutation_does_not_recombine_axis_components(self) -> None:
        candidates = sample_increment_candidates((0.0, 0.0, 0.2), (0.24, -0.12, 0.32), seed=9, config=self.config)
        for candidate in candidates:
            vectors = candidate.increment_vectors
            self.assertEqual(len(set(vectors)), len(vectors))

    def test_complete_increment_vectors_are_jointly_permuted(self) -> None:
        candidates = sample_increment_candidates((0.0, 0.0, 0.2), (0.24, -0.12, 0.32), seed=9, config=self.config)
        self.assertTrue(candidates)
        self.assertTrue(any(candidate.increment_permutation != tuple(range(self.config.segment_count)) for candidate in candidates))
        self.assertTrue(all(len(candidate.increment_permutation) == self.config.segment_count for candidate in candidates))

    def test_vertical_first_style_frontloads_z_progress(self) -> None:
        candidates = sample_increment_candidates((0.0, 0.0, 0.7), (0.3, 0.2, 0.3), seed=9, config=self.config)
        vertical = next(candidate for candidate in candidates if candidate.path_style == "vertical_first")
        z_progress = [vector[2] for vector in vertical.increment_vectors]
        xy_magnitude = [vector[0] ** 2 + vector[1] ** 2 for vector in vertical.increment_vectors]
        self.assertGreaterEqual(abs(z_progress[0]), abs(z_progress[-1]))
        self.assertLessEqual(xy_magnitude[0], xy_magnitude[-1])

    def test_default_config_uses_high_variation_budget(self) -> None:
        config = IncrementPlanConfig()
        self.assertEqual(config.segment_count, 6)
        self.assertEqual(config.candidate_count, 16)
        self.assertGreaterEqual(config.reverse_probability, 0.75)
        self.assertGreaterEqual(config.max_reverse_absolute_m, 0.08)

    def test_all_candidates_are_audited_and_invalid_set_falls_back(self) -> None:
        fallback = WaypointPlan(waypoints=((0.0, 0.0, 0.3), (0.3, 0.0, 0.3)))
        candidates = sample_increment_candidates(fallback.waypoints[0], fallback.waypoints[-1], seed=9, config=self.config)
        selected, evaluations = choose_valid_waypoint_candidate(
            candidates,
            validator=lambda _: ("workspace_rejected",),
            fallback=fallback,
        )
        self.assertFalse(selected.augmented)
        self.assertEqual(selected.fallback_reason, "no_valid_increment_plan")
        self.assertEqual(len(evaluations), len(candidates))


if __name__ == "__main__":
    unittest.main()
