from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from geomind_lab.mimic.utils.family_runner_registry import (  # noqa: E402
    RUNNER_DISPATCH_MODE_PERMISSIVE,
    RUNNER_DISPATCH_MODE_STRICT,
    RUNNER_FAMILY_REGISTRY,
    resolve_runner_family,
    resolve_runner_family_for_entry,
    runner_family_config,
)


class TestFamilyRunnerRegistry(unittest.TestCase):
    def test_registry_declares_current_and_future_runner_families(self) -> None:
        self.assertIn("state_refresh", RUNNER_FAMILY_REGISTRY)
        self.assertIn("geometric_structure", RUNNER_FAMILY_REGISTRY)
        self.assertIn("alignment", RUNNER_FAMILY_REGISTRY)
        self.assertIn("routing", RUNNER_FAMILY_REGISTRY)
        self.assertEqual(RUNNER_DISPATCH_MODE_PERMISSIVE, "permissive")
        self.assertEqual(RUNNER_DISPATCH_MODE_STRICT, "strict")

    def test_skill_tasks_route_to_specialized_runner_families(self) -> None:
        self.assertEqual(
            resolve_runner_family(family="shape_completion", skill_type="alignment"),
            "alignment",
        )
        self.assertEqual(
            resolve_runner_family(family="spatial_placement", skill_type="routing"),
            "routing",
        )

    def test_state_and_geometric_tasks_map_to_implemented_runners(self) -> None:
        self.assertEqual(resolve_runner_family(family="state_reindexing"), "state_refresh")
        self.assertEqual(resolve_runner_family(family="historical_identity_tracking"), "state_refresh")
        self.assertEqual(resolve_runner_family(family="explicit_rule_extrapolation"), "geometric_structure")
        self.assertEqual(resolve_runner_family(task_id="square_completion", family="shape_completion"), "geometric_structure")
        self.assertTrue(bool(runner_family_config("state_refresh").get("implemented")))
        self.assertTrue(bool(runner_family_config("geometric_structure").get("implemented")))

    def test_entry_level_resolution_prefers_existing_runner_family(self) -> None:
        self.assertEqual(
            resolve_runner_family_for_entry(
                {
                    "task_id": "place_directional",
                    "family": "spatial_placement",
                    "skill_type": "routing",
                }
            ),
            "routing",
        )


if __name__ == "__main__":
    unittest.main()
