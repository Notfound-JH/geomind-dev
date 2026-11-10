from __future__ import annotations

from collections import defaultdict
import unittest

from geomind.core.registry import GeoMindRegistry
from geomind.core.schema import normalize_task_config, upgrade_task_config
from geomind.core.taxonomy import CANONICAL_TASK_FAMILIES


class TestCanonicalTaskFamilies(unittest.TestCase):
    def test_every_registered_condition_and_interface_matches_final_families(self) -> None:
        registry = GeoMindRegistry()
        registry.load()
        actual = defaultdict(set)
        for registration_id in registry.list_task_ids():
            spec = registry.get(registration_id)
            actual[spec.task_family].add(spec.public_task_id)
        expected = {family: set(tasks) for family, tasks in CANONICAL_TASK_FAMILIES.items()}
        self.assertEqual(dict(actual), expected)
        self.assertEqual([len(expected[family]) for family in CANONICAL_TASK_FAMILIES], [4, 5, 5, 7, 3, 3])
        self.assertEqual(sum(map(len, expected.values())), 27)

    def test_public_reclassification_preserves_runtime_oracle_families(self) -> None:
        registry = GeoMindRegistry()
        registry.load()
        for task_id, family in (
            ("SelectByOrderFirst_v0", "TargetIdentification"),
            ("SelectByGridIndex_v0", "TargetIdentification"),
            ("SelectWithinSubset_v0", "ConstraintComposition"),
        ):
            with self.subTest(task_id=task_id):
                spec = registry.get(task_id)
                self.assertEqual(spec.task_family, family)
                self.assertEqual(spec.family, "state_reindexing")

    def test_generated_definitions_use_task_family_and_reject_wrong_public_family(self) -> None:
        definition = upgrade_task_config({
            "task_id": "SelectByOrder_v0", "task_name": "SelectByOrder", "version": 0,
            "family": "state_reindexing", "canonical_task_id": "ordinal_insert_static",
            "variant": "base", "task_version": "base",
        })
        self.assertEqual(definition["family"], "TargetIdentification")
        definition["family"] = "RelationalGrounding"
        with self.assertRaisesRegex(ValueError, "belongs to TargetIdentification"):
            normalize_task_config(definition)


if __name__ == "__main__":
    unittest.main()
