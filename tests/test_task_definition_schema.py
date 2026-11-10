from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from geomind.core.registry import GeoMindRegistry
from geomind.core.schema import normalize_task_config
from scripts.run_vla_eval import load_vla_tasks


class TestTaskDefinitionSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = GeoMindRegistry()
        self.registry.load()

    def test_all_definitions_have_independent_fields_and_runtime_dispatch(self) -> None:
        for registration_id in self.registry.list_task_ids():
            definition = self.registry.get_definition(registration_id)
            with self.subTest(registration_id=registration_id):
                self.assertEqual(definition["schema_version"], 3)
                self.assertNotIn("runtime", definition)
                self.assertNotIn("canonical_task_id", definition)
                self.assertNotIn("_v", definition["task_id"])
                payload = normalize_task_config(definition)
                self.assertEqual(normalize_task_config(payload), payload)
                self.assertEqual(payload["task_id"], registration_id)
                spec = self.registry.get(registration_id)
                self.assertEqual(spec.public_task_id, definition["task_id"])
                compat = self.registry.get_compat_definition(registration_id)
                self.assertEqual(spec.family, compat["runtime"]["protocol_family"])
                self.assertEqual(spec.task_family, compat["family"])

    def test_bare_names_default_to_base_standard_and_select_variants(self) -> None:
        self.assertEqual(self.registry.resolve_task_id("SelectByDirection"), "SelectByDirection_v0")
        self.assertEqual(self.registry.resolve_task_id("TrackMovingObject", condition="Visual"),
                         "TrackMovingObjectVisual_v0")
        self.assertEqual(self.registry.resolve_task_id("TrackMovingObject", condition="Visual", interface="Vla"),
                         "TrackMovingObjectVisualVla_v0")
        self.assertEqual(self.registry.get("AnchorDistance").condition, "Semantic")
        with self.assertRaises(KeyError):
            self.registry.resolve_task_id("SelectByDirection", layout="Line")
        with self.assertRaises(ValueError):
            self.registry.resolve_task_id("SelectByDirection_v0", condition="Visual")

    def test_independent_version_drives_derived_and_legacy_registration_keys(self) -> None:
        definition = self.registry.get_compat_definition("SelectByDirection")
        definition["runtime"].pop("registration_id")
        definition["version"] = 2
        self.assertEqual(normalize_task_config(definition)["task_id"], "SelectByDirection_v2")
        definition["runtime"]["registration_id"] = "SelectByDirection_v0"
        self.assertEqual(normalize_task_config(definition)["task_id"], "SelectByDirection_v2")
        definition["runtime"]["registration_id"] = "not-versioned"
        with self.assertRaisesRegex(ValueError, "must end"):
            normalize_task_config(definition)

    def test_missing_identity_fields_and_invalid_interfaces_fail(self) -> None:
        original = self.registry.get_compat_definition("SelectByDirection")
        for key in ("task_id", "family", "condition", "layout", "version"):
            definition = deepcopy(original)
            definition.pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Missing task definition"):
                normalize_task_config(definition)
        original["runtime"]["interface"] = "Vla"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            normalize_task_config(original)

    def test_multiple_layouts_require_an_explicit_layout_selector(self) -> None:
        cross = self.registry.get_compat_definition("SelectByDirection")
        line = deepcopy(cross)
        line["layout"] = "Line"
        # An internal registration key distinguishes two layout definitions.
        line["runtime"]["registration_id"] = "SelectByDirectionLine_v0"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "configs" / "tasks"
            directory.mkdir(parents=True)
            for index, definition in enumerate((cross, line)):
                (directory / f"task_{index}.json").write_text(json.dumps(definition), encoding="utf-8")
            registry = GeoMindRegistry(root=root)
            registry.load()
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                registry.resolve_task_id("SelectByDirection")
            self.assertEqual(registry.resolve_task_id("SelectByDirection", layout="Line"),
                             "SelectByDirectionLine_v0")
            self.assertEqual(registry.resolve_task_id("SelectByDirection", layout="Cross"),
                             "SelectByDirection_v0")

    def test_vla_evaluation_reads_runtime_fields_without_collapsing_conditions(self) -> None:
        tasks = load_vla_tasks(self.registry.root.parent)
        self.assertEqual(len(tasks), len({task["task_id"] for task in tasks}))
        by_id = {task["task_id"]: task for task in tasks}
        base = by_id["TrackMovingObjectVla_v0"]
        visual = by_id["TrackMovingObjectVisualVla_v0"]
        self.assertEqual(base["public_task_id"], visual["public_task_id"])
        self.assertEqual((base["condition"], visual["condition"]), ("Base", "Visual"))
        self.assertEqual(base["family"], "historical_identity_tracking")
        self.assertTrue(base["annotation_layout_key"])


if __name__ == "__main__":
    unittest.main()
