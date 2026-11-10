from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from geomind.core.ids import is_public_task_id, make_task_id, parse_task_id
from geomind.core.registry import GeoMindRegistry
from geomind.core.schema import normalize_task_config
from geomind.tools import generate_vla_task_suite as vla_generator


class TestPublicTaskIds(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = GeoMindRegistry()
        self.registry.load()

    def test_registered_ids_are_unique_compact_filenames(self) -> None:
        ids = self.registry.list_task_ids()
        config_count = sum(
            "task_id" in json.loads(path.read_text(encoding="utf-8"))
            for directory in (self.registry.tasks_dir, self.registry.root / "configs" / "vla_tasks")
            for path in directory.glob("*.json")
        )
        self.assertTrue(ids)
        self.assertEqual(len(ids), config_count)
        for task_id in ids:
            with self.subTest(task_id=task_id):
                self.assertTrue(is_public_task_id(task_id))
                self.assertEqual(task_id.count("_"), 1)
                self.assertEqual(self.registry.get_config_path(task_id).stem, task_id)
                spec = self.registry.get(task_id)
                self.assertEqual(self.registry.get_definition(task_id)["task_id"], spec.task_name)

    def test_public_aliases_resolve_to_their_own_task(self) -> None:
        for task_id in self.registry.list_task_ids():
            for alias in self.registry.get_payload(task_id).get("aliases", []):
                if not is_public_task_id(alias):
                    continue
                with self.subTest(task_id=task_id, alias=alias):
                    self.assertEqual(self.registry.resolve_task_id(alias), task_id)

    def test_old_binding_ids_keep_base_and_vla_separate(self) -> None:
        expected = {
            "BindingBaseBase_AnchorDirection_v0": "SelectByDirection_v0",
            "BindingBase_SelectByDirection_v0": "SelectByDirection_v0",
            "BindingBaseBase_AnchorDirectionVla_v0": "SelectByDirectionVla_v0",
            "BindingAugBase_AnchorDistance_v0": "SelectByDistance_v0",
            "BindingAugBase_AnchorDistanceVla_v0": "SelectByDistanceVla_v0",
            "TrackBaseSem_TrackSingle_v0": "TrackMovingObjectSem_v0",
            "TrackBaseVisual_TrackSingleVla_v0": "TrackMovingObjectVisualVla_v0",
        }
        for old_id, new_id in expected.items():
            with self.subTest(old_id=old_id):
                self.assertEqual(self.registry.resolve_task_id(old_id), new_id)

    def test_scaffold_ids_use_variant_suffixes_without_category(self) -> None:
        for variant, expected in (
            ("base", "SelectByDirection_v2"),
            ("semantic", "SelectByDirectionSem_v2"),
            ("visual", "SelectByDirectionVisual_v2"),
            ("execution", "SelectByDirectionExecution_v2"),
        ):
            with self.subTest(variant=variant):
                self.assertEqual(
                    make_task_id(family="Binding", split="base", variant=variant,
                                 task_name="SelectByDirection", version=2),
                    expected,
                )

    def test_registry_rejects_compact_id_variant_mismatch(self) -> None:
        payload = self.registry.get_payload("SelectByDirection_v0")
        payload["task_id"] = "SelectByDirectionSem_v0"
        with self.assertRaisesRegex(ValueError, "task_id encoded variant mismatch"):
            self.registry._to_spec(payload)

    def test_vla_regeneration_keeps_legacy_aliases_without_base_shadowing(self) -> None:
        base = self.registry.get_definition("SelectByDirection_v0")
        existing = self.registry.get_definition("SelectByDirectionVla_v0")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "source.json"
            src.write_text(json.dumps(base), encoding="utf-8")
            out = root / "vla"
            out.mkdir()
            destination = out / "SelectByDirectionVla_v0.json"
            destination.write_text(json.dumps(existing), encoding="utf-8")
            with patch.object(vla_generator, "OUT_DIR", out):
                vla_generator.emit_task_json(src, {})
            generated = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(generated["task_id"], "SelectByDirection")
        self.assertEqual(normalize_task_config(generated)["task_id"], "SelectByDirectionVla_v0")
        self.assertIn("BindingBaseBase_AnchorDirectionVla_v0", generated["aliases"])
        self.assertNotIn("BindingBaseBase_AnchorDirection_v0", generated["aliases"])
        self.assertNotIn("SelectByDirection_v0", generated["aliases"])
        self.assertEqual(vla_generator._vla_task_id("TrackMovingObjectSem_v0"),
                         "TrackMovingObjectSemVla_v0")
        self.assertEqual(vla_generator._vla_task_id("SelectByDirectionVla_v0"),
                         "SelectByDirectionVla_v0")


if __name__ == "__main__":
    unittest.main()
