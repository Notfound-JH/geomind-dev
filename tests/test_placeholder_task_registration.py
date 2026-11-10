from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym

from geomind.core.registry import registry
from geomind.core.resolver import resolve_task_request
from geomind.tools import generate_vla_task_suite as vla_generator


NATIVE_TASKS = {
    "SelectByBranchPosition": "TargetIdentification",
    "SelectByRegion": "TargetIdentification",
    "SelectByPathDistance": "RelationalGrounding",
    "SelectByTwoRelations": "ConstraintComposition",
    "SelectByIndirectReference": "ConstraintComposition",
    "ReflectAcrossLine": "GeometricInference",
    "ProjectOntoLine": "GeometricInference",
    "DetectSceneChange": "SceneUpdating",
    "UpdateEmptySlot": "SceneUpdating",
    "TrackObjectOrigin": "IdentityTracking",
}


class TestNativeTaskRegistration(unittest.TestCase):
    def test_names_and_versioned_ids_resolve_with_families(self) -> None:
        for name, family in NATIVE_TASKS.items():
            with self.subTest(name=name):
                spec = registry.get(name)
                self.assertEqual(spec.task_id, 'TrackObjectOriginClutter_v0' if name=='TrackObjectOrigin' else f"{name}_v0")
                self.assertEqual(registry.get(spec.task_id), spec)
                self.assertEqual(spec.task_family, family)
                self.assertEqual(spec.implementation_status, "implemented")
                self.assertTrue(spec.instruction)
                self.assertEqual(spec.evaluator_key, "assembled_object_state")
                self.assertTrue(spec.meta['native_assembly']['recipe_module'].startswith('geomind.episodes.extended_'))

    def test_resolver_selects_native_manipulation_and_rejects_mock_fallback(self) -> None:
        for name in NATIVE_TASKS:
            with self.subTest(name=name):
                for backend in (None,'isaaclab','assembled_native'):
                    request=resolve_task_request(name,backend=backend)
                    self.assertEqual(request.env_cfg.runtime_mode,'assembled_native_pipeline')
                    self.assertEqual(request.env_cfg.backend,'assembled_native')
                    self.assertEqual(request.env_cfg.backend_kwargs['runner_entrypoint'],'geomind.tools.regenerate_registered_tasks')
                with self.assertRaisesRegex(ValueError, 'native assembled manipulation'):
                    resolve_task_request(name,backend='mock')
                capabilities=registry.capabilities(name)
                self.assertTrue(capabilities['isaaclab_assembled'])
                self.assertFalse(capabilities['isaaclab_legacy'])

    def test_existing_adjacency_keeps_implementation(self) -> None:
        spec = registry.get("SelectByAdjacency")
        self.assertEqual(spec.task_family, "RelationalGrounding")
        self.assertEqual(spec.implementation_status, "implemented")
        self.assertEqual(spec.canonical_task_id, "reindex_neighbor")

    def test_distance_placeholder_is_replaced_by_the_existing_implementation(self) -> None:
        spec = registry.get("SelectByDistance")
        self.assertEqual(spec.implementation_status, "implemented")
        self.assertEqual(spec.task_family, "RelationalGrounding")
        self.assertEqual(spec.canonical_task_id, "grasp_anchor_distance")
        self.assertEqual(registry.resolve_task_id("AnchorDistance"), "SelectByDistance_v0")
        self.assertEqual(registry.resolve_task_id("AnchorDistance", interface="Vla"), "SelectByDistanceVla_v0")
        self.assertEqual(registry.resolve_task_id("AnchorDistanceVla_v0"), "SelectByDistanceVla_v0")

    def test_between_is_a_relation_variant_outside_the_canonical_registry(self) -> None:
        import json
        for task_id in ("SelectBetweenObjects", "SelectBetweenObjects_v0"):
            with self.subTest(task_id=task_id), self.assertRaises(KeyError):
                registry.resolve_task_id(task_id)
        relations = json.loads((registry.root / "configs" / "relational_variants.json").read_text())
        self.assertEqual(relations["family"], "RelationalGrounding")
        self.assertFalse(relations["relations"]["Between"]["standalone_task"])

    def test_implemented_native_recipes_are_visible_in_the_registry(self) -> None:
        native = {registry.get(name).task_id for name in NATIVE_TASKS}
        self.assertTrue(native <= set(registry.list_task_ids(include_placeholders=False)))
        self.assertTrue(native <= set(registry.select_by_split("base")))
        standard=[registry.get(name) for name in registry.list_task_ids()
                  if registry.get(name).interface=='Standard']
        self.assertEqual(len({spec.public_task_id for spec in standard}),27)
        self.assertTrue(all(spec.implementation_status=='implemented' for spec in standard))

    def test_vla_generator_does_not_fabricate_legacy_manager_bindings_for_native_recipes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch.object(vla_generator, "OUT_DIR", output):
                for name in NATIVE_TASKS:
                    vla_generator.emit_task_json(registry.get_config_path(name), {})
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
