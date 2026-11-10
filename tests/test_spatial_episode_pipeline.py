from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from geomind import create_env
from geomind.core.registry import GeoMindRegistry
from geomind.episodes import ContractError, EpisodeResolver, SemanticEpisode, Snapshot, evaluate_answer, solve
from geomind.tools.generate_spatial_suite import generate, verify
from geomind.tools.migrate_episode_configs import migrate, write


CONFIGS = Path(__file__).resolve().parents[1] / "geomind/configs"


class TestSpatialEpisodePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.manifest = generate(cls.root, [11, 12])
        cls.resolver = EpisodeResolver(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def memory(self):
        return SemanticEpisode(self.resolver.resolve("DetectSceneChange_v0"))

    def ready(self, env):
        obs, _ = env.reset()
        while not obs["allow_response"]:
            obs, *_ = env.step()
        return obs

    def test_automated_generation_and_independent_oracle_verification(self):
        report = verify(self.root, self.manifest)
        self.assertEqual(report["verified_cases"], 6)
        self.assertTrue(report["passed"])
        self.assertFalse(report["physical_execution_verified"])
        for record in report["results"]:
            self.assertLess(record["oracle_error_m"], 1e-12)
            self.assertTrue(record["incorrect_answer_rejected"])
            self.assertTrue(record["evidence_replay_equal"])

    def test_selected_tasks_have_explicit_native_assembled_registration_and_keep_original_episode_api(self):
        registry = GeoMindRegistry()
        for name in ("DetectSceneChange", "ProjectOntoLine", "ReflectAcrossLine"):
            spec = registry.get(name)
            self.assertEqual(spec.implementation_status, "implemented")
            self.assertTrue(registry.capabilities(name)["semantic"])
            self.assertTrue(registry.capabilities(name)["isaaclab"])
            self.assertTrue(registry.capabilities(name)["isaaclab_episode"])
            self.assertFalse(registry.capabilities(name)["isaaclab_legacy"])
            self.assertTrue(registry.capabilities(name)['isaaclab_assembled'])
        self.assertEqual(registry.get("DetectSceneChange").domain, "SpatialMemory")
        self.assertEqual(registry.get("ProjectOntoLine").domain, "SpatialReasoning")

    def test_public_factory_executes_native_episode_without_legacy_gt_lookup(self):
        env = create_env("ProjectOntoLine", backend="semantic", root=self.root)
        obs, _ = env.reset()
        self.assertTrue(obs["allow_response"])
        self.assertNotIn("ground_truth", obs)
        self.assertEqual(env.result["task_id"], "ProjectOntoLine")

    def test_semantic_factory_does_not_silently_ignore_parameter_overrides(self):
        with self.assertRaisesRegex(ValueError, "unsupported overrides"):
            create_env("ProjectOntoLine", backend="semantic", root=self.root, parameters={})

    def test_projection_uses_infinite_line_not_clamped_segment(self):
        resolved = self.resolver.resolve("ProjectOntoLine_v0")
        env = SemanticEpisode(resolved)
        env.reset()
        state = env.snapshot().to_dict()
        state["positions_m"] = {"line_a": [-0.1, 0, 0.2], "line_b": [0.1, 0, 0.2], "point": [0.3, 0.2, 0.2]}
        result = solve(resolved, Snapshot.from_dict(state))
        self.assertLess(math.dist(result["ground_truth"]["position_m"], [0.3, 0.0, 0.2]), 1e-12)

    def test_reflection_is_involution_in_three_dimensions(self):
        resolved = self.resolver.resolve("ReflectAcrossLine_v0")
        env = SemanticEpisode(resolved)
        env.reset()
        original = env.snapshot().to_dict()
        reflected = solve(resolved, Snapshot.from_dict(original))
        updated = deepcopy(original)
        updated["positions_m"]["point"] = reflected["ground_truth"]["position_m"]
        twice = solve(resolved, Snapshot.from_dict(updated))
        self.assertLess(math.dist(twice["ground_truth"]["position_m"], original["positions_m"]["point"]), 1e-12)

    def test_coincident_line_endpoints_rejected_for_both_tasks(self):
        for name in ("ProjectOntoLine", "ReflectAcrossLine"):
            resolved = self.resolver.resolve(name + "_v0")
            env = SemanticEpisode(resolved)
            env.reset()
            state = env.snapshot().to_dict()
            state["positions_m"]["line_b"] = state["positions_m"]["line_a"]
            with self.subTest(task=name), self.assertRaisesRegex(ContractError, "Degenerate line"):
                solve(resolved, Snapshot.from_dict(state))

    def test_line_endpoint_order_does_not_change_answer(self):
        resolved = self.resolver.resolve("ProjectOntoLine_v0")
        env = SemanticEpisode(resolved)
        env.reset()
        state = env.snapshot().to_dict()
        expected = solve(resolved, Snapshot.from_dict(state))["ground_truth"]["position_m"]
        state["positions_m"]["line_a"], state["positions_m"]["line_b"] = state["positions_m"]["line_b"], state["positions_m"]["line_a"]
        actual = solve(resolved, Snapshot.from_dict(state))["ground_truth"]["position_m"]
        self.assertLess(math.dist(actual, expected), 1e-12)

    def test_memory_visibility_query_timing_and_event_application(self):
        env = self.memory()
        observed, _ = env.reset()
        self.assertTrue(observed["scene_visible"])
        self.assertFalse(observed["allow_response"])
        self.assertNotIn("moved", observed["instruction"])
        env.step()
        delayed, *_ = env.step()
        self.assertFalse(delayed["scene_visible"])
        self.assertEqual(delayed["objects"], [])
        reference = env.history["observe.End"].to_dict()
        for _ in range(3):
            final, *_ = env.step()
        self.assertTrue(final["scene_visible"])
        self.assertTrue(final["allow_response"])
        self.assertEqual(set(final), {"instruction", "phase", "step", "scene_visible", "objects", "frame", "allow_response"})
        self.assertNotEqual(env.snapshot().to_dict()["positions_m"], reference["positions_m"])
        self.assertEqual(env.history["observe.End"].to_dict(), reference)

    def test_memory_rejects_response_during_observation(self):
        env = self.memory()
        env.reset()
        with self.assertRaisesRegex(ContractError, "before retrieval"):
            env.step({"object_id": "block_0"})

    def test_memory_resets_history_and_run_identity(self):
        env = self.memory()
        self.ready(env)
        old_history = deepcopy(env.history)
        old_run = env.instance_id
        env.reset()
        self.assertEqual(env.history, {})
        self.assertNotEqual(old_run, env.instance_id)
        while not env.phase().get("allow_response", False):
            env.step()
        with self.assertRaisesRegex(ContractError, "different run"):
            solve(env.resolved, env.snapshot(), history=old_history)

    def test_change_detection_rejects_no_change_multiple_changes_and_threshold_boundary(self):
        env = self.memory()
        self.ready(env)
        reference = env.history["observe.End"].to_dict()["positions_m"]
        for count, shift, message in ((0, 0.12, "exactly one"), (2, 0.12, "exactly one"), (1, 0.03, "Ambiguous change")):
            state = env.snapshot().to_dict()
            state["positions_m"] = deepcopy(reference)
            for identity in list(reference)[:count]:
                state["positions_m"][identity][1] += shift
            with self.subTest(count=count, shift=shift), self.assertRaisesRegex(ContractError, message):
                solve(env.resolved, Snapshot.from_dict(state), history=env.history)

    def test_point_evaluator_rejects_bad_frame_and_nonfinite_predictions(self):
        resolved = self.resolver.resolve("ProjectOntoLine_v0")
        env = SemanticEpisode(resolved)
        env.reset()
        for answer, error in (({"position_m": [0, 0, 0], "frame": "camera"}, "frame mismatch"),
                              ({"position_m": [float("nan"), 0, 0], "frame": "table"}, "finite")):
            with self.subTest(error=error), self.assertRaisesRegex(ContractError, error):
                evaluate_answer(resolved, env.result, answer)

    def test_finished_episode_requires_reset(self):
        env = self.memory()
        self.ready(env)
        env.step({"object_id": env.result["ground_truth"]["object_id"]})
        with self.assertRaisesRegex(ContractError, "reset"):
            env.step({"object_id": "block_0"})

    def test_exported_state_round_trip_replays_the_same_gt(self):
        env = self.memory()
        self.ready(env)
        exported = env.export_evidence()
        result = solve(env.resolved, Snapshot.from_dict(exported["current"]), history={
            k: Snapshot.from_dict(v) for k, v in exported["history"].items()})
        self.assertEqual(result, env.result)

    def test_cli_replays_exported_evidence(self):
        env = self.memory()
        self.ready(env)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            write(path, env.export_evidence())
            result = subprocess.run([sys.executable, "-m", "geomind.tools.resolve_episode",
                "DetectSceneChange_v0", "--root", str(self.root), "--state", str(path)],
                capture_output=True, text=True, check=True)
            actual = json.loads(result.stdout)
            self.assertEqual(actual["ground_truth"], env.result["ground_truth"])
            self.assertEqual(actual["evidence_backend"], "semantic")

    def test_scaffold_cli_validates_before_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "configs"
            shutil.copytree(self.root, root)
            params = deepcopy(self.resolver.resolve("ProjectOntoLine_v0").episode["parameters"])
            parameter_file = Path(directory) / "parameters.json"
            write(parameter_file, params)
            command = [sys.executable, "-m", "geomind.tools.scaffold_task", "--root", str(root),
                       "--task-id", "ProjectOntoLine", "--scene-id", "semantic_table", "--layout-id", "geometry_s11",
                       "--parameters-file", str(parameter_file), "--response", "ReportPoint",
                       "--position-tolerance-m", "0.001", "--seed", "11", "--episode-id", "new_projection"]
            subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(EpisodeResolver(root).resolve("new_projection").episode["parameters"], params)
            params["typo"] = 1
            write(parameter_file, params)
            command[-1] = "invalid_projection"
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2)
            self.assertFalse((root / "episodes/invalid_projection.json").exists())

    def test_invalid_event_object_and_duplicate_events_rejected_at_load_time(self):
        source = self.resolver.resolve("DetectSceneChange_v0")
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "configs"
            shutil.copytree(self.root, copied)
            protocol_path = copied / "protocols" / f"{source.protocol['protocol_id']}.json"
            bad = deepcopy(source.protocol)
            bad["phases"][1]["events"][0]["object_id"] = "missing"
            write(protocol_path, bad)
            with self.assertRaisesRegex(ContractError, "Unknown event object"):
                EpisodeResolver(copied).resolve("DetectSceneChange_v0")
            bad = deepcopy(source.protocol)
            bad["phases"][1]["events"] *= 2
            write(protocol_path, bad)
            with self.assertRaisesRegex(ContractError, "Conflicting simultaneous"):
                EpisodeResolver(copied).resolve("DetectSceneChange_v0")


class TestV3Migration(unittest.TestCase):
    def test_all_active_registrations_are_v3_and_keep_their_runtime_identity(self):
        registry = GeoMindRegistry()
        for identity in registry.list_task_ids():
            definition = registry.get_definition(identity)
            self.assertEqual(definition["schema_version"], 3)
            self.assertEqual(definition["episode_id"], identity)
            self.assertFalse({"runtime", "canonical_task_id", "legacy_task_id", "protocol_family", "evaluator_key"} & definition.keys())
            if definition["execution"] == "LegacyIsaacLab":
                self.assertIn("scene_id", definition)
                layout = json.loads((CONFIGS / "layouts" / f"{definition['layout_id']}.json").read_text())
                self.assertNotIn("tasks", layout)
                self.assertFalse({"preview_proxy_goal", "layout_constraints"} & layout["initialization"].keys())
                self.assertEqual(registry.get_payload(identity)["backend_kwargs"]["seed"], definition["seed"])

    def test_migration_is_idempotent_and_does_not_rewrite_active_episode_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "configs"
            shutil.copytree(CONFIGS, root)
            before = {(folder, p.name): p.read_bytes() for folder in ("tasks", "vla_tasks")
                      for p in (root / folder).glob("*.json")}
            result = migrate(root)
            self.assertEqual(result["migrated"], [])
            self.assertEqual(len(result["already_v3"]), len(before))
            self.assertTrue(all((root / folder / name).read_bytes() == contents for (folder, name), contents in before.items()))

    def test_legacy_layout_override_is_read_by_the_old_runtime_without_moving_gt_into_layout(self):
        from geomind_lab.tasks.manager_based.manipulation.geomind import task_specs
        source = task_specs.load_annotated_tasks()["state_reindex_blocks_line_4"]
        layout = json.loads((CONFIGS / "layouts/state_reindex_blocks_line_4.json").read_text())
        name = layout["initialization"]["runtime_objects"][0]["name"]
        layout["initialization"]["runtime_objects"][0]["pose"]["xy"][0] += 0.01
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_file = root / "annotations.json"
            write(annotation_file, {"state_reindex_blocks_line_4": source})
            write(root / "geomind/configs/layouts/state_reindex_blocks_line_4.json", layout)
            with patch.object(task_specs, "repo_root", return_value=root), \
                 patch.object(task_specs, "resolve_annotated_tasks_path", return_value=annotation_file):
                result = task_specs.load_annotated_tasks()["state_reindex_blocks_line_4"]
            actual = result["tasks"][0]["annotation"]["scene_adaptation"]["runtime_objects"]
            expected = layout["initialization"]["runtime_objects"][0]["pose"]["xy"]
            self.assertEqual(next(o for o in actual if o["name"] == name)["pose"]["xy"], expected)
            self.assertEqual(result["tasks"][0]["annotation"]["protocol"], source["tasks"][0]["annotation"]["protocol"])

    def test_default_standard_and_explicit_vla_catalogs_do_not_mix(self):
        from geomind_lab.mimic.utils.benchmark_manifest_utils import resolve_task_entries
        self.assertTrue(all(e["interface"] == "Standard" for e in resolve_task_entries("grasp_anchor_distance")))
        vla = resolve_task_entries("grasp_anchor_distance_scene10_base_vla")
        self.assertEqual(len(vla), 1)
        self.assertEqual(vla[0]["interface"], "Vla")


if __name__ == "__main__":
    unittest.main()
