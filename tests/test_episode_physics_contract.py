from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from geomind import create_env
from geomind.episodes import ContractError, EpisodeResolver
from geomind.episodes.isaaclab_physics import IsaacLabEpisode
from geomind.tools.generate_spatial_suite import generate
from geomind.tools.migrate_episode_configs import write
from geomind.tools.run_episode_physics import supervise, verification_complete


class TestCaseCompleteness(unittest.TestCase):
    def test_fresh_negative_run_requires_a_real_verified_control(self):
        control={'negative_control':True,'verification_passed':True}
        self.assertTrue(verification_complete([control],None,0,1))
        self.assertFalse(verification_complete([],None,0,1))
        self.assertFalse(verification_complete([dict(control,verification_passed=False)],None,0,1))
        self.assertFalse(verification_complete([control],'worker failure',0,1))

    def test_combined_run_cannot_pass_with_missing_or_duplicate_control(self):
        positive={'negative_control':False,'verification_passed':True}
        control={'negative_control':True,'verification_passed':True}
        self.assertTrue(verification_complete([positive,control],None,1,1))
        self.assertFalse(verification_complete([positive],None,1,1))
        self.assertFalse(verification_complete([positive,control,control],None,1,1))


class TestPhysicalConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        generate(cls.root, [0])
        cls.resolver = EpisodeResolver(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_factory_dispatches_to_native_physics_with_explicit_headless_options(self):
        with patch("geomind.episodes.isaaclab_physics.IsaacLabEpisode") as backend:
            result = create_env("ProjectOntoLine", backend="isaaclab", root=self.root, headless=True, device="cuda:0")
            self.assertIs(result, backend.return_value)
            self.assertEqual(backend.call_args.args[0].episode["episode_id"], "ProjectOntoLine_v0")
            self.assertEqual(backend.call_args.kwargs, {"headless": True, "device": "cuda:0"})

    def test_nonheadless_request_rejected_before_simulator_imports(self):
        resolved = self.resolver.resolve("ProjectOntoLine_v0")
        with self.assertRaisesRegex(ContractError, "explicitly headless"):
            IsaacLabEpisode(resolved, headless=False)

    def test_no_implicit_physical_action_profile(self):
        resolved = deepcopy(self.resolver.resolve("ProjectOntoLine_v0"))
        resolved.episode.pop("physical_execution")
        with self.assertRaisesRegex(ContractError, "explicit physical_execution"):
            IsaacLabEpisode(resolved)

    def test_physical_tolerances_and_offsets_are_validated_separately_from_semantic_tolerance(self):
        from geomind.episodes.physical_contract import validate_execution
        original = self.resolver.resolve("ProjectOntoLine_v0")
        self.assertNotEqual(original.episode["evaluation"]["position_tolerance_m"],
                            original.episode["physical_execution"]["position_tolerance_m"])
        for field, value in (("position_tolerance_m", 0), ("dwell_steps", True),
                             ("target_offset_m", [0, 0, 0]), ("max_steps", 1)):
            episode = deepcopy(original.episode)
            episode["physical_execution"][field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_execution(episode, original.scene)

    def test_physical_clock_must_be_exact_integer_multiple(self):
        from geomind.episodes.physical_contract import validate_physics
        scene = deepcopy(self.resolver.resolve("ProjectOntoLine_v0").scene)
        scene["physics"]["protocol_step_s"] = 0.5001
        with self.assertRaisesRegex(ContractError, "integer number"):
            validate_physics(scene)

    def test_unsupported_scene_mode_cannot_silently_replace_requested_usd(self):
        from geomind.episodes.physical_contract import validate_physics
        scene = deepcopy(self.resolver.resolve("ProjectOntoLine_v0").scene)
        scene["physics"]["scene_mode"] = "UnknownUSD"
        with self.assertRaisesRegex(ContractError, "scene mode"):
            validate_physics(scene)

    def test_fr3_reach_is_supported_but_unknown_robots_and_placement_are_rejected(self):
        from geomind.episodes.physical_contract import validate_physics, validate_execution
        resolved = deepcopy(self.resolver.resolve("ProjectOntoLine_v0"))
        resolved.scene["robot"] = "fr3"
        validate_physics(resolved.scene)
        validate_execution(resolved.episode, resolved.scene)
        resolved.scene["robot"] = "unknown"
        with self.assertRaisesRegex(ContractError, "Panda and FR3"):
            validate_physics(resolved.scene)
        resolved.scene["robot"] = "fr3"
        resolved.episode["physical_execution"].update(action="PickPlaceMidpoint", moving_object="point")
        with self.assertRaisesRegex(ContractError, "Placement currently supports Panda only"):
            validate_execution(resolved.episode, resolved.scene)

    def test_piper_target_profile_is_accepted_without_simulator_imports(self):
        from geomind.episodes.physical_contract import validate_physics, validate_execution
        resolved = deepcopy(self.resolver.resolve("ProjectOntoLine_v0"))
        resolved.scene["robot"] = "piper"
        validate_physics(resolved.scene)
        validate_execution(resolved.episode, resolved.scene)
        resolved.episode["physical_execution"].update(action="PickPlaceMidpoint", moving_object="point")
        with self.assertRaisesRegex(ContractError, "Placement currently supports Panda only"):
            validate_execution(resolved.episode, resolved.scene)

    def test_dual_piper_target_scene_is_accepted(self):
        from geomind.episodes.physical_contract import validate_physics, validate_execution
        resolved = deepcopy(self.resolver.resolve("ProjectOntoLine_v0"))
        resolved.scene["robot"] = "piper_dual"
        resolved.episode["physical_execution"]["active_arm"] = "left"
        validate_physics(resolved.scene)
        validate_execution(resolved.episode, resolved.scene)

    def test_memory_initial_geometry_has_no_elevated_position_markers(self):
        resolved = self.resolver.resolve("DetectSceneChange_v0")
        expected = resolved.scene["physics"]["object_size_m"][2] / 2
        self.assertTrue(all(obj["position_m"][2] == expected for obj in resolved.layout["objects"]))


class TestHeadlessSupervisor(unittest.TestCase):
    def run_fake(self, *, verified=True, bounded=False, stale=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "report.json", {"run_token": "old" if stale else "fresh", "physical_execution_verified": verified})
            process = MagicMock()
            process.pid = 12345
            process.returncode = -9 if bounded else 0
            process.poll.side_effect = [None, -9] if bounded else [0, 0]
            args = SimpleNamespace(timeout_s=600.0, output=root)
            with patch("geomind.tools.run_episode_physics.uuid.uuid4", return_value=SimpleNamespace(hex="fresh")), \
                 patch("geomind.tools.run_episode_physics.subprocess.Popen", return_value=process) as spawn, \
                 patch("geomind.tools.run_episode_physics.os.killpg") as kill, \
                 patch("geomind.tools.run_episode_physics.time.monotonic", side_effect=[0, 1, 20]):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    code = supervise(args)
                self.assertTrue(spawn.call_args.kwargs["start_new_session"])
                if bounded:
                    kill.assert_called_once_with(12345, signal.SIGKILL)
            return code, json.loads((root / "report.json").read_text())

    def test_clean_worker_reports_success(self):
        code, report = self.run_fake()
        self.assertEqual(code, 0)
        self.assertEqual(report["shutdown"], "worker_exited")

    def test_bounded_cleanup_preserves_only_completed_physical_evidence(self):
        code, report = self.run_fake(bounded=True)
        self.assertEqual(code, 0)
        self.assertEqual(report["shutdown"], "bounded_cleanup_after_report")

    def test_failed_physical_trial_cannot_be_reported_as_success(self):
        code, report = self.run_fake(verified=False)
        self.assertEqual(code, 1)
        self.assertFalse(report["physical_execution_verified"])

    def test_old_success_report_cannot_mask_a_new_worker_failure(self):
        code, report = self.run_fake(stale=True)
        self.assertEqual(code, 1)
        self.assertEqual(report["run_token"], "fresh")
        self.assertFalse(report["physical_execution_verified"])


if __name__ == "__main__":
    unittest.main()
