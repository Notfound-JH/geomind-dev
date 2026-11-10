from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from geomind_lab.data.trajectory import EpisodeCollector, export_episode_h5, resample_episode
from geomind_lab.data.trajectory.validator import validate_dataset_manifest, validate_episode_h5


@unittest.skipUnless(importlib.util.find_spec("h5py"), "requires the trajectory optional dependency")
class TestTrajectoryValidator(unittest.TestCase):
    def _sampled_episode(self) -> dict:
        collector = EpisodeCollector(
            metadata={
                "task_id": "test",
                "variant": "base",
                "layout_key": "layout",
                "scene_id": 65,
                "task_index": 0,
                "layout_index": 0,
                "rollout_index": 0,
                "trajectory_seed": 123,
                "layout_jitter_seed": 1,
                "layout_shift_seed": 2,
                "layout_randomization_record": {},
                "initial_runtime_object_poses": {},
                "sample_frequency_hz": 10.0,
                "action_scale": 0.5,
                "object_info": {},
                "ground_truth": {"gt_spec": {}, "goal_sequence": []},
            },
            sample_hz=10.0,
        )
        for index, timestamp_s in enumerate((0.0, 0.1, 0.2)):
            collector.record_step(
                timestamp_s=timestamp_s,
                sim_step_index=index,
                stage="approach",
                action_eef_abs=[timestamp_s, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                eef_pose_world=[timestamp_s, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                gripper_command=1.0,
                joint_pos=[0.0],
                joint_vel=[0.0],
                gripper_state=[0.04],
            )
        return resample_episode(collector.as_dict())

    def test_validate_episode_h5_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            h5_path = Path(directory) / "episode.h5"
            export_episode_h5(self._sampled_episode(), h5_path)
            report = validate_episode_h5(h5_path)
            self.assertTrue(report["passed"])
            self.assertEqual(report["errors"], [])

    def test_validate_dataset_manifest_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = root / "episode.h5"
            export_episode_h5(self._sampled_episode(), h5_path)

            manifest_payload = {
                "config": {},
                "layouts": [
                    {
                        "layout_index": 0,
                        "episodes": [
                            {
                                "episode_uid": "uid-1",
                                "h5_path": str(h5_path),
                            }
                        ],
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            report = validate_dataset_manifest(manifest_path)
            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["failed_count"], 0)

    def test_validate_episode_h5_rejects_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            h5_path = Path(directory) / "episode.h5"
            export_episode_h5(self._sampled_episode(), h5_path)
            import h5py

            with h5py.File(h5_path, "r+") as handle:
                handle["actions/eef_abs"][0, 0] = math.nan
            report = validate_episode_h5(h5_path)
            self.assertFalse(report["passed"])
            self.assertIn("non_finite_values:actions/eef_abs", report["errors"])


if __name__ == "__main__":
    unittest.main()
