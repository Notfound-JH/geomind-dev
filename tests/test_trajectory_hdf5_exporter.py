from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from geomind_lab.data.trajectory import CameraH5Stream, EpisodeCollector, export_episode_h5, resample_episode


@unittest.skipUnless(importlib.util.find_spec("h5py"), "requires the trajectory optional dependency")
class TestTrajectoryHdf5Exporter(unittest.TestCase):
    def _sampled_episode(self) -> dict:
        collector = EpisodeCollector(metadata={"task_id": "test"}, sample_hz=10.0)
        for index, timestamp_s in enumerate((0.0, 0.1)):
            collector.record_step(
                timestamp_s=timestamp_s,
                sim_step_index=index,
                stage="approach",
                action_eef_abs=[timestamp_s, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                eef_pose_world=[timestamp_s - 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                gripper_command=1.0,
                joint_pos=[0.0],
                joint_vel=[0.0],
                gripper_state=[0.04],
            )
        return resample_episode(collector.as_dict())

    def test_h5_contains_actual_eef_pose(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            export_episode_h5(self._sampled_episode(), path)
            with h5py.File(path, "r") as handle:
                self.assertIn("observations/eef_pose_world", handle)
                self.assertEqual(handle["observations/eef_pose_world"].shape, (2, 7))
                self.assertIn("sim_step_index", handle["steps"])
                self.assertIn("events/object_grasped", handle)
                self.assertEqual(handle["events/object_grasped"].shape, (2,))
                self.assertIn("events/grasp_transition", handle)
                self.assertIn("events/raw_grasp_transition_kind", handle)

    def test_strict_camera_export_rejects_never_captured_camera(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            stream = CameraH5Stream(path, ["front_camera"])
            with self.assertRaisesRegex(ValueError, "never_captured"):
                export_episode_h5(self._sampled_episode(), path, camera_stream=stream, strict_cameras=True)

    def test_camera_missing_frame_is_filled_and_marked(self) -> None:
        import h5py
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            stream = CameraH5Stream(path, ["front_camera"])
            stream.append(source_index=0, timestamp_s=0.0, frames={"front_camera": np.zeros((2, 2, 3), dtype=np.uint8)})
            export_episode_h5(self._sampled_episode(), path, camera_stream=stream, strict_cameras=True)
            with h5py.File(path, "r") as handle:
                camera = handle["cameras/front_camera"]
                self.assertEqual(camera["rgb"].shape, (2, 2, 2, 3))
                self.assertEqual(camera["valid_mask"][:].tolist(), [True, False])
                self.assertEqual(camera["filled_mask"][:].tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
