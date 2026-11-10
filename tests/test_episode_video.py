from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from geomind.episodes import ContractError
from geomind.episodes.video import EpisodeVideo, validate_camera_pose, validate_recording


CONFIG = {"fps": 30, "width": 64, "height": 48, "eye_w_m": [1.6, 1.3, 1.75],
          "look_at_w_m": [0.4, 0, 1.05], "focal_length_mm": 18.0}
HAS_ENCODER = bool(importlib.util.find_spec("imageio") and importlib.util.find_spec("imageio_ffmpeg"))


def camera_pose(x=0.):
    return {"position_w_m": [x, .2, .7], "quaternion_wxyz_ros": [1., 0., 0., 0.],
            "parent_position_w_m": [x, .2, .6], "parent_quaternion_wxyz": [1., 0., 0., 0.],
            "mount_position_m": [0., 0., .1], "mount_quaternion_wxyz_ros": [1., 0., 0., 0.]}


class TestVideoContract(unittest.TestCase):
    def test_cfr_video_rate_is_exactly_aligned_to_physics(self):
        self.assertEqual(validate_recording(CONFIG, 1 / 120), 4)

    def test_invalid_parameters_do_not_start_rendering(self):
        for key, value in (("fps", 29), ("fps", True), ("width", 63), ("height", 0),
                           ("focal_length_mm", 0), ("eye_w_m", CONFIG["look_at_w_m"])):
            config = deepcopy(CONFIG)
            config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ContractError):
                validate_recording(config, 1 / 120)

    def test_unknown_camera_parameter_rejected(self):
        config = {**CONFIG, "unknown": 1}
        with self.assertRaisesRegex(ContractError, "Unknown fields"):
            validate_recording(config, 1 / 120)

    def test_camera_pose_is_validated_and_copied_without_aliasing(self):
        pose = camera_pose()
        clean = validate_camera_pose(pose)
        self.assertEqual(clean, pose)
        pose["position_w_m"][0] = 100.
        self.assertEqual(clean["position_w_m"][0], 0.)

    def test_invalid_pose_fields_vectors_and_quaternions_rejected(self):
        for key, value in (("position_w_m", [0., 0.]),
                           ("parent_position_w_m", [0., float("nan"), 0.]),
                           ("mount_position_m", [True, 0., 0.]),
                           ("quaternion_wxyz_ros", [0., 0., 0., 0.]),
                           ("parent_quaternion_wxyz", [2., 0., 0., 0.]),
                           ("mount_quaternion_wxyz_ros", [1., 0., float("inf"), 0.])):
            pose = camera_pose(); pose[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ContractError):
                validate_camera_pose(pose)
        for pose in ({}, {**camera_pose(), "unknown": 1}):
            with self.assertRaises(ContractError):
                validate_camera_pose(pose)


@unittest.skipUnless(HAS_ENCODER, "Optional imageio/ffmpeg video dependencies not installed")
class TestEncodedEpisodeVideo(unittest.TestCase):
    def record(self, directory, *, moving=False):
        recorder = EpisodeVideo(Path(directory) / "test", CONFIG, {"intrinsic_matrix": np.eye(3).tolist(),
                                "camera_name": "wrist", "parent_prim_path": "/World/Robot/panda_hand"},
                                start_tick=120, dt=1 / 120)
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.arange(64)[None, :] * 4
        rgb[:, :, 1] = 100
        for frame in range(5):
            recorder.append(rgb, tick=120 + 4 * frame, protocol_step=frame,
                            phase="delay" if frame in (1, 2) else "observe",
                            scene_visible=frame not in (1, 2), stage=1, sensor_frame=frame + 1,
                            camera_pose=camera_pose(frame * .01) if moving else None)
        return recorder.close(), rgb

    def test_real_mp4_frame_count_resolution_and_hidden_scene_mask(self):
        import imageio_ffmpeg
        def decode(path):
            result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path,
                                     "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                    capture_output=True, check=True)
            return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 48, 64, 3)
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.record(directory)
            observer, agent = decode(result["observer"]), decode(result["agent"])
            self.assertEqual(len(observer), 5)
            self.assertEqual(len(agent), 5)
            self.assertEqual(observer[0].shape, (48, 64, 3))
            self.assertGreater(float(observer[1].mean()), 20)
            self.assertLessEqual(int(agent[1].max()), 2)
            self.assertLessEqual(int(agent[2].max()), 2)
            self.assertGreater(float(agent[3].mean()), 20)
            self.assertEqual(result["agent_masked_frames"], 2)

    def test_frame_metadata_and_hashes_match_the_saved_files(self):
        import imageio.v2 as imageio
        with tempfile.TemporaryDirectory() as directory:
            result, rgb = self.record(directory)
            rows = [json.loads(line) for line in Path(result["frames"]).read_text().splitlines()]
            self.assertEqual([r["physics_tick"] for r in rows], [120, 124, 128, 132, 136])
            for index, row in enumerate(rows):
                self.assertAlmostEqual(row["episode_time_s"], index / 30)
                self.assertAlmostEqual(row["episode_time_s"], row["video_time_s"])
            for key in ("observer", "agent", "frames"):
                self.assertEqual(hashlib.sha256(Path(result[key]).read_bytes()).hexdigest(), result["sha256"][key])
            self.assertTrue(np.array_equal(imageio.imread(result["first_frame"]), rgb))
            self.assertFalse(result["dynamic_extrinsics"])
            self.assertFalse(result["camera_motion_observed"])
            self.assertNotIn("camera_mount", result)
            self.assertTrue(all("camera_pose" not in row for row in rows))

    def test_moving_camera_records_per_frame_extrinsics_and_fixed_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.record(directory, moving=True)
            rows = [json.loads(line) for line in Path(result["frames"]).read_text().splitlines()]
            self.assertEqual([row["camera_pose"]["position_w_m"][0] for row in rows], [0., .01, .02, .03, .04])
            self.assertEqual(result["extrinsics_frame_count"], result["frame_count"])
            self.assertTrue(result["dynamic_extrinsics"])
            self.assertTrue(result["camera_motion_observed"])
            self.assertEqual(result["camera_mount"], {"position_m": [0., 0., .1],
                "quaternion_wxyz_ros": [1., 0., 0., 0.], "camera_name": "wrist",
                "parent_prim_path": "/World/Robot/panda_hand"})
            self.assertEqual(result["agent_masked_frames"], 2)
            self.assertEqual(hashlib.sha256(Path(result["frames"]).read_bytes()).hexdigest(), result["sha256"]["frames"])

    def test_fixed_pose_does_not_claim_motion_and_incomplete_or_changed_mount_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeVideo(Path(directory) / "wrist", CONFIG, {}, start_tick=120, dt=1 / 120)
            rgb = np.zeros((48, 64, 3), dtype=np.uint8); rgb[:, :32, :] = 200
            args = dict(protocol_step=0, phase="observe", scene_visible=True, stage=1, sensor_frame=1)
            pose = camera_pose()
            recorder.append(rgb, tick=120, camera_pose=pose, **args)
            with self.assertRaisesRegex(ContractError, "every frame"):
                recorder.append(rgb, tick=124, **args)
            moved_mount = deepcopy(pose); moved_mount["mount_position_m"][0] = .01
            with self.assertRaisesRegex(ContractError, "local mount transform changed"):
                recorder.append(rgb, tick=124, camera_pose=moved_mount, **args)
            rotated_mount = deepcopy(pose); rotated_mount["mount_quaternion_wxyz_ros"] = [0., 1., 0., 0.]
            with self.assertRaisesRegex(ContractError, "local mount transform changed"):
                recorder.append(rgb, tick=124, camera_pose=rotated_mount, **args)
            # The quaternion sign may legitimately change without any rotation.
            opposite_sign = {key: ([-x for x in value] if "quaternion" in key else value)
                             for key, value in pose.items()}
            recorder.append(rgb, tick=124, camera_pose=opposite_sign, **args)
            result = recorder.close()
            self.assertEqual(result["frame_count"], 2)
            self.assertTrue(result["dynamic_extrinsics"])
            self.assertFalse(result["camera_motion_observed"])

    def test_pose_cannot_start_after_untracked_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeVideo(Path(directory) / "table", CONFIG, {}, start_tick=120, dt=1 / 120)
            rgb = np.zeros((48, 64, 3), dtype=np.uint8); rgb[:, :32, :] = 200
            args = dict(protocol_step=0, phase="observe", scene_visible=True, stage=1, sensor_frame=1)
            recorder.append(rgb, tick=120, **args)
            with self.assertRaisesRegex(ContractError, "every frame"):
                recorder.append(rgb, tick=124, camera_pose=camera_pose(), **args)
            recorder.append(rgb, tick=124, **args)
            self.assertFalse(recorder.close()["dynamic_extrinsics"])

    def test_missing_or_duplicate_physics_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeVideo(Path(directory) / "test", CONFIG, {}, start_tick=120, dt=1 / 120)
            rgb = np.zeros((48, 64, 3), dtype=np.uint8)
            rgb[:, :32, :] = 200
            arguments = dict(protocol_step=0, phase="observe", scene_visible=True, stage=1, sensor_frame=1)
            recorder.append(rgb, tick=120, **arguments)
            for tick in (120, 128):
                with self.subTest(tick=tick), self.assertRaisesRegex(ContractError, "physics clock"):
                    recorder.append(rgb, tick=tick, **arguments)
            recorder.append(rgb, tick=124, **arguments)
            recorder.close()


if __name__ == "__main__":
    unittest.main()
