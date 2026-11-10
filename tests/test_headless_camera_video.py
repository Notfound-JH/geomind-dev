from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local test environment
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if torch is not None:
    import geomind_lab.utils.headless_camera_video as camera_video
    from geomind_lab.utils.headless_camera_video import RolloutCameraRecorder
else:  # pragma: no cover - depends on local test environment
    camera_video = None
    RolloutCameraRecorder = None


class _FakeSensor:
    def __init__(self) -> None:
        self.frame_value = 0
        self.data = type("SensorData", (), {"output": {"rgb": None}})()

    def update(self, dt: float = 0.0, force_recompute: bool = False) -> None:
        del dt, force_recompute
        image = torch.full((1, 3, 4, 3), self.frame_value, dtype=torch.uint8)
        self.data.output["rgb"] = image


class _FakeScene:
    def __init__(self, sensor: _FakeSensor) -> None:
        self.sensors = {"front_camera": sensor, "scene_cam": _FakeSensor()}


class _FakeEnv:
    def __init__(self) -> None:
        self.sensor = _FakeSensor()
        self.scene = _FakeScene(self.sensor)
        self.step_count = 0

    def step(self, action=None):
        del action
        self.step_count += 1
        self.sensor.frame_value = self.step_count
        return None, None, False, False, {}


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class TestRolloutCameraRecorder(unittest.TestCase):
    def test_recorder_captures_full_step_sequence(self) -> None:
        env = _FakeEnv()
        recorder = RolloutCameraRecorder(env).install()
        recorder.capture_now()
        for _ in range(4):
            env.step(None)
        recorder.uninstall()

        frames = recorder.frame_buffers["front_camera"]
        values = [int(frame[0, 0, 0].item()) for frame in frames]
        self.assertEqual(values, [0, 1, 2, 3, 4])
        self.assertEqual(recorder.frame_count, 5)

    def test_recorder_honors_max_frame_cap(self) -> None:
        env = _FakeEnv()
        recorder = RolloutCameraRecorder(env, max_frames=3).install()
        recorder.capture_now()
        for _ in range(5):
            env.step(None)
        recorder.uninstall()

        frames = recorder.frame_buffers["front_camera"]
        values = [int(frame[0, 0, 0].item()) for frame in frames]
        self.assertEqual(values, [0, 1, 2])
        self.assertEqual(recorder.frame_count, 3)

    def test_save_mp4s_can_filter_camera_names(self) -> None:
        env = _FakeEnv()
        recorder = RolloutCameraRecorder(env).install()
        recorder.capture_now()
        recorder.uninstall()
        written: list[str] = []
        original_write = camera_video._write_mp4_clip
        try:
            camera_video._write_mp4_clip = lambda path, clip, fps: written.append(Path(path).name)  # type: ignore[assignment]
            paths = recorder.save_mp4s("/tmp/geomind-camera-test", fps=30.0, camera_names=("scene_cam",))
        finally:
            camera_video._write_mp4_clip = original_write  # type: ignore[assignment]
        self.assertEqual(written, ["scene_cam.mp4"])
        self.assertEqual([path.name for path in paths], ["scene_cam.mp4"])

    def test_recorder_matches_scene_camera_resolution(self) -> None:
        env = _FakeEnv()
        env.scene.sensors["scene_cam"].cfg = type("CameraCfg", (), {"height": 6, "width": 8})()
        recorder = camera_video.create_rollout_camera_recorder(env).install()
        recorder.capture_now()
        recorder.uninstall()
        self.assertEqual(tuple(recorder.frame_buffers["front_camera"][0].shape), (6, 8, 3))
        self.assertEqual(tuple(recorder.frame_buffers["scene_cam"][0].shape), (6, 8, 3))

    def test_stream_mode_does_not_retain_frames(self) -> None:
        env = _FakeEnv()
        output_dir = Path("/tmp/geomind-camera-stream-test")
        recorder = RolloutCameraRecorder(env, stream_video_dir=output_dir, stream_video_fps=30.0).install()
        original_popen = camera_video.subprocess.Popen

        class _Pipe:
            def __init__(self) -> None:
                self.stdin = self
                self.stderr = self
                self.writes = 0

            def write(self, payload: bytes) -> None:
                self.writes += len(payload)

            def close(self) -> None:
                pass

            def read(self) -> bytes:
                return b""

            def wait(self) -> int:
                return 0

        pipes: list[_Pipe] = []
        try:
            camera_video.subprocess.Popen = lambda *args, **kwargs: pipes.append(_Pipe()) or pipes[-1]  # type: ignore[assignment]
            recorder.capture_now()
            recorder.uninstall()
            paths = recorder.save_mp4s(output_dir)
        finally:
            camera_video.subprocess.Popen = original_popen  # type: ignore[assignment]
        self.assertEqual([path.name for path in paths], ["front_camera.mp4", "scene_cam.mp4"])
        self.assertEqual(recorder.frame_buffers, {})
        self.assertEqual(len(pipes), 2)


if __name__ == "__main__":
    unittest.main()
