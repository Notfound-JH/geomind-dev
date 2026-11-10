"""Streaming RGB recordings with frame-to-physics alignment and visibility masks."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .resolver import _check, _fields, _integer, _number, _position


def validate_recording(config: dict, dt: float) -> int:
    _fields(config, {"fps", "width", "height", "eye_w_m", "look_at_w_m", "focal_length_mm"})
    _check(_number(config["focal_length_mm"]) and config["focal_length_mm"] > 0, "Camera focal length must be positive")
    _check(_integer(config["fps"], 1), "Video fps must be a positive integer")
    for key in ("width", "height"):
        _check(_integer(config[key], 2) and config[key] % 2 == 0, f"Video {key} must be positive and even")
    eye, look = _position(config["eye_w_m"]), _position(config["look_at_w_m"])
    _check(eye != look, "Camera eye and target must differ")
    _check(_number(dt) and dt > 0, "Invalid video physics dt")
    stride = 1 / (config["fps"] * dt)
    _check(stride >= 1 and abs(stride - round(stride)) < 1e-6, "Video fps must divide the physics rate exactly")
    return round(stride)


_CAMERA_POSE_FIELDS = {
    "position_w_m", "quaternion_wxyz_ros", "parent_position_w_m", "parent_quaternion_wxyz",
    "mount_position_m", "mount_quaternion_wxyz_ros",
}


def validate_camera_pose(pose: dict) -> dict:
    """Copy one measured optical pose and its parent/local mounting transforms.

    The camera quaternion uses ROS optical axes; the parent quaternion is the
    rigid body's ordinary world orientation. This records extrinsics without
    changing the sensor or claiming that a mounted camera necessarily moved.
    """
    _fields(pose, _CAMERA_POSE_FIELDS)
    clean = {}
    for key in ("position_w_m", "parent_position_w_m", "mount_position_m"):
        clean[key] = list(_position(pose[key]))
    for key in ("quaternion_wxyz_ros", "parent_quaternion_wxyz", "mount_quaternion_wxyz_ros"):
        value = pose[key]
        _check(isinstance(value, (list, tuple)) and len(value) == 4
               and all(_number(component) for component in value),
               f"Camera pose {key} must contain four finite numbers")
        norm = math.hypot(*value)
        _check(abs(norm - 1.0) <= 1e-4, f"Camera pose {key} must be a normalized quaternion")
        clean[key] = [float(component) for component in value]
    return clean


def _same_rotation(a, b, *, tolerance=1e-12):
    # q and -q represent the same orientation. Use normalized dot products so
    # ordinary float32 norm residuals do not appear as camera motion.
    cosine = abs(sum(x * y for x, y in zip(a, b))) / (math.hypot(*a) * math.hypot(*b))
    return cosine >= 1.0 - tolerance


class EpisodeVideo:
    def __init__(self, prefix: Path, config: dict, calibration: dict, *, start_tick: int, dt: float):
        import imageio.v2 as imageio
        self.imageio = imageio
        self.config = dict(config)
        self.calibration = calibration
        self.start_tick = start_tick
        self.dt = dt
        self.stride = validate_recording(config, dt)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.observer_path = prefix.with_name(prefix.name + "_observer.mp4")
        self.agent_path = prefix.with_name(prefix.name + "_agent.mp4")
        self.metadata_path = prefix.with_name(prefix.name + "_frames.jsonl")
        self.first_path = prefix.with_name(prefix.name + "_first.png")
        self.last_path = prefix.with_name(prefix.name + "_last.png")
        options = dict(fps=config["fps"], codec="libx264", quality=8, pixelformat="yuv420p",
                       macro_block_size=1, ffmpeg_log_level="error")
        self.observer = imageio.get_writer(str(self.observer_path), **options)
        self.agent = imageio.get_writer(str(self.agent_path), **options)
        self.metadata = self.metadata_path.open("w", encoding="utf-8")
        self.frame_count = 0
        self.masked_frames = 0
        self.last_tick = None
        self.first_rgb = None
        self.last_rgb = None
        self._pose_mode = None
        self._first_camera_pose = None
        self._camera_motion_observed = False

    def append(self, rgb, *, tick: int, protocol_step: int, phase: str, scene_visible: bool, stage: int,
               sensor_frame: int, camera_pose: dict | None = None):
        import numpy as np
        _check(rgb.shape == (self.config["height"], self.config["width"], 3) and rgb.dtype == np.uint8,
               "Unexpected RGB camera frame")
        _check(tick == self.start_tick + self.frame_count * self.stride, "Video frames do not match physics clock")
        pose = validate_camera_pose(camera_pose) if camera_pose is not None else None
        has_pose = pose is not None
        _check(self._pose_mode is None or has_pose == self._pose_mode,
               "Camera pose metadata must be present on every frame or absent on every frame")
        if pose is not None and self._first_camera_pose is not None:
            initial = self._first_camera_pose
            _check(math.dist(pose["mount_position_m"], initial["mount_position_m"]) <= 1e-9
                   and _same_rotation(pose["mount_quaternion_wxyz_ros"], initial["mount_quaternion_wxyz_ros"]),
                   "Camera local mount transform changed during recording")
        if self.first_rgb is None:
            _check(float(rgb.std()) > 1.0, "Camera returned a blank image")
            self.first_rgb = rgb.copy()
        if self._pose_mode is None:
            self._pose_mode = has_pose
            self._first_camera_pose = pose
        if pose is not None:
            initial = self._first_camera_pose
            self._camera_motion_observed |= (
                math.dist(pose["position_w_m"], initial["position_w_m"]) > 1e-7
                or not _same_rotation(pose["quaternion_wxyz_ros"], initial["quaternion_wxyz_ros"], tolerance=1e-10))
        self.last_rgb = rgb.copy()
        self.observer.append_data(rgb)
        self.agent.append_data(rgb if scene_visible else np.zeros_like(rgb))
        self.masked_frames += int(not scene_visible)
        row = {"frame_index": self.frame_count, "physics_tick": tick, "simulation_time_s": tick * self.dt,
               "episode_time_s": (tick - self.start_tick) * self.dt, "video_time_s": self.frame_count / self.config["fps"],
               "protocol_step": protocol_step, "phase": phase, "scene_visible_to_agent": scene_visible,
               "motion_stage": stage, "sensor_frame": sensor_frame}
        if pose is not None:
            row["camera_pose"] = pose
        self.metadata.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.last_tick = tick
        self.frame_count += 1

    def close(self) -> dict:
        self.observer.close()
        self.agent.close()
        self.metadata.close()
        _check(self.frame_count > 1, "Video contains fewer than two frames")
        self.imageio.imwrite(self.first_path, self.first_rgb)
        self.imageio.imwrite(self.last_path, self.last_rgb)
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
            ("observer", self.observer_path), ("agent", self.agent_path), ("frames", self.metadata_path))}
        result = {"observer": str(self.observer_path), "agent": str(self.agent_path), "frames": str(self.metadata_path),
                "first_frame": str(self.first_path), "last_frame": str(self.last_path),
                "frame_count": self.frame_count, "agent_masked_frames": self.masked_frames,
                "fps": self.config["fps"], "width": self.config["width"], "height": self.config["height"],
                "duration_s": self.frame_count / self.config["fps"], "start_physics_tick": self.start_tick,
                "last_physics_tick": self.last_tick, "camera": self.config, "calibration": self.calibration,
                "sha256": hashes, "observer_privileged": True,
                "dynamic_extrinsics": bool(self._pose_mode),
                "camera_motion_observed": self._camera_motion_observed}
        if self._first_camera_pose is not None:
            pose = self._first_camera_pose
            result["camera_mount"] = {
                "position_m": pose["mount_position_m"],
                "quaternion_wxyz_ros": pose["mount_quaternion_wxyz_ros"],
                **{key: self.calibration[key] for key in ("camera_name", "parent_prim_path") if key in self.calibration},
            }
            result["extrinsics_frame_count"] = self.frame_count
        return result
