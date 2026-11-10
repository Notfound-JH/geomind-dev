"""Atomic HDF5 trajectory export and streamed, auditable camera capture."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .segmenter import ATOMIC_ACTION_NAMES

SCHEMA_VERSION = "geomind_trajectory_v1"


def _require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "HDF5 export requires h5py. Install the optional dependency with `pip install -e '.[trajectory]'`."
        ) from exc
    return h5py


class CameraH5Stream:
    """Append 10Hz camera frames to a sidecar H5 without retaining RGB in RAM."""

    def __init__(self, output_path: str | Path, expected_camera_names: list[str]):
        self.h5py = _require_h5py()
        output = Path(output_path).expanduser().resolve()
        self.path = output.with_suffix(f"{output.suffix}.camera-stream.tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.expected_camera_names = tuple(sorted(set(str(name) for name in expected_camera_names)))
        self._handle = self.h5py.File(self.path, "w")
        self._handle.attrs["expected_camera_names"] = json.dumps(self.expected_camera_names)

    def append(self, *, source_index: int, timestamp_s: float, frames: dict[str, Any]) -> None:
        for name, frame in frames.items():
            if str(name) not in self.expected_camera_names:
                continue
            image = frame.detach().cpu().numpy() if hasattr(frame, "detach") else frame
            if getattr(image, "ndim", 0) != 3 or int(image.shape[-1]) != 3:
                continue
            group = self._handle.require_group(str(name))
            if "rgb" not in group:
                group.create_dataset(
                    "rgb",
                    shape=(0, *image.shape),
                    maxshape=(None, *image.shape),
                    chunks=(1, *image.shape),
                    dtype="u1",
                    compression="gzip",
                    compression_opts=4,
                )
                group.create_dataset("source_index", shape=(0,), maxshape=(None,), chunks=(128,), dtype="i4")
                group.create_dataset("timestamp_s", shape=(0,), maxshape=(None,), chunks=(128,), dtype="f8")
            row = int(group["rgb"].shape[0])
            group["rgb"].resize(row + 1, axis=0)
            group["source_index"].resize(row + 1, axis=0)
            group["timestamp_s"].resize(row + 1, axis=0)
            group["rgb"][row] = image
            group["source_index"][row] = int(source_index)
            group["timestamp_s"][row] = float(timestamp_s)
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _write_scalar(group, key: str, value: Any, string_dtype) -> None:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, str):
        group.create_dataset(key, data=value, dtype=string_dtype)
    elif value is not None:
        group.create_dataset(key, data=value)


def _write_camera_datasets(handle, stream: CameraH5Stream, steps: dict[str, Any], *, strict: bool, string_dtype) -> None:
    source_indices = [int(value) for value in steps["source_index"]]
    timestamps = [float(value) for value in steps["timestamp_s"]]
    cameras_group = handle.create_group("cameras")
    statuses: dict[str, str] = {}
    # HDF5 chunk dimensions must not exceed data dimensions for fixed-size datasets.
    # Use adaptive chunking so short episodes (e.g., <128 sampled steps) remain writable.
    sample_count = len(source_indices)
    mask_chunks = (min(128, sample_count),) if sample_count > 0 else None
    with stream.h5py.File(stream.path, "r") as source:
        for name in stream.expected_camera_names:
            if name not in source or "rgb" not in source[name]:
                statuses[name] = "never_captured"
                continue
            source_group = source[name]
            source_to_row = {int(value): row for row, value in enumerate(source_group["source_index"][:])}
            raw_rgb = source_group["rgb"]
            camera_group = cameras_group.create_group(name)
            rgb = camera_group.create_dataset(
                "rgb",
                shape=(len(source_indices), *raw_rgb.shape[1:]),
                chunks=(1, *raw_rgb.shape[1:]) if sample_count > 0 else None,
                dtype="u1",
                compression="gzip",
                compression_opts=4,
            )
            valid = camera_group.create_dataset(
                "valid_mask",
                shape=(sample_count,),
                dtype="?",
                chunks=mask_chunks,
            )
            filled = camera_group.create_dataset(
                "filled_mask",
                shape=(sample_count,),
                dtype="?",
                chunks=mask_chunks,
            )
            last_row: int | None = None
            missing_initial = False
            for output_index, source_index in enumerate(source_indices):
                row = source_to_row.get(source_index)
                valid[output_index] = row is not None
                if row is not None:
                    last_row = row
                if last_row is None:
                    missing_initial = True
                    continue
                rgb[output_index] = raw_rgb[last_row]
                filled[output_index] = row is None
            if missing_initial:
                statuses[name] = "missing_initial_frame"
                del cameras_group[name]
                continue
            statuses[name] = "complete"
            camera_group.create_dataset("timestamp_s", data=timestamps, dtype="f8")
    failures = {name: status for name, status in statuses.items() if status != "complete"}
    if failures and strict:
        raise ValueError(f"Camera capture incomplete: {json.dumps(failures, sort_keys=True)}")
    handle["metadata"].create_dataset("camera_status", data=json.dumps(statuses, sort_keys=True), dtype=string_dtype)


def _write_mp4_dataset(dataset, output_path: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export trajectory camera MP4 files.")
    height, width = int(dataset.shape[1]), int(dataset.shape[2])
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(float(fps)), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    for frame in dataset:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg camera export failed: {stderr[:800]}")


def _write_camera_videos(temp_path: Path, output_dir: Path, fps: float) -> None:
    h5py = _require_h5py()
    with h5py.File(temp_path, "r+") as handle:
        cameras = handle.get("cameras")
        if cameras is None:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for name in sorted(cameras.keys()):
            path = output_dir / f"{name}.mp4"
            _write_mp4_dataset(cameras[name]["rgb"], path, fps)
            cameras[name].create_dataset("video_path", data=str(path), dtype=string_dtype)


def export_episode_h5(
    sampled_episode: dict[str, Any],
    output_path: str | Path,
    *,
    camera_stream: CameraH5Stream | None = None,
    strict_cameras: bool = True,
    camera_video_dir: str | Path | None = None,
) -> Path:
    """Atomically write one sampled episode and optionally finalize streamed RGB."""
    h5py = _require_h5py()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(f"{output.suffix}.tmp")
    metadata = dict(sampled_episode.get("metadata", {}))
    steps = dict(sampled_episode.get("steps", {}))
    events = dict(sampled_episode.get("events", {}))
    required = {
        "index", "source_index", "timestamp_s", "sim_step_index", "eef_abs", "eef_pose_world", "eef_delta_rel",
        "gripper_cmd", "joint_pos", "joint_vel", "gripper_state", "object_grasped", "grasp_transition", "atomic_id", "segment_id",
    }
    missing = sorted(required - set(steps))
    if missing:
        raise ValueError(f"Sampled episode is missing required fields: {', '.join(missing)}")
    length = len(steps["index"])
    if any(len(steps[key]) != length for key in required):
        raise ValueError("All step datasets must have the same length.")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        if camera_stream is not None:
            camera_stream.close()
        with h5py.File(temp, "w") as handle:
            metadata_group = handle.create_group("metadata")
            metadata.setdefault("schema_version", SCHEMA_VERSION)
            for key, value in sorted(metadata.items()):
                _write_scalar(metadata_group, str(key), value, string_dtype)
            steps_group = handle.create_group("steps")
            for key, dtype in (("index", "i4"), ("source_index", "i4"), ("timestamp_s", "f8"), ("sim_step_index", "i4")):
                steps_group.create_dataset(key, data=steps[key], dtype=dtype)
            actions = handle.create_group("actions")
            actions.create_dataset("eef_abs", data=steps["eef_abs"], dtype="f4")
            actions.create_dataset("eef_delta_rel", data=steps["eef_delta_rel"], dtype="f4")
            actions.create_dataset("gripper_cmd", data=steps["gripper_cmd"], dtype="f4")
            observations = handle.create_group("observations")
            observations.create_dataset("eef_pose_world", data=steps["eef_pose_world"], dtype="f4")
            robot = handle.create_group("robot")
            for key in ("joint_pos", "joint_vel", "gripper_state"):
                robot.create_dataset(key, data=steps[key], dtype="f4")
            events_group = handle.create_group("events")
            events_group.create_dataset("object_grasped", data=steps["object_grasped"], dtype="?")
            events_group.create_dataset("grasp_transition", data=steps["grasp_transition"], dtype="i1")
            for key, dtype in (
                ("raw_grasp_transition_source_index", "i4"),
                ("raw_grasp_transition_timestamp_s", "f8"),
                ("raw_grasp_transition_kind", "i1"),
            ):
                events_group.create_dataset(key, data=events.get(key, []), dtype=dtype)
            labels = handle.create_group("labels")
            labels.create_dataset("atomic_id", data=steps["atomic_id"], dtype="i2")
            labels.create_dataset("segment_id", data=steps["segment_id"], dtype="i4")
            labels.create_dataset("atomic_name_table", data=list(ATOMIC_ACTION_NAMES), dtype=string_dtype)
            segments = list(sampled_episode.get("segments", []))
            segments_group = handle.create_group("segments")
            for key, dtype in (("start_step", "i4"), ("end_step", "i4"), ("atomic_id", "i2")):
                segments_group.create_dataset(key, data=[segment[key] for segment in segments], dtype=dtype)
            augmented_atomic_ids = {
                int(action_id)
                for action_id in metadata.get("augmented_atomic_ids", [])
                if isinstance(action_id, (int, float))
            }
            segments_group.create_dataset(
                "is_augmented",
                data=[int(int(segment["atomic_id"]) in augmented_atomic_ids) for segment in segments],
                dtype="u1",
            )
            segments_group.create_dataset(
                "start_reason",
                data=[str(segment.get("start_reason", "")) for segment in segments],
                dtype=string_dtype,
            )
            segments_group.create_dataset(
                "end_reason",
                data=[str(segment.get("end_reason", "")) for segment in segments],
                dtype=string_dtype,
            )
            if camera_stream is not None:
                _write_camera_datasets(handle, camera_stream, steps, strict=strict_cameras, string_dtype=string_dtype)
        if camera_stream is not None and camera_video_dir is not None:
            _write_camera_videos(temp, Path(camera_video_dir), float(metadata["sample_hz"]))
        os.replace(temp, output)
        return output
    finally:
        if temp.exists():
            temp.unlink()
        if camera_stream is not None and camera_stream.path.exists():
            camera_stream.path.unlink()
