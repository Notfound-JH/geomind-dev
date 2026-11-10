"""Time-aligned trajectory resampling and relative end-effector deltas."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from typing import Any

from .segmenter import SEGMENTER_VERSION_STAGE_V1, SegmentationConfig, segment_ids, segment_records


def _left_indices(timestamps: Sequence[float], target_hz: float) -> list[int]:
    if not timestamps:
        return []
    interval = 1.0 / float(target_hz)
    final_time = float(timestamps[-1])
    targets = [index * interval for index in range(int(math.floor(final_time / interval)) + 1)]
    indices: list[int] = []
    source_index = 0
    for target in targets:
        while source_index + 1 < len(timestamps) and float(timestamps[source_index + 1]) <= target:
            source_index += 1
        indices.append(source_index)
    if indices[-1] != len(timestamps) - 1:
        indices.append(len(timestamps) - 1)
    return indices


def _quat_conjugate(quat: Sequence[float]) -> list[float]:
    return [-float(quat[0]), -float(quat[1]), -float(quat[2]), float(quat[3])]


def _quat_multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    lx, ly, lz, lw = (float(value) for value in left)
    rx, ry, rz, rw = (float(value) for value in right)
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _rotate_vector(quat: Sequence[float], vector: Sequence[float]) -> list[float]:
    rotated = _quat_multiply(_quat_multiply(quat, [*vector, 0.0]), _quat_conjugate(quat))
    return rotated[:3]


def compute_relative_eef_deltas(eef_actions: Sequence[Sequence[float]]) -> list[list[float]]:
    """Compute previous-EEF-frame translation and relative quaternion deltas."""
    result: list[list[float]] = []
    previous: list[float] | None = None
    for action in eef_actions:
        pose = [float(value) for value in action]
        if len(pose) != 7:
            raise ValueError(f"EEF action must be [x,y,z,qx,qy,qz,qw], got {len(pose)} values.")
        if previous is None:
            result.append([0.0] * 7)
        else:
            delta_world = [pose[index] - previous[index] for index in range(3)]
            delta_local = _rotate_vector(_quat_conjugate(previous[3:]), delta_world)
            delta_quat = _quat_multiply(_quat_conjugate(previous[3:]), pose[3:])
            result.append([*delta_local, *delta_quat])
        previous = pose
    return result


def _grasp_transition(previous: dict[str, Any], current: dict[str, Any]) -> int:
    was_grasped = bool(previous.get("object_grasped", False))
    is_grasped = bool(current.get("object_grasped", False))
    if is_grasped and not was_grasped:
        return 1
    if was_grasped and not is_grasped:
        return -1
    return 0


def _raw_grasp_events(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for source_index in range(1, len(records)):
        kind = _grasp_transition(records[source_index - 1], records[source_index])
        if kind == 0:
            continue
        events.append(
            {
                "source_index": int(source_index),
                "timestamp_s": float(records[source_index]["timestamp_s"]),
                "kind": int(kind),
            }
        )
    return events


def _project_events_to_samples(
    selected: list[dict[str, Any]],
    source_indices: Sequence[int],
    raw_events: Sequence[dict[str, Any]],
) -> tuple[list[int], dict[str, int]]:
    """Assign each raw event to retained 10Hz frames without overwrite.

    The primary target for each event is the first retained frame at or after
    its source index. If multiple events collide on one target frame, later
    events spill forward to the next available frames to preserve ordering.
    """
    projected = [0] * len(selected)
    collisions = 0
    spilled = 0
    dropped = 0
    next_min_target = 0
    for event in raw_events:
        base_target = bisect_left(source_indices, int(event["source_index"]))
        if base_target >= len(selected):
            base_target = len(selected) - 1
        kind = int(event["kind"])
        target_index = max(base_target, next_min_target)
        while target_index < len(selected) and projected[target_index] != 0:
            target_index += 1
        if target_index >= len(selected):
            dropped += 1
            continue
        if target_index != base_target:
            collisions += 1
            spilled += int(target_index > base_target)
        projected[target_index] = kind
        selected[target_index]["grasp_transition"] = kind
        next_min_target = target_index + 1
    return projected, {
        "raw_event_count": int(len(raw_events)),
        "projected_event_count": int(sum(1 for value in projected if int(value) != 0)),
        "collision_count": int(collisions),
        "spill_count": int(spilled),
        "dropped_event_count": int(dropped),
    }


def resample_episode(raw_episode: dict[str, Any], *, target_hz: float = 10.0, segment_source: str | None = None) -> dict[str, Any]:
    """Left-sample records onto a deterministic target clock and derive labels."""
    if target_hz <= 0.0:
        raise ValueError("target_hz must be positive.")
    records = list(raw_episode.get("records", []))
    if not records:
        raise ValueError("Cannot resample an episode with no recorded steps.")
    timestamps = [float(record["timestamp_s"]) for record in records]
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Raw trajectory timestamps must be monotonic.")

    requested_indices = [int(index) for index in raw_episode.get("sample_indices", [])]
    source_indices = requested_indices or _left_indices(timestamps, target_hz)
    if any(index < 0 or index >= len(records) for index in source_indices):
        raise ValueError("Sample indices are outside the raw trajectory range.")
    if any(right <= left for left, right in zip(source_indices, source_indices[1:])):
        raise ValueError("Sample indices must be strictly increasing.")
    selected = [dict(records[index]) for index in source_indices]
    raw_grasp_events = _raw_grasp_events(records)
    projected_grasp_transitions, projection_diagnostics = _project_events_to_samples(
        selected,
        source_indices,
        raw_grasp_events,
    )
    requested_segment_source = str(segment_source or raw_episode.get("metadata", {}).get("segment_source", "") or SEGMENTER_VERSION_STAGE_V1)
    segmentation = segment_records(selected, SegmentationConfig(source=requested_segment_source))
    atomic_ids = segmentation.atomic_ids
    segments = segmentation.segments
    eef_abs = [[float(value) for value in record["action_eef_abs"]] for record in selected]
    metadata = {**dict(raw_episode.get("metadata", {})), "sample_hz": float(target_hz)}
    metadata["segmenter_version"] = segmentation.version
    metadata["segment_source"] = segmentation.source
    metadata["segmenter_diagnostics"] = {
        **dict(segmentation.diagnostics),
        "event_projection": projection_diagnostics,
    }
    return {
        "metadata": metadata,
        "steps": {
            "index": list(range(len(selected))),
            "source_index": source_indices,
            "timestamp_s": [float(record["timestamp_s"]) for record in selected],
            "sim_step_index": [int(record["sim_step_index"]) for record in selected],
            "eef_abs": eef_abs,
            "eef_pose_world": [[float(value) for value in record["eef_pose_world"]] for record in selected],
            "eef_delta_rel": compute_relative_eef_deltas(eef_abs),
            "gripper_cmd": [[float(record["gripper_command"])] for record in selected],
            "joint_pos": [[float(value) for value in record["joint_pos"]] for record in selected],
            "joint_vel": [[float(value) for value in record["joint_vel"]] for record in selected],
            "gripper_state": [[float(value) for value in record["gripper_state"]] for record in selected],
            "object_grasped": [bool(record.get("object_grasped", False)) for record in selected],
            "grasp_transition": projected_grasp_transitions,
            "atomic_id": atomic_ids,
            "segment_id": segment_ids(atomic_ids),
        },
        "segments": [segment.__dict__ for segment in segments],
        "events": {
            "raw_grasp_transition_source_index": [int(event["source_index"]) for event in raw_grasp_events],
            "raw_grasp_transition_timestamp_s": [float(event["timestamp_s"]) for event in raw_grasp_events],
            "raw_grasp_transition_kind": [int(event["kind"]) for event in raw_grasp_events],
        },
    }
