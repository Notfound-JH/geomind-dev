"""Validation helpers for trajectory H5 episodes and dataset manifests.

These checks are read-only and intended for Phase2/Phase3 quality gates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .hdf5_exporter import _require_h5py


DEFAULT_REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "task_id",
    "variant",
    "layout_key",
    "scene_id",
    "task_index",
    "layout_index",
    "rollout_index",
    "trajectory_seed",
    "layout_jitter_seed",
    "layout_shift_seed",
    "layout_randomization_record",
    "initial_runtime_object_poses",
    "sample_frequency_hz",
    "sample_hz",
    "action_scale",
    "object_info",
    "ground_truth",
)


def _read_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON at {path}, got {type(payload).__name__}")
    return payload


def validate_episode_h5(
    episode_h5_path: str | Path,
    *,
    required_metadata_fields: tuple[str, ...] = DEFAULT_REQUIRED_METADATA_FIELDS,
) -> dict[str, Any]:
    """Validate one exported trajectory H5 file.

    Returns a report with ``passed`` and human-readable ``errors``.
    """
    path = Path(episode_h5_path).expanduser().resolve()
    report: dict[str, Any] = {
        "episode_h5_path": str(path),
        "exists": path.is_file(),
        "passed": False,
        "errors": [],
        "checks": {},
    }
    if not path.is_file():
        report["errors"].append("h5_file_missing")
        return report

    h5py = _require_h5py()
    required_datasets = (
        "steps/index",
        "steps/source_index",
        "steps/timestamp_s",
        "steps/sim_step_index",
        "actions/eef_abs",
        "actions/eef_delta_rel",
        "actions/gripper_cmd",
        "observations/eef_pose_world",
        "robot/joint_pos",
        "robot/joint_vel",
        "robot/gripper_state",
        "events/object_grasped",
        "events/grasp_transition",
        "events/raw_grasp_transition_source_index",
        "events/raw_grasp_transition_timestamp_s",
        "events/raw_grasp_transition_kind",
        "labels/atomic_id",
        "labels/segment_id",
        "labels/atomic_name_table",
        "segments/start_step",
        "segments/end_step",
        "segments/atomic_id",
        "segments/is_augmented",
        "segments/start_reason",
        "segments/end_reason",
    )

    with h5py.File(path, "r") as handle:
        missing_ds = [name for name in required_datasets if name not in handle]
        report["checks"]["required_datasets_present"] = len(missing_ds) == 0
        if missing_ds:
            report["errors"].append(f"missing_datasets:{','.join(missing_ds)}")

        metadata_group = handle.get("metadata")
        metadata_keys = set(metadata_group.keys()) if metadata_group is not None else set()
        missing_meta = [key for key in required_metadata_fields if key not in metadata_keys]
        report["checks"]["required_metadata_present"] = len(missing_meta) == 0
        if missing_meta:
            report["errors"].append(f"missing_metadata:{','.join(missing_meta)}")

        if "steps/index" in handle:
            T = int(handle["steps/index"].shape[0])
        else:
            T = -1
        report["checks"]["non_empty_steps"] = T > 0
        if T <= 0:
            report["errors"].append("empty_steps")

        step_first_dim_datasets = (
            "steps/source_index",
            "steps/timestamp_s",
            "steps/sim_step_index",
            "actions/eef_abs",
            "actions/eef_delta_rel",
            "actions/gripper_cmd",
            "observations/eef_pose_world",
            "robot/joint_pos",
            "robot/joint_vel",
            "robot/gripper_state",
            "events/object_grasped",
            "events/grasp_transition",
            "labels/atomic_id",
            "labels/segment_id",
        )
        shape_mismatch = []
        if T > 0:
            for name in step_first_dim_datasets:
                if name not in handle:
                    continue
                if int(handle[name].shape[0]) != T:
                    shape_mismatch.append(name)
        report["checks"]["step_axis_shapes_consistent"] = len(shape_mismatch) == 0
        if shape_mismatch:
            report["errors"].append(f"step_axis_shape_mismatch:{','.join(shape_mismatch)}")

        if "steps/timestamp_s" in handle and T > 1:
            timestamps = handle["steps/timestamp_s"][:]
            monotonic = bool((timestamps[1:] > timestamps[:-1]).all())
        else:
            monotonic = True
        report["checks"]["timestamps_strictly_increasing"] = monotonic
        if not monotonic:
            report["errors"].append("timestamps_not_strictly_increasing")

        if "steps/source_index" in handle and T > 1:
            source_indices = handle["steps/source_index"][:]
            source_monotonic = bool((source_indices[1:] > source_indices[:-1]).all())
        else:
            source_monotonic = True
        report["checks"]["source_index_strictly_increasing"] = source_monotonic
        if not source_monotonic:
            report["errors"].append("source_index_not_strictly_increasing")

        non_finite_datasets: list[str] = []
        for name in (
            "steps/timestamp_s",
            "actions/eef_abs",
            "actions/eef_delta_rel",
            "actions/gripper_cmd",
            "observations/eef_pose_world",
            "robot/joint_pos",
            "robot/joint_vel",
            "robot/gripper_state",
        ):
            if name not in handle:
                continue
            values = handle[name][:]
            if not all(math.isfinite(float(value)) for value in values.flat):
                non_finite_datasets.append(name)
        report["checks"]["numeric_datasets_finite"] = len(non_finite_datasets) == 0
        if non_finite_datasets:
            report["errors"].append(f"non_finite_values:{','.join(non_finite_datasets)}")

        raw_event_lengths = []
        for name in (
            "events/raw_grasp_transition_source_index",
            "events/raw_grasp_transition_timestamp_s",
            "events/raw_grasp_transition_kind",
        ):
            if name in handle:
                raw_event_lengths.append(int(handle[name].shape[0]))
        raw_event_shapes_consistent = len(set(raw_event_lengths)) <= 1
        report["checks"]["raw_event_shapes_consistent"] = raw_event_shapes_consistent
        if not raw_event_shapes_consistent:
            report["errors"].append("raw_grasp_event_shape_mismatch")

        cameras_group = handle.get("cameras")
        camera_checks_ok = True
        camera_errors: list[str] = []
        if cameras_group is not None and T > 0:
            for camera_name in cameras_group.keys():
                camera = cameras_group[camera_name]
                for field in ("rgb", "timestamp_s", "valid_mask", "filled_mask"):
                    if field not in camera:
                        camera_checks_ok = False
                        camera_errors.append(f"{camera_name}:{field}_missing")
                if "rgb" in camera and int(camera["rgb"].shape[0]) != T:
                    camera_checks_ok = False
                    camera_errors.append(f"{camera_name}:rgb_len_mismatch")
                if "timestamp_s" in camera and int(camera["timestamp_s"].shape[0]) != T:
                    camera_checks_ok = False
                    camera_errors.append(f"{camera_name}:timestamp_len_mismatch")
                if "valid_mask" in camera and int(camera["valid_mask"].shape[0]) != T:
                    camera_checks_ok = False
                    camera_errors.append(f"{camera_name}:valid_mask_len_mismatch")
                if "filled_mask" in camera and int(camera["filled_mask"].shape[0]) != T:
                    camera_checks_ok = False
                    camera_errors.append(f"{camera_name}:filled_mask_len_mismatch")
        report["checks"]["camera_shapes_consistent"] = camera_checks_ok
        if camera_errors:
            report["errors"].append(f"camera_errors:{';'.join(camera_errors)}")

    report["passed"] = all(bool(value) for value in report["checks"].values()) and not report["errors"]
    return report


def validate_dataset_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Validate episode-level manifest entries and linked H5 files."""
    path = Path(manifest_path).expanduser().resolve()
    report: dict[str, Any] = {
        "manifest_path": str(path),
        "exists": path.is_file(),
        "passed": False,
        "errors": [],
        "episode_reports": [],
        "summary": {},
    }
    if not path.is_file():
        report["errors"].append("manifest_missing")
        return report

    payload = _read_json_file(path)
    layouts = payload.get("layouts", [])
    if not isinstance(layouts, list):
        report["errors"].append("manifest_layouts_not_list")
        return report

    episode_entries: list[dict[str, Any]] = []
    for layout in layouts:
        if not isinstance(layout, dict):
            continue
        episodes = layout.get("episodes", [])
        if not isinstance(episodes, list):
            continue
        for episode in episodes:
            if isinstance(episode, dict):
                episode_entries.append(episode)

    if not episode_entries:
        report["errors"].append("no_episode_entries")
        return report

    pass_count = 0
    for entry in episode_entries:
        h5_path = str(entry.get("h5_path", "")).strip()
        if not h5_path:
            episode_report = {
                "episode_uid": str(entry.get("episode_uid", "")),
                "episode_h5_path": "",
                "passed": False,
                "errors": ["missing_h5_path"],
                "checks": {},
            }
        else:
            episode_report = validate_episode_h5(h5_path)
            episode_report["episode_uid"] = str(entry.get("episode_uid", ""))
        report["episode_reports"].append(episode_report)
        if bool(episode_report.get("passed", False)):
            pass_count += 1

    total = len(report["episode_reports"])
    report["summary"] = {
        "episode_count": int(total),
        "passed_count": int(pass_count),
        "failed_count": int(total - pass_count),
    }
    report["passed"] = pass_count == total and total > 0
    return report


def _metadata_value(handle, key: str) -> Any:
    value = handle["metadata"][key][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value.item() if hasattr(value, "item") else value


def validate_phase5_dataset_manifest(
    manifest_path: str | Path,
    *,
    min_rollouts_per_layout: int = 5,
) -> dict[str, Any]:
    """Validate the Phase 5 fixed-layout diversity contract from exported H5 files."""
    path = Path(manifest_path).expanduser().resolve()
    report: dict[str, Any] = {
        "manifest_path": str(path),
        "exists": path.is_file(),
        "passed": False,
        "errors": [],
        "layouts": [],
    }
    if not path.is_file():
        report["errors"].append("manifest_missing")
        return report

    payload = _read_json_file(path)
    layouts = payload.get("layouts", [])
    if not isinstance(layouts, list):
        report["errors"].append("manifest_layouts_not_list")
        return report

    h5py = _require_h5py()
    required_contact_order = (2, 3, 4)
    for layout in layouts:
        if not isinstance(layout, dict):
            continue
        layout_index = int(layout.get("layout_index", -1))
        episodes = layout.get("episodes", [])
        layout_report: dict[str, Any] = {
            "layout_index": layout_index,
            "episode_count": len(episodes) if isinstance(episodes, list) else 0,
            "errors": [],
            "checks": {},
        }
        if not isinstance(episodes, list):
            layout_report["errors"].append("episodes_not_list")
            report["layouts"].append(layout_report)
            continue

        initial_pose_signatures: set[str] = set()
        approach_signatures: set[tuple[tuple[float, float, float], ...]] = set()
        contact_order_ok = True
        augmentation_record_ok = True
        augmentation_executed_count = 0
        endpoint_errors: list[float] = []
        increment_signatures: set[tuple[str, ...]] = set()
        for episode in episodes:
            if not isinstance(episode, dict):
                layout_report["errors"].append("episode_not_object")
                continue
            h5_path = Path(str(episode.get("h5_path", ""))).expanduser()
            if not h5_path.is_file():
                layout_report["errors"].append(f"h5_missing:{h5_path}")
                continue
            with h5py.File(h5_path, "r") as handle:
                required = ("metadata/initial_runtime_object_poses", "metadata/augmentation_record", "actions/eef_abs", "labels/atomic_id")
                missing = [name for name in required if name not in handle]
                if missing:
                    layout_report["errors"].append(f"missing_fields:{','.join(missing)}")
                    continue
                initial_pose_signatures.add(json.dumps(_metadata_value(handle, "initial_runtime_object_poses"), sort_keys=True))
                augmentation_record = _metadata_value(handle, "augmentation_record")
                if not isinstance(augmentation_record, dict) or not bool(augmentation_record.get("requested", False)):
                    augmentation_record_ok = False
                elif bool(augmentation_record.get("augmentation_enabled", False)):
                    augmentation_executed_count += 1
                    endpoint_errors.append(float(augmentation_record.get("endpoint_error_m", float("inf"))))
                    signature = augmentation_record.get("axis_signature", [])
                    if isinstance(signature, list) and signature:
                        increment_signatures.add(tuple(str(value) for value in signature))
                atomic_ids = [int(value) for value in handle["labels/atomic_id"][:].tolist()]
                compact_ids = [action_id for index, action_id in enumerate(atomic_ids) if index == 0 or action_id != atomic_ids[index - 1]]
                cursor = 0
                for required_id in required_contact_order:
                    try:
                        cursor = compact_ids.index(required_id, cursor) + 1
                    except ValueError:
                        contact_order_ok = False
                        break
                approach_positions = handle["actions/eef_abs"][:]
                approach_signature = tuple(
                    (round(float(pose[0]), 5), round(float(pose[1]), 5), round(float(pose[2]), 5))
                    for pose, action_id in zip(approach_positions, atomic_ids)
                    if action_id == 1
                )
                approach_signatures.add(approach_signature)

        layout_report["checks"] = {
            "minimum_rollouts": len(episodes) >= int(min_rollouts_per_layout),
            "initial_object_poses_identical": len(initial_pose_signatures) == 1,
            "at_least_two_approach_paths": len(approach_signatures) >= 2,
            "at_least_three_approach_paths": len(approach_signatures) >= 3,
            "contact_order_preserved": contact_order_ok,
            "augmentation_records_present": augmentation_record_ok,
            "augmentation_executed_majority": augmentation_executed_count >= (len(episodes) + 1) // 2,
            "increment_endpoint_error_within_1mm": bool(endpoint_errors) and max(endpoint_errors) <= 0.001,
            "at_least_three_increment_signatures": len(increment_signatures) >= min(3, len(episodes)),
        }
        if not layout_report["checks"]["minimum_rollouts"]:
            layout_report["errors"].append("insufficient_rollouts")
        if not layout_report["checks"]["initial_object_poses_identical"]:
            layout_report["errors"].append("initial_object_poses_differ")
        if not layout_report["checks"]["at_least_two_approach_paths"]:
            layout_report["errors"].append("approach_paths_not_diverse")
        if not layout_report["checks"]["at_least_three_approach_paths"]:
            layout_report["errors"].append("fewer_than_three_approach_paths")
        if not layout_report["checks"]["contact_order_preserved"]:
            layout_report["errors"].append("contact_order_not_preserved")
        if not layout_report["checks"]["augmentation_records_present"]:
            layout_report["errors"].append("augmentation_record_missing_or_disabled")
        if not layout_report["checks"]["augmentation_executed_majority"]:
            layout_report["errors"].append("augmentation_not_executed_for_majority")
        if not layout_report["checks"]["increment_endpoint_error_within_1mm"]:
            layout_report["errors"].append("increment_endpoint_error_exceeds_1mm")
        if not layout_report["checks"]["at_least_three_increment_signatures"]:
            layout_report["errors"].append("increment_signatures_not_diverse")
        report["layouts"].append(layout_report)

    report["passed"] = bool(report["layouts"]) and all(
        not layout_report["errors"] and all(bool(value) for value in layout_report["checks"].values())
        for layout_report in report["layouts"]
    )
    return report
