"""Stable atomic-action labels and deterministic trajectory segmentation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

SEGMENTER_VERSION_STAGE_V1 = "stage_v1"
SEGMENTER_VERSION_EVENT_KINEMATICS_V2 = "event_kinematics_v2"

ATOMIC_ACTION_NAMES: tuple[str, ...] = (
    "IDLE_HOLD",
    "REACH_ABOVE",
    "DESCEND_TO_GRASP",
    "GRIPPER_CLOSE",
    "LIFT_OBJECT",
    "TRANSIT_XY",
    "ALIGN_POSE",
    "DESCEND_TO_PLACE",
    "GRIPPER_OPEN",
    "RETREAT",
)

_IDLE_HOLD = 0
_REACH_ABOVE = 1
_DESCEND_TO_GRASP = 2
_GRIPPER_CLOSE = 3
_LIFT_OBJECT = 4
_TRANSIT_XY = 5
_ALIGN_POSE = 6
_DESCEND_TO_PLACE = 7
_GRIPPER_OPEN = 8
_RETREAT = 9

_STAGE_TO_ACTION_ID = {
    "approach": _REACH_ABOVE,
    "descend": _DESCEND_TO_GRASP,
    "close": _GRIPPER_CLOSE,
    "lift": _LIFT_OBJECT,
    "transit_xy": _TRANSIT_XY,
    "align_pose": _ALIGN_POSE,
    "descend_place": _DESCEND_TO_PLACE,
    "open": _GRIPPER_OPEN,
    "retreat": _RETREAT,
    "hold": _IDLE_HOLD,
    "idle": _IDLE_HOLD,
}


@dataclass(frozen=True)
class Segment:
    """Inclusive start/end indices for one contiguous atomic action."""

    start_step: int
    end_step: int
    atomic_id: int
    start_reason: str = ""
    end_reason: str = ""


@dataclass(frozen=True)
class SegmentationConfig:
    source: str = SEGMENTER_VERSION_STAGE_V1
    idle_speed_mps: float = 0.015
    z_speed_mps: float = 0.02
    xy_speed_mps: float = 0.02
    gripper_close_threshold: float = -0.2
    gripper_open_threshold: float = 0.2
    gripper_state_close_threshold: float = 0.015
    gripper_state_open_threshold: float = 0.03
    min_segment_steps: int = 2


@dataclass(frozen=True)
class SegmentationResult:
    atomic_ids: list[int]
    segments: list[Segment]
    source: str
    version: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def stage_to_atomic_id(stage_name: str | None) -> int:
    """Translate scripted stage names; unknown stages intentionally become hold."""
    return _STAGE_TO_ACTION_ID.get(str(stage_name or "").strip().lower(), _IDLE_HOLD)


def build_segments(atomic_ids: Sequence[int], transition_reasons: Sequence[str] | None = None) -> list[Segment]:
    """Return maximal contiguous runs; every input index belongs to exactly one run."""
    if not atomic_ids:
        return []
    segments: list[Segment] = []
    start = 0
    current = int(atomic_ids[0])
    start_reason = "episode_start"
    for index, action_id in enumerate(atomic_ids[1:], start=1):
        if int(action_id) == current:
            continue
        reason = "stage_change"
        if transition_reasons is not None and index - 1 < len(transition_reasons):
            reason = str(transition_reasons[index - 1] or reason)
        segments.append(
            Segment(
                start_step=start,
                end_step=index - 1,
                atomic_id=current,
                start_reason=start_reason,
                end_reason=reason,
            )
        )
        start, current = index, int(action_id)
        start_reason = reason
    segments.append(
        Segment(
            start_step=start,
            end_step=len(atomic_ids) - 1,
            atomic_id=current,
            start_reason=start_reason,
            end_reason="episode_end",
        )
    )
    return segments


def _record_stage(record: dict[str, Any]) -> str:
    return str(record.get("stage", "") or "").strip().lower()


def _record_position(record: dict[str, Any]) -> tuple[float, float, float] | None:
    for key in ("eef_pose_world", "action_eef_abs"):
        values = record.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) >= 3:
            try:
                return (float(values[0]), float(values[1]), float(values[2]))
            except (TypeError, ValueError):
                return None
    return None


def _record_gripper_command(record: dict[str, Any]) -> float:
    try:
        return float(record.get("gripper_command", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _record_gripper_state(record: dict[str, Any]) -> float | None:
    values = record.get("gripper_state")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) > 0:
        try:
            return float(values[0])
        except (TypeError, ValueError):
            return None
    if values is None:
        return None
    try:
        return float(values)
    except (TypeError, ValueError):
        return None


def _record_object_grasped(record: dict[str, Any]) -> bool:
    return bool(record.get("object_grasped", False))


def _record_grasp_transition(record: dict[str, Any]) -> int:
    try:
        transition = int(record.get("grasp_transition", 0))
    except (TypeError, ValueError):
        return 0
    return 1 if transition > 0 else (-1 if transition < 0 else 0)


def _kinematic_action_id(records: Sequence[dict[str, Any]], index: int, cfg: SegmentationConfig) -> int:
    if index <= 0:
        return _IDLE_HOLD
    current = _record_position(records[index])
    previous = _record_position(records[index - 1])
    if current is None or previous is None:
        return _IDLE_HOLD
    try:
        dt = float(records[index].get("timestamp_s", 0.0)) - float(records[index - 1].get("timestamp_s", 0.0))
    except (TypeError, ValueError):
        dt = 0.0
    if dt <= 1e-9:
        return _IDLE_HOLD
    dx = current[0] - previous[0]
    dy = current[1] - previous[1]
    dz = current[2] - previous[2]
    xy_speed = (dx * dx + dy * dy) ** 0.5 / dt
    z_speed = dz / dt
    if xy_speed < cfg.idle_speed_mps and abs(z_speed) < cfg.idle_speed_mps:
        return _IDLE_HOLD
    if z_speed >= cfg.z_speed_mps and xy_speed < cfg.xy_speed_mps:
        return _LIFT_OBJECT
    if z_speed <= -cfg.z_speed_mps and xy_speed < cfg.xy_speed_mps:
        stage = _record_stage(records[index])
        return _DESCEND_TO_PLACE if "place" in stage else _DESCEND_TO_GRASP
    if xy_speed >= cfg.xy_speed_mps:
        return _TRANSIT_XY
    return _ALIGN_POSE


def _event_kinematics_action_id(records: Sequence[dict[str, Any]], index: int, cfg: SegmentationConfig) -> int:
    stage = _record_stage(records[index])
    grasp_transition = _record_grasp_transition(records[index])
    if grasp_transition > 0:
        return _GRIPPER_CLOSE
    if grasp_transition < 0:
        return _GRIPPER_OPEN
    object_grasped = _record_object_grasped(records[index])
    previous_grasped = _record_object_grasped(records[index - 1]) if index > 0 else object_grasped
    if object_grasped and not previous_grasped:
        return _GRIPPER_CLOSE
    if not object_grasped and previous_grasped:
        return _GRIPPER_OPEN

    state_value = _record_gripper_state(records[index])
    previous_state = _record_gripper_state(records[index - 1]) if index > 0 else state_value
    if state_value is not None and previous_state is not None:
        if state_value <= cfg.gripper_state_close_threshold < previous_state:
            return _GRIPPER_CLOSE
        if state_value >= cfg.gripper_state_open_threshold > previous_state:
            return _GRIPPER_OPEN

    command = _record_gripper_command(records[index])
    previous_command = _record_gripper_command(records[index - 1]) if index > 0 else command
    if command <= cfg.gripper_close_threshold and previous_command > cfg.gripper_close_threshold:
        return _GRIPPER_CLOSE
    if command >= cfg.gripper_open_threshold and previous_command < cfg.gripper_open_threshold:
        return _GRIPPER_OPEN

    if stage in {"close", "open"}:
        return stage_to_atomic_id(stage)
    if stage == "approach":
        return _REACH_ABOVE
    if stage in {"hold", "idle"}:
        return _IDLE_HOLD

    kinematic = _kinematic_action_id(records, index, cfg)
    if kinematic != _IDLE_HOLD:
        return kinematic

    if stage in {"descend", "descend_place", "lift", "retreat", "transit_xy", "align_pose"}:
        return stage_to_atomic_id(stage)
    return _IDLE_HOLD


def _transition_reason(records: Sequence[dict[str, Any]], index: int, cfg: SegmentationConfig) -> str:
    grasp_transition = _record_grasp_transition(records[index])
    if grasp_transition > 0:
        return "grasp_event"
    if grasp_transition < 0:
        return "release_event"
    if _record_object_grasped(records[index]) != _record_object_grasped(records[index - 1]):
        return "grasp_event" if _record_object_grasped(records[index]) else "release_event"
    state_value = _record_gripper_state(records[index])
    previous_state = _record_gripper_state(records[index - 1])
    if state_value is not None and previous_state is not None:
        crossed_close_state = state_value <= cfg.gripper_state_close_threshold < previous_state
        crossed_open_state = state_value >= cfg.gripper_state_open_threshold > previous_state
        if crossed_close_state or crossed_open_state:
            return "gripper_state_change"
    command = _record_gripper_command(records[index])
    previous_command = _record_gripper_command(records[index - 1])
    crossed_close = command <= cfg.gripper_close_threshold < previous_command
    crossed_open = command >= cfg.gripper_open_threshold > previous_command
    if crossed_close or crossed_open:
        return "gripper_command_change"
    if _record_stage(records[index]) != _record_stage(records[index - 1]):
        return "stage_change"
    return "kinematic_class_change"


def _merge_short_semantic_fragments(
    atomic_ids: Sequence[int],
    transition_reasons: Sequence[str],
    *,
    min_segment_steps: int,
) -> tuple[list[int], int]:
    """Merge short noise runs only when equal semantic neighbors surround them.

    Gripper actions and event boundaries are never changed. This intentionally
    avoids guessing across semantically distinct phases.
    """
    merged = [int(action_id) for action_id in atomic_ids]
    if min_segment_steps <= 1:
        return merged, 0

    hard_action_ids = {_GRIPPER_CLOSE, _GRIPPER_OPEN}
    hard_reasons = {"grasp_event", "release_event"}
    merge_count = 0
    changed = True
    while changed:
        changed = False
        for segment in build_segments(merged, transition_reasons=transition_reasons):
            segment_length = int(segment.end_step - segment.start_step + 1)
            if segment_length >= min_segment_steps or segment.atomic_id in hard_action_ids:
                continue
            if segment.start_step == 0 or segment.end_step >= len(merged) - 1:
                continue
            left_id = merged[segment.start_step - 1]
            right_id = merged[segment.end_step + 1]
            left_reason = str(transition_reasons[segment.start_step - 1] or "")
            right_reason = str(transition_reasons[segment.end_step] or "")
            if left_id != right_id or left_reason in hard_reasons or right_reason in hard_reasons:
                continue
            for index in range(segment.start_step, segment.end_step + 1):
                merged[index] = left_id
            merge_count += 1
            changed = True
            break
    return merged, merge_count


def segment_records(records: Sequence[dict[str, Any]], config: SegmentationConfig | None = None) -> SegmentationResult:
    """Segment records using either stable stage labels or event+kinematic heuristics."""
    cfg = config or SegmentationConfig()
    normalized = [dict(record) for record in records]
    if cfg.source == SEGMENTER_VERSION_EVENT_KINEMATICS_V2:
        atomic_ids = [_event_kinematics_action_id(normalized, index, cfg) for index in range(len(normalized))]
        transition_reasons = [_transition_reason(normalized, index, cfg) for index in range(1, len(normalized))]
        atomic_ids, merged_fragment_count = _merge_short_semantic_fragments(
            atomic_ids,
            transition_reasons,
            min_segment_steps=max(1, int(cfg.min_segment_steps)),
        )
        version = SEGMENTER_VERSION_EVENT_KINEMATICS_V2
    else:
        atomic_ids = [stage_to_atomic_id(_record_stage(record)) for record in normalized]
        transition_reasons = ["stage_change" for _ in range(max(0, len(normalized) - 1))]
        merged_fragment_count = 0
        version = SEGMENTER_VERSION_STAGE_V1
    return SegmentationResult(
        atomic_ids=atomic_ids,
        segments=build_segments(atomic_ids, transition_reasons=transition_reasons),
        source=cfg.source,
        version=version,
        diagnostics={"record_count": len(normalized), "merged_fragment_count": merged_fragment_count},
    )


def segment_ids(atomic_ids: Iterable[int]) -> list[int]:
    """Return the contiguous segment index associated with each atomic action."""
    ids = [int(value) for value in atomic_ids]
    result: list[int] = []
    segment_id = -1
    previous: int | None = None
    for action_id in ids:
        if action_id != previous:
            segment_id += 1
            previous = action_id
        result.append(segment_id)
    return result
