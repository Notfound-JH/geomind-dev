"""Small, framework-neutral runtime trajectory collector.

The rollout owns control and calls :meth:`EpisodeCollector.record_step` after
``env.step``. Keeping the collector passive prevents it from changing task
behavior when collection is disabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StepRecord:
    timestamp_s: float
    sim_step_index: int
    stage: str
    action_eef_abs: list[float]
    gripper_command: float
    eef_pose_world: list[float]
    joint_pos: list[float]
    joint_vel: list[float]
    gripper_state: list[float]
    object_grasped: bool = False
    terminated: bool = False
    truncated: bool = False


class EpisodeCollector:
    """Accumulates one raw episode and attaches immutable task metadata."""

    def __init__(self, *, metadata: dict[str, Any], sample_hz: float = 10.0):
        self.metadata = dict(metadata)
        if sample_hz <= 0.0:
            raise ValueError("sample_hz must be positive.")
        self.sample_hz = float(sample_hz)
        self._records: list[StepRecord] = []
        self._sample_indices: list[int] = []
        self._next_sample_time_s: float | None = None

    @property
    def records(self) -> tuple[StepRecord, ...]:
        return tuple(self._records)

    @property
    def sample_indices(self) -> tuple[int, ...]:
        return tuple(self._sample_indices)

    def is_sample_time(self, timestamp_s: float) -> bool:
        """Return whether this record belongs to the shared 10Hz data clock."""
        if self._next_sample_time_s is None:
            return True
        return float(timestamp_s) + 1e-9 >= self._next_sample_time_s

    def record_step(
        self,
        *,
        timestamp_s: float,
        sim_step_index: int,
        stage: str,
        action_eef_abs: list[float],
        gripper_command: float,
        eef_pose_world: list[float],
        joint_pos: list[float],
        joint_vel: list[float],
        gripper_state: list[float],
        object_grasped: bool = False,
        terminated: bool = False,
        truncated: bool = False,
    ) -> bool:
        frame_index = len(self._records)
        timestamp_s = float(timestamp_s)
        if self._records and timestamp_s < self._records[-1].timestamp_s:
            raise ValueError("Trajectory timestamps must be monotonic.")
        is_sample = self.is_sample_time(timestamp_s)
        if is_sample:
            self._sample_indices.append(frame_index)
            interval = 1.0 / self.sample_hz
            self._next_sample_time_s = timestamp_s + interval
        self._records.append(
            StepRecord(
                timestamp_s=timestamp_s,
                sim_step_index=int(sim_step_index),
                stage=str(stage),
                action_eef_abs=[float(value) for value in action_eef_abs],
                gripper_command=float(gripper_command),
                eef_pose_world=[float(value) for value in eef_pose_world],
                joint_pos=[float(value) for value in joint_pos],
                joint_vel=[float(value) for value in joint_vel],
                gripper_state=[float(value) for value in gripper_state],
                object_grasped=bool(object_grasped),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )
        return is_sample

    def as_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "records": [asdict(record) for record in self._records],
            "sample_indices": list(self._sample_indices),
        }
