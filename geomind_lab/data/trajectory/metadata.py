"""Structured, JSON-serializable metadata for exported trajectories."""

from __future__ import annotations

from typing import Any


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def configured_action_scale(env_cfg: object) -> float | None:
    """Read the runtime arm-action scale without assuming a concrete cfg type."""
    actions = getattr(env_cfg, "actions", None)
    arm_action = getattr(actions, "arm_action", None)
    scale = getattr(arm_action, "scale", None)
    try:
        return float(scale) if scale is not None else None
    except (TypeError, ValueError):
        return None


def build_trajectory_metadata_contract(
    *,
    env_cfg: object,
    task_spec: dict[str, Any],
    sample_hz: float,
    initial_object_poses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return task facts required to interpret one sampled trajectory."""
    protocol = _mapping(task_spec.get("protocol"))
    return {
        "sample_frequency_hz": float(sample_hz),
        "action_scale": configured_action_scale(env_cfg),
        "object_info": initial_object_poses,
        "ground_truth": {
            "gt_spec": _mapping(protocol.get("gt_spec")),
            "goal_sequence": list(task_spec.get("goal_sequence", []) or []),
        },
    }