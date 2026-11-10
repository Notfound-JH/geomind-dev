from __future__ import annotations

from typing import Any


def build_raw_observation(
    *,
    instruction: str,
    step: int,
    max_steps: int,
    state: dict[str, Any] | None = None,
    cameras: dict[str, Any] | None = None,
    action_info: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "instruction": str(instruction),
        "cameras": dict(cameras or {}),
        "timestep": {
            "episode_step": int(step),
            "max_episode_steps": int(max_steps),
        },
        "action": {
            "joint_action": (action_info or {}).get("joint_action", []),
            "ee_action": (action_info or {}).get("ee_action", []),
            "joint_velocity": (action_info or {}).get("joint_velocity", []),
            "extra": (action_info or {}).get("extra", {}),
        },
        "state": dict(state or {}),
        "meta": dict(meta or {}),
    }


def to_agent_ready_observation(raw_obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": raw_obs.get("instruction", ""),
        "state": raw_obs.get("state", {}),
        "cameras": raw_obs.get("cameras", {}),
        "episode_step": ((raw_obs.get("timestep") or {}).get("episode_step", 0)),
    }
