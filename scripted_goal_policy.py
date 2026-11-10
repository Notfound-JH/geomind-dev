"""Scripted straight-to-goal control policy.

Used as a harness sanity control: it is injected through the exact same
``GEOMIND_MODEL_POLICY_HOOK`` / ``--policy model`` path as the remote VLA
clients, so it exercises the identical env.step / evaluator / max_steps
pipeline. If this policy cannot solve a task, the failure is in the
benchmark harness rather than in the policy under evaluation.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from geomind.runners.policy_runtime import _extract_delta_xy_from_native_obs


def _magnitude() -> float:
    try:
        value = float(os.environ.get("SCRIPTED_POLICY_MAGNITUDE", "1.0"))
    except ValueError:
        return 1.0
    return float(np.clip(value, 0.0, 1.0))


# 夹爪维恒为 -1.0（闭合），与 run_eval_scripted_ctl 的 24 条 trace 逐字一致。
# gripper_mode=hold_initial 的改写发生在 geomind_client 侧，而这个 hook 走
# policy_runtime 的 model_hook 直接注入、绕过 client，所以必须自己写死这一维，
# 否则新旧两批 scripted 数据的动作不同源（目标物体是 preloaded_in_gripper）。
GRIPPER_ACTION = -1.0


def policy(env: Any, raw_obs: dict[str, Any], step: int, trace: list[dict[str, Any]]) -> np.ndarray:
    action = np.zeros((7,), dtype=np.float32)
    action[6] = GRIPPER_ACTION
    delta_xy = _extract_delta_xy_from_native_obs(raw_obs)
    if delta_xy is None:
        return action
    delta = np.asarray(delta_xy[:2], dtype=np.float32)
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return action
    action[:2] = np.clip(delta / norm * _magnitude(), -1.0, 1.0)
    return action
