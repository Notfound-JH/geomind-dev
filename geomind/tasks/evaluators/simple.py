from __future__ import annotations

import os
from math import inf
from typing import Any

from .base import BaseEvaluator

DEFAULT_SUCCESS_DWELL_STEPS = 5


def _resolve_dwell_steps(default: int) -> int:
    """Consecutive in-radius steps required for success (GEOMIND_SUCCESS_DWELL_STEPS)."""
    raw = str(os.environ.get("GEOMIND_SUCCESS_DWELL_STEPS", "")).strip()
    if raw:
        try:
            return max(int(raw), 1)
        except ValueError:
            pass
    return max(int(default), 1)


def _step_goal_distance(step: dict[str, Any]) -> float | None:
    raw = dict(step.get("meta", {})).get("goal_distance")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _step_goal_distance_l2(step: dict[str, Any]) -> float | None:
    """Plain euclidean goal distance in metres.

    Anisotropic goals rescale ``goal_distance`` so that ``<= success_radius`` matches the
    tolerance ellipse, which makes it a unitless-ish score rather than a length. SGE and the
    other physical-distance diagnostics must read this instead.
    """
    meta = dict(step.get("meta", {}))
    raw = meta.get("goal_distance_l2_m")
    if raw is None:
        raw = meta.get("goal_distance")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _step_within_radius(step: dict[str, Any]) -> bool:
    meta = dict(step.get("meta", {}))
    success_radius = float(meta.get("success_radius", 0.0) or 0.0)
    if success_radius <= 0.0:
        return False
    goal_distance = _step_goal_distance(step)
    if goal_distance is None:
        return False
    return goal_distance <= success_radius


def _first_dwell_step(episode_trace: list[dict[str, Any]], dwell_steps: int) -> int | None:
    """Index of the last step of the first run of `dwell_steps` consecutive in-radius steps."""
    streak = 0
    for index, step in enumerate(episode_trace):
        streak = streak + 1 if _step_within_radius(step) else 0
        if streak >= dwell_steps:
            return index
    return None


class RewardThresholdEvaluator(BaseEvaluator):
    key = "reward_threshold"

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)

    def evaluate(self, episode_trace: list[dict[str, Any]]) -> dict[str, Any]:
        reward_sum = float(sum(float(step.get("reward", 0.0)) for step in episode_trace))
        return {
            "success": reward_sum >= self.threshold,
            "score": reward_sum,
            "threshold": self.threshold,
            "trace_len": len(episode_trace),
            "implementation": "reward_threshold",
        }


class LayoutAwareEvaluator(BaseEvaluator):
    def __init__(
        self,
        key: str,
        *,
        expected_goal_kind: str = "",
        expected_resolvers: tuple[str, ...] = (),
        dwell_steps: int = DEFAULT_SUCCESS_DWELL_STEPS,
        bucket: str = "specialized",
    ) -> None:
        self.key = str(key).strip()
        self.expected_goal_kind = str(expected_goal_kind).strip().lower()
        self.expected_resolvers = tuple(str(item).strip().lower() for item in expected_resolvers if str(item).strip())
        self.dwell_steps = max(int(dwell_steps), 1)
        self.bucket = str(bucket).strip() or "specialized"

    def evaluate(self, episode_trace: list[dict[str, Any]]) -> dict[str, Any]:
        final_step = episode_trace[-1] if episode_trace else {}
        final_meta = dict(final_step.get("meta", {})) if episode_trace else {}
        raw_goal_distance = _step_goal_distance(final_step) if episode_trace else None
        goal_distance = raw_goal_distance if raw_goal_distance is not None else inf
        success_radius = float(final_meta.get("success_radius", 0.0) or 0.0)
        goal_kind = str(final_meta.get("goal_kind", "")).strip().lower()
        resolver = str(final_meta.get("protocol_resolver", "")).strip().lower()
        goal_kind_ok = (not self.expected_goal_kind) or goal_kind == self.expected_goal_kind
        resolver_ok = (not self.expected_resolvers) or resolver in self.expected_resolvers
        # "Metric unavailable" must never be reported as success: the delegate returns
        # goal_distance=None when the target/anchor pose could not be resolved.
        metric_valid = bool(final_meta.get("goal_metric_valid", raw_goal_distance is not None))
        # The runner re-evaluates the growing trace every step, so requiring the last N steps to be
        # inside the radius means the end effector has to stay there instead of merely passing through.
        dwell_steps = _resolve_dwell_steps(self.dwell_steps)
        window = episode_trace[-dwell_steps:]
        dwell_ok = len(window) >= dwell_steps and all(_step_within_radius(step) for step in window)
        success = bool(goal_kind_ok and resolver_ok and metric_valid and success_radius > 0.0 and dwell_ok)
        # Reach criterion: the gripper never releases the object under the current eval config
        # (gripper_mode=hold_initial), so "placed and settled" is unreachable by construction.
        # reach_success therefore asks whether the object was ever held inside the goal region for
        # dwell_steps consecutive steps, which still separates models without rewarding a flyby.
        reach_step = _first_dwell_step(episode_trace, dwell_steps)
        reach_success = bool(goal_kind_ok and resolver_ok and metric_valid and success_radius > 0.0 and reach_step is not None)
        distances = [d for d in (_step_goal_distance(step) for step in episode_trace) if d is not None]
        l2_distances = [d for d in (_step_goal_distance_l2(step) for step in episode_trace) if d is not None]
        final_l2 = _step_goal_distance_l2(final_step) if episode_trace else None
        score = float(success_radius - goal_distance) if goal_distance < inf else None
        return {
            "success": success,
            "reach_success": reach_success,
            "reach_step": reach_step,
            "min_goal_distance": min(distances) if distances else None,
            "score": score,
            "goal_distance": None if goal_distance == inf else goal_distance,
            "goal_distance_l2_m": final_l2,
            "min_goal_distance_l2_m": min(l2_distances) if l2_distances else None,
            "success_radius": success_radius,
            "goal_kind": goal_kind,
            "protocol_resolver": resolver,
            "trace_len": len(episode_trace),
            "dwell_steps_required": dwell_steps,
            "dwell_satisfied": dwell_ok,
            "metric_valid": metric_valid,
            "invalid_reasons": [] if metric_valid else list(final_meta.get("goal_metric_issues", []) or ["goal_distance_unavailable"]),
            "implementation": "temporary_layout_aware",
            "bucket": self.bucket,
        }


class BucketedLegacyEvaluator(BaseEvaluator):
    """Temporary bucket evaluator for legacy keys not implemented yet."""

    def __init__(self, key: str, *, bucket: str) -> None:
        self.key = str(key).strip()
        self.bucket = str(bucket).strip() or "legacy_placeholder"

    def evaluate(self, episode_trace: list[dict[str, Any]]) -> dict[str, Any]:
        final_meta = dict(episode_trace[-1].get("meta", {})) if episode_trace else {}
        reward_sum = float(sum(float(step.get("reward", 0.0)) for step in episode_trace))
        metric_valid = bool(final_meta.get("goal_metric_valid", False))
        return {
            "success": bool(final_meta.get("goal_achieved", False)),
            "score": reward_sum,
            "trace_len": len(episode_trace),
            "goal_kind": str(final_meta.get("goal_kind", "")).strip().lower(),
            "protocol_resolver": str(final_meta.get("protocol_resolver", "")).strip().lower(),
            # This bucket has no task-semantic criterion yet; `success` only mirrors the native
            # termination flag, so downstream aggregation must filter on metric_valid.
            "metric_valid": metric_valid,
            "invalid_reasons": []
            if metric_valid
            else list(final_meta.get("goal_metric_issues", []) or ["evaluator_not_implemented"]),
            "implementation": "temporary_bucketed_legacy",
            "bucket": self.bucket,
        }
