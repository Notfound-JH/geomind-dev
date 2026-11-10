from __future__ import annotations

from geomind_lab.mimic.utils.task_metadata_registry import TASK_METADATA_REGISTRY

from .base import BaseEvaluator
from .simple import BucketedLegacyEvaluator, LayoutAwareEvaluator, RewardThresholdEvaluator

_EVALUATORS = {
    "always_success": BaseEvaluator(),
    "reward_threshold": RewardThresholdEvaluator(0.1),
    "anchor_reference_evaluator": LayoutAwareEvaluator(
        "anchor_reference_evaluator",
        expected_goal_kind="anchor_selection",
        expected_resolvers=("directional_anchor_selection", "nearest_anchor_selection"),
    ),
    "directional_placement_evaluator": LayoutAwareEvaluator(
        "directional_placement_evaluator",
        expected_goal_kind="placement",
        expected_resolvers=("directional_region_placement",),
    ),
    "move_to_object_evaluator": LayoutAwareEvaluator(
        "move_to_object_evaluator",
        expected_goal_kind="move_to",
        expected_resolvers=("move_to_object", "move_to_object_proximity", "object_proximity"),
    ),
    "identity_tracking_evaluator": LayoutAwareEvaluator(
        "identity_tracking_evaluator",
        expected_goal_kind="rank_selection",
        expected_resolvers=("initial_rank",),
    ),
    "square_completion_evaluator": LayoutAwareEvaluator(
        "square_completion_evaluator",
        expected_goal_kind="square_completion",
    ),
    "state_reindexing_evaluator": LayoutAwareEvaluator(
        "state_reindexing_evaluator",
        expected_goal_kind="rank_selection",
        expected_resolvers=("rank_after_intervention", "structural_adjacency", "rank_initial_layout"),
    ),
}

_LEGACY_EVALUATOR_BUCKETS = {
    "stack_evaluator": "spatial_stack_bucket",
    "containment_evaluator": "spatial_containment_bucket",
    "constraint_selection_evaluator": "constraint_selection_bucket",
    "geometric_inference_evaluator": "geometric_inference_bucket",
    "point_symmetry_evaluator": "geometric_point_symmetry_bucket",
    "trend_continuation_evaluator": "trend_continuation_bucket",
}


def _register_legacy_evaluators() -> None:
    for metadata in TASK_METADATA_REGISTRY.values():
        if not isinstance(metadata, dict):
            continue
        key = str(metadata.get("evaluator_key", "")).strip().lower()
        if not key or key in _EVALUATORS:
            continue
        bucket = _LEGACY_EVALUATOR_BUCKETS.get(key)
        if bucket is None:
            _EVALUATORS[key] = BucketedLegacyEvaluator(key, bucket="legacy_unbucketed_placeholder")
            continue
        # These families already carry a ground-truth target in annotation.goal_sequence (or a
        # placement resolver), so the delegate emits a real goal_distance/success_radius. Route them
        # through the distance evaluator instead of falling back to the raw termination flag.
        _EVALUATORS[key] = LayoutAwareEvaluator(key, bucket=bucket)


_register_legacy_evaluators()


def resolve_evaluator(key: str) -> BaseEvaluator:
    normalized = str(key).strip().lower()
    if normalized not in _EVALUATORS:
        raise KeyError(f"Unknown evaluator key: {key!r}")
    return _EVALUATORS[normalized]


def has_evaluator(key: str) -> bool:
    return str(key).strip().lower() in _EVALUATORS


def list_evaluator_keys() -> tuple[str, ...]:
    return tuple(sorted(_EVALUATORS.keys()))
