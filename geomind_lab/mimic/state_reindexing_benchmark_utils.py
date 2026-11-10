from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


POST_ACTION_RELOCATION_MODULE_NAME = "post_action_relocation"
STATE_REINDEXING_FAMILY = "state_reindexing"
HISTORICAL_IDENTITY_TRACKING_FAMILY = "historical_identity_tracking"
DISCRETE_SELECTION_TASK_TYPE = "discrete_selection"

STATE_TRACK_NAMES = frozenset(
    {
        "state_reindexing",
        "object_position_update",
        "reorder_with_distractors",
    }
)
HISTORICAL_TRACK_NAMES = frozenset(
    {
        "historical_identity_tracking",
        "object_identity_tracking",
    }
)

SCRIPT_TO_VARIANT = {
    "leftmost_to_right_of_rightmost": "edge_translation",
    "swap_adjacent_pairs_over_4": "adjacent_pair_swaps",
    "rightmost_to_left_of_leftmost": "edge_translation_reverse",
    "rightmost_to_between_first_and_second": "internal_insertion",
    "middle_to_left_of_leftmost": "single_step_tracking",
    "swap_first_and_second_over_2s": "single_step_tracking",
    "swap_middle_with_seeded_neighbor_over_2s": "single_step_tracking",
    "swap_middle_with_seeded_neighbor_middle_direct_over_2s": "single_step_tracking",
    "middle_to_left_then_rightmost_to_left": "historical_two_step_tracking",
    "leftmost_to_middle_then_rightmost_to_left_of_leftmost": "multi_step_shell_game",
}
SCRIPT_TO_INTERVENTION_TYPE = {
    "leftmost_to_right_of_rightmost": "edge_move",
    "swap_adjacent_pairs_over_4": "pair_swap",
    "rightmost_to_left_of_leftmost": "edge_move",
    "rightmost_to_between_first_and_second": "insertion",
    "middle_to_left_of_leftmost": "edge_move",
    "swap_first_and_second_over_2s": "timed_swap",
    "swap_middle_with_seeded_neighbor_over_2s": "timed_swap",
    "swap_middle_with_seeded_neighbor_middle_direct_over_2s": "timed_swap",
    "middle_to_left_then_rightmost_to_left": "multi_step_relocation",
    "leftmost_to_middle_then_rightmost_to_left_of_leftmost": "multi_step_relocation",
}
SCRIPT_TO_INTERVENTION_STEPS = {
    "leftmost_to_right_of_rightmost": 1,
    "swap_adjacent_pairs_over_4": 1,
    "rightmost_to_left_of_leftmost": 1,
    "rightmost_to_between_first_and_second": 1,
    "middle_to_left_of_leftmost": 1,
    "swap_first_and_second_over_2s": 1,
    "swap_middle_with_seeded_neighbor_over_2s": 1,
    "swap_middle_with_seeded_neighbor_middle_direct_over_2s": 1,
    "middle_to_left_then_rightmost_to_left": 2,
    "leftmost_to_middle_then_rightmost_to_left_of_leftmost": 4,
}


def _as_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _target_subset_ids(protocol: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(protocol, Mapping):
        return []
    subset_spec = protocol.get("target_subset_spec", {})
    if not isinstance(subset_spec, Mapping):
        return []
    object_ids = subset_spec.get("object_ids", [])
    if not isinstance(object_ids, Sequence) or isinstance(object_ids, (str, bytes)):
        return []
    return [str(object_id) for object_id in object_ids if isinstance(object_id, str) and object_id]


def protocol_track_name(protocol: Mapping[str, Any] | None) -> str:
    return _as_str(protocol.get("track_name")) if isinstance(protocol, Mapping) else ""


def protocol_family(protocol: Mapping[str, Any] | None) -> str:
    if not isinstance(protocol, Mapping):
        return ""
    family = _as_str(protocol.get("family"))
    if family:
        return family

    track_name = protocol_track_name(protocol)
    if track_name in STATE_TRACK_NAMES:
        return STATE_REINDEXING_FAMILY
    if track_name in HISTORICAL_TRACK_NAMES:
        return HISTORICAL_IDENTITY_TRACKING_FAMILY
    return track_name


def protocol_variant(protocol: Mapping[str, Any] | None) -> str:
    if not isinstance(protocol, Mapping):
        return ""
    variant = _as_str(protocol.get("variant"))
    if variant:
        return variant

    diagnostic_tags = protocol.get("diagnostic_tags", [])
    diagnostic_text = " ".join(str(tag) for tag in diagnostic_tags if isinstance(tag, str)).lower()
    subset_ids = _target_subset_ids(protocol)
    if subset_ids:
        if "semantic" in diagnostic_text:
            return "semantic_isolation"
        if protocol_track_name(protocol) == "reorder_with_distractors":
            return "clutter_filtering"

    script_name = ""
    intervention = protocol.get("intervention", {})
    if isinstance(intervention, Mapping):
        script_name = _as_str(intervention.get("script_name"))
    if script_name in SCRIPT_TO_VARIANT:
        return SCRIPT_TO_VARIANT[script_name]
    if subset_ids:
        return "clutter_filtering"
    return protocol_track_name(protocol)


def protocol_visual_mode(protocol: Mapping[str, Any] | None) -> str:
    if not isinstance(protocol, Mapping):
        return "identical_objects"
    visual_mode = _as_str(protocol.get("visual_mode"))
    return visual_mode if visual_mode else "identical_objects"


def protocol_intervention_type(protocol: Mapping[str, Any] | None) -> str:
    if not isinstance(protocol, Mapping):
        return ""
    intervention_type = _as_str(protocol.get("intervention_type"))
    if intervention_type:
        return intervention_type
    intervention = protocol.get("intervention", {})
    if isinstance(intervention, Mapping):
        script_name = _as_str(intervention.get("script_name"))
        return SCRIPT_TO_INTERVENTION_TYPE.get(script_name, _as_str(intervention.get("type")))
    return ""


def protocol_intervention_steps(protocol: Mapping[str, Any] | None) -> int | None:
    if not isinstance(protocol, Mapping):
        return None
    resolved = _as_int(protocol.get("intervention_steps"))
    if resolved is not None:
        return resolved
    intervention = protocol.get("intervention", {})
    if isinstance(intervention, Mapping):
        script_name = _as_str(intervention.get("script_name"))
        return SCRIPT_TO_INTERVENTION_STEPS.get(script_name)
    return None


def protocol_noise_level(protocol: Mapping[str, Any] | None) -> str:
    if not isinstance(protocol, Mapping):
        return ""
    noise_level = _as_str(protocol.get("noise_level"))
    return noise_level if noise_level else "clean"


def protocol_diagnostic_tags(protocol: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(protocol, Mapping):
        return []
    diagnostic_tags = protocol.get("diagnostic_tags", [])
    if not isinstance(diagnostic_tags, list):
        return []
    return [str(tag) for tag in diagnostic_tags if isinstance(tag, str) and str(tag).strip()]


def protocol_difficulty_factors(protocol: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(protocol, Mapping):
        return {}
    difficulty_factors = protocol.get("difficulty_factors", {})
    if isinstance(difficulty_factors, Mapping):
        return {str(key): value for key, value in difficulty_factors.items()}
    return {}


def protocol_object_count(
    protocol: Mapping[str, Any] | None,
    *,
    total_object_count: int | None = None,
) -> int | None:
    if not isinstance(protocol, Mapping):
        return total_object_count
    resolved = _as_int(protocol.get("object_count"))
    if resolved is not None:
        return resolved
    subset_count = len(_target_subset_ids(protocol))
    if subset_count > 0:
        return subset_count
    return total_object_count


def protocol_distractor_count(
    protocol: Mapping[str, Any] | None,
    *,
    total_object_count: int | None = None,
) -> int | None:
    if not isinstance(protocol, Mapping):
        return None
    resolved = _as_int(protocol.get("distractor_count"))
    if resolved is not None:
        return resolved

    target_count = protocol_object_count(protocol, total_object_count=total_object_count)
    if total_object_count is None or target_count is None:
        return None
    return max(int(total_object_count) - int(target_count), 0)


def build_state_refresh_summary_metadata(
    *,
    protocol: Mapping[str, Any] | None,
    total_object_count: int | None = None,
    task_name: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "family": protocol_family(protocol),
        "variant": protocol_variant(protocol),
        "task_name": task_name,
        "task_type": DISCRETE_SELECTION_TASK_TYPE,
        "visual_mode": protocol_visual_mode(protocol),
        "intervention_type": protocol_intervention_type(protocol),
        "noise_level": protocol_noise_level(protocol),
        "diagnostic_tags": protocol_diagnostic_tags(protocol),
        "difficulty_factors": protocol_difficulty_factors(protocol),
    }
    object_count = protocol_object_count(protocol, total_object_count=total_object_count)
    distractor_count = protocol_distractor_count(protocol, total_object_count=total_object_count)
    intervention_steps = protocol_intervention_steps(protocol)
    if object_count is not None:
        metadata["object_count"] = int(object_count)
    if distractor_count is not None:
        metadata["distractor_count"] = int(distractor_count)
    if intervention_steps is not None:
        metadata["intervention_steps"] = int(intervention_steps)
    return metadata


__all__ = [
    "DISCRETE_SELECTION_TASK_TYPE",
    "HISTORICAL_IDENTITY_TRACKING_FAMILY",
    "POST_ACTION_RELOCATION_MODULE_NAME",
    "STATE_REINDEXING_FAMILY",
    "build_state_refresh_summary_metadata",
    "protocol_distractor_count",
    "protocol_diagnostic_tags",
    "protocol_difficulty_factors",
    "protocol_family",
    "protocol_intervention_steps",
    "protocol_intervention_type",
    "protocol_noise_level",
    "protocol_object_count",
    "protocol_track_name",
    "protocol_variant",
    "protocol_visual_mode",
]
