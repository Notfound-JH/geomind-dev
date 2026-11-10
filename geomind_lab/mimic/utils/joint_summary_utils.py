from __future__ import annotations

import json
from typing import Any

from geomind_lab.mimic.state_reindexing_benchmark_utils import DISCRETE_SELECTION_TASK_TYPE
from geomind_lab.mimic.utils.benchmark_manifest_utils import (
    EXPLICIT_RULE_EXTRAPOLATION_FAMILY,
    GEOMETRIC_STRUCTURE_TASK_FAMILIES,
    IMPLICIT_TREND_FITTING_FAMILY,
    STATE_REFRESH_TASK_FAMILIES,
)
from geomind_lab.mimic.utils.config_semantics_adapter import (
    augment_record_with_semantics,
    infer_task_type as infer_task_type_from_semantics,
    serialize_semantic_payload,
)


CONTINUOUS_PLACEMENT_TASK_TYPE = "continuous_placement"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _serialize_metadata(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def infer_task_type(record: dict[str, Any]) -> str:
    semantic_task_type = infer_task_type_from_semantics(record.get("family"))
    if semantic_task_type:
        return semantic_task_type
    family = str(record.get("family", "")).strip()
    if family in STATE_REFRESH_TASK_FAMILIES:
        return DISCRETE_SELECTION_TASK_TYPE
    if family in GEOMETRIC_STRUCTURE_TASK_FAMILIES:
        return CONTINUOUS_PLACEMENT_TASK_TYPE
    if any(key in record for key in ("intervention_success_rate", "accidental_displacement_rate")):
        return DISCRETE_SELECTION_TASK_TYPE
    if any(key in record for key in ("mean_l2_error", "mean_orthogonal_distance", "direction_correct_rate", "trend_consistent_rate")):
        return CONTINUOUS_PLACEMENT_TASK_TYPE
    return str(record.get("task_type") or "")


def normalize_joint_row(record: dict[str, Any]) -> dict[str, Any]:
    normalized = augment_record_with_semantics(record)
    episode_count = _safe_int(normalized.get("episode_count"))
    if episode_count is None:
        episode_count = _safe_int(normalized.get("num_episodes"))

    valid_rate = _safe_float(normalized.get("valid_rate"))
    if valid_rate is None:
        valid_rate = _safe_float(normalized.get("valid_episode_rate"))

    row = {
        "family": str(normalized.get("family", "")),
        "variant": str(normalized.get("variant", "")),
        "semantic_module": str(normalized.get("semantic_module") or ""),
        "task_id": str(normalized.get("task_id") or ""),
        "task_version": str(normalized.get("task_version") or ""),
        "skill_type": str(normalized.get("skill_type") or ""),
        "phase": str(normalized.get("phase") or ""),
        "parent_task_id": str(normalized.get("parent_task_id") or ""),
        "paired_control_id": str(normalized.get("paired_control_id") or ""),
        "evaluator_key": str(normalized.get("evaluator_key") or ""),
        "task_name": str(
            normalized.get("task_name")
            or normalized.get("task_label")
            or normalized.get("task_id")
            or normalized.get("layout_key")
            or "unknown_task"
        ),
        "raw_task_name": str(normalized.get("raw_task_name") or normalized.get("task_name") or ""),
        "canonical_task_id": str(normalized.get("canonical_task_id") or ""),
        "agent_name": str(normalized.get("agent_name") or normalized.get("agent_alias") or normalized.get("agent") or ""),
        "episode_count": episode_count,
        "valid_rate": valid_rate,
        "success_rate": _safe_float(normalized.get("success_rate")),
        "task_type": str(normalized.get("task_type") or infer_task_type(normalized)),
        "core_skill": str(normalized.get("core_skill") or ""),
        "query_semantics": str(normalized.get("query_semantics") or ""),
        "gt_mode": str(normalized.get("gt_mode") or ""),
        "canonical_role": str(normalized.get("canonical_role") or ""),
        "scene_name": str(normalized.get("scene_name") or ""),
        "scene_capability_tags": _serialize_metadata(normalized.get("scene_capability_tags")),
        "scene_risk_tags": _serialize_metadata(normalized.get("scene_risk_tags")),
        "instance_id": str(normalized.get("instance_id") or ""),
        "instance_layout": str(normalized.get("instance_layout") or ""),
        "instance_axis_variant": str(normalized.get("instance_axis_variant") or ""),
        "enabled_probes": _serialize_metadata(normalized.get("enabled_probes")),
        "stress_factors": _serialize_metadata(normalized.get("stress_factors")),
        "notes": str(normalized.get("notes") or ""),
    }

    for field_name in (
        "intervention_success_rate",
        "accidental_displacement_rate",
        "mean_l2_error",
        "mean_orthogonal_distance",
        "direction_correct_rate",
        "trend_consistent_rate",
    ):
        value = _safe_float(normalized.get(field_name))
        if value is not None:
            row[field_name] = value

    for field_name in ("diagnostic_tags", "difficulty_factors"):
        value = _serialize_metadata(normalized.get(field_name))
        if value is not None:
            row[field_name] = value
    for field_name in ("task_semantics", "scene_semantics", "instance_semantics"):
        value = serialize_semantic_payload(normalized.get(field_name))
        if value is not None:
            row[field_name] = value
    return row


__all__ = [
    "CONTINUOUS_PLACEMENT_TASK_TYPE",
    "EXPLICIT_RULE_EXTRAPOLATION_FAMILY",
    "IMPLICIT_TREND_FITTING_FAMILY",
    "infer_task_type",
    "normalize_joint_row",
]
