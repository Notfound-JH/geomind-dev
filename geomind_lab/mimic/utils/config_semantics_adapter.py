from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any

from geomind_lab.mimic.utils.task_metadata_registry import (
    build_task_metadata,
    resolve_task_id as resolve_benchmark_task_id,
    resolve_task_version as resolve_benchmark_task_version,
)
from geomind_lab.mimic.state_reindexing_benchmark_utils import (
    DISCRETE_SELECTION_TASK_TYPE,
    HISTORICAL_IDENTITY_TRACKING_FAMILY,
    POST_ACTION_RELOCATION_MODULE_NAME,
    STATE_REINDEXING_FAMILY,
    build_state_refresh_summary_metadata,
)


GEOMETRIC_INFERENCE_MODULE_NAME = "geometric_inference"
ANCHOR_REFERENCE_MODULE_NAME = "anchor_reference"
EXPLICIT_RULE_EXTRAPOLATION_FAMILY = "explicit_rule_extrapolation"
IMPLICIT_TREND_FITTING_FAMILY = "implicit_trend_fitting"
SHAPE_COMPLETION_FAMILY = "shape_completion"
CONTINUOUS_PLACEMENT_TASK_TYPE = "continuous_placement"


CANONICAL_TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "state_reindexing_extreme_after_edge_shift": {
        "family": STATE_REINDEXING_FAMILY,
        "task_type": DISCRETE_SELECTION_TASK_TYPE,
        "core_skill": "post_action_relocation_rank_resolution",
        "query_semantics": "extreme_rank_selection_after_edge_shift",
        "gt_mode": "rank_after_intervention",
        "default_query_template": "Pick the current extreme object after the scripted edge shift.",
        "diagnostic_tags": ["edge_shift", "extreme_rank_resolution"],
        "difficulty_axes": ["object_count", "visual_mode", "order_axis", "intervention_steps"],
        "canonical_role": "task_prototype",
        "notes": "Covers edge relocation tasks that ask for the current leftmost/frontmost or analogous extreme.",
    },
    "state_reindexing_internal_insertion": {
        "family": STATE_REINDEXING_FAMILY,
        "task_type": DISCRETE_SELECTION_TASK_TYPE,
        "core_skill": "post_action_relocation_rank_resolution",
        "query_semantics": "internal_rank_selection_after_insertion",
        "gt_mode": "rank_after_intervention",
        "default_query_template": "Pick the object occupying the requested internal slot after insertion.",
        "diagnostic_tags": ["internal_insertion", "rank_update"],
        "difficulty_axes": ["object_count", "visual_mode", "order_axis"],
        "canonical_role": "task_prototype",
        "notes": "Covers insertion-style topology updates where a moved object changes the internal ordering.",
    },
    "ordinal_insert_static_layout": {
        "family": STATE_REINDEXING_FAMILY,
        "task_type": DISCRETE_SELECTION_TASK_TYPE,
        "core_skill": "static_ordinal_grounding",
        "query_semantics": "ordinal_rank_without_anchor_or_intervention",
        "gt_mode": "rank_initial_layout",
        "default_query_template": "Pick the k-th object from the left along a fixed tabletop axis in a static layout.",
        "diagnostic_tags": ["ordinal-selection", "no_anchor"],
        "difficulty_axes": ["object_count", "spacing", "visual_mode"],
        "canonical_role": "task_prototype",
        "notes": "Static 1D ordinal grounding; no scripted reorder, GT from initial sorted layout.",
    },
    "grid_rowcol_grounding_layout": {
        "family": STATE_REINDEXING_FAMILY,
        "task_type": DISCRETE_SELECTION_TASK_TYPE,
        "core_skill": "static_grid_rowcol_grounding",
        "query_semantics": "row_column_index_without_anchor",
        "gt_mode": "grid_rowcol_slot",
        "default_query_template": "Pick the object at the requested row and column of a tabletop grid.",
        "diagnostic_tags": ["grid-indexing", "no_anchor"],
        "difficulty_axes": ["grid_shape", "spacing", "visual_mode"],
        "canonical_role": "task_prototype",
        "notes": "Static 2D grid; rows use world x ascending (front-to-back), columns use y ascending (left-to-right).",
    },
        "state_reindexing_subset_middle_with_distractors": {
        "family": STATE_REINDEXING_FAMILY,
        "task_type": DISCRETE_SELECTION_TASK_TYPE,
        "core_skill": "post_action_relocation_subset_filtering",
        "query_semantics": "subset_rank_selection_with_distractors",
        "gt_mode": "rank_after_intervention",
        "default_query_template": "Within the designated subset, select the requested object after reordering.",
        "diagnostic_tags": ["subset_filtering", "distractor_filtering"],
        "difficulty_axes": ["object_count", "distractor_count", "visual_mode", "target_subset"],
        "canonical_role": "task_prototype",
        "notes": "Covers distractor-aware subset ranking tasks, including semantic or marked-target filtering variants.",
    },
    "historical_identity_tracking_single_step": {
        "family": HISTORICAL_IDENTITY_TRACKING_FAMILY,
        "task_type": DISCRETE_SELECTION_TASK_TYPE,
        "core_skill": "historical_identity_tracking",
        "query_semantics": "select_object_by_initial_order_identity",
        "gt_mode": "initial_rank",
        "default_query_template": "Pick the object that originally occupied the referenced rank.",
        "diagnostic_tags": ["identity_tracking", "initial_state_reasoning"],
        "difficulty_axes": ["object_count", "intervention_steps", "order_axis"],
        "canonical_role": "task_prototype",
        "notes": "Covers minimal historical-identity tasks where GT is resolved from the cached initial ordering.",
    },
    "historical_identity_tracking_multistep": {
        "family": HISTORICAL_IDENTITY_TRACKING_FAMILY,
        "task_type": DISCRETE_SELECTION_TASK_TYPE,
        "core_skill": "historical_identity_tracking",
        "query_semantics": "select_object_by_initial_order_identity_after_multistep_relocation",
        "gt_mode": "initial_rank",
        "default_query_template": "Track the queried object identity through multiple relocations and select it.",
        "diagnostic_tags": ["identity_tracking", "multi_step_temporal_tracking"],
        "difficulty_axes": ["object_count", "intervention_steps", "order_axis", "temporal_depth"],
        "canonical_role": "task_prototype",
        "notes": "Covers higher-depth shell-game style identity tracking tasks.",
    },
    "perfect_midpoint": {
        "family": EXPLICIT_RULE_EXTRAPOLATION_FAMILY,
        "task_type": CONTINUOUS_PLACEMENT_TASK_TYPE,
        "core_skill": "explicit_geometric_reasoning",
        "query_semantics": "place_at_midpoint_between_anchors",
        "gt_mode": "analytic_midpoint",
        "default_query_template": "Place the target exactly at the midpoint between the anchors.",
        "diagnostic_tags": ["center_interpolation", "explicit_rule"],
        "difficulty_axes": ["spacing", "anchor_count", "layout"],
        "canonical_role": "task_prototype",
        "notes": "Covers midpoint placement tasks even when historical labels use legacy midpoint naming.",
    },
    "isometric_extension": {
        "family": EXPLICIT_RULE_EXTRAPOLATION_FAMILY,
        "task_type": CONTINUOUS_PLACEMENT_TASK_TYPE,
        "core_skill": "explicit_geometric_reasoning",
        "query_semantics": "place_by_isometric_extension",
        "gt_mode": "analytic_extension",
        "default_query_template": "Extend the anchor pattern while preserving direction and spacing.",
        "diagnostic_tags": ["directional_extrapolation", "explicit_rule"],
        "difficulty_axes": ["spacing", "anchor_count", "layout", "axis_variant"],
        "canonical_role": "task_prototype",
        "notes": "Covers extension variants including primary, vertical, and grid-carried instances.",
    },
    "point_symmetry": {
        "family": EXPLICIT_RULE_EXTRAPOLATION_FAMILY,
        "task_type": CONTINUOUS_PLACEMENT_TASK_TYPE,
        "core_skill": "explicit_geometric_reasoning",
        "query_semantics": "place_by_point_symmetry_about_center",
        "gt_mode": "analytic_point_symmetry",
        "default_query_template": "Place the target at the point-symmetric location about the red center block.",
        "diagnostic_tags": ["center_reflection", "explicit_rule"],
        "difficulty_axes": ["spacing", "anchor_count", "layout", "obstacle_proximity"],
        "canonical_role": "task_prototype",
        "notes": "Covers explicit point-symmetry placement tasks using a red center anchor and a blue reference anchor.",
    },
    "trend_continuation": {
        "family": IMPLICIT_TREND_FITTING_FAMILY,
        "task_type": CONTINUOUS_PLACEMENT_TASK_TYPE,
        "core_skill": "implicit_geometric_reasoning",
        "query_semantics": "continue_noisy_geometric_trend",
        "gt_mode": "pca_trend_continuation",
        "default_query_template": "Continue the geometric trend to the next implied location.",
        "diagnostic_tags": ["global_trend_extraction", "implicit_trend"],
        "difficulty_axes": ["anchor_count", "noise_level", "spacing", "layout"],
        "canonical_role": "task_prototype",
        "notes": "Covers the main implicit trend-continuation task family.",
    },
    "weak_destruction_control": {
        "family": IMPLICIT_TREND_FITTING_FAMILY,
        "task_type": CONTINUOUS_PLACEMENT_TASK_TYPE,
        "core_skill": "implicit_geometric_reasoning",
        "query_semantics": "continue_trend_under_structure_damage",
        "gt_mode": "pca_trend_continuation",
        "default_query_template": "Continue the latent trend despite weak structural corruption.",
        "diagnostic_tags": ["global_trend_extraction", "structure_destroyed_control"],
        "difficulty_axes": ["anchor_count", "noise_level", "spacing", "layout", "destruction_level"],
        "canonical_role": "task_prototype",
        "notes": "Control task for stress-testing trend continuation under degraded structure.",
    },
    "square_completion": {
        "family": SHAPE_COMPLETION_FAMILY,
        "task_type": CONTINUOUS_PLACEMENT_TASK_TYPE,
        "core_skill": "shape_completion_reasoning",
        "query_semantics": "complete_missing_square_corner_with_optional_inward_yaw",
        "gt_mode": "analytic_square_completion",
        "default_query_template": "Complete the missing square corner from the three anchor blocks.",
        "diagnostic_tags": ["shape_completion", "square_corner_inference", "orientation_upgrade"],
        "difficulty_axes": ["anchor_spacing", "obstacle_proximity", "yaw_tolerance", "layout"],
        "canonical_role": "task_prototype",
        "notes": "Covers square-corner completion tasks, including the inward-yaw upgrade variants.",
    },
}

RAW_TASK_ID_TO_CANONICAL = {
    "state_reindexing_current_leftmost": "state_reindexing_extreme_after_edge_shift",
    "state_reindexing_current_rightmost": "state_reindexing_extreme_after_edge_shift",
    "state_reindexing_current_leftmost_after_rightmost_shift": "state_reindexing_extreme_after_edge_shift",
    "state_reindexing_current_leftmost_blocks": "state_reindexing_extreme_after_edge_shift",
    "state_reindexing_current_leftmost_blocks_stress": "state_reindexing_extreme_after_edge_shift",
    "object_position_update_edge_frontmost": "state_reindexing_extreme_after_edge_shift",
    "state_reindexing_insert_second_blocks": "state_reindexing_internal_insertion",
    "state_reindexing_insert_second_blocks_stress": "state_reindexing_internal_insertion",
    "object_position_update_insert_second": "state_reindexing_internal_insertion",
    "ordinal_insert_static_blocks_base": "ordinal_insert_static_layout",
    "ordinal_insert_static_blocks_stress": "ordinal_insert_static_layout",
    "ordinal_insert_static_blocks_skill": "ordinal_insert_static_layout",
    "grid_rowcol_grounding_grid_2x3_base": "grid_rowcol_grounding_layout",
    "grid_rowcol_grounding_grid_2x3_stress": "grid_rowcol_grounding_layout",
    "grid_rowcol_grounding_grid_3x3_skill": "grid_rowcol_grounding_layout",
    "reorder_with_distractors_red_subset_middle_blocks": "state_reindexing_subset_middle_with_distractors",
    "reorder_with_distractors_red_subset_middle_blocks_stress": "state_reindexing_subset_middle_with_distractors",
    "reorder_with_distractors_filtered_frontmost": "state_reindexing_subset_middle_with_distractors",
    "reorder_with_distractors_middle_of_subset": "state_reindexing_subset_middle_with_distractors",
    "historical_identity_tracking_original_middle": "historical_identity_tracking_single_step",
    "object_identity_tracking_original_middle_single_step": "historical_identity_tracking_single_step",
    "object_identity_tracking_original_frontmost_multistep": "historical_identity_tracking_multistep",
    "geometric_inference_perfect_midpoint_primary": "perfect_midpoint",
    "geometric_inference_midpoint_primary": "perfect_midpoint",
    "geometric_inference_isometric_extension_primary": "isometric_extension",
    "geometric_inference_extension_primary": "isometric_extension",
    "geometric_inference_extension_vertical_primary": "isometric_extension",
    "geometric_inference_extension_grid_top_row": "isometric_extension",
    "geometric_inference_extension_grid_left_column": "isometric_extension",
    "geometric_inference_point_symmetry_primary": "point_symmetry",
    "geometric_inference_point_symmetry_skill": "point_symmetry",
    "geometric_inference_point_symmetry_stress": "point_symmetry",
    "geometric_inference_implicit_trend_primary": "trend_continuation",
    "geometric_inference_implicit_trend_stress": "trend_continuation",
    "geometric_inference_implicit_trend_weak_destruction": "weak_destruction_control",
    "geometric_inference_implicit_trend_weak_destruction_stress": "weak_destruction_control",
    "geometric_inference_square_completion_primary": "square_completion",
    "geometric_inference_square_completion_skill": "square_completion",
    "geometric_inference_square_completion_stress": "square_completion",
}

FAMILY_VARIANT_TO_CANONICAL = {
    (STATE_REINDEXING_FAMILY, "edge_translation"): "state_reindexing_extreme_after_edge_shift",
    (STATE_REINDEXING_FAMILY, "edge_translation_reverse"): "state_reindexing_extreme_after_edge_shift",
    (STATE_REINDEXING_FAMILY, "internal_insertion"): "state_reindexing_internal_insertion",
    (STATE_REINDEXING_FAMILY, "clutter_filtering"): "state_reindexing_subset_middle_with_distractors",
    (STATE_REINDEXING_FAMILY, "semantic_isolation"): "state_reindexing_subset_middle_with_distractors",
    (HISTORICAL_IDENTITY_TRACKING_FAMILY, "single_step_tracking"): "historical_identity_tracking_single_step",
    (HISTORICAL_IDENTITY_TRACKING_FAMILY, "historical_two_step_tracking"): "historical_identity_tracking_single_step",
    (HISTORICAL_IDENTITY_TRACKING_FAMILY, "multi_step_shell_game"): "historical_identity_tracking_multistep",
    (EXPLICIT_RULE_EXTRAPOLATION_FAMILY, "perfect_midpoint"): "perfect_midpoint",
    (EXPLICIT_RULE_EXTRAPOLATION_FAMILY, "midpoint"): "perfect_midpoint",
    (EXPLICIT_RULE_EXTRAPOLATION_FAMILY, "isometric_extension"): "isometric_extension",
    (EXPLICIT_RULE_EXTRAPOLATION_FAMILY, "point_symmetry"): "point_symmetry",
    (IMPLICIT_TREND_FITTING_FAMILY, "trend_continuation"): "trend_continuation",
    (IMPLICIT_TREND_FITTING_FAMILY, "weak_destruction_control"): "weak_destruction_control",
    (SHAPE_COMPLETION_FAMILY, "square_completion"): "square_completion",
}

CANONICAL_SCENE_CAPABILITY_TAGS = {
    "state_reindexing_extreme_after_edge_shift": ("linear_friendly",),
    "state_reindexing_internal_insertion": ("linear_friendly", "insertion_friendly"),
    "ordinal_insert_static_layout": ("linear_friendly", "ordinal_static_friendly"),
    "grid_rowcol_grounding_layout": ("linear_friendly", "grid_grounding_friendly"),
    "state_reindexing_subset_middle_with_distractors": ("linear_friendly", "subset_filtering_friendly"),
    "historical_identity_tracking_single_step": ("identity_tracking_friendly", "linear_friendly"),
    "historical_identity_tracking_multistep": ("identity_tracking_friendly",),
    "perfect_midpoint": ("geometry_friendly",),
    "isometric_extension": ("geometry_friendly", "extension_friendly"),
    "point_symmetry": ("geometry_friendly", "symmetry_friendly"),
    "trend_continuation": ("geometry_friendly", "trend_friendly"),
    "weak_destruction_control": ("geometry_friendly", "trend_friendly"),
    "square_completion": ("geometry_friendly", "shape_completion_friendly"),
}

SCENE_METADATA_OVERRIDES = {
    1: {"scene_name": "scene_1", "capability_tags": ["linear_friendly", "geometry_friendly"]},
    4: {"scene_name": "scene_4", "capability_tags": ["linear_friendly", "geometry_friendly", "trend_friendly"]},
    5: {"scene_name": "scene_5", "capability_tags": ["geometry_friendly", "extension_friendly"]},
    10: {"scene_name": "scene_10", "capability_tags": ["linear_friendly", "identity_tracking_friendly"]},
    17: {"scene_name": "scene_17", "capability_tags": ["linear_friendly", "insertion_friendly", "identity_tracking_friendly"]},
    21: {"scene_name": "scene_21", "capability_tags": ["subset_filtering_friendly", "linear_friendly"]},
    23: {"scene_name": "scene_23", "capability_tags": ["identity_tracking_friendly"]},
    26: {"scene_name": "scene_26", "capability_tags": ["subset_filtering_friendly"]},
}


def _as_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_slug(value: Any) -> str:
    text = _as_str(value).lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _ensure_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _ensure_jsonable_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): inner for key, inner in value.items()}
    return {}


def _protocol_gt_mode(protocol: Mapping[str, Any] | None) -> str:
    if not isinstance(protocol, Mapping):
        return ""
    gt_spec = protocol.get("gt_spec", {})
    if not isinstance(gt_spec, Mapping):
        gt_spec = {}
    resolver = _as_str(gt_spec.get("resolver"))
    if resolver:
        return resolver
    rule_type = _as_str(gt_spec.get("rule_type"))
    if rule_type:
        return rule_type
    return ""


def _state_refresh_layout_object_count(scene_structure: Mapping[str, Any] | None) -> int | None:
    if not isinstance(scene_structure, Mapping):
        return None
    object_positions = scene_structure.get("object_positions", {})
    if isinstance(object_positions, Mapping):
        return len(object_positions)
    return _as_int(scene_structure.get("num_of_objects"))


def _resolve_family_variant(
    *,
    protocol: Mapping[str, Any] | None,
    raw_task_id: str,
    family: str,
    variant: str,
    scene_structure: Mapping[str, Any] | None,
) -> tuple[str, str]:
    resolved_family = _as_str(family)
    resolved_variant = _as_str(variant)
    if resolved_family and resolved_variant:
        return resolved_family, resolved_variant

    if isinstance(protocol, Mapping):
        module_name = _as_str(protocol.get("module_name"))
        if module_name == POST_ACTION_RELOCATION_MODULE_NAME:
            metadata = build_state_refresh_summary_metadata(
                protocol=protocol,
                total_object_count=_state_refresh_layout_object_count(scene_structure),
                task_name=raw_task_id or None,
            )
            resolved_family = resolved_family or _as_str(metadata.get("family"))
            resolved_variant = resolved_variant or _as_str(metadata.get("variant"))
        elif module_name == GEOMETRIC_INFERENCE_MODULE_NAME:
            resolved_family = resolved_family or _as_str(protocol.get("family") or protocol.get("track_name"))
            gt_spec = protocol.get("gt_spec", {})
            if not isinstance(gt_spec, Mapping):
                gt_spec = {}
            resolved_variant = resolved_variant or _as_str(
                protocol.get("variant") or protocol.get("subtask_name") or gt_spec.get("rule_type")
            )
        elif module_name == ANCHOR_REFERENCE_MODULE_NAME:
            resolved_family = resolved_family or _as_str(protocol.get("family"))
            resolved_variant = resolved_variant or _as_str(protocol.get("variant"))
        elif module_name == "spatial_placement":
            resolved_family = resolved_family or _as_str(protocol.get("family"))
            resolved_variant = resolved_variant or _as_str(protocol.get("variant"))

    if not resolved_family or not resolved_variant:
        canonical_task_id = RAW_TASK_ID_TO_CANONICAL.get(raw_task_id)
        if canonical_task_id:
            registry = CANONICAL_TASK_REGISTRY.get(canonical_task_id, {})
            resolved_family = resolved_family or _as_str(registry.get("family"))
    return resolved_family, resolved_variant


def infer_task_type(family: Any) -> str:
    normalized = _as_str(family)
    if normalized in {STATE_REINDEXING_FAMILY, HISTORICAL_IDENTITY_TRACKING_FAMILY, ANCHOR_REFERENCE_MODULE_NAME}:
        return DISCRETE_SELECTION_TASK_TYPE
    if normalized in {EXPLICIT_RULE_EXTRAPOLATION_FAMILY, IMPLICIT_TREND_FITTING_FAMILY, "spatial_placement"}:
        return CONTINUOUS_PLACEMENT_TASK_TYPE
    return ""


def canonicalize_task_id(raw_task_id: Any, *, family: Any = None, variant: Any = None) -> str:
    task_name = _as_str(raw_task_id)
    if task_name in CANONICAL_TASK_REGISTRY:
        return task_name
    mapped = RAW_TASK_ID_TO_CANONICAL.get(task_name)
    if mapped:
        return mapped
    family_variant = (_as_str(family), _as_str(variant))
    mapped = FAMILY_VARIANT_TO_CANONICAL.get(family_variant)
    if mapped:
        return mapped
    return task_name


def _infer_task_diagnostic_tags(
    *,
    canonical_task_id: str,
    protocol: Mapping[str, Any] | None,
) -> list[str]:
    registry_tags = _ensure_str_list(CANONICAL_TASK_REGISTRY.get(canonical_task_id, {}).get("diagnostic_tags"))
    protocol_tags = _ensure_str_list(protocol.get("diagnostic_tags")) if isinstance(protocol, Mapping) else []
    merged = registry_tags + protocol_tags
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in merged:
        if tag in seen:
            continue
        deduped.append(tag)
        seen.add(tag)
    return deduped


def _infer_scene_risk_tags(
    *,
    raw_task_id: str,
    layout: str,
    distractor_count: int | None,
) -> list[str]:
    risk_tags: list[str] = []
    lowered = " ".join([raw_task_id, layout]).lower()
    if distractor_count is not None and int(distractor_count) > 0:
        risk_tags.append("occlusion_risk")
    if "grid" in lowered:
        risk_tags.append("camera_ambiguous")
    if "vertical" in lowered:
        risk_tags.append("edge_risk")
    if "subset" in lowered:
        risk_tags.append("workspace_tight")
    return risk_tags


def _infer_instance_layout(
    *,
    layout_key: str,
    scene_structure: Mapping[str, Any] | None,
    raw_task_id: str,
) -> str:
    if isinstance(scene_structure, Mapping):
        scene_layout = _normalize_slug(scene_structure.get("layout"))
        if scene_layout:
            return scene_layout
    lowered = " ".join([layout_key, raw_task_id]).lower()
    if "grid" in lowered:
        return "grid"
    if "diagonal" in lowered:
        return "diagonal_line"
    if "front_targets_back_distractors" in lowered:
        return "front_targets_back_distractors"
    if "line" in lowered:
        return "linear"
    return ""


def _infer_axis_variant(
    *,
    protocol: Mapping[str, Any] | None,
    raw_task_id: str,
    layout_key: str,
) -> str:
    if isinstance(protocol, Mapping):
        gt_spec = protocol.get("gt_spec", {})
        if isinstance(gt_spec, Mapping):
            order_axis = _normalize_slug(gt_spec.get("order_axis"))
            if order_axis:
                return order_axis
        anchor_layout = _normalize_slug(protocol.get("anchor_layout"))
        if anchor_layout:
            return anchor_layout
    lowered = " ".join([raw_task_id, layout_key]).lower()
    if "top_row" in lowered:
        return "top_row"
    if "left_column" in lowered:
        return "left_column"
    if "vertical" in lowered:
        return "vertical"
    if "grid" in lowered:
        return "grid"
    return ""


def normalize_task_semantics(
    raw_task_spec: Mapping[str, Any] | None = None,
    *,
    raw_task_id: Any = None,
    canonical_task_id: Any = None,
    family: Any = None,
    variant: Any = None,
    protocol: Mapping[str, Any] | None = None,
    instruction: Any = None,
    task_version: Any = None,
) -> dict[str, Any]:
    existing = {}
    if isinstance(raw_task_spec, Mapping):
        existing = _ensure_jsonable_dict(raw_task_spec.get("task_semantics", {}))
    raw_task_name = _as_str(raw_task_id) or _as_str(existing.get("raw_task_id")) or _as_str(existing.get("task_id"))
    resolved_family = _as_str(family) or _as_str(existing.get("family"))
    resolved_variant = _as_str(variant) or _as_str(existing.get("variant"))
    legacy_canonical_task_id = (
        _as_str(canonical_task_id) or canonicalize_task_id(raw_task_name, family=resolved_family, variant=resolved_variant)
    )
    registry = dict(CANONICAL_TASK_REGISTRY.get(legacy_canonical_task_id, {}))
    task_metadata_overrides = {}
    if isinstance(raw_task_spec, Mapping) and isinstance(raw_task_spec.get("task_metadata"), Mapping):
        task_metadata_overrides = _ensure_jsonable_dict(raw_task_spec.get("task_metadata"))

    benchmark_task_id = resolve_benchmark_task_id(
        raw_task_name,
        canonical_task_id=legacy_canonical_task_id,
        family=resolved_family,
        variant=resolved_variant,
        protocol=protocol,
    )
    resolved_task_version = resolve_benchmark_task_version(
        benchmark_task_id,
        explicit_task_version=_as_str(task_version) or _as_str(existing.get("task_version")) or task_metadata_overrides.get("task_version"),
        raw_task_id=raw_task_name,
        protocol=protocol,
    )
    task_metadata = build_task_metadata(
        benchmark_task_id,
        task_version=resolved_task_version,
        overrides=task_metadata_overrides,
    )

    protocol_query_text = _as_str(protocol.get("query_text")) if isinstance(protocol, Mapping) else ""
    default_query = (
        _as_str(existing.get("default_query_template"))
        or protocol_query_text
        or _as_str(instruction)
        or _as_str(task_metadata.get("instruction"))
        or _as_str(registry.get("default_query_template"))
    )

    task_semantics = {
        "task_id": benchmark_task_id,
        "canonical_task_id": legacy_canonical_task_id,
        "family": resolved_family or _as_str(registry.get("family")) or _as_str(task_metadata.get("family")),
        "task_type": _as_str(existing.get("task_type")) or _as_str(registry.get("task_type")) or infer_task_type(
            resolved_family or registry.get("family")
        ),
        "core_skill": _as_str(existing.get("core_skill")) or _as_str(registry.get("core_skill")),
        "query_semantics": _as_str(existing.get("query_semantics")) or _as_str(registry.get("query_semantics")),
        "gt_mode": _as_str(existing.get("gt_mode")) or _as_str(registry.get("gt_mode")) or _protocol_gt_mode(protocol),
        "default_query_template": default_query,
        "diagnostic_tags": _ensure_str_list(existing.get("diagnostic_tags"))
        or _ensure_str_list(registry.get("diagnostic_tags")),
        "difficulty_axes": _ensure_str_list(existing.get("difficulty_axes"))
        or _ensure_str_list(registry.get("difficulty_axes")),
        "canonical_role": _as_str(existing.get("canonical_role")) or _as_str(registry.get("canonical_role")),
        "semantic_module": _as_str(task_metadata.get("semantic_module")),
        "task_version": _as_str(task_metadata.get("task_version")),
        "skill_type": _as_str(task_metadata.get("skill_type")),
        "phase": _as_str(task_metadata.get("phase")),
        "instruction": _as_str(task_metadata.get("instruction")),
        "parent_task_id": _as_str(task_metadata.get("parent_task_id")),
        "paired_control_id": _as_str(task_metadata.get("paired_control_id")),
        "enabled_probes": _ensure_str_list(task_metadata.get("enabled_probes")),
        "evaluator_key": _as_str(task_metadata.get("evaluator_key")),
        "stress_factors": _ensure_str_list(task_metadata.get("stress_factors")),
        "notes": _as_str(existing.get("notes")) or _as_str(task_metadata.get("notes")) or _as_str(registry.get("notes")),
        "is_runnable": bool(task_metadata.get("is_runnable")),
    }
    return task_semantics


def normalize_scene_semantics(
    raw_scene_info: Mapping[str, Any] | None = None,
    *,
    scene_id: Any = None,
    canonical_task_id: Any = None,
    family: Any = None,
    variant: Any = None,
    scene_structure: Mapping[str, Any] | None = None,
    raw_task_id: Any = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    existing = {}
    if isinstance(raw_scene_info, Mapping):
        if isinstance(raw_scene_info.get("scene_semantics"), Mapping):
            existing = _ensure_jsonable_dict(raw_scene_info.get("scene_semantics"))
        else:
            existing = _ensure_jsonable_dict(raw_scene_info)

    resolved_scene_id = _as_int(scene_id) or _as_int(existing.get("scene_id"))
    canonical_task_name = _as_str(canonical_task_id)
    override = dict(SCENE_METADATA_OVERRIDES.get(int(resolved_scene_id), {})) if resolved_scene_id is not None else {}
    provided_capability_tags = _ensure_str_list(existing.get("capability_tags"))
    override_capability_tags = _ensure_str_list(override.get("capability_tags"))
    canonical_capability_tags = _ensure_str_list(CANONICAL_SCENE_CAPABILITY_TAGS.get(canonical_task_name))

    capability_tags: list[str] = []
    for source in (provided_capability_tags, override_capability_tags, canonical_capability_tags):
        for tag in source:
            if tag not in capability_tags:
                capability_tags.append(tag)

    object_count = _state_refresh_layout_object_count(scene_structure)
    distractor_count = None
    if isinstance(protocol, Mapping):
        distractor_count = _as_int(protocol.get("distractor_count"))
    layout = _infer_instance_layout(
        layout_key="",
        scene_structure=scene_structure,
        raw_task_id=_as_str(raw_task_id),
    )
    risk_tags = _ensure_str_list(existing.get("risk_tags")) or _infer_scene_risk_tags(
        raw_task_id=_as_str(raw_task_id),
        layout=layout,
        distractor_count=distractor_count if distractor_count is not None else max((object_count or 0) - 3, 0)
        if object_count is not None and canonical_task_name.startswith("state_reindexing_subset")
        else distractor_count,
    )

    scene_semantics = {
        "scene_id": resolved_scene_id,
        "scene_name": _as_str(existing.get("scene_name")) or _as_str(override.get("scene_name")) or (
            f"scene_{resolved_scene_id}" if resolved_scene_id is not None else ""
        ),
        "room_type": _as_str(existing.get("room_type")),
        "table_shape": _as_str(existing.get("table_shape")),
        "table_material": _as_str(existing.get("table_material")),
        "background_level": _as_str(existing.get("background_level")),
        "reflection_level": _as_str(existing.get("reflection_level")),
        "texture_strength": _as_str(existing.get("texture_strength")),
        "workspace_margin": _as_float(existing.get("workspace_margin")),
        "camera_profile": _as_str(existing.get("camera_profile")),
        "capability_tags": capability_tags,
        "risk_tags": risk_tags,
        "recommended_for": _ensure_str_list(existing.get("recommended_for")) or ([canonical_task_name] if canonical_task_name else []),
        "not_recommended_for": _ensure_str_list(existing.get("not_recommended_for")),
        "notes": _as_str(existing.get("notes")) or _as_str(override.get("notes")),
    }
    if family and not scene_semantics["recommended_for"]:
        scene_semantics["recommended_for"] = [_as_str(family)]
    if variant:
        scene_semantics.setdefault("notes", "")
    return scene_semantics


def normalize_instance_semantics(
    raw_annotation: Mapping[str, Any] | None = None,
    *,
    raw_task_id: Any = None,
    canonical_task_id: Any = None,
    layout_key: Any = None,
    task_index: Any = None,
    scene_id: Any = None,
    family: Any = None,
    variant: Any = None,
    protocol: Mapping[str, Any] | None = None,
    scene_structure: Mapping[str, Any] | None = None,
    instruction: Any = None,
) -> dict[str, Any]:
    existing = {}
    if isinstance(raw_annotation, Mapping):
        existing = _ensure_jsonable_dict(raw_annotation.get("instance_semantics", {}))

    resolved_layout_key = _as_str(layout_key) or _as_str(existing.get("layout_key"))
    resolved_task_index = _as_int(task_index)
    resolved_scene_id = _as_int(scene_id)
    raw_task_name = _as_str(raw_task_id)
    canonical_task_name = _as_str(canonical_task_id)

    object_positions = {}
    if isinstance(scene_structure, Mapping):
        maybe_positions = scene_structure.get("object_positions", {})
        if isinstance(maybe_positions, Mapping):
            object_positions = maybe_positions

    object_count = _as_int(existing.get("object_count"))
    if object_count is None:
        object_count = _as_int(protocol.get("object_count")) if isinstance(protocol, Mapping) else None
    if object_count is None and object_positions:
        object_count = len(object_positions)
    if object_count is None and isinstance(scene_structure, Mapping):
        object_count = _as_int(scene_structure.get("num_of_objects"))

    anchor_count = _as_int(existing.get("anchor_count"))
    if anchor_count is None and isinstance(protocol, Mapping):
        anchor_count = _as_int(protocol.get("anchor_count"))
        if anchor_count is None:
            anchors = protocol.get("anchors", [])
            if isinstance(anchors, list):
                anchor_count = len(anchors)

    distractor_count = _as_int(existing.get("distractor_count"))
    if distractor_count is None and isinstance(protocol, Mapping):
        distractor_count = _as_int(protocol.get("distractor_count"))
    if distractor_count is None and isinstance(protocol, Mapping):
        subset_spec = protocol.get("target_subset_spec", {})
        if isinstance(subset_spec, Mapping):
            object_ids = subset_spec.get("object_ids", [])
            if isinstance(object_ids, Sequence) and not isinstance(object_ids, (str, bytes)):
                subset_count = len([item for item in object_ids if isinstance(item, str) and item])
                if object_count is not None and subset_count > 0:
                    distractor_count = max(int(object_count) - int(subset_count), 0)

    difficulty_factors = {}
    if isinstance(protocol, Mapping):
        difficulty_factors = _ensure_jsonable_dict(protocol.get("difficulty_factors"))
    difficulty_factors.update(_ensure_jsonable_dict(existing.get("difficulty_factors")))
    for key, value in (
        ("object_count", object_count),
        ("anchor_count", anchor_count),
        ("distractor_count", distractor_count),
        ("spacing", _as_float(existing.get("spacing"))),
    ):
        if key not in difficulty_factors and value is not None:
            difficulty_factors[key] = value

    spacing = _as_float(existing.get("spacing"))
    if spacing is None and isinstance(protocol, Mapping):
        spacing = _as_float(protocol.get("spacing"))
    noise_level = _as_str(existing.get("noise_level"))
    if not noise_level and isinstance(protocol, Mapping):
        noise_level = _as_str(protocol.get("noise_level"))
    sigma_parallel = _as_float(existing.get("sigma_parallel"))
    if sigma_parallel is None and isinstance(protocol, Mapping):
        sigma_parallel = _as_float(protocol.get("sigma_parallel"))
    sigma_perpendicular = _as_float(existing.get("sigma_perpendicular"))
    if sigma_perpendicular is None and isinstance(protocol, Mapping):
        sigma_perpendicular = _as_float(protocol.get("sigma_perpendicular"))
    intervention_type = _as_str(existing.get("intervention_type"))
    if not intervention_type and isinstance(protocol, Mapping):
        intervention_type = _as_str(protocol.get("intervention_type"))
        if not intervention_type:
            intervention = protocol.get("intervention", {})
            if isinstance(intervention, Mapping):
                intervention_type = _as_str(intervention.get("type"))
    intervention_steps = _as_int(existing.get("intervention_steps"))
    if intervention_steps is None and isinstance(protocol, Mapping):
        intervention_steps = _as_int(protocol.get("intervention_steps"))
    query_template = _as_str(existing.get("query_template"))
    if not query_template and isinstance(protocol, Mapping):
        query_template = _as_str(protocol.get("query_text"))
    if not query_template:
        query_template = _as_str(instruction)

    layout = _as_str(existing.get("layout")) or _infer_instance_layout(
        layout_key=resolved_layout_key,
        scene_structure=scene_structure,
        raw_task_id=raw_task_name,
    )
    axis_variant = _as_str(existing.get("axis_variant")) or _infer_axis_variant(
        protocol=protocol,
        raw_task_id=raw_task_name,
        layout_key=resolved_layout_key,
    )
    gt_generator = _as_str(existing.get("gt_generator")) or _protocol_gt_mode(protocol)
    success_tolerance_l2 = _as_float(existing.get("success_tolerance_l2"))
    if success_tolerance_l2 is None and isinstance(protocol, Mapping):
        success_tolerance_l2 = _as_float(protocol.get("success_tolerance_l2"))
    success_tolerance_orthogonal = _as_float(existing.get("success_tolerance_orthogonal"))
    if success_tolerance_orthogonal is None and isinstance(protocol, Mapping):
        success_tolerance_orthogonal = _as_float(protocol.get("success_tolerance_orthogonal"))

    diagnostic_tags = _ensure_str_list(existing.get("diagnostic_tags"))
    if not diagnostic_tags and isinstance(protocol, Mapping):
        diagnostic_tags = _ensure_str_list(protocol.get("diagnostic_tags"))

    instance_id = _as_str(existing.get("instance_id"))
    if not instance_id:
        if resolved_layout_key and resolved_task_index is not None:
            instance_id = f"{resolved_layout_key}:{resolved_task_index}"
        elif raw_task_name:
            instance_id = raw_task_name

    protocol_visual_mode = _as_str(protocol.get("visual_mode")) if isinstance(protocol, Mapping) else ""

    instance_semantics = {
        "instance_id": instance_id,
        "task_id": canonical_task_name,
        "raw_task_id": raw_task_name,
        "scene_id": resolved_scene_id,
        "layout_key": resolved_layout_key,
        "task_index": resolved_task_index,
        "layout": layout,
        "object_count": object_count,
        "anchor_count": anchor_count,
        "distractor_count": distractor_count,
        "visual_mode": _as_str(existing.get("visual_mode")) or protocol_visual_mode,
        "spacing": spacing,
        "axis_variant": axis_variant,
        "noise_level": noise_level,
        "sigma_parallel": sigma_parallel,
        "sigma_perpendicular": sigma_perpendicular,
        "intervention_type": intervention_type,
        "intervention_steps": intervention_steps,
        "query_template": query_template,
        "gt_generator": gt_generator,
        "success_tolerance_l2": success_tolerance_l2,
        "success_tolerance_orthogonal": success_tolerance_orthogonal,
        "diagnostic_tags": diagnostic_tags,
        "difficulty_factors": difficulty_factors,
        "family": _as_str(family),
        "variant": _as_str(variant),
    }
    return instance_semantics


def normalize_config_semantics(
    *,
    raw_task_id: Any = None,
    layout_key: Any = None,
    task_index: Any = None,
    scene_id: Any = None,
    protocol: Mapping[str, Any] | None = None,
    task_payload: Mapping[str, Any] | None = None,
    annotation: Mapping[str, Any] | None = None,
    scene_structure: Mapping[str, Any] | None = None,
    family: Any = None,
    variant: Any = None,
    instruction: Any = None,
    raw_scene_info: Mapping[str, Any] | None = None,
    task_version: Any = None,
) -> dict[str, Any]:
    raw_task_name = _as_str(raw_task_id)
    resolved_family, resolved_variant = _resolve_family_variant(
        protocol=protocol,
        raw_task_id=raw_task_name,
        family=_as_str(family),
        variant=_as_str(variant),
        scene_structure=scene_structure,
    )
    canonical_task_id = canonicalize_task_id(
        raw_task_name or _as_str(layout_key),
        family=resolved_family,
        variant=resolved_variant,
    )
    task_semantics = normalize_task_semantics(
        annotation if isinstance(annotation, Mapping) else task_payload,
        raw_task_id=raw_task_name,
        canonical_task_id=canonical_task_id,
        family=resolved_family,
        variant=resolved_variant,
        protocol=protocol,
        instruction=instruction,
        task_version=task_version,
    )
    scene_semantics = normalize_scene_semantics(
        raw_scene_info,
        scene_id=scene_id,
        canonical_task_id=canonical_task_id,
        family=resolved_family,
        variant=resolved_variant,
        scene_structure=scene_structure,
        raw_task_id=raw_task_name,
        protocol=protocol,
    )
    instance_semantics = normalize_instance_semantics(
        annotation,
        raw_task_id=raw_task_name,
        canonical_task_id=canonical_task_id,
        layout_key=layout_key,
        task_index=task_index,
        scene_id=scene_id,
        family=resolved_family,
        variant=resolved_variant,
        protocol=protocol,
        scene_structure=scene_structure,
        instruction=instruction,
    )
    task_metadata = {
        "task_id": _as_str(task_semantics.get("task_id")),
        "semantic_module": _as_str(task_semantics.get("semantic_module")),
        "task_version": _as_str(task_semantics.get("task_version")),
        "skill_type": _as_str(task_semantics.get("skill_type")),
        "phase": _as_str(task_semantics.get("phase")),
        "instruction": _as_str(task_semantics.get("instruction")) or _as_str(task_semantics.get("default_query_template")),
        "parent_task_id": _as_str(task_semantics.get("parent_task_id")),
        "paired_control_id": _as_str(task_semantics.get("paired_control_id")),
        "enabled_probes": _ensure_str_list(task_semantics.get("enabled_probes")),
        "evaluator_key": _as_str(task_semantics.get("evaluator_key")),
        "stress_factors": _ensure_str_list(task_semantics.get("stress_factors")),
        "notes": _as_str(task_semantics.get("notes")),
        "is_runnable": bool(task_semantics.get("is_runnable")),
    }
    diagnostic_tags = _infer_task_diagnostic_tags(canonical_task_id=canonical_task_id, protocol=protocol)
    for tag in _ensure_str_list(instance_semantics.get("diagnostic_tags")):
        if tag not in diagnostic_tags:
            diagnostic_tags.append(tag)

    difficulty_factors = dict(_ensure_jsonable_dict(instance_semantics.get("difficulty_factors")))
    if not difficulty_factors and isinstance(protocol, Mapping):
        difficulty_factors = _ensure_jsonable_dict(protocol.get("difficulty_factors"))

    return {
        "raw_task_name": raw_task_name,
        "canonical_task_id": canonical_task_id,
        "task_id": task_metadata["task_id"],
        "family": resolved_family or _as_str(task_semantics.get("family")),
        "variant": resolved_variant,
        "task_type": _as_str(task_semantics.get("task_type")) or infer_task_type(resolved_family),
        "semantic_module": task_metadata["semantic_module"],
        "task_version": task_metadata["task_version"],
        "skill_type": task_metadata["skill_type"],
        "phase": task_metadata["phase"],
        "parent_task_id": task_metadata["parent_task_id"],
        "paired_control_id": task_metadata["paired_control_id"],
        "enabled_probes": task_metadata["enabled_probes"],
        "evaluator_key": task_metadata["evaluator_key"],
        "stress_factors": task_metadata["stress_factors"],
        "notes": task_metadata["notes"],
        "is_runnable": task_metadata["is_runnable"],
        "task_metadata": task_metadata,
        "task_semantics": task_semantics,
        "scene_semantics": scene_semantics,
        "instance_semantics": instance_semantics,
        "diagnostic_tags": diagnostic_tags,
        "difficulty_factors": difficulty_factors,
    }


def normalize_annotation_entry(
    *,
    layout_key: Any,
    task_index: Any,
    layout: Mapping[str, Any],
    task_payload: Mapping[str, Any],
) -> dict[str, Any]:
    scene_id = _as_int(layout.get("scene_id"))
    scene_structure = layout.get("scene_structure", {})
    if not isinstance(scene_structure, Mapping):
        scene_structure = {}
    annotation = task_payload.get("annotation", {})
    if not isinstance(annotation, Mapping):
        annotation = {}
    protocol = annotation.get("protocol", {})
    if not isinstance(protocol, Mapping):
        protocol = {}
    raw_task_id = _as_str(annotation.get("task_id")) or f"{_as_str(layout_key)}__task_{int(_as_int(task_index) or 0)}"
    instruction = _as_str(task_payload.get("instruction"))
    return normalize_config_semantics(
        raw_task_id=raw_task_id,
        layout_key=layout_key,
        task_index=task_index,
        scene_id=scene_id,
        protocol=protocol,
        task_payload=task_payload,
        annotation=annotation,
        scene_structure=scene_structure,
        instruction=instruction,
        raw_scene_info=layout.get("scene_semantics") if isinstance(layout.get("scene_semantics"), Mapping) else layout,
    )


def augment_record_with_semantics(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    task_name = _as_str(
        normalized.get("task_name")
        or normalized.get("task_label")
        or normalized.get("task_id")
        or normalized.get("raw_task_name")
    )
    semantic_view = normalize_config_semantics(
        raw_task_id=task_name,
        layout_key=normalized.get("layout_key"),
        task_index=normalized.get("task_index"),
        scene_id=normalized.get("scene_id"),
        family=normalized.get("family"),
        variant=normalized.get("variant"),
        instruction=normalized.get("instruction"),
        task_version=normalized.get("task_version"),
    )
    normalized.setdefault("raw_task_name", semantic_view.get("raw_task_name") or task_name)
    normalized.setdefault("canonical_task_id", semantic_view.get("canonical_task_id"))
    if semantic_view.get("task_id") not in (None, ""):
        normalized["task_id"] = semantic_view.get("task_id")
    else:
        normalized.setdefault("task_id", semantic_view.get("task_id"))
    normalized.setdefault("task_type", semantic_view.get("task_type"))
    normalized.setdefault("family", semantic_view.get("family"))
    normalized.setdefault("variant", semantic_view.get("variant"))
    normalized.setdefault("semantic_module", semantic_view.get("semantic_module"))
    normalized.setdefault("task_version", semantic_view.get("task_version"))
    normalized.setdefault("skill_type", semantic_view.get("skill_type"))
    normalized.setdefault("phase", semantic_view.get("phase"))
    normalized.setdefault("parent_task_id", semantic_view.get("parent_task_id"))
    normalized.setdefault("paired_control_id", semantic_view.get("paired_control_id"))
    normalized.setdefault("enabled_probes", semantic_view.get("enabled_probes"))
    normalized.setdefault("evaluator_key", semantic_view.get("evaluator_key"))
    normalized.setdefault("stress_factors", semantic_view.get("stress_factors"))
    normalized.setdefault("notes", semantic_view.get("notes"))
    normalized.setdefault("is_runnable", semantic_view.get("is_runnable"))

    task_semantics = semantic_view.get("task_semantics", {})
    scene_semantics = semantic_view.get("scene_semantics", {})
    instance_semantics = semantic_view.get("instance_semantics", {})

    normalized.setdefault("core_skill", _as_str(task_semantics.get("core_skill")))
    normalized.setdefault("query_semantics", _as_str(task_semantics.get("query_semantics")))
    normalized.setdefault("gt_mode", _as_str(task_semantics.get("gt_mode")))
    normalized.setdefault("canonical_role", _as_str(task_semantics.get("canonical_role")))
    normalized.setdefault("scene_name", _as_str(scene_semantics.get("scene_name")))
    normalized.setdefault("scene_capability_tags", _ensure_str_list(scene_semantics.get("capability_tags")))
    normalized.setdefault("scene_risk_tags", _ensure_str_list(scene_semantics.get("risk_tags")))
    normalized.setdefault("instance_id", _as_str(instance_semantics.get("instance_id")))
    normalized.setdefault("instance_layout", _as_str(instance_semantics.get("layout")))
    normalized.setdefault("instance_axis_variant", _as_str(instance_semantics.get("axis_variant")))

    if normalized.get("diagnostic_tags") in (None, ""):
        normalized["diagnostic_tags"] = semantic_view.get("diagnostic_tags", [])
    if normalized.get("difficulty_factors") in (None, ""):
        normalized["difficulty_factors"] = semantic_view.get("difficulty_factors", {})

    normalized.setdefault("task_semantics", task_semantics)
    normalized.setdefault("task_metadata", semantic_view.get("task_metadata", {}))
    normalized.setdefault("scene_semantics", scene_semantics)
    normalized.setdefault("instance_semantics", instance_semantics)
    return normalized


def serialize_semantic_payload(value: Any) -> str | None:
    if value in (None, "", [], {}, ()):
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


__all__ = [
    "CANONICAL_TASK_REGISTRY",
    "CONTINUOUS_PLACEMENT_TASK_TYPE",
    "EXPLICIT_RULE_EXTRAPOLATION_FAMILY",
    "GEOMETRIC_INFERENCE_MODULE_NAME",
    "IMPLICIT_TREND_FITTING_FAMILY",
    "RAW_TASK_ID_TO_CANONICAL",
    "augment_record_with_semantics",
    "canonicalize_task_id",
    "infer_task_type",
    "normalize_annotation_entry",
    "normalize_config_semantics",
    "normalize_instance_semantics",
    "normalize_scene_semantics",
    "normalize_task_semantics",
    "serialize_semantic_payload",
]
