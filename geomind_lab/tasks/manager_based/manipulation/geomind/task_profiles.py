from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCENE_ID_TO_ALIASES: dict[int, tuple[str, ...]] = {
    1: ("scene_1", "study_table"),
    2: ("scene_2",),
    3: ("scene_3",),
    4: ("scene_4", "grid_table"),
    5: ("scene_5",),
    7: ("scene_7",),
    10: ("scene_10", "lab_table", "benchmark_lab"),
    12: ("scene_12",),
    15: ("scene_15",),
    17: ("scene_17", "clutter_table", "obstacle_table"),
    21: ("scene_21",),
    23: ("scene_23",),
    26: ("scene_26",),
    65: ("scene_65", "kitchen_table", "clean_table", "runtime_table"),
}

SCENE_ID_TO_NAME: dict[int, str] = {
    scene_id: aliases[0]
    for scene_id, aliases in SCENE_ID_TO_ALIASES.items()
}

SCENE_NAME_TO_ID: dict[str, int] = {}
for _scene_id, _scene_aliases in SCENE_ID_TO_ALIASES.items():
    for _alias in _scene_aliases + (
        str(_scene_id),
        f"scene{_scene_id}",
        f"scene_{_scene_id}",
    ):
        SCENE_NAME_TO_ID.setdefault(_alias, _scene_id)


@dataclass(frozen=True)
class TaskVariantProfile:
    name: str
    layout_key: str
    scene_id: int
    task_index: int = 0
    scene_note: str = ""
    aliases: tuple[str, ...] = ()
    env_defaults: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskProfile:
    task_id: str
    backend: str
    log_root: str
    default_variant: str
    default_object_mode: str
    variants: dict[str, TaskVariantProfile]
    env_defaults: dict[str, str] = field(default_factory=dict)


def _variant(
    *,
    name: str,
    layout_key: str,
    scene_id: int,
    task_index: int = 0,
    scene_note: str = "",
    aliases: tuple[str, ...] = (),
    env_defaults: dict[str, str] | None = None,
) -> TaskVariantProfile:
    return TaskVariantProfile(
        name=name,
        layout_key=layout_key,
        scene_id=int(scene_id),
        task_index=int(task_index),
        scene_note=str(scene_note),
        aliases=tuple(str(alias) for alias in aliases if str(alias).strip()),
        env_defaults=dict(env_defaults or {}),
    )


TASK_REQUEST_ALIASES: dict[str, tuple[str, str | None]] = {
    "track_clutter_base": ("track_clutter", "standard"),
    "track_clutter_stress": ("track_clutter", "stress"),
}


TASK_PROFILES: dict[str, TaskProfile] = {
    "grasp_anchor_direction": TaskProfile(
        task_id="grasp_anchor_direction",
        backend="anchor_reference",
        log_root="anchor_reference_rollouts",
        default_variant="base",
        default_object_mode="concrete",
        variants={
            "base": _variant(
                name="base",
                layout_key="grasp_anchor_direction_scene65_base",
                scene_id=65,
                aliases=("standard",),
            ),
            "stress": _variant(
                name="stress",
                layout_key="grasp_anchor_direction_scene17_stress",
                scene_id=17,
                aliases=("hard",),
            ),
        },
    ),
    "grasp_anchor_distance": TaskProfile(
        task_id="grasp_anchor_distance",
        backend="anchor_reference",
        log_root="anchor_reference_rollouts",
        default_variant="base",
        default_object_mode="geometric",
        variants={
            "base": _variant(
                name="base",
                layout_key="grasp_anchor_distance_scene10_base",
                scene_id=10,
                aliases=("standard",),
            ),
            "stress": _variant(
                name="stress",
                layout_key="grasp_anchor_distance_scene10_stress",
                scene_id=10,
                aliases=("hard",),
            ),
        },
    ),
    "place_directional": TaskProfile(
        task_id="place_directional",
        backend="spatial_placement",
        log_root="spatial_placement_rollouts",
        default_variant="base",
        default_object_mode="concrete",
        variants={
            "base": _variant(
                name="base",
                layout_key="place_directional_scene65_base",
                scene_id=65,
                aliases=("standard",),
            ),
            "skill": _variant(
                name="skill",
                layout_key="place_directional_scene10_skill",
                scene_id=10,
            ),
            "stress": _variant(
                name="stress",
                layout_key="place_directional_scene17_stress",
                scene_id=17,
                aliases=("hard",),
            ),
            "semantic": _variant(
                name="semantic",
                layout_key="place_directional_scene65_semantic",
                scene_id=65,
            ),
            "visual": _variant(
                name="visual",
                layout_key="place_directional_scene65_visual",
                scene_id=65,
            ),
            "execution": _variant(
                name="execution",
                layout_key="place_directional_scene65_execution",
                scene_id=65,
            ),
        },
    ),
    "place_on_top": TaskProfile(
        task_id="place_on_top",
        backend="spatial_placement",
        log_root="spatial_placement_rollouts",
        default_variant="skill",
        default_object_mode="geometric",
        variants={
            "skill": _variant(
                name="skill",
                layout_key="place_on_top_scene65_skill",
                scene_id=65,
                aliases=("standard",),
            ),
            "stress": _variant(
                name="stress",
                layout_key="place_on_top_scene17_stress",
                scene_id=17,
                aliases=("hard",),
            ),
        },
    ),
    "place_inside": TaskProfile(
        task_id="place_inside",
        backend="spatial_placement",
        log_root="spatial_placement_rollouts",
        default_variant="skill",
        default_object_mode="geometric",
        variants={
            "skill": _variant(
                name="skill",
                layout_key="place_inside_scene65_skill",
                scene_id=65,
                aliases=("standard",),
            ),
            "stress": _variant(
                name="stress",
                layout_key="place_inside_scene17_stress",
                scene_id=17,
                aliases=("hard",),
            ),
        },
    ),
    "place_inside_boundary_control": TaskProfile(
        task_id="place_inside_boundary_control",
        backend="spatial_placement",
        log_root="spatial_placement_rollouts",
        default_variant="control",
        default_object_mode="geometric",
        variants={
            "control": _variant(
                name="control",
                layout_key="place_inside_boundary_control_scene65_control",
                scene_id=65,
                aliases=("standard",),
            ),
            "stress": _variant(
                name="stress",
                layout_key="place_inside_boundary_control_scene17_stress",
                scene_id=17,
            ),
        },
    ),
    "select_visual_spatial": TaskProfile(
        task_id="select_visual_spatial",
        backend="constraint_selection",
        log_root="constraint_selection_rollouts",
        default_variant="base",
        default_object_mode="geometric",
        variants={
            "base": _variant(
                name="base",
                layout_key="select_visual_spatial_scene65_base",
                scene_id=65,
                aliases=("standard",),
            ),
            "stress": _variant(
                name="stress",
                layout_key="select_visual_spatial_scene17_stress",
                scene_id=17,
                aliases=("hard",),
            ),
        },
    ),
    "select_visual_spatial_control": TaskProfile(
        task_id="select_visual_spatial_control",
        backend="constraint_selection",
        log_root="constraint_selection_rollouts",
        default_variant="control",
        default_object_mode="geometric",
        variants={
            "control": _variant(
                name="control",
                layout_key="select_visual_spatial_control_scene65_control",
                scene_id=65,
                aliases=("standard",),
            ),
            "stress": _variant(
                name="stress",
                layout_key="select_visual_spatial_control_scene17_stress",
                scene_id=17,
            ),
        },
    ),
    "select_exclusion_unique": TaskProfile(
        task_id="select_exclusion_unique",
        backend="constraint_selection",
        log_root="constraint_selection_rollouts",
        default_variant="base",
        default_object_mode="concrete",
        variants={
            "base": _variant(
                name="base",
                layout_key="select_exclusion_unique_scene65_base",
                scene_id=65,
                aliases=("standard",),
            ),
            "stress": _variant(
                name="stress",
                layout_key="select_exclusion_unique_scene17_stress",
                scene_id=17,
                aliases=("hard",),
            ),
        },
    ),
    "reindex_extreme": TaskProfile(
        task_id="reindex_extreme",
        backend="state_selection",
        log_root="state_reindex_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={"AGENT": "oracle", "AGENT_EPISODES": "1"},
        variants={
            "standard": _variant(
                name="standard",
                layout_key="state_reindex_blocks_line_4",
                scene_id=65,
                scene_note="scene 65 runtime-injected blocks only; four aligned blocks are ordered along y, with more-negative y treated as farther left",
                aliases=("base", "phase1_standard", "scene65"),
                env_defaults={
                    "PROTOCOL_DISPLACEMENT_TOL": "0.02",
                    "PROTOCOL_MIN_REORDER_GAP": "0.08",
                },
            ),
            "semantic": _variant(
                name="semantic",
                layout_key="state_reindex_blocks_line_4",
                scene_id=65,
                scene_note="scene 65 semantic variant placeholder; keeps current-leftmost recomputation and layout semantics aligned with the standard profile",
                aliases=("scene65_semantic",),
                env_defaults={
                    "PROTOCOL_DISPLACEMENT_TOL": "0.02",
                    "PROTOCOL_MIN_REORDER_GAP": "0.08",
                },
            ),
            "visual": _variant(
                name="visual",
                layout_key="state_reindex_blocks_line_4",
                scene_id=65,
                scene_note="scene 65 visual variant placeholder; keeps current-leftmost recomputation and layout semantics aligned with the standard profile",
                aliases=("scene65_visual",),
                env_defaults={
                    "PROTOCOL_DISPLACEMENT_TOL": "0.02",
                    "PROTOCOL_MIN_REORDER_GAP": "0.08",
                },
            ),
            "stress": _variant(
                name="stress",
                layout_key="state_reindex_blocks_line_4_tight",
                scene_id=10,
                scene_note="scene 10 runtime-injected blocks only; same y-axis left/right convention as base, with tighter y spacing to make the current-leftmost recomputation less forgiving",
                aliases=("hard", "phase1_hard", "scene10"),
                env_defaults={
                    "PROTOCOL_DISPLACEMENT_TOL": "0.015",
                    "PROTOCOL_MIN_REORDER_GAP": "0.08",
                },
            ),
        },
    ),
    "reindex_insert": TaskProfile(
        task_id="reindex_insert",
        backend="state_selection",
        log_root="state_reindex_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={"AGENT": "oracle", "AGENT_EPISODES": "5"},
        variants={
            "standard": _variant(
                name="standard",
                layout_key="state_reindex_insert_blocks_line_4",
                scene_id=65,
                scene_note="scene 65 runtime-injected blocks only; y-axis order defines left-to-right, and the current rightmost block is inserted between the first two blocks",
                aliases=("base", "phase2_standard", "scene65"),
                env_defaults={"PROTOCOL_DISPLACEMENT_TOL": "0.02"},
            ),
            "stress": _variant(
                name="stress",
                layout_key="state_reindex_insert_blocks_line_4_tight",
                scene_id=10,
                scene_note="scene 10 runtime-injected blocks only; same y-axis insertion rule as base, with tighter spacing so current second and third are harder to separate",
                aliases=("hard", "phase2_hard", "scene10"),
                env_defaults={"PROTOCOL_DISPLACEMENT_TOL": "0.015"},
            ),
        },
    ),
    "grid_rowcol_grounding": TaskProfile(
        task_id="grid_rowcol_grounding",
        backend="state_selection",
        log_root="state_reindex_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={"AGENT": "oracle", "AGENT_EPISODES": "5"},
        variants={
            "standard": _variant(
                name="standard",
                layout_key="grid_rowcol_grounding_grid_2x3_base",
                scene_id=65,
                scene_note="2x3 grid; row=x asc (front row first), col=y asc (left to right); GT cell (1,2)",
                aliases=("base", "phase2_standard", "scene65"),
                env_defaults={"PROTOCOL_DISPLACEMENT_TOL": "0.02"},
            ),
            "stress": _variant(
                name="stress",
                layout_key="grid_rowcol_grounding_grid_2x3_stress",
                scene_id=10,
                scene_note="2x3 tighter spacing + jitter; same (row,col) semantics",
                aliases=("hard", "phase2_hard", "scene10"),
                env_defaults={"PROTOCOL_DISPLACEMENT_TOL": "0.015"},
            ),
            "skill": _variant(
                name="skill",
                layout_key="grid_rowcol_grounding_grid_3x3_skill",
                scene_id=10,
                scene_note="3x3 denser grid; target remains first row, second column",
                aliases=("dense", "3x3"),
                env_defaults={"PROTOCOL_DISPLACEMENT_TOL": "0.012"},
            ),
        },
    ),
    "ordinal_insert_static": TaskProfile(
        task_id="ordinal_insert_static",
        backend="state_selection",
        log_root="state_reindex_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={"AGENT": "oracle", "AGENT_EPISODES": "5"},
        variants={
            "standard": _variant(
                name="standard",
                layout_key="ordinal_insert_static_blocks_base",
                scene_id=65,
                scene_note="static 1D ordinal; five runtime blocks; GT = y-sorted index_1; no scripted reorder",
                aliases=("base", "phase2_standard", "scene65"),
                env_defaults={"PROTOCOL_DISPLACEMENT_TOL": "0.02"},
            ),
            "stress": _variant(
                name="stress",
                layout_key="ordinal_insert_static_blocks_stress",
                scene_id=10,
                scene_note="tighter y spacing between rank 1 and 2; same static GT resolver",
                aliases=("hard", "phase2_hard", "scene10"),
                env_defaults={"PROTOCOL_DISPLACEMENT_TOL": "0.015"},
            ),
            "skill": _variant(
                name="skill",
                layout_key="ordinal_insert_static_blocks_skill",
                scene_id=10,
                scene_note="mild x jitter + similar colors; y-order still unique for index_1",
                aliases=("perceptual",),
                env_defaults={"PROTOCOL_DISPLACEMENT_TOL": "0.015"},
            ),
        },
    ),
    "reindex_subset": TaskProfile(
        task_id="reindex_subset",
        backend="state_selection",
        log_root="state_reindex_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={"AGENT": "oracle", "AGENT_EPISODES": "5"},
        variants={
            "standard": _variant(
                name="standard",
                layout_key="state_reindex_red_subset_blocks_line_5",
                scene_id=65,
                scene_note="scene 65 runtime-injected blocks only; only the red blocks count, and the current middle red block is recomputed after subset-only reordering",
                aliases=("base", "phase1_standard", "scene65"),
                env_defaults={"PROTOCOL_DISPLACEMENT_TOL": "0.02"},
            ),
            "stress": _variant(
                name="stress",
                layout_key="state_reindex_red_subset_blocks_line_6_tight",
                scene_id=10,
                scene_note="scene 10 runtime-injected blocks only; same red-only subset rule as base, with denser non-red distractors and tighter red spacing",
                aliases=("hard", "phase1_hard", "scene10"),
                env_defaults={"PROTOCOL_DISPLACEMENT_TOL": "0.015"},
            ),
        },
    ),
    "reindex_subset_control": TaskProfile(
        task_id="reindex_subset_control",
        backend="state_selection",
        log_root="state_reindex_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        variants={
            "standard": _variant(
                name="standard",
                layout_key="reindex_subset_control_scene65_control",
                scene_id=65,
            ),
            "stress": _variant(
                name="stress",
                layout_key="reindex_subset_control_scene10_stress",
                scene_id=10,
            ),
        },
    ),
    "reindex_neighbor": TaskProfile(
        task_id="reindex_neighbor",
        backend="state_selection",
        log_root="state_reindex_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={"AGENT": "oracle", "AGENT_EPISODES": "5"},
        variants={
            "standard": _variant(
                name="standard",
                layout_key="state_reindex_neighbor_red_left_blocks_5",
                scene_id=65,
                scene_note="scene 65 runtime-injected blocks; GT is the planar-nearest block strictly left of the red anchor (y-order), with distractors rigid-enabled",
                aliases=("base", "phase2_standard", "scene65"),
                env_defaults={"PROTOCOL_DISPLACEMENT_TOL": "0.02"},
            ),
            "stress": _variant(
                name="stress",
                layout_key="state_reindex_neighbor_red_left_blocks_6_tight",
                scene_id=10,
                scene_note="scene 10 runtime-injected blocks only; red-anchor neighborhood is denser with left-front distractor and nearby static blockers, while direct-left neighbor remains uniquely defined",
                aliases=("hard", "phase2_hard", "scene10"),
                env_defaults={"PROTOCOL_DISPLACEMENT_TOL": "0.015"},
            ),
        },
    ),
    "reindex_neighbor_control": TaskProfile(
        task_id="reindex_neighbor_control",
        backend="state_selection",
        log_root="state_reindex_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        variants={
            "standard": _variant(
                name="standard",
                layout_key="reindex_neighbor_control_scene65_control",
                scene_id=65,
            ),
            "stress": _variant(
                name="stress",
                layout_key="reindex_neighbor_control_scene10_stress",
                scene_id=10,
            ),
        },
    ),
    "track_single": TaskProfile(
        task_id="track_single",
        backend="state_selection",
        log_root="state_reindex_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={"AGENT": "oracle", "AGENT_EPISODES": "5"},
        variants={
            "standard": _variant(
                name="standard",
                layout_key="historical_identity_blocks_line_3",
                scene_id=65,
                scene_note="scene 65 runtime-injected blocks only; three aligned blocks start ordered along y, and the initially middle block swaps with either the left or right neighbor over 2.0 seconds based on the task seed while the agent must keep tracking that same initial-middle block",
                aliases=("base", "phase1_standard", "scene65"),
                env_defaults={
                    "PROTOCOL_DISPLACEMENT_TOL": "0.02",
                    "PROTOCOL_MIN_REORDER_GAP": "0.08",
                },
            ),
            "stress": _variant(
                name="stress",
                layout_key="historical_identity_blocks_line_3_tight",
                scene_id=10,
                scene_note="scene 10 runtime-injected blocks only; same initial-middle identity rule as base, but with tighter y spacing while the initially middle block still swaps with a seed-selected left or right neighbor over 2.0 seconds",
                aliases=("hard", "phase1_hard", "scene10"),
                env_defaults={
                    "PROTOCOL_DISPLACEMENT_TOL": "0.015",
                    "PROTOCOL_MIN_REORDER_GAP": "0.06",
                },
            ),
            "semantic": _variant(
                name="semantic",
                layout_key="track_single_scene65_semantic_anchors",
                scene_id=65,
                scene_note="scene 65: three irregular cubes plus static red bowl and blue can anchors +X; ranking uses y on cubes only via target_subset_spec; timed swap uses swap_middle_with_seeded_neighbor_middle_direct_over_2s (no arc interpolation; middle slot uses −X waypoint detour via middle_detour_dx to avoid +X anchor collisions)",
                aliases=("semantic_anchors", "scene65_semantic"),
                env_defaults={
                    "PROTOCOL_DISPLACEMENT_TOL": "0.02",
                    "PROTOCOL_MIN_REORDER_GAP": "0.08",
                },
            ),
            "visual": _variant(
                name="visual",
                layout_key="historical_identity_blocks_line_3_visual_occlusion",
                scene_id=65,
                scene_note="scene 65: same homogeneous three cubes and initial-middle GT as base, with moderately tighter y spacing; static gray cylinder at y-midpoint between left and initial-middle with −X rear offset, radius 0.0192 (1.2× prior), center spacing to both exchange cubes > cube edge 0.046 m with surface gap margin; force_neighbor_swap left + middle_direct motion; target_subset_spec keeps GT on the three cubes only",
                aliases=("scene65_visual", "occlusion_visual"),
                env_defaults={
                    "PROTOCOL_DISPLACEMENT_TOL": "0.02",
                    "PROTOCOL_MIN_REORDER_GAP": "0.08",
                },
            ),
        },
    ),
    "track_single_control": TaskProfile(
        task_id="track_single_control",
        backend="state_selection",
        log_root="state_reindex_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        variants={
            "standard": _variant(
                name="standard",
                layout_key="track_single_control_scene65_control",
                scene_id=65,
            ),
            "stress": _variant(
                name="stress",
                layout_key="track_single_control_scene10_stress",
                scene_id=10,
            ),
        },
    ),
    "track_multi": TaskProfile(
        task_id="track_multi",
        backend="state_selection",
        log_root="state_reindex_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={
            "AGENT": "oracle",
            "AGENT_EPISODES": "5",
            "POST_INTERVENTION_SETTLE_STEPS": "120",
            "PROTOCOL_DEBUG_INTERVAL": "0",
        },
        variants={
            "standard": _variant(
                name="standard",
                layout_key="historical_identity_multistep_blocks_line_5",
                scene_id=65,
                scene_note="scene 65 runtime-injected blocks only; five aligned blocks support multi-step identity tracking where the initially leftmost block must be recovered after two scripted reorders",
                aliases=("base", "phase2_standard", "scene65"),
                env_defaults={
                    "PROTOCOL_DISPLACEMENT_TOL": "0.02",
                    "PROTOCOL_MIN_REORDER_GAP": "0.08",
                },
            ),
            "stress": _variant(
                name="stress",
                layout_key="historical_identity_multistep_blocks_line_5_tight",
                scene_id=10,
                scene_note="scene 10 runtime-injected blocks only; same initial-leftmost identity chain as base, with tighter spacing, denser lookalike clutter, and a less intuitive final arrangement",
                aliases=("hard", "phase2_hard", "scene10"),
                env_defaults={
                    "PROTOCOL_DISPLACEMENT_TOL": "0.015",
                    "PROTOCOL_MIN_REORDER_GAP": "0.06",
                },
            ),
        },
    ),
    "track_clutter": TaskProfile(
        task_id="track_clutter",
        backend="state_selection",
        log_root="state_reindex_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={
            "AGENT": "oracle",
            "AGENT_EPISODES": "5",
            "POST_INTERVENTION_SETTLE_STEPS": "120",
            "PROTOCOL_DEBUG_INTERVAL": "0",
        },
        variants={
            "standard": _variant(
                name="standard",
                layout_key="clutter_robust_identity_blocks_line_3",
                scene_id=65,
                scene_note="scene 65: three movable blocks define the tracked initial middle identity while surrounding static clutter remains irrelevant to the ground-truth chain",
                aliases=("base", "phase1_standard", "scene65"),
                env_defaults={
                    "PROTOCOL_DISPLACEMENT_TOL": "0.02",
                    "PROTOCOL_MIN_REORDER_GAP": "0.08",
                },
            ),
            "stress": _variant(
                name="stress",
                layout_key="clutter_robust_identity_blocks_line_3_tight",
                scene_id=10,
                scene_note="scene 10: denser static obstacles and tighter grasp channels preserve the same two-step identity chain while removing shortcut opportunities",
                aliases=("hard", "phase1_hard", "scene10"),
                env_defaults={
                    "PROTOCOL_DISPLACEMENT_TOL": "0.015",
                    "PROTOCOL_MIN_REORDER_GAP": "0.06",
                },
            ),
        },
    ),
    "perfect_midpoint": TaskProfile(
        task_id="perfect_midpoint",
        backend="structure_reasoning",
        log_root="geometric_inference_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        variants={
            "standard": _variant(
                name="standard",
                layout_key="geometric_inference_perfect_midpoint_line_2",
                scene_id=65,
                scene_note="scene 65 runtime blocks; red and blue anchors plus one movable target block; no scene-native tabletop objects",
                aliases=("base", "line", "line_primary", "scene65"),
            ),
            "grid": _variant(
                name="grid",
                layout_key="geometric_inference_midpoint_grid_2_2_top_row",
                scene_id=10,
                scene_note="scene 10 runtime blocks; top-row red/blue anchors with lower-row distractor and movable target block",
                aliases=("grid_top_row", "hard", "scene10"),
            ),
        },
    ),
    "isometric_extension": TaskProfile(
        task_id="isometric_extension",
        backend="structure_reasoning",
        log_root="geometric_inference_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={"AGENT": "explicit_extension_oracle"},
        variants={
            "standard": _variant(
                name="standard",
                layout_key="geometric_inference_isometric_extension_line_2",
                scene_id=65,
                scene_note="scene 65 runtime-injected blocks only; clean two-anchor equal-spacing extension with no scene-native tabletop objects",
                aliases=("base", "line_primary", "phase2_standard", "scene65"),
                env_defaults={"EPSILON_POS": "0.05", "TAU_LINE": "0.02"},
            ),
            "grid": _variant(
                name="grid",
                layout_key="geometric_inference_extension_grid_2_2_top_row",
                scene_id=10,
                scene_note="scene 10 runtime-injected blocks only; top-row extension with a lower-row distractor block for anchor-binding checks",
                aliases=("grid_top_row", "scene10"),
                env_defaults={"EPSILON_POS": "0.05", "TAU_LINE": "0.02"},
            ),
            "stress": _variant(
                name="stress",
                layout_key="geometric_inference_extension_grid_2_2_top_row",
                scene_id=10,
                scene_note="scene 10 runtime-injected blocks only; same top-row extension setup with tighter tolerances for harder debugging",
                aliases=("hard", "phase2_hard"),
                env_defaults={"EPSILON_POS": "0.04", "TAU_LINE": "0.015"},
            ),
        },
    ),
    "point_symmetry": TaskProfile(
        task_id="point_symmetry",
        backend="structure_reasoning",
        log_root="geometric_inference_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={"AGENT": "oracle"},
        variants={
            "standard": _variant(
                name="standard",
                layout_key="geometric_inference_point_symmetry_line_2",
                scene_id=65,
                scene_note="scene 65 runtime-injected blocks only; red center block plus blue reference block define a clean diagonal point-symmetry target",
                aliases=("base", "line_primary", "phase2_standard", "scene65"),
                env_defaults={"EPSILON_POS": "0.05", "TAU_LINE": "0.02"},
            ),
            "stress": _variant(
                name="stress",
                layout_key="geometric_inference_point_symmetry_tight_obstacle",
                scene_id=10,
                scene_note="scene 10 runtime-injected blocks only; mirrored target sits beside a static barrier block with one extra distractor block",
                aliases=("hard", "tight_obstacle", "phase2_hard", "scene10"),
                env_defaults={"EPSILON_POS": "0.04", "TAU_LINE": "0.015"},
            ),
        },
    ),
    "trend_continuation": TaskProfile(
        task_id="trend_continuation",
        backend="structure_reasoning",
        log_root="geometric_inference_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={"AGENT": "implicit_trend_oracle"},
        variants={
            "standard": _variant(
                name="standard",
                layout_key="geometric_inference_implicit_trend_line_3",
                scene_id=65,
                scene_note="scene 65 runtime-injected blocks only; three anchor blocks form a clean diagonal trend and the next continuation slot is fully open",
                aliases=("base", "phase1_standard", "scene65"),
                env_defaults={"EPSILON_POS": "0.05", "TAU_LINE": "0.02", "TAU_TREND": "0.03"},
            ),
            "stress": _variant(
                name="stress",
                layout_key="geometric_inference_implicit_trend_tight_obstacle",
                scene_id=10,
                scene_note="scene 10 runtime-injected blocks only; clear trend is preserved but the continuation zone is tighter because of one distractor block and a nearby static side barrier",
                aliases=("hard", "phase2_hard", "scene10"),
                env_defaults={"EPSILON_POS": "0.04", "TAU_LINE": "0.015", "TAU_TREND": "0.02"},
            ),
        },
    ),
    "weak_destruction_control": TaskProfile(
        task_id="weak_destruction_control",
        backend="structure_reasoning",
        log_root="geometric_inference_rollouts",
        default_variant="standard",
        default_object_mode="geometric",
        env_defaults={"AGENT": "implicit_trend_oracle"},
        variants={
            "standard": _variant(
                name="standard",
                layout_key="geometric_inference_implicit_trend_weak_destruction_line_3",
                scene_id=65,
                scene_note="scene 65 runtime-injected blocks only; same object set and workspace template as trend_continuation base, but anchor structure is weakened",
                aliases=("base", "control", "scene65"),
                env_defaults={"EPSILON_POS": "0.05", "TAU_LINE": "0.02", "TAU_TREND": "0.03"},
            ),
            "stress": _variant(
                name="stress",
                layout_key="geometric_inference_implicit_trend_weak_destruction_tight_obstacle",
                scene_id=10,
                scene_note="scene 10 runtime-injected blocks only; same distractor, same barrier, same tolerances as trend_continuation stress, with structure legibility intentionally degraded",
                aliases=("hard", "phase2_hard", "scene10"),
                env_defaults={"EPSILON_POS": "0.04", "TAU_LINE": "0.015", "TAU_TREND": "0.02"},
            ),
        },
    ),
    "square_completion": TaskProfile(
        task_id="square_completion",
        backend="structure_reasoning",
        log_root="geometric_inference_rollouts",
        default_variant="base",
        default_object_mode="geometric",
        env_defaults={"AGENT": "oracle"},
        variants={
            "base": _variant(
                name="base",
                layout_key="geometric_inference_square_completion_corner_3",
                scene_id=65,
                scene_note="scene 65 runtime-injected blocks only; complete the missing square corner from three anchors",
                aliases=("standard",),
                env_defaults={"EPSILON_POS": "0.05", "TAU_LINE": "0.02", "TAU_YAW": "0.35"},
            ),
            "skill": _variant(
                name="skill",
                layout_key="geometric_inference_square_completion_corner_3_skill",
                scene_id=65,
                scene_note="scene 65 same square-completion layout as base, with inward principal-axis yaw alignment enabled",
                aliases=("yaw",),
                env_defaults={"EPSILON_POS": "0.05", "TAU_LINE": "0.02", "TAU_YAW": "0.35"},
            ),
            "stress": _variant(
                name="stress",
                layout_key="geometric_inference_square_completion_tight_obstacle",
                scene_id=10,
                scene_note="scene 10 square-completion stress layout with one distractor, one side barrier, and tighter yaw tolerance",
                aliases=("hard",),
                env_defaults={"EPSILON_POS": "0.04", "TAU_LINE": "0.015", "TAU_YAW": "0.18"},
            ),
        },
    ),
    "square_completion_decoy_control": TaskProfile(
        task_id="square_completion_decoy_control",
        backend="structure_reasoning",
        log_root="geometric_inference_rollouts",
        default_variant="control",
        default_object_mode="geometric",
        variants={
            "control": _variant(
                name="control",
                layout_key="geometric_inference_square_completion_decoy_control_base",
                scene_id=65,
            ),
            "skill": _variant(
                name="skill",
                layout_key="geometric_inference_square_completion_decoy_control_skill",
                scene_id=65,
            ),
            "stress": _variant(
                name="stress",
                layout_key="geometric_inference_square_completion_decoy_control_stress",
                scene_id=10,
            ),
        },
    ),
}


def normalize_scene_value(scene: Any) -> int | None:
    if scene in (None, "", 0, "0"):
        return None
    try:
        return int(scene)
    except (TypeError, ValueError):
        normalized = str(scene).strip().lower().replace("-", "_")
        return SCENE_NAME_TO_ID.get(normalized)


def scene_name_for_id(scene_id: int | None) -> str:
    if scene_id is None:
        return ""
    return SCENE_ID_TO_NAME.get(int(scene_id), f"scene_{int(scene_id)}")


def resolve_profile_task_id(task_id: Any) -> tuple[str, str | None]:
    requested = str(task_id or "").strip()
    if not requested:
        return "", None
    alias = TASK_REQUEST_ALIASES.get(requested)
    if alias is not None:
        return alias
    return requested, None


def get_task_profile(task_id: Any) -> TaskProfile | None:
    resolved_task_id, _ = resolve_profile_task_id(task_id)
    return TASK_PROFILES.get(resolved_task_id)


def resolve_profile_variant(task_id: Any, variant: Any = None) -> str:
    profile = get_task_profile(task_id)
    if profile is None:
        return str(variant or "").strip()

    requested = str(variant or "").strip()
    if not requested:
        return profile.default_variant
    if requested in profile.variants:
        return requested

    lowered = requested.lower().replace("-", "_")
    for variant_name, variant_profile in profile.variants.items():
        if lowered == variant_name:
            return variant_name
        if lowered in {alias.lower().replace("-", "_") for alias in variant_profile.aliases}:
            return variant_name
    return requested


def resolve_profile_variant_profile(task_id: Any, variant: Any = None) -> TaskVariantProfile | None:
    profile = get_task_profile(task_id)
    if profile is None:
        return None
    return profile.variants.get(resolve_profile_variant(task_id, variant))


def list_task_profiles() -> tuple[str, ...]:
    return tuple(sorted(TASK_PROFILES))


def resolve_task_env_defaults(task_id: Any, variant: Any = None) -> dict[str, str]:
    profile = get_task_profile(task_id)
    if profile is None:
        return {}
    merged = dict(profile.env_defaults)
    variant_profile = resolve_profile_variant_profile(task_id, variant)
    if variant_profile is not None:
        merged.update(variant_profile.env_defaults)
    return merged


def resolve_task_log_root(task_id: Any) -> str:
    profile = get_task_profile(task_id)
    return str(profile.log_root) if profile is not None else ""


def resolve_task_backend(task_id: Any) -> str:
    profile = get_task_profile(task_id)
    return str(profile.backend) if profile is not None else ""


def registered_annotation_layout_keys() -> frozenset[str]:
    """Layout keys referenced by task profiles (annotation-backed core tasks)."""
    keys: set[str] = set()
    for profile in TASK_PROFILES.values():
        for variant_profile in profile.variants.values():
            keys.add(str(variant_profile.layout_key))
    return frozenset(keys)


# Extra layouts required by unit tests or legacy protocol smoke keys not wired as profile variants.
ANNOTATION_LAYOUT_KEYS_TEST_EXTRA: frozenset[str] = frozenset(
    {
        "geometric_inference_extension_line_3",
        "geometric_inference_extension_vertical_line_3",
        "geometric_inference_extension_grid_2_2_top_row",
        "geometric_inference_extension_grid_2_3_left_column",
    }
)


__all__ = [
    "SCENE_ID_TO_ALIASES",
    "SCENE_ID_TO_NAME",
    "SCENE_NAME_TO_ID",
    "TASK_PROFILES",
    "TASK_REQUEST_ALIASES",
    "TaskProfile",
    "TaskVariantProfile",
    "ANNOTATION_LAYOUT_KEYS_TEST_EXTRA",
    "get_task_profile",
    "list_task_profiles",
    "registered_annotation_layout_keys",
    "normalize_scene_value",
    "resolve_task_backend",
    "resolve_task_env_defaults",
    "resolve_task_log_root",
    "resolve_profile_task_id",
    "resolve_profile_variant",
    "resolve_profile_variant_profile",
    "scene_name_for_id",
]
