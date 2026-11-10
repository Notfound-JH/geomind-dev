from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TASK_VERSION_BASE = "base"
TASK_VERSION_SEMANTIC = "semantic"
TASK_VERSION_VISUAL = "visual"
TASK_VERSION_EXECUTION = "execution"
TASK_VERSION_SKILL = "skill"
TASK_VERSION_STRESS = "stress"
TASK_VERSION_CONTROL = "control"
TASK_VERSION_VLA = "vla"
TASK_VERSION_ORDER = (
    TASK_VERSION_BASE,
    TASK_VERSION_SEMANTIC,
    TASK_VERSION_VISUAL,
    TASK_VERSION_EXECUTION,
    TASK_VERSION_SKILL,
    TASK_VERSION_STRESS,
    TASK_VERSION_CONTROL,
    TASK_VERSION_VLA,
)

PHASE_1 = "phase1"
PHASE_2 = "phase2"


def _task(
    *,
    semantic_module: str,
    family: str,
    evaluator_key: str,
    instruction: str,
    enabled_probes: list[str],
    notes: str,
    versions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "semantic_module": semantic_module,
        "family": family,
        "evaluator_key": evaluator_key,
        "instruction": instruction,
        "enabled_probes": list(enabled_probes),
        "notes": notes,
        "versions": dict(versions),
    }


TASK_METADATA_REGISTRY: dict[str, dict[str, Any]] = {
    "grasp_anchor_direction": _task(
        semantic_module="anchor_reference",
        family="anchor_reference",
        evaluator_key="anchor_reference_evaluator",
        instruction="拿起蓝色碗右边的那个方块。",
        enabled_probes=["anchor_object_grounding", "directional_reference_resolution"],
        notes="T1 方向参照抓取任务；stress 仅增加静态杂乱，不改变语义。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_1,
                "skill_type": "directional_anchor_grasp",
                "stress_factors": [],
                "is_runnable": False,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "directional_anchor_grasp",
                "parent_task_id": "grasp_anchor_direction",
                "stress_factors": ["denser_static_clutter"],
                "is_runnable": False,
            },
        },
    ),
    "grasp_anchor_distance": _task(
        semantic_module="anchor_reference",
        family="anchor_reference",
        evaluator_key="anchor_reference_evaluator",
        instruction="拿起离红色圆柱体最近的小球。",
        enabled_probes=["anchor_object_grounding", "distance_reference_resolution"],
        notes="T1 距离参照抓取任务；stress 仅增加极窄静态间隙，不改变语义。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_1,
                "skill_type": "distance_anchor_grasp",
                "stress_factors": [],
                "is_runnable": False,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "distance_anchor_grasp",
                "parent_task_id": "grasp_anchor_distance",
                "stress_factors": ["narrow_static_grasp_gap"],
                "is_runnable": False,
            },
        },
    ),
    "place_directional": _task(
        semantic_module="spatial_placement",
        family="spatial_placement",
        evaluator_key="directional_placement_evaluator",
        instruction="将手中的方块放到蓝盘前侧区域。",
        enabled_probes=["target_region_grounding", "directional_placement"],
        notes="T2 方向放置任务；skill 版本要求越过低矮静态障碍，stress 仅收紧目标区并增加静态杂物。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_1,
                "instruction": "移动到方块附近。",
                "skill_type": "move_to_object",
                "evaluator_key": "move_to_object_evaluator",
                "stress_factors": [],
                "is_runnable": False,
            },
            TASK_VERSION_SKILL: {
                "phase": PHASE_1,
                "skill_type": "routing",
                "parent_task_id": "place_directional",
                "enabled_probes": ["obstacle_aware_routing"],
                "stress_factors": [],
                "is_runnable": False,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_1,
                "skill_type": "directional_region_placement",
                "parent_task_id": "place_directional",
                "stress_factors": ["smaller_target_zone", "denser_static_clutter"],
                "is_runnable": False,
            },
            TASK_VERSION_SEMANTIC: {
                "phase": PHASE_1,
                "skill_type": "directional_region_placement",
                "stress_factors": [],
                "is_runnable": False,
                "notes": "与 base 相同的指令与 gt_spec；场景 65 增加第二只蓝盘与锚盘后侧（世界 −X）红色碗作为静态语义干扰。",
            },
            TASK_VERSION_VISUAL: {
                "phase": PHASE_1,
                "skill_type": "directional_region_placement",
                "stress_factors": [],
                "is_runnable": False,
                "notes": "与 base 相同指令、query_text 与 gt_spec；单锚盘；若干无语义静态几何干扰物贴近合法区边界，可轻度遮挡视线但不完全遮挡锚盘。",
            },
            TASK_VERSION_EXECUTION: {
                "phase": PHASE_1,
                "skill_type": "directional_region_placement",
                "instruction": "把方块放到蓝盘子的前方。",
                "stress_factors": [],
                "is_runnable": False,
                "notes": "低拥挤单盘；更小前方合法区；gt_spec 收紧：AABB 完全落入、与锚盘最小间距、释放后低速角速稳定。",
            },
        },
    ),
    "place_on_top": _task(
        semantic_module="spatial_placement",
        family="spatial_placement",
        evaluator_key="stack_evaluator",
        instruction="把手中的方块平稳地叠放在红色木块上。",
        enabled_probes=["support_surface_grounding", "support_top_placement"],
        notes="T2 顶置放置任务；skill 版本作为主评估版本，stress 仅增加静态支撑难度。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_2,
                "skill_type": "support_top_placement",
                "stress_factors": [],
                "is_runnable": False,
            },
            TASK_VERSION_SKILL: {
                "phase": PHASE_2,
                "skill_type": "stacking",
                "parent_task_id": "place_on_top",
                "enabled_probes": ["stack_stability"],
                "stress_factors": [],
                "is_runnable": False,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "stacking",
                "parent_task_id": "place_on_top",
                "stress_factors": ["narrower_support_surface", "reduced_surface_friction"],
                "is_runnable": False,
            },
        },
    ),
    "place_inside": _task(
        semantic_module="spatial_placement",
        family="spatial_placement",
        evaluator_key="containment_evaluator",
        instruction="把手中的方块放进蓝色的盒子里。",
        enabled_probes=["container_opening_grounding", "containment_placement"],
        notes="T2 容器放置任务；skill 版本作为主评估版本，stress 仅收紧盒口和内部静态间隙。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_2,
                "skill_type": "container_placement",
                "stress_factors": [],
                "is_runnable": False,
            },
            TASK_VERSION_SKILL: {
                "phase": PHASE_2,
                "skill_type": "containment",
                "parent_task_id": "place_inside",
                "paired_control_id": "place_inside_boundary_control",
                "enabled_probes": ["aperture_alignment"],
                "stress_factors": [],
                "is_runnable": False,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "containment",
                "parent_task_id": "place_inside",
                "paired_control_id": "place_inside_boundary_control",
                "stress_factors": ["narrower_box_opening", "smaller_internal_clearance"],
                "is_runnable": False,
            },
        },
    ),
    "place_inside_boundary_control": _task(
        semantic_module="spatial_placement",
        family="spatial_placement",
        evaluator_key="containment_evaluator",
        instruction="把手中的方块放进蓝色的盒子里。",
        enabled_probes=["container_opening_grounding", "containment_placement"],
        notes="place_inside 的 paired control；强调盒口边界与半进半出失败，复用 containment evaluator。",
        versions={
            TASK_VERSION_CONTROL: {
                "phase": PHASE_2,
                "skill_type": "containment",
                "parent_task_id": "place_inside",
                "paired_control_id": "place_inside",
                "enabled_probes": ["aperture_alignment"],
                "stress_factors": ["rim_contact_and_near_miss_layout", "stricter_containment_thresholds"],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "containment",
                "parent_task_id": "place_inside",
                "paired_control_id": "place_inside",
                "stress_factors": ["narrower_box_opening", "smaller_internal_clearance"],
                "is_runnable": True,
            },
        },
    ),
    "select_visual_spatial": _task(
        semantic_module="constraint_selection",
        family="constraint_selection",
        evaluator_key="constraint_selection_evaluator",
        instruction="拿起红色方块右边的木纹圆柱体。",
        enabled_probes=["visual_attribute_grounding", "spatial_constraint_resolution"],
        notes="T3 视觉加空间联合筛选任务；stress 只增加静态相似干扰物。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_1,
                "skill_type": "visual_spatial_constraint_selection",
                "paired_control_id": "select_visual_spatial_control",
                "stress_factors": [],
                "is_runnable": False,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "visual_spatial_constraint_selection",
                "parent_task_id": "select_visual_spatial",
                "paired_control_id": "select_visual_spatial_control",
                "stress_factors": ["more_static_lookalike_distractors"],
                "is_runnable": False,
            },
        },
    ),
    "select_visual_spatial_control": _task(
        semantic_module="constraint_selection",
        family="constraint_selection",
        evaluator_key="constraint_selection_evaluator",
        instruction="拿起红色方块右边的木纹圆柱体。",
        enabled_probes=["visual_attribute_grounding", "spatial_constraint_resolution"],
        notes="select_visual_spatial 的 paired control；显式加入「仅属性 / 仅空间」干扰项以打击捷径。",
        versions={
            TASK_VERSION_CONTROL: {
                "phase": PHASE_1,
                "skill_type": "visual_spatial_constraint_selection",
                "parent_task_id": "select_visual_spatial",
                "paired_control_id": "select_visual_spatial",
                "stress_factors": ["attribute_only_and_spatial_only_distractors"],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "visual_spatial_constraint_selection",
                "parent_task_id": "select_visual_spatial",
                "paired_control_id": "select_visual_spatial",
                "stress_factors": ["more_static_lookalike_distractors"],
                "is_runnable": True,
            },
        },
    ),
    "select_exclusion_unique": _task(
        semantic_module="constraint_selection",
        family="constraint_selection",
        evaluator_key="constraint_selection_evaluator",
        instruction="拿起离蓝盘最近且不是最左边的红方块。",
        enabled_probes=["visual_attribute_grounding", "exclusion_constraint_resolution"],
        notes="T3 排除式唯一目标筛选任务；stress 只增加静态排除干扰项。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_1,
                "skill_type": "exclusion_constraint_selection",
                "stress_factors": [],
                "is_runnable": False,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "exclusion_constraint_selection",
                "parent_task_id": "select_exclusion_unique",
                "stress_factors": ["more_static_exclusion_distractors"],
                "is_runnable": False,
            },
        },
    ),
    "reindex_extreme": _task(
        semantic_module="state_refresh",
        family="state_reindexing",
        evaluator_key="state_reindexing_evaluator",
        instruction="拿起当前最左侧的方块。",
        enabled_probes=["current_state_reindexing", "extreme_rank_resolution"],
        notes="状态刷新：极值重索引（离散选取）。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_1,
                "skill_type": "extreme_state_reindexing",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_SEMANTIC: {
                "phase": PHASE_1,
                "skill_type": "extreme_state_reindexing",
                "parent_task_id": "reindex_extreme",
                "stress_factors": [],
                "is_runnable": True,
                "notes": "语义扰动变体；保持当前最左侧重索引规则不变，仅调整语义上下文。",
            },
            TASK_VERSION_VISUAL: {
                "phase": PHASE_1,
                "skill_type": "extreme_state_reindexing",
                "parent_task_id": "reindex_extreme",
                "stress_factors": [],
                "is_runnable": True,
                "notes": "视觉扰动变体；保持当前最左侧重索引规则不变，仅调整视觉干扰。",
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_1,
                "skill_type": "extreme_state_reindexing",
                "parent_task_id": "reindex_extreme",
                "stress_factors": ["tighter_static_object_packing"],
                "is_runnable": True,
            },
        },
    ),
    "reindex_insert": _task(
        semantic_module="state_refresh",
        family="state_reindexing",
        evaluator_key="state_reindexing_evaluator",
        instruction="拿起当前左起第二个方块。",
        enabled_probes=["current_state_reindexing", "ordinal_rank_resolution"],
        notes="状态刷新：插入后的序位重索引。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_2,
                "skill_type": "ordinal_state_reindexing",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "ordinal_state_reindexing",
                "parent_task_id": "reindex_insert",
                "stress_factors": ["smaller_static_target_gaps"],
                "is_runnable": True,
            },
        },
    ),
    "ordinal_insert_static": _task(
        semantic_module="state_refresh",
        family="state_reindexing",
        evaluator_key="state_reindexing_evaluator",
        instruction="拿起左侧第二个方块。",
        enabled_probes=["ordinal_selection", "ordinal_rank_resolution"],
        notes="静态场景无锚点 1D 序位定位（无 current-update / 插入干预）。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_2,
                "skill_type": "ordinal_selection",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "ordinal_selection",
                "parent_task_id": "ordinal_insert_static",
                "stress_factors": ["smaller_static_target_gaps"],
                "is_runnable": True,
            },
            TASK_VERSION_SKILL: {
                "phase": PHASE_2,
                "skill_type": "ordinal_selection",
                "parent_task_id": "ordinal_insert_static",
                "stress_factors": ["perceptual_homogeneity", "layout_jitter"],
                "is_runnable": True,
            },
        },
    ),
    "grid_rowcol_grounding": _task(
        semantic_module="state_refresh",
        family="state_reindexing",
        evaluator_key="state_reindexing_evaluator",
        instruction="拿起第一排第二列的方块。",
        enabled_probes=["grid-indexing", "object_grounding"],
        notes="静态无锚点 2D 行列网格定位；行=world x 升序（近→远），列=y 升序（左→右）。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_2,
                "skill_type": "grid_indexing",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "grid_indexing",
                "parent_task_id": "grid_rowcol_grounding",
                "stress_factors": ["tighter_grid_spacing", "layout_jitter"],
                "is_runnable": True,
            },
            TASK_VERSION_SKILL: {
                "phase": PHASE_2,
                "skill_type": "grid_indexing",
                "parent_task_id": "grid_rowcol_grounding",
                "stress_factors": ["denser_grid", "perceptual_homogeneity"],
                "is_runnable": True,
            },
        },
    ),
    "reindex_subset": _task(
        semantic_module="state_refresh",
        family="state_reindexing",
        evaluator_key="state_reindexing_evaluator",
        instruction="只看红色方块，拿起其中位于中间的那个。",
        enabled_probes=["target_subset_grounding", "subset_rank_resolution"],
        notes="状态刷新：子集内重索引。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_1,
                "skill_type": "subset_state_reindexing",
                "paired_control_id": "reindex_subset_control",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_1,
                "skill_type": "subset_state_reindexing",
                "parent_task_id": "reindex_subset",
                "paired_control_id": "reindex_subset_control",
                "stress_factors": ["static_occluders_around_target_subset"],
                "is_runnable": True,
            },
        },
    ),
    "reindex_subset_control": _task(
        semantic_module="state_refresh",
        family="state_reindexing",
        evaluator_key="state_reindexing_evaluator",
        instruction="只看红色方块，拿起其中位于中间的那个。",
        enabled_probes=["target_subset_grounding", "subset_rank_resolution"],
        notes="reindex_subset 的 paired control；全局中间位与红色子集中位分离以打击全局序捷径。",
        versions={
            TASK_VERSION_CONTROL: {
                "phase": PHASE_1,
                "skill_type": "subset_state_reindexing",
                "parent_task_id": "reindex_subset",
                "paired_control_id": "reindex_subset",
                "stress_factors": ["global_median_differs_from_red_subset_median"],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_1,
                "skill_type": "subset_state_reindexing",
                "parent_task_id": "reindex_subset",
                "paired_control_id": "reindex_subset",
                "stress_factors": ["static_occluders_around_target_subset"],
                "is_runnable": True,
            },
        },
    ),
    "reindex_neighbor": _task(
        semantic_module="state_refresh",
        family="state_reindexing",
        evaluator_key="state_reindexing_evaluator",
        instruction="请拿起红色方块左边离它最近的物体",
        enabled_probes=["current_state_reindexing", "neighbor_relation_resolution"],
        notes="状态刷新：邻接关系重索引。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_2,
                "skill_type": "neighbor_state_reindexing",
                "paired_control_id": "reindex_neighbor_control",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "neighbor_state_reindexing",
                "parent_task_id": "reindex_neighbor",
                "paired_control_id": "reindex_neighbor_control",
                "stress_factors": ["static_occluders_around_reference_object"],
                "is_runnable": True,
            },
        },
    ),
    "reindex_neighbor_control": _task(
        semantic_module="state_refresh",
        family="state_reindexing",
        evaluator_key="state_reindexing_evaluator",
        instruction="请拿起红色方块左边离它最近的物体",
        enabled_probes=["current_state_reindexing", "neighbor_relation_resolution"],
        notes="reindex_neighbor 的 paired control；强化左前邻与直左邻混淆。",
        versions={
            TASK_VERSION_CONTROL: {
                "phase": PHASE_2,
                "skill_type": "neighbor_state_reindexing",
                "parent_task_id": "reindex_neighbor",
                "paired_control_id": "reindex_neighbor",
                "stress_factors": ["left_front_neighbor_decoy_emphasis"],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "neighbor_state_reindexing",
                "parent_task_id": "reindex_neighbor",
                "paired_control_id": "reindex_neighbor",
                "stress_factors": ["static_occluders_around_reference_object"],
                "is_runnable": True,
            },
        },
    ),
    "track_single": _task(
        semantic_module="identity_tracking",
        family="historical_identity_tracking",
        evaluator_key="identity_tracking_evaluator",
        instruction="拿起最初在中间的那个方块。",
        enabled_probes=["historical_identity_binding", "single_step_identity_tracking"],
        notes="历史身份追踪：单步身份绑定。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_1,
                "skill_type": "single_step_identity_tracking",
                "paired_control_id": "track_single_control",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_SEMANTIC: {
                "phase": PHASE_1,
                "skill_type": "single_step_identity_tracking",
                "parent_task_id": "track_single",
                "paired_control_id": "track_single_control",
                "stress_factors": ["semantic_static_anchor_corridor", "irregular_three_cube_field"],
                "is_runnable": True,
                "notes": "场景 65：同质三方块略不规则摆放；红碗与蓝罐头为静态锚点（不参与 y 排序）；仅初始 y-middle 方块落在两锚 y 走廊内，种子邻居交换后非目标占据该走廊。",
            },
            TASK_VERSION_VISUAL: {
                "phase": PHASE_1,
                "skill_type": "single_step_identity_tracking",
                "parent_task_id": "track_single",
                "paired_control_id": "track_single_control",
                "stress_factors": ["static_depth_occluder_near_swap_arc"],
                "is_runnable": True,
                "notes": "与 base 相同指令与 GT（initial_rank middle）；略收紧 y 间距；+X 灰色细高静态圆柱不参与排序，用于交换轨迹附近短时轻度遮挡。",
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "single_step_identity_tracking",
                "parent_task_id": "track_single",
                "paired_control_id": "track_single_control",
                "stress_factors": ["more_static_lookalike_distractors", "smaller_target_spacing"],
                "is_runnable": True,
            },
        },
    ),
    "track_single_control": _task(
        semantic_module="identity_tracking",
        family="historical_identity_tracking",
        evaluator_key="identity_tracking_evaluator",
        instruction="拿起最初在中间的那个方块。",
        enabled_probes=["historical_identity_binding", "single_step_identity_tracking"],
        notes="track_single 的 paired control；终态中间位被非目标占据，强调初始身份绑定。",
        versions={
            TASK_VERSION_CONTROL: {
                "phase": PHASE_1,
                "skill_type": "single_step_identity_tracking",
                "parent_task_id": "track_single",
                "paired_control_id": "track_single",
                "stress_factors": ["final_middle_position_is_decoy"],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "single_step_identity_tracking",
                "parent_task_id": "track_single",
                "paired_control_id": "track_single",
                "stress_factors": ["more_static_lookalike_distractors", "smaller_target_spacing"],
                "is_runnable": True,
            },
        },
    ),
    "track_multi": _task(
        semantic_module="identity_tracking",
        family="historical_identity_tracking",
        evaluator_key="identity_tracking_evaluator",
        instruction="拿起最初在最左侧的那个方块。",
        enabled_probes=["historical_identity_binding", "multi_step_identity_tracking"],
        notes="历史身份追踪：多步身份绑定。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_2,
                "skill_type": "multi_step_identity_tracking",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "multi_step_identity_tracking",
                "parent_task_id": "track_multi",
                "stress_factors": ["denser_static_interleaved_layout"],
                "is_runnable": True,
            },
        },
    ),
    "track_clutter": _task(
        semantic_module="identity_tracking",
        family="historical_identity_tracking",
        evaluator_key="identity_tracking_evaluator",
        instruction="在周围物体变化后，拿起最初在中间的那个方块。",
        enabled_probes=["historical_identity_binding", "clutter_robust_identity_tracking"],
        notes="历史身份追踪：静态杂乱下的鲁棒绑定。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_1,
                "skill_type": "clutter_robust_identity_tracking",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_1,
                "skill_type": "clutter_robust_identity_tracking",
                "parent_task_id": "track_clutter",
                "stress_factors": ["tighter_static_obstacles_along_grasp_path"],
                "is_runnable": True,
            },
        },
    ),
    "perfect_midpoint": _task(
        semantic_module="structure_inference",
        family="explicit_rule_extrapolation",
        evaluator_key="geometric_inference_evaluator",
        instruction="把手中的方块放在红蓝两块的正中间。",
        enabled_probes=["geometric_anchor_grounding", "midpoint_relation_inference"],
        notes="显式几何规则：中点推断。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_1,
                "skill_type": "midpoint_inference",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "midpoint_inference",
                "parent_task_id": "perfect_midpoint",
                "stress_factors": ["smaller_placement_zone", "denser_static_clutter"],
                "is_runnable": False,
            },
        },
    ),
    "isometric_extension": _task(
        semantic_module="structure_inference",
        family="explicit_rule_extrapolation",
        evaluator_key="geometric_inference_evaluator",
        instruction="把手中的方块放到红蓝连线的延长线上，并与蓝块保持相同间距。",
        enabled_probes=["geometric_anchor_grounding", "equal_spacing_extension_inference"],
        notes="显式几何规则：等距延长。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_2,
                "skill_type": "isometric_extension_inference",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "isometric_extension_inference",
                "parent_task_id": "isometric_extension",
                "stress_factors": ["tighter_l2_tolerance", "denser_static_clutter"],
                "is_runnable": False,
            },
        },
    ),
    "point_symmetry": _task(
        semantic_module="structure_inference",
        family="explicit_rule_extrapolation",
        evaluator_key="point_symmetry_evaluator",
        instruction="以红块为中心，把手中的方块放在与蓝块对称的位置。",
        enabled_probes=["geometric_anchor_grounding", "point_symmetry_inference"],
        notes="显式几何规则：点对称；skill 版本额外要求 180° 朝向翻转。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_2,
                "skill_type": "point_symmetry_inference",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_SKILL: {
                "phase": PHASE_2,
                "skill_type": "alignment",
                "parent_task_id": "point_symmetry",
                "instruction": "以红块为中心，把手中的方块放在与蓝块对称的位置，并将物体朝向旋转 180°。",
                "enabled_probes": ["orientation_alignment"],
                "stress_factors": [],
                "is_runnable": False,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "point_symmetry_inference",
                "parent_task_id": "point_symmetry",
                "stress_factors": ["static_occluders_near_mirrored_target"],
                "is_runnable": True,
            },
        },
    ),
    "trend_continuation": _task(
        semantic_module="structure_inference",
        family="implicit_trend_fitting",
        evaluator_key="trend_continuation_evaluator",
        instruction="顺着方块的排列走势，把手中的方块放在下一个位置。",
        enabled_probes=["global_trend_grounding", "trend_extrapolation"],
        notes="隐式趋势拟合主任务；与 weak_destruction_control 共享指令、probe 和 evaluator。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_1,
                "skill_type": "implicit_trend_extrapolation",
                "paired_control_id": "weak_destruction_control",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "implicit_trend_extrapolation",
                "parent_task_id": "trend_continuation",
                "paired_control_id": "weak_destruction_control",
                "stress_factors": ["tighter_l2_tolerance", "narrower_static_approach_channel"],
                "is_runnable": True,
            },
        },
    ),
    "weak_destruction_control": _task(
        semantic_module="structure_inference",
        family="implicit_trend_fitting",
        evaluator_key="trend_continuation_evaluator",
        instruction="顺着方块的排列走势，把手中的方块放在下一个位置。",
        enabled_probes=["global_trend_grounding", "trend_extrapolation"],
        notes="trend_continuation 的 paired control；除弱结构破坏外，其余控制变量应保持一致。",
        versions={
            TASK_VERSION_CONTROL: {
                "phase": PHASE_1,
                "skill_type": "implicit_trend_extrapolation",
                "parent_task_id": "trend_continuation",
                "paired_control_id": "trend_continuation",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_2,
                "skill_type": "implicit_trend_extrapolation",
                "parent_task_id": "trend_continuation",
                "paired_control_id": "trend_continuation",
                "stress_factors": ["tighter_l2_tolerance", "narrower_static_approach_channel"],
                "is_runnable": True,
            },
        },
    ),
    "square_completion": _task(
        semantic_module="structure_inference",
        family="shape_completion",
        evaluator_key="square_completion_evaluator",
        instruction="Place the held block in the correct position so that the four blocks form a square.",
        enabled_probes=["geometric_anchor_grounding", "shape_completion_inference"],
        notes="形状补全；skill 版本额外要求偏航角向心对齐。",
        versions={
            TASK_VERSION_BASE: {
                "phase": PHASE_1,
                "skill_type": "shape_completion",
                "paired_control_id": "square_completion_decoy_control",
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_SKILL: {
                "phase": PHASE_1,
                "skill_type": "alignment",
                "parent_task_id": "square_completion",
                "paired_control_id": "square_completion_decoy_control",
                "instruction": "把手中的方块放到合适的位置，使这四块方块构成一个正方形，并让方块偏航角向心对齐。",
                "enabled_probes": ["orientation_alignment"],
                "stress_factors": [],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_1,
                "skill_type": "alignment",
                "parent_task_id": "square_completion",
                "paired_control_id": "square_completion_decoy_control",
                "instruction": "把手中的方块放到合适的位置，使这四块方块构成一个正方形，并让方块偏航角向心对齐。",
                "stress_factors": ["tighter_yaw_tolerance", "denser_static_clutter"],
                "is_runnable": True,
            },
        },
    ),
    "square_completion_decoy_control": _task(
        semantic_module="structure_inference",
        family="shape_completion",
        evaluator_key="square_completion_evaluator",
        instruction="Place the held block in the correct position so that the four blocks form a square.",
        enabled_probes=["geometric_anchor_grounding", "shape_completion_inference"],
        notes=(
            "square_completion 的 paired control；与 skill/stress 同为「位置+向心偏航」能力层，仅场景加 decoy；"
            "control/skill/stress 三档 instruction 一致（对齐主任务 skill 文案）。"
        ),
        versions={
            TASK_VERSION_CONTROL: {
                "phase": PHASE_1,
                "skill_type": "alignment",
                "parent_task_id": "square_completion",
                "paired_control_id": "square_completion",
                "instruction": "把手中的方块放到合适的位置，使这四块方块构成一个正方形，并让方块偏航角向心对齐。",
                "enabled_probes": ["orientation_alignment"],
                "stress_factors": ["nearby_decoy_corner_block"],
                "is_runnable": True,
            },
            TASK_VERSION_SKILL: {
                "phase": PHASE_1,
                "skill_type": "alignment",
                "parent_task_id": "square_completion",
                "paired_control_id": "square_completion",
                "instruction": "把手中的方块放到合适的位置，使这四块方块构成一个正方形，并让方块偏航角向心对齐。",
                "enabled_probes": ["orientation_alignment"],
                "stress_factors": ["nearby_decoy_corner_block"],
                "is_runnable": True,
            },
            TASK_VERSION_STRESS: {
                "phase": PHASE_1,
                "skill_type": "alignment",
                "parent_task_id": "square_completion",
                "paired_control_id": "square_completion",
                "instruction": "把手中的方块放到合适的位置，使这四块方块构成一个正方形，并让方块偏航角向心对齐。",
                "stress_factors": ["tighter_yaw_tolerance", "denser_static_clutter"],
                "is_runnable": True,
            },
        },
    ),
}


RAW_TASK_ID_TO_TASK_ID = {
    "state_reindexing_current_leftmost": "reindex_extreme",
    "state_reindexing_current_rightmost": "reindex_extreme",
    "state_reindexing_current_leftmost_after_rightmost_shift": "reindex_extreme",
    "state_reindexing_current_leftmost_blocks": "reindex_extreme",
    "state_reindexing_current_leftmost_blocks_stress": "reindex_extreme",
    "object_position_update_edge_frontmost": "reindex_extreme",
    "state_reindexing_insert_second_blocks": "reindex_insert",
    "state_reindexing_insert_second_blocks_stress": "reindex_insert",
    "object_position_update_insert_second": "reindex_insert",
    "ordinal_insert_static_blocks_base": "ordinal_insert_static",
    "ordinal_insert_static_blocks_stress": "ordinal_insert_static",
    "ordinal_insert_static_blocks_skill": "ordinal_insert_static",
    "grid_rowcol_grounding_grid_2x3_base": "grid_rowcol_grounding",
    "grid_rowcol_grounding_grid_2x3_stress": "grid_rowcol_grounding",
    "grid_rowcol_grounding_grid_3x3_skill": "grid_rowcol_grounding",
    "reorder_with_distractors_red_subset_middle_blocks": "reindex_subset",
    "reorder_with_distractors_red_subset_middle_blocks_stress": "reindex_subset",
    "reorder_with_distractors_filtered_frontmost": "reindex_subset",
    "reorder_with_distractors_middle_of_subset": "reindex_subset",
    "historical_identity_tracking_original_middle": "track_single",
    "object_identity_tracking_original_middle_single_step": "track_single",
    "object_identity_tracking_original_frontmost_multistep": "track_multi",
    "geometric_inference_perfect_midpoint_primary": "perfect_midpoint",
    "geometric_inference_midpoint_primary": "perfect_midpoint",
    "geometric_inference_isometric_extension_primary": "isometric_extension",
    "geometric_inference_extension_primary": "isometric_extension",
    "geometric_inference_extension_vertical_primary": "isometric_extension",
    "geometric_inference_extension_grid_top_row": "isometric_extension",
    "geometric_inference_extension_grid_left_column": "isometric_extension",
    "geometric_inference_implicit_trend_primary": "trend_continuation",
    "geometric_inference_implicit_trend_stress": "trend_continuation",
    "geometric_inference_implicit_trend_weak_destruction": "weak_destruction_control",
    "geometric_inference_implicit_trend_weak_destruction_stress": "weak_destruction_control",
    "geometric_inference_square_completion_primary": "square_completion",
    "geometric_inference_square_completion_skill": "square_completion",
    "geometric_inference_square_completion_stress": "square_completion",
    "track_clutter_base": "track_clutter",
    "track_clutter_stress": "track_clutter",
}


LEGACY_CANONICAL_TASK_ID_TO_TASK_ID = {
    "state_reindexing_extreme_after_edge_shift": "reindex_extreme",
    "state_reindexing_internal_insertion": "reindex_insert",
    "state_reindexing_subset_middle_with_distractors": "reindex_subset",
    "historical_identity_tracking_single_step": "track_single",
    "historical_identity_tracking_multistep": "track_multi",
    "perfect_midpoint": "perfect_midpoint",
    "isometric_extension": "isometric_extension",
    "trend_continuation": "trend_continuation",
    "weak_destruction_control": "weak_destruction_control",
    "square_completion": "square_completion",
}


FAMILY_VARIANT_TO_TASK_ID = {
    ("state_reindexing", "edge_translation"): "reindex_extreme",
    ("state_reindexing", "edge_translation_reverse"): "reindex_extreme",
    ("state_reindexing", "internal_insertion"): "reindex_insert",
    ("state_reindexing", "clutter_filtering"): "reindex_subset",
    ("state_reindexing", "semantic_isolation"): "reindex_subset",
    ("historical_identity_tracking", "single_step_tracking"): "track_single",
    ("historical_identity_tracking", "historical_two_step_tracking"): "track_single",
    ("historical_identity_tracking", "multi_step_shell_game"): "track_multi",
    ("historical_identity_tracking", "clutter_robust_two_step"): "track_clutter",
    ("explicit_rule_extrapolation", "perfect_midpoint"): "perfect_midpoint",
    ("explicit_rule_extrapolation", "midpoint"): "perfect_midpoint",
    ("explicit_rule_extrapolation", "isometric_extension"): "isometric_extension",
    ("implicit_trend_fitting", "trend_continuation"): "trend_continuation",
    ("implicit_trend_fitting", "weak_destruction_control"): "weak_destruction_control",
    ("shape_completion", "square_completion"): "square_completion",
}


SCRIPT_NAME_TO_TASK_ID = {
    "leftmost_to_right_of_rightmost": "reindex_extreme",
    "swap_adjacent_pairs_over_4": "reindex_extreme",
    "rightmost_to_left_of_leftmost": "reindex_extreme",
    "rightmost_to_between_first_and_second": "reindex_insert",
    "middle_to_left_of_leftmost": "track_single",
    "swap_first_and_second_over_2s": "track_single",
    "swap_middle_with_seeded_neighbor_over_2s": "track_single",
    "swap_middle_with_seeded_neighbor_middle_direct_over_2s": "track_single",
    "middle_to_left_then_rightmost_to_left": "track_single",
    "leftmost_to_middle_then_rightmost_to_left_of_leftmost": "track_multi",
}


def _as_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _unique_str_list(*values: Any) -> list[str]:
    merged: list[str] = []
    for value in values:
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, (list, tuple)):
            candidates = list(value)
        else:
            candidates = []
        for candidate in candidates:
            text = _as_str(candidate)
            if not text or text in merged:
                continue
            merged.append(text)
    return merged


def declared_task_ids() -> tuple[str, ...]:
    return tuple(TASK_METADATA_REGISTRY.keys())


def supported_task_versions(task_id: str) -> tuple[str, ...]:
    entry = TASK_METADATA_REGISTRY.get(_as_str(task_id), {})
    versions = entry.get("versions", {})
    if isinstance(versions, Mapping):
        ordered = tuple(version for version in TASK_VERSION_ORDER if version in versions)
        if ordered and TASK_VERSION_VLA not in ordered:
            return (*ordered, TASK_VERSION_VLA)
        return ordered
    return ()


def default_task_version(task_id: str) -> str:
    versions = supported_task_versions(task_id)
    if TASK_VERSION_BASE in versions:
        return TASK_VERSION_BASE
    if TASK_VERSION_CONTROL in versions:
        return TASK_VERSION_CONTROL
    return versions[0] if versions else TASK_VERSION_BASE


def resolve_task_id(
    raw_task_id: Any = None,
    *,
    canonical_task_id: Any = None,
    family: Any = None,
    variant: Any = None,
    protocol: Mapping[str, Any] | None = None,
) -> str:
    requested = _as_str(raw_task_id)
    if requested in TASK_METADATA_REGISTRY:
        return requested

    canonical = _as_str(canonical_task_id)
    if canonical in TASK_METADATA_REGISTRY:
        return canonical
    if canonical in LEGACY_CANONICAL_TASK_ID_TO_TASK_ID:
        return LEGACY_CANONICAL_TASK_ID_TO_TASK_ID[canonical]

    if requested in RAW_TASK_ID_TO_TASK_ID:
        return RAW_TASK_ID_TO_TASK_ID[requested]

    resolved_family = _as_str(family)
    resolved_variant = _as_str(variant)
    mapped = FAMILY_VARIANT_TO_TASK_ID.get((resolved_family, resolved_variant))
    if mapped:
        return mapped

    if isinstance(protocol, Mapping):
        protocol_variant = _as_str(protocol.get("variant"))
        mapped = FAMILY_VARIANT_TO_TASK_ID.get((resolved_family, protocol_variant))
        if mapped:
            return mapped

        gt_spec = protocol.get("gt_spec", {})
        if isinstance(gt_spec, Mapping):
            rule_type = _as_str(gt_spec.get("rule_type"))
            mapped = FAMILY_VARIANT_TO_TASK_ID.get((resolved_family, rule_type))
            if mapped:
                return mapped

        intervention = protocol.get("intervention", {})
        if isinstance(intervention, Mapping):
            script_name = _as_str(intervention.get("script_name"))
            if script_name == "static_scene_noop":
                gt_spec_check = protocol.get("gt_spec", {})
                if isinstance(gt_spec_check, Mapping) and _as_str(gt_spec_check.get("resolver")) == "grid_rowcol_slot":
                    return "grid_rowcol_grounding"
                return "ordinal_insert_static"
            mapped = SCRIPT_NAME_TO_TASK_ID.get(script_name)
            if mapped:
                return mapped

    return requested or canonical


def resolve_task_version(
    task_id: Any,
    *,
    explicit_task_version: Any = None,
    raw_task_id: Any = None,
    protocol: Mapping[str, Any] | None = None,
) -> str:
    resolved_task_id = _as_str(task_id)
    supported_versions = set(supported_task_versions(resolved_task_id))
    explicit = _as_str(explicit_task_version)
    if explicit and explicit in supported_versions:
        return explicit

    if isinstance(protocol, Mapping):
        protocol_version = _as_str(protocol.get("task_version"))
        if protocol_version in supported_versions:
            return protocol_version

    raw_name = _as_str(raw_task_id)
    if resolved_task_id == "weak_destruction_control" or raw_name == "geometric_inference_implicit_trend_weak_destruction":
        return TASK_VERSION_CONTROL if TASK_VERSION_CONTROL in supported_versions else default_task_version(resolved_task_id)

    return default_task_version(resolved_task_id)


def _vla_base_version_payload(entry: dict[str, Any]) -> dict[str, Any]:
    versions = entry.get("versions", {})
    if not isinstance(versions, Mapping):
        return {}
    canonical_evaluator = _as_str(entry.get("evaluator_key")).strip().lower()

    def _contradicts_canonical(payload: Mapping[str, Any]) -> bool:
        """True when a version is a curriculum stub for a *different* skill than the task.

        ``place_directional``'s base level is a non-runnable "move near the block" reach stub
        carrying ``move_to_object_evaluator``. Deriving the VLA profile from it turns a placement
        task into a reach task, and ``LayoutAwareEvaluator``'s goal_kind/resolver guard then
        forces ``success=False`` for every episode of that task.
        """
        version_evaluator = _as_str(payload.get("evaluator_key")).strip().lower()
        return bool(version_evaluator and canonical_evaluator and version_evaluator != canonical_evaluator)

    base_payload = versions.get(TASK_VERSION_BASE)
    if isinstance(base_payload, Mapping):
        if not _contradicts_canonical(base_payload):
            return dict(base_payload)
        for version in TASK_VERSION_ORDER:
            if version in {TASK_VERSION_BASE, TASK_VERSION_VLA}:
                continue
            payload = versions.get(version)
            if not isinstance(payload, Mapping):
                continue
            # `parent_task_id` marks a derived sub-skill (e.g. obstacle routing), which is no
            # better a template for the VLA profile than the stub we just rejected.
            if _contradicts_canonical(payload) or _as_str(payload.get("parent_task_id")).strip():
                continue
            substitute = dict(payload)
            # Notes describe that donor version's own scene tweaks, which do not apply here.
            substitute.pop("notes", None)
            return substitute
        return dict(base_payload)
    for version in TASK_VERSION_ORDER:
        if version in versions and version != TASK_VERSION_VLA:
            return dict(versions[version])
    return {}


def _infer_vla_version_payload(entry: dict[str, Any], base_vp: dict[str, Any]) -> dict[str, Any]:
    """Derive VLA registry fields from the canonical base skill metadata."""
    out = dict(base_vp)
    base_eval = str(out.get("evaluator_key") or entry.get("evaluator_key", "")).strip().lower()
    vla_note = (
        "VLA profile: end-effector proximity to target where applicable; "
        "placement/geometric tasks keep protocol evaluators (often with pre-attached block in layout)."
    )
    prior_notes = _as_str(out.get("notes"))
    out["notes"] = f"{prior_notes} {vla_note}".strip() if prior_notes else vla_note
    out["stress_factors"] = []

    if base_eval in {
        "anchor_reference_evaluator",
        "identity_tracking_evaluator",
        "state_reindexing_evaluator",
        "constraint_selection_evaluator",
        "move_to_object_evaluator",
    }:
        out["evaluator_key"] = "move_to_object_evaluator"
        st = _as_str(out.get("skill_type"))
        if st and not st.endswith("_vla"):
            out["skill_type"] = f"{st}_vla"
        return out

    return out


def build_task_metadata(
    task_id: Any,
    *,
    task_version: Any = None,
    instruction: Any = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_task_id = _as_str(task_id)
    entry = TASK_METADATA_REGISTRY.get(resolved_task_id)
    explicit_overrides = dict(overrides) if isinstance(overrides, Mapping) else {}
    if entry is None:
        fallback_version = _as_str(task_version) or TASK_VERSION_BASE
        return {
            "task_id": resolved_task_id,
            "semantic_module": "",
            "task_version": fallback_version,
            "skill_type": "",
            "phase": "",
            "instruction": _as_str(instruction),
            "parent_task_id": "",
            "paired_control_id": "",
            "enabled_probes": [],
            "evaluator_key": "",
            "stress_factors": [],
            "notes": "",
            "family": "",
            "is_runnable": False,
            "supported_task_versions": [],
        }

    resolved_version = resolve_task_version(
        resolved_task_id,
        explicit_task_version=task_version,
    )
    if resolved_version == TASK_VERSION_VLA:
        version_payload = _infer_vla_version_payload(entry, _vla_base_version_payload(entry))
    else:
        version_payload = dict(entry["versions"].get(resolved_version, {}))

    metadata = {
        "task_id": resolved_task_id,
        "semantic_module": _as_str(entry.get("semantic_module")),
        "task_version": resolved_version,
        "skill_type": _as_str(version_payload.get("skill_type")),
        "phase": _as_str(version_payload.get("phase")),
        "instruction": _as_str(instruction) or _as_str(version_payload.get("instruction")) or _as_str(entry.get("instruction")),
        "parent_task_id": _as_str(version_payload.get("parent_task_id")),
        "paired_control_id": _as_str(version_payload.get("paired_control_id")),
        "enabled_probes": _unique_str_list(version_payload.get("enabled_probes"), entry.get("enabled_probes")),
        "evaluator_key": _as_str(version_payload.get("evaluator_key")) or _as_str(entry.get("evaluator_key")),
        "stress_factors": _unique_str_list(version_payload.get("stress_factors")),
        "notes": _as_str(version_payload.get("notes")) or _as_str(entry.get("notes")),
        "family": _as_str(version_payload.get("family")) or _as_str(entry.get("family")),
        "is_runnable": bool(version_payload.get("is_runnable", False)),
        "supported_task_versions": list(supported_task_versions(resolved_task_id)),
    }

    for key, value in explicit_overrides.items():
        if key in {"enabled_probes", "stress_factors"}:
            metadata[key] = _unique_str_list(value)
        elif key in metadata:
            metadata[key] = value
    return metadata


def _validate_paired_control_registry() -> None:
    shared_pairs = (
        ("trend_continuation", TASK_VERSION_BASE, "weak_destruction_control", TASK_VERSION_CONTROL),
        ("trend_continuation", TASK_VERSION_STRESS, "weak_destruction_control", TASK_VERSION_STRESS),
        ("select_visual_spatial", TASK_VERSION_BASE, "select_visual_spatial_control", TASK_VERSION_CONTROL),
        ("select_visual_spatial", TASK_VERSION_STRESS, "select_visual_spatial_control", TASK_VERSION_STRESS),
        ("track_single", TASK_VERSION_BASE, "track_single_control", TASK_VERSION_CONTROL),
        ("track_single", TASK_VERSION_STRESS, "track_single_control", TASK_VERSION_STRESS),
        ("place_inside", TASK_VERSION_SKILL, "place_inside_boundary_control", TASK_VERSION_CONTROL),
        ("place_inside", TASK_VERSION_STRESS, "place_inside_boundary_control", TASK_VERSION_STRESS),
        # decoy_control 与主任务 skill 同为「位置+朝向」层；base 为仅位置层，不与 decoy control 做 instruction 对齐校验
        ("square_completion", TASK_VERSION_SKILL, "square_completion_decoy_control", TASK_VERSION_CONTROL),
        ("square_completion", TASK_VERSION_SKILL, "square_completion_decoy_control", TASK_VERSION_SKILL),
        ("square_completion", TASK_VERSION_STRESS, "square_completion_decoy_control", TASK_VERSION_STRESS),
        ("reindex_neighbor", TASK_VERSION_BASE, "reindex_neighbor_control", TASK_VERSION_CONTROL),
        ("reindex_neighbor", TASK_VERSION_STRESS, "reindex_neighbor_control", TASK_VERSION_STRESS),
        ("reindex_subset", TASK_VERSION_BASE, "reindex_subset_control", TASK_VERSION_CONTROL),
        ("reindex_subset", TASK_VERSION_STRESS, "reindex_subset_control", TASK_VERSION_STRESS),
    )
    shared_fields = ("semantic_module", "family", "evaluator_key", "enabled_probes", "phase", "instruction", "skill_type")
    for lhs_task_id, lhs_version, rhs_task_id, rhs_version in shared_pairs:
        lhs = build_task_metadata(lhs_task_id, task_version=lhs_version)
        rhs = build_task_metadata(rhs_task_id, task_version=rhs_version)
        for field_name in shared_fields:
            if lhs.get(field_name) != rhs.get(field_name):
                raise ValueError(
                    "Paired control metadata mismatch for "
                    f"{lhs_task_id}:{lhs_version} vs {rhs_task_id}:{rhs_version} on field {field_name!r}"
                )
        if lhs_version == TASK_VERSION_STRESS and rhs_version == TASK_VERSION_STRESS and lhs.get("stress_factors") != rhs.get("stress_factors"):
            raise ValueError(
                "Paired control stress_factors must match for "
                f"{lhs_task_id}:{lhs_version} vs {rhs_task_id}:{rhs_version}"
            )


_validate_paired_control_registry()


__all__ = [
    "FAMILY_VARIANT_TO_TASK_ID",
    "LEGACY_CANONICAL_TASK_ID_TO_TASK_ID",
    "PHASE_1",
    "PHASE_2",
    "RAW_TASK_ID_TO_TASK_ID",
    "TASK_METADATA_REGISTRY",
    "TASK_VERSION_BASE",
    "TASK_VERSION_CONTROL",
    "TASK_VERSION_ORDER",
    "TASK_VERSION_SEMANTIC",
    "TASK_VERSION_VISUAL",
    "TASK_VERSION_EXECUTION",
    "TASK_VERSION_SKILL",
    "TASK_VERSION_STRESS",
    "build_task_metadata",
    "declared_task_ids",
    "default_task_version",
    "resolve_task_id",
    "resolve_task_version",
    "supported_task_versions",
]
