from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from geomind_lab.mimic.utils.task_metadata_registry import (  # noqa: E402
    TASK_METADATA_REGISTRY,
    TASK_VERSION_BASE,
    TASK_VERSION_CONTROL,
    TASK_VERSION_SEMANTIC,
    TASK_VERSION_VISUAL,
    TASK_VERSION_EXECUTION,
    TASK_VERSION_SKILL,
    TASK_VERSION_STRESS,
    TASK_VERSION_VLA,
    build_task_metadata,
    resolve_task_id,
)
from geomind_lab.mimic.utils.config_semantics_adapter import normalize_config_semantics  # noqa: E402
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import (  # noqa: E402
    load_annotated_tasks,
    load_task,
    load_task_spec,
    resolve_task_request,
    resolve_task_spec_object_names,
)


def _dummy_scene_profile_for_layout(layout_key: str) -> SimpleNamespace:
    payload = json.loads((REPO_ROOT / "assets" / "annotations" / "annotated_tasks.json").read_text(encoding="utf-8"))
    layout = payload[layout_key]
    scene_structure = layout["scene_structure"]
    object_positions = dict(scene_structure.get("object_positions", {}))
    object_names = tuple(object_positions.keys())
    xforms = {
        name: {
            "pos": (float(position["position"][0]), float(position["position"][1]), 0.75),
            "rot": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        }
        for name, position in object_positions.items()
    }
    return SimpleNamespace(
        object_names=object_names,
        xforms=xforms,
        basket_name=None,
    )


class TestTaskMetadataRegistry(unittest.TestCase):
    def test_legacy_raw_task_ids_resolve_to_new_atomic_task_ids(self) -> None:
        self.assertEqual(resolve_task_id("object_position_update_edge_frontmost"), "reindex_extreme")
        self.assertEqual(resolve_task_id("state_reindexing_current_leftmost_blocks"), "reindex_extreme")
        self.assertEqual(resolve_task_id("state_reindexing_current_leftmost_blocks_stress"), "reindex_extreme")
        self.assertEqual(resolve_task_id("object_position_update_insert_second"), "reindex_insert")
        self.assertEqual(resolve_task_id("state_reindexing_insert_second_blocks"), "reindex_insert")
        self.assertEqual(resolve_task_id("state_reindexing_insert_second_blocks_stress"), "reindex_insert")
        self.assertEqual(resolve_task_id("reorder_with_distractors_red_subset_middle_blocks"), "reindex_subset")
        self.assertEqual(resolve_task_id("reorder_with_distractors_red_subset_middle_blocks_stress"), "reindex_subset")
        self.assertEqual(resolve_task_id("object_identity_tracking_original_middle_single_step"), "track_single")
        self.assertEqual(resolve_task_id("geometric_inference_implicit_trend_weak_destruction"), "weak_destruction_control")
        self.assertEqual(resolve_task_id("geometric_inference_square_completion_skill"), "square_completion")
        self.assertEqual(resolve_task_id("ordinal_insert_static_blocks_base"), "ordinal_insert_static")
        self.assertEqual(resolve_task_id("grid_rowcol_grounding_grid_2x3_base"), "grid_rowcol_grounding")
        self.assertEqual(
            resolve_task_id(
                protocol={
                    "intervention": {"script_name": "static_scene_noop"},
                    "gt_spec": {"resolver": "rank_initial_layout"},
                }
            ),
            "ordinal_insert_static",
        )
        self.assertEqual(
            resolve_task_id(
                protocol={
                    "intervention": {"script_name": "static_scene_noop"},
                    "gt_spec": {"resolver": "grid_rowcol_slot"},
                }
            ),
            "grid_rowcol_grounding",
        )

    def test_trend_and_weak_destruction_share_control_pair_metadata(self) -> None:
        trend_base = build_task_metadata("trend_continuation", task_version=TASK_VERSION_BASE)
        weak_control = build_task_metadata("weak_destruction_control", task_version=TASK_VERSION_CONTROL)

        self.assertEqual(trend_base["paired_control_id"], "weak_destruction_control")
        self.assertEqual(weak_control["paired_control_id"], "trend_continuation")
        self.assertEqual(weak_control["parent_task_id"], "trend_continuation")
        self.assertEqual(trend_base["instruction"], weak_control["instruction"])
        self.assertEqual(trend_base["skill_type"], weak_control["skill_type"])
        self.assertEqual(trend_base["family"], weak_control["family"])
        self.assertEqual(trend_base["semantic_module"], weak_control["semantic_module"])
        self.assertEqual(trend_base["evaluator_key"], weak_control["evaluator_key"])
        self.assertEqual(trend_base["enabled_probes"], weak_control["enabled_probes"])

    def test_declared_versions_have_non_empty_instruction_and_skill_type(self) -> None:
        for task_id, entry in TASK_METADATA_REGISTRY.items():
            versions = entry.get("versions", {})
            self.assertTrue(versions, msg=f"{task_id} should declare at least one version")
            for task_version in versions:
                metadata = build_task_metadata(task_id, task_version=task_version)
                self.assertTrue(
                    str(metadata.get("instruction", "")).strip(),
                    msg=f"{task_id}:{task_version} should have a non-empty canonical instruction",
                )
                self.assertTrue(
                    str(metadata.get("skill_type", "")).strip(),
                    msg=f"{task_id}:{task_version} should have a non-empty skill_type",
                )

    def test_key_frozen_tasks_align_to_registry(self) -> None:
        place_base = build_task_metadata("place_directional", task_version=TASK_VERSION_BASE)
        place_semantic = build_task_metadata("place_directional", task_version=TASK_VERSION_SEMANTIC)
        place_visual = build_task_metadata("place_directional", task_version=TASK_VERSION_VISUAL)
        place_execution = build_task_metadata("place_directional", task_version=TASK_VERSION_EXECUTION)
        place_skill = build_task_metadata("place_directional", task_version=TASK_VERSION_SKILL)
        iso_base = build_task_metadata("isometric_extension", task_version=TASK_VERSION_BASE)
        point_base = build_task_metadata("point_symmetry", task_version=TASK_VERSION_BASE)
        point_skill = build_task_metadata("point_symmetry", task_version=TASK_VERSION_SKILL)
        point_stress = build_task_metadata("point_symmetry", task_version=TASK_VERSION_STRESS)
        reindex_base = build_task_metadata("reindex_extreme", task_version=TASK_VERSION_BASE)
        reindex_stress = build_task_metadata("reindex_extreme", task_version=TASK_VERSION_STRESS)
        reindex_insert_base = build_task_metadata("reindex_insert", task_version=TASK_VERSION_BASE)
        reindex_insert_stress = build_task_metadata("reindex_insert", task_version=TASK_VERSION_STRESS)
        reindex_subset_base = build_task_metadata("reindex_subset", task_version=TASK_VERSION_BASE)
        reindex_subset_stress = build_task_metadata("reindex_subset", task_version=TASK_VERSION_STRESS)
        trend_base = build_task_metadata("trend_continuation", task_version=TASK_VERSION_BASE)
        weak_control = build_task_metadata("weak_destruction_control", task_version=TASK_VERSION_CONTROL)
        square_base = build_task_metadata("square_completion", task_version=TASK_VERSION_BASE)
        square_skill = build_task_metadata("square_completion", task_version=TASK_VERSION_SKILL)
        square_stress = build_task_metadata("square_completion", task_version=TASK_VERSION_STRESS)

        self.assertEqual(place_base["instruction"], "移动到方块附近。")
        self.assertEqual(place_base["skill_type"], "move_to_object")
        self.assertEqual(place_base["evaluator_key"], "move_to_object_evaluator")
        self.assertEqual(place_semantic["instruction"], "将手中的方块放到蓝盘前侧区域。")
        self.assertEqual(place_semantic["skill_type"], "directional_region_placement")
        self.assertEqual(place_visual["instruction"], place_semantic["instruction"])
        self.assertEqual(place_visual["skill_type"], place_semantic["skill_type"])
        self.assertEqual(place_execution["instruction"], "把方块放到蓝盘子的前方。")
        self.assertEqual(place_execution["skill_type"], "directional_region_placement")
        self.assertEqual(place_skill["skill_type"], "routing")
        self.assertIn("obstacle_aware_routing", place_skill["enabled_probes"])

        # The `base` curriculum level is a deliberate "move near the block" reach stub, but the
        # VLA variant must stay the placement task it is named after, otherwise
        # `directional_placement_evaluator`'s goal_kind/resolver guard rejects every episode.
        place_vla = build_task_metadata("place_directional", task_version=TASK_VERSION_VLA)
        self.assertEqual(place_vla["evaluator_key"], "directional_placement_evaluator")
        self.assertEqual(place_vla["skill_type"], "directional_region_placement")
        self.assertEqual(place_vla["instruction"], "将手中的方块放到蓝盘前侧区域。")

        self.assertEqual(iso_base["instruction"], "把手中的方块放到红蓝连线的延长线上，并与蓝块保持相同间距。")
        self.assertEqual(iso_base["phase"], "phase2")
        self.assertEqual(iso_base["skill_type"], "isometric_extension_inference")

        self.assertEqual(point_base["instruction"], "以红块为中心，把手中的方块放在与蓝块对称的位置。")
        self.assertEqual(point_base["skill_type"], "point_symmetry_inference")
        self.assertEqual(
            point_skill["instruction"],
            "以红块为中心，把手中的方块放在与蓝块对称的位置，并将物体朝向旋转 180°。",
        )
        self.assertEqual(point_skill["skill_type"], "alignment")
        self.assertIn("orientation_alignment", point_skill["enabled_probes"])
        self.assertEqual(point_stress["stress_factors"], ["static_occluders_near_mirrored_target"])

        self.assertEqual(reindex_base["instruction"], "拿起当前最左侧的方块。")
        self.assertEqual(reindex_stress["skill_type"], "extreme_state_reindexing")
        self.assertEqual(reindex_stress["phase"], "phase1")
        self.assertTrue(bool(reindex_base["is_runnable"]))
        self.assertTrue(bool(reindex_stress["is_runnable"]))

        self.assertEqual(reindex_insert_base["instruction"], "拿起当前左起第二个方块。")
        self.assertEqual(reindex_insert_stress["phase"], "phase2")
        self.assertTrue(bool(reindex_insert_base["is_runnable"]))
        self.assertTrue(bool(reindex_insert_stress["is_runnable"]))

        ordinal_base = build_task_metadata("ordinal_insert_static", task_version=TASK_VERSION_BASE)
        ordinal_stress = build_task_metadata("ordinal_insert_static", task_version=TASK_VERSION_STRESS)
        ordinal_skill = build_task_metadata("ordinal_insert_static", task_version=TASK_VERSION_SKILL)
        self.assertEqual(ordinal_base["instruction"], "拿起左侧第二个方块。")
        self.assertTrue(bool(ordinal_base["is_runnable"]))
        self.assertTrue(bool(ordinal_stress["is_runnable"]))
        self.assertTrue(bool(ordinal_skill["is_runnable"]))

        grid_base = build_task_metadata("grid_rowcol_grounding", task_version=TASK_VERSION_BASE)
        self.assertEqual(grid_base["instruction"], "拿起第一排第二列的方块。")
        self.assertTrue(bool(grid_base["is_runnable"]))

        self.assertEqual(reindex_subset_base["instruction"], "只看红色方块，拿起其中位于中间的那个。")
        self.assertEqual(reindex_subset_stress["phase"], "phase1")
        self.assertTrue(bool(reindex_subset_base["is_runnable"]))
        self.assertTrue(bool(reindex_subset_stress["is_runnable"]))

        self.assertEqual(trend_base["instruction"], "顺着方块的排列走势，把手中的方块放在下一个位置。")
        self.assertEqual(trend_base["instruction"], weak_control["instruction"])
        self.assertEqual(trend_base["enabled_probes"], weak_control["enabled_probes"])

        self.assertEqual(
            square_base["instruction"],
            "Place the held block in the correct position so that the four blocks form a square.",
        )
        self.assertEqual(
            square_skill["instruction"],
            "把手中的方块放到合适的位置，使这四块方块构成一个正方形，并让方块偏航角向心对齐。",
        )
        self.assertEqual(square_skill["skill_type"], "alignment")
        self.assertEqual(square_stress["skill_type"], "alignment")
        self.assertEqual(square_stress["phase"], "phase1")
        self.assertTrue(bool(square_base["is_runnable"]))
        self.assertTrue(bool(square_skill["is_runnable"]))
        self.assertTrue(bool(square_stress["is_runnable"]))

    def test_registry_instruction_flows_into_registry_only_semantics(self) -> None:
        semantic_view = normalize_config_semantics(raw_task_id="place_directional")
        task_semantics = dict(semantic_view.get("task_semantics", {}))
        task_metadata = dict(semantic_view.get("task_metadata", {}))

        self.assertEqual(task_semantics["default_query_template"], "移动到方块附近。")
        self.assertEqual(task_metadata["instruction"], "移动到方块附近。")

    def test_load_task_spec_attaches_flat_task_metadata(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_blocks_line_4")
        task_spec = load_task_spec("state_reindex_blocks_line_4", 0, scene_profile)
        task_metadata = dict(task_spec.get("task_metadata", {}))

        self.assertEqual(task_spec["instruction"], "拿起当前最左侧的方块。")
        self.assertEqual(task_metadata["task_id"], "reindex_extreme")
        self.assertEqual(task_metadata["task_version"], "base")
        self.assertEqual(task_metadata["semantic_module"], "state_refresh")
        self.assertEqual(task_metadata["phase"], "phase1")
        self.assertIn("runtime_objects", task_spec)

    def test_load_task_falls_back_without_runtime_scene_import(self) -> None:
        loaded = load_task(task_id="track_single")

        self.assertEqual(loaded.request.task_id, "track_single")
        self.assertEqual(loaded.request.variant, "standard")
        self.assertEqual(loaded.request.scene_id, 65)
        self.assertEqual(loaded.task_spec["task_metadata"]["scene"], "scene_65")
        self.assertEqual(loaded.task_spec["task_metadata"]["scene_name"], "scene_65")
        self.assertEqual(loaded.task_spec["task_metadata"]["recommended_scene"], "scene_65")
        self.assertFalse(bool(loaded.task_spec["task_metadata"]["scene_overridden"]))
        self.assertIn("runtime_objects", loaded.task_spec)

    def test_resolved_task_request_tracks_scene_override_state(self) -> None:
        loaded = load_task(task_id="track_single", scene="lab_table")

        self.assertEqual(loaded.request.scene_id, 10)
        self.assertEqual(loaded.request.recommended_scene_id, 65)
        self.assertTrue(loaded.request.scene_overridden)
        self.assertEqual(loaded.task_spec["task_metadata"]["scene"], "scene_10")
        self.assertEqual(loaded.task_spec["task_metadata"]["recommended_scene"], "scene_65")

    def test_resolved_profile_variants_hide_stage_specific_names(self) -> None:
        self.assertEqual(resolve_task_request(task_id="perfect_midpoint", variant="standard").variant, "standard")
        self.assertEqual(resolve_task_request(task_id="perfect_midpoint", variant="hard").variant, "grid")
        self.assertEqual(resolve_task_request(task_id="isometric_extension", variant="standard").variant, "standard")
        self.assertEqual(resolve_task_request(task_id="isometric_extension", variant="hard").variant, "stress")
        self.assertEqual(resolve_task_request(task_id="point_symmetry", variant="standard").variant, "standard")
        self.assertEqual(resolve_task_request(task_id="point_symmetry", variant="hard").variant, "stress")

    def test_annotation_task_ids_are_canonical(self) -> None:
        payload = load_annotated_tasks()
        for layout_key, layout in payload.items():
            self.assertIsInstance(layout, dict, msg=layout_key)
            for task_index, task_payload in enumerate(layout.get("tasks", [])):
                self.assertIsInstance(task_payload, dict, msg=f"{layout_key}:{task_index}")
                annotation = task_payload.get("annotation", {})
                self.assertIsInstance(annotation, dict, msg=f"{layout_key}:{task_index}")
                raw_task_id = str(annotation.get("task_id") or "").strip()
                if not raw_task_id:
                    continue
                metadata = normalize_config_semantics(raw_task_id=raw_task_id, protocol=annotation.get("protocol"))
                canonical_task_id = str(metadata.get("task_metadata", {}).get("task_id") or "")
                self.assertEqual(
                    raw_task_id,
                    canonical_task_id,
                    msg=f"{layout_key}:{task_index} should store canonical task_id",
                )

    def test_runtime_task_object_names_override_scene_profile_names(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_blocks_line_4")
        task_spec = load_task_spec("state_reindex_blocks_line_4", 0, scene_profile)

        self.assertEqual(
            resolve_task_spec_object_names(task_spec, scene_profile),
            ["reindex_block_0", "reindex_block_1", "reindex_block_2", "reindex_block_3"],
        )

    def test_annotation_backed_task_metadata_uses_canonical_registry_instruction(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("geometric_inference_isometric_extension_line_2")
        task_spec = load_task_spec("geometric_inference_isometric_extension_line_2", 0, scene_profile)
        task_metadata = dict(task_spec.get("task_metadata", {}))

        self.assertEqual(task_spec["instruction"], "把手中的方块放到红蓝连线的延长线上，并与蓝块保持相同间距。")
        self.assertEqual(task_metadata["task_id"], "isometric_extension")
        self.assertEqual(task_metadata["instruction"], "把手中的方块放到红蓝连线的延长线上，并与蓝块保持相同间距。")

    def test_annotation_backed_point_symmetry_metadata_uses_canonical_registry_instruction(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("geometric_inference_point_symmetry_line_2")
        task_spec = load_task_spec("geometric_inference_point_symmetry_line_2", 0, scene_profile)
        task_metadata = dict(task_spec.get("task_metadata", {}))

        self.assertEqual(task_spec["instruction"], "以红块为中心，把手中的方块放在与蓝块对称的位置。")
        self.assertEqual(task_metadata["task_id"], "point_symmetry")
        self.assertEqual(task_metadata["task_version"], "base")
        self.assertEqual(task_metadata["instruction"], "以红块为中心，把手中的方块放在与蓝块对称的位置。")

    def test_annotation_backed_trend_metadata_uses_canonical_registry_instruction(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("geometric_inference_implicit_trend_line_3")
        task_spec = load_task_spec("geometric_inference_implicit_trend_line_3", 0, scene_profile)
        task_metadata = dict(task_spec.get("task_metadata", {}))

        self.assertEqual(task_spec["instruction"], "顺着方块的排列走势，把手中的方块放在下一个位置。")
        self.assertEqual(task_metadata["task_id"], "trend_continuation")
        self.assertEqual(task_metadata["task_version"], "base")
        self.assertEqual(task_metadata["instruction"], "顺着方块的排列走势，把手中的方块放在下一个位置。")
        self.assertIn("runtime_objects", task_spec)

    def test_annotation_backed_reindex_stress_metadata_uses_canonical_registry_instruction(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_blocks_line_4_tight")
        task_spec = load_task_spec("state_reindex_blocks_line_4_tight", 0, scene_profile)
        task_metadata = dict(task_spec.get("task_metadata", {}))

        self.assertEqual(task_spec["instruction"], "拿起当前最左侧的方块。")
        self.assertEqual(task_metadata["task_id"], "reindex_extreme")
        self.assertEqual(task_metadata["task_version"], "stress")
        self.assertEqual(task_metadata["instruction"], "拿起当前最左侧的方块。")
        self.assertIn("runtime_objects", task_spec)

    def test_annotation_backed_reindex_insert_metadata_uses_canonical_registry_instruction(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_insert_blocks_line_4")
        task_spec = load_task_spec("state_reindex_insert_blocks_line_4", 0, scene_profile)
        task_metadata = dict(task_spec.get("task_metadata", {}))

        self.assertEqual(task_spec["instruction"], "拿起当前左起第二个方块。")
        self.assertEqual(task_metadata["task_id"], "reindex_insert")
        self.assertEqual(task_metadata["task_version"], "base")
        self.assertEqual(task_metadata["instruction"], "拿起当前左起第二个方块。")
        self.assertIn("runtime_objects", task_spec)

    def test_annotation_backed_reindex_insert_stress_metadata_uses_canonical_registry_instruction(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_insert_blocks_line_4_tight")
        task_spec = load_task_spec("state_reindex_insert_blocks_line_4_tight", 0, scene_profile)
        task_metadata = dict(task_spec.get("task_metadata", {}))

        self.assertEqual(task_spec["instruction"], "拿起当前左起第二个方块。")
        self.assertEqual(task_metadata["task_id"], "reindex_insert")
        self.assertEqual(task_metadata["task_version"], "stress")
        self.assertEqual(task_metadata["instruction"], "拿起当前左起第二个方块。")
        self.assertIn("runtime_objects", task_spec)

    def test_annotation_backed_ordinal_insert_static_metadata_matches_registry(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("ordinal_insert_static_blocks_base")
        task_spec = load_task_spec("ordinal_insert_static_blocks_base", 0, scene_profile)
        task_metadata = dict(task_spec.get("task_metadata", {}))
        self.assertEqual(task_metadata["task_id"], "ordinal_insert_static")
        self.assertEqual(task_metadata["instruction"], "拿起左侧第二个方块。")

    def test_annotation_backed_grid_rowcol_metadata_matches_registry(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("grid_rowcol_grounding_grid_2x3_base")
        task_spec = load_task_spec("grid_rowcol_grounding_grid_2x3_base", 0, scene_profile)
        task_metadata = dict(task_spec.get("task_metadata", {}))
        self.assertEqual(task_metadata["task_id"], "grid_rowcol_grounding")
        self.assertEqual(task_metadata["instruction"], "拿起第一排第二列的方块。")

    def test_annotation_backed_reindex_subset_metadata_uses_canonical_registry_instruction(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_red_subset_blocks_line_5")
        task_spec = load_task_spec("state_reindex_red_subset_blocks_line_5", 0, scene_profile)
        task_metadata = dict(task_spec.get("task_metadata", {}))

        self.assertEqual(task_spec["instruction"], "只看红色方块，拿起其中位于中间的那个。")
        self.assertEqual(task_metadata["task_id"], "reindex_subset")
        self.assertEqual(task_metadata["task_version"], "base")
        self.assertEqual(task_metadata["instruction"], "只看红色方块，拿起其中位于中间的那个。")
        self.assertIn("runtime_objects", task_spec)

    def test_annotation_backed_reindex_subset_stress_metadata_uses_canonical_registry_instruction(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("state_reindex_red_subset_blocks_line_6_tight")
        task_spec = load_task_spec("state_reindex_red_subset_blocks_line_6_tight", 0, scene_profile)
        task_metadata = dict(task_spec.get("task_metadata", {}))

        self.assertEqual(task_spec["instruction"], "只看红色方块，拿起其中位于中间的那个。")
        self.assertEqual(task_metadata["task_id"], "reindex_subset")
        self.assertEqual(task_metadata["task_version"], "stress")
        self.assertEqual(task_metadata["instruction"], "只看红色方块，拿起其中位于中间的那个。")
        self.assertIn("runtime_objects", task_spec)

    def test_annotation_backed_square_completion_metadata_uses_canonical_registry_instruction(self) -> None:
        scene_profile = _dummy_scene_profile_for_layout("geometric_inference_square_completion_corner_3_skill")
        task_spec = load_task_spec("geometric_inference_square_completion_corner_3_skill", 0, scene_profile)
        task_metadata = dict(task_spec.get("task_metadata", {}))

        self.assertEqual(
            task_spec["instruction"],
            "把手中的方块放到合适的位置，使这四块方块构成一个正方形，并让方块偏航角向心对齐。",
        )
        self.assertEqual(task_metadata["task_id"], "square_completion")
        self.assertEqual(task_metadata["task_version"], "skill")
        self.assertEqual(
            task_metadata["instruction"],
            "把手中的方块放到合适的位置，使这四块方块构成一个正方形，并让方块偏航角向心对齐。",
        )


if __name__ == "__main__":
    unittest.main()
