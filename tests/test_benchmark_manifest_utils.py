from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from geomind_lab.mimic.utils.benchmark_manifest_utils import (  # noqa: E402
    DISPATCH_STATUS_DECLARED_RUNNER_UNIMPLEMENTED,
    RUNNER_DISPATCH_MODE_PERMISSIVE,
    RUNNER_DISPATCH_MODE_STRICT,
    plan_runner_dispatch,
    resolve_task_entries,
    split_task_names_by_runner,
)


class TestBenchmarkManifestUtils(unittest.TestCase):
    def test_reindex_extreme_resolves_block_backed_entries_across_versions(self) -> None:
        entries = resolve_task_entries("reindex_extreme")
        self.assertTrue(entries)
        self.assertTrue(any(entry.get("task_id") == "reindex_extreme" for entry in entries))
        self.assertTrue(any(bool(entry.get("is_runnable")) for entry in entries))
        layout_keys = {str(entry.get("layout_key")) for entry in entries}
        self.assertIn("state_reindex_blocks_line_4", layout_keys)
        self.assertIn("state_reindex_blocks_line_4_tight", layout_keys)
        stress_entries = [entry for entry in entries if str(entry.get("task_version")) == "stress"]
        self.assertEqual(
            {str(entry.get("layout_key")) for entry in stress_entries},
            {"state_reindex_blocks_line_4_tight"},
        )
        self.assertTrue(all(bool(entry.get("is_runnable")) for entry in stress_entries))
        self.assertTrue(all(str(entry.get("runner_family")) == "state_refresh" for entry in stress_entries))

    def test_reindex_insert_resolves_block_backed_entries_across_versions(self) -> None:
        entries = resolve_task_entries("reindex_insert")
        self.assertTrue(entries)
        layout_keys = {str(entry.get("layout_key")) for entry in entries}
        self.assertIn("state_reindex_insert_blocks_line_4", layout_keys)
        self.assertIn("state_reindex_insert_blocks_line_4_tight", layout_keys)
        by_layout = {str(entry.get("layout_key")): entry for entry in entries if str(entry.get("layout_key")).startswith("state_reindex_insert_blocks")}
        self.assertEqual(str(by_layout["state_reindex_insert_blocks_line_4"].get("task_version")), "base")
        self.assertEqual(str(by_layout["state_reindex_insert_blocks_line_4_tight"].get("task_version")), "stress")
        self.assertTrue(bool(by_layout["state_reindex_insert_blocks_line_4"].get("is_runnable")))
        self.assertTrue(bool(by_layout["state_reindex_insert_blocks_line_4_tight"].get("is_runnable")))
        self.assertTrue(all(str(entry.get("runner_family")) == "state_refresh" for entry in by_layout.values()))

    def test_reindex_subset_resolves_block_backed_entries_across_versions(self) -> None:
        entries = resolve_task_entries("reindex_subset")
        self.assertTrue(entries)
        layout_keys = {str(entry.get("layout_key")) for entry in entries}
        self.assertIn("state_reindex_red_subset_blocks_line_5", layout_keys)
        self.assertIn("state_reindex_red_subset_blocks_line_6_tight", layout_keys)
        by_layout = {str(entry.get("layout_key")): entry for entry in entries if str(entry.get("layout_key")).startswith("state_reindex_red_subset_blocks")}
        self.assertEqual(str(by_layout["state_reindex_red_subset_blocks_line_5"].get("task_version")), "base")
        self.assertEqual(str(by_layout["state_reindex_red_subset_blocks_line_6_tight"].get("task_version")), "stress")
        self.assertTrue(bool(by_layout["state_reindex_red_subset_blocks_line_5"].get("is_runnable")))
        self.assertTrue(bool(by_layout["state_reindex_red_subset_blocks_line_6_tight"].get("is_runnable")))
        self.assertTrue(all(str(entry.get("runner_family")) == "state_refresh" for entry in by_layout.values()))

    def test_ordinal_insert_static_resolves_annotation_backed_entries(self) -> None:
        entries = resolve_task_entries("ordinal_insert_static")
        self.assertTrue(entries)
        layout_keys = {str(entry.get("layout_key")) for entry in entries}
        self.assertIn("ordinal_insert_static_blocks_base", layout_keys)
        self.assertIn("ordinal_insert_static_blocks_stress", layout_keys)
        self.assertIn("ordinal_insert_static_blocks_skill", layout_keys)
        self.assertTrue(all(bool(entry.get("is_runnable")) for entry in entries))
        self.assertTrue(all(str(entry.get("runner_family")) == "state_refresh" for entry in entries))

    def test_grid_rowcol_grounding_resolves_annotation_backed_entries(self) -> None:
        entries = resolve_task_entries("grid_rowcol_grounding")
        self.assertTrue(entries)
        layout_keys = {str(entry.get("layout_key")) for entry in entries}
        self.assertIn("grid_rowcol_grounding_grid_2x3_base", layout_keys)
        self.assertIn("grid_rowcol_grounding_grid_2x3_stress", layout_keys)
        self.assertIn("grid_rowcol_grounding_grid_3x3_skill", layout_keys)
        self.assertTrue(all(bool(entry.get("is_runnable")) for entry in entries))
        self.assertTrue(all(str(entry.get("runner_family")) == "state_refresh" for entry in entries))

    def test_square_completion_resolves_annotation_backed_entries_across_versions(self) -> None:
        entries = resolve_task_entries("square_completion")
        self.assertEqual(
            {str(entry.get("layout_key")) for entry in entries},
            {
                "geometric_inference_square_completion_corner_3",
                "geometric_inference_square_completion_corner_3_skill",
                "geometric_inference_square_completion_tight_obstacle",
            },
        )
        by_version = {str(entry.get("task_version")): entry for entry in entries}
        self.assertTrue(bool(by_version["base"].get("is_runnable")))
        self.assertTrue(bool(by_version["skill"].get("is_runnable")))
        self.assertTrue(bool(by_version["stress"].get("is_runnable")))
        self.assertEqual(str(by_version["base"].get("runner_family")), "geometric_structure")
        self.assertEqual(str(by_version["skill"].get("runner_family")), "geometric_structure")
        self.assertEqual(str(by_version["stress"].get("runner_family")), "geometric_structure")

    def test_anchor_reference_distance_task_resolves_annotation_backed_preview_entries(self) -> None:
        entries = resolve_task_entries("grasp_anchor_distance")
        self.assertEqual(
            {str(entry.get("layout_key")) for entry in entries},
            {"grasp_anchor_distance_scene10_base", "grasp_anchor_distance_scene10_stress"},
        )
        self.assertTrue(all(bool(entry.get("preview_runnable")) for entry in entries))
        self.assertTrue(all(not bool(entry.get("is_runnable")) for entry in entries))
        self.assertTrue(all(str(entry.get("runner_family")) == "anchor_reference" for entry in entries))

    def test_point_symmetry_resolves_annotation_backed_entries_across_versions(self) -> None:
        entries = resolve_task_entries("point_symmetry")
        self.assertEqual(
            {str(entry.get("layout_key")) for entry in entries},
            {
                "geometric_inference_point_symmetry_line_2",
                "geometric_inference_point_symmetry_tight_obstacle",
            },
        )
        by_version = {str(entry.get("task_version")): entry for entry in entries}
        self.assertTrue(bool(by_version["base"].get("is_runnable")))
        self.assertTrue(bool(by_version["stress"].get("is_runnable")))
        self.assertEqual(str(by_version["base"].get("runner_family")), "geometric_structure")
        self.assertEqual(str(by_version["stress"].get("runner_family")), "geometric_structure")

    def test_trend_continuation_resolves_annotation_backed_entries_across_versions(self) -> None:
        entries = resolve_task_entries("trend_continuation")
        self.assertEqual(
            {str(entry.get("layout_key")) for entry in entries},
            {
                "geometric_inference_implicit_trend_line_3",
                "geometric_inference_implicit_trend_tight_obstacle",
            },
        )
        by_version = {str(entry.get("task_version")): entry for entry in entries}
        self.assertTrue(bool(by_version["base"].get("is_runnable")))
        self.assertTrue(bool(by_version["stress"].get("is_runnable")))
        self.assertEqual(str(by_version["base"].get("runner_family")), "geometric_structure")
        self.assertEqual(str(by_version["stress"].get("runner_family")), "geometric_structure")

    def test_weak_destruction_control_resolves_annotation_backed_entries_across_versions(self) -> None:
        entries = resolve_task_entries("weak_destruction_control")
        self.assertEqual(
            {str(entry.get("layout_key")) for entry in entries},
            {
                "geometric_inference_implicit_trend_weak_destruction_line_3",
                "geometric_inference_implicit_trend_weak_destruction_tight_obstacle",
            },
        )
        by_version = {str(entry.get("task_version")): entry for entry in entries}
        self.assertTrue(bool(by_version["control"].get("is_runnable")))
        self.assertTrue(bool(by_version["stress"].get("is_runnable")))
        self.assertEqual(str(by_version["control"].get("runner_family")), "geometric_structure")
        self.assertEqual(str(by_version["stress"].get("runner_family")), "geometric_structure")

    def test_constraint_selection_task_resolves_annotation_backed_preview_entries(self) -> None:
        entries = resolve_task_entries("select_visual_spatial")
        self.assertEqual(
            {str(entry.get("layout_key")) for entry in entries},
            {"select_visual_spatial_scene65_base", "select_visual_spatial_scene17_stress"},
        )
        self.assertTrue(all(bool(entry.get("preview_runnable")) for entry in entries))
        self.assertTrue(all(not bool(entry.get("is_runnable")) for entry in entries))
        self.assertTrue(all(str(entry.get("runner_family")) == "constraint_selection" for entry in entries))

    def test_exclusion_constraint_selection_task_resolves_annotation_backed_preview_entries(self) -> None:
        entries = resolve_task_entries("select_exclusion_unique")
        self.assertEqual(
            {str(entry.get("layout_key")) for entry in entries},
            {"select_exclusion_unique_scene65_base", "select_exclusion_unique_scene17_stress"},
        )
        self.assertTrue(all(bool(entry.get("preview_runnable")) for entry in entries))
        self.assertTrue(all(not bool(entry.get("is_runnable")) for entry in entries))
        self.assertTrue(all(str(entry.get("runner_family")) == "constraint_selection" for entry in entries))

    def test_runner_split_skips_registry_only_tasks_without_marking_unknown(self) -> None:
        grouped, unknown = split_task_names_by_runner(
            ["square_completion", "reindex_extreme", "reindex_insert", "reindex_subset"],
            phases=["phase1", "phase2"],
        )
        self.assertEqual(unknown, [])
        self.assertIn("reindex_extreme", grouped["state_refresh"])
        self.assertIn("reindex_insert", grouped["state_refresh"])
        self.assertIn("reindex_subset", grouped["state_refresh"])
        self.assertIn("square_completion", grouped["geometric_structure"])

    def test_dispatch_plan_surfaces_declared_unimplemented_families(self) -> None:
        dispatch = plan_runner_dispatch(
            ["square_completion", "place_directional", "reindex_extreme"],
            ["state_refresh_core_oracle"],
            phases=["phase1"],
            dispatch_mode=RUNNER_DISPATCH_MODE_PERMISSIVE,
        )
        self.assertEqual(dispatch["unknown_tasks"], [])
        self.assertEqual(dispatch["unknown_agents"], [])
        self.assertIn("reindex_extreme", dispatch["task_groups"]["state_refresh"])
        self.assertIn("square_completion", dispatch["task_groups"]["geometric_structure"])
        skipped = dispatch["skipped_task_descriptors"]
        self.assertTrue(any(entry.get("runner_family") == "spatial_placement" for entry in skipped))
        self.assertTrue(
            all(
                entry.get("dispatch_status") == DISPATCH_STATUS_DECLARED_RUNNER_UNIMPLEMENTED
                for entry in skipped
                if entry.get("runner_family") in {"routing", "spatial_placement"}
            )
        )

    def test_dispatch_plan_validates_mode_and_carries_strict_setting(self) -> None:
        with self.assertRaises(ValueError):
            plan_runner_dispatch(
                ["reindex_extreme"],
                ["state_refresh_core_oracle"],
                dispatch_mode="not-a-mode",
            )
        dispatch = plan_runner_dispatch(
            ["square_completion"],
            [],
            phases=["phase1"],
            dispatch_mode=RUNNER_DISPATCH_MODE_STRICT,
        )
        self.assertEqual(dispatch["dispatch_mode"], RUNNER_DISPATCH_MODE_STRICT)
        self.assertFalse(dispatch["skipped_task_descriptors"])
        self.assertTrue(dispatch["runnable_task_descriptors"])


if __name__ == "__main__":
    unittest.main()
