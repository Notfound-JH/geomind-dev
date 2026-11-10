"""CPU adapter tests. Synthetic movement samples are not physical evidence."""
from copy import deepcopy
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from geomind.episodes.registered_tasks import (
    INITIAL_REFERENCE, REGISTERED_TASK_IDS, is_registered, load_registration,
    plan_temporal, public_instruction, signature, solve_registered, task_reference_view,
    validate_parameters, validate_relation_ground_truth,
)
from geomind.episodes.resolver import ContractError, Snapshot


class RegisteredTaskGTTests(unittest.TestCase):
    def fixture(self, task_id, seed=2):
        registration = load_registration(task_id, seed=seed)
        spec = registration["task_spec"]
        actual = {obj["name"]: [*obj["pose"]["xy"], .82] for obj in spec["runtime_objects"]}
        # Deliberately unrelated opaque order: names never enter public IDs.
        mapping = {f"opaque_{len(actual)-index:02d}": name for index, name in enumerate(actual)}
        temporal = plan_temporal(spec, actual, seed)
        params = {"frame": "world", "registered_task_id": task_id,
                  "legacy_task_spec": spec, "object_id_map": mapping,
                  "temporal_plan": temporal, "goal_z_m": .83}
        if temporal["required"]:
            params.update(objects=list(mapping), reference={"phase": "observe", "boundary": "End"})
        resolved = SimpleNamespace(task={"task_id": task_id},
            episode={"parameters": params, "seed": seed},
            scene={"workspace": {"frame": "world"}},
            layout={"objects": [{"id": alias} for alias in mapping]},
            protocol={"phases": [{"id": "observe", "type": "Observe", "duration_steps": 1},
                                  {"id": "delay", "type": "Delay", "duration_steps": 10},
                                  {"id": "respond", "type": "Respond"}]} if temporal["required"] else None)
        snapshot = Snapshot("cpu_fixture", "a" * 64, "world", 1,
                            {alias: tuple(actual[name]) for alias, name in mapping.items()})
        return registration, resolved, snapshot

    def measured_after_planned_motion(self, resolved, initial):
        """Synthetic endpoint solely for CPU contract tests, never a physics run."""
        mapping = resolved.episode["parameters"]["object_id_map"]
        temporal = resolved.episode["parameters"]["temporal_plan"]
        actual = {name: list(initial.positions_m[alias]) for alias, name in mapping.items()}
        for name, xy in temporal["actual_movement"]["waypoints"][-1]["positions_xy_m"].items():
            actual[name][:2] = xy
        return Snapshot(initial.instance_id, initial.contract_hash, initial.frame, 30,
                        {alias: tuple(actual[name]) for alias, name in mapping.items()})

    def test_exact_twenty_keep_public_taxonomy_and_conditions(self):
        self.assertEqual(len(REGISTERED_TASK_IDS), 20)
        self.assertEqual(len(set(REGISTERED_TASK_IDS)), 20)
        plans = [load_registration(name) for name in REGISTERED_TASK_IDS]
        self.assertEqual(sum(plan["temporal_required"] for plan in plans), 7)
        self.assertEqual([plan["registration"]["task_id"] for plan in plans], list(REGISTERED_TASK_IDS))
        self.assertEqual(signature("UpdateObjectOrder_v0"), ("SpatialMemory", "SceneUpdating", "SelectObject"))
        self.assertEqual(signature("TrackMovingObject_v0"), ("SpatialMemory", "IdentityTracking", "SelectObject"))
        self.assertEqual(signature("SelectByOrderFirst_v0"), ("SpatialReasoning", "TargetIdentification", "SelectObject"))
        self.assertEqual(load_registration("SelectByDistance_v0")["registration"]["condition"], "Semantic")
        for plan in plans:
            self.assertTrue(plan["source_hashes"])
            self.assertTrue(all(len(digest) == 64 for digest in plan["source_hashes"].values()))
            json.dumps(plan, allow_nan=False)
        self.assertFalse(is_registered("SelectByOrder"))
        self.assertFalse(is_registered({}))
        with self.assertRaises(ContractError):
            load_registration("SelectByOrder")

    def test_compilation_is_deterministic_and_ignores_ambient_layout_flags(self):
        baseline = load_registration("SelectByGridIndex_v0", seed=12)
        with patch.dict(os.environ, {"GEOMIND_LAYOUT_JITTER_ENABLE": "1",
                                    "GEOMIND_LAYOUT_JITTER_SEED": "987",
                                    "GEOMIND_LAYOUT_SHIFT_ENABLE": "1"}):
            again = load_registration("SelectByGridIndex_v0", seed=12)
            self.assertEqual(os.environ["GEOMIND_LAYOUT_JITTER_SEED"], "987")
        self.assertEqual(baseline, again)
        self.assertNotIn("applied_layout_randomization", again["task_spec"])

    def test_authored_bowl_plate_and_material_cues_survive_compilation(self):
        for task_id, name, asset in (("SelectByDirection_v0", "anchor_bowl", "bowl_1"),
                                    ("SelectAfterExclusion_v0", "blue_plate", "plate_2")):
            plan = load_registration(task_id)
            obj = next(obj for obj in plan["task_spec"]["runtime_objects"] if obj["name"] == name)
            self.assertEqual(obj["kind"], "usd")
            self.assertEqual(obj["asset_key"], asset)
            self.assertEqual(obj["material_override"]["diffuse_color"], [.12, .28, .82])
            self.assertTrue(plan["asset_preserving"]["authored_materials_preserved"])
            self.assertFalse(plan["asset_preserving"]["object_mode_conversion_applied"])
        plan = load_registration("SelectWithVisualCue_v0")
        objects = {obj["name"]: obj for obj in plan["task_spec"]["runtime_objects"]}
        self.assertEqual(objects["wood_cylinder_target"]["visual_material"]["type"], "mdl")
        self.assertNotEqual(objects["wood_cylinder_target"]["visual_material"], objects["plain_brown_cylinder_oblique"]["visual_material"])

    def test_public_task_reference_preserves_the_distinct_legacy_axis_signs(self):
        order = ("SelectByOrderFirst_v0", "SelectByOrderSecond_v0", "SelectByOrderLast_v0",
                 "UpdateObjectOrder_v0", "TrackObjectSwap_v0")
        for name in order:
            spec = load_registration(name)["task_spec"]
            reference = task_reference_view(spec)
            self.assertEqual(reference["left_axis"], "-y")
            self.assertEqual(reference["right_axis"], "+y")
            self.assertIn("世界Y较小端", public_instruction(spec))
            self.assertNotIn("世界Y较小端", spec["instruction"])
        for name in ("SelectByDirection_v0", "SelectWithVisualCue_v0"):
            spec = load_registration(name)["task_spec"]
            self.assertEqual(task_reference_view(spec)["right_axis"], "-y")
            self.assertIn("右方指世界Y减小", public_instruction(spec))
        exclusion = load_registration("SelectAfterExclusion_v0")["task_spec"]
        self.assertEqual(task_reference_view(exclusion)["left_axis"], "-x")
        grid = load_registration("SelectByGridIndex_v0")["task_spec"]
        self.assertEqual(task_reference_view(grid)["row_axis"], "+x")
        self.assertEqual(task_reference_view(grid)["column_axis"], "+y")
        self.assertIn("世界X从小到大", public_instruction(grid))
        adjacency = load_registration("SelectByAdjacency_v0")["task_spec"]
        self.assertEqual(task_reference_view(adjacency)["direction"], "visible_arrows")
        for name in ("TrackMovingObject_v0", "SelectWithinSubset_v0", "SelectByDistance_v0", "InferMidpoint_v0"):
            self.assertEqual(task_reference_view(load_registration(name)["task_spec"])["instruction_suffix"], "")

    def test_all_thirteen_static_contracts_solve_from_measured_opaque_objects(self):
        count = 0
        for task_id in REGISTERED_TASK_IDS:
            registration, resolved, current = self.fixture(task_id)
            if registration["temporal_required"]:
                continue
            with self.subTest(task_id=task_id):
                solved = solve_registered(resolved, current)
                gt = solved["ground_truth"]
                self.assertTrue(solved["instruction"])
                self.assertFalse(solved["calculation"]["planned_positions_used_as_observations"])
                if gt["type"] == "Object":
                    self.assertIn(gt["object_id"], current.positions_m)
                    self.assertTrue(gt["object_id"].startswith("opaque_"))
                else:
                    self.assertEqual(gt["position_m"][2], .83)
                    self.assertEqual(gt["frame"], "world")
                count += 1
        self.assertEqual(count, 13)

    def test_geometry_reacts_to_measured_anchor_translation(self):
        _, resolved, current = self.fixture("CompleteSquare_v0")
        first = solve_registered(resolved, current)
        translated = Snapshot(current.instance_id, current.contract_hash, current.frame, 2,
            {name: (point[0] + .07, point[1] - .03, point[2]) for name, point in current.positions_m.items()})
        second = solve_registered(resolved, translated)
        self.assertAlmostEqual(second["ground_truth"]["position_m"][0] - first["ground_truth"]["position_m"][0], .07)
        self.assertAlmostEqual(second["ground_truth"]["position_m"][1] - first["ground_truth"]["position_m"][1], -.03)

    def test_all_seven_dynamic_tasks_require_history_and_completed_measured_intervention(self):
        count = 0
        for task_id in REGISTERED_TASK_IDS:
            registration, resolved, initial = self.fixture(task_id)
            if not registration["temporal_required"]:
                continue
            with self.subTest(task_id=task_id):
                current = self.measured_after_planned_motion(resolved, initial)
                with self.assertRaisesRegex(ContractError, "history unavailable"):
                    solve_registered(resolved, current)
                unchanged = Snapshot(initial.instance_id, initial.contract_hash, initial.frame, 30, initial.positions_m)
                with self.assertRaisesRegex(ContractError, "invalid in measured"):
                    solve_registered(resolved, unchanged, {INITIAL_REFERENCE: initial})
                solved = solve_registered(resolved, current, {INITIAL_REFERENCE: initial})
                self.assertTrue(solved["calculation"]["terminal_validation"]["valid_episode"][0])
                self.assertIn(solved["ground_truth"]["object_id"], initial.positions_m)
                self.assertEqual(solved["calculation"]["initial_step"], 1)
                count += 1
        self.assertEqual(count, 7)

    def test_tracking_selects_initial_identity_rather_than_current_middle(self):
        _, resolved, initial = self.fixture("TrackMovingObject_v0")
        current = self.measured_after_planned_motion(resolved, initial)
        selected = solve_registered(resolved, current, {INITIAL_REFERENCE: initial})["ground_truth"]["object_id"]
        ordered_initial = sorted(initial.positions_m, key=lambda name: initial.positions_m[name][1])
        ordered_current = sorted(current.positions_m, key=lambda name: current.positions_m[name][1])
        self.assertEqual(selected, ordered_initial[1])
        self.assertNotEqual(selected, ordered_current[1])

    def test_grid_is_row_column_not_one_dimensional_rank(self):
        _, resolved, current = self.fixture("SelectByGridIndex_v0")
        selected = solve_registered(resolved, current)["ground_truth"]["object_id"]
        mapping = resolved.episode["parameters"]["object_id_map"]
        selected_name = mapping[selected]
        actual = {mapping[alias]: point for alias, point in current.positions_m.items()}
        first_row = sorted(actual, key=lambda name: actual[name][0])[:3]
        self.assertEqual(selected_name, sorted(first_row, key=lambda name: actual[name][1])[1])
        broken = dict(current.positions_m)
        point = broken[selected]
        broken[selected] = (point[0] + .06, point[1], point[2])
        with self.assertRaises(ContractError):
            solve_registered(resolved, Snapshot(current.instance_id, current.contract_hash, current.frame, 2, broken))

    def test_adjacency_keeps_directed_chain_and_order_variants_keep_rank(self):
        adjacency = load_registration("SelectByAdjacency_v0")["task_spec"]["protocol"]["gt_spec"]
        self.assertEqual(adjacency["resolver"], "structural_adjacency")
        self.assertEqual(adjacency["structure"]["type"], "DirectedChain")
        self.assertTrue(adjacency["structure"]["draw_edges"])
        first = load_registration("SelectByOrderFirst_v0")["task_spec"]["protocol"]
        last = load_registration("SelectByOrderLast_v0")["task_spec"]["protocol"]
        self.assertEqual(first["intervention"]["script_name"], "swap_adjacent_pairs_over_4")
        self.assertNotEqual(first["gt_spec"]["target_rank"], last["gt_spec"]["target_rank"])

    def test_topology_preserves_physical_relation_predicate_not_just_a_point(self):
        for task_id, resolver in (("PlaceByTopologyInside_v0", "containment_placement"),
                                  ("PlaceByTopologyOnTop_v0", "support_top_placement")):
            _, resolved, current = self.fixture(task_id)
            solved = solve_registered(resolved, current)
            relation = solved["calculation"]["relation_contract"]
            self.assertEqual(solved["ground_truth"]["type"], "Relation")
            self.assertEqual(solved["ground_truth"]["point_role"], "representative_action_goal")
            self.assertEqual(solved["ground_truth"]["contract"], relation)
            self.assertEqual(solved["ground_truth"]["reference_object_id"], solved["calculation"]["relation_reference_object"])
            self.assertEqual(relation["resolver"], resolver)
            self.assertTrue(relation["must_be_released"])
            self.assertEqual(relation["stable_after_settle_steps"], 24)
            self.assertIn("original relation predicate", solved["calculation"]["semantic_point_policy"])

    def test_relation_gt_rejects_schema_contract_alias_and_physical_target_drift(self):
        _, resolved, current = self.fixture("PlaceByTopologyInside_v0")
        gt = solve_registered(resolved, current)["ground_truth"]
        validate_relation_ground_truth(resolved, gt)
        mutations = []
        altered = deepcopy(gt); altered["unknown"] = True; mutations.append(altered)
        altered = deepcopy(gt); altered["contract"]["max_floor_gap"] *= 2; mutations.append(altered)
        altered = deepcopy(gt); altered["target_object_id"] = altered["reference_object_id"]; mutations.append(altered)
        altered = deepcopy(gt); altered["reference_object_id"] = "missing_alias"; mutations.append(altered)
        altered = deepcopy(gt); altered["frame"] = "table"; mutations.append(altered)
        altered = deepcopy(gt); altered["position_m"][0] = float("nan"); mutations.append(altered)
        for altered in mutations:
            with self.subTest(altered=altered):
                with self.assertRaises(ContractError):
                    validate_relation_ground_truth(resolved, altered)
        resolved.episode["physical_execution"] = {"action": "PickPlaceObject", "moving_object": gt["reference_object_id"]}
        with self.assertRaisesRegex(ContractError, "Physical moving object"):
            validate_relation_ground_truth(resolved, gt)

    def test_invalid_bindings_history_and_temporal_evidence_are_rejected(self):
        _, resolved, initial = self.fixture("UpdateObjectOrder_v0")
        current = self.measured_after_planned_motion(resolved, initial)
        other = Snapshot("another_run", initial.contract_hash, initial.frame, initial.step, initial.positions_m)
        with self.assertRaisesRegex(ContractError, "instance_id mismatch"):
            solve_registered(resolved, current, {INITIAL_REFERENCE: other})
        params = deepcopy(resolved.episode["parameters"])
        params["object_id_map"][next(iter(params["object_id_map"]))] = "missing_name"
        with self.assertRaisesRegex(ContractError, "bijectively"):
            validate_parameters("UpdateObjectOrder_v0", params)
        params = deepcopy(resolved.episode["parameters"])
        params["temporal_plan"]["planned_event_is_ground_truth"] = True
        with self.assertRaisesRegex(ContractError, "cannot be GT"):
            validate_parameters("UpdateObjectOrder_v0", params)
        params = deepcopy(resolved.episode["parameters"])
        params.pop("reference")
        with self.assertRaisesRegex(ContractError, "observe.End"):
            validate_parameters("UpdateObjectOrder_v0", params)

    def test_history_must_be_exactly_observe_end_and_query_not_before_respond(self):
        _, resolved, initial = self.fixture("UpdateObjectOrder_v0")
        current = self.measured_after_planned_motion(resolved, initial)
        early_initial = Snapshot(initial.instance_id, initial.contract_hash, initial.frame, 0, initial.positions_m)
        with self.assertRaisesRegex(ContractError, "exactly at Observe.End"):
            solve_registered(resolved, current, {INITIAL_REFERENCE: early_initial})
        early_query = Snapshot(current.instance_id, current.contract_hash, current.frame, 5, current.positions_m)
        with self.assertRaisesRegex(ContractError, "before Respond"):
            solve_registered(resolved, early_query, {INITIAL_REFERENCE: initial})
        resolved.protocol = None
        with self.assertRaisesRegex(ContractError, "native temporal protocol"):
            solve_registered(resolved, current, {INITIAL_REFERENCE: initial})


if __name__ == "__main__":
    unittest.main()
