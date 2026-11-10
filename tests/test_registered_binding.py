"""Synthetic CPU evidence fixtures; this file does not run or claim PhysX."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geomind.episodes.registered_tasks import load_registration
from geomind.episodes.resolver import ContractError, EpisodeResolver, Snapshot, solve
from geomind.episodes.runtime import SemanticEpisode, evaluate_answer
from geomind.episodes.physical_contract import physics_profile
from geomind.scene_assembly.audit import measured_bounds
from geomind.scene_assembly.registered_binding import bind_and_solve
from geomind.scene_assembly.robot_task_config import prepare_robot_configs, temporal_execution_timing


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RegisteredBindingTests(unittest.TestCase):
    def fixture(self, root, task_id):
        source = root / "assembly"; source.mkdir()
        plan = load_registration(task_id)
        spec = plan["task_spec"]
        objects = []
        metrics = []
        clouds = {}
        raw = np.zeros((600, len(spec["runtime_objects"]), 13)); raw[:, :, 3] = 1
        for index, descriptor in enumerate(spec["runtime_objects"]):
            # The offset/asymmetric clouds exercise body-specific centre binding.
            sx = .012 + index * .001; sy = .011 + index * .001; sz = .015
            cloud = np.asarray(list(itertools.product((-.5*sx, 1.5*sx), (-sy, sy), (-sz, sz))))
            key = f"points_m_{index:03d}"; clouds[key] = cloud
            position = [*descriptor["pose"]["xy"], .6 + sz]
            path = f"/World/AssembledObjects/{descriptor['name']}"
            raw[:, index, :3] = position
            low, high = measured_bounds(cloud, position, [1., 0., 0., 0.])
            objects.append({"object_id": descriptor["name"], "prim_path": path,
                "asset_name": descriptor.get("asset_key", descriptor["kind"]),
                "body_points_key": key, "position_w_m": position,
                "orientation_wxyz": [1., 0., 0., 0.], "bbox_center_xy_m": ((low+high)*.5)[:2].tolist()})
            metrics.append({"prim_path": path, "measured_bbox_min_w_m": low.tolist(), "measured_bbox_max_w_m": high.tolist()})
        clouds["points_m"] = clouds["points_m_000"]
        np.savez_compressed(source / "asset_body_vertices.npz", **clouds)
        columns = ["x","y","z","qw","qx","qy","qz","vx","vy","vz","wx","wy","wz"]
        np.savez_compressed(source / "object_states.npz", data=raw, dt_s=1/120,
                            columns=columns, prim_paths=[obj["prim_path"] for obj in objects])
        (source / "scene.usda").write_text("#usda 1.0\n# CPU fixture; not a simulated scene.\n")
        for name in ("preview.png", "detail_preview.png", "settling.mp4"):
            (source / name).write_bytes(b"Synthetic CPU test artifact, not rendered physics evidence")
        hashes = {name: digest(source/name) for name in ("preview.png", "detail_preview.png", "settling.mp4", "object_states.npz", "asset_body_vertices.npz")}
        report = {"executed": True, "passed": True, "backend": "isaacsim_physx", "kind": "initial_layout_settling",
            "physics_steps": 600, "warmup_steps": 20, "stable_steps": 600, "dt_s": 1/120,
            "object_count": len(objects), "objects": metrics, "state_trace": str(source/"object_states.npz"), "sha256": hashes}
        offline = {"passed": True, "stable_steps_recomputed": 600, "offline_wrong_layout_control_rejected": True}
        manifest = {"kind": "initial_scene_assembly", "status": "assembled_physics_passed",
            "registration": plan["registration"], "registered_task_spec": spec,
            "request": {"object": {"count": len(objects)}, "seed": 0}, "asset": {"name": "registered_runtime"},
            "objects": objects, "physics": report, "offline_audit": offline,
            "table": {"height_m": .6, "bounds_xy_m": [-1., 1., -1., 1.]},
            "scene_usda": str(source/"scene.usda"), "asset_body_vertices": str(source/"asset_body_vertices.npz"),
            "provenance": {"scene_usda_sha256": digest(source/"scene.usda"),
                "asset_body_vertices_sha256": hashes["asset_body_vertices.npz"],
                "registered_sources_sha256": plan["source_hashes"]}}
        write(source/"physics_report.json", report); write(source/"offline_audit.json", offline)
        write(source/"manifest.json", manifest)
        return source/"manifest.json", manifest

    def test_static_mixed_clouds_bind_measured_centres_and_write_replayable_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest_path, _ = self.fixture(root, "CompleteSquare_v0")
            original_hash = digest(manifest_path)
            output = root/"binding"
            result = bind_and_solve(manifest_path, output)
            self.assertEqual(result["kind"], "assembled_task_binding")
            self.assertEqual(result["status"], "gt_solved_task_action_unverified")
            self.assertFalse(result["gt_pending_runtime"])
            self.assertFalse(result["task_completed"])
            self.assertTrue(Path(result["ground_truth_path"]).is_file())
            self.assertTrue(result["checks"]["wrong_answer_rejected"])
            self.assertIsNone(result["checks"]["independent_gt_agrees"])
            first = result["objects"][0]
            self.assertAlmostEqual(first["position_m"][0] - first["rigid_origin_world_m"][0], .006)
            self.assertEqual(len(result["source"]["identity_mapping"]["semantic_to_legacy"]), 4)
            resolved = EpisodeResolver(output/"configs").resolve("assembled_task")
            snapshot = Snapshot.from_dict(json.loads((output/"evaluator/query_snapshot.json").read_text()))
            self.assertEqual(solve(resolved, snapshot)["ground_truth"], result["ground_truth"])
            self.assertEqual(digest(manifest_path), original_hash)
            self.assertEqual(json.loads((output/"task_result.json").read_text())["ground_truth"], result["ground_truth"])
            public = json.loads((output/"observations/initial.json").read_text())
            self.assertFalse({"parameters", "operation", "ground_truth", "source", "temporal_plan"} & public.keys())
            self.assertNotIn("held block", public["instruction"])
            self.assertFalse(any("_anchor_" in obj["label"] or "_target" in obj["label"] for obj in public["objects"]))
            self.assertEqual(public["axes"], {"x": "+x", "y": "+y", "z": "+z"})

    def test_relation_commands_are_deferred_and_cpu_never_claims_task_success(self):
        for task_id in ("PlaceByTopologyInside_v0", "PlaceByTopologyOnTop_v0"):
            with self.subTest(task_id=task_id), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);manifest_path,_=self.fixture(root,task_id)
                result=bind_and_solve(manifest_path,root/"binding")
                self.assertEqual(result["ground_truth"]["type"],"Relation")
                self.assertIsNone(result["checks"]["wrong_answer_rejected"])
                self.assertNotIn("semantic_wrong_answer",result["checks"])
                self.assertFalse(result["checks"]["command_diagnostics_are_semantic_task_acceptance"])
                self.assertTrue((root/"binding/evaluator/command_diagnostics.json").is_file())
                resolved=EpisodeResolver(root/"binding/configs").resolve("assembled_task")
                solved=json.loads((root/"binding/evaluator/ground_truth.json").read_text())
                point=list(solved["ground_truth"]["position_m"]);point[0]+=.03
                answer={"position_m":point,"frame":"world"}
                parsed=evaluate_answer(resolved,solved,answer)
                self.assertTrue(parsed["command_valid"])
                self.assertTrue(parsed["semantic_deferred_to_relation"])
                self.assertIsNone(parsed["score"])
                for invalid in ({"position_m":[2.,0.,.6],"frame":"world"},
                                {"position_m":point,"frame":"table"},
                                {"position_m":[float("nan"),0.,.6],"frame":"world"},
                                {**answer,"extra":True}):
                    with self.assertRaises(ContractError):evaluate_answer(resolved,solved,invalid)
                env=SemanticEpisode(resolved);env.reset()
                _,reward,done,_,info=env.step(answer)
                self.assertTrue(done);self.assertEqual(reward,0.)
                self.assertIsNone(info["success"]);self.assertIsNone(info["task_success"])
                self.assertFalse(info["task_completed"]);self.assertFalse(info["physical_execution"])

    def test_all_seven_robot_protocols_restore_materialized_two_second_command_horizon(self):
        tasks=("SelectByAdjacency_v0","SelectByOrderFirst_v0","SelectByOrderLast_v0",
               "SelectWithinSubset_v0","TrackMovingObject_v0","TrackObjectSwap_v0","UpdateObjectOrder_v0")
        with tempfile.TemporaryDirectory() as directory:
            for task_id in tasks:
                with self.subTest(task_id=task_id):
                    root=Path(directory)/task_id;root.mkdir()
                    manifest_path,_=self.fixture(root,task_id)
                    bind_and_solve(manifest_path,root/"binding")
                    prepared=prepare_robot_configs(root/"binding",
                        {"embodiment":"franka_robotiq","base_position_w_m":[-.12,.32,.6]},
                        {"action":"PickLiftSelectedObject"},root/"binding/robot_configs")
                    resolved=EpisodeResolver(prepared["root"]).resolve("assembled_task")
                    timing=json.loads(Path(prepared["binding_path"]).read_text())["temporal_execution_timing"]
                    self.assertAlmostEqual(resolved.scene["physics"]["protocol_step_s"],1/6)
                    self.assertAlmostEqual(resolved.scene["physics"]["dt_s"],1/240)
                    self.assertEqual(timing["physics_ticks_per_protocol_step"],40)
                    self.assertAlmostEqual(timing["physics_ticks_per_protocol_step"]*
                        timing["command_horizon_steps"]*resolved.scene["physics"]["dt_s"],2.)
                    self.assertEqual(timing["command_horizon_s"],2.)
                    self.assertFalse(timing["measured_motion_duration_verified"])
                    expected="explicit_legacy_intervention" if task_id=="TrackMovingObject_v0" else "materialized_legacy_default"
                    self.assertEqual(timing["duration_source"],expected)
                    changed=deepcopy(resolved)
                    changed.episode["parameters"]["legacy_task_spec"]["protocol"]["intervention"]["duration_s"]=2.001
                    with self.assertRaisesRegex(ValueError,"integer number"):
                        temporal_execution_timing(changed,physics_profile())

    def test_dynamic_binding_never_constructs_gt_from_planned_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest_path, _ = self.fixture(root, "UpdateObjectOrder_v0")
            output = root/"binding"
            result = bind_and_solve(manifest_path, output)
            self.assertEqual(result["status"], "gt_pending_runtime_task_action_unverified")
            self.assertTrue(result["gt_pending_runtime"])
            self.assertIsNone(result["ground_truth"])
            self.assertIsNone(result["ground_truth_path"])
            self.assertFalse((output/"evaluator/ground_truth.json").exists())
            self.assertFalse((output/"evaluator/query_snapshot.json").exists())
            self.assertTrue((output/"evaluator/initial_snapshot.json").is_file())
            resolved = EpisodeResolver(output/"configs").resolve("assembled_task")
            self.assertEqual(resolved.task["domain"], "SpatialMemory")
            self.assertEqual(resolved.protocol["phases"][0]["id"], "observe")
            events = resolved.protocol["phases"][1]["events"]
            self.assertEqual(len(events), 12)
            self.assertEqual({event["type"] for event in events}, {"MoveObject"})
            snapshot = Snapshot.from_dict(json.loads((output/"evaluator/initial_snapshot.json").read_text()))
            with self.assertRaisesRegex(ContractError, "history unavailable"):
                solve(resolved, snapshot)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                bind_and_solve(manifest_path, output)

    def test_tracking_exposes_the_actual_scene_motion_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest_path, _ = self.fixture(root, "TrackMovingObject_v0")
            result = bind_and_solve(manifest_path, root/"binding")
            resolved = EpisodeResolver(result["native_config_root"]).resolve(result["episode_id"])
            self.assertTrue(resolved.protocol["phases"][1]["scene_visible"])
            self.assertEqual(len(resolved.protocol["phases"][1]["events"]), 24)
            self.assertTrue(result["gt_pending_runtime"])

    def test_native_registered_history_provenance_hashes_the_referenced_snapshot(self):
        # Symbolic CPU commands test provenance only, not a physical execution.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest_path,_=self.fixture(root,"UpdateObjectOrder_v0")
            bound=bind_and_solve(manifest_path,root/"binding")
            resolved=EpisodeResolver(bound["native_config_root"]).resolve(bound["episode_id"])
            env=SemanticEpisode(resolved);observation,_=env.reset()
            while not observation["allow_response"]:
                observation,*_=env.step()
            initial=env.history["observe.End"];current=env.snapshot()
            fingerprint=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
            provenance=env.result["provenance"]
            self.assertEqual(provenance["reference"],"observe.End")
            self.assertEqual(provenance["evidence_step"],initial.step)
            self.assertEqual(provenance["evidence_step"],1)
            self.assertEqual(provenance["query_step"],current.step)
            self.assertEqual(provenance["evidence_hash"],fingerprint(initial.to_dict()))
            self.assertEqual(provenance["current_state_hash"],fingerprint(current.to_dict()))
            self.assertNotEqual(provenance["evidence_hash"],provenance["current_state_hash"])

    def test_hash_modified_raw_geometry_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest_path, _ = self.fixture(root, "InferMidpoint_v0")
            (manifest_path.parent/"asset_body_vertices.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                bind_and_solve(manifest_path, root/"binding")


if __name__ == "__main__":
    unittest.main()
