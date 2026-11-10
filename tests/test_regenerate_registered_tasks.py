"""Batch invariants: registration identity, bounded work and honest completion."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from geomind.tools import regenerate_registered_tasks as batch


class RegenerateRegisteredTasksTest(unittest.TestCase):
    def test_selection_uses_exact_instances_without_placeholders_or_aug_duplicates(self):
        all_tasks = batch._selected_tasks(None)
        self.assertEqual(len(all_tasks), 20)
        self.assertIn("SelectByDistance_v0", all_tasks)
        self.assertIn("SelectByOrderLast_v0", all_tasks)
        self.assertNotIn("TrackObjectOriginClutter_v0", all_tasks)
        self.assertNotIn("TrackMovingObjectSem_v0", all_tasks)
        selected = batch._selected_tasks(["SelectByOrderLast_v0", "InferMidpoint_v0"])
        self.assertEqual(selected, ["InferMidpoint_v0", "SelectByOrderLast_v0"])
        for bad in (["unknown"], [], ["InferMidpoint_v0", "InferMidpoint_v0"]):
            with self.assertRaises(ValueError):
                batch._selected_tasks(bad)

    def test_gt_or_an_optimistic_robot_flag_cannot_count_as_action_success(self):
        self.assertFalse(batch._action_passed({"passed": True, "offline_audit": {"passed": True},
            "report": {"physical_execution_verified": True, "results": []}}, "PickPlaceObject"))
        positive = {"negative_control": False, "success": True,
                    "physical": {"action": "PickPlaceObject", "success": True, "frozen_ground_truth": {"type": "Point"}},
                    "independent_gt_check": {"success": True}}
        robot = {"passed": True, "offline_audit": {"passed": True},
                 "report": {"physical_execution_verified": True, "results": [positive]}}
        self.assertTrue(batch._action_passed(robot, "PickPlaceObject"))
        self.assertTrue(batch._runtime_gt_verified(robot))
        self.assertFalse(batch._action_passed(robot, "PickLiftSelectedObject"))
        wrong = deepcopy(robot); wrong["report"]["results"][0]["physical"]["success"] = False
        self.assertFalse(batch._action_passed(wrong, "PickPlaceObject"))
        wrong = deepcopy(robot); wrong["offline_audit"]["passed"] = False
        self.assertFalse(batch._action_passed(wrong, "PickPlaceObject"))
        wrong = deepcopy(robot); del wrong["report"]["results"][0]["physical"]["frozen_ground_truth"]
        self.assertFalse(batch._runtime_gt_verified(wrong))

    def test_robot_request_preserves_task_identity_and_robotiq_height(self):
        manifest = {"table": {"height_m": .93}, "scene": {"scene_id": 65}, "scene_usda": "/tmp/case/assembly/scene.usda"}
        plan = {"registration": {"response": "SelectObject", "task_id": "SelectByOrderSecond_v0"}}
        request = batch._execution_request(manifest, plan, 420, 6000, True)
        self.assertEqual(request["task"]["task_id"], "SelectByOrderSecond_v0")
        self.assertEqual(request["robot"], {"embodiment": "franka_robotiq", "base_position_w_m": [0., -.45, .93]})
        self.assertEqual(request["execution"]["action"], "PickLiftSelectedObject")
        self.assertEqual(request["execution"]["max_steps"], 6000)
        override = batch._execution_request(manifest, plan, 420, 6000, True, [.1, -.4, .9])
        self.assertEqual(override["robot"]["base_position_w_m"], [.1, -.4, .9])
        plan["registration"]["response"] = "ReportPoint"
        self.assertEqual(batch._execution_request(manifest, plan, 420, 6000, True)["execution"]["action"], "PickPlaceObject")

    def test_prepare_only_stops_before_physics_binding_and_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            plan = {"registration": {"task_id": "InferMidpoint_v0", "response": "ReportPoint"}, "source_hashes": {}}
            manifest = {"table": {"height_m": .93}, "scene": {"scene_id": 65}, "scene_usda": str(case / "assembly/scene.usda")}
            exported = {"report": manifest, "report_path": str(case / "assembly/manifest.json"), "exit_code": 0}
            with patch("geomind.episodes.registered_tasks.load_registration", return_value=plan), \
                 patch.object(batch, "_snapshot", return_value={}), \
                 patch.object(batch, "_finish_provenance", return_value={"passed": True}), \
                 patch.object(batch, "_worker", return_value=exported) as worker:
                result = batch._run_case("InferMidpoint_v0", case, scene_id=65, seed=0, timeout_s=420,
                    max_steps=6000, negative_controls=True, prepare_only=True, isaac_python=None, command=["test"])
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(worker.call_args.args[0], "export")
            self.assertEqual(result["status"], "prepared_physics_unexecuted")
            self.assertTrue(result["design_generated"])
            self.assertFalse(result["gt_verified"])
            self.assertFalse(result["task_action_physics_executed"])
            self.assertFalse(result["task_completed"])

    def test_failed_case_is_preserved_and_does_not_stop_the_requested_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            def fake_case(task_id, case, **kwargs):
                case.mkdir()
                row = {"task_id": task_id, "status": "export_failed", "output": str(case),
                       "task_completed": False, "task_action_physics_executed": False,
                       "failure": {"reason": "kept failure"}}
                batch._write(case / "result.json", row)
                return row
            with patch.object(batch, "_run_case", side_effect=fake_case) as run_case:
                report = batch.run_batch(root, tasks=["InferMidpoint_v0", "SelectByOrderFirst_v0"])
            self.assertEqual(run_case.call_count, 2)
            self.assertEqual(report["counts"]["completed_physics_passed"], 0)
            self.assertEqual(report["counts"]["failed_before_action"], 2)
            self.assertEqual(report["counts"]["action_unexecuted"], 2)
            self.assertTrue((root / "InferMidpoint_v0/result.json").is_file())
            self.assertTrue((root / "SelectByOrderFirst_v0/result.json").is_file())
            self.assertIn("kept failure", (root / "index.html").read_text())
            with self.assertRaises(FileExistsError):
                batch.run_batch(root, prepare_only=True)

    def test_invalid_budget_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bad"
            for options in ({"timeout_s": 421}, {"timeout_s": float("inf")}, {"max_steps": 20}, {"scene_id": 0},
                            {"robot_base": [0, 0]}, {"robot_base": [0, False, .9]}, {"robot_base": [0, 0, float("nan")]},
                            {"robot_base": [0, 0, 11]}):
                with self.assertRaises(ValueError):
                    batch.run_batch(root, **options)
                self.assertFalse(root.exists())

    def test_prepare_only_exit_requires_a_valid_source_snapshot(self):
        report = {"counts": {"processed": 1, "requested": 1},
                  "cases": [{"status": "prepared_physics_unexecuted", "provenance_check": {"passed": False}}]}
        with patch.object(batch, "run_batch", return_value=report):
            self.assertEqual(batch.main(["--output", "/tmp/unused", "--prepare-only"]), 1)
        report["cases"][0]["provenance_check"]["passed"] = True
        with patch.object(batch, "run_batch", return_value=report):
            self.assertEqual(batch.main(["--output", "/tmp/unused", "--prepare-only"]), 0)
        report["cases"][0]["status"] = "provenance_failed"
        with patch.object(batch, "run_batch", return_value=report):
            self.assertEqual(batch.main(["--output", "/tmp/unused", "--prepare-only"]), 1)

    def test_layout_scale_flag_reaches_the_batch_and_export_worker(self):
        report = {"counts": {"processed": 1, "requested": 1},
                  "cases": [{"status": "prepared_physics_unexecuted", "provenance_check": {"passed": True}}]}
        with patch.object(batch, "run_batch", return_value=report) as run:
            self.assertEqual(batch.main(["--output", "/tmp/unused", "--prepare-only", "--layout-scale", "2.5"]), 0)
            self.assertEqual(run.call_args.kwargs["layout_scale"], 2.5)
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            batch._write(case / "registered_plan.json", {"registration": {"scene_id": 65}, "seed": 0, "layout_scale": 2.5})
            with patch("geomind.scene_assembly.registered_export.export_registered") as export:
                self.assertEqual(batch.main(["--worker", "export", "--case-dir", str(case)]), 0)
                self.assertEqual(export.call_args.kwargs["layout_scale"], 2.5)

    def test_layout_scale_invalid_values_are_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            for factor in (0, -1, True, float("inf"), float("nan")):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    batch.run_batch(root, layout_scale=factor)
                self.assertFalse(root.exists())

    def test_prepare_only_retains_agent_transform_in_case_plan_and_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            plan = {"registration": {"task_id": "SelectByAdjacency_v0", "response": "SelectObject"}, "source_hashes": {}}
            transform = {"layout_scale": 2.5, "randomizer_additional_calls": 0, "reason": "Robotiq clearance"}
            manifest = {"table": {"height_m": .93}, "scene": {"scene_id": 65},
                        "scene_usda": str(case / "assembly/scene.usda"), "agent_layout_transform": transform}
            exported = {"report": manifest, "report_path": str(case / "assembly/manifest.json"), "exit_code": 0}
            with patch("geomind.episodes.registered_tasks.load_registration", return_value=plan), \
                 patch.object(batch, "_snapshot", return_value={}), \
                 patch.object(batch, "_finish_provenance", return_value={"passed": True}), \
                 patch.object(batch, "_worker", return_value=exported) as worker:
                result = batch._run_case("SelectByAdjacency_v0", case, scene_id=65, seed=0, timeout_s=420,
                    max_steps=6000, negative_controls=True, prepare_only=True, isaac_python=None, command=["test"], layout_scale=2.5)
            self.assertEqual(worker.call_count, 1)
            self.assertFalse(result["task_completed"])
            self.assertEqual(batch._load(case / "registered_plan.json")["agent_layout_transform"], transform)
            self.assertEqual(batch._load(case / "request.json")["layout_scale"], 2.5)
    def test_a_complete_report_before_deadline_survives_bounded_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            batch._write(case / "assembly/physics_report.json", {"run_token": "token", "executed": True,
                "passed": True, "physics_steps": 600})
            process = SimpleNamespace(pid=123, returncode=None)
            process.poll = lambda: process.returncode
            process.wait = lambda timeout=None: process.returncode
            def stop(pid, sig):
                process.returncode = -15
            with patch("geomind.tools.assemble_scene._python_environment", return_value=("python", {})), \
                 patch.object(batch.uuid, "uuid4", return_value=SimpleNamespace(hex="token")), \
                 patch.object(batch.subprocess, "Popen", return_value=process), \
                 patch.object(batch.os, "killpg", side_effect=stop), \
                 patch.object(batch.time, "monotonic", side_effect=[0., 419.9, 420.01, 420.02]):
                result = batch._worker("physics", case, 420)
            self.assertTrue(result["bounded_cleanup_after_report"])
            self.assertTrue(result["report"]["passed"])
            self.assertEqual(result["exit_code"], -15)

    def test_deadline_without_a_complete_report_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            # A stale/incomplete report cannot authorize renderer cleanup as a pass.
            batch._write(case / "assembly/physics_report.json", {"run_token": "token"})
            process = SimpleNamespace(pid=123, returncode=None)
            process.poll = lambda: process.returncode
            process.wait = lambda timeout=None: process.returncode
            def stop(pid, sig):
                process.returncode = -15
            with patch("geomind.tools.assemble_scene._python_environment", return_value=("python", {})), \
                 patch.object(batch.uuid, "uuid4", return_value=SimpleNamespace(hex="token")), \
                 patch.object(batch.subprocess, "Popen", return_value=process), \
                 patch.object(batch.os, "killpg", side_effect=stop), \
                 patch.object(batch.time, "monotonic", side_effect=[0., 420.01]):
                with self.assertRaisesRegex(TimeoutError, "exceeded"):
                    batch._worker("physics", case, 420)


if __name__ == "__main__":
    unittest.main()
