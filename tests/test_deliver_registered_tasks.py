"""Offline delivery never replaces failures with old successes or rewrites raw evidence."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from geomind.tools import deliver_registered_tasks as delivery


class DeliveryTest(unittest.TestCase):
    def report(self):
        physical = {"success": True, "action": "PickPlaceObject", "command_finger_joint_names": ["finger_joint"],
                    "finger_joint_names": ["finger_joint", "right_outer_knuckle_joint"]}
        return {"physical_execution_verified": True, "results": [
            {"negative_control": False, "success": True, "expected_success": True, "verification_passed": True,
             "semantic": {"success": True}, "robot_name": "franka_robotiq", "physical": physical},
            {"negative_control": True, "success": False, "expected_success": False, "verification_passed": True,
             "physical": {"success": False}}]}

    def gates(self):
        return {"initial_verified": True, "report": self.report(), "new_audit_passed": True,
                "integrity": {"saved_source_snapshot_complete": True, "saved_config_hashes_complete": True,
                              "current_workspace_source_drift": [{"path": "audit.py"}]},
                "run_end_provenance_passed": True, "gt_verified": True}

    def test_latest_explicit_retry_wins_even_if_failed_or_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"; retry = Path(tmp) / "retry"
            (base / "InferMidpoint_v0").mkdir(parents=True)
            (retry / "InferMidpoint_v0").mkdir(parents=True)
            delivery._write(base / "InferMidpoint_v0/result.json", {"task_completed": True})
            roots, selected = delivery.select_versions(base, [retry])
            case = next(r for r in selected if r["task_id"] == "InferMidpoint_v0")
            self.assertEqual(case["selected"]["case"], str(retry / "InferMidpoint_v0"))
            self.assertEqual(len(case["versions"]), 2)
            self.assertEqual(len(selected), 20)
            with self.assertRaises(ValueError):
                delivery.select_versions(base, [base])

    def test_all_physical_and_integrity_gates_are_required_but_later_source_drift_is_separate(self):
        valid = self.gates()
        self.assertTrue(delivery.qualification(**valid)["task_completed"])
        for key in ("initial_verified", "new_audit_passed", "run_end_provenance_passed", "gt_verified", "identity_verified"):
            bad = deepcopy(valid); bad[key] = False
            self.assertFalse(delivery.qualification(**bad)["task_completed"], key)
        for key in ("saved_source_snapshot_complete", "saved_config_hashes_complete"):
            bad = deepcopy(valid); bad["integrity"][key] = False
            self.assertFalse(delivery.qualification(**bad)["task_completed"], key)
        bad = deepcopy(valid); bad["report"]["physical_execution_verified"] = False
        self.assertFalse(delivery.qualification(**bad)["task_completed"])
        bad = deepcopy(valid); bad["report"]["results"].pop()
        self.assertFalse(delivery.qualification(**bad)["task_completed"])
        bad = deepcopy(valid); bad["report"]["results"][1]["success"] = True
        self.assertFalse(delivery.qualification(**bad)["task_completed"])
        bad = deepcopy(valid); bad["report"]["results"][0]["robot_name"] = "franka"
        self.assertFalse(delivery.qualification(**bad)["task_completed"])

    def test_saved_source_integrity_does_not_hide_current_workspace_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"; case = Path(tmp) / "case"
            (repo / "module").mkdir(parents=True)
            (case / "source_snapshot/module").mkdir(parents=True)
            (repo / "module/code.py").write_text("old source")
            (case / "source_snapshot/module/code.py").write_text("old source")
            configs = {}
            for filename in ("request.json", "registered_plan.json", "task_result.json", "task_binding.json",
                             "configs/episodes/episode.json", "robot_configs/episodes/episode.json"):
                delivery._write(case / filename, {})
                configs[filename] = delivery._digest(case / filename)
            sources = [{"path": "module/code.py", "sha256": delivery._digest(repo / "module/code.py")}]
            provenance = {"source_files": sources, "source_snapshot_id": hashlib.sha256(delivery._encoded({r["path"]:r["sha256"] for r in sources})).hexdigest(),
                          "config_files": configs, "configuration_id": hashlib.sha256(delivery._encoded(configs)).hexdigest()}
            delivery._write(case / "provenance.json", provenance)
            self.assertTrue(delivery.saved_integrity(case, repository=repo)["current_workspace_matches"])
            (repo / "module/code.py").write_text("new auditor source")
            reviewed = delivery.saved_integrity(case, repository=repo)
            self.assertTrue(reviewed["saved_source_snapshot_complete"])
            self.assertTrue(reviewed["saved_config_hashes_complete"])
            self.assertFalse(reviewed["current_workspace_matches"])
            self.assertEqual(reviewed["current_workspace_source_drift"][0]["path"], "module/code.py")
            (case / "source_snapshot/module/code.py").write_text("tampered snapshot")
            self.assertFalse(delivery.saved_integrity(case, repository=repo)["saved_source_snapshot_complete"])
            (case / "request.json").write_text("tampered config")
            self.assertFalse(delivery.saved_integrity(case, repository=repo)["saved_config_hashes_complete"])

    def test_runtime_gt_comparison_is_typed_exact_for_ids_and_tightly_bounded_for_points(self):
        a = {"type": "Point", "position_m": [.5256578347235923, .1941252936625268, .9554389073171518], "frame": "world"}
        b = deepcopy(a); b["position_m"][1] = .19412529366252673
        same, delta = delivery._same_gt(a, b)
        self.assertTrue(same); self.assertLess(delta, 1e-15)
        bad = deepcopy(a); bad["position_m"][0] += 1e-6
        self.assertFalse(delivery._same_gt(a, bad)[0])
        bad = deepcopy(a); bad["frame"] = "robot"
        self.assertFalse(delivery._same_gt(a, bad)[0])
        bad = deepcopy(a); bad["unexpected"] = True
        self.assertFalse(delivery._same_gt(a, bad)[0])
        for invalid in (True, "0.525", float("inf")):
            bad = deepcopy(a); bad["position_m"][0] = invalid
            self.assertFalse(delivery._same_gt(a, bad)[0])
        self.assertFalse(delivery._same_gt({"type": "Object", "object_id": "obj_001"}, {"type": "Object", "object_id": "obj_002"})[0])

    def test_relation_gt_keeps_contract_ids_exact_and_does_not_define_a_success_region_by_point(self):
        relation = {"type": "Relation", "relation": "containment_placement", "target_object_id": "obj_001",
                    "reference_object_id": "obj_002", "contract": {"must_be_released": True, "max_floor_gap": .018},
                    "position_m": [.4, .1, .96], "frame": "world", "point_role": "representative_action_goal"}
        same = deepcopy(relation); same["position_m"][0] += .25e-12
        self.assertTrue(delivery._same_gt(relation, same)[0])
        for field, value in (("reference_object_id", "obj_003"), ("point_role", "unique_goal"),
                             ("relation", "support_top_placement"), ("contract", {"must_be_released": 1, "max_floor_gap": .018})):
            bad = deepcopy(relation); bad[field] = value
            self.assertFalse(delivery._same_gt(relation, bad)[0], field)
        # The comparison only checks frozen representative consistency. Actual
        # region membership is an independent raw-state auditor requirement.
        gates = self.gates(); gates["new_audit_passed"] = False
        self.assertFalse(delivery.qualification(**gates)["task_completed"])

    def test_framework_restore_removes_new_episode_files_instead_of_faking_implementation_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"; case = Path(tmp) / "case"
            source_data = {"geomind/episodes/__init__.py": "# saved package\n",
                           "geomind/episodes/resolver.py": "# saved resolver\n",
                           "geomind/episodes/registered_tasks.py": "# saved registered solver\n"}
            sources = []
            for filename, text in source_data.items():
                live = repo / filename; live.parent.mkdir(parents=True, exist_ok=True); live.write_text(text)
                saved = case / "source_snapshot" / filename; saved.parent.mkdir(parents=True, exist_ok=True); saved.write_text(text)
                sources.append({"path": filename, "sha256": delivery._digest(saved)})
            (repo / "geomind/episodes/new_module.py").write_text("# must not enter the old implementation hash\n")
            (repo / "geomind/episodes/resolver.py").write_text("# changed current resolver\n")
            fallback = repo / "geomind/core/registry.py"; fallback.parent.mkdir(parents=True); fallback.write_text("# explicit fallback\n")
            configs = {}
            for filename in ("request.json", "registered_plan.json", "task_result.json", "task_binding.json",
                             "configs/episodes/episode.json", "robot_configs/episodes/episode.json"):
                delivery._write(case / filename, {}); configs[filename] = delivery._digest(case / filename)
            source_hash = hashlib.sha256(delivery._encoded({row["path"]: row["sha256"] for row in sources})).hexdigest()
            delivery._write(case / "provenance.json", {"source_files": sources, "source_snapshot_id": source_hash,
                "config_files": configs, "configuration_id": hashlib.sha256(delivery._encoded(configs)).hexdigest()})
            dest = Path(tmp) / "restored"
            restored = delivery.prepare_framework(case, dest, repository=repo)
            self.assertFalse((dest / "geomind/episodes/new_module.py").exists())
            self.assertEqual((dest / "geomind/episodes/resolver.py").read_text(), source_data["geomind/episodes/resolver.py"])
            self.assertIn("geomind/core/registry.py", restored["current_fallback_files_sha256"])
            self.assertNotIn("geomind/episodes/resolver.py", restored["current_fallback_files_sha256"])
            self.assertFalse(restored["episode_hash_constant_overridden"])
            expected = hashlib.sha256(b"".join(Path(name).name.encode() + b"\0" + source_data[name].encode() + b"\0"
                                                for name in sorted(source_data))).hexdigest()
            self.assertEqual(restored["expected_original_episode_implementation_hash"], expected)

    def test_robot_auditor_writes_only_into_a_new_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"; original = case / "robot_physics"
            (original / "evidence").mkdir(parents=True)
            (original / "observations").mkdir()
            delivery._write(original / "report.json", {"raw": "unchanged"})
            delivery._write(original / "assembled_robot_audit.json", {"passed": False, "old": True})
            old_report = delivery._digest(original / "report.json")
            old_audit = delivery._digest(original / "assembled_robot_audit.json")
            target = Path(tmp) / "derived/robot"
            def fake_audit(configs, view):
                self.assertNotEqual(view, original)
                self.assertEqual((view / "evidence").resolve(), (original / "evidence").resolve())
                self.assertEqual((view / "report.json").read_bytes(), (original / "report.json").read_bytes())
                delivery._write(view / "assembled_robot_audit.json", {"passed": True})
                return {"passed": True}
            with patch("geomind.tools.audit_assembled_robot.audit", side_effect=fake_audit):
                self.assertTrue(delivery.audit_robot(case, target)["passed"])
            self.assertEqual(delivery._digest(original / "report.json"), old_report)
            self.assertEqual(delivery._digest(original / "assembled_robot_audit.json"), old_audit)
            self.assertTrue((target / "assembled_robot_audit.json").exists())

    def test_selected_request_metadata_and_initial_links_come_from_newest_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, retry = Path(tmp) / "base", Path(tmp) / "retry"
            for root, scene, seed, scale in ((base, 65, 7, 1.), (retry, 91, 0, 2.5)):
                case = root / "InferMidpoint_v0"
                delivery._write(case / "request.json", {"scene_id": scene, "seed": seed,
                    "layout_scale": scale, "robot": {"embodiment": "franka_robotiq"}})
                assembly = case / "assembly"
                for filename in ("preview.png", "detail_preview.png", "scene.usda"):
                    assembly.mkdir(parents=True, exist_ok=True)
                    (assembly / filename).write_bytes(b"original evidence")
                delivery._write(assembly / "manifest.json", {"request": {"scene_id": 1},
                    "preview": str(assembly / "preview.png"), "detail_preview": "detail_preview.png"})
                delivery._write(assembly / "physics_report.json", {"passed": False})
            _, selected = delivery.select_versions(base, [retry])
            row = next(item for item in selected if item["task_id"] == "InferMidpoint_v0")
            case = Path(row["selected"]["case"])
            before = {p: delivery._digest(p) for p in case.rglob("*") if p.is_file()}
            metadata, links = delivery._case_presentation(case, "sources/latest")
            self.assertEqual((metadata["scene_id"], metadata["seed"], metadata["layout_scale"], metadata["robot"]),
                             (91, 0, 2.5, "franka_robotiq"))
            self.assertEqual(metadata["sources"]["scene_id"], "request.json.scene_id")
            self.assertEqual(metadata["request_link"], "sources/latest/request.json")
            self.assertIn({"label": "初始桌面细节", "href": "sources/latest/assembly/detail_preview.png"}, links)
            self.assertIn({"label": "初始组装来源", "href": "sources/latest/assembly/manifest.json"}, links)
            self.assertTrue(all(item["href"].startswith("sources/latest/") for item in links))
            self.assertEqual(before, {p: delivery._digest(p) for p in case.rglob("*") if p.is_file()})

    def test_old_request_metadata_has_explicit_fallback_and_missing_case_stays_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            delivery._write(case / "request.json", {"seed": 0, "robot": "franka_robotiq"})
            delivery._write(case / "assembly/manifest.json", {"scene": {"scene_id": 65}})
            metadata, _ = delivery._case_presentation(case, "case")
            self.assertEqual(metadata["scene_id"], 65)
            self.assertEqual(metadata["sources"]["scene_id"], "assembly/manifest.json")
            self.assertEqual(metadata["layout_scale"], 1.)
            self.assertTrue(metadata["sources"]["layout_scale"].startswith("default 1"))
            unknown, links = delivery._case_presentation(Path(tmp) / "missing", "missing")
            self.assertIsNone(unknown["scene_id"])
            self.assertIsNone(unknown["layout_scale"])
            self.assertIsNone(unknown["robot"])
            self.assertEqual(links, [])

    def test_html_labels_metadata_and_privileged_actual_gt_without_plan_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp); case = output / "selected"
            video = case / "robot_physics/observer.mp4"
            video.parent.mkdir(parents=True); video.write_bytes(b"existing source video")
            common = {"task_completed": False, "task_action_physics_executed": True, "gt_verified": False,
                "selected": {"case": str(case)}, "source_link": "source/current", "versions": [],
                "integrity": {}, "qualification": {"failed_gates": ["positive_action_passed"]},
                "initial_ground_truth": {"type": "Object", "object_id": "PLAN_ONLY_DO_NOT_RENDER"}}
            actual = {**deepcopy(common), "task_id": "InferMidpoint_v0",
                "selected_request": {"scene_id": 91, "seed": 0, "layout_scale": 2.5, "robot": "franka_robotiq"},
                "initial_assembly_links": [{"label": "初始组装来源", "href": "source/current/assembly/manifest.json"}],
                "runtime_cases": [{"negative_control": False, "ground_truth": {"type": "Point", "position_m": [9.87654321, .123456789, .9999], "frame": "world"},
                                   "video": {"observer": str(video)}, "physical": {"position_error_m": .005260290963, "dwell_steps": 0}}]}
            absent = {**deepcopy(common), "task_id": "SelectWithVisualCue_v0", "runtime_cases": [],
                      "task_action_physics_executed": False}
            report = {"counts": {"completed_physics_passed": 0, "gt_verified": 0, "action_failed": 1, "action_unexecuted": 1},
                      "cases": [actual, absent]}
            delivery._make_html(report, output)
            page = (output / "delivery.html").read_text()
            self.assertIn("scene 91 · seed 0 · layout_scale 2.5 · robot franka_robotiq", page)
            self.assertIn("5.260 mm", page)
            self.assertIn("[9.877, 0.123, 1.000] m", page)
            self.assertIn("研究者特权：真实运行时GT与关系验收", page)
            self.assertLess(page.index("研究者特权："), page.index("9.87654321"))
            self.assertNotIn("PLAN_ONLY_DO_NOT_RENDER", page)
            self.assertIn("GT：未验证", page)
            self.assertIn("没有保存真实运行时GT；未从计划或初始绑定结果补值", page)
            self.assertIn("scene 未记录", page)
            self.assertIn("初始组装来源", page)
            self.assertIn("没有再次执行物理", page)
            self.assertEqual(video.read_bytes(), b"existing source video")
            for value in (None, True, float("nan"), float("inf")):
                self.assertEqual(delivery._distance_mm(value), "未记录")

    def test_initial_auditor_preserves_original_report_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"; original = case / "assembly"
            delivery._write(original / "manifest.json", {"physics": {"passed": True}})
            delivery._write(original / "physics_report.json", {"passed": True, "sha256": {"preview.png": "placeholder"}})
            delivery._write(original / "offline_audit.json", {"old": True})
            for name in ("asset_body_vertices.npz", "object_states.npz", "preview.png"):
                (original / name).write_bytes(b"raw evidence")
            old_audit = delivery._digest(original / "offline_audit.json")
            def fake_audit(view):
                self.assertEqual((view / "object_states.npz").resolve(), (original / "object_states.npz").resolve())
                delivery._write(view / "offline_audit.json", {"passed": True})
                return {"passed": True}
            with patch("geomind.scene_assembly.audit.audit", side_effect=fake_audit):
                self.assertTrue(delivery.audit_initial(case, Path(tmp) / "derived/initial")["passed"])
            self.assertEqual(delivery._digest(original / "offline_audit.json"), old_audit)


if __name__ == "__main__":
    unittest.main()
