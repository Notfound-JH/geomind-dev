"""Regenerate registered tasks with Panda + Robotiq on actual assembled scenes.

Every case keeps its own source snapshot, assembly, GT and action evidence.
Workers run serially with bounded deadlines. An exported scene or solved GT is
never counted as a physically completed task.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from geomind.scene_assembly.assets import ROOT


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _selected_tasks(requested):
    from geomind.episodes.registered_tasks import REGISTERED_TASK_IDS,supported_registered_task_ids
    if requested is None:
        return list(REGISTERED_TASK_IDS)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--tasks must contain distinct registered task IDs")
    supported=supported_registered_task_ids()
    unknown = set(requested) - set(supported)
    if unknown:
        raise ValueError(f"Unknown registered tasks: {sorted(unknown)}")
    return [task for task in supported if task in requested]


def _worker(kind, case, timeout_s, isaac_python=None):
    """Use the existing Isaac environment; retain and bound the whole process group."""
    from geomind.tools.assemble_scene import _python_environment
    python, environment = _python_environment(isaac_python, usd_only=kind == "export")
    token = uuid.uuid4().hex
    command = [python, "-m", "geomind.tools.regenerate_registered_tasks", "--worker", kind,
               "--case-dir", str(case), "--run-token", token]
    _write(case / f"{kind}_command.json", {"argv": command, "timeout_s": timeout_s, "run_token": token})
    logfile = case / f"{kind}.log"
    report_path = case / "assembly" / ("physics_report.json" if kind == "physics" else "manifest.json")
    cleanup = False
    with logfile.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout_s
        finished_at = None
        while process.poll() is None:
            if kind == "physics" and report_path.is_file() and finished_at is None:
                try:
                    report = _load(report_path)
                    if (report.get("run_token") == token and report.get("executed") is True
                            and type(report.get("passed")) is bool and report.get("physics_steps") == 600):
                        finished_at = time.monotonic()
                except (OSError, json.JSONDecodeError):
                    pass
            expired = time.monotonic() > deadline
            cleanup_due = finished_at is not None and time.monotonic() - finished_at > 8
            if expired or cleanup_due:
                # A complete measured report written before the deadline can
                # survive bounded renderer shutdown, as in assemble_scene.
                cleanup = finished_at is not None
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                if expired and not cleanup:
                    raise TimeoutError(f"{kind} exceeded {timeout_s}s; see {logfile}")
                break
            time.sleep(.3)
    if not report_path.is_file():
        raise RuntimeError(f"{kind} produced no report (exit {process.returncode}); see {logfile}")
    report = _load(report_path)
    if kind == "physics" and report.get("run_token") != token:
        raise RuntimeError("Initial physics report token mismatch")
    if kind == "export" and process.returncode:
        raise RuntimeError(f"Export failed (exit {process.returncode}); see {logfile}")
    return {"exit_code": process.returncode, "bounded_cleanup_after_report": cleanup,
            "report_path": str(report_path), "report": report}


def _snapshot(case, command, plan):
    from geomind.tools.research_pairs import snapshot
    bundle_data=plan.get("asset_bundle_provenance")
    if bundle_data:
        import hashlib
        contents=Path(bundle_data["manifest_path"]).read_bytes()
        if hashlib.sha256(contents).hexdigest()!=bundle_data["manifest_sha256"]:
            raise ValueError("Asset bundle manifest changed before source snapshot")
        (case/"input_assets").mkdir(exist_ok=True)
        (case/"input_assets/asset_bundle_manifest.json").write_bytes(contents)
    paths = set(Path(name).relative_to(ROOT).as_posix() for name in plan.get("source_hashes", {})
                if Path(name).is_relative_to(ROOT))
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "geomind/scene_assembly").glob("*.py"))
    paths.update({"geomind/tools/regenerate_registered_tasks.py", "geomind/tools/assemble_scene.py",
                  "geomind/tools/run_episode_physics.py", "geomind/tools/audit_assembled_robot.py",
                  "geomind/tools/audit_negative_pick.py", "geomind_lab/utils/portable_paths.py",
                  "tests/test_negative_pick_command_audit.py", "tests/test_native_registration_routing.py",
                  "tests/test_canonical_contact_height.py",
                  "tests/test_explicit_negative_pick_candidate.py",
                  "tests/test_regenerate_registered_tasks.py"})
    return snapshot(case, command, index_filename="request.json", extra_paths=sorted(paths))


def _finish_provenance(case, metadata):
    from geomind.tools.research_pairs import check_provenance, digest, encoded
    metadata["config_files"] = {p.relative_to(case).as_posix(): digest(p.read_bytes())
                                for folder in ("configs", "robot_configs", "evaluator", "observations", "input_assets")
                                for p in sorted((case / folder).rglob("*.json"))}
    for name in ("request.json", "registered_plan.json", "task_result.json", "task_binding.json"):
        path = case / name
        if path.is_file():
            metadata["config_files"][name] = digest(path.read_bytes())
    metadata["configuration_id"] = digest(encoded(metadata["config_files"]))
    _write(case / "provenance.json", metadata)
    return check_provenance(case)


def _execution_request(manifest, plan, timeout_s, max_steps, negative_controls, robot_base=None):
    response = plan["registration"]["response"]
    action = "PickLiftSelectedObject" if response == "SelectObject" else "PickPlaceObject"
    height = float(manifest["table"]["height_m"])
    return {"schema_version": 1, "stage": "assemble_solve_execute",
            "assembly_manifest": str(Path(manifest["scene_usda"]).parent / "manifest.json"),
            "task": {"task_id": plan["registration"]["task_id"]},
            "selection": {"actor": "agent", "reason": "Reuse this registered task's authored objects and structural planner; execute real Robotiq grasp and verify measured outcome.",
                          "considered_scene_ids": [manifest["scene"]["scene_id"]], "considered_layouts": ["registered"]},
            "robot": {"embodiment": "franka_robotiq", "base_position_w_m": list(robot_base) if robot_base is not None else [0., -.45, height]},
            "execution": {"action": action, "negative_controls": negative_controls, "timeout_s": timeout_s,
                          "target_offset_m": [0., 0., .1], "position_tolerance_m": .01,
                          "orientation_tolerance_rad": .15, "dwell_steps": 60, "max_steps": max_steps,
                          "joint_step_limit_rad": .04, "approach_clearance_m": .15}}


def _positive_cases(robot):
    return [row for row in robot.get("report", {}).get("results", []) if row.get("negative_control") is False]


def _runtime_gt_verified(robot):
    positives = _positive_cases(robot)
    return bool(positives and all(row.get("independent_gt_check", {}).get("success") is True
                                and bool(row.get("physical", {}).get("frozen_ground_truth"))
                                for row in positives))


def _action_passed(robot, action):
    positives = _positive_cases(robot)
    return bool(robot.get("passed") and robot.get("offline_audit", {}).get("passed")
                and robot.get("report", {}).get("physical_execution_verified") and positives
                and all(row.get("success") is True and row.get("physical", {}).get("action") == action
                        and row.get("physical", {}).get("success") is True for row in positives))


def _run_case(task_id, case, *, scene_id, seed, timeout_s, max_steps, negative_controls,
              prepare_only, isaac_python, command, robot_base=None, layout_scale=1.):
    from geomind.episodes.registered_tasks import load_registration
    case.mkdir(parents=True, exist_ok=False)
    result = {"task_id": task_id, "stage": "assemble_solve_execute", "status": "initializing",
              "robot": "franka_robotiq", "scene_id": scene_id, "seed": seed,
              "design_generated": False, "initial_layout_physics_verified": False,
              "gt_verified": False, "gt_pending_runtime": False,
              "task_action_physics_executed": False, "task_action_physics_verified": False,
              "task_completed": False, "model_evaluated": False, "output": str(case)}
    metadata = None
    stage = "registration"
    try:
        plan = load_registration(task_id, seed=seed, scene_id=scene_id)
        if layout_scale != 1.:
            plan.update(layout_scale=float(layout_scale), layout_scale_reason=
                "Agent selected a uniformly scaled version of the existing seeded layout for Robotiq clearance, preserving task/GT rules and all assets.")
        _write(case / "registered_plan.json", plan)
        _write(case / "request.json", {"task_id": task_id, "scene_id": scene_id, "seed": seed,
            "robot": "franka_robotiq", "prepare_only": prepare_only,
            "timeout_s": timeout_s, "max_steps": max_steps, "negative_controls": negative_controls})
        if layout_scale != 1.:
            initial_request = _load(case / "request.json")
            initial_request["layout_scale"] = float(layout_scale)
            _write(case / "request.json", initial_request)
        metadata = _snapshot(case, command, plan)
        stage = "export"
        exported = _worker("export", case, timeout_s, isaac_python)
        manifest = exported["report"]
        result.update(design_generated=True, status="assembled_physics_unverified",
                      assembly_manifest=exported["report_path"], export_worker={k: v for k, v in exported.items() if k != "report"})
        request = _execution_request(manifest, plan, timeout_s, max_steps, negative_controls, robot_base)
        if layout_scale != 1.:
            request["layout_scale"] = float(layout_scale)
            request["agent_layout_transform"] = manifest["agent_layout_transform"]
            plan["agent_layout_transform"] = manifest["agent_layout_transform"]
            _write(case / "registered_plan.json", plan)
            request["selection"]["reason"] += " The agent explicitly selected uniform layout_scale=" + str(float(layout_scale)) + " before measured GT solving."
        _write(case / "request.json", {**request, "output": str(case), "prepare_only": prepare_only,
                                      "seed": seed, "registered_task_id": task_id})
        _write(case / "result.json", result)
        if prepare_only:
            result["status"] = "prepared_physics_unexecuted"
            return result
        stage = "initial_layout_physics"
        settled = _worker("physics", case, timeout_s, isaac_python)
        report = settled["report"]
        manifest["physics"] = report
        manifest["physics_worker"] = {k: v for k, v in settled.items() if k != "report"}
        manifest["status"] = "assembled_physics_passed" if report.get("passed") else "assembled_physics_failed"
        for key in ("preview", "detail_preview", "video"):
            manifest[key] = report.get(key)
        _write(Path(result["assembly_manifest"]), manifest)
        if settled["exit_code"] and not settled["bounded_cleanup_after_report"]:
            raise RuntimeError(f"Initial physics worker exited unexpectedly after reporting: {settled['exit_code']}")
        if not report.get("passed"):
            raise ValueError("Initial layout did not pass measured PhysX settling; robot action was not started")
        from geomind.scene_assembly.audit import audit
        initial_audit = audit(case / "assembly")
        manifest["offline_audit"] = initial_audit
        _write(Path(result["assembly_manifest"]), manifest)
        if not initial_audit.get("passed"):
            raise ValueError("Independent initial-layout audit did not pass")
        result["initial_layout_physics_verified"] = True
        stage = "gt_binding"
        from geomind.scene_assembly.registered_binding import bind_and_solve
        solved = bind_and_solve(Path(result["assembly_manifest"]), case)
        result.update(task=solved, status=solved["status"],
                      gt_verified=bool(solved.get("ground_truth") is not None and not solved.get("gt_pending_runtime")),
                      gt_pending_runtime=bool(solved.get("gt_pending_runtime")))
        _write(case / "result.json", result)
        stage = "robot_action_physics"
        from geomind.scene_assembly.robot_execution import execute_robot_stage
        extension=plan['task_spec']['protocol'].get('module_name')=='canonical_extension'
        robot = execute_robot_stage(case, request, isolate_negative_controls=extension,
                                    negative_placement_policy='offsets_first' if extension else 'initial_first', isaac_python=isaac_python)
        action_passed = _action_passed(robot, request["execution"]["action"])
        gt_verified = result["gt_verified"] or _runtime_gt_verified(robot)
        result.update(robot_execution=robot, task_action_physics_executed=bool(robot.get("executed")),
                      task_action_physics_verified=action_passed, task_completed=action_passed,
                      gt_verified=gt_verified, gt_pending_runtime=bool(result["gt_pending_runtime"] and not gt_verified),
                      status="task_action_physics_passed" if action_passed else "task_action_physics_failed")
        solved.update(action_physics_executed=result["task_action_physics_executed"],
                      task_completed=action_passed, status=result["status"])
        _write(case / "task_result.json", solved)
        return result
    except Exception as exc:
        partial = case / "robot_physics/report.json"
        if stage == "robot_action_physics" and partial.is_file():
            try:
                report = _load(partial)
                evidence = {"report": report, "passed": False, "report_path": str(partial)}
                positives = _positive_cases(evidence)
                executed = any(row.get("physical", {}).get("motion_steps", 0) > 0 for row in positives)
                result.update(robot_execution=evidence, task_action_physics_executed=executed,
                              gt_verified=result["gt_verified"] or _runtime_gt_verified(evidence))
            except (ValueError, OSError, TypeError):
                pass
        result.update(status=f"{stage}_failed", task_completed=False, task_action_physics_verified=False,
                      failure={"stage": stage, "type": type(exc).__name__, "reason": str(exc),
                               "category": "invalid_instance" if isinstance(exc, ValueError) else "system_failure"})
        _write(case / "failure.json", result["failure"])
        return result
    finally:
        if metadata is not None:
            try:
                provenance = _finish_provenance(case, metadata)
                result["provenance_check"] = provenance
                if not provenance["passed"]:
                    result.update(status="provenance_failed", task_completed=False, task_action_physics_verified=False)
            except Exception as exc:
                result.update(status="provenance_failed", task_completed=False, task_action_physics_verified=False,
                              provenance_failure=f"{type(exc).__name__}: {exc}")
        _write(case / "result.json", result)


def _summary(root, request, rows):
    completed = sum(row.get("task_completed") is True for row in rows)
    counts = {"requested": len(request["tasks"]), "processed": len(rows), "completed_physics_passed": completed,
              "design_generated": sum(bool(row.get("design_generated")) for row in rows),
              "initial_layout_physics_passed": sum(bool(row.get("initial_layout_physics_verified")) for row in rows),
              "gt_verified": sum(bool(row.get("gt_verified")) for row in rows),
              "gt_pending_runtime": sum(bool(row.get("gt_pending_runtime")) for row in rows),
              "action_physics_failed": sum(bool(row.get("task_action_physics_executed")) and not row.get("task_completed") for row in rows),
              "action_unexecuted": sum(not row.get("task_action_physics_executed") for row in rows),
              "failed_before_action": sum(bool(row.get("failure")) and not row.get("task_action_physics_executed") for row in rows),
              "unprocessed": len(request["tasks"]) - len(rows)}
    result = {"schema_version": 1, "kind": "registered_task_regeneration", "request": request,
              "robot": "franka_robotiq", "model_evaluated": False, "counts": counts, "cases": rows,
              "status": "physics_passed" if completed == len(request["tasks"]) else "incomplete",
              "updated_at_utc": datetime.now(timezone.utc).isoformat(), "webpage": str(root / "index.html")}
    _write(root / "batch.json", result)
    lines = ["# 注册任务重新生成", "", f"机械臂：Panda + Robotiq 2F-85。物理完成 {completed}/{counts['requested']}；动作未执行 {counts['action_unexecuted']}。", "",
             "|任务|状态|GT|动作物理|", "|---|---|---|---|"]
    cards = []
    for row in rows:
        task_id = row["task_id"]
        gt = "通过" if row.get("gt_verified") else "运行时待解" if row.get("gt_pending_runtime") else "未验证"
        physical = "通过" if row.get("task_completed") else "未通过" if row.get("task_action_physics_executed") else "未执行"
        lines.append(f"|[{task_id}]({task_id}/result.json)|{row['status']}|{gt}|{physical}|")
        case = root / task_id
        instruction = row.get("task", {}).get("instruction", "")
        previews = list((case / "assembly").glob("preview.png"))
        videos = sorted(case.rglob("*.mp4"))
        media = "".join(f'<img loading="lazy" src="{html.escape(p.relative_to(root).as_posix(), quote=True)}" alt="实际场景">' for p in previews)
        media += "".join(f'<p>{html.escape(p.relative_to(case).as_posix())}</p><video controls preload="none" src="{html.escape(p.relative_to(root).as_posix(), quote=True)}"></video>' for p in videos)
        failure = row.get("failure", {}).get("reason", row.get("provenance_failure", ""))
        cards.append(f'<article><h2>{html.escape(task_id)}</h2><p>{html.escape(instruction)}</p><p>{html.escape(row["status"])} · GT {gt} · 动作 {physical}</p><p>{html.escape(failure)}</p><a href="{task_id}/result.json">原始报告与来源</a>{media}</article>')
    (root / "batch.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>注册任务 · Robotiq 重新生成</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui;background:#f4f5f6;color:#18212a}article{background:white;padding:24px;margin:20px 0;border-radius:12px}h2{font-size:20px}img,video{width:min(100%,640px);display:block;margin:12px 0}a{color:#175a9a}</style>'
            f'<h1>注册任务 · Panda + Robotiq 2F-85</h1><p>物理完成 {completed}/{counts["requested"]}；GT已验证 {counts["gt_verified"]}；动作未执行 {counts["action_unexecuted"]}。未接学习模型。场景落稳与GT通过分别记录，只有动作物理通过计入完成。</p>'
            '<p><a href="batch.json">批次原始报告</a> · <a href="request.json">请求</a></p>' + "".join(cards) + '</html>')
    (root / "index.html").write_text(page, encoding="utf-8")
    return result


def run_batch(output, *, tasks=None, scene_id=65, seed=0, timeout_s=420., max_steps=6000,
              negative_controls=True, prepare_only=False, isaac_python=None, command=None, robot_base=None, layout_scale=1.):
    tasks = _selected_tasks(tasks)
    if type(scene_id) is not int or not 1 <= scene_id <= 102 or type(seed) is not int or seed < 0:
        raise ValueError("Expected scene_id in 1..102 and a nonnegative seed")
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 1 <= timeout_s <= 420:
        raise ValueError("timeout_s must be in 1..420 per worker")
    if type(max_steps) is not int or not 120 <= max_steps <= 12000:
        raise ValueError("max_steps must be in 120..12000")
    try:
        scale_valid = type(layout_scale) in (int, float) and math.isfinite(layout_scale) and layout_scale > 0
    except OverflowError:
        scale_valid = False
    if not scale_valid:
        raise ValueError("layout_scale must be a finite positive number")
    if robot_base is not None:
        if (not isinstance(robot_base, (tuple, list)) or len(robot_base) != 3
                or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 10 for x in robot_base)):
            raise ValueError("robot_base must contain three finite world metre values in [-10,10]")
        robot_base = [float(x) for x in robot_base]
    root = Path(output).resolve()
    if root.is_relative_to(ROOT / "geomind/configs"):
        raise ValueError("Do not overwrite formal task configurations")
    root.mkdir(parents=True, exist_ok=False)
    command = command or [sys.executable, "-m", "geomind.tools.regenerate_registered_tasks"]
    request = {"tasks": tasks, "scene_id": scene_id, "seed": seed, "robot": "franka_robotiq",
               "timeout_s": timeout_s, "max_steps": max_steps, "negative_controls": negative_controls,
               "robot_base_w_m": robot_base, "robot_base_policy": "explicit_world_xyz" if robot_base is not None else "tabletop_height_x0_y_minus0.45",
               "prepare_only": prepare_only, "gpu_concurrency": 1, "source_snapshot_phase": "before_each_case_export"}
    _write(root / "request.json", request)
    if layout_scale != 1.:
        request["layout_scale"] = float(layout_scale)
        _write(root / "request.json", request)
    rows = []
    _summary(root, request, rows)
    for index, task_id in enumerate(tasks):
        print(f"[{index + 1}/{len(tasks)}] {task_id}: assembling scene {scene_id}, franka_robotiq", flush=True)
        row = _run_case(task_id, root / task_id, scene_id=scene_id, seed=seed, timeout_s=timeout_s,
                        max_steps=max_steps, negative_controls=negative_controls, prepare_only=prepare_only,
                        isaac_python=isaac_python, command=command, robot_base=robot_base, layout_scale=layout_scale)
        rows.append(row)
        _summary(root, request, rows)
        print(json.dumps({"task_id": task_id, "status": row["status"], "task_completed": row["task_completed"]}), flush=True)
    result = _summary(root, request, rows)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "webpage": result["webpage"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--scene-id", type=int, default=65)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=420.)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--layout-scale", type=float, default=1.,
                        help="Explicit uniform XY layout scale after one legacy randomizer sample; no resampling or scale-to-fit")
    parser.add_argument("--no-negative-controls", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Export real USD on CPU only; no GT or action completion claim")
    parser.add_argument("--isaac-python")
    parser.add_argument("--robot-base", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Explicit base world XYZ in metres; default [0,-0.45,measured tabletop height]")
    parser.add_argument("--worker", choices=("export", "physics"), help=argparse.SUPPRESS)
    parser.add_argument("--case-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-token", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if args.case_dir is None:
                raise ValueError("Worker requires --case-dir")
            case = args.case_dir.resolve()
            if args.worker == "export":
                from geomind.scene_assembly.registered_export import export_registered
                plan = _load(case / "registered_plan.json")
                export_registered(plan, case / "assembly", scene_id=plan["registration"]["scene_id"], seed=plan["seed"],
                                  layout_scale=plan.get("layout_scale", 1.))
                return 0
            from geomind.scene_assembly.physics import verify
            report = verify(case / "assembly", args.run_token)
            return 0 if report["passed"] else 1
        if args.output is None:
            raise ValueError("--output is required")
        command = [sys.executable, "-m", "geomind.tools.regenerate_registered_tasks", *(sys.argv[1:] if argv is None else argv)]
        result = run_batch(args.output, tasks=args.tasks, scene_id=args.scene_id, seed=args.seed,
                           timeout_s=args.timeout_s, max_steps=args.max_steps,
                           negative_controls=not args.no_negative_controls,
                           prepare_only=args.prepare_only, isaac_python=args.isaac_python, command=command,
                           robot_base=args.robot_base, layout_scale=args.layout_scale)
        if args.prepare_only:
            return 0 if (result["counts"]["processed"] == result["counts"]["requested"]
                         and all(row.get("status") == "prepared_physics_unexecuted"
                                 and row.get("provenance_check", {}).get("passed") is True
                                 for row in result["cases"])) else 1
        return 0 if result["status"] == "physics_passed" else 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
