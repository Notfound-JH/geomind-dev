#!/usr/bin/env python3
"""Inspect or explicitly generate fresh cases from the accepted 30-case matrix.

This stdlib-only wrapper never restores historical source or replays old bytes.
Only execution mode starts child simulation processes; --list/--dry-run do not.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs/reproduction/canonical27_verified.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_cases(path):
    matrix = json.loads(path.read_text())
    cases = matrix["cases"]
    ids = [case["task_id"] for case in cases]
    if len(cases) != 30 or len(set(ids)) != 30:
        raise ValueError("Expected the 30 distinct accepted instances covering 27 types")
    # Public type coverage is separate from legacy solver identity.  Historical
    # solver names remain source_solver_task_id metadata and are not regrouped.
    canonical = [case.get("canonical_task_id") for case in cases]
    if any(not isinstance(value, str) or re.fullmatch(r"[A-Z][A-Za-z0-9]*", value) is None
           for value in canonical):
        raise ValueError("Expected public canonical task type names in matrix metadata")
    if matrix.get("canonical_task_types") != 27 or len(set(canonical)) != 27:
        raise ValueError("Expected exactly 27 public canonical types across the 30 matrix instances")
    for case in cases:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", case["task_id"]):
            raise ValueError("Invalid task identity in reproduction matrix")
        if case["robot"] != "franka_robotiq":
            raise ValueError("This registered runner supports franka_robotiq only")
        base = case["robot_base_w_m"]
        if len(base) != 3 or not all(math.isfinite(float(x)) for x in base):
            raise ValueError("Expected each case's actual finite world XYZ base")
        suggested = case["current_source_suggested_execution"]
        if not 0 < int(suggested["max_steps"]) <= 12000:
            raise ValueError("Current-source action budget must be at most 12000 ticks")
        if not 0 < float(suggested["worker_timeout_s"]) <= 420:
            raise ValueError("Each simulation worker must remain bounded by 420 seconds")
    return matrix, cases


def plan_case(case, index, output):
    task = case["task_id"]
    run = output / "runs" / f"{index:02d}_{task}"
    destination = run / task
    suggested = case["current_source_suggested_execution"]
    command = [sys.executable, "-m", "geomind.tools.regenerate_registered_tasks",
               "--output", str(run), "--tasks", task,
               "--scene-id", str(case["scene_id"]), "--seed", str(case["seed"]),
               "--timeout-s", str(suggested["worker_timeout_s"]),
               "--max-steps", str(suggested["max_steps"]),
               "--layout-scale", str(case["layout_scale"]),
               "--robot-base", *(str(x) for x in case["robot_base_w_m"])]
    if case["original_execution"].get("negative_controls") is False:
        command.append("--no-negative-controls")
    audit = [sys.executable, "-m", "geomind.tools.verify_wrist_cameras", "audit",
             "--physics", str(destination / "robot_physics")]
    return {"task_id": task, "case_directory": str(destination),
            "physics_argv": command, "wrist_audit_argv": audit,
            "prior_saved_source_snapshot_id": case.get("source_snapshot_id"),
            "status": "not_executed"}


def run_case(item, logs, index):
    task = item["task_id"]
    for stage, argv in (("physics", item["physics_argv"]), ("wrist_audit", item["wrist_audit_argv"])):
        log = logs / f"{index:02d}_{task}_{stage}.log"
        item[stage + "_log"] = str(log)
        item["status"] = stage + "_running"
        try:
            with log.open("w") as stream:
                result = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=180 if stage == "wrist_audit" else None)
            item[stage + "_returncode"] = result.returncode
            if result.returncode:
                item["status"] = stage + "_failed"
                return False
            case = Path(item["case_directory"])
            evidence = case / ("result.json" if stage == "physics" else "robot_physics/wrist_camera_audit.json")
            record = json.loads(evidence.read_text())
            key = "task_completed" if stage == "physics" else "passed"
            if record.get(key) is not True:
                raise ValueError(f"Exit zero without verified {key} evidence")
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            item["status"] = stage + "_failed"
            item["failure"] = f"{type(error).__name__}: {error}"
            return False
    item["status"] = "passed"
    item["completed_at_utc"] = now()
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    inspect = parser.add_mutually_exclusive_group()
    inspect.add_argument("--list", action="store_true", help="List matrix cases without importing the SDK")
    inspect.add_argument("--dry-run", action="store_true", help="Print commands only; create no output directory")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tasks", nargs="+", help="Explicit registered instance IDs")
    selection.add_argument("--all", action="store_true", help="Explicitly select all 30 instances")
    parser.add_argument("--output", type=Path, help="New, nonexistent output root required for execution")
    parser.add_argument("--continue-on-failure", action="store_true", help="Keep failures and continue serially")
    args = parser.parse_args(argv)
    try:
        matrix, cases = load_cases(args.matrix)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    by_id = {case["task_id"]: case for case in cases}
    if args.tasks:
        if len(set(args.tasks)) != len(args.tasks) or any(task not in by_id for task in args.tasks):
            parser.error("--tasks must contain distinct IDs from --list")
        cases = [by_id[task] for task in args.tasks]
    if args.list:
        print(json.dumps({"canonical_task_types": matrix["canonical_task_types"], "instance_count": len(cases),
                          "sdk_started": False, "cases": [{key: case[key] for key in
                          ("task_id", "scene_id", "seed", "robot", "robot_base_w_m", "layout_scale", "validation_status")}
                          for case in cases]}, ensure_ascii=False, indent=2))
        return 0
    if not args.dry_run and (not (args.tasks or args.all) or args.output is None):
        parser.error("Execution requires explicit --tasks ID... or --all, plus --output NEW_DIRECTORY")
    output = (args.output or ROOT / "outputs/canonical27_NEW_DIRECTORY").expanduser().absolute()
    if not args.dry_run and (output.exists() or output.is_symlink()):
        parser.error("--output must not exist; failed evidence is never overwritten")
    items = [plan_case(case, index, output) for index, case in enumerate(cases, 1)]
    report = {"schema_version": 1, "mode": "dry_run" if args.dry_run else "fresh_current_source",
              "fresh_current_source": True, "original_evidence_replayed": False,
              "model_evaluated": False, "physical_execution_performed": False,
              "scope": "New generation under the current source; historical byte replay is not claimed. Prior acceptance is not a result of this run.",
              "matrix": str(args.matrix.resolve()), "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
              "created_at_utc": now(), "gpu_concurrency": 1, "stop_on_first_failure": not args.continue_on_failure,
              "selected_instances": len(items), "passed_instances": 0, "status": "not_executed", "cases": items}
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    destination = output / "reproduction_report.json"
    report["status"] = "running"
    write(destination, report)
    try:
        for index, item in enumerate(items, 1):
            report["physical_execution_performed"] = True
            passed = run_case(item, logs, index)
            report["passed_instances"] += int(passed)
            write(destination, report)
            print(json.dumps({"task_id": item["task_id"], "status": item["status"]}), flush=True)
            if not passed and not args.continue_on_failure:
                break
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        write(destination, report)
        raise
    report["status"] = "passed" if report["passed_instances"] == len(items) else "failed"
    report["finished_at_utc"] = now()
    write(destination, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
