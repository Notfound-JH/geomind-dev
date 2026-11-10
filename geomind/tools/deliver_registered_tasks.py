"""Re-audit a registered-task batch without changing its original evidence.

Only offline CPU replay is performed. The newest explicitly supplied retry
directory wins for each of the twenty registration IDs, including failed or
incomplete retries. Saved-source integrity, run-end provenance and current
workspace drift are reported separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from geomind.scene_assembly.assets import ROOT


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe saved provenance path: {value!r}")
    return path


def saved_integrity(case, *, repository=ROOT):
    """Verify saved bytes independently from whether today's source has changed."""
    case = Path(case); repository = Path(repository)
    result = {"saved_source_snapshot_complete": False, "saved_config_hashes_complete": False,
              "saved_source_mismatches": [], "saved_config_mismatches": [],
              "current_workspace_source_drift": [], "current_workspace_matches": False}
    try:
        provenance = _read(case / "provenance.json")
        sources = provenance["source_files"]; configs = provenance["config_files"]
        if not isinstance(sources, list) or not sources or not isinstance(configs, dict) or not configs:
            raise ValueError("Saved source/config manifests are empty")
        source_names = [item["path"] for item in sources]
        if len(source_names) != len(set(source_names)):
            raise ValueError("Duplicate saved source paths")
        manifest_hash = hashlib.sha256(_encoded({item["path"]: item["sha256"] for item in sources})).hexdigest()
        if manifest_hash != provenance["source_snapshot_id"]:
            result["saved_source_mismatches"].append("source_manifest_hash")
        for item in sources:
            relative = _relative_path(item["path"])
            saved = case / "source_snapshot" / relative
            if not saved.is_file() or _digest(saved) != item["sha256"]:
                result["saved_source_mismatches"].append(item["path"])
            current = repository / relative
            actual = _digest(current) if current.is_file() else None
            if actual != item["sha256"]:
                result["current_workspace_source_drift"].append({"path": item["path"],
                    "run_sha256": item["sha256"], "current_sha256": actual})
        if hashlib.sha256(_encoded(configs)).hexdigest() != provenance["configuration_id"]:
            result["saved_config_mismatches"].append("config_manifest_hash")
        for filename, expected in configs.items():
            relative = _relative_path(filename)
            if not (case / relative).is_file() or _digest(case / relative) != expected:
                result["saved_config_mismatches"].append(filename)
        required = {"request.json", "registered_plan.json", "task_result.json", "task_binding.json"}
        required.update(p.relative_to(case).as_posix()
                        for folder in ("configs", "robot_configs", "evaluator", "observations")
                        for p in (case / folder).rglob("*.json"))
        missing = sorted(required - configs.keys())
        if missing:
            result["saved_config_mismatches"].extend("unhashed:" + name for name in missing)
        if not any(name.startswith("configs/") for name in configs) or not any(name.startswith("robot_configs/") for name in configs):
            result["saved_config_mismatches"].append("missing_semantic_or_robot_configs")
        result.update(saved_source_snapshot_complete=not result["saved_source_mismatches"],
                      saved_config_hashes_complete=not result["saved_config_mismatches"],
                      current_workspace_matches=not result["current_workspace_source_drift"],
                      source_snapshot_id=provenance["source_snapshot_id"],
                      configuration_id=provenance["configuration_id"],
                      saved_source_file_count=len(sources), saved_config_file_count=len(configs),
                      provenance_file_sha256=_digest(case / "provenance.json"))
    except (KeyError, OSError, ValueError, TypeError) as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
    return result


def select_versions(batch, retries=()):
    from geomind.episodes.registered_tasks import REGISTERED_TASK_IDS
    roots = [Path(batch).resolve(), *(Path(path).resolve() for path in retries)]
    if len(roots) != len(set(roots)):
        raise ValueError("Batch and retry directories must be distinct and explicitly ordered")
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Missing batch/retry directory: {root}")
    selected = []
    for task_id in REGISTERED_TASK_IDS:
        # A partial newer retry still wins. Never search backwards for success.
        versions = [{"root": str(root), "case": str(root / task_id), "order": index}
                    for index, root in enumerate(roots) if (root / task_id).is_dir()]
        selected.append({"task_id": task_id, "versions": versions,
                         "selected": versions[-1] if versions else None})
    return roots, selected


def _copy_file(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _link(source, destination):
    source = Path(source).resolve(); destination = Path(destination)
    if not source.exists():
        raise ValueError(f"Missing original evidence: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def audit_initial(case, destination):
    """Run the existing settling auditor in a fresh, isolated output view."""
    from geomind.scene_assembly.audit import audit
    case = Path(case); destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    assembly = case / "assembly"
    manifest = _read(assembly / "manifest.json")
    report = _read(assembly / "physics_report.json")
    for name in ("manifest.json", "physics_report.json"):
        _copy_file(assembly / name, destination / name)
    files = set(report.get("sha256", {})) | {"asset_body_vertices.npz", "object_states.npz"}
    for filename in files:
        relative = _relative_path(filename)
        _link(assembly / relative, destination / relative)
    checked = audit(destination)
    return {"passed": bool(manifest.get("physics", {}).get("passed") and report.get("passed") and checked.get("passed")),
            "audit": checked, "report": str(destination / "offline_audit.json"),
            "original_manifest_sha256": _digest(assembly / "manifest.json"),
            "original_physics_report_sha256": _digest(assembly / "physics_report.json")}


def audit_robot(case, destination):
    """All writes from the existing auditor land in the new evidence view."""
    from geomind.tools.audit_assembled_robot import audit
    case = Path(case); destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    original = case / "robot_physics"
    _copy_file(original / "report.json", destination / "report.json")
    for folder in ("evidence", "observations"):
        _link(original / folder, destination / folder)
    checked = audit(case / "robot_configs", destination)
    return {"passed": checked.get("passed") is True, "audit": checked,
            "report": str(destination / "assembled_robot_audit.json"),
            "original_report_sha256": _digest(original / "report.json"),
            "mode": "new offline audit of original raw states, controls and media; no physical re-execution"}


def _same_gt(expected, actual):
    """Exact identities and frames; a one picometre bound for numeric replay."""
    import numpy as np
    if not isinstance(expected, dict) or not isinstance(actual, dict) or expected.get("type") != actual.get("type"):
        return False, None
    if expected["type"] == "Object":
        return (set(expected) == set(actual) == {"type", "object_id"}
                and isinstance(expected.get("object_id"), str) and bool(expected["object_id"])
                and expected == actual), None
    if expected["type"] == "Relation":
        fields = {"type", "relation", "target_object_id", "reference_object_id", "contract", "position_m", "frame", "point_role"}
        if (set(expected) != fields or set(actual) != fields
                or expected.get("point_role") != "representative_action_goal"
                or actual.get("point_role") != "representative_action_goal"
                or any(not isinstance(expected.get(key), str) or not expected[key]
                       or expected[key] != actual.get(key) for key in ("relation", "target_object_id", "reference_object_id"))
                or not isinstance(expected.get("contract"), dict) or not isinstance(actual.get("contract"), dict)
                or json.dumps(expected["contract"], sort_keys=True, separators=(",", ":"), allow_nan=False)
                    != json.dumps(actual["contract"], sort_keys=True, separators=(",", ":"), allow_nan=False)):
            return False, None
        # This compares only the frozen representative action command. Region
        # membership, release and stability are checked by the raw-state audit.
        expected = {key: value for key, value in expected.items() if key in {"position_m", "frame"}} | {"type": "Point"}
        actual = {key: value for key, value in actual.items() if key in {"position_m", "frame"}} | {"type": "Point"}
    if (expected["type"] != "Point" or set(expected) != {"type", "position_m", "frame"}
            or set(actual) != set(expected) or expected.get("frame") != "world" or expected.get("frame") != actual.get("frame")):
        return False, None
    for point in (expected["position_m"], actual["position_m"]):
        if (not isinstance(point, list) or len(point) != 3
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in point)):
            return False, None
    a = np.asarray(expected.get("position_m", []), dtype=float)
    b = np.asarray(actual.get("position_m", []), dtype=float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return False, None
    delta = float(np.max(np.abs(a - b)))
    return delta <= 1e-12, delta


def replay_gt(case, report, *, auditor=None):
    """Verify saved query snapshots against raw object clocks before GT replay."""
    import numpy as np
    from geomind.episodes import EpisodeResolver
    if auditor is None:
        from geomind.tools import audit_assembled_robot as auditor
    checked = []
    for row in report.get("results", []):
        resolved = EpisodeResolver(Path(case) / "robot_configs").resolve(row["episode_id"])
        trace = Path(row["trace"])
        with np.load(trace.with_name(trace.stem + "_objects.npz"), allow_pickle=False) as archive:
            raw = archive["data"]; ids = archive["object_ids"].tolist()
        states = raw[:, 2:].reshape(len(raw), len(ids), 13)
        _, points = auditor._load_body_geometry(resolved.scene["physics"], ids)
        evidence = _read(Path(case) / "robot_physics/evidence" / (trace.stem + ".json"))
        solved = auditor._replay_query_gt(resolved, evidence, raw[:, 0], states, ids, points)
        equivalent, delta = _same_gt(solved["ground_truth"], row["ground_truth"])
        frozen_equal, frozen_delta = _same_gt(row.get("physical", {}).get("frozen_ground_truth"), row["ground_truth"])
        checked.append({"negative_control": row.get("negative_control"), "passed": equivalent and frozen_equal,
            "runtime_ground_truth": row["ground_truth"], "replayed_ground_truth": solved["ground_truth"],
            "max_coordinate_difference_m": delta, "frozen_coordinate_difference_m": frozen_delta,
            "instance_id": row["instance_id"], "query_tick": evidence["physical_clock"]["current_tick"]})
    return {"passed": bool(checked) and all(row["passed"] for row in checked), "cases": checked,
            "scope": "original registered solver replay; query/history geometry checked against original raw traces",
            "numeric_comparison": {"point_atol_m": 1e-12, "point_rtol": 0., "object_identity": "exact", "frame": "exact"}}


def prepare_framework(case, destination, *, repository=ROOT):
    """Restore original episode modules, with explicitly recorded fallback files."""
    case = Path(case); destination = Path(destination); repository = Path(repository)
    verified = saved_integrity(case, repository=repository)
    if not verified.get("saved_source_snapshot_complete"):
        raise ValueError("Cannot replay an incomplete or modified saved source snapshot")
    destination.mkdir(parents=True, exist_ok=False)
    provenance = _read(case / "provenance.json")
    copied = {}
    paths = {p for package in ("geomind", "geomind_lab") for p in (repository / package).rglob("*.py")
             if "__pycache__" not in p.parts}
    paths.update(p for p in (repository / "geomind/configs").rglob("*")
                 if p.is_file() and p.suffix in {".json", ".yaml", ".yml"})
    for relative in ("assets/assets_catalog_with_metadata.json", "assets/scene_profiles/assets_catalog_with_metadata.json"):
        if (repository / relative).is_file():
            paths.add(repository / relative)
    for source in sorted(paths):
        relative = source.relative_to(repository)
        _copy_file(source, destination / relative)
        copied[relative.as_posix()] = _digest(source)
    episode_sources = [item for item in provenance["source_files"]
                       if _relative_path(item["path"]).parts[:2] == ("geomind", "episodes") and item["path"].endswith(".py")]
    names = {item["path"] for item in episode_sources}
    if not {"geomind/episodes/__init__.py", "geomind/episodes/resolver.py"} <= names:
        raise ValueError("Saved snapshot lacks the original episode package/resolver")
    # Merely overlaying old files leaves newly added modules in glob('*.py'),
    # changing IMPLEMENTATION_HASH. Rebuild this package entirely from saved
    # original files instead of patching the hash constant in memory.
    shutil.rmtree(destination / "geomind/episodes", ignore_errors=True)
    copied = {key: value for key, value in copied.items() if not key.startswith("geomind/episodes/")}
    restored = {}
    for item in provenance["source_files"]:
        relative = _relative_path(item["path"])
        source = case / "source_snapshot" / relative
        _copy_file(source, destination / relative)
        if _digest(destination / relative) != item["sha256"]:
            raise ValueError(f"Saved source restoration failed: {relative}")
        restored[item["path"]] = item["sha256"]
        copied.pop(item["path"], None)
    episode_files = sorted((destination / "geomind/episodes").glob("*.py"))
    expected_hash = hashlib.sha256(b"".join(path.name.encode() + b"\0" + path.read_bytes() + b"\0"
                                            for path in episode_files)).hexdigest()
    record = {"framework_root": str(destination), "original_source_snapshot_id": provenance["source_snapshot_id"],
        "restored_saved_files_sha256": restored, "current_fallback_files_sha256": copied,
        "expected_original_episode_implementation_hash": expected_hash,
        "episode_package_files": [p.name for p in episode_files], "episode_hash_constant_overridden": False,
        "fallback_policy": "Current package Python/config/catalog files absent from the original saved snapshot are copied and hashed explicitly; they are not claimed to be execution-time snapshots. The entire episode package comes only from saved original sources."}
    _write(destination.parent / "framework_restore.json", record)
    return record


def _worker_review(job_path):
    """Entry point in the isolated CPU process; never starts a simulation."""
    job = _read(job_path)
    result = {"initial": {"passed": False}, "robot": {"passed": False}, "gt": {"passed": False}}
    try:
        import geomind.episodes
        import geomind.episodes.resolver as resolver
        import geomind_lab
        framework = Path(job["framework_root"]).resolve()
        modules = {"geomind.episodes": Path(geomind.episodes.__file__).resolve(),
                   "geomind.episodes.resolver": Path(resolver.__file__).resolve(),
                   "geomind_lab": Path(geomind_lab.__file__).resolve()}
        if any(not path.is_relative_to(framework) for path in modules.values()):
            raise ValueError("A framework import escaped the saved-source replay environment")
        if resolver.IMPLEMENTATION_HASH != job["expected_episode_hash"]:
            raise ValueError("Restored episode implementation hash differs from the original snapshot")
        auditor_path = Path(job["auditor_path"])
        if _digest(auditor_path) != job["auditor_sha256"]:
            raise ValueError("New independent auditor snapshot changed")
        spec = importlib.util.spec_from_file_location("registered_delivery_latest_auditor", auditor_path)
        auditor = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = auditor
        spec.loader.exec_module(auditor)
        result["framework"] = {"module_paths": {key: str(value) for key, value in modules.items()},
            "episode_implementation_hash": resolver.IMPLEMENTATION_HASH,
            "new_auditor_path": str(auditor_path), "new_auditor_sha256": job["auditor_sha256"],
            "global_implementation_hash_modified": False}
        case = Path(job["case"])
        try:
            result["initial"] = audit_initial(case, Path(job["output"]) / "initial_layout")
        except Exception as exc:
            result["initial"] = {"passed": False, "failure": f"{type(exc).__name__}: {exc}"}
        original = case / "robot_physics"
        if (original / "report.json").is_file():
            report = _read(original / "report.json")
            try:
                result["gt"] = replay_gt(case, report, auditor=auditor)
            except Exception as exc:
                result["gt"] = {"passed": False, "failure": f"{type(exc).__name__}: {exc}"}
            try:
                view = Path(job["output"]) / "robot"
                view.mkdir(parents=True, exist_ok=False)
                _copy_file(original / "report.json", view / "report.json")
                for folder in ("evidence", "observations"):
                    _link(original / folder, view / folder)
                checked = auditor.audit(case / "robot_configs", view)
                result["robot"] = {"passed": checked.get("passed") is True, "audit": checked,
                    "report": str(view / "assembled_robot_audit.json"),
                    "original_report_sha256": _digest(original / "report.json"),
                    "mode": "new independent auditor + this case's original episode/framework snapshot; original raw trace replay only"}
            except Exception as exc:
                result["robot"] = {"passed": False, "failure": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        result["isolation_failure"] = f"{type(exc).__name__}: {exc}"
    _write(job["result_path"], result)
    return 0 if "isolation_failure" not in result else 1


def isolated_review(case, destination, auditor_path, *, timeout_s=180):
    destination = Path(destination)
    restore = prepare_framework(case, destination / "framework")
    auditor_path = Path(auditor_path).resolve()
    # This immutable copy of the delivery worker runs outside both the live
    # repository package and the restored package's potentially old tools.
    worker = destination / "delivery_worker.py"
    worker_source = auditor_path.with_name("deliver_registered_tasks.py")
    if not worker_source.is_file():
        worker_source = Path(__file__).resolve()
    _copy_file(worker_source, worker)
    job = {"case": str(Path(case).resolve()), "output": str(destination),
           "framework_root": restore["framework_root"],
           "expected_episode_hash": restore["expected_original_episode_implementation_hash"],
           "auditor_path": str(auditor_path), "auditor_sha256": _digest(auditor_path),
           "delivery_worker_sha256": _digest(worker),
           "result_path": str(destination / "isolated_review.json")}
    _write(destination / "worker_job.json", job)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = restore["framework_root"]
    command = [sys.executable, str(worker), "--offline-worker", str(destination / "worker_job.json")]
    _write(destination / "worker_command.json", {"argv": command, "timeout_s": timeout_s,
        "cwd": restore["framework_root"], "pythonpath_policy": "only the restored framework; site packages remain available"})
    try:
        with (destination / "worker.log").open("w", encoding="utf-8") as log:
            run = subprocess.run(command, cwd=restore["framework_root"], env=environment,
                                 stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
        reviewed = _read(job["result_path"])
        reviewed.update(framework_restore=restore, worker_returncode=run.returncode)
        return reviewed
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"initial": {"passed": False}, "robot": {"passed": False}, "gt": {"passed": False},
                "framework_restore": restore, "isolation_failure": f"{type(exc).__name__}: {exc}"}


def qualification(*, initial_verified, report, new_audit_passed, integrity, run_end_provenance_passed, gt_verified,
                  identity_verified=True):
    positives = [row for row in report.get("results", []) if row.get("negative_control") is False]
    negatives = [row for row in report.get("results", []) if row.get("negative_control") is True]
    positive_passed = bool(positives) and all(row.get("success") is True and row.get("expected_success") is True
        and row.get("verification_passed") is True and row.get("semantic", {}).get("success") is True
        and row.get("physical", {}).get("success") is True
        and row.get("physical", {}).get("action") in {"PickLiftSelectedObject", "PickPlaceObject"} for row in positives)
    negative_rejected = bool(negatives) and all(row.get("success") is False and row.get("expected_success") is False
        and row.get("verification_passed") is True and row.get("physical", {}).get("success") is False for row in negatives)
    robot_verified = bool(positives) and all(row.get("robot_name") == "franka_robotiq"
        and row.get("physical", {}).get("command_finger_joint_names") == ["finger_joint"]
        and "right_outer_knuckle_joint" in row.get("physical", {}).get("finger_joint_names", []) for row in positives)
    gates = {"initial_layout_verified": bool(initial_verified), "worker_physical_execution_verified": report.get("physical_execution_verified") is True,
             "positive_action_passed": positive_passed, "negative_controls_rejected": negative_rejected,
             "new_offline_robot_audit_passed": bool(new_audit_passed),
             "runtime_gt_replayed": gt_verified is True,
             "saved_source_snapshot_complete": integrity.get("saved_source_snapshot_complete") is True,
             "saved_config_hashes_complete": integrity.get("saved_config_hashes_complete") is True,
             "run_end_provenance_passed": run_end_provenance_passed is True,
             "registered_identity_verified": identity_verified is True, "robotiq_evidence_verified": robot_verified}
    return {"task_completed": all(gates.values()), "gates": gates,
            "failed_gates": [name for name, passed in gates.items() if not passed],
            "current_workspace_drift_blocks_completion": False,
            "current_workspace_drift_policy": "Saved original sources/configs must be intact and run-end provenance must pass; later workspace drift is reported separately."}


def _capture_auditor(destination):
    paths = {Path(__file__).resolve(), ROOT / "geomind/tools/audit_assembled_robot.py"}
    paths.update((ROOT / "geomind/episodes").glob("*.py"))
    paths.update(ROOT / relative for relative in ("geomind/scene_assembly/audit.py", "geomind/scene_assembly/usd.py",
        "geomind/scene_assembly/body_geometry.py", "geomind/scene_assembly/assets.py"))
    hashes = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        _copy_file(path, destination / relative)
        hashes[relative.as_posix()] = _digest(path)
    import numpy as np
    result = {"source_files_sha256": hashes, "source_snapshot_id": hashlib.sha256(_encoded(hashes)).hexdigest(),
              "python": sys.version, "numpy": np.__version__, "physical_execution_performed": False}
    _write(destination.parent / "audit_implementation.json", result)
    return result


def _href(path, output):
    return Path(os.path.relpath(Path(path), output)).as_posix()


def _case_presentation(case, source_link):
    """Read display metadata only from the selected case, never a prior retry."""
    case = Path(case)
    documents = {}
    for name in ("request.json", "assembly/manifest.json"):
        try:
            value = _read(case / name)
            documents[name] = value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            documents[name] = {}
    request, manifest = documents["request.json"], documents["assembly/manifest.json"]
    manifest_request = manifest.get("request") or {}
    scene = request.get("scene_id")
    scene_source = "request.json.scene_id"
    if scene is None:
        scene = manifest_request.get("scene_id", (manifest.get("scene") or {}).get("scene_id"))
        scene_source = "assembly/manifest.json"
    seed = request.get("seed", manifest_request.get("seed"))
    seed_source = "request.json.seed" if "seed" in request else "assembly/manifest.json.request.seed"
    scale = request.get("layout_scale")
    scale_source = "request.json.layout_scale"
    if scale is None:
        scale = (manifest.get("agent_layout_transform") or {}).get("layout_scale")
        scale_source = "assembly/manifest.json.agent_layout_transform.layout_scale"
    if scale is None:
        scale = (manifest_request.get("layout") or {}).get("agent_scale")
        scale_source = "assembly/manifest.json.request.layout.agent_scale"
    if scale is None and (request or manifest):
        scale, scale_source = 1., "default 1; selected request/manifest have no explicit scale"
    robot = request.get("robot")
    embodiment = robot.get("embodiment") if isinstance(robot, dict) else robot if isinstance(robot, str) else None
    metadata = {"scene_id": scene, "seed": seed, "layout_scale": scale, "robot": embodiment,
                "sources": {"scene_id": scene_source, "seed": seed_source, "layout_scale": scale_source,
                            "robot": "request.json.robot"}, "request_link": source_link + "/request.json" if (case / "request.json").is_file() else None}
    links = []
    for label, filename in (("初始组装来源", "assembly/manifest.json"), ("原始落稳报告", "assembly/physics_report.json"),
                            ("组装USD", "assembly/scene.usda")):
        if (case / filename).is_file():
            links.append({"label": label, "href": source_link + "/" + filename})
    for label, key in (("初始全景", "preview"), ("初始桌面细节", "detail_preview"), ("初始落稳录像", "video")):
        raw = manifest.get(key) or (manifest.get("physics") or {}).get(key)
        if not isinstance(raw, str):
            continue
        path = Path(raw)
        if not path.is_absolute():
            candidates = [case / path, case / "assembly" / path]
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if path.is_file() and path.is_relative_to(case):
            links.append({"label": label, "href": source_link + "/" + path.relative_to(case).as_posix()})
    return metadata, links


def _distance_mm(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        return "未记录"
    return f"{value * 1000:.3f} mm"


def _runtime_gt_display(ground_truth):
    if not isinstance(ground_truth, dict):
        return "没有保存真实运行时GT"
    if ground_truth.get("type") == "Object":
        return "Object · " + str(ground_truth.get("object_id", "未记录"))
    point = ground_truth.get("position_m")
    if (isinstance(point, (list, tuple)) and len(point) == 3
            and all(type(x) in (int, float) and math.isfinite(x) for x in point)):
        role = "代表执行点（不限定合法区域）" if ground_truth.get("type") == "Relation" else "目标点"
        return role + " · [" + ", ".join(f"{x:.3f}" for x in point) + "] m · " + str(ground_truth.get("frame", "未记录"))
    return str(ground_truth.get("type", "未知GT类型")) + " · 坐标未记录"


def _review_case(selection, audit_dir, output, auditor_path):
    task_id = selection["task_id"]
    row = {"task_id": task_id, "selected": selection["selected"], "versions": selection["versions"],
           "task_completed": False, "gt_verified": False, "initial_gt_solved": False,
           "task_action_physics_executed": False, "model_evaluated": False, "issues": []}
    if selection["selected"] is None:
        row.update(status="missing", qualification={"task_completed": False, "failed_gates": ["missing_case"]})
        return row
    case = Path(selection["selected"]["case"])
    own = audit_dir / "cases" / task_id
    own.mkdir(parents=True, exist_ok=False)
    for version in row["versions"]:
        original = Path(version["case"])
        link = audit_dir / "source_cases" / f"run_{version['order']:02d}" / task_id
        _link(original, link)
        version["source_link"] = _href(link, output)
        if (original / "result.json").is_file():
            version["original_status"] = _read(original / "result.json").get("status")
        else:
            version["original_status"] = "missing_result"
    selected_link = row["versions"][-1]["source_link"]
    row["source_link"] = selected_link
    row["selected_request"], row["initial_assembly_links"] = _case_presentation(case, selected_link)
    report = {}; initial = {"passed": False}; audited = {"passed": False}
    original = {}; identity = False
    integrity = saved_integrity(case)
    row["integrity"] = integrity
    try:
        original = _read(case / "result.json")
        row.update(original_status=original.get("status"), original_task_completed=original.get("task_completed"),
            run_end_provenance=original.get("provenance_check") or {},
            task_action_physics_executed=bool(original.get("task_action_physics_executed")))
        task = _read(case / "task_result.json") if (case / "task_result.json").is_file() else {}
        row.update(instruction=task.get("instruction", ""), operation=task.get("operation"),
                   initial_ground_truth=task.get("ground_truth"), gt_pending_runtime=task.get("gt_pending_runtime", False),
                   initial_gt_solved=bool(task.get("ground_truth")) and not task.get("gt_pending_runtime"))
        manifest = _read(case / "assembly/manifest.json") if (case / "assembly/manifest.json").is_file() else {}
        identity = original.get("task_id") == task_id and task.get("task_id") == task_id and manifest.get("registration", {}).get("task_id") == task_id
        try:
            isolated = isolated_review(case, own / "isolated", auditor_path)
        except Exception as exc:
            isolated = {"initial": {"passed": False}, "robot": {"passed": False}, "gt": {"passed": False},
                        "isolation_failure": f"{type(exc).__name__}: {exc}"}
        row["replay_environment"] = {key: value for key, value in isolated.items() if key not in {"initial", "robot", "gt"}}
        initial = isolated.get("initial", {"passed": False})
        if (case / "robot_physics/report.json").is_file():
            report = _read(case / "robot_physics/report.json")
            positives = [r for r in report.get("results", []) if r.get("negative_control") is False]
            row["task_action_physics_executed"] = bool(positives and any(r.get("physical", {}).get("motion_steps", 0) > 0 for r in positives))
            identity = identity and bool(positives) and all(r.get("task_id") == task_id for r in positives)
            row["runtime_cases"] = [{"negative_control": r.get("negative_control"), "success": r.get("success"),
                "expected_success": r.get("expected_success"), "verification_passed": r.get("verification_passed"),
                "ground_truth": r.get("ground_truth"), "robot_name": r.get("robot_name"), "physical": {
                    key: r.get("physical", {}).get(key) for key in ("action", "success", "position_error_m", "position_tolerance_m",
                        "lift_height_m", "grasp_verified", "dwell_steps", "required_dwell_steps", "termination_reason")},
                "video": r.get("video")} for r in report.get("results", [])]
            gt = isolated.get("gt", {"passed": False})
            row["runtime_gt_review"] = gt
            row["gt_verified"] = gt.get("passed") is True
            _write(own / "runtime_gt_review.json", gt)
            audited = isolated.get("robot", {"passed": False})
            row["original_robot_audit"] = original.get("robot_execution", {}).get("offline_audit")
    except (KeyError, OSError, ValueError, TypeError) as exc:
        row["issues"].append(f"{type(exc).__name__}: {exc}")
    row.update(initial_layout_review=initial, new_robot_audit=audited)
    checks = qualification(initial_verified=initial.get("passed"), report=report,
        new_audit_passed=audited.get("passed"), integrity=integrity,
        run_end_provenance_passed=(original.get("provenance_check") or {}).get("passed"),
        gt_verified=row["gt_verified"], identity_verified=identity)
    row.update(qualification=checks, task_completed=checks["task_completed"],
               status="physics_passed" if checks["task_completed"] else "physics_failed" if row["task_action_physics_executed"] else "unexecuted")
    _write(own / "review.json", row)
    return row


def _make_html(report, output):
    esc = lambda value: html.escape(str(value), quote=True)
    counts = report["counts"]
    cards = []
    for case in report["cases"]:
        task_id = case["task_id"]
        state = "通过" if case["task_completed"] else "动作未通过" if case["task_action_physics_executed"] else "未执行"
        runtime = case.get("runtime_cases", [])
        media = []
        privileged = []
        source_case = Path(case["selected"]["case"]) if case["selected"] else None
        for raw in runtime:
            label = "错误目标对照" if raw.get("negative_control") else "真实正例动作"
            actual_gt = raw.get("ground_truth")
            privileged.append(f'<p>{label}：{esc(_runtime_gt_display(actual_gt))}</p>'
                f'<pre>{esc(json.dumps(actual_gt, ensure_ascii=False, indent=2))}</pre>')
            video = raw.get("video") or {}
            source = Path(video["observer"]) if video.get("observer") else None
            if source is not None and source.is_file() and source_case is not None and source.is_relative_to(source_case):
                link = case["source_link"] + "/" + source.relative_to(source_case).as_posix()
                label = "错误目标对照" if raw["negative_control"] else "真实正例动作"
                metrics = raw.get("physical") or {}
                gt_type = (raw.get("ground_truth") or {}).get("type")
                if gt_type == "Object":
                    text = f'抓起高度 {esc(_distance_mm(metrics.get("lift_height_m")))}'
                elif gt_type == "Relation":
                    text = f'代表点误差（不决定关系成功）{esc(_distance_mm(metrics.get("position_error_m")))}'
                else:
                    text = f'目标点误差 {esc(_distance_mm(metrics.get("position_error_m")))}'
                media.append(f'<p>{label} · {text} · 稳定 {esc(metrics.get("dwell_steps"))} 步</p>'
                             f'<video controls preload="none" src="{esc(link)}"></video>')
                if gt_type == "Relation":
                    relation = [item.get("relation_predicate_recomputed") for item in
                        case.get("new_robot_audit", {}).get("audit", {}).get("cases", [])
                        if item.get("negative_control") == raw["negative_control"]]
                    if relation:
                        privileged.append('<details><summary>物体末态关系验收</summary><pre>' +
                            esc(json.dumps(relation[0], ensure_ascii=False, indent=2)) + '</pre></details>')
        metadata = case.get("selected_request") or {}
        def shown(value):
            return "未记录" if value is None else str(value)
        scale_value = metadata.get("layout_scale")
        scale_label = f"{scale_value:g}" if type(scale_value) in (int, float) and math.isfinite(scale_value) else shown(scale_value)
        if str(metadata.get("sources", {}).get("layout_scale", "")).startswith("default"):
            scale_label += "（默认）"
        metadata_html = (f'<p>所选请求：scene {esc(shown(metadata.get("scene_id")))} · seed {esc(shown(metadata.get("seed")))} · '
            f'layout_scale {esc(scale_label)} · robot {esc(shown(metadata.get("robot")))}</p>')
        if metadata.get("request_link"):
            metadata_html += f'<p><a href="{esc(metadata["request_link"])}">所选请求原文</a></p>'
        initial_links = " · ".join(f'<a href="{esc(item["href"])}">{esc(item["label"])}</a>'
                                 for item in case.get("initial_assembly_links", []))
        if not privileged:
            privileged.append('<p>没有保存真实运行时GT；未从计划或初始绑定结果补值。</p>')
        privileged_html = ('<details><summary>研究者特权：真实运行时GT与关系验收</summary>'
            '<p>仅用于证据复核，不作为受评agent输入。坐标摘要保留三位小数，完整GT保留原始精度；复算状态以卡片标记为准。</p>'
            + "".join(privileged) + '</details>')
        versions = " · ".join(f'<a href="{esc(v["source_link"] + ("/result.json" if (Path(v["case"]) / "result.json").is_file() else "/"))}">版本{v["order"] + 1}：{esc(v.get("original_status"))}</a>' for v in case["versions"] if "source_link" in v)
        drift = case.get("integrity", {}).get("current_workspace_source_drift", [])
        gt = "已按原始查询复算" if case["gt_verified"] else "未验证"
        failures = ", ".join(case.get("qualification", {}).get("failed_gates", []))
        saved = case.get("integrity", {})
        cards.append(f'<article><h2>{esc(task_id)} <span class="{"pass" if case["task_completed"] else "fail"}">{state}</span></h2>'
            f'<p>{esc(case.get("instruction", ""))}</p>' + metadata_html + f'<p>{initial_links}</p><p>GT：{gt}。原始结束时来源检查：{esc(case.get("run_end_provenance", {}).get("passed"))}；'
            f'保存源码完整：{esc(saved.get("saved_source_snapshot_complete"))}；配置完整：{esc(saved.get("saved_config_hashes_complete"))}；当前工作区变化：{len(drift)} 文件。</p>'
            f'<p>{esc(failures)}</p><p>{versions}</p>' + "".join(media) + privileged_html +
            f'<details><summary>审计与来源变化</summary><pre>{esc(json.dumps({"qualification":case.get("qualification"),"selected_request_sources":metadata.get("sources"),"source_drift":drift,"issues":case.get("issues")},ensure_ascii=False,indent=2))}</pre></details></article>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>20注册任务 · Robotiq 验收交付</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#f2f4f5;color:#18212a;font:16px/1.65 system-ui}article{background:white;margin:20px 0;padding:24px;border-radius:12px}h2{font-size:20px}.pass{color:#17693e}.fail{color:#a22e2e}video{display:block;width:min(100%,680px);margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#185b9b}</style>'
        f'<h1>20注册任务 · Panda + Robotiq 2F-85</h1><p>物理完成 {counts["completed_physics_passed"]}/20；运行时GT复算 {counts["gt_verified"]}/20；动作未通过 {counts["action_failed"]}；未执行 {counts["action_unexecuted"]}。</p>'
        '<p>本页重新审计原始状态、控制和视频，没有再次执行物理。保存源码、结束时来源检查和当前工作区变化分别列出。未接学习模型。</p><p><a href="delivery.json">完整验收与版本选择</a></p>' + "".join(cards) + '</html>')
    (output / "delivery.html").write_text(page, encoding="utf-8")


def deliver(batch, retries=(), output=None):
    roots, selected = select_versions(batch, retries)
    output = Path(output).resolve() if output is not None else roots[0]
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    audits = output / "delivery_audits" / stamp
    audits.mkdir(parents=True, exist_ok=False)
    implementation = _capture_auditor(audits / "source_snapshot")
    # Only earlier derived deliveries are archived; source batch/case artifacts
    # are never rewritten, copied over, or reclassified in place.
    for name in ("delivery.json", "delivery.html"):
        if (output / name).is_file():
            _copy_file(output / name, audits / ("previous_" + name))
    rows = []
    for selection in selected:
        print(f"Offline review: {selection['task_id']}", flush=True)
        try:
            row = _review_case(selection, audits, output,
                               audits / "source_snapshot/geomind/tools/audit_assembled_robot.py")
        except Exception as exc:
            row = {"task_id": selection["task_id"], "selected": selection["selected"], "versions": selection["versions"],
                "task_completed": False, "gt_verified": False, "initial_gt_solved": False, "task_action_physics_executed": False,
                "status": "review_system_failure", "issues": [f"{type(exc).__name__}: {exc}"],
                "qualification": {"failed_gates": ["review_system_failure"]}}
        rows.append(row)
    counts = {"requested": len(rows), "selected": sum(row["selected"] is not None for row in rows),
        "completed_physics_passed": sum(row["task_completed"] for row in rows),
        "gt_verified": sum(row["gt_verified"] for row in rows),
        "initial_gt_solved_recorded": sum(row["initial_gt_solved"] for row in rows),
        "action_executed": sum(row["task_action_physics_executed"] for row in rows),
        "action_failed": sum(row["task_action_physics_executed"] and not row["task_completed"] for row in rows),
        "action_unexecuted": sum(not row["task_action_physics_executed"] for row in rows),
        "snapshot_integrity_passed": sum(row.get("integrity", {}).get("saved_source_snapshot_complete") is True
            and row.get("integrity", {}).get("saved_config_hashes_complete") is True for row in rows),
        "run_end_provenance_passed": sum(row.get("run_end_provenance", {}).get("passed") is True for row in rows),
        "current_workspace_source_drift_cases": sum(bool(row.get("integrity", {}).get("current_workspace_source_drift")) for row in rows)}
    result = {"schema_version": 1, "kind": "registered_task_delivery", "counts": counts, "cases": rows,
        "status": "physics_passed" if counts["completed_physics_passed"] == 20 else "incomplete",
        "robot": "franka_robotiq", "model_evaluated": False, "physical_reexecution_performed": False,
        "batch_roots_in_selection_order": list(map(str, roots)),
        "selection_policy": "For each exact registered task ID, the newest explicitly supplied root containing its directory wins, including incomplete or failed retries.",
        "audit_directory": str(audits), "audit_implementation": implementation,
        "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _write(audits / "delivery.json", result)
    _write(output / "delivery.json", result)
    _make_html(result, output)
    print(json.dumps({"counts": counts, "status": result["status"], "html": str(output / "delivery.html")}, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--retry", type=Path, action="append", default=[], help="Explicit retry roots, oldest to newest; may be repeated")
    parser.add_argument("--output", type=Path, help="Derived delivery directory; defaults to the base batch")
    parser.add_argument("--offline-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.offline_worker:
            return _worker_review(args.offline_worker)
        if args.batch is None:
            raise ValueError("--batch is required")
        result = deliver(args.batch, args.retry, args.output)
        return 0 if result["status"] == "physics_passed" else 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
