#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SUCCESS_LEVELS = ("loose", "medium", "strict")
# Per (task, success_level): trial_index runs 0 .. runs_per_level-1 with deterministic seeds (see derived_seed).
DEFAULT_RUNS_PER_LEVEL = 8
DEFAULT_BASE_SEED = 20260502
DEFAULT_MAX_STEPS = 40
DEFAULT_PERTURB_RANGE_M = 0.015
DEFAULT_REMOTE_POLICY_URL = "http://127.0.0.1:8001/infer"
DEFAULT_REMOTE_POLICY_PORT_MODEL_MAP = {
    "8000": "spatialforcing",
    "8001": "spatialvla",
    "8002": "internvla",
}

FALLBACK_FAMILY_THRESHOLDS_M = {
    "spatial_placement": {"loose": 0.05, "medium": 0.035, "strict": 0.025},
    "anchor_reference": {"loose": 0.045, "medium": 0.035, "strict": 0.025},
    "explicit_rule_extrapolation": {"loose": 0.05, "medium": 0.035, "strict": 0.025},
    "implicit_trend_fitting": {"loose": 0.055, "medium": 0.04, "strict": 0.03},
    "shape_completion": {"loose": 0.05, "medium": 0.035, "strict": 0.025},
}

OBJECT_MARGIN_BY_LEVEL_M = {"loose": 0.020, "medium": 0.012, "strict": 0.005}
REGION_SCALE_BY_LEVEL = {"loose": 1.30, "medium": 1.00, "strict": 0.70}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_vla_layouts(root: Path) -> dict[str, Any]:
    layouts_path = root / "geomind" / "configs" / "legacy" / "layouts" / "vla.json"
    payload = json.loads(layouts_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_vla_tasks(root: Path) -> list[dict[str, Any]]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from geomind.core.schema import normalize_task_config
    tasks_dir = root / "geomind" / "configs" / "vla_tasks"
    layouts = load_vla_layouts(root)
    tasks: list[dict[str, Any]] = []
    for path in sorted(tasks_dir.glob("*.json")):
        if path.name == "legacy_layouts.json":
            continue
        payload = normalize_task_config(json.loads(path.read_text(encoding="utf-8")),
                                        config_root=root / "geomind" / "configs")
        if payload.get("canonical_task_id") == "place_directional":
            continue
        if payload.get("implementation_status") == "placeholder":
            continue
        if not isinstance(payload, dict) or not payload.get("task_id"):
            continue
        payload["_config_path"] = str(path)
        layout_key = str(payload.get("annotation_layout_key", "")).strip()
        task_index = int(payload.get("annotation_task_index", 0) or 0)
        layout_payload = layouts.get(layout_key, {})
        if isinstance(layout_payload, dict):
            payload["_layout_payload"] = layout_payload
            layout_tasks = layout_payload.get("tasks", [])
            if isinstance(layout_tasks, list) and 0 <= task_index < len(layout_tasks):
                layout_task = layout_tasks[task_index]
                if isinstance(layout_task, dict):
                    payload["_annotation"] = layout_task.get("annotation", {})
        tasks.append(payload)
    return tasks


def parse_task_filter(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_")


def parse_port_model_map(raw: str) -> dict[str, str]:
    mapping = dict(DEFAULT_REMOTE_POLICY_PORT_MODEL_MAP)
    for item in str(raw or "").split(","):
        port, sep, model = item.strip().partition("=")
        if sep and port.strip() and model.strip():
            mapping[port.strip()] = safe_name(model.strip())
    return mapping


REMOTE_POLICY_PORT_MODEL_MAP = parse_port_model_map(os.environ.get("REMOTE_POLICY_PORT_MODEL_MAP", ""))


def remote_policy_model_from_url(url: str) -> str:
    try:
        port = str(urlparse(str(url)).port or "")
    except ValueError:
        port = ""
    if port in REMOTE_POLICY_PORT_MODEL_MAP:
        return REMOTE_POLICY_PORT_MODEL_MAP[port]
    return f"remote_{port or 'unknown'}"


def load_eval_model_config(root: Path, model_name: str, config_dir: str = "") -> dict[str, Any]:
    name = safe_name(str(model_name).strip())
    if not name:
        return {}
    base_dir = Path(config_dir) if str(config_dir).strip() else root / "geomind" / "configs" / "vla_eval_models"
    if not base_dir.is_absolute():
        base_dir = root / base_dir
    path = base_dir / f"{name}.json"
    if not path.is_file():
        return {}
    payload = read_json(path)
    payload["_config_path"] = str(path)
    return payload


SEED_DERIVATION_EXPR = "base_seed + task_index * 100003 + level_index * 1009 + trial_index"


def derived_seed(base_seed: int, task_index: int, level_index: int, trial_index: int) -> int:
    """Deterministic per-trial seed; same inputs always yield the same ``seed`` (reproducible rollouts)."""
    return int(base_seed) + int(task_index) * 100003 + int(level_index) * 1009 + int(trial_index)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def nested_dict_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def as_xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = as_float(value[0])
    y = as_float(value[1])
    if x is None or y is None:
        return None
    return (x, y)


def task_protocol(task: dict[str, Any]) -> dict[str, Any]:
    annotation = task.get("_annotation", {})
    if not isinstance(annotation, dict):
        return {}
    protocol = annotation.get("protocol", {})
    return protocol if isinstance(protocol, dict) else {}


def scene_adaptation(task: dict[str, Any]) -> dict[str, Any]:
    annotation = task.get("_annotation", {})
    if not isinstance(annotation, dict):
        return {}
    adaptation = annotation.get("scene_adaptation", {})
    return adaptation if isinstance(adaptation, dict) else {}


def runtime_objects(task: dict[str, Any]) -> list[dict[str, Any]]:
    objects = scene_adaptation(task).get("runtime_objects", [])
    return [obj for obj in objects if isinstance(obj, dict)] if isinstance(objects, list) else []


def object_by_id(task: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    wanted = str(object_id).strip()
    if not wanted:
        return None
    for obj in runtime_objects(task):
        if str(obj.get("name", "")).strip() == wanted:
            return obj
    return None


def object_contact_radius_xy(obj: dict[str, Any] | None) -> float | None:
    if not isinstance(obj, dict):
        return None
    radius = as_float(obj.get("radius"))
    if radius is not None and radius > 0.0:
        return radius
    size = obj.get("size")
    if not isinstance(size, (list, tuple)) or len(size) < 2:
        size = obj.get("dimensions")
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        sx = as_float(size[0])
        sy = as_float(size[1])
        if sx is not None and sy is not None and sx > 0.0 and sy > 0.0:
            return 0.5 * max(sx, sy)
    return None


def protocol_target_object_id(protocol: dict[str, Any]) -> str:
    gt_spec = protocol.get("gt_spec", {})
    gt_spec = gt_spec if isinstance(gt_spec, dict) else {}
    target = str(gt_spec.get("target_object_id", "")).strip()
    if target:
        return target
    target_object = protocol.get("target_object", {})
    if isinstance(target_object, dict):
        return str(target_object.get("object_id", "")).strip()
    return ""


def is_place_task(task: dict[str, Any]) -> bool:
    protocol = task_protocol(task)
    annotation = task.get("_annotation", {})
    task_names = {
        str(task.get("canonical_task_id", "")).strip().lower(),
        str(task.get("legacy_task_id", "")).strip().lower(),
        str(protocol.get("variant", "")).strip().lower(),
        str(protocol.get("task_id", "")).strip().lower(),
    }
    if isinstance(annotation, dict):
        task_names.add(str(annotation.get("task_id", "")).strip().lower())
    return any(name.startswith("place") for name in task_names if name)


def has_preloaded_gripper_target(task: dict[str, Any]) -> bool:
    protocol = task_protocol(task)
    target_object = protocol.get("target_object", {})
    if not isinstance(target_object, dict):
        return False
    preloaded = target_object.get("preloaded", False)
    if isinstance(preloaded, str):
        preloaded = preloaded.strip().lower() in {"1", "true", "yes", "y", "on"}
    return (
        str(target_object.get("carrier", "")).strip().lower() == "gripper"
        and bool(preloaded)
        and bool(str(target_object.get("object_id", "")).strip())
    )


def preview_pos_radius(task: dict[str, Any]) -> float | None:
    preview = scene_adaptation(task).get("preview_proxy_goal", {})
    if not isinstance(preview, dict):
        return None
    tolerance = preview.get("tolerance", {})
    if not isinstance(tolerance, dict):
        return None
    radius = as_float(tolerance.get("pos_r"))
    return radius if radius is not None and radius > 0.0 else None


def target_zone_center_radius(task: dict[str, Any]) -> float | None:
    protocol = task_protocol(task)
    gt_spec = protocol.get("gt_spec", {})
    gt_spec = gt_spec if isinstance(gt_spec, dict) else {}
    zone = gt_spec.get("target_zone", {})
    if not isinstance(zone, dict):
        return None
    half_extents = as_xy(zone.get("half_extents"))
    if half_extents is None:
        return None
    base = min(abs(half_extents[0]), abs(half_extents[1]))
    target_radius = object_contact_radius_xy(object_by_id(task, protocol_target_object_id(protocol))) or 0.0
    if bool(gt_spec.get("xy_aabb_fully_inside_zone", False)):
        base = max(base - target_radius, 0.0)
    return base if base > 0.0 else None


def extract_distance(result: dict[str, Any]) -> float | None:
    candidates = (
        ("evaluation", "goal_distance"),
        ("evaluation", "l2_error"),
        ("evaluation", "protocol_episode_result", "goal_distance"),
        ("evaluation", "protocol_episode_result", "l2_error"),
        ("evaluation", "protocol_episode_result", "final_l2_error"),
    )
    for path in candidates:
        value = as_float(nested_dict_get(result, path))
        if value is not None:
            return value
    return None


def extract_raw_success(result: dict[str, Any]) -> bool:
    if "success" in result:
        return bool(result.get("success"))
    evaluation = result.get("evaluation")
    if isinstance(evaluation, dict) and "success" in evaluation:
        return bool(evaluation.get("success"))
    return False


def classify_success(*, task: dict[str, Any], level: str, result: dict[str, Any]) -> dict[str, Any]:
    distance = extract_distance(result)
    raw_success = extract_raw_success(result)
    threshold = threshold_for_task_level(task, level)
    if distance is not None and threshold is not None:
        success = distance <= threshold
        method = "distance_threshold"
    else:
        success = raw_success
        method = "raw_task_success"
    return {
        "success": bool(success),
        "raw_success": bool(raw_success),
        "distance_m": distance,
        "threshold_m": threshold,
        "method": method,
    }


def threshold_for_task_level(task: dict[str, Any], level: str, fallback: float | None = None) -> float | None:
    protocol = task_protocol(task)
    gt_spec = protocol.get("gt_spec", {})
    gt_spec = gt_spec if isinstance(gt_spec, dict) else {}
    resolver = str(gt_spec.get("resolver", "")).strip().lower()

    if isinstance(gt_spec.get("target_zone", {}), dict) and gt_spec.get("target_zone"):
        region_radius = preview_pos_radius(task) or target_zone_center_radius(task)
        if region_radius is not None:
            return float(region_radius * REGION_SCALE_BY_LEVEL.get(level, REGION_SCALE_BY_LEVEL["medium"]))

    if resolver in {"move_to_object", "move_to_object_proximity", "object_proximity"}:
        target_radius = object_contact_radius_xy(object_by_id(task, protocol_target_object_id(protocol)))
        if target_radius is not None:
            return float(target_radius + OBJECT_MARGIN_BY_LEVEL_M.get(level, OBJECT_MARGIN_BY_LEVEL_M["medium"]))

    region_radius = preview_pos_radius(task) or target_zone_center_radius(task)
    if region_radius is not None:
        return float(region_radius * REGION_SCALE_BY_LEVEL.get(level, REGION_SCALE_BY_LEVEL["medium"]))

    family = str(task.get("family", "")).strip().lower()
    value = FALLBACK_FAMILY_THRESHOLDS_M.get(family, {}).get(level)
    parsed = as_float(value)
    if parsed is not None:
        return parsed
    return fallback


def reclassify_records(records: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_by_id = {str(task.get("task_id", "")): task for task in tasks}
    reclassified: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        task = task_by_id.get(str(item.get("task_id", "")))
        level = str(item.get("success_level", "")).strip()
        result_path = Path(str(item.get("result_json", "")))
        if task is not None and level in SUCCESS_LEVELS and result_path.is_file():
            result = read_json(result_path)
            classified = classify_success(task=task, level=level, result=result)
            item.update(classified)
            item["level_success"] = bool(classified["success"])
        reclassified.append(item)
    return reclassified


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(records: list[dict[str, Any]], tasks: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_task_level: dict[tuple[str, str], dict[str, Any]] = {}
    by_level: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        for level in SUCCESS_LEVELS:
            subset = [r for r in records if r.get("task_id") == task_id and r.get("success_level") == level]
            completed = [r for r in subset if r.get("run_status") == "completed"]
            attempted_trials = len(subset)
            failed_trials = sum(1 for r in subset if r.get("run_status") not in {"completed", "dry_run"})
            not_run_trials = max(int(args.runs_per_level) - attempted_trials, 0)
            success_count = sum(1 for r in completed if bool(r.get("level_success")))
            by_task_level[(task_id, level)] = {
                "task_id": task_id,
                "canonical_task_id": task.get("canonical_task_id", ""),
                "family": task.get("family", ""),
                "success_level": level,
                "planned_trials": int(args.runs_per_level),
                "attempted_trials": attempted_trials,
                "completed_trials": len(completed),
                "failed_trials": failed_trials,
                "not_run_trials": not_run_trials,
                "success_count": success_count,
                "success_rate": (success_count / len(completed)) if completed else None,
            }
    for level in SUCCESS_LEVELS:
        subset = [r for r in records if r.get("success_level") == level]
        completed = [r for r in subset if r.get("run_status") == "completed"]
        planned_trials = len(tasks) * int(args.runs_per_level)
        attempted_trials = len(subset)
        failed_trials = sum(1 for r in subset if r.get("run_status") not in {"completed", "dry_run"})
        not_run_trials = max(planned_trials - attempted_trials, 0)
        success_count = sum(1 for r in completed if bool(r.get("level_success")))
        by_level[level] = {
            "success_level": level,
            "planned_trials": planned_trials,
            "attempted_trials": attempted_trials,
            "completed_trials": len(completed),
            "failed_trials": failed_trials,
            "not_run_trials": not_run_trials,
            "success_count": success_count,
            "success_rate": (success_count / len(completed)) if completed else None,
        }
    planned_trials = len(tasks) * len(SUCCESS_LEVELS) * int(args.runs_per_level)
    attempted_trials = len(records)
    failed_trials = sum(1 for r in records if r.get("run_status") not in {"completed", "dry_run"})
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(Path(args.output_root)),
        "policy": args.policy,
        "backend": args.backend,
        "max_steps": int(args.max_steps),
        "base_seed": int(args.base_seed),
        "runs_per_level": int(args.runs_per_level),
        "native_object_pose_perturb_range_m": float(args.native_object_pose_perturb_range),
        "success_threshold_policy": {
            "object_margin_by_level_m": OBJECT_MARGIN_BY_LEVEL_M,
            "region_scale_by_level": REGION_SCALE_BY_LEVEL,
            "fallback_family_thresholds_m": FALLBACK_FAMILY_THRESHOLDS_M,
        },
        "task_count": len(tasks),
        "planned_trials": planned_trials,
        "attempted_trials": attempted_trials,
        "completed_trials": sum(1 for r in records if r.get("run_status") == "completed"),
        "failed_trials": failed_trials,
        "not_run_trials": max(planned_trials - attempted_trials, 0),
        "by_level": list(by_level.values()),
        "by_task_level": list(by_task_level.values()),
    }


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = list(summary.get("by_task_level", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "canonical_task_id",
                "family",
                "success_level",
                "planned_trials",
                "attempted_trials",
                "completed_trials",
                "failed_trials",
                "not_run_trials",
                "success_count",
                "success_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def fmt_rate(value: Any) -> str:
    number = as_float(value)
    return "n/a" if number is None else f"{number:.3f}"


def write_markdown(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    failures = [r for r in records if r.get("run_status") != "completed"]
    lines = [
        "# VLA Eval Spatiaforcing Summary",
        "",
        f"- created_at: {summary.get('created_at')}",
        f"- policy: {summary.get('policy')}",
        f"- backend: {summary.get('backend')}",
        f"- max_steps: {summary.get('max_steps')}",
        f"- task_count: {summary.get('task_count')}",
        f"- planned_trials: {summary.get('planned_trials')}",
        f"- attempted_trials: {summary.get('attempted_trials')}",
        f"- completed_trials: {summary.get('completed_trials')}",
        f"- failed_trials: {summary.get('failed_trials')}",
        f"- not_run_trials: {summary.get('not_run_trials')}",
        f"- base_seed: {summary.get('base_seed')}",
        f"- native_object_pose_perturb_range_m: {summary.get('native_object_pose_perturb_range_m')}",
        "",
        "## Overall By Success Level",
        "",
        "| level | attempted | completed | failed | not_run | success | success_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("by_level", []):
        lines.append(
            "| {success_level} | {attempted_trials} | {completed_trials} | {failed_trials} | {not_run_trials} | {success_count} | {success_rate} |".format(
                **{**row, "success_rate": fmt_rate(row.get("success_rate"))}
            )
        )
    lines.extend(
        [
            "",
            "## Per Task",
            "",
            "| task | family | level | attempted | completed | failed | not_run | success | success_rate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("by_task_level", []):
        lines.append(
            "| {task_id} | {family} | {success_level} | {attempted_trials} | {completed_trials} | {failed_trials} | {not_run_trials} | {success_count} | {success_rate} |".format(
                **{**row, "success_rate": fmt_rate(row.get("success_rate"))}
            )
        )
    lines.extend(["", "## Notes", ""])
    if int(summary.get("not_run_trials", 0) or 0) > 0:
        lines.append(
            f"- Run is incomplete: {summary.get('not_run_trials')} planned trials have not been started yet. "
            "Re-run the same command with resume enabled to continue from existing result files."
        )
    if failures:
        first = failures[0]
        lines.append(
            f"- Runs are incomplete. First failure: task={first.get('task_id')} level={first.get('success_level')} "
            f"trial={first.get('trial_index')} seed={first.get('seed')} returncode={first.get('returncode')}."
        )
        lines.append(f"- First failure log: `{first.get('run_log', '')}`")
    elif int(summary.get("not_run_trials", 0) or 0) == 0:
        lines.append("- All planned trials completed.")
    lines.append(
        "- Continuous placement/geometric tasks use family-specific distance thresholds for loose/medium/strict. "
        "Discrete selection/tracking tasks fall back to the task's raw success flag when no distance metric is available."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_trial(
    *,
    root: Path,
    task: dict[str, Any],
    task_index: int,
    level: str,
    level_index: int,
    trial_index: int,
    args: argparse.Namespace,
    task_list_fingerprint: str = "",
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    max_steps = max(int(args.max_steps), int(task.get("minimum_rollout_steps", 0)))
    ablation_id = str(getattr(args, "ablation", "base") or "base")
    pair_id = f"{task_id}|{level}|{int(trial_index)}"
    seed = derived_seed(int(args.base_seed), task_index, level_index, trial_index)
    output_root = Path(args.output_root).resolve()
    trial_root = output_root / "runs" / safe_name(task_id) / level / f"seed_{seed}"
    episode_dir = trial_root / "episode_0000"
    result_path = episode_dir / "result.json"
    run_log = trial_root / "run.log"
    perturb_range = 0.0 if has_preloaded_gripper_target(task) else float(args.native_object_pose_perturb_range)
    metadata = {
        "task_id": task_id,
        "canonical_task_id": task.get("canonical_task_id", ""),
        "family": task.get("family", ""),
        "success_level": level,
        "trial_index": int(trial_index),
        "seed": int(seed),
        "base_seed": int(args.base_seed),
        "task_index": int(task_index),
        "level_index": int(level_index),
        "seed_derivation": SEED_DERIVATION_EXPR,
        "task_list_fingerprint": str(task_list_fingerprint),
        "ablation_id": ablation_id,
        "pair_id": pair_id,
        "max_steps": max_steps,
        "remote_policy_url": str(args.remote_policy_url),
        "remote_policy_model": str(args.remote_policy_model),
        "remote_policy_action_scale": float(args.remote_policy_action_scale),
        "remote_policy_disable_rotation": bool(getattr(args, "remote_policy_disable_rotation", False)),
        "remote_policy_gripper_mode": str(getattr(args, "remote_policy_gripper_mode", "")),
        "native_object_pose_perturb_range_m": perturb_range,
        "target_preloaded_in_gripper": has_preloaded_gripper_target(task),
        "move_to_success_radius_max_m": threshold_for_task_level(
            task,
            level,
            fallback=float(args.move_to_success_radius_max) if float(args.move_to_success_radius_max) > 0.0 else None,
        ),
        "output_root": str(trial_root),
        "episode_dir": str(episode_dir),
    }
    # Resume before touching trial_metadata or launching IsaacLab (subprocess).
    if args.resume and result_path.is_file():
        write_json(trial_root / "trial_metadata.json", metadata)
        result = read_json(result_path)
        classified = classify_success(task=task, level=level, result=result)
        return {
            **metadata,
            **classified,
            "level_success": classified["success"],
            "run_status": "completed",
            "returncode": 0,
            "result_json": str(result_path),
            "run_log": str(run_log),
            "resumed": True,
        }

    write_json(trial_root / "trial_metadata.json", metadata)

    if args.dry_run:
        return {
            **metadata,
            "level_success": False,
            "run_status": "dry_run",
            "returncode": None,
            "result_json": str(result_path),
            "run_log": str(run_log),
            "resumed": False,
        }

    env = dict(os.environ)
    env.update(
        {
            "TASK_ID_ENV": task_id,
            "POLICY": str(args.policy),
            "BACKEND": str(args.backend),
            "MAX_STEPS": str(max_steps),
            "SEED": str(int(seed)),
            "ABLATION_ID": ablation_id,
            "PAIR_ID": pair_id,
            "RUN_NAME": f"{safe_name(args.remote_policy_model)}_vla_eval_{safe_name(task_id)}_{level}_seed_{seed}",
            "OUTPUT_ROOT": str(trial_root),
            "OUTPUT_DIR": str(episode_dir),
            "EPISODE_INDEX": "0",
            "REMOTE_POLICY_URL": str(args.remote_policy_url),
            "REMOTE_POLICY_MODEL": str(args.remote_policy_model),
            "REMOTE_POLICY_ACTION_SCALE": str(float(args.remote_policy_action_scale)),
            "REMOTE_POLICY_DISABLE_ROTATION": "1" if bool(getattr(args, "remote_policy_disable_rotation", False)) else "0",
            "REMOTE_POLICY_GRIPPER_MODE": str(getattr(args, "remote_policy_gripper_mode", "") or "isaac"),
            "REMOTE_POLICY_PORT_MODEL_MAP": ",".join(
                f"{port}={name}" for port, name in REMOTE_POLICY_PORT_MODEL_MAP.items()
            ),
            **(
                {"OPENPI_WEBSOCKET_URL": str(args.openpi_websocket_url).strip()}
                if str(getattr(args, "openpi_websocket_url", "") or "").strip()
                else {}
            ),
            "NATIVE_OBJECT_POSE_PERTURB_RANGE": str(perturb_range),
            "GEOMIND_MOVE_TO_SUCCESS_RADIUS_MAX": str(metadata["move_to_success_radius_max_m"] or 0.0),
            # Bash runner exits before conda/IsaacLab if result.json exists (defense in depth with --resume).
            "VLA_EVAL_SKIP_IF_RESULT_JSON": "1" if args.resume else "0",
        }
    )
    cmd = ["bash", str(root / "scripts" / "run_remote_policy_isaaclab_vla.sh")]
    completed = subprocess.run(cmd, cwd=str(root), env=env)
    if completed.returncode != 0 or not result_path.is_file():
        return {
            **metadata,
            "level_success": False,
            "raw_success": False,
            "distance_m": None,
            "threshold_m": threshold_for_task_level(task, level),
            "method": "run_failed",
            "run_status": "failed",
            "returncode": int(completed.returncode),
            "result_json": str(result_path),
            "run_log": str(run_log),
            "resumed": False,
        }
    result = read_json(result_path)
    classified = classify_success(task=task, level=level, result=result)
    return {
        **metadata,
        **classified,
        "level_success": classified["success"],
        "run_status": "completed",
        "returncode": int(completed.returncode),
        "result_json": str(result_path),
        "run_log": str(run_log),
        "resumed": False,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run all GeoMind VLA tasks with reproducible seed sweeps.")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--eval-config-dir", default=os.environ.get("VLA_EVAL_CONFIG_DIR", ""))
    parser.add_argument("--remote-policy-url", default=os.environ.get("REMOTE_POLICY_URL", DEFAULT_REMOTE_POLICY_URL))
    parser.add_argument(
        "--remote-policy-model",
        default=(os.environ.get("VLA_EVAL_MODEL") or os.environ.get("REMOTE_POLICY_MODEL", "")).strip(),
        help="Selects geomind/configs/vla_eval_models/<name>.json. Defaults: VLA_EVAL_MODEL, else REMOTE_POLICY_MODEL, else inferred from --remote-policy-url port map.",
    )
    parser.add_argument("--remote-policy-action-scale", type=float, default=None)
    parser.add_argument(
        "--runs-per-level",
        type=int,
        default=int(os.environ.get("VLA_EVAL_RUNS_PER_LEVEL", DEFAULT_RUNS_PER_LEVEL)),
        help=(
            "Rollouts per (task, success_level); each uses trial_index 0..N-1 and a deterministic seed from "
            f"{SEED_DERIVATION_EXPR}."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.environ.get("MAX_STEPS", str(DEFAULT_MAX_STEPS))),
    )
    parser.add_argument("--base-seed", type=int, default=int(os.environ.get("VLA_EVAL_BASE_SEED", DEFAULT_BASE_SEED)))
    parser.add_argument("--policy", default=os.environ.get("POLICY", "model"))
    parser.add_argument("--backend", default=os.environ.get("BACKEND", "isaaclab"))
    parser.add_argument(
        "--native-object-pose-perturb-range",
        type=float,
        default=float(os.environ.get("NATIVE_OBJECT_POSE_PERTURB_RANGE", DEFAULT_PERTURB_RANGE_M)),
    )
    parser.add_argument(
        "--move-to-success-radius-max",
        type=float,
        default=float(os.environ.get("GEOMIND_MOVE_TO_SUCCESS_RADIUS_MAX", "0.045")),
        help="Clamp native move-to-object success radius in meters. Use <=0 to disable clamp.",
    )
    parser.add_argument("--tasks", default="", help="Comma-separated task ids or canonical task ids. Default: all VLA tasks.")
    parser.add_argument(
        "--ablation",
        default=os.environ.get("VLA_EVAL_ABLATION", "base"),
        choices=("base", "direct"),
        help=(
            "'direct' runs the direct-reference variant: identical scene/poses/seed, only the instruction "
            "and target appearance change. Default output root gets a '_direct' suffix."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="Rebuild summary files from trial_results.jsonl.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parse_args(raw_argv)
    root = repo_root()

    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, str):
            raw = value.strip().lower()
            if not raw:
                return bool(default)
            return raw in {"1", "true", "yes", "on"}
        return bool(value)

    url_overridden = bool(os.environ.get("REMOTE_POLICY_URL_OVERRIDE")) or any(
        arg == "--remote-policy-url" or arg.startswith("--remote-policy-url=") for arg in raw_argv
    )
    action_scale_overridden = bool(os.environ.get("REMOTE_POLICY_ACTION_SCALE_OVERRIDE")) or any(
        arg == "--remote-policy-action-scale" or arg.startswith("--remote-policy-action-scale=") for arg in raw_argv
    )
    if not str(args.remote_policy_model).strip():
        args.remote_policy_model = remote_policy_model_from_url(str(args.remote_policy_url))
    eval_model_config = load_eval_model_config(root, str(args.remote_policy_model), str(args.eval_config_dir))
    args.openpi_websocket_url = str((eval_model_config or {}).get("openpi_websocket_url", "") or "").strip()
    if eval_model_config:
        args.remote_policy_model = str(eval_model_config.get("model_name") or args.remote_policy_model)
        if not url_overridden and eval_model_config.get("remote_policy_url"):
            args.remote_policy_url = str(eval_model_config["remote_policy_url"])
        elif not url_overridden and not eval_model_config.get("remote_policy_url"):
            # Libero-style host + port (e.g. openpi_eval_models/*.json) without remote_policy_url.
            # OpenPI's default policy server speaks WebSocket on this port, not HTTP POST /infer.
            hk = str(eval_model_config.get("host", "")).strip()
            pr = eval_model_config.get("port")
            if hk and pr is not None and str(pr).strip() != "":
                try:
                    pn = int(pr)
                    args.remote_policy_url = f"ws://{hk}:{pn}"
                except (TypeError, ValueError):
                    pass
        if not action_scale_overridden and eval_model_config.get("remote_policy_action_scale") is not None:
            args.remote_policy_action_scale = float(eval_model_config["remote_policy_action_scale"])
    args.remote_policy_disable_rotation = _as_bool(
        eval_model_config.get(
            "remote_policy_disable_rotation",
            str(os.environ.get("REMOTE_POLICY_DISABLE_ROTATION", "")).strip().lower() in {"1", "true", "yes", "on"},
        )
    )
    args.remote_policy_gripper_mode = str(
        eval_model_config.get("remote_policy_gripper_mode", os.environ.get("REMOTE_POLICY_GRIPPER_MODE", "isaac"))
        or "isaac"
    )
    mn_lc = str(args.remote_policy_model).strip().lower()
    if mn_lc.startswith("openpi") and not url_overridden:
        # OpenPI policy servers are WebSocket-only; normalize legacy http(s)://…/infer configs.
        w = str(args.openpi_websocket_url or "").strip()
        r = str(args.remote_policy_url or "").strip()
        w_scheme = (urlparse(w).scheme or "").lower() if w else ""
        r_scheme = (urlparse(r).scheme or "").lower() if r else ""
        if w and w_scheme in ("ws", "wss"):
            args.remote_policy_url = w
        elif r and r_scheme in ("http", "https"):
            p = urlparse(r)
            if p.hostname and p.port is not None:
                args.remote_policy_url = f"ws://{p.hostname}:{p.port}"
        args.openpi_websocket_url = ""
    if args.remote_policy_action_scale is None:
        args.remote_policy_action_scale = float(
            os.environ.get("REMOTE_POLICY_ACTION_SCALE_OVERRIDE", os.environ.get("REMOTE_POLICY_ACTION_SCALE", "1.0"))
        )
    if not str(args.output_root).strip():
        args.output_root = str(eval_model_config.get("output_dir_name") or f"run_eval_{safe_name(str(args.remote_policy_model))}")
        if str(args.ablation) == "direct":
            args.output_root = f"{args.output_root}_direct"
    output_root = (root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    args.output_root = str(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = load_vla_tasks(root)
    # task_index feeds the seed derivation, so it must be the position in the *full*, unfiltered
    # task list. Deriving it from the filtered list made --tasks silently change every seed and
    # broke base/direct pairing.
    canonical_task_ids = [str(task.get("task_id", "")) for task in tasks]
    task_index_by_id = {task_id: index for index, task_id in enumerate(canonical_task_ids)}
    task_list_fingerprint = hashlib.sha256("\n".join(canonical_task_ids).encode("utf-8")).hexdigest()
    selected = parse_task_filter(args.tasks)
    if selected:
        tasks = [
            task
            for task in tasks
            if str(task.get("task_id", "")) in selected or str(task.get("canonical_task_id", "")) in selected
            or str(task.get("public_task_id", "")) in selected
        ]
    if not tasks:
        raise SystemExit("No VLA tasks selected.")

    if args.summarize_only:
        records_path = output_root / "trial_results.jsonl"
        records = []
        if records_path.is_file():
            records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = reclassify_records(records, tasks)
        write_jsonl(records_path, records)
        summary = summarize(records, tasks, args)
        write_json(output_root / "summary.json", summary)
        write_csv(output_root / "summary_by_task_level.csv", summary)
        write_markdown(output_root / "RUN_SUMMARY.md", summary, records)
        return 0

    write_json(
        output_root / "eval_config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "eval_model_config": eval_model_config,
            "remote_policy_port_model_map": REMOTE_POLICY_PORT_MODEL_MAP,
            "success_levels": SUCCESS_LEVELS,
            "success_threshold_policy": {
                "object_margin_by_level_m": OBJECT_MARGIN_BY_LEVEL_M,
                "region_scale_by_level": REGION_SCALE_BY_LEVEL,
                "fallback_family_thresholds_m": FALLBACK_FAMILY_THRESHOLDS_M,
            },
            "seed_schedule": {
                "base_seed": int(args.base_seed),
                "runs_per_level": int(args.runs_per_level),
                "success_levels": list(SUCCESS_LEVELS),
                "derivation": SEED_DERIVATION_EXPR,
                "task_index_source": "position in the full unfiltered load_vla_tasks() list",
                "task_list_fingerprint": task_list_fingerprint,
                "repro_note": (
                    "Re-run with the same --base-seed, task order, and --runs-per-level to reproduce every trial; "
                    "each line in trial_results.jsonl includes seed, task_index, level_index, trial_index."
                ),
            },
            "tasks": [
                {
                    "task_id": task.get("task_id", ""),
                    "canonical_task_id": task.get("canonical_task_id", ""),
                    "family": task.get("family", ""),
                    "config_path": task.get("_config_path", ""),
                    "thresholds_m": {
                        level: threshold_for_task_level(task, level)
                        for level in SUCCESS_LEVELS
                    },
                }
                for task in tasks
            ],
        },
    )

    records: list[dict[str, Any]] = []
    for task in tasks:
        task_index = task_index_by_id[str(task.get("task_id", ""))]
        for level_index, level in enumerate(SUCCESS_LEVELS):
            for trial_index in range(int(args.runs_per_level)):
                record = run_trial(
                    root=root,
                    task=task,
                    task_index=task_index,
                    level=level,
                    level_index=level_index,
                    trial_index=trial_index,
                    args=args,
                    task_list_fingerprint=task_list_fingerprint,
                )
                records.append(record)
                write_jsonl(output_root / "trial_results.jsonl", records)
                if args.stop_on_error and record.get("run_status") == "failed":
                    summary = summarize(records, tasks, args)
                    write_json(output_root / "summary.json", summary)
                    write_csv(output_root / "summary_by_task_level.csv", summary)
                    write_markdown(output_root / "RUN_SUMMARY.md", summary, records)
                    return 1

    summary = summarize(records, tasks, args)
    write_json(output_root / "summary.json", summary)
    write_csv(output_root / "summary_by_task_level.csv", summary)
    write_markdown(output_root / "RUN_SUMMARY.md", summary, records)
    if args.dry_run:
        return 0
    return 0 if all(r.get("run_status") == "completed" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
