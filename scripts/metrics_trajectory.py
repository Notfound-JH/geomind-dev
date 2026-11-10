#!/usr/bin/env python3
"""
Trajectory-level target tracking 后处理诊断。

仅读取 trial_results.jsonl、result.json、layout、trace.json（必须含完整 eef_pos 序列）。
不修改环境 / task JSON；不把 layout_constraints 当 SCS；不把 nearest_final_eef 当作 pred 主指标。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import metrics_lite as ml  # noqa: E402


def _layout_suggests_dynamic_objects(
    layout_key: str,
    task_payload: dict[str, Any] | None,
    *,
    result: dict[str, Any] | None,
) -> bool:
    """用于标记 static_layout_proxy：仅基于 layout / task / result 的显式字段，避免误报。"""
    tokens = ("track", "clutter", "reindex", "scene_update", "identity_track")
    blobs: list[str] = [layout_key or ""]
    if isinstance(result, dict):
        for k in (
            "canonical_task_id",
            "legacy_task_id",
            "task_id",
            "semantic_module",
            "skill_type",
            "evaluator_key",
        ):
            blobs.append(str(result.get(k) or ""))
    if isinstance(task_payload, dict):
        ann = task_payload.get("annotation")
        if isinstance(ann, dict):
            for k in ("family", "taxonomy", "task_type"):
                blobs.append(str(ann.get(k) or ""))
    hay = " ".join(blobs).lower()
    return any(tok in hay for tok in tokens)


def _policy_from_step(step: dict[str, Any]) -> dict[str, Any]:
    aos = step.get("agent_obs")
    if not isinstance(aos, dict):
        return {}
    st = aos.get("state")
    if not isinstance(st, dict):
        return {}
    no = st.get("native_obs")
    if not isinstance(no, dict):
        return {}
    pol = no.get("policy")
    return pol if isinstance(pol, dict) else {}


def _parse_object_poses_explicit(
    pol: dict[str, Any], default_z: float
) -> dict[str, tuple[float, float, float]] | None:
    for key in ("object_poses", "object_pose_world", "object_world_poses"):
        raw = pol.get(key)
        if raw is None:
            continue
        if isinstance(raw, dict):
            out: dict[str, tuple[float, float, float]] = {}
            for oid, val in raw.items():
                v = ml._vec3_from_value(val, default_z)  # type: ignore[attr-defined]
                if v is not None:
                    out[str(oid)] = v
            return out or None
        if isinstance(raw, list):
            out = {}
            for item in raw:
                if not isinstance(item, dict):
                    continue
                oid = str(item.get("object_id") or item.get("name") or item.get("id") or "").strip()
                if not oid:
                    continue
                pos = item.get("position") or item.get("pos") or item.get("xyz")
                v = ml._vec3_from_value(pos, default_z)  # type: ignore[attr-defined]
                if v is None and "pose" in item:
                    v = ml._vec3_from_value(item.get("pose"), default_z)  # type: ignore[attr-defined]
                if v is not None:
                    out[oid] = v
            return out or None
    return None


def _parse_policy_object_tensor(
    pol: dict[str, Any], ordered_names: list[str]
) -> dict[str, tuple[float, float, float]] | None:
    raw = pol.get("object")
    if isinstance(raw, dict) and raw.get("omitted"):
        return None
    if not isinstance(raw, list) or not raw or not ordered_names:
        return None
    n = len(ordered_names)
    if len(raw) % n != 0:
        return None
    dpo = len(raw) // n
    if dpo < 3:
        return None
    out: dict[str, tuple[float, float, float]] = {}
    for i, name in enumerate(ordered_names):
        base = i * dpo
        try:
            out[name] = (float(raw[base]), float(raw[base + 1]), float(raw[base + 2]))
        except (TypeError, ValueError, IndexError):
            return None
    return out


def _eef_from_step(step: dict[str, Any]) -> tuple[float, float, float] | None:
    pol = _policy_from_step(step)
    eef = pol.get("eef_pos")
    if isinstance(eef, (list, tuple)) and len(eef) >= 3:
        try:
            return (float(eef[0]), float(eef[1]), float(eef[2]))
        except (TypeError, ValueError):
            return None
    return None


def _load_trace_list(path: Path) -> list[dict[str, Any]] | None:
    data = ml._read_json(path)  # type: ignore[attr-defined]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return None


def _resolve_trace_path(
    trial: dict[str, Any],
    result_path: Path,
    *,
    eval_root: Path,
    traces_mirror: Path | None,
) -> Path | None:
    ep_dir = trial.get("episode_dir")
    ep_dir_s = str(ep_dir) if isinstance(ep_dir, str) else None
    for tp in ml._iter_trace_candidates(  # type: ignore[attr-defined]
        result_path=result_path,
        episode_dir=ep_dir_s,
        eval_root=eval_root,
        traces_mirror=traces_mirror,
    ):
        if tp.is_file():
            return tp
    return None


def _per_step_object_positions(
    trace: list[dict[str, Any]],
    ordered_names: list[str],
    static_pos: dict[str, tuple[float, float, float]],
    default_z: float,
) -> tuple[list[dict[str, tuple[float, float, float]]], str, str | None]:
    """
    返回 (每步物体 xyz 字典, object_pos_source, parse_note)。
    优先 object_poses；其次 policy.object 张量；否则每步使用 layout 静态位姿。
    """
    if not trace:
        return [], "static_layout", "empty_trace"

    series_explicit: list[dict[str, tuple[float, float, float]] | None] = []
    for step in trace:
        pol = _policy_from_step(step)
        series_explicit.append(_parse_object_poses_explicit(pol, default_z))
    if all(p is not None for p in series_explicit) and series_explicit:
        return [p for p in series_explicit if p is not None], "dynamic_trace_object_poses", None

    series_tensor: list[dict[str, tuple[float, float, float]] | None] = []
    for step in trace:
        pol = _policy_from_step(step)
        series_tensor.append(_parse_policy_object_tensor(pol, ordered_names))
    if all(p is not None for p in series_tensor) and series_tensor:
        return [p for p in series_tensor if p is not None], "dynamic_trace_policy_object", None

    static_series = [dict(static_pos) for _ in trace]
    note: str | None = None
    pol0 = _policy_from_step(trace[0])
    raw = pol0.get("object")
    if isinstance(raw, dict) and raw.get("omitted"):
        note = "policy_object_omitted_in_trace"
    elif isinstance(raw, list) and ordered_names and len(raw) % len(ordered_names) != 0:
        note = "policy_object_length_not_multiple_of_num_objects"
    elif series_explicit and any(p is not None for p in series_explicit):
        note = "incomplete_object_poses_sequence_fell_back_static"
    elif any(p is not None for p in series_tensor):
        note = "policy_object_tensor_parse_failed_fell_back_static"
    return static_series, "static_layout", note


def _gt_success_radius(annotation: dict[str, Any] | None, result: dict[str, Any], default_radius: float) -> float:
    proto = (annotation or {}).get("protocol") if isinstance(annotation, dict) else None
    proto = proto if isinstance(proto, dict) else {}
    gt_spec = proto.get("gt_spec") if isinstance(proto.get("gt_spec"), dict) else {}
    sr = gt_spec.get("success_radius")
    try:
        success_radius = float(sr) if sr is not None else default_radius
    except (TypeError, ValueError):
        success_radius = default_radius
    if success_radius == default_radius:
        ev = result.get("evaluation")
        if isinstance(ev, dict) and ev.get("success_radius") is not None:
            try:
                success_radius = float(ev["success_radius"])
            except (TypeError, ValueError):
                pass
    return success_radius


def process_episode_trajectory(
    trial: dict[str, Any],
    layout_bundle: dict[str, Any],
    *,
    eval_root: Path,
    traces_mirror: Path | None,
    default_z: float,
    default_radius: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {k: v for k, v in trial.items() if not str(k).startswith("_")}
    row["metrics_trajectory_version"] = 1

    def _fail(reason: str, **extra: Any) -> dict[str, Any]:
        row["trajectory_valid"] = False
        row["trajectory_missing_reason"] = reason
        row.update(extra)
        for k in (
            "mean_target_distance",
            "final_target_distance",
            "min_target_distance",
            "target_near_fraction",
            "target_closest_rate",
            "mean_target_margin",
        ):
            if k not in row:
                row[k] = None
        return row

    if trial.get("_parse_error"):
        return _fail("trial_json_parse_error")

    rp = trial.get("result_json")
    result_path = Path(str(rp)) if isinstance(rp, str) and rp.strip() else None
    result = ml._read_json(result_path) if result_path is not None else None  # type: ignore[attr-defined]
    if not isinstance(result, dict):
        return _fail("missing_or_unreadable_result_json")

    layout_key = ml._layout_key_from_result(result)  # type: ignore[attr-defined]
    tidx = ml._task_index_from_result(result)  # type: ignore[attr-defined]
    resolved_key: str | None = None
    annotation: dict[str, Any] | None = None
    task_payload: dict[str, Any] | None = None
    layout_entry: dict[str, Any] | None = None
    positions: dict[str, tuple[float, float, float]] = {}

    if layout_key and layout_bundle:
        resolved_key, layout_entry = ml.resolve_layout(layout_bundle, layout_key)  # type: ignore[attr-defined]
        if layout_entry and isinstance(layout_entry.get("tasks"), list):
            tasks = layout_entry["tasks"]
            if 0 <= tidx < len(tasks) and isinstance(tasks[tidx], dict):
                task_payload = tasks[tidx]
                ann = task_payload.get("annotation")
                if isinstance(ann, dict):
                    annotation = ann
                    positions = ml.build_object_positions(layout_entry, task_payload, default_z)  # type: ignore[attr-defined]

    row["layout_key_requested"] = layout_key or None
    row["layout_key_resolved"] = resolved_key
    row["annotation_task_index"] = tidx

    gt_id = ml.infer_gt_target_object_id(annotation) if isinstance(annotation, dict) else None  # type: ignore[attr-defined]
    row["gt_target_object_id"] = gt_id

    proto = (annotation or {}).get("protocol") if isinstance(annotation, dict) else {}
    proto = proto if isinstance(proto, dict) else {}
    gt_spec = proto.get("gt_spec") if isinstance(proto.get("gt_spec"), dict) else {}
    xy_only = bool(gt_spec.get("xy_only"))
    success_radius = _gt_success_radius(annotation, result, default_radius)
    row["success_radius_used"] = success_radius
    row["xy_only"] = xy_only

    if result_path is None:
        return _fail("missing_result_json_path")

    trace_path = _resolve_trace_path(trial, result_path, eval_root=eval_root, traces_mirror=traces_mirror)
    row["trace_path_used"] = str(trace_path) if trace_path else None
    if trace_path is None:
        return _fail("missing_trace_json")

    trace = _load_trace_list(trace_path)
    if trace is None:
        return _fail("trace_json_unreadable")
    if not trace:
        return _fail("empty_trace_json")

    eef_seq: list[tuple[float, float, float] | None] = []
    for step in trace:
        eef_seq.append(_eef_from_step(step))
    if any(e is None for e in eef_seq):
        return _fail("incomplete_eef_pos_sequence")
    eef_list = [e for e in eef_seq if e is not None]

    ordered_names = list(positions.keys())
    static_pos = dict(positions)
    step_poses, object_pos_source, dyn_note = _per_step_object_positions(
        trace, ordered_names, static_pos, default_z
    )
    row["object_pos_source"] = object_pos_source
    row["object_pos_parse_note"] = dyn_note

    suggests_dyn = _layout_suggests_dynamic_objects(layout_key or "", task_payload, result=result)
    row["static_layout_proxy"] = bool(object_pos_source == "static_layout" and suggests_dyn)
    row["layout_suggests_dynamic_objects"] = suggests_dyn

    if not gt_id:
        return _fail("missing_gt_target_object_id", object_pos_source=object_pos_source)

    gt_static = static_pos.get(gt_id)
    if gt_static is None and object_pos_source == "static_layout":
        return _fail("missing_gt_target_object_pos", object_pos_source=object_pos_source)

    T = len(eef_list)
    if len(step_poses) != T:
        return _fail("trace_length_mismatch_eef_vs_object_poses", object_pos_source=object_pos_source)

    dists: list[float] = []
    near_ct = 0
    closest_hits = 0
    margin_terms: list[float] = []
    distractor_ids = [oid for oid in ordered_names if oid != gt_id]
    can_multi_object = len(distractor_ids) > 0
    mult_ok = True
    if can_multi_object:
        for t in range(T):
            pt = step_poses[t]
            if not all(oid in pt for oid in ordered_names):
                mult_ok = False
                break

    for t in range(T):
        eef = eef_list[t]
        poses_t = step_poses[t]
        if gt_id in poses_t:
            tgt = poses_t[gt_id]
        elif gt_static is not None:
            tgt = gt_static
        else:
            return _fail("missing_gt_target_object_pos", object_pos_source=object_pos_source, timestep=t)

        d = ml.distance_eef_to_target(eef, tgt, xy_only)  # type: ignore[attr-defined]
        dists.append(d)
        if d <= success_radius:
            near_ct += 1

        if can_multi_object and mult_ok:
            best_oid: str | None = None
            best_d = math.inf
            for oid in ordered_names:
                pos = poses_t.get(oid)
                if pos is None:
                    continue
                dd = ml.distance_eef_to_target(eef, pos, xy_only)  # type: ignore[attr-defined]
                if dd < best_d:
                    best_d = dd
                    best_oid = oid
            if best_oid == gt_id:
                closest_hits += 1

            d_targ = ml.distance_eef_to_target(eef, tgt, xy_only)  # type: ignore[attr-defined]
            d_other = [
                ml.distance_eef_to_target(eef, poses_t[oid], xy_only)  # type: ignore[attr-defined]
                for oid in distractor_ids
            ]
            margin_terms.append(min(d_other) - d_targ)

    mean_d = sum(dists) / len(dists)
    final_d = dists[-1]
    min_d = min(dists)
    near_frac = near_ct / len(dists)

    row["mean_target_distance"] = mean_d
    row["final_target_distance"] = final_d
    row["min_target_distance"] = min_d
    row["target_near_fraction"] = near_frac

    if not can_multi_object:
        row["target_closest_rate"] = None
        row["mean_target_margin"] = None
        row["trajectory_mult_object_metrics_reason"] = "no_distractor_objects_in_layout"
    elif not mult_ok:
        row["target_closest_rate"] = None
        row["mean_target_margin"] = None
        row["trajectory_mult_object_metrics_reason"] = "incomplete_object_positions_for_mult_metrics"
    elif not margin_terms:
        row["target_closest_rate"] = None
        row["mean_target_margin"] = None
        row["trajectory_mult_object_metrics_reason"] = "no_margin_timesteps"
    else:
        row["target_closest_rate"] = closest_hits / T
        row["mean_target_margin"] = sum(margin_terms) / len(margin_terms)
        row["trajectory_mult_object_metrics_reason"] = None

    row["trajectory_timesteps"] = T
    row["trajectory_valid"] = True
    row["trajectory_missing_reason"] = None
    return row


def summarize_trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    valid = [r for r in rows if r.get("trajectory_valid") is True]
    n_valid = len(valid)

    def _pull(key: str) -> list[float]:
        xs: list[float] = []
        for r in valid:
            v = r.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                xs.append(float(v))
        return xs

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    def _median(xs: list[float]) -> float | None:
        return float(statistics.median(xs)) if xs else None

    mtd = _pull("mean_target_distance")
    tcr_vals = [float(r["target_closest_rate"]) for r in valid if r.get("target_closest_rate") is not None]
    mm_vals = [float(r["mean_target_margin"]) for r in valid if r.get("mean_target_margin") is not None]

    miss = Counter()
    for r in rows:
        mr = r.get("trajectory_missing_reason")
        if isinstance(mr, str) and mr:
            miss[mr] += 1

    return {
        "total_episodes": total,
        "valid_trajectory_metric_count": n_valid,
        "mean_target_distance": _mean(mtd),
        "median_target_distance": _median(mtd),
        "final_target_distance_mean": _mean(_pull("final_target_distance")),
        "min_target_distance_mean": _mean(_pull("min_target_distance")),
        "target_near_fraction_mean": _mean(_pull("target_near_fraction")),
        "target_closest_rate_mean": _mean(tcr_vals),
        "mean_target_margin": _mean(mm_vals),
        "missing_reason_counts": dict(miss),
        "object_pos_source_counts": dict(Counter(str(r.get("object_pos_source") or "") for r in rows)),
        "static_layout_proxy_count": sum(1 for r in rows if r.get("static_layout_proxy") is True),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Trajectory-level target tracking 诊断（后处理）")
    p.add_argument(
        "--load-trace-eef",
        action="store_true",
        help="必须指定：从 trace.json 读取完整 eef_pos 时间序列（与 metrics_lite 习惯一致；未指定则退出）",
    )
    p.add_argument("--results", type=Path, required=True, help="trial_results.jsonl")
    p.add_argument(
        "--layouts",
        type=Path,
        action="append",
        required=True,
        help="legacy_layouts.json（可多次指定）",
    )
    p.add_argument(
        "--traces",
        type=Path,
        default=None,
        help="可选：trace 镜像根目录（与 metrics_lite 相同语义）",
    )
    p.add_argument("--out", type=Path, required=True, help="输出 metrics_trajectory.jsonl")
    p.add_argument("--csv", type=Path, default=None, help="可选：同时写 CSV")
    p.add_argument("--tabletop-z", type=float, default=ml.DEFAULT_TABLETOP_Z_M, help="布局仅 xy 时的默认 z（米）")
    p.add_argument(
        "--default-radius",
        type=float,
        default=ml.DEFAULT_SUCCESS_RADIUS_M,
        help="gt_spec / evaluation 无 success_radius 时的默认半径（米）",
    )
    args = p.parse_args()
    if not args.load_trace_eef:
        print("错误: 轨迹诊断必须指定 --load-trace-eef（从 trace.json 读取 eef_pos 全序列）", file=sys.stderr)
        return 2

    trials = ml.load_trial_lines(args.results)  # type: ignore[attr-defined]
    layout_bundle = ml.merge_layout_files(list(args.layouts))  # type: ignore[attr-defined]
    eval_root = args.results.resolve().parent

    out_rows: list[dict[str, Any]] = []
    for trial in trials:
        out_rows.append(
            process_episode_trajectory(
                trial,
                layout_bundle,
                eval_root=eval_root,
                traces_mirror=args.traces,
                default_z=float(args.tabletop_z),
                default_radius=float(args.default_radius),
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        preferred = [
            "episode_dir",
            "result_json",
            "layout_key_resolved",
            "gt_target_object_id",
            "trace_path_used",
            "object_pos_source",
            "static_layout_proxy",
            "trajectory_valid",
            "trajectory_missing_reason",
            "trajectory_timesteps",
            "mean_target_distance",
            "final_target_distance",
            "min_target_distance",
            "target_near_fraction",
            "target_closest_rate",
            "mean_target_margin",
            "success_radius_used",
            "xy_only",
        ]
        keys = preferred + [k for k in out_rows[0].keys() if k not in preferred] if out_rows else preferred
        with args.csv.open("w", encoding="utf-8", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in out_rows:
                flat = {k: r.get(k) for k in keys}
                w.writerow(flat)

    summ = summarize_trajectory(out_rows)
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
