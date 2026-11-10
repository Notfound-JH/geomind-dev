#!/usr/bin/env python3
"""
最小后处理：从 trial_results.jsonl + result.json + legacy layout（可选 trace）计算
MTA-lite 与 Target Reach Satisfaction。不修改环境、不修改 task JSON、不解析 SCS constraints。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

# 布局仅有 xy 时补 z（米）；仅用于与 layout 中物体参考位置对齐
DEFAULT_TABLETOP_Z_M = 0.784
# protocol.gt_spec 无 success_radius 时的默认（米）
DEFAULT_SUCCESS_RADIUS_M = 0.10

PRED_RESULT_KEYS = (
    "manipulated_object_id",
    "grasped_object_id",
    "picked_object_id",
    "moved_object_id",
    "pred_target_object_id",
)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_trial_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"_parse_error": True, "_line": line[:200]})
    return rows


def merge_layout_files(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for p in paths:
        data = _read_json(p)
        if isinstance(data, dict):
            merged.update(data)
    return merged


def _layout_key_candidates(key: str) -> list[str]:
    k = str(key or "").strip()
    if not k:
        return []
    out = [k]
    if k.endswith("_vla"):
        out.append(k[:-4])
    else:
        out.append(f"{k}_vla")
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def resolve_layout(bundle: dict[str, Any], layout_key: str) -> tuple[str | None, dict[str, Any] | None]:
    for cand in _layout_key_candidates(layout_key):
        entry = bundle.get(cand)
        if isinstance(entry, dict) and isinstance(entry.get("tasks"), list):
            return cand, entry
    return None, None


def _task_index_from_result(result: dict[str, Any]) -> int:
    if isinstance(result.get("annotation_task_index"), int):
        return int(result["annotation_task_index"])
    lr = result.get("layout_ref")
    if isinstance(lr, dict) and isinstance(lr.get("task_index"), int):
        return int(lr["task_index"])
    return 0


def _layout_key_from_result(result: dict[str, Any]) -> str:
    k = str(result.get("annotation_layout_key") or "").strip()
    if k:
        return k
    lr = result.get("layout_ref")
    if isinstance(lr, dict):
        k2 = str(lr.get("layout_key") or "").strip()
        if k2:
            return k2
    return ""


def _vec3_from_value(val: Any, default_z: float) -> tuple[float, float, float] | None:
    if isinstance(val, dict):
        if isinstance(val.get("xy"), (list, tuple)) and len(val["xy"]) >= 2:
            z = float(val.get("z", default_z))
            return (float(val["xy"][0]), float(val["xy"][1]), z)
        if "position" in val:
            return _vec3_from_value(val["position"], default_z)
        return None
    if isinstance(val, (list, tuple)):
        if len(val) >= 3:
            try:
                return (float(val[0]), float(val[1]), float(val[2]))
            except (TypeError, ValueError):
                return None
        if len(val) >= 2:
            try:
                return (float(val[0]), float(val[1]), default_z)
            except (TypeError, ValueError):
                return None
    return None


def build_object_positions(
    layout_entry: dict[str, Any],
    task_payload: dict[str, Any],
    default_z: float,
) -> dict[str, tuple[float, float, float]]:
    poses: dict[str, tuple[float, float, float]] = {}
    scene = layout_entry.get("scene_structure")
    if isinstance(scene, dict):
        obj_pos = scene.get("object_positions")
        if isinstance(obj_pos, dict):
            for oid, val in obj_pos.items():
                if isinstance(val, dict) and "position" in val:
                    v = _vec3_from_value(val["position"], default_z)
                else:
                    v = _vec3_from_value(val, default_z)
                if v is not None:
                    poses[str(oid)] = v
    sa = task_payload.get("scene_adaptation")
    if isinstance(sa, dict):
        for obj in sa.get("runtime_objects") or []:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name") or "").strip()
            if not name:
                continue
            pose = obj.get("pose")
            if isinstance(pose, dict):
                v = _vec3_from_value(pose, default_z)
                if v is not None:
                    poses[name] = v
    return poses


def infer_gt_target_object_id(annotation: dict[str, Any]) -> str | None:
    """layout annotation.protocol.gt_spec 优先级：target_object_id > expected_target > targets[0]。"""
    proto = annotation.get("protocol")
    if not isinstance(proto, dict):
        return None
    gt = proto.get("gt_spec")
    if isinstance(gt, dict):
        for key in ("target_object_id", "expected_target"):
            v = gt.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        tgs = gt.get("targets")
        if isinstance(tgs, list) and tgs:
            t0 = tgs[0]
            if isinstance(t0, str) and t0.strip():
                return t0.strip()
    to = proto.get("target_object")
    if isinstance(to, dict):
        oid = to.get("object_id")
        if isinstance(oid, str) and oid.strip():
            return oid.strip()
    targets = annotation.get("targets")
    if isinstance(targets, list) and targets:
        t0 = targets[0]
        if isinstance(t0, str) and t0.strip():
            return t0.strip()
    return None


def _get_nonempty_str(d: dict[str, Any], key: str) -> str | None:
    v = d.get(key)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def infer_pred_from_result(result: dict[str, Any]) -> str | None:
    """在 result 顶层与常见子 dict 中按优先级查找物体 id。"""
    buckets: list[dict[str, Any]] = [result]
    for k in ("evaluation", "execution_binding", "episode_summary", "outcome", "extras"):
        sub = result.get(k)
        if isinstance(sub, dict):
            buckets.append(sub)
    for d in buckets:
        for key in PRED_RESULT_KEYS:
            v = _get_nonempty_str(d, key)
            if v is not None:
                return v
    return None


def extract_last_eef_pos(trace_path: Path) -> tuple[float, float, float] | None:
    """从 trace.json 文本解析最后一次 eef_pos（避免整文件 json 解析）。"""
    try:
        text = trace_path.read_text(encoding="utf-8")
    except OSError:
        return None
    key = '"eef_pos"'
    idx = text.rfind(key)
    if idx < 0:
        return None
    lb = text.find("[", idx)
    rb = text.find("]", lb + 1)
    if lb < 0 or rb < 0:
        return None
    chunk = text[lb : rb + 1]
    try:
        arr = json.loads(chunk)
        if isinstance(arr, list) and len(arr) >= 3:
            return (float(arr[0]), float(arr[1]), float(arr[2]))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def _iter_trace_candidates(
    *,
    result_path: Path,
    episode_dir: str | None,
    eval_root: Path,
    traces_mirror: Path | None,
) -> Iterator[Path]:
    parents: list[Path] = [result_path.parent]
    if episode_dir:
        ep = Path(episode_dir)
        parents.append(ep)
        if (ep / "episode_0000").is_dir():
            parents.append(ep / "episode_0000")
    seen: set[Path] = set()
    for par in parents:
        p = par / "trace.json"
        rp = p.resolve()
        if p.is_file() and rp not in seen:
            seen.add(rp)
            yield p
    if traces_mirror is not None and traces_mirror.is_dir() and episode_dir:
        try:
            rel = Path(episode_dir).resolve().relative_to(eval_root.resolve())
            alt = (traces_mirror / rel / "trace.json").resolve()
            if alt.is_file() and alt not in seen:
                yield alt
        except ValueError:
            pass


def nearest_object_id(
    eef: tuple[float, float, float],
    positions: dict[str, tuple[float, float, float]],
) -> str | None:
    best: str | None = None
    best_d = math.inf
    for oid, pos in positions.items():
        d = math.dist(eef, pos)
        if d < best_d:
            best_d = d
            best = oid
    return best


def distance_eef_to_target(
    eef: tuple[float, float, float],
    target_pos: tuple[float, float, float],
    xy_only: bool,
) -> float:
    if xy_only:
        return math.hypot(eef[0] - target_pos[0], eef[1] - target_pos[1])
    return math.dist(eef, target_pos)


def probe_sample_fields(rows: list[dict[str, Any]], n: int = 3) -> dict[str, Any]:
    """检查前几条的 result / trace 实际字段（stdout 用）。"""
    out: dict[str, Any] = {"sampled_trials": n, "result_key_sets": [], "pred_hits": []}
    for row in rows[:n]:
        if not isinstance(row, dict) or row.get("_parse_error"):
            continue
        rp = row.get("result_json")
        if not isinstance(rp, str) or not rp.strip():
            out["result_key_sets"].append({"error": "no_result_json"})
            continue
        res = _read_json(Path(rp))
        if not isinstance(res, dict):
            out["result_key_sets"].append({"error": "result_read_failed"})
            continue
        top = sorted(res.keys())
        ev = res.get("evaluation")
        evk = sorted(ev.keys()) if isinstance(ev, dict) else []
        eb = res.get("execution_binding")
        ebk = sorted(eb.keys()) if isinstance(eb, dict) else []
        pred = infer_pred_from_result(res)
        out["result_key_sets"].append({"top_keys": top, "evaluation_keys": evk, "execution_binding_keys": ebk})
        out["pred_hits"].append(
            {
                "pred_target_object_id": pred,
                "evaluation_goal_distance": (ev or {}).get("goal_distance") if isinstance(ev, dict) else None,
            }
        )
    return out


def process_episode(
    trial: dict[str, Any],
    layout_bundle: dict[str, Any],
    *,
    eval_root: Path,
    load_trace_eef: bool,
    traces_mirror: Path | None,
    default_z: float,
    default_radius: float,
) -> dict[str, Any]:
    """单条 trial + 已加载 layout bundle -> 输出行 dict。"""
    row: dict[str, Any] = {k: v for k, v in trial.items() if not str(k).startswith("_")}
    row["metrics_lite_version"] = 1

    if trial.get("_parse_error"):
        row["mta_lite"] = None
        row["mta_lite_missing_reason"] = "trial_json_parse_error"
        row["pred_target_object_id"] = None
        row["pred_target_source"] = "missing"
        row["gt_target_object_id"] = None
        row["target_distance"] = None
        row["target_reach_satisfaction"] = None
        row["target_reach_missing_reason"] = "trial_json_parse_error"
        return row

    rp = trial.get("result_json")
    result_path = Path(str(rp)) if isinstance(rp, str) and rp.strip() else None
    result: dict[str, Any] | None = None
    if result_path is not None:
        data = _read_json(result_path)
        if isinstance(data, dict):
            result = data

    if not isinstance(result, dict):
        row["mta_lite"] = None
        row["mta_lite_missing_reason"] = "missing_or_unreadable_result_json"
        row["pred_target_object_id"] = None
        row["pred_target_source"] = "missing"
        row["gt_target_object_id"] = None
        row["layout_resolution"] = None
        row["target_distance"] = None
        row["target_reach_satisfaction"] = None
        row["target_reach_missing_reason"] = "missing_or_unreadable_result_json"
        return row

    layout_key = _layout_key_from_result(result)
    tidx = _task_index_from_result(result)
    resolved_key: str | None = None
    annotation: dict[str, Any] | None = None
    task_payload: dict[str, Any] | None = None
    layout_entry: dict[str, Any] | None = None
    positions: dict[str, tuple[float, float, float]] = {}

    if layout_key and layout_bundle:
        resolved_key, layout_entry = resolve_layout(layout_bundle, layout_key)
        if layout_entry and isinstance(layout_entry.get("tasks"), list):
            tasks = layout_entry["tasks"]
            if 0 <= tidx < len(tasks) and isinstance(tasks[tidx], dict):
                task_payload = tasks[tidx]
                ann = task_payload.get("annotation")
                if isinstance(ann, dict):
                    annotation = ann
                    positions = build_object_positions(layout_entry, task_payload, default_z)

    row["layout_key_requested"] = layout_key or None
    row["layout_key_resolved"] = resolved_key
    row["annotation_task_index"] = tidx

    gt_id: str | None = None
    if isinstance(annotation, dict):
        gt_id = infer_gt_target_object_id(annotation)
    row["gt_target_object_id"] = gt_id

    proto = (annotation or {}).get("protocol") if isinstance(annotation, dict) else None
    proto = proto if isinstance(proto, dict) else {}
    gt_spec = proto.get("gt_spec") if isinstance(proto.get("gt_spec"), dict) else {}
    xy_only = bool(gt_spec.get("xy_only"))
    sr = gt_spec.get("success_radius")
    try:
        success_radius = float(sr) if sr is not None else default_radius
    except (TypeError, ValueError):
        success_radius = default_radius
    ev = result.get("evaluation")
    if success_radius == default_radius and isinstance(ev, dict) and ev.get("success_radius") is not None:
        try:
            success_radius = float(ev["success_radius"])
        except (TypeError, ValueError):
            pass
    row["success_radius_used"] = success_radius
    row["xy_only"] = xy_only

    pred_source = "missing"
    pred: str | None = infer_pred_from_result(result)
    if pred is not None:
        pred_source = "explicit_log"

    eef: tuple[float, float, float] | None = None
    trace_used: str | None = None
    if load_trace_eef and result_path is not None:
        ep_dir = trial.get("episode_dir")
        ep_dir_s = str(ep_dir) if isinstance(ep_dir, str) else None
        for tp in _iter_trace_candidates(
            result_path=result_path,
            episode_dir=ep_dir_s,
            eval_root=eval_root,
            traces_mirror=traces_mirror,
        ):
            eef = extract_last_eef_pos(tp)
            if eef is not None:
                trace_used = str(tp)
                break

    if pred is None and eef is not None and positions:
        pred = nearest_object_id(eef, positions)
        if pred is not None:
            pred_source = "nearest_final_eef"

    row["pred_target_object_id"] = pred
    row["pred_target_source"] = pred_source
    row["trace_path_used"] = trace_used
    row["final_eef_pos"] = list(eef) if eef is not None else None

    gt_pos = positions.get(gt_id) if gt_id else None
    row["gt_target_object_pos"] = list(gt_pos) if gt_pos is not None else None

    # MTA-lite
    if not gt_id:
        row["mta_lite"] = None
        row["mta_lite_missing_reason"] = "missing_gt_target_object_id"
    elif not pred:
        row["mta_lite"] = None
        row["mta_lite_missing_reason"] = "missing_pred_target_object_id"
    else:
        row["mta_lite"] = 1 if pred == gt_id else 0
        row["mta_lite_missing_reason"] = None

    # Target reach
    if eef is None:
        row["target_distance"] = None
        row["target_reach_satisfaction"] = None
        row["target_reach_missing_reason"] = "missing_final_eef_pos"
    elif gt_pos is None:
        row["target_distance"] = None
        row["target_reach_satisfaction"] = None
        row["target_reach_missing_reason"] = "missing_gt_target_object_pos"
    else:
        td = distance_eef_to_target(eef, gt_pos, xy_only)
        row["target_distance"] = td
        row["target_reach_satisfaction"] = bool(td <= success_radius)
        row["target_reach_missing_reason"] = None

    return row


def summarize(rows_out: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows_out)
    mta_avail = [r for r in rows_out if r.get("mta_lite") is not None]
    reach_avail = [r for r in rows_out if r.get("target_distance") is not None]

    def _mean_int(xs: list[dict[str, Any]], key: str) -> float | None:
        vals = [r[key] for r in xs if isinstance(r.get(key), int)]
        if not vals:
            return None
        return sum(vals) / len(vals)

    miss = Counter()
    for r in rows_out:
        mr = r.get("mta_lite_missing_reason")
        if isinstance(mr, str) and mr:
            miss[f"mta:{mr}"] += 1
        tr = r.get("target_reach_missing_reason")
        if isinstance(tr, str) and tr:
            miss[f"reach:{tr}"] += 1

    return {
        "total_episodes": total,
        "mta_lite_coverage": (len(mta_avail) / total) if total else 0.0,
        "mta_lite_accuracy_on_available": _mean_int(mta_avail, "mta_lite"),
        "target_reach_coverage": (len(reach_avail) / total) if total else 0.0,
        "target_reach_satisfaction_rate_on_available": (
            sum(1 for r in reach_avail if r.get("target_reach_satisfaction") is True) / len(reach_avail)
            if reach_avail
            else None
        ),
        "missing_reason_counts": dict(miss),
        "pred_target_source_counts": dict(Counter(str(r.get("pred_target_source") or "") for r in rows_out)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="MTA-lite + Target Reach Satisfaction（后处理）")
    p.add_argument("--results", type=Path, required=True, help="trial_results.jsonl")
    p.add_argument(
        "--layouts",
        type=Path,
        action="append",
        required=True,
        help="legacy_layouts.json（可多次指定，后写覆盖同名 layout_key）",
    )
    p.add_argument(
        "--traces",
        type=Path,
        default=None,
        help="可选：与 eval 根目录同结构的镜像根；在其下用 episode_dir 相对路径找 trace.json",
    )
    p.add_argument("--load-trace-eef", action="store_true", help="解析 trace.json 最后一次 eef_pos")
    p.add_argument("--out", type=Path, required=True, help="输出 metrics_lite.jsonl")
    p.add_argument("--tabletop-z", type=float, default=DEFAULT_TABLETOP_Z_M, help="布局仅 xy 时的默认 z（米）")
    p.add_argument(
        "--default-radius",
        type=float,
        default=DEFAULT_SUCCESS_RADIUS_M,
        help="gt_spec 无 success_radius 且 evaluation 也无时的默认半径（米）",
    )
    p.add_argument("--verbose", action="store_true", help="打印前几条的 result 字段探测")
    args = p.parse_args()

    trials = load_trial_lines(args.results)
    if args.verbose:
        probe = probe_sample_fields(trials, n=3)
        print("=== probe (first trial lines) ===", file=sys.stderr)
        print(json.dumps(probe, ensure_ascii=False, indent=2), file=sys.stderr)

    layout_bundle = merge_layout_files(list(args.layouts))
    eval_root = args.results.resolve().parent

    out_rows: list[dict[str, Any]] = []
    for trial in trials:
        out_rows.append(
            process_episode(
                trial,
                layout_bundle,
                eval_root=eval_root,
                load_trace_eef=bool(args.load_trace_eef),
                traces_mirror=args.traces,
                default_z=float(args.tabletop_z),
                default_radius=float(args.default_radius),
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summ = summarize(out_rows)
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
