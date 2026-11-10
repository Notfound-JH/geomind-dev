#!/usr/bin/env python3
"""
基于已生成的 metrics_lite.jsonl 做可信度 sanity check（只读，不改环境、不写 task JSON）。

说明：本报告仅涉及 endpoint-based diagnostic（MTA-lite、target distance / reach），
不把 layout_constraints 当作 SCS，也不报告正式 SCS。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_METRICS_DIR = _REPO / "analysis_outputs" / "metrics_lite"
_DEFAULT_LAYOUTS = [
    _REPO / "geomind" / "configs" / "legacy" / "layouts" / "standard.json",
    _REPO / "geomind" / "configs" / "legacy" / "layouts" / "vla.json",
]
_DEFAULT_MAPPING = _REPO / "geomind" / "configs" / "current_task_mapping.json"
_DEFAULT_TABLETOP_Z = 0.784


def _load_metrics_lite_module():
    path = _REPO / "scripts" / "metrics_lite.py"
    spec = importlib.util.spec_from_file_location("metrics_lite", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load metrics_lite")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def episode_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """跨 run 对齐：同一任务、档位、种子、trial。"""
    return (
        str(row.get("task_id") or ""),
        str(row.get("success_level") or ""),
        int(row.get("seed") or -1),
        int(row.get("trial_index") or -1),
    )


def normalize_task_id_for_mapping(task_id: str) -> str:
    return re.sub(r"Vla_v(\d+)$", r"_v\1", str(task_id).strip())


def load_task_family_map(mapping_path: Path) -> dict[str, str]:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for t in data.get("tasks") or []:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("task_id") or "").strip()
        fam = str(t.get("taxonomy_family") or "").strip()
        if tid:
            out[tid] = fam
    return out


def task_family_for_row(row: dict[str, Any], fam_map: dict[str, str]) -> str:
    tid = str(row.get("task_id") or "")
    return fam_map.get(normalize_task_id_for_mapping(tid)) or fam_map.get(tid) or str(row.get("family") or "unknown")


def dist_xy_or_3d(
    eef: list[float],
    tgt: list[float],
    xy_only: bool,
) -> float | None:
    if len(eef) < 3 or len(tgt) < 3:
        return None
    if xy_only:
        return math.hypot(float(eef[0]) - float(tgt[0]), float(eef[1]) - float(tgt[1]))
    return math.dist(tuple(map(float, eef[:3])), tuple(map(float, tgt[:3])))


def reach_bool(d: float | None, radius: float) -> bool | None:
    if d is None:
        return None
    return bool(d <= radius)


def quantile_summary(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None, "n": 0}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    mean = float(sum(xs_sorted) / n)
    median = float(statistics.median(xs_sorted))
    if n == 1:
        p25 = p75 = median
    else:
        p25 = float(xs_sorted[max(0, int(0.25 * (n - 1)))])
        p75 = float(xs_sorted[min(n - 1, int(0.75 * (n - 1)))])
    return {
        "mean": mean,
        "median": median,
        "p25": p25,
        "p75": p75,
        "min": float(xs_sorted[0]),
        "max": float(xs_sorted[-1]),
        "n": n,
    }


def positions_for_row(ml: Any, bundle: dict[str, Any], row: dict[str, Any], default_z: float) -> dict[str, tuple[float, float, float]]:
    lk = str(row.get("layout_key_requested") or row.get("layout_key_resolved") or "").strip()
    tidx = int(row.get("annotation_task_index") or 0)
    if not lk or not bundle:
        return {}
    _rk, entry = ml.resolve_layout(bundle, lk)
    if not entry or not isinstance(entry.get("tasks"), list):
        return {}
    tasks = entry["tasks"]
    if tidx < 0 or tidx >= len(tasks) or not isinstance(tasks[tidx], dict):
        return {}
    return ml.build_object_positions(entry, tasks[tidx], default_z)


def main() -> int:
    p = argparse.ArgumentParser(description="metrics_lite 结果 sanity check")
    p.add_argument(
        "--metrics-dir",
        type=Path,
        default=_DEFAULT_METRICS_DIR,
        help="含 run_eval_*.jsonl 的目录",
    )
    p.add_argument(
        "--layouts",
        type=Path,
        action="append",
        default=[],
        help="legacy_layouts.json，可多次指定；默认两份标准路径",
    )
    p.add_argument("--mapping", type=Path, default=_DEFAULT_MAPPING)
    p.add_argument("--tabletop-z", type=float, default=_DEFAULT_TABLETOP_Z)
    p.add_argument("--sample-coords", type=int, default=20, help="随机抽样打印坐标/距离的条数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=_DEFAULT_METRICS_DIR / "sanity_check_report.json")
    args = p.parse_args()

    layout_paths = list(args.layouts) if args.layouts else list(_DEFAULT_LAYOUTS)
    ml = _load_metrics_lite_module()
    bundle = ml.merge_layout_files(layout_paths)
    fam_map = load_task_family_map(args.mapping)

    metrics_dir = args.metrics_dir.resolve()
    jsonl_files = sorted(metrics_dir.glob("run_eval_*.jsonl"))
    if not jsonl_files:
        print(f"No run_eval_*.jsonl under {metrics_dir}", file=sys.stderr)
        return 1

    runs: dict[str, list[dict[str, Any]]] = {}
    for fp in jsonl_files:
        name = fp.stem  # run_eval_internvla
        runs[name] = _read_jsonl(fp)

    # --- 1) episode sets ---
    per_run_sets: dict[str, set[tuple[Any, ...]]] = {k: set() for k in runs}
    per_run_task_ids: dict[str, set[str]] = {k: set() for k in runs}
    for name, rows in runs.items():
        for row in rows:
            per_run_sets[name].add(episode_key(row))
            per_run_task_ids[name].add(str(row.get("task_id") or ""))

    key_lists = list(per_run_sets.values())
    common_keys: set[tuple[Any, ...]] = set.intersection(*key_lists) if key_lists else set()

    report: dict[str, Any] = {
        "disclaimer": (
            "Endpoint-based diagnostic only (MTA-lite, target reach). "
            "layout_constraints are NOT used as SCS. Not an official SCS metric."
        ),
        "runs": {},
        "episode_set_alignment": {
            "per_run_num_episodes": {k: len(v) for k, v in per_run_sets.items()},
            "per_run_num_unique_task_ids": {k: len(v) for k, v in per_run_task_ids.items()},
        "intersection_size_across_all_runs": len(common_keys),
        "runs_with_same_cardinality": len(set(len(s) for s in per_run_sets.values())) == 1,
        "keys_unique_to_run_not_in_intersection": {
            name: len(per_run_sets[name] - common_keys) for name in sorted(runs.keys())
        },
    },
        "subset_common_episodes": {},
        "coord_sample": [],
        "target_distance_stats": {},
        "success_radius_audit": {},
        "reach_sensitivity_radii": {},
        "random_baseline_mta": {},
    }

    for name in sorted(runs.keys()):
        report["runs"][name] = {
            "num_episodes": len(runs[name]),
            "task_id_list_sorted": sorted(per_run_task_ids[name]),
            "episode_key_count": len(per_run_sets[name]),
        }

    # Common subset stats
    subset_stats: dict[str, Any] = {}
    radii = (0.05, 0.10, 0.15, 0.20)
    reach_by_radius_by_run: dict[str, dict[float, list[bool | None]]] = {
        name: {r: [] for r in radii} for name in runs
    }

    all_distances: list[float] = []
    dist_by_run: dict[str, list[float]] = {k: [] for k in runs}
    dist_by_family_by_run: dict[str, dict[str, list[float]]] = {
        k: {} for k in runs
    }

    sr_values: dict[str, list[float]] = {k: [] for k in runs}
    sr_default_010_count: dict[str, int] = {k: 0 for k in runs}

    num_objs_list: dict[str, list[int]] = {k: [] for k in runs}
    mta_values: dict[str, list[int]] = {k: [] for k in runs}
    mta_subset: dict[str, list[int]] = {k: [] for k in runs}

    for name, rows in runs.items():
        for row in rows:
            ek = episode_key(row)
            td = row.get("target_distance")
            if isinstance(td, (int, float)) and not isinstance(td, bool):
                d = float(td)
                all_distances.append(d)
                dist_by_run[name].append(d)
                fam = task_family_for_row(row, fam_map)
                dist_by_family_by_run[name].setdefault(fam, []).append(d)

            sr = row.get("success_radius_used")
            if isinstance(sr, (int, float)):
                sr_f = float(sr)
                sr_values[name].append(sr_f)
                if math.isclose(sr_f, 0.10, rel_tol=0.0, abs_tol=1e-6):
                    sr_default_010_count[name] += 1

            eef = row.get("final_eef_pos")
            tgt = row.get("gt_target_object_pos")
            xy_only = bool(row.get("xy_only"))
            dcalc = None
            if isinstance(eef, list) and isinstance(tgt, list):
                dcalc = dist_xy_or_3d(eef, tgt, xy_only)
            for r in radii:
                reach_by_radius_by_run[name][r].append(reach_bool(dcalc, r))

            pos = positions_for_row(ml, bundle, row, float(args.tabletop_z))
            nobj = len(pos)
            num_objs_list[name].append(nobj)

            m = row.get("mta_lite")
            if m in (0, 1):
                mta_values[name].append(int(m))
                if ek in common_keys:
                    mta_subset[name].append(int(m))

        reach_on_sub = [
            1.0 if row.get("target_reach_satisfaction") is True else 0.0
            for row in rows
            if episode_key(row) in common_keys and isinstance(row.get("target_reach_satisfaction"), bool)
        ]
        subset_stats[name] = {
            "num_in_intersection": sum(1 for row in rows if episode_key(row) in common_keys),
            "mta_lite_mean_on_intersection": (
                statistics.mean(mta_subset[name]) if mta_subset[name] else None
            ),
            "target_reach_mean_on_intersection": statistics.mean(reach_on_sub) if reach_on_sub else None,
        }

    report["subset_common_episodes"] = {
        "intersection_keys_count": len(common_keys),
        "per_run_on_subset": subset_stats,
    }

    report["target_distance_stats"]["all_runs_pooled"] = quantile_summary(all_distances)
    report["target_distance_stats"]["by_run"] = {k: quantile_summary(v) for k, v in dist_by_run.items()}
    report["target_distance_stats"]["by_run_and_task_family"] = {
        run: {fam: quantile_summary(vs) for fam, vs in sorted(fam.items())}
        for run, fam in dist_by_family_by_run.items()
    }

    # success_radius audit
    sr_audit: dict[str, Any] = {}
    for name in runs:
        vals = sr_values[name]
        if not vals:
            sr_audit[name] = {"n": 0, "note": "no success_radius_used in rows"}
            continue
        ctr = Counter(round(v, 6) for v in vals)
        sr_audit[name] = {
            "n": len(vals),
            "unique_rounded_6": len(ctr),
            "most_common": ctr.most_common(8),
            "count_eq_default_0_10_exact": sr_default_010_count[name],
            "min": min(vals),
            "max": max(vals),
        }
    report["success_radius_audit"] = sr_audit
    report["success_radius_note"] = (
        "metrics_lite 中 success_radius_used 来自 gt_spec 或 evaluation；"
        "脚本默认 0.10 仅在两者皆缺时生效，需从原始 trial 行无法直接看出是否触发默认。"
    )

    # reach sensitivity
    sens: dict[str, dict[str, float | None]] = {}
    for r in radii:
        sens[str(r)] = {}
        for name in runs:
            xs = [x for x in reach_by_radius_by_run[name][r] if x is not None]
            if not xs:
                sens[str(r)][name] = None
            else:
                sens[str(r)][name] = sum(1 for b in xs if b) / len(xs)
    report["reach_sensitivity_radii"] = sens

    # random baseline
    rb: dict[str, Any] = {}
    for name in runs:
        inv = [1.0 / n for n in num_objs_list[name] if n > 0]
        rb[name] = {
            "mean_inv_num_objects_random_mta_baseline": statistics.mean(inv) if inv else None,
            "mta_lite_mean_observed": statistics.mean(mta_values[name]) if mta_values[name] else None,
            "mta_lite_mean_on_common_subset": subset_stats[name]["mta_lite_mean_on_intersection"],
            "mean_num_layout_objects": statistics.mean(num_objs_list[name]) if num_objs_list[name] else None,
        }
        if inv and mta_values[name]:
            rb[name]["observed_minus_random"] = statistics.mean(mta_values[name]) - statistics.mean(inv)
    report["random_baseline_mta"] = rb

    # Ranking stability across radii (by mean reach rate)
    ranking_per_r: dict[str, list[str]] = {}
    for r in radii:
        scores = [(name, statistics.mean([float(x) for x in reach_by_radius_by_run[name][r] if x is not None]))
                  for name in runs
                  if any(x is not None for x in reach_by_radius_by_run[name][r])]
        scores.sort(key=lambda t: t[1], reverse=True)
        ranking_per_r[str(r)] = [s[0] for s in scores]
    report["reach_ranking_by_radius"] = ranking_per_r
    r05 = tuple(ranking_per_r.get("0.05", ()))
    r20 = tuple(ranking_per_r.get("0.2", ()))
    report["reach_ranking_stable_0_05_vs_0_20"] = r05 == r20 if r05 and r20 else None

    # --- 2) random sample rows for coordinate print ---
    rng = random.Random(int(args.seed))
    pool: list[tuple[str, dict[str, Any]]] = []
    for name, rows in runs.items():
        for row in rows:
            pool.append((name, row))
    sample_n = min(int(args.sample_coords), len(pool))
    sample = rng.sample(pool, sample_n) if pool else []

    coord_rows: list[dict[str, Any]] = []
    for run_name, row in sample:
        eef = row.get("final_eef_pos")
        tgt = row.get("gt_target_object_pos")
        pred_id = row.get("pred_target_object_id")
        pos = positions_for_row(ml, bundle, row, float(args.tabletop_z))
        near_pos = None
        if isinstance(pred_id, str) and pred_id in pos:
            near_pos = list(pos[pred_id])
        td = row.get("target_distance")
        coord_rows.append(
            {
                "run": run_name,
                "model": row.get("remote_policy_model"),
                "episode_key": episode_key(row),
                "final_eef_pos": eef,
                "gt_target_pos": tgt,
                "pred_target_object_id": pred_id,
                "nearest_object_pos": near_pos,
                "target_distance_in_jsonl": td,
                "num_layout_objects": len(pos),
            }
        )
    report["coord_sample_n"] = sample_n
    report["coord_sample"] = coord_rows

    # distance sanity flags
    absurd_hi = sum(1 for d in all_distances if d > 1.0)
    absurd_lo = sum(1 for d in all_distances if d < 1e-6)
    report["distance_sanity_flags"] = {
        "count_target_distance_gt_1m": absurd_hi,
        "count_target_distance_lt_1e-6": absurd_lo,
        "fraction_gt_1m": (absurd_hi / len(all_distances)) if all_distances else 0.0,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["episode_set_alignment"], ensure_ascii=False, indent=2))
    print(json.dumps({"subset_common_episodes": report["subset_common_episodes"]}, ensure_ascii=False, indent=2))
    print(json.dumps(report["distance_sanity_flags"], ensure_ascii=False, indent=2))
    print(json.dumps(report["reach_ranking_by_radius"], ensure_ascii=False, indent=2))
    print(json.dumps({"reach_ranking_stable_0_05_vs_0_20": report["reach_ranking_stable_0_05_vs_0_20"]}, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
