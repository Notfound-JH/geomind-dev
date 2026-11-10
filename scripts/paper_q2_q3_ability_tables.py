#!/usr/bin/env python3
"""
Q2/Q3 论文用 ability-level 汇总：基于 trial_results + per_episode_diagnostics + metrics_lite。
仅使用五套 eval 的共有 episode 子集（task_id, success_level, seed, trial_index）= 292。

NTA-lite：当前 metrics_lite 管线未定义；输出列 `nta_lite_mean` 留空（NaN 占位），仅报告 MTA-lite。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]

SIX_FAMILIES = (
    "Target Identification",
    "Relational Grounding",
    "Scene Updating",
    "Identity Tracking",
    "Constraint Composition",
    "Geometric Inference",
)
MEMORY_FAMILIES = ("Scene Updating", "Identity Tracking")
REASONING_FAMILIES = (
    "Target Identification",
    "Relational Grounding",
    "Constraint Composition",
    "Geometric Inference",
)

DEFAULT_RUNS = (
    "run_eval_internvla",
    "run_eval_openpi_0",
    "run_eval_openvla",
    "run_eval_spatialforcing",
    "run_eval_spatialvla",
)


def episode_key(trial: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(trial.get("task_id") or ""),
        str(trial.get("success_level") or ""),
        int(trial.get("seed") if trial.get("seed") is not None else -1),
        int(trial.get("trial_index") if trial.get("trial_index") is not None else -1),
    )


def load_jsonl_metrics(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        k = episode_key(row)
        out[k] = row
    return out


def load_diag_by_result_json(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rj = str(row.get("result_json") or "").strip()
            if rj:
                out[rj] = {k: (v if v is not None else "") for k, v in row.items()}
    return out


def load_trials(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _parse_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s or s.lower() == "nan":
        return None
    try:
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except ValueError:
        return None


def merge_run_rows(
    trials: list[dict[str, Any]],
    diag: dict[str, dict[str, str]],
    ml: dict[tuple[Any, ...], dict[str, Any]],
    common_keys: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for t in trials:
        k = episode_key(t)
        if k not in common_keys:
            continue
        rj = str(t.get("result_json") or "").strip()
        drow = diag.get(rj, {})
        mlrow = ml.get(k, {})
        sge_raw = drow.get("sge", "").strip()
        sge = _parse_float(sge_raw)
        success = bool(t.get("success")) if "success" in t else None
        fam = str(drow.get("task_family") or "").strip()
        model = str(t.get("remote_policy_model") or drow.get("agent") or drow.get("method") or "").strip()
        tr = mlrow.get("target_reach_satisfaction")
        mta = mlrow.get("mta_lite")
        merged.append(
            {
                "episode_key": k,
                "model": model,
                "task_family": fam,
                "success": success,
                "sge": sge,
                "task_id": str(t.get("task_id") or ""),
                "success_level": str(t.get("success_level") or ""),
                "target_reach_satisfaction": tr if isinstance(tr, bool) else None,
                "mta_lite": mta if mta in (0, 1) else None,
            }
        )
    return merged


def aggregate_family(
    rows: list[dict[str, Any]],
    families: tuple[str, ...],
    models_order: list[str],
) -> list[dict[str, Any]]:
    """按 model x task_family 聚合（仅 families 内）；缺失组合填 n_episodes=0。"""
    bucket: dict[str, dict[str, dict[str, list[Any]]]] = defaultdict(
        lambda: defaultdict(lambda: {"succ": [], "sge": [], "tr": [], "mta": []})
    )
    for r in rows:
        fam = r["task_family"]
        if fam not in families:
            continue
        m = r["model"]
        if r["success"] is not None:
            bucket[m][fam]["succ"].append(1.0 if r["success"] else 0.0)
        if r["sge"] is not None:
            bucket[m][fam]["sge"].append(float(r["sge"]))
        if r["target_reach_satisfaction"] is not None:
            bucket[m][fam]["tr"].append(1.0 if r["target_reach_satisfaction"] else 0.0)
        if r["mta_lite"] is not None:
            bucket[m][fam]["mta"].append(float(r["mta_lite"]))

    out_rows: list[dict[str, Any]] = []
    for model in models_order:
        for fam in families:
            b = bucket[model][fam]
            ns = len(b["succ"])
            nv = len(b["sge"])
            ntr = len(b["tr"])
            nmta = len(b["mta"])
            out_rows.append(
                {
                    "model": model,
                    "task_family": fam,
                    "subset": "common_episode_intersection",
                    "n_episodes": ns,
                    "success_rate": (sum(b["succ"]) / ns) if ns else float("nan"),
                    "mean_sge": (sum(b["sge"]) / nv) if nv else float("nan"),
                    "valid_sge_count": nv,
                    "target_reach_rate": (sum(b["tr"]) / ntr) if ntr else float("nan"),
                    "n_target_reach_defined": ntr,
                    "mta_lite_mean": (sum(b["mta"]) / nmta) if nmta else float("nan"),
                    "n_mta_lite_defined": nmta,
                    "nta_lite_mean": "",
                    "notes": "diagnostics_auxiliary; nta_lite_mean not defined in metrics_lite",
                }
            )
    return out_rows


def family_difficulty(
    rows: list[dict[str, Any]],
    families: tuple[str, ...],
    models: list[str],
) -> list[dict[str, Any]]:
    """各 family：五模型 success 均值，从低到高为更难（rank 1 = 最难）。"""
    fam_models: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["task_family"] not in families:
            continue
        if r["success"] is None:
            continue
        fam_models[r["task_family"]].append(1.0 if r["success"] else 0.0)
    # need per-model mean then mean of model means - user said 所有模型平均 success
    per_fam_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["task_family"] not in families or r["success"] is None:
            continue
        per_fam_model[r["task_family"]][r["model"]].append(1.0 if r["success"] else 0.0)

    scored: list[tuple[str, float]] = []
    for fam in families:
        model_means: list[float] = []
        for mod in models:
            xs = per_fam_model[fam].get(mod, [])
            if xs:
                model_means.append(sum(xs) / len(xs))
        if not model_means:
            continue
        scored.append((fam, sum(model_means) / len(model_means)))
    scored.sort(key=lambda x: x[1])
    out: list[dict[str, Any]] = []
    for rank, (fam, ms) in enumerate(scored, start=1):
        out.append(
            {
                "difficulty_rank": rank,
                "task_family": fam,
                "mean_success_across_models": ms,
                "n_models_in_mean": len(models),
                "interpretation": "rank 1 = lowest mean success = hardest among listed families",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser(description="Q2/Q3 ability tables (common 292 subset)")
    p.add_argument("--repo-root", type=Path, default=_REPO)
    p.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="run_eval_* 名称（与 repo 下目录一致）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "analysis_outputs" / "paper_q2_q3",
    )
    args = p.parse_args()
    root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    runs = list(args.runs)

    key_sets: list[set[tuple[Any, ...]]] = []
    per_run_trials: dict[str, list[dict[str, Any]]] = {}
    per_run_diag: dict[str, dict[str, dict[str, str]]] = {}
    per_run_ml: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}

    for name in runs:
        trial_path = root / name / "trial_results.jsonl"
        diag_path = root / "analysis_outputs" / "eval_diagnostics" / name / "per_episode_diagnostics.csv"
        ml_path = root / "analysis_outputs" / "metrics_lite" / f"{name}.jsonl"
        trials = load_trials(trial_path)
        per_run_trials[name] = trials
        per_run_diag[name] = load_diag_by_result_json(diag_path)
        per_run_ml[name] = load_jsonl_metrics(ml_path)
        key_sets.append({episode_key(t) for t in trials})

    common_keys = set.intersection(*key_sets) if key_sets else set()
    n_common = len(common_keys)

    merged_all: list[dict[str, Any]] = []
    models_order: list[str] = []
    for name in runs:
        rows = merge_run_rows(
            per_run_trials[name],
            per_run_diag[name],
            per_run_ml[name],
            common_keys,
        )
        merged_all.extend(rows)
    models_order = sorted({r["model"] for r in merged_all if r.get("model")})

    # Part A: six families — wide summary optional; user asked B and C CSVs primarily
    q2_rows = aggregate_family(merged_all, MEMORY_FAMILIES, models_order)
    q3_rows = aggregate_family(merged_all, REASONING_FAMILIES, models_order)

    diff_rows: list[dict[str, Any]] = []
    for pool, fams in (
        ("memory", MEMORY_FAMILIES),
        ("reasoning", REASONING_FAMILIES),
        ("all_six", SIX_FAMILIES),
    ):
        for row in family_difficulty(merged_all, fams, models_order):
            diff_rows.append({"pool": pool, **row})

    meta = {
        "common_episode_subset_size": n_common,
        "episode_key_fields": ["task_id", "success_level", "seed", "trial_index"],
        "runs": runs,
        "models": models_order,
        "note": "success_rate from trial_results.success; diagnostics are auxiliary.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "q2_q3_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    q_all_rows = aggregate_family(merged_all, SIX_FAMILIES, models_order)
    write_csv(out_dir / "q2_memory_ability_table.csv", q2_rows)
    write_csv(out_dir / "q3_reasoning_ability_table.csv", q3_rows)
    write_csv(out_dir / "part_a_six_families_by_model.csv", q_all_rows)
    write_csv(out_dir / "family_difficulty_ranking.csv", diff_rows)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Wrote CSVs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
