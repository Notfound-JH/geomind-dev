#!/usr/bin/env python3
"""
从仓库内已有评估产物生成 paper/data/appendix/*.csv，供论文附录与绘图脚本使用。

数据源（默认相对仓库根）：
  - statistics/.../run_eval_models_task_level_matrix_family_by_model_long.csv
  - run_eval_models_task_level_matrix.csv（strict 列；用于 Semantic / Visual / Execution 与 Base 配对 drop）
  - analysis_outputs/eval_diagnostics/run_eval_<slug>/aggregate_diagnostics_by_family.csv
  - analysis_outputs/eval_diagnostics/run_eval_<slug>/aggregate_diagnostics_by_aug_type.csv（若存在）

Oracle 行：仓库中无独立 oracle 汇总时，写入 100.0 表示「特权上界占位」，请在定稿前替换为实测 oracle（若可用）。

模块化智能体（COPA/MOKA）：本仓库 eval 矩阵未包含时，不写入 base_family_results.csv（由 LaTeX 单独标为未评测）。
"""
from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

FAMILY_DISPLAY = [
    "Target Identification",
    "Relational Grounding",
    "Constraint Composition",
    "Geometric Inference",
    "Scene Updating",
    "Identity Tracking",
]

FAMILY_SLUG = {
    "Target Identification": "target_identification",
    "Relational Grounding": "relational_grounding",
    "Constraint Composition": "constraint_composition",
    "Geometric Inference": "geometric_inference",
    "Scene Updating": "scene_updating",
    "Identity Tracking": "identity_tracking",
}

# taxonomy_family -> (memory, reasoning) 设计标注（与附录 caption 一致：非心理测量声明）
CAPABILITY_DEMAND: dict[str, tuple[str, str]] = {
    "Target Identification": ("Low", "Medium"),
    "Relational Grounding": ("Low", "High"),
    "Constraint Composition": ("Medium", "High"),
    "Geometric Inference": ("Low", "High"),
    "Scene Updating": ("High", "Medium"),
    "Identity Tracking": ("High", "Medium"),
}

# matrix 列名（strict）配对：aug 列前缀片段 -> base 列前缀片段（均匹配 __strict 结尾）
SEM_PAIRS = (
    ("RelationAugSem_PlaceDirectionalVla_v0__strict", "RelationBaseBase_PlaceDirectionalVla_v0__strict"),
    ("ReindexBaseSem_ReindexExtremeVla_v0__strict", "ReindexBaseBase_ReindexExtremeVla_v0__strict"),
    ("TrackBaseSem_TrackSingleVla_v0__strict", "TrackBaseBase_TrackSingleVla_v0__strict"),
)
VIS_PAIRS = (
    ("RelationAugVisual_PlaceDirectionalVla_v0__strict", "RelationBaseBase_PlaceDirectionalVla_v0__strict"),
    ("ReindexBaseVisual_ReindexExtremeVla_v0__strict", "ReindexBaseBase_ReindexExtremeVla_v0__strict"),
    ("TrackBaseVisual_TrackSingleVla_v0__strict", "TrackBaseBase_TrackSingleVla_v0__strict"),
)
EXE_PAIRS = (("BindingAugBase_AnchorDistanceVla_v0__strict", "BindingBaseBase_AnchorDirectionVla_v0__strict"),)

VLA_MODEL_ORDER: list[tuple[str, str, str]] = [
    ("OpenVLA", "vla", "openvla"),
    ("pi_0", "vla", "openpi_0"),
    ("pi_0_5", "vla", "openpi_05"),
    ("SpatialVLA", "spatial_vla", "spatialvla"),
    ("InternVLA-M1", "spatial_vla", "internvla"),
    ("Spatial-Forcing", "spatial_vla", "spatialforcing"),
]

SPATIAL_VLA_SLUGS = {"spatialvla", "internvla", "spatialforcing"}

# summary_by_task_level.csv 的 family 字段 -> 附录 taxonomy_family（与 matrix_radar 宽表一致）
INTERNAL_FAMILY_TO_TAXONOMY: dict[str, str] = {
    "anchor_reference": "Target Identification",
    "spatial_placement": "Relational Grounding",
    "constraint_selection": "Constraint Composition",
    "explicit_rule_extrapolation": "Geometric Inference",
    "implicit_trend_fitting": "Geometric Inference",
    "shape_completion": "Geometric Inference",
    "state_reindexing": "Scene Updating",
    "historical_identity_tracking": "Identity Tracking",
}


def _fmt_pp(x: float) -> str:
    return f"{x:+.1f}"


def _read_family_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _pivot_family_rates(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """slug -> {family_display -> rate 0..1}."""
    out: dict[str, dict[str, float]] = {}
    for row in df.itertuples(index=False):
        slug = str(getattr(row, "model_slug"))
        fam = str(getattr(row, "taxonomy_family"))
        rate = float(getattr(row, "family_success_rate"))
        out.setdefault(slug, {})[fam] = rate
    return out


def _load_matrix_numeric(repo: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    """返回 (列名列表, model_slug -> {col: float or nan}). 跳过说明行与占位 run。"""
    path = repo / "run_eval_models_task_level_matrix.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        raise SystemExit(f"矩阵过短: {path}")
    header = rows[0]
    data_rows = [r for r in rows[2:] if r and r[0] and not r[0].startswith("任务")]
    col_names = header[2:]
    by_slug: dict[str, dict[str, float]] = {}
    for r in data_rows:
        slug = (r[1] or "").strip()
        if not slug or slug == "all_models_suites":
            continue
        cells = r[2:]
        m: dict[str, float] = {}
        for cname, v in zip(col_names, cells, strict=False):
            if not cname:
                continue
            s = (v or "").strip()
            if s in {"", "-"}:
                continue
            try:
                m[cname] = float(s)
            except ValueError:
                continue
        by_slug[slug] = m
    return header[2:], by_slug


def _mean_pairs(row: Mapping[str, float], pairs: Iterable[tuple[str, str]]) -> tuple[float | None, float | None]:
    aug_vals: list[float] = []
    base_vals: list[float] = []
    for a_col, b_col in pairs:
        if a_col in row and b_col in row:
            aug_vals.append(float(row[a_col]))
            base_vals.append(float(row[b_col]))
    if not aug_vals:
        return None, None
    return sum(aug_vals) / len(aug_vals), sum(base_vals) / len(base_vals)


def _weighted_base_success(by_fam: pd.DataFrame) -> float | None:
    """suite==Base 的 family 表按 num_episodes 加权平均成功率。"""
    sub = by_fam[by_fam["suite"].astype(str) == "Base"]
    if sub.empty:
        return None
    w = sub["num_episodes"].astype(float)
    s = sub["success_rate"].astype(float)
    if w.sum() <= 0:
        return float(s.mean())
    return float((s * w).sum() / w.sum())


def _weighted_aug_success(by_fam: pd.DataFrame) -> float | None:
    sub = by_fam[by_fam["suite"].astype(str) == "Aug"]
    if sub.empty:
        return None
    w = sub["num_episodes"].astype(float)
    s = sub["success_rate"].astype(float)
    if w.sum() <= 0:
        return float(s.mean())
    return float((s * w).sum() / w.sum())


def export_base_family(
    long_df: pd.DataFrame,
    out: Path,
    *,
    oracle_percent: float,
) -> None:
    rates = _pivot_family_rates(long_df)
    rows: list[dict[str, object]] = []
    oracle_row: dict[str, object] = {"method": "Oracle", "paradigm": "oracle"}
    for fam in FAMILY_DISPLAY:
        oracle_row[FAMILY_SLUG[fam]] = f"{oracle_percent:.1f}"
    oracle_row["spatial_reasoning_avg"] = f"{oracle_percent:.1f}"
    oracle_row["spatial_memory_avg"] = f"{oracle_percent:.1f}"
    oracle_row["overall_avg"] = f"{oracle_percent:.1f}"
    rows.append(oracle_row)

    for display, paradigm, slug in VLA_MODEL_ORDER:
        fam_rates = rates.get(slug, {})
        row: dict[str, object] = {"method": display, "paradigm": paradigm}
        vals: list[float] = []
        for fam in FAMILY_DISPLAY:
            r = fam_rates.get(fam)
            if r is None:
                row[FAMILY_SLUG[fam]] = ""
            else:
                row[FAMILY_SLUG[fam]] = f"{100.0 * r:.1f}"
                vals.append(r)
        if len(vals) == 6:
            reasoning = [fam_rates[f] for f in FAMILY_DISPLAY[:4]]
            memory = [fam_rates[f] for f in FAMILY_DISPLAY[4:]]
            row["spatial_reasoning_avg"] = f"{100.0 * (sum(reasoning) / 4):.1f}"
            row["spatial_memory_avg"] = f"{100.0 * (sum(memory) / 2):.1f}"
            row["overall_avg"] = f"{100.0 * (sum(vals) / 6):.1f}"
        else:
            row["spatial_reasoning_avg"] = ""
            row["spatial_memory_avg"] = ""
            row["overall_avg"] = ""
        rows.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "paradigm",
        *FAMILY_SLUG.values(),
        "spatial_reasoning_avg",
        "spatial_memory_avg",
        "overall_avg",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def export_aug_and_drops(
    repo: Path,
    matrix_by_slug: dict[str, dict[str, float]],
    out_aug: Path,
    out_drop: Path,
    long_rates: dict[str, dict[str, float]],
) -> None:
    """Aug 成功率与 (Aug-Base) 百分点优先用 strict 矩阵配对列；整体 Base/Aug 均值用 diagnostics（若有）。"""
    rows_aug: list[dict[str, str]] = []
    rows_drop: list[dict[str, str]] = []
    for display, paradigm, slug in VLA_MODEL_ORDER:
        diag_slug = slug
        diag_dir = repo / "analysis_outputs" / "eval_diagnostics" / f"run_eval_{diag_slug}"
        if not diag_dir.is_dir() and slug == "openpi_05":
            diag_slug = "openpi_0"
            diag_dir = repo / "analysis_outputs" / "eval_diagnostics" / f"run_eval_{diag_slug}"
        fam_path = diag_dir / "aggregate_diagnostics_by_family.csv"
        base_rate = None
        aug_overall = None
        if fam_path.is_file():
            bf = pd.read_csv(fam_path)
            base_rate = _weighted_base_success(bf)
            aug_overall = _weighted_aug_success(bf)

        row_m = matrix_by_slug.get(slug, {})
        sem_aug_m, sem_base_m = _mean_pairs(row_m, SEM_PAIRS)
        vis_aug_m, vis_base_m = _mean_pairs(row_m, VIS_PAIRS)
        exe_aug_m, exe_base_m = _mean_pairs(row_m, EXE_PAIRS)

        def pair_drop(aug_m: float | None, base_m: float | None) -> tuple[float | None, float | None]:
            if aug_m is None or base_m is None:
                return None, None
            return aug_m, (aug_m - base_m) * 100.0

        sem_succ, sem_drop_pp = pair_drop(sem_aug_m, sem_base_m)
        vis_succ, vis_drop_pp = pair_drop(vis_aug_m, vis_base_m)
        exe_succ, exe_drop_pp = pair_drop(exe_aug_m, exe_base_m)

        # 与 Base 总表一致：优先用 family long 的六类平均（与 base_family_results.overall_avg 同源）
        fams_lr = long_rates.get(slug, {})
        base_bench_pp = None
        if len(fams_lr) == 6:
            base_bench_pp = 100.0 * sum(fams_lr[f] for f in FAMILY_DISPLAY) / 6.0
        elif base_rate is not None:
            base_bench_pp = base_rate * 100.0

        base_matched_pp = None
        bases = [x for x in (sem_base_m, vis_base_m, exe_base_m) if x is not None]
        if bases:
            base_matched_pp = 100.0 * (sum(bases) / len(bases))

        aug_parts = [x for x in (sem_succ, vis_succ, exe_succ) if x is not None]
        aug_avg_pp = (aug_overall * 100.0) if aug_overall is not None else None
        if aug_avg_pp is None and aug_parts:
            aug_avg_pp = 100.0 * (sum(aug_parts) / len(aug_parts))

        def cell(succ: float | None, drop_pp: float | None) -> tuple[str, str, float | None]:
            if succ is None:
                return "", "", None
            s = f"{100.0 * succ:.1f}"
            d = _fmt_pp(drop_pp) if drop_pp is not None else ""
            return s, d, drop_pp

        s1, d1, raw_d1 = cell(sem_succ, sem_drop_pp)
        s2, d2, raw_d2 = cell(vis_succ, vis_drop_pp)
        s3, d3, raw_d3 = cell(exe_succ, exe_drop_pp)
        drops_raw = [x for x in (raw_d1, raw_d2, raw_d3) if x is not None]
        worst = min(drops_raw) if drops_raw else None

        base_col = base_bench_pp if base_bench_pp is not None else base_matched_pp

        rows_aug.append(
            {
                "method": display,
                "paradigm": paradigm,
                "base_avg": f"{base_col:.1f}" if base_col is not None else "",
                "semantic_success": s1,
                "semantic_drop": d1,
                "visual_success": s2,
                "visual_drop": d2,
                "execution_success": s3,
                "execution_drop": d3,
                "aug_avg": f"{aug_avg_pp:.1f}" if aug_avg_pp is not None else "",
                "worst_drop": _fmt_pp(worst) if worst is not None else "",
            }
        )

        def pos_degradation(drop_pp: float | None) -> str:
            if drop_pp is None:
                return "0.0"
            return f"{-drop_pp:.1f}"

        rows_drop.append(
            {
                "method": display,
                "semantic_drop": pos_degradation(sem_drop_pp),
                "visual_drop": pos_degradation(vis_drop_pp),
                "execution_drop": pos_degradation(exe_drop_pp),
            }
        )

    out_aug.parent.mkdir(parents=True, exist_ok=True)
    with out_aug.open("w", newline="", encoding="utf-8") as f:
        fn = [
            "method",
            "paradigm",
            "base_avg",
            "semantic_success",
            "semantic_drop",
            "visual_success",
            "visual_drop",
            "execution_success",
            "execution_drop",
            "aug_avg",
            "worst_drop",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows_aug)
    with out_drop.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "semantic_drop", "visual_drop", "execution_drop"])
        w.writeheader()
        w.writerows(rows_drop)


def export_capability_demand(long_df: pd.DataFrame, out: Path, *, oracle_percent: float) -> None:
    rates = _pivot_family_rates(long_df)
    slugs = [s for _, _, s in VLA_MODEL_ORDER]
    rows = []
    for fam in FAMILY_DISPLAY:
        mem, reas = CAPABILITY_DEMAND[fam]
        per = [rates[s][fam] for s in slugs if fam in rates[s]]
        best_vla = max((rates[s][fam] for s in slugs if s in ("openvla", "openpi_0", "openpi_05") and fam in rates[s]), default=None)
        best_sp = max((rates[s][fam] for s in slugs if s in SPATIAL_VLA_SLUGS and fam in rates[s]), default=None)
        oracle_r = oracle_percent / 100.0
        overall = sum(per) / len(per) if per else None
        rows.append(
            {
                "family": fam,
                "memory_demand": mem,
                "reasoning_demand": reas,
                "oracle": f"{oracle_percent:.1f}",
                "best_vla": f"{100.0 * best_vla:.1f}" if best_vla is not None else "",
                "best_spatial_vla": f"{100.0 * best_sp:.1f}" if best_sp is not None else "",
                "best_modular": "",
                "overall_avg": f"{100.0 * overall:.1f}" if overall is not None else "",
            }
        )
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "memory_demand",
                "reasoning_demand",
                "oracle",
                "best_vla",
                "best_spatial_vla",
                "best_modular",
                "overall_avg",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def export_diagnostics(repo: Path, out: Path) -> None:
    """以 OpenVLA 的 GeoMind-Base（suite=Base）聚合诊断为主表。"""
    path = repo / "analysis_outputs" / "eval_diagnostics" / "run_eval_openvla" / "aggregate_diagnostics_by_family.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    rows = []
    failure_types = {
        "Target Identification": "Spatial Relation Grounding Error",
        "Relational Grounding": "Spatial Relation Reasoning Error",
        "Constraint Composition": "Spatial Relation Reasoning Error",
        "Geometric Inference": "Spatial Relation Reasoning Error",
        "Scene Updating": "Historical State Memory Error",
        "Identity Tracking": "Object Identity Error",
    }
    cap_dim = {
        "Target Identification": "Spatial Reasoning",
        "Relational Grounding": "Spatial Reasoning",
        "Constraint Composition": "Spatial Reasoning",
        "Geometric Inference": "Spatial Reasoning",
        "Scene Updating": "Spatial Memory",
        "Identity Tracking": "Spatial Memory",
    }
    for fam in FAMILY_DISPLAY:
        sub = df[(df["task_family"] == fam) & (df["suite"].astype(str) == "Base")]
        if sub.empty:
            continue
        r = sub.iloc[0]
        sr = float(r["success_rate"])
        fail = 100.0 * (1.0 - sr)
        sge = float(r["mean_sge"]) if pd.notna(r.get("mean_sge")) else ""
        tte = float(r["mean_mta"]) if pd.notna(r.get("mean_mta")) else ""
        rows.append(
            {
                "family": fam,
                "capability_dimension": cap_dim[fam],
                "failure_rate": f"{fail:.1f}",
                "sge": f"{sge:.4f}" if isinstance(sge, float) else "",
                "tte": f"{tte:.4f}" if isinstance(tte, float) else "",
                "main_failure_type": failure_types[fam],
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["family", "capability_dimension", "failure_rate", "sge", "tte", "main_failure_type"],
        )
        w.writeheader()
        w.writerows(rows)


def export_task_level_sample(repo: Path, out: Path) -> None:
    """从 OpenVLA summary 导出 strict 档 task 级成功率（长表）。"""
    path = repo / "run_eval_openvla" / "summary_by_task_level.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    sub = df[df["success_level"].astype(str).str.lower() == "strict"]
    rows = []
    for r in sub.itertuples(index=False):
        tid = str(getattr(r, "task_id", "") or "")
        fam_raw = str(getattr(r, "family", "") or "")
        fam = INTERNAL_FAMILY_TO_TAXONOMY.get(fam_raw, fam_raw)
        can = str(getattr(r, "canonical_task_id", "") or "")
        sr = getattr(r, "success_rate", None)
        sc = getattr(r, "success_count", None)
        n = getattr(r, "completed_trials", None) or getattr(r, "planned_trials", None)
        if sr is None or str(sr).strip() == "":
            continue
        try:
            rate = float(sr)
        except ValueError:
            continue
        rows.append(
            {
                "task_id": tid,
                "family": fam,
                "structure": can,
                "instruction_summary": "",
                "method": "OpenVLA",
                "success_rate": f"{100.0 * rate:.1f}",
                "num_trials": str(int(n)) if n is not None and str(n).isdigit() else str(n or ""),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "family",
                "structure",
                "instruction_summary",
                "method",
                "success_rate",
                "num_trials",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 paper/data/appendix/*.csv")
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument(
        "--family-long",
        type=Path,
        default=None,
        help="family_by_model_long.csv 路径",
    )
    ap.add_argument("--oracle-percent", type=float, default=100.0, help="Oracle 行占位成功率 (%)")
    args = ap.parse_args()
    root = args.repo_root.resolve()
    long_path = args.family_long or (
        root
        / "statistics"
        / "analysis_outputs"
        / "matrix_radar"
        / "run_eval_models_task_level_matrix_family_by_model_long.csv"
    )
    if not long_path.is_file():
        print(f"缺少 family long CSV: {long_path}", flush=True)
        return 1
    long_df = _read_family_long(long_path)
    long_pivot = _pivot_family_rates(long_df)
    _, matrix_by_slug = _load_matrix_numeric(root)
    if "openpi_05" not in matrix_by_slug and "openpi_0" in matrix_by_slug:
        matrix_by_slug["openpi_05"] = dict(matrix_by_slug["openpi_0"])

    out_dir = root / "paper" / "data" / "appendix"
    export_base_family(long_df, out_dir / "base_family_results.csv", oracle_percent=args.oracle_percent)
    export_aug_and_drops(
        root,
        matrix_by_slug,
        out_dir / "aug_results.csv",
        out_dir / "aug_drop_results.csv",
        long_pivot,
    )
    export_capability_demand(long_df, out_dir / "capability_demand_results.csv", oracle_percent=args.oracle_percent)
    export_diagnostics(root, out_dir / "diagnostics_results.csv")
    export_task_level_sample(root, out_dir / "task_level_results.csv")

    print(
        "已写入:\n"
        f"  {out_dir / 'base_family_results.csv'}\n"
        f"  {out_dir / 'aug_results.csv'}\n"
        f"  {out_dir / 'aug_drop_results.csv'}\n"
        f"  {out_dir / 'capability_demand_results.csv'}\n"
        f"  {out_dir / 'diagnostics_results.csv'}\n"
        f"  {out_dir / 'task_level_results.csv'}\n"
        f"Oracle 占位: {args.oracle_percent:.1f}%（可用 --oracle-percent 覆盖）",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
