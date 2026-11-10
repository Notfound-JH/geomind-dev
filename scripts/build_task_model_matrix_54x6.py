#!/usr/bin/env python3
"""
从 run_eval_models_task_level_matrix.csv 构造 54×6 任务×模型成功率表（CSV）。

规则（与用户约定一致）：
  - 27 个 task_id（矩阵中每个任务三档 loose / medium / strict）。
  - 每个任务两行：
      (1) base：在 loose 与 medium 上成功率的算术平均（缺失档用另一档；两档皆缺则空）。
      (2) aug_execution：在 strict 上的成功率，视为该任务的「aug execution」版本。
  - 共 54 行；首列 task_name 来自矩阵第二行层级说明（去掉「宽松/中等/严格(loose|…)」后缀）。
  - 完整 CSV 列序：task_name, task_id, row_kind, row_label, + 6 个 VLA。
  - models_only CSV：task_name + 6 个 VLA。
  - 若矩阵中无 openpi_05 行，则用 openpi_0 的数值复制填充（与附录导出逻辑一致）。
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# 矩阵第二行：如 anchor_reference-grasp_anchor_direction-宽松(loose)
_TIER_SUFFIX = re.compile(r"-(?:宽松|中等|严格)\((?:loose|medium|strict)\)\s*$")

MODEL_SLUGS = ["openvla", "openpi_0", "openpi_05", "spatialvla", "internvla", "spatialforcing"]
ROW_KIND_BASE = "base_loose_medium_mean"
ROW_KIND_AUG = "aug_execution_strict"


def _parse_cell(s: str) -> float | None:
    t = (s or "").strip()
    if t in {"", "-"}:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _mean_loose_medium(a: float | None, b: float | None) -> float | None:
    if a is not None and b is not None:
        return (a + b) / 2.0
    if a is not None:
        return a
    if b is not None:
        return b
    return None


def _pct(x: float | None) -> str:
    if x is None:
        return ""
    return f"{100.0 * x:.1f}"


def task_name_from_tier(tid: str, tier_row: list[str], col_index: dict[str, int]) -> str:
    col = f"{tid}__loose"
    i = col_index.get(col)
    if i is None or i >= len(tier_row):
        return tid
    raw = (tier_row[i] or "").strip()
    if not raw:
        return tid
    return _TIER_SUFFIX.sub("", raw) or tid


def task_order_from_header(header: list[str]) -> list[str]:
    seen: list[str] = []
    for c in header[2:]:
        if not c or "__" not in c:
            continue
        tid = c.rsplit("__", 1)[0]
        if tid not in seen:
            seen.append(tid)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--numeric-only-out",
        type=Path,
        default=None,
        help="另存 54 行：首列 task_name + 6 列模型成功率（%%）",
    )
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root = args.repo_root.resolve()
    matrix_path = args.matrix or (root / "run_eval_models_task_level_matrix.csv")
    out_path = args.out or (root / "paper" / "data" / "appendix" / "task_model_success_54x6.csv")

    with matrix_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        raise SystemExit(f"矩阵过短: {matrix_path}")
    header = rows[0]
    tier_row = rows[1] if len(rows) > 1 else []
    task_ids = task_order_from_header(header)
    if len(task_ids) != 27:
        raise SystemExit(f"期望 27 个 task_id，实际 {len(task_ids)}")

    col_index = {name: i for i, name in enumerate(header)}

    def cell(row: list[str], col_name: str) -> float | None:
        i = col_index.get(col_name)
        if i is None or i >= len(row):
            return None
        return _parse_cell(row[i])

    by_slug: dict[str, list[str]] = {}
    for r in rows[2:]:
        if not r or len(r) < 2:
            continue
        slug = (r[1] or "").strip()
        if not slug or slug == "all_models_suites":
            continue
        by_slug[slug] = r

    if "openpi_05" not in by_slug and "openpi_0" in by_slug:
        by_slug["openpi_05"] = list(by_slug["openpi_0"])

    out_rows: list[dict[str, str]] = []
    for tid in task_ids:
        tname = task_name_from_tier(tid, tier_row, col_index)
        c_lo = f"{tid}__loose"
        c_md = f"{tid}__medium"
        c_st = f"{tid}__strict"
        base_vals: dict[str, str] = {}
        aug_vals: dict[str, str] = {}
        for slug in MODEL_SLUGS:
            r = by_slug.get(slug)
            if not r:
                base_vals[slug] = ""
                aug_vals[slug] = ""
                continue
            lo = cell(r, c_lo)
            md = cell(r, c_md)
            st = cell(r, c_st)
            base_vals[slug] = _pct(_mean_loose_medium(lo, md))
            aug_vals[slug] = _pct(st)
        out_rows.append(
            {
                "task_name": tname,
                "task_id": tid,
                "row_kind": ROW_KIND_BASE,
                "row_label": f"{tid}__base_loose_medium",
                **{slug: base_vals[slug] for slug in MODEL_SLUGS},
            }
        )
        out_rows.append(
            {
                "task_name": tname,
                "task_id": tid,
                "row_kind": ROW_KIND_AUG,
                "row_label": f"{tid}__aug_execution_strict",
                **{slug: aug_vals[slug] for slug in MODEL_SLUGS},
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task_name", "task_id", "row_kind", "row_label", *MODEL_SLUGS]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {out_path} ({len(out_rows)} rows, task_name+task_id+kind + {len(MODEL_SLUGS)} models)")

    num_out = args.numeric_only_out or (out_path.with_name(out_path.stem + "_models_only.csv"))
    with num_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_name", *MODEL_SLUGS])
        w.writeheader()
        for rec in out_rows:
            row = {"task_name": rec["task_name"], **{slug: rec[slug] for slug in MODEL_SLUGS}}
            w.writerow(row)
    print(f"Wrote {num_out} (54 rows: task_name + 6 models, success rate %%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
