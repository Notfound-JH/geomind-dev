#!/usr/bin/env python3
"""
汇总仓库根目录下所有 run_eval_* 子目录的按任务 x 档位指标，生成宽表 CSV：
  每行 = 一个 run_eval_<name> 目录（模型 / 一次 eval）
  每列 = <task_id>__<loose|medium|strict>（默认 success_rate）

第 2 行为层级说明：任务类别(family)-任务名称(canonical_task_id)-判别标准(宽松/中等/严格)。
列按「任务类别 -> canonical -> task_id -> 档位」排序，便于同类任务聚在一起。

缺失文件或缺失 (task, level) 行时填 "-"。
列集为各 run 的 summary CSV 中出现过的 (task_id, success_level) 并集。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LEVEL_ORDER = ("loose", "medium", "strict")
LEVEL_RANK = {lv: i for i, lv in enumerate(LEVEL_ORDER)}
LEVEL_LABEL_CN = {"loose": "宽松", "medium": "中等", "strict": "严格"}


def _pick_summary_csv(run_dir: Path, prefer_human: bool) -> Path | None:
    human = run_dir / "summary_by_task_level_human.csv"
    auto = run_dir / "summary_by_task_level.csv"
    if prefer_human:
        if human.is_file():
            return human
        if auto.is_file():
            return auto
        return None
    if auto.is_file():
        return auto
    if human.is_file():
        return human
    return None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_column_key_set(run_dirs: list[Path], prefer_human: bool) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for d in run_dirs:
        p = _pick_summary_csv(d, prefer_human)
        if not p:
            continue
        for row in _read_rows(p):
            tid = (row.get("task_id") or "").strip()
            lv = (row.get("success_level") or "").strip().lower()
            if tid and lv in LEVEL_RANK:
                keys.add((tid, lv))
    return keys


def load_task_taxonomy_meta(run_dirs: list[Path], prefer_human: bool) -> dict[str, tuple[str, str]]:
    """task_id -> (family, canonical_task_id)，任取首次出现的 summary 行。"""
    meta: dict[str, tuple[str, str]] = {}
    for d in run_dirs:
        p = _pick_summary_csv(d, prefer_human)
        if not p:
            continue
        for row in _read_rows(p):
            tid = (row.get("task_id") or "").strip()
            if not tid or tid in meta:
                continue
            fam = (row.get("family") or "").strip() or "-"
            can = (row.get("canonical_task_id") or "").strip() or tid
            meta[tid] = (fam, can)
    return meta


def sort_keys_by_taxonomy(
    keys: set[tuple[str, str]],
    meta: dict[str, tuple[str, str]],
) -> list[tuple[str, str]]:
    def sort_key(item: tuple[str, str]) -> tuple[str, str, str, int]:
        tid, lv = item
        fam, can = meta.get(tid, ("-", tid))
        return (fam, can, tid, LEVEL_RANK[lv])

    return sorted(keys, key=sort_key)


def build_tier_label_row(
    col_keys: list[tuple[str, str]],
    col_names: list[str],
    meta: dict[str, tuple[str, str]],
) -> dict[str, str]:
    """第二行：任务类别-任务名称(semantic)-判别标准(中文+英文档位)。"""
    row: dict[str, str] = {
        "run_dir": "任务类别-任务名称-判别标准",
        "model_slug": "",
    }
    for (tid, lv), cname in zip(col_keys, col_names, strict=True):
        fam, can = meta.get(tid, ("-", tid))
        cn = LEVEL_LABEL_CN.get(lv, lv)
        row[cname] = f"{fam}-{can}-{cn}({lv})"
    return row


def load_run_values(
    run_dir: Path,
    prefer_human: bool,
    metric: str,
) -> dict[tuple[str, str], str]:
    p = _pick_summary_csv(run_dir, prefer_human)
    out: dict[tuple[str, str], str] = {}
    if not p:
        return out
    for row in _read_rows(p):
        tid = (row.get("task_id") or "").strip()
        lv = (row.get("success_level") or "").strip().lower()
        if not tid or lv not in LEVEL_RANK:
            continue
        val = (row.get(metric) or "").strip()
        if val == "":
            continue
        out[(tid, lv)] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="汇总 run_eval_* 为模型 x 任务 x 档位宽表")
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="仓库根目录（其下匹配 run_eval_*）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 CSV，默认 <repo-root>/run_eval_models_task_level_matrix.csv",
    )
    ap.add_argument(
        "--metric",
        choices=("success_rate", "success_count"),
        default="success_rate",
        help="从 summary CSV 读取的列名",
    )
    ap.add_argument(
        "--prefer-human",
        action="store_true",
        help="各 run 优先 summary_by_task_level_human.csv，否则 summary_by_task_level.csv",
    )
    args = ap.parse_args()
    root = args.repo_root.resolve()
    out_path = (args.out or (root / "run_eval_models_task_level_matrix.csv")).resolve()

    run_dirs = sorted([p for p in root.glob("run_eval_*") if p.is_dir()])
    if not run_dirs:
        print(f"在 {root} 下未找到 run_eval_* 目录。", flush=True)
        return 1

    key_set = collect_column_key_set(run_dirs, args.prefer_human)
    if not key_set:
        print("未从任何 summary CSV 解析到列，退出。", flush=True)
        return 1

    meta = load_task_taxonomy_meta(run_dirs, args.prefer_human)
    col_keys = sort_keys_by_taxonomy(key_set, meta)
    col_names = [f"{tid}__{lv}" for tid, lv in col_keys]

    rows_out: list[dict[str, str]] = []
    for d in run_dirs:
        vals = load_run_values(d, args.prefer_human, args.metric)
        row: dict[str, str] = {
            "run_dir": d.name,
            "model_slug": d.name[len("run_eval_") :] if d.name.startswith("run_eval_") else d.name,
        }
        for (tid, lv), cname in zip(col_keys, col_names, strict=True):
            v = vals.get((tid, lv))
            row[cname] = v if v is not None and v != "" else "-"
        rows_out.append(row)

    fields = ["run_dir", "model_slug"] + col_names
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tier_row = build_tier_label_row(col_keys, col_names, meta)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({k: tier_row.get(k, "") for k in fields})
        for r in rows_out:
            w.writerow({k: r.get(k, "-") for k in fields})

    print(
        f"已写入: {out_path}\n"
        f"  run 数: {len(run_dirs)}  总列数(含 run_dir, model_slug): {len(fields)}  即 {len(col_keys)} 个 task x level\n"
        f"  已插入第 2 行：任务类别-任务名称-判别标准；列已按任务类别聚合排序",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
