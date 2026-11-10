#!/usr/bin/env python3
"""
将 suite 输出的 summary.json（顶层 tasks[] + episodes）与
geomind/configs/current_task_mapping.json 对齐，生成分类统计：

1) <stem>_suite_tasks_with_taxonomy.csv — 每条 task 一行：成功率 + taxonomy 字段
2) <stem>_suite_by_taxonomy_family.csv — 按 taxonomy_family 聚合（任务数、成功数、成功率）

默认只汇总 mapping 中 suite 与 summary.json 顶层 group 一致的任务（常见 group=base）。

可选 --record-augmented-oracle-success：summary 未跑 augmented 时，将 mapping 中
suite==augmented 且尚未出现在 summary 的任务按「oracle 已核对为成功」各记 1/1，
并入 per-task 与 by-family 表（stats_source=oracle_verified_augmented）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_mapping(path: Path) -> tuple[dict[str, dict], str | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        raise SystemExit(f"mapping 缺少 tasks 列表: {path}")
    by_id: dict[str, dict] = {}
    for row in tasks:
        tid = row.get("task_id")
        if isinstance(tid, str) and tid.strip():
            by_id[tid.strip()] = row
    schema = data.get("schema_version")
    return by_id, schema if isinstance(schema, str) else None


def episode_success_01(summary: dict) -> bool | None:
    ev = summary.get("evaluation")
    if isinstance(ev, dict) and "success" in ev:
        return bool(ev["success"])
    if "success" in summary:
        return bool(summary["success"])
    return None


def task_aggregate_success(task_block: dict) -> tuple[int, int, float | None]:
    """(success_count, n_episodes, rate) 优先用 task 级字段，否则从 episodes 推断。"""
    eps = task_block.get("episodes") or []
    n = int(task_block.get("episodes_total") or len(eps))
    sc = task_block.get("success_count")
    sr = task_block.get("success_rate")
    if isinstance(sc, (int, float)) and n > 0:
        s = int(sc)
        rate = float(s) / float(n)
        return s, n, rate
    if isinstance(sr, (int, float)) and n > 0:
        s = int(round(float(sr) * n))
        return s, n, float(sr)
    ok_list: list[bool] = []
    for ep in eps:
        sm = ep.get("summary")
        if not isinstance(sm, dict):
            continue
        v = episode_success_01(sm)
        if v is not None:
            ok_list.append(v)
    if not ok_list:
        return 0, 0, None
    s = sum(1 for x in ok_list if x)
    m = len(ok_list)
    return s, m, (s / m) if m else None


def augmented_oracle_rows(
    mapping_by_id: dict[str, dict],
    already: set[str],
    *,
    episodes_per_task: int,
    success_per_task: int,
) -> list[dict[str, object]]:
    """mapping 中 suite==augmented 且不在 summary 中的任务，按 oracle 核对成功写入。"""
    out: list[dict[str, object]] = []
    for tid, meta in sorted(mapping_by_id.items()):
        if str(meta.get("suite") or "") != "augmented":
            continue
        if tid in already:
            continue
        fam = str(meta.get("taxonomy_family") or "")
        can = str(meta.get("canonical_task_id") or meta.get("task_id") or "")
        dim = str(meta.get("dimension") or "")
        ax = str(meta.get("augmentation_axis") or "")
        n = int(episodes_per_task)
        s = int(success_per_task)
        if s > n or n < 1:
            raise SystemExit("augmented oracle 参数无效: success_per_task <= episodes_per_task 且 n>=1")
        rate = float(s) / float(n)
        out.append(
            {
                "task_id": tid,
                "taxonomy_family": fam,
                "canonical_task_id": can,
                "dimension": dim,
                "augmentation_axis": ax,
                "mapping_suite": "augmented",
                "summary_group": "augmented_oracle_verified",
                "success_count": s,
                "episode_count": n,
                "success_rate": f"{rate:.6f}",
                "stats_source": "oracle_verified_augmented",
            }
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="suite summary.json × current_task_mapping 分类报表")
    p.add_argument(
        "--suite-summary",
        type=Path,
        required=True,
        help="suite 的 summary.json 路径",
    )
    p.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="current_task_mapping.json（默认仓库 geomind/configs/current_task_mapping.json）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录（默认同 summary.json 所在目录）",
    )
    p.add_argument(
        "--include-unmapped",
        action="store_true",
        help="仍输出 mapping 中无 task_id 的行（taxonomy 列为空）",
    )
    p.add_argument(
        "--record-augmented-oracle-success",
        action="store_true",
        help="将 mapping 中 augmented 且未出现在 summary 的任务按各 1/1 成功并入统计（oracle 已核对）",
    )
    p.add_argument(
        "--augmented-episodes-per-task",
        type=int,
        default=1,
        metavar="N",
        help="与 --record-augmented-oracle-success 联用：每个 augmented 任务计为 N 条 episode（默认 1）",
    )
    p.add_argument(
        "--augmented-success-count",
        type=int,
        default=None,
        metavar="K",
        help="与上一参数联用：每个 augmented 任务成功条数（默认等于 episodes-per-task 即全成功）",
    )
    args = p.parse_args()

    summary_path = args.suite_summary.resolve()
    repo = Path(__file__).resolve().parents[1]
    mapping_path = (args.mapping or (repo / "geomind/configs/current_task_mapping.json")).resolve()
    if not summary_path.is_file():
        raise SystemExit(f"找不到 summary: {summary_path}")
    if not mapping_path.is_file():
        raise SystemExit(f"找不到 mapping: {mapping_path}")

    out_dir = (args.out_dir or summary_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = summary_path.stem
    per_task_csv = out_dir / f"{stem}_suite_tasks_with_taxonomy.csv"
    by_fam_csv = out_dir / f"{stem}_suite_by_taxonomy_family.csv"

    mapping_by_id, schema_ver = load_mapping(mapping_path)
    suite_doc = json.loads(summary_path.read_text(encoding="utf-8"))
    group = suite_doc.get("group")
    if not isinstance(group, str):
        group = ""

    rows_out: list[dict[str, object]] = []
    unmapped: list[str] = []

    for task_block in suite_doc.get("tasks") or []:
        if not isinstance(task_block, dict):
            continue
        tid = str(task_block.get("task_id") or "").strip()
        if not tid:
            continue
        meta = mapping_by_id.get(tid)
        if meta is None:
            unmapped.append(tid)
            if not args.include_unmapped:
                continue
            fam = can = dim = ax = suite_m = ""
        else:
            suite_m = str(meta.get("suite") or "")
            if group and suite_m and suite_m != group:
                continue
            fam = str(meta.get("taxonomy_family") or "")
            can = str(meta.get("canonical_task_id") or meta.get("task_id") or "")
            dim = str(meta.get("dimension") or "")
            ax = str(meta.get("augmentation_axis") or "")

        s, n, rate = task_aggregate_success(task_block)
        rows_out.append(
            {
                "task_id": tid,
                "taxonomy_family": fam,
                "canonical_task_id": can,
                "dimension": dim,
                "augmentation_axis": ax,
                "mapping_suite": suite_m if meta is not None else "",
                "summary_group": group,
                "success_count": s,
                "episode_count": n,
                "success_rate": "" if rate is None else f"{float(rate):.6f}",
                "stats_source": "suite_summary",
            }
        )

    if unmapped and not args.include_unmapped:
        print(f"[warn] mapping 中无条目的 task_id（已跳过 {len(unmapped)} 个）:", file=sys.stderr)
        for u in sorted(set(unmapped))[:30]:
            print(f"  {u}", file=sys.stderr)
        if len(set(unmapped)) > 30:
            print(f"  ... 共 {len(set(unmapped))} 个", file=sys.stderr)

    augmented_appended: list[str] = []
    if args.record_augmented_oracle_success:
        n_ep = max(1, int(args.augmented_episodes_per_task))
        k_succ = args.augmented_success_count
        if k_succ is None:
            k_succ = n_ep
        extra = augmented_oracle_rows(
            mapping_by_id,
            {str(r["task_id"]) for r in rows_out},
            episodes_per_task=n_ep,
            success_per_task=int(k_succ),
        )
        augmented_appended = [str(r["task_id"]) for r in extra]
        rows_out.extend(extra)
        print(
            f"[info] 已并入 augmented oracle 成功条目: {len(extra)} 个 task",
            file=sys.stderr,
        )

    fieldnames = [
        "task_id",
        "taxonomy_family",
        "canonical_task_id",
        "dimension",
        "augmentation_axis",
        "mapping_suite",
        "summary_group",
        "success_count",
        "episode_count",
        "success_rate",
        "stats_source",
    ]
    with per_task_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sorted(rows_out, key=lambda r: (str(r["taxonomy_family"]), str(r["task_id"]))):
            w.writerow({k: row.get(k, "") for k in fieldnames})

    agg: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for row in rows_out:
        fam = str(row["taxonomy_family"]) or "<unmapped>"
        s = int(row["success_count"] or 0)
        n = int(row["episode_count"] or 0)
        a, b = agg[fam]
        agg[fam] = (a + s, b + n)

    with by_fam_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "taxonomy_family",
                "success_count",
                "episode_count",
                "success_rate",
                "mapping_schema_version",
                "suite_summary_path",
            ]
        )
        for fam in sorted(agg.keys()):
            s, n = agg[fam]
            rate = (s / n) if n else 0.0
            w.writerow([fam, s, n, f"{rate:.6f}", schema_ver or "", str(summary_path)])
    meta_json = out_dir / f"{stem}_suite_taxonomy_report_meta.json"
    meta_json.write_text(
        json.dumps(
            {
                "suite_summary": str(summary_path),
                "mapping": str(mapping_path),
                "summary_group_filter": group,
                "mapping_schema_version": schema_ver,
                "suite_head": {
                    k: suite_doc.get(k)
                    for k in (
                        "policy",
                        "interactive",
                        "task_count",
                        "episode_count",
                        "success_count",
                        "success_rate",
                        "backend",
                    )
                    if k in suite_doc
                },
                "per_task_csv": str(per_task_csv),
                "by_family_csv": str(by_fam_csv),
                "n_tasks_written": len(rows_out),
                "n_unmapped_skipped": len(set(unmapped)) if not args.include_unmapped else 0,
                "record_augmented_oracle_success": bool(args.record_augmented_oracle_success),
                "oracle_augmented_task_ids_appended": augmented_appended,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[info] wrote: {per_task_csv}")
    print(f"[info] wrote: {by_fam_csv}")
    print(f"[info] wrote: {meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
