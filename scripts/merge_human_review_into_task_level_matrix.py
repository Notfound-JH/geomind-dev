#!/usr/bin/env python3
"""
将某次 eval 的 human_review_state.json 与 trial_results.jsonl 合并后，
写回仓库根目录的 run_eval_models_task_level_matrix.csv 中对应模型行。

规则（与「按 task×level 成功率」一致）：
- 对每个 (task_id, success_level) 列：仅当该组内至少有一条 trial 在 state.labels 中有条目时才重算该格；
  否则保留 CSV 原值（含 \"-\"）。
- 组内各 trial：labels 中有则用 human_success（0/1），否则用 jsonl 的 success 布尔。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


EPISODE_DIR_RE = re.compile(r"episode_(\d+)$")


def trial_id_from_jsonl(rec: dict) -> str:
    tid = rec["task_id"]
    level = str(rec["success_level"]).strip().lower()
    seed = str(rec["seed"])
    ep_dir = rec.get("episode_dir") or ""
    name = Path(ep_dir).name
    m = EPISODE_DIR_RE.match(name)
    ep = m.group(1) if m else "0"
    return f"{tid}|{level}|{seed}|{ep}"


def load_labels_int(state: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in (state.get("labels") or {}).items():
        if isinstance(v, dict) and "human_success" in v:
            out[str(k)] = int(v["human_success"])
        elif isinstance(v, int):
            out[str(k)] = int(v)
    return out


def parse_matrix_column(col: str) -> tuple[str, str] | None:
    if "__" not in col or col in ("run_dir", "model_slug"):
        return None
    tid, lv = col.rsplit("__", 1)
    tid, lv = tid.strip(), lv.strip().lower()
    if not tid or lv not in ("loose", "medium", "strict"):
        return None
    return tid, lv


def format_rate(r: float) -> str:
    if abs(r - round(r)) < 1e-12 and 0 <= r <= 1:
        return str(int(round(r)))
    s = f"{r:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def main() -> int:
    p = argparse.ArgumentParser(description="用 human_review + trial_results 更新 task-level 矩阵 CSV 中一行")
    p.add_argument("--eval-root", type=Path, required=True, help="如 run_eval_openpi_0")
    p.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="矩阵 CSV（默认仓库根下 run_eval_models_task_level_matrix.csv）",
    )
    p.add_argument(
        "--match-run-dir",
        type=str,
        default=None,
        help="要更新的行的 run_dir 列值（默认与 --eval-root 目录名一致）",
    )
    p.add_argument(
        "--human-state",
        type=Path,
        default=None,
        help="默认 <eval-root>/human_review_state.json",
    )
    p.add_argument(
        "--trial-results",
        type=Path,
        default=None,
        help="默认 <eval-root>/trial_results.jsonl",
    )
    p.add_argument("--dry-run", action="store_true", help="只打印将改动的列，不写文件")
    args = p.parse_args()

    eval_root: Path = args.eval_root.resolve()
    repo_root = eval_root.parent
    matrix_path = (args.matrix or (repo_root / "run_eval_models_task_level_matrix.csv")).resolve()
    state_path = (args.human_state or (eval_root / "human_review_state.json")).resolve()
    jsonl_path = (args.trial_results or (eval_root / "trial_results.jsonl")).resolve()

    if not matrix_path.is_file():
        raise SystemExit(f"找不到矩阵: {matrix_path}")
    if not state_path.is_file():
        raise SystemExit(f"找不到人工状态: {state_path}")
    if not jsonl_path.is_file():
        raise SystemExit(f"找不到 trial_results: {jsonl_path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    labels = load_labels_int(state)

    # (task_id, level) -> list of (trial_id, auto_01)
    groups: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    with jsonl_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{jsonl_path}:{line_no} JSON 错误: {e}") from e
            if not isinstance(rec, dict) or "task_id" not in rec:
                continue
            tid = rec.get("task_id")
            sl = str(rec.get("success_level") or "").strip().lower()
            if not tid or sl not in ("loose", "medium", "strict"):
                continue
            trial_id = trial_id_from_jsonl(rec)
            auto = 1 if bool(rec.get("success")) else 0
            groups[(str(tid), sl)].append((trial_id, auto))

    run_dir_match = (args.match_run_dir or eval_root.name).strip()

    with matrix_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        raise SystemExit("矩阵 CSV 至少需要表头 + 说明行 + 一行数据")

    header = rows[0]
    try:
        run_i = header.index("run_dir")
        slug_i = header.index("model_slug")
    except ValueError as e:
        raise SystemExit("矩阵首行缺少 run_dir 或 model_slug") from e

    target_row_idx: int | None = None
    for i in range(2, len(rows)):
        if len(rows[i]) <= run_i:
            continue
        if rows[i][run_i].strip() == run_dir_match:
            target_row_idx = i
            break
    if target_row_idx is None:
        raise SystemExit(f"矩阵中未找到 run_dir={run_dir_match!r} 的行")

    old_row = rows[target_row_idx]
    new_row = list(old_row)
    while len(new_row) < len(header):
        new_row.append("")

    updated_cols: list[str] = []
    for j, col in enumerate(header):
        if j <= slug_i:
            continue
        pk = parse_matrix_column(col)
        if pk is None:
            continue
        g = groups.get(pk)
        if not g:
            continue
        reviewed_here = [tid for tid, _ in g if tid in labels]
        if not reviewed_here:
            continue
        num = den = 0
        for trial_id, auto in g:
            if trial_id in labels:
                num += labels[trial_id]
            else:
                num += auto
            den += 1
        rate = num / den if den else 0.0
        new_row[j] = format_rate(rate)
        updated_cols.append(col)

    if args.dry_run:
        print(f"将更新 run_dir={run_dir_match} 共 {len(updated_cols)} 列（有复核 trial 的 task×level）:")
        for c in updated_cols[:40]:
            print(f"  {c}")
        if len(updated_cols) > 40:
            print(f"  ... 共 {len(updated_cols)} 列")
        return 0

    rows[target_row_idx] = new_row
    tmp = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerows(rows)
    tmp.replace(matrix_path)
    print(f"已写回: {matrix_path}（{run_dir_match} 行更新 {len(updated_cols)} 列）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
