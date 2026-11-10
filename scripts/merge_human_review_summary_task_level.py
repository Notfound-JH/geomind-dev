#!/usr/bin/env python3
"""
将 human_review_state.json 与 trial_results.jsonl（判别器 / 自动 trial 成功）按 trial 合并，
再按 task_id + success_level 聚合为 summary_by_task_level 风格 CSV。

合并规则（单条 trial 是否计为成功）：
- 若该 trial 落在 meta.human_sample_pools 任一 task 的池内：以状态文件中的 human_success 为准
  （含手标与脚本写入的 auto_rule_success）。
- 若不在抽样池内：以 trial_results.jsonl 的 success 为准；
  但若状态里存在非 auto_rule_success 的人工条目（例如全量复核），仍采用该 human_success。

默认输出：<eval_root>/summary_by_task_level_human.csv
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
    level = rec["success_level"]
    seed = str(rec["seed"])
    ep_dir = rec.get("episode_dir") or ""
    name = Path(ep_dir).name
    m = EPISODE_DIR_RE.match(name)
    ep = m.group(1) if m else "0"
    return f"{tid}|{level}|{seed}|{ep}"


def load_in_pool_trial_ids(state: dict) -> set[str]:
    meta = state.get("meta") or {}
    pools = meta.get("human_sample_pools")
    if not isinstance(pools, dict):
        return set()
    out: set[str] = set()
    for _task, ids in pools.items():
        if isinstance(ids, list):
            out.update(str(x) for x in ids)
    return out


def load_labels_raw(state: dict) -> dict[str, dict]:
    raw: dict[str, dict] = {}
    for k, v in (state.get("labels") or {}).items():
        if isinstance(v, dict) and "human_success" in v:
            raw[str(k)] = v
    return raw


def merged_success_for_trial(
    trial_id: str,
    task_id: str,
    in_pool: bool,
    label_rec: dict | None,
    auto_success: bool,
) -> bool:
    if label_rec is None:
        return bool(auto_success)
    src = label_rec.get("source")
    if in_pool:
        return bool(int(label_rec["human_success"]))
    if src != "auto_rule_success":
        return bool(int(label_rec["human_success"]))
    return bool(auto_success)


def main() -> int:
    p = argparse.ArgumentParser(description="合并人工复核状态与 trial_results，生成 summary_by_task_level_human.csv")
    p.add_argument(
        "--eval-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "run_eval_internvla",
        help="eval 输出根目录（含 trial_results.jsonl、summary_by_task_level.csv）",
    )
    p.add_argument(
        "--human-state",
        type=Path,
        default=None,
        help="human_review_state.json，默认 <eval-root>/human_review_state.json",
    )
    p.add_argument(
        "--trial-results",
        type=Path,
        default=None,
        help="trial_results.jsonl，默认 <eval-root>/trial_results.jsonl",
    )
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help="原 summary_by_task_level.csv，默认 <eval-root>/summary_by_task_level.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出路径，默认 <eval-root>/summary_by_task_level_human.csv",
    )
    args = p.parse_args()
    root = args.eval_root.resolve()
    state_path = (args.human_state or (root / "human_review_state.json")).resolve()
    tr_path = (args.trial_results or (root / "trial_results.jsonl")).resolve()
    base_path = (args.baseline_csv or (root / "summary_by_task_level.csv")).resolve()
    out_path = (args.out or (root / "summary_by_task_level_human.csv")).resolve()

    if not state_path.is_file():
        raise SystemExit(f"找不到状态文件: {state_path}")
    if not tr_path.is_file():
        raise SystemExit(f"找不到 trial_results: {tr_path}")
    if not base_path.is_file():
        raise SystemExit(f"找不到 baseline CSV: {base_path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    in_pool = load_in_pool_trial_ids(state)
    labels = load_labels_raw(state)

    # task_id|level -> list of (trial_id, auto_s, merged_s)
    groups: dict[tuple[str, str], list[tuple[str, bool, bool]]] = defaultdict(list)

    with tr_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = trial_id_from_jsonl(rec)
            task_id = rec["task_id"]
            level = rec["success_level"]
            auto_s = bool(rec.get("success"))
            ip = tid in in_pool
            merged = merged_success_for_trial(tid, task_id, ip, labels.get(tid), auto_s)
            groups[(task_id, level)].append((tid, auto_s, merged))

    # baseline order + metadata
    rows_out: list[dict[str, str]] = []
    with base_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_in = reader.fieldnames or []
        for row in reader:
            task_id = row["task_id"]
            level = row["success_level"]
            key = (task_id, level)
            trials = groups.get(key, [])
            n = len(trials)
            out_row = dict(row)
            if not trials:
                out_row["success_count_discriminator"] = row.get("success_count", "")
                out_row["success_rate_discriminator"] = row.get("success_rate", "")
                out_row["success_count"] = row.get("success_count", "")
                out_row["success_rate"] = row.get("success_rate", "")
                rows_out.append(out_row)
                continue

            auto_cnt = sum(1 for _, a, _ in trials if a)
            merged_cnt = sum(1 for _, _, m in trials if m)
            auto_rate = (auto_cnt / n) if n else 0.0
            merged_rate = (merged_cnt / n) if n else 0.0
            # 保留原列名语义：用判别器列 + 合并列
            out_row["success_count_discriminator"] = str(auto_cnt)
            out_row["success_rate_discriminator"] = f"{auto_rate:.6f}"
            out_row["success_count"] = str(merged_cnt)
            out_row["success_rate"] = f"{merged_rate:.6f}"
            rows_out.append(out_row)

    # 输出列：在原有列基础上插入 discriminator 两列（若原 CSV 已有 success_count，先保留其它列再写新 success_*）
    extra = ["success_count_discriminator", "success_rate_discriminator"]
    base_cols = [c for c in fieldnames_in if c not in ("success_count", "success_rate")]
    out_fields = base_cols + extra + ["success_count", "success_rate"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in out_fields})

    print(f"已写入: {out_path}")
    print(f"  trial 条数（jsonl）: {sum(len(v) for v in groups.values())}")
    print(f"  抽样池 trial 数: {len(in_pool)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
