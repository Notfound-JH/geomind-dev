"""reach 假阳性审计：起手位姿是否已经落在成功半径内。

背景
  评测全程 dwell_steps_required = 1，也就是说 `reach_success` 退化成「任意单步进入
  success_radius」，没有任何停留要求。当一个 episode 的初始位姿本来就在半径内时，
  reach 在第 0~2 步就被触发，策略一步都不用走对——这不是能力，是布局。

判据（从严到宽）
  start_inside : d0 <= success_radius        起手即在半径内，reach 与策略无关
  reach_le3    : reach_step <= 3             起手附近触发（含刚好压线后一两步抖进去）
  net_reach    : reach_success and not start_inside

d0 取环境真值 goal 的 xy 距离 ||goal_error_0||，与 SGE/TTE 同源（见 paper/rebuttal_metrics.md §1.3）。

用法
  /home/elm/miniconda3/envs/isaac_curobo/bin/python scripts/pi05_reach_false_positive_audit.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

RUNS = {"base": "run_eval_pi05_v4", "direct": "run_eval_pi05_v4_direct"}
PATH_RE = re.compile(r"/runs/([^/]+)/([^/]+)/seed_(\d+)/")
FAMILY_MAP = [
    ("Binding", "Target Identification"),
    ("Relation", "Relational Grounding"),
    ("Constraint", "Constraint Composition"),
    ("Geometric", "Geometric Inference"),
    ("Reindex", "Scene Updating"),
    ("Track", "Identity Tracking"),
]


def family_of(task: str) -> str:
    for prefix, name in FAMILY_MAP:
        if task.startswith(prefix):
            return name
    return "Unknown"


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确 McNemar：n=b+c 次投币里出现 >=max(b,c) 次一面的概率。"""
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2.0 * tail / (2**n))


DWELL_LEVELS = (1, 3, 5)


def _in_radius_flags(trace: list) -> list[bool]:
    flags: list[bool] = []
    for step in trace:
        meta = step.get("meta") or {}
        try:
            distance = float(meta.get("goal_distance"))
            radius = float(meta.get("success_radius"))
        except (TypeError, ValueError):
            flags.append(False)
            continue
        flags.append(radius > 0.0 and distance <= radius)
    return flags


def _first_dwell_step(flags: list[bool], dwell: int) -> int | None:
    """首次出现 dwell 个连续在半径内的步时，那一段的末步下标。与 evaluators/simple.py 同语义。"""
    streak = 0
    for index, flag in enumerate(flags):
        streak = streak + 1 if flag else 0
        if streak >= dwell:
            return index
    return None


def episode_row(result_path: str, variant: str) -> dict | None:
    m = PATH_RE.search(result_path)
    if m is None:
        return None
    try:
        result = json.load(open(result_path))
        trace = json.load(open(result_path.replace("result.json", "trace.json")))
    except (OSError, ValueError):
        return None
    if not trace:
        return None
    first = trace[0].get("agent_obs", {}).get("state", {}).get("native_obs", {}).get("policy", {})
    goal_error = first.get("goal_error")
    if not isinstance(goal_error, list) or len(goal_error) < 2:
        return None
    d0 = math.hypot(float(goal_error[0]), float(goal_error[1]))

    evaluation = result.get("evaluation") or {}
    radius = float(evaluation.get("success_radius") or 0.0)
    reach = bool(evaluation.get("reach_success"))
    reach_step = evaluation.get("reach_step")
    task = m.group(1)
    start_inside = radius > 0.0 and d0 <= radius
    row = {
        "variant": variant,
        "family": family_of(task),
        "task": task,
        "level": m.group(2),
        "seed": m.group(3),
        "d0": round(d0, 6),
        "success_radius": radius,
        "d0_minus_radius": round(d0 - radius, 6),
        "reach_success": int(reach),
        "reach_step": reach_step if reach_step is not None else "",
        "dwell_steps_required": evaluation.get("dwell_steps_required"),
        "success": int(bool(evaluation.get("success"))),
        "start_inside": int(start_inside),
        "reach_le3": int(reach and isinstance(reach_step, int) and reach_step <= 3),
        "net_reach": int(reach and not start_inside),
    }

    # 离线重算不同 dwell 阈值下的判定。evaluators/simple.py 的默认值是 5，本次评测被
    # GEOMIND_SUCCESS_DWELL_STEPS=1 覆盖，使 reach 退化成「任意单步进入半径」即 flyby 也算达成。
    #
    # LayoutAwareEvaluator 另有 goal_kind/resolver/metric_valid 守卫，reach_step 无论守卫是否通过
    # 都会被算出来，所以「守卫通过」等价于 reach_success == (reach_step is not None)。据此把
    # 守卫效应（修复 1 所针对的那个错路由）与 dwell 效应分开报。
    guard_ok = reach == (reach_step is not None)
    row["guard_ok"] = int(guard_ok)
    flags = _in_radius_flags(trace)
    for dwell in DWELL_LEVELS:
        step = _first_dwell_step(flags, dwell)
        window = flags[-dwell:]
        row[f"reach_dwell{dwell}"] = int(step is not None)
        row[f"reach_dwell{dwell}_guarded"] = int(guard_ok and step is not None)
        row[f"reach_step_dwell{dwell}"] = step if step is not None else ""
        row[f"success_dwell{dwell}"] = int(len(window) >= dwell and all(window))
    return row


def main() -> None:
    rows: list[dict] = []
    for variant, root in RUNS.items():
        for path in sorted(glob.glob(f"{root}/runs/*/*/seed_*/episode_0000/result.json")):
            row = episode_row(path, variant)
            if row is not None:
                rows.append(row)
        print(f"载入 {variant:7s} {sum(1 for r in rows if r['variant'] == variant)} 条")

    os.makedirs("analysis_outputs", exist_ok=True)
    out_csv = "analysis_outputs/pi05_reach_false_positive_audit.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"逐 episode: {out_csv}  ({len(rows)} 行)\n")

    print("dwell_steps_required:", dict(Counter(r["dwell_steps_required"] for r in rows)))
    print()

    print("=" * 104)
    print("[表 1] 起手即在半径内的 episode（start_inside）与 reach 的关系")
    print("=" * 104)
    for variant in RUNS:
        sub = [r for r in rows if r["variant"] == variant]
        inside = [r for r in sub if r["start_inside"]]
        reached = [r for r in sub if r["reach_success"]]
        fp = [r for r in reached if r["start_inside"]]
        net = [r for r in reached if not r["start_inside"]]
        print(
            f"  {variant:7s} n={len(sub):3d}  start_inside={len(inside):3d} ({len(inside)/len(sub)*100:4.1f}%)  "
            f"reach={len(reached):3d} ({len(reached)/len(sub)*100:4.1f}%)  "
            f"其中起手即在内={len(fp):3d}  净 reach={len(net):3d} ({len(net)/len(sub)*100:4.1f}%)"
        )
        print(f"          起手即在内的 episode 里有 {sum(r['reach_success'] for r in inside)}/{len(inside)} 触发了 reach")
    print()

    print("=" * 104)
    print("[表 2] 逐族净 reach（分母为该族 episode 数）")
    print("=" * 104)
    header = f"  {'族':<24s}"
    for variant in RUNS:
        header += f" {variant + ' reach':>14s} {variant + ' 起手内':>14s} {variant + ' 净':>14s}"
    print(header)
    for _, family in FAMILY_MAP:
        line = f"  {family:<24s}"
        for variant in RUNS:
            sub = [r for r in rows if r["variant"] == variant and r["family"] == family]
            if not sub:
                line += f" {'-':>14s} {'-':>14s} {'-':>14s}"
                continue
            reached = sum(r["reach_success"] for r in sub)
            inside = sum(r["start_inside"] for r in sub)
            net = sum(r["net_reach"] for r in sub)
            line += f" {f'{reached}/{len(sub)}':>14s} {f'{inside}/{len(sub)}':>14s} {f'{net}/{len(sub)}':>14s}"
        print(line)
    print()

    base_map = {(r["task"], r["level"], r["seed"]): r for r in rows if r["variant"] == "base"}
    direct_map = {(r["task"], r["level"], r["seed"]): r for r in rows if r["variant"] == "direct"}
    shared = sorted(set(base_map) & set(direct_map))

    print("=" * 104)
    print(f"[表 3] 配对子集 n={len(shared)} 上的 McNemar 精确检验：原始 reach vs 净 reach")
    print("=" * 104)
    for label, field in (("原始 reach_success", "reach_success"), ("净 reach（剔除起手即在内）", "net_reach")):
        b = sum(1 for k in shared if not base_map[k][field] and direct_map[k][field])
        c = sum(1 for k in shared if base_map[k][field] and not direct_map[k][field])
        base_rate = sum(base_map[k][field] for k in shared) / len(shared)
        direct_rate = sum(direct_map[k][field] for k in shared) / len(shared)
        p = mcnemar_exact(b, c)
        mark = "★显著" if p < 0.05 else " 不显著"
        print(
            f"  {label:<28s} base {base_rate*100:5.1f}%  direct {direct_rate*100:5.1f}%  "
            f"Δ{(direct_rate-base_rate)*100:+5.1f}  翻转 {b}/{c}  p={p:.4f} {mark}"
        )
    print()

    print("=" * 104)
    print("[表 4] 起手即在半径内的任务清单（按发生次数排序）")
    print("=" * 104)
    counter: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        if r["start_inside"]:
            counter[(r["family"], r["task"])].append(r)
    for (family, task), sub in sorted(counter.items(), key=lambda kv: -len(kv[1])):
        margins = sorted(r["d0_minus_radius"] for r in sub)
        triggered = sum(r["reach_success"] for r in sub)
        print(
            f"  {family:<22s} {task:<44s} {len(sub):>2d} 次  触发 reach {triggered:>2d}  "
            f"d0-radius ∈ [{margins[0]:+.4f}, {margins[-1]:+.4f}]"
        )

    print("\n" + "=" * 104)
    print("[表 5] 离线重算 dwell 阈值（evaluators/simple.py 默认 5，本次评测被环境变量压成 1）")
    print("       guarded = 保留 LayoutAwareEvaluator 的 goal_kind/resolver 守卫（与现有 result.json 同口径）")
    print("       裸值   = 去掉该守卫，相当于修复 1（place_directional 错路由）之后的预期值")
    print("=" * 104)
    verify = sum(r["reach_dwell1_guarded"] for r in rows if r["variant"] == "base")
    recorded = sum(r["reach_success"] for r in rows if r["variant"] == "base")
    print(f"  [自检] base dwell=1 guarded 重算={verify}  result.json 记录={recorded}  "
          f"{'一致 ✓' if verify == recorded else '不一致 ✗'}")
    print()
    for field_suffix, label in (("_guarded", "guarded"), ("", "裸值")):
        print(f"  --- {label} ---")
        print(f"  {'dwell':>5s}  {'base reach':>14s} {'direct reach':>14s}  {'配对 base':>10s} {'配对 direct':>12s} {'翻转 b/c':>10s} {'p(McNemar)':>12s}")
        for dwell in DWELL_LEVELS:
            field = f"reach_dwell{dwell}{field_suffix}"
            cells = ""
            for variant in RUNS:
                sub = [r for r in rows if r["variant"] == variant]
                hit = sum(r[field] for r in sub)
                cells += f" {f'{hit}/{len(sub)} ({hit/len(sub)*100:.1f}%)':>14s}"
            b = sum(1 for k in shared if not base_map[k][field] and direct_map[k][field])
            c = sum(1 for k in shared if base_map[k][field] and not direct_map[k][field])
            base_rate = sum(base_map[k][field] for k in shared) / len(shared)
            direct_rate = sum(direct_map[k][field] for k in shared) / len(shared)
            p = mcnemar_exact(b, c)
            mark = "★" if p < 0.05 else " "
            print(
                f"  {dwell:>5d} {cells}  {base_rate*100:>9.1f}% {direct_rate*100:>11.1f}% "
                f"{f'{b}/{c}':>10s} {p:>11.4f}{mark}"
            )
        print()

    print("=" * 104)
    print("[表 6] 逐族 reach（guarded 口径）：dwell=1（当前）vs dwell=5（默认）")
    print("=" * 104)
    header = f"  {'族':<24s}"
    for variant in RUNS:
        header += f" {variant + ' dwell1':>14s} {variant + ' dwell5':>14s}"
    print(header)
    for _, family in FAMILY_MAP:
        line = f"  {family:<24s}"
        for variant in RUNS:
            sub = [r for r in rows if r["variant"] == variant and r["family"] == family]
            if not sub:
                line += f" {'-':>14s} {'-':>14s}"
                continue
            d1 = sum(r["reach_dwell1_guarded"] for r in sub)
            d5 = sum(r["reach_dwell5_guarded"] for r in sub)
            line += f" {f'{d1}/{len(sub)}':>14s} {f'{d5}/{len(sub)}':>14s}"
        print(line)


if __name__ == "__main__":
    main()
