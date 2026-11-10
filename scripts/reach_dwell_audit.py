"""reach 口径离线重算与起始态审计（rebuttal 用）。

要回答的审稿意见
  "How do the authors disentangle failures caused by deficient spatial understanding
   from failures caused by low-level control, action-interface mismatch, or policy
   deployment artifacts?"

本脚本只负责把「口径」钉死，让后续对比站得住：

1. dwell 口径
   评测全程被 GEOMIND_SUCCESS_DWELL_STEPS=1 覆盖，reach_success 退化成「任意单步进入
   success_radius」，即 flyby 也算达成。这里离线重算 K = 1/3/5 三档。
2. 起始态伪影
   起手位姿本来就在半径内时（d0 <= success_radius），策略一步都不用走对，reach 就成立。
   reach_kK_excl_start = reach_kK and not start_inside，把这类剔掉。
   注意首达步的机械下界就是 K-1，所以「首达步靠前」在 K=5 下不构成额外判据，
   真正要排除的只有起手即在半径内这一种。
3. 步数预算
   oracle 控制器是恒速直线（每步位移 = magnitude 1.0 x action_scale 0.1），50 步是硬上限。
   near_budget 标出「贴着天花板才到」的 episode，这类失败是预算问题不是能力问题。

距离口径
  逐步判定一律用 trace[t].meta.goal_distance 与 success_radius 比较——这正是
  evaluators/simple.py 看到的量（对各向异性目标已做过重标定，是「判决」而非「长度」）。
  不使用 ||goal_error|| 做判定，也不使用 metrics_trajectory.py 的 *_target_distance
  系列（那套依赖名义布局坐标，有坐标系问题）。

强制验证门
  先用 K=1 重算，逐 episode 比对 result.json 里已有的 reach_success / reach_step。
  必须 100% 一致才输出 K=3/5 的结论；不一致会以非零退出码中止。

用法
  /home/elm/miniconda3/envs/isaac_curobo/bin/python scripts/reach_dwell_audit.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

RUNS = {
    # 旧的 4 task 版本，保留做复现性交叉验证（新旧应逐条一致）
    "scripted": "run_eval_scripted_ctl",
    # 全量 27 task，参数与 pi05_base 逐字一致
    "scripted_full": "run_eval_scripted_full",
    "pi05_base": "run_eval_pi05_v4",
    "pi05_direct": "run_eval_pi05_v4_direct",
    "pi0_base": "run_eval_openpi_05_v3",
}
ORACLE_RUN = "scripted_full"
LEARNED_RUN = "pi05_base"

# 接口 / 部署消融。全部是同一个 checkpoint（π0.5+镜像），只改动作接口或部署侧的一个旋钮。
# 用来量化「action-interface mismatch / deployment artifact」这条通道能解释多少失败，
# 从而说明主结果是 best-of-config 而不是「接口无所谓」。
IFACE_RUNS = {
    "noflip": "run_eval_pi05_noflip",          # 不做图像水平翻转
    "hflip": "run_eval_pi05_hflip",            # 做翻转（= 主配置的做法）
    "rot_on": "run_eval_pi05_rotON",           # 放开 action[3:6] 旋转维
    "proprio_libero": "run_eval_openpi_05_proprio_fix",  # LIBERO 风格 observation/state
    "chunk5": "run_eval_openpi_05_chunk_check",          # 动作块长度 5
    "scale005": "run_eval_openpi_05_scale_check",        # action_scale 0.05
    "scale01": "run_eval_openpi_05_scale01_check",       # action_scale 0.1
}
IFACE_REF = "pi05_base"

PATH_RE = re.compile(r"/runs/([^/]+)/([^/]+)/seed_(\d+)/")
DWELL_LEVELS = (1, 3, 5)
EARLY_STEP_MAX = 2
NEAR_BUDGET_FRAC = 0.7
OUT_DIR = "analysis_outputs/rebuttal_disentangle"

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


def _step_radius(meta: dict, fallback: float) -> float:
    value = meta.get("success_radius")
    try:
        radius = float(value)
    except (TypeError, ValueError):
        return fallback
    return radius if radius > 0.0 else fallback


def _distance_series(trace: list, fallback_radius: float, align: bool) -> tuple[list[float], list[bool]]:
    """逐步距离与「是否在半径内」标志。取不到距离的步按不在半径内处理。

    align=False：一律用 meta.goal_distance，即 evaluator 当时看到的量。用于验证门。
    align=True ：对 generic 类改用 meta.goal_distance_l2_m。

    为什么需要 align：评测器在 pi05_base/pi0_base 跑完之后升级过。新版把
    goal_distance 改成各向异性重标定后的「判决量」，并把原始欧氏距离另存为
    goal_distance_l2_m。已逐条核实：新版的 goal_distance_l2_m 恰等于旧版的
    goal_distance（同 task/level/seed，到浮点位）。move_to / square_completion 两个版本
    本来就不重标定，placement 两个版本都重标定，只有 generic 是新增的，因此只需对
    generic 回退。不这么做，新跑的 oracle 会在 24 条 generic 上被用更严的尺子量。
    """
    distances: list[float] = []
    flags: list[bool] = []
    for step in trace:
        meta = step.get("meta") or {}
        radius = _step_radius(meta, fallback_radius)
        raw = meta.get("goal_distance")
        if align and meta.get("goal_kind") == "generic" and meta.get("goal_distance_l2_m") is not None:
            raw = meta.get("goal_distance_l2_m")
        try:
            distance = float(raw)
        except (TypeError, ValueError):
            distances.append(float("nan"))
            flags.append(False)
            continue
        distances.append(distance)
        flags.append(radius > 0.0 and distance <= radius)
    return distances, flags


def _first_dwell_step(flags: list[bool], dwell: int) -> int | None:
    """首次凑满 dwell 个连续在半径内的步时，该段末步的下标。与 evaluators/simple.py 同语义。"""
    streak = 0
    for index, flag in enumerate(flags):
        streak = streak + 1 if flag else 0
        if streak >= dwell:
            return index
    return None


def episode_row(result_path: str, run: str) -> dict | None:
    match = PATH_RE.search(result_path)
    if match is None:
        return None
    try:
        result = json.load(open(result_path, encoding="utf-8"))
        trace = json.load(open(result_path.replace("result.json", "trace.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not trace:
        return None

    evaluation = result.get("evaluation") or {}
    radius = float(evaluation.get("success_radius") or 0.0)
    # native = evaluator 当时的尺子（过验证门）；aligned = 抹平评测器版本差异后的尺子（出结论）。
    distances_native, flags_native = _distance_series(trace, radius, align=False)
    distances, flags = _distance_series(trace, radius, align=True)
    finite = [d for d in distances if math.isfinite(d)]
    if not finite:
        return None
    realigned = any(
        (step.get("meta") or {}).get("goal_kind") == "generic"
        and (step.get("meta") or {}).get("goal_distance_l2_m") is not None
        for step in trace
    )

    task = match.group(1)
    max_steps = int(result.get("max_steps") or len(trace))
    ref_reach = bool(evaluation.get("reach_success"))
    ref_step = evaluation.get("reach_step")
    # LayoutAwareEvaluator 的 goal_kind/resolver/metric_valid 守卫会挡掉 reach_success，
    # 但 reach_step 照算不误，所以「守卫通过」等价于 reach_success == (reach_step is not None)。
    guard_ok = ref_reach == (ref_step is not None)

    row: dict = {
        "run": run,
        "task": task,
        "family": family_of(task),
        "level": match.group(2),
        "seed": match.group(3),
        "episode_dir": os.path.dirname(result_path),
        "n_steps": len(trace),
        "max_steps": max_steps,
        "native_object_pose_perturb_range": result.get("native_object_pose_perturb_range"),
        "goal_kind": (result.get("execution_binding") or {}).get("goal_kind") or "",
        "success_radius": radius,
        "d0": round(distances[0], 6) if math.isfinite(distances[0]) else "",
        "min_d": round(min(finite), 6),
        "min_d_native": round(min(d for d in distances_native if math.isfinite(d)), 6),
        "caliber_realigned": int(realigned),
        "argmin_d_step": int(min(range(len(distances)), key=lambda i: distances[i] if math.isfinite(distances[i]) else math.inf)),
        "start_inside": int(bool(flags[0])),
        "ref_success": int(bool(evaluation.get("success"))),
        "ref_reach_success": int(ref_reach),
        "ref_reach_step": ref_step if ref_step is not None else "",
        "ref_dwell_steps_required": evaluation.get("dwell_steps_required"),
        "ref_min_goal_distance": evaluation.get("min_goal_distance"),
        "ref_metric_valid": int(bool(evaluation.get("metric_valid"))),
        "guard_ok": int(guard_ok),
    }

    start_inside = bool(flags[0])
    for dwell in DWELL_LEVELS:
        step = _first_dwell_step(flags, dwell)
        reached = step is not None
        row[f"reach_k{dwell}"] = int(reached)
        row[f"reach_step_k{dwell}"] = step if reached else ""
        row[f"reach_k{dwell}_excl_start"] = int(reached and not start_inside)
        row[f"reach_k{dwell}_early"] = int(reached and step <= EARLY_STEP_MAX)

    step5 = _first_dwell_step(flags, 5)
    row["reach_step_frac_k5"] = round(step5 / max_steps, 4) if step5 is not None and max_steps else ""
    row["near_budget"] = int(step5 is not None and max_steps > 0 and step5 / max_steps >= NEAR_BUDGET_FRAC)

    # 验证门：K=1 的重算必须与 result.json 记录一致。守卫挡掉的 episode 只比对 reach_step。
    # 用 native 尺子，否则对齐后的尺子必然对不上新版评测器写入的 reach_success。
    step1 = _first_dwell_step(flags_native, 1)
    verify_reach = (step1 is not None) == ref_reach if guard_ok else True
    verify_step = (step1 if step1 is not None else None) == (ref_step if isinstance(ref_step, int) else None)
    row["verify_ok"] = int(verify_reach and verify_step)
    row["verify_note"] = "" if row["verify_ok"] else f"recomputed_k1_step={step1} ref_step={ref_step} ref_reach={ref_reach}"
    return row


def _rate(rows: list[dict], key: str) -> str:
    if not rows:
        return "-"
    hit = sum(r[key] for r in rows)
    return f"{hit}/{len(rows)} ({hit / len(rows) * 100:.1f}%)"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def main() -> int:
    rows: list[dict] = []
    for run, root in list(RUNS.items()) + list(IFACE_RUNS.items()):
        if not os.path.isdir(root):
            print(f"[跳过] {run}: 目录不存在 {root}")
            continue
        found = 0
        for path in sorted(glob.glob(f"{root}/runs/*/*/seed_*/episode_0000/result.json")):
            row = episode_row(path, run)
            if row is not None:
                rows.append(row)
                found += 1
        tag = "" if run in RUNS else "  [接口消融]"
        print(f"载入 {run:14s} {found:4d} 条  <- {root}{tag}")
    if not rows:
        print("没有可用 episode", file=sys.stderr)
        return 1
    print()

    # ---- 强制验证门 ----
    bad = [r for r in rows if not r["verify_ok"]]
    print("=" * 108)
    print("[验证门] K=1 重算 vs result.json 记录的 reach_success / reach_step")
    print("=" * 108)
    if bad:
        print(f"  ✗ 不一致 {len(bad)}/{len(rows)} 条。前 10 条：")
        for r in bad[:10]:
            print(f"    {r['run']:12s} {r['task']:<42s} {r['level']:<7s} seed={r['seed']}  {r['verify_note']}")
        print("\n  口径对不上，K=3/5 的结论不可信。中止。", file=sys.stderr)
        _write_audit(rows)
        return 2
    print(f"  ✓ 全部一致 ({len(rows)}/{len(rows)})")
    print(f"  dwell_steps_required 分布: "
          f"{dict((k, sum(1 for r in rows if r['ref_dwell_steps_required'] == k)) for k in sorted({r['ref_dwell_steps_required'] for r in rows}, key=str))}")
    print()

    _write_audit(rows)
    _write_summary(rows)
    _print_tables(rows)
    return 0


def _write_audit(rows: list[dict]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/reach_dwell_audit.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"逐 episode: {path}  ({len(rows)} 行)")


def _write_summary(rows: list[dict]) -> None:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["run"], "level", row["level"])].append(row)
        groups[(row["run"], "family", row["family"])].append(row)
        groups[(row["run"], "all", "all")].append(row)

    out: list[dict] = []
    for (run, kind, value), sub in sorted(groups.items()):
        steps = [r["reach_step_k5"] for r in sub if r["reach_step_k5"] != ""]
        record = {
            "run": run,
            "group_kind": kind,
            "group": value,
            "n": len(sub),
            "perturb_values": "|".join(sorted({str(r["native_object_pose_perturb_range"]) for r in sub})),
        }
        for dwell in DWELL_LEVELS:
            hit = sum(r[f"reach_k{dwell}"] for r in sub)
            record[f"reach_k{dwell}"] = hit
            record[f"reach_k{dwell}_rate"] = round(hit / len(sub), 4)
            excl = sum(r[f"reach_k{dwell}_excl_start"] for r in sub)
            record[f"reach_k{dwell}_excl_start"] = excl
            record[f"reach_k{dwell}_excl_start_rate"] = round(excl / len(sub), 4)
        record["reach_k1_early"] = sum(r["reach_k1_early"] for r in sub)
        record["start_inside"] = sum(r["start_inside"] for r in sub)
        record["start_inside_rate"] = round(sum(r["start_inside"] for r in sub) / len(sub), 4)
        record["near_budget"] = sum(r["near_budget"] for r in sub)
        record["reach_step_k5_median"] = round(_percentile(steps, 0.5), 2) if steps else ""
        record["reach_step_k5_p25"] = round(_percentile(steps, 0.25), 2) if steps else ""
        record["reach_step_k5_p75"] = round(_percentile(steps, 0.75), 2) if steps else ""
        record["min_d_median"] = round(_percentile([r["min_d"] for r in sub], 0.5), 5)
        out.append(record)

    path = f"{OUT_DIR}/reach_dwell_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print(f"汇总:       {path}  ({len(out)} 行)\n")


def _print_tables(rows: list[dict]) -> None:
    runs = [r for r in RUNS if any(x["run"] == r for x in rows)]

    print("=" * 108)
    print("[表 1] dwell 阈值的影响（分母为该 run 的全部 episode）")
    print("=" * 108)
    print(f"  {'run':<13s}{'n':>5s}{'reach@1':>18s}{'reach@3':>18s}{'reach@5':>18s}{'reach@5 排除起始态':>24s}")
    for run in runs:
        sub = [r for r in rows if r["run"] == run]
        print(f"  {run:<13s}{len(sub):>5d}"
              f"{_rate(sub, 'reach_k1'):>18s}{_rate(sub, 'reach_k3'):>18s}"
              f"{_rate(sub, 'reach_k5'):>18s}{_rate(sub, 'reach_k5_excl_start'):>22s}")
    print()

    print("=" * 108)
    print("[表 1b] 条件口径：只统计起手在半径外的 episode（起手即命中的从分母剔除）")
    print("=" * 108)
    print("  表 1 的 excl_start 把「起手即在半径内」当失败留在分母里，而各 run 的起手命中率差异悬殊，")
    print("  会系统性惩罚起手命中率高的 run（oracle 远高于 learned）。这里改为把这些 episode 剔出分母。")
    print(f"  {'run':<13s}{'子集 n':>9s}{'剔除数':>9s}{'reach@1':>18s}{'reach@3':>18s}{'reach@5':>18s}")
    for run in runs:
        sub = [r for r in rows if r["run"] == run]
        cond = [r for r in sub if not r["start_inside"]]
        if not cond:
            continue
        print(f"  {run:<13s}{len(cond):>9d}{len(sub) - len(cond):>9d}"
              f"{_rate(cond, 'reach_k1'):>18s}{_rate(cond, 'reach_k3'):>18s}{_rate(cond, 'reach_k5'):>18s}")
    print()

    print("=" * 108)
    print("[表 2] 起始态伪影与步数预算")
    print("=" * 108)
    print(f"  {'run':<13s}{'起手即在半径内':>18s}{'reach@1 首达<=2步':>20s}{'reach@5 步数中位':>20s}{'贴预算(>=70%)':>18s}")
    for run in runs:
        sub = [r for r in rows if r["run"] == run]
        early = [r for r in sub if r["reach_k1_early"]]
        steps = [r["reach_step_k5"] for r in sub if r["reach_step_k5"] != ""]
        median = f"{_percentile(steps, 0.5):.1f}" if steps else "-"
        early_cell = f"{len(early)}/{len(sub)}"
        budget_cell = f"{sum(r['near_budget'] for r in sub)}/{len(sub)}"
        print(f"  {run:<13s}{_rate(sub, 'start_inside'):>18s}"
              f"{early_cell:>20s}{median:>20s}{budget_cell:>18s}")
    print()

    print("=" * 108)
    print("[表 3] 逐族 reach@5（排除起始态）")
    print("=" * 108)
    header = f"  {'族':<24s}"
    for run in runs:
        header += f"{run:>20s}"
    print(header)
    for _, family in FAMILY_MAP:
        line = f"  {family:<24s}"
        for run in runs:
            sub = [r for r in rows if r["run"] == run and r["family"] == family]
            line += f"{_rate(sub, 'reach_k5_excl_start'):>20s}" if sub else f"{'-':>20s}"
        print(line)
    print()

    print("=" * 108)
    print("[表 4] 条件匹配检查：native_object_pose_perturb_range")
    print("=" * 108)
    for run in runs:
        sub = [r for r in rows if r["run"] == run]
        counts: dict = defaultdict(int)
        for r in sub:
            counts[str(r["native_object_pose_perturb_range"])] += 1
        print(f"  {run:<13s}{dict(sorted(counts.items()))}")
    print()

    _reproducibility_check(rows)

    print("=" * 108)
    print("[判定门] 在 reach@5 且排除起始态之后")
    print("=" * 108)
    scripted = [r for r in rows if r["run"] == ORACLE_RUN]
    base = [r for r in rows if r["run"] == LEARNED_RUN]
    if scripted and base:
        tasks = {r["task"] for r in scripted}
        base_matched = [r for r in base if r["task"] in tasks]
        s_rate = sum(r["reach_k5_excl_start"] for r in scripted) / len(scripted)
        b_rate = sum(r["reach_k5_excl_start"] for r in base) / len(base)
        print(f"  scripted (全部 {len(scripted)} 条, {len(tasks)} 个 task): {s_rate * 100:.1f}%")
        print(f"  pi05_base(全部 {len(base)} 条): {b_rate * 100:.1f}%")
        if base_matched:
            m_rate = sum(r["reach_k5_excl_start"] for r in base_matched) / len(base_matched)
            print(f"  pi05_base(限于 scripted 覆盖的同 {len(tasks)} 个 task, {len(base_matched)} 条): {m_rate * 100:.1f}%")
            print(f"\n  → 同 task 差距 {(s_rate - m_rate) * 100:+.1f} 个百分点")
            _paired_test(scripted, base_matched)
    else:
        print(f"  {ORACLE_RUN} 或 {LEARNED_RUN} 缺失，无法判定")
    print()

    _interface_sweep(rows)


def _interface_sweep(rows: list[dict]) -> None:
    """接口 / 部署旋钮各自能解释多少 reach 差异。

    每个变体只覆盖 2~27 个 task，绝不能拿它的整体率去和 27 task 的主配置比。
    这里一律先取共同 (task, level, seed) 三元组再配对，未配上的条数如实打印。
    """
    key = lambda r: (r["task"], r["level"], r["seed"])
    ref = {key(r): r for r in rows if r["run"] == IFACE_REF}
    print("=" * 108)
    print(f"[表 5] 接口 / 部署消融（同 checkpoint，逐 (task,level,seed) 配对 vs {IFACE_REF}）")
    print("=" * 108)
    if not ref:
        print(f"  参照 {IFACE_REF} 缺失")
        return
    print(f"  {'变体':<16s}{'task':>6s}{'配对n':>7s}{'变体 reach@5*':>16s}{'参照 reach@5*':>16s}"
          f"{'Δpp':>8s}{'b/c':>9s}{'p':>9s}")
    for name in IFACE_RUNS:
        sub = {key(r): r for r in rows if r["run"] == name}
        if not sub:
            continue
        shared = sorted(set(sub) & set(ref))
        if not shared:
            print(f"  {name:<16s}{len({k[0] for k in sub}):>6d}{0:>7d}   (与参照无重合 task/level/seed)")
            continue
        v_hit = sum(sub[k]["reach_k5_excl_start"] for k in shared)
        r_hit = sum(ref[k]["reach_k5_excl_start"] for k in shared)
        b = sum(1 for k in shared if sub[k]["reach_k5_excl_start"] and not ref[k]["reach_k5_excl_start"])
        c = sum(1 for k in shared if ref[k]["reach_k5_excl_start"] and not sub[k]["reach_k5_excl_start"])
        n = len(shared)
        delta = (v_hit - r_hit) / n * 100
        print(f"  {name:<16s}{len({k[0] for k in shared}):>6d}{n:>7d}"
              f"{f'{v_hit}/{n} ({v_hit / n * 100:.1f}%)':>16s}{f'{r_hit}/{n} ({r_hit / n * 100:.1f}%)':>16s}"
              f"{delta:>+8.1f}{f'{b}/{c}':>9s}{mcnemar_exact(b, c):>9.4f}")
    print("  * reach@5 排除起始态。b = 仅变体达成，c = 仅参照达成。")
    print()


def _reproducibility_check(rows: list[dict]) -> None:
    """新旧两批 scripted 在重合的 task/level/seed 上必须给出相同判定，否则两批不同源。"""
    key = lambda r: (r["task"], r["level"], r["seed"])
    old = {key(r): r for r in rows if r["run"] == "scripted"}
    new = {key(r): r for r in rows if r["run"] == "scripted_full"}
    shared = sorted(set(old) & set(new))
    print("=" * 108)
    print("[复现性] 新旧 scripted 在重合 episode 上的一致性")
    print("=" * 108)
    if not shared:
        print(f"  重合 0 条（旧 {len(old)} / 新 {len(new)}），无法交叉验证")
        print()
        return
    diff_reach = [k for k in shared if old[k]["reach_k5"] != new[k]["reach_k5"]]
    diff_min = [(k, abs(old[k]["min_d"] - new[k]["min_d"])) for k in shared]
    worst = max(diff_min, key=lambda x: x[1])
    print(f"  重合 {len(shared)} 条（旧 {len(old)} / 新 {len(new)}）")
    print(f"  reach@5 判定不一致: {len(diff_reach)} 条")
    print(f"  min_d 最大偏差: {worst[1]:.6f} m  @ {worst[0]}")
    for k in diff_reach[:5]:
        print(f"    {k}  旧 reach@5={old[k]['reach_k5']} min_d={old[k]['min_d']}  "
              f"新 reach@5={new[k]['reach_k5']} min_d={new[k]['min_d']}")
    print()


def _paired_test(oracle: list[dict], learned: list[dict]) -> None:
    """按 (task, level, seed) 配对做精确 McNemar。配不上的 episode 直接报出来，不静默丢弃。

    两个口径都报：
    - 全量：用 reach_k5_excl_start（起手即命中 = 0），对 oracle 偏保守
    - 条件：只留双方起手都在半径外的配对，用 reach_k5，这才是无偏的对比
    """
    key = lambda r: (r["task"], r["level"], r["seed"])
    left = {key(r): r for r in oracle}
    right = {key(r): r for r in learned}
    shared = sorted(set(left) & set(right))
    if not shared:
        print("  (无法配对：task/level/seed 不重合)")
        return
    b = sum(1 for k in shared if left[k]["reach_k5_excl_start"] and not right[k]["reach_k5_excl_start"])
    c = sum(1 for k in shared if right[k]["reach_k5_excl_start"] and not left[k]["reach_k5_excl_start"])
    print(f"  配对 n={len(shared)}（oracle 未配上 {len(set(left) - set(right))}，learned 未配上 {len(set(right) - set(left))}）")
    print(f"  [全量口径] 仅 oracle 达成 b={b}，仅 learned 达成 c={c}，精确 McNemar p={mcnemar_exact(b, c):.4f}")

    cond = [k for k in shared if not left[k]["start_inside"] and not right[k]["start_inside"]]
    if not cond:
        return
    ob = sum(left[k]["reach_k5"] for k in cond)
    lb = sum(right[k]["reach_k5"] for k in cond)
    b2 = sum(1 for k in cond if left[k]["reach_k5"] and not right[k]["reach_k5"])
    c2 = sum(1 for k in cond if right[k]["reach_k5"] and not left[k]["reach_k5"])
    n = len(cond)
    print(f"  [条件口径] 双方起手都在半径外的配对 n={n}（剔除 {len(shared) - n} 对）")
    print(f"             oracle {ob}/{n} ({ob / n * 100:.1f}%)  vs  learned {lb}/{n} ({lb / n * 100:.1f}%)"
          f"  Δ{(ob - lb) / n * 100:+.1f} pp")
    print(f"             b={b2}, c={c2}, 精确 McNemar p={mcnemar_exact(b2, c2):.4f}")


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确 McNemar：n=b+c 次投币里出现 >=max(b,c) 次同一面的概率。"""
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2.0 * tail / (2**n))


if __name__ == "__main__":
    raise SystemExit(main())
