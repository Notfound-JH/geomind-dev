"""统一重算 rebuttal 所需的全部指标。

一个入口、一套口径，避免各指标来自不同临时脚本。

指标定义
--------
二元
  success        : result.json evaluation.success（夹爪锁定下几乎恒为 0，仅作说明）
  reach          : evaluation.reach_success（主指标）
  tte            : 末端最终最近的物体 == GT 目标物体；仅 goal_kind == move_to
连续（越小越好）
  sge_xy_final   : 末步 |goal_error[:2]|
  sge_xy_best    : 全回合最小 |goal_error[:2]|
  sge_3d_final   : evaluation.final_goal_distance（含 z，受悬停高度污染，仅作对照）
  min_goal_dist  : evaluation.min_goal_distance
诊断
  cos_goal       : 净位移方向 vs 初始目标方向的余弦
  net_dir        : 净位移单位向量（用于先验坍缩检验）

用法
  python scripts/rebuttal_metrics.py
"""

from __future__ import annotations

import glob
import json
import math
import re
from collections import Counter, defaultdict

import numpy as np

RUNS = {
    "pi05_base": "run_eval_pi05_v4",
    "pi05_direct": "run_eval_pi05_v4_direct",
    "pi0_base": "run_eval_openpi_05_v3",
    # GT goal 向量 + 定幅直线控制器（scripted_goal_policy:policy）。参数与 pi05_base 逐字一致，
    # 只换策略，因此它给出的是同一套环境/接口/步数预算下的控制上界。
    "scripted": "run_eval_scripted_full",
}
PATH_RE = re.compile(r"/runs/([^/]+)/([^/]+)/seed_(\d+)/")


def _first_not_none(mapping: dict, *keys: str):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _episode(result_path: str) -> dict | None:
    d = json.load(open(result_path))
    m = PATH_RE.search(result_path)
    if m is None:
        return None
    ev = d.get("evaluation") or {}
    eb = d.get("execution_binding") or {}
    da = d.get("direct_ablation") or {}
    trace = json.load(open(result_path.replace("result.json", "trace.json")))
    if not trace:
        return None

    def pol(step):
        return step["agent_obs"]["state"]["native_obs"]["policy"]

    ge = np.array([np.asarray(pol(s)["goal_error"][:2], float) for s in trace])
    sge_xy = np.linalg.norm(ge, axis=1)
    eef = np.array([np.asarray(pol(s)["eef_pos"][:3], float) for s in trace])
    net = eef[-1, :2] - eef[0, :2]
    goal_dir = ge[0]
    denom = np.linalg.norm(net) * np.linalg.norm(goal_dir)
    cos_goal = float(net @ goal_dir / denom) if denom > 1e-9 else float("nan")
    net_dir = net / np.linalg.norm(net) if np.linalg.norm(net) > 1e-9 else np.array([np.nan, np.nan])

    # TTE：仅当目标点与某个物体重合（move_to）时有定义
    tte = None
    n_obj = int(eb.get("runtime_object_count") or 0)
    if eb.get("goal_kind") == "move_to" and n_obj >= 2:
        v = np.asarray(pol(trace[-1])["object"], float)
        P = np.array([v[i * 7 : i * 7 + 2] for i in range(n_obj)])
        goal = eef[-1, :2] + ge[-1]
        tte = int(np.argmin(np.linalg.norm(P - eef[-1, :2], axis=1)) == np.argmin(np.linalg.norm(P - goal, axis=1)))

    return {
        "task": m.group(1),
        "level": m.group(2),
        "seed": m.group(3),
        "success": int(bool(ev.get("success"))),
        "reach": int(bool(ev.get("reach_success"))),
        "tte": tte,
        "n_obj": n_obj,
        "goal_kind": eb.get("goal_kind"),
        "sge_xy_final": float(sge_xy[-1]),
        "sge_xy_best": float(sge_xy.min()),
        # *_l2_m are the plain euclidean metres; goal_distance itself is rescaled for anisotropic
        # goals and is therefore a verdict, not a length.
        "sge_3d_final": _first_not_none(ev, "final_goal_distance_l2_m", "final_goal_distance"),
        "min_goal_dist": _first_not_none(ev, "min_goal_distance_l2_m", "min_goal_distance"),
        "success_radius": ev.get("success_radius"),
        "reach_step": ev.get("reach_step"),
        "cos_goal": cos_goal,
        "net_dir": net_dir,
        "family": m.group(1).split("Base")[0].split("Aug")[0],
        "mechanism": da.get("mechanism", "none"),
        "tier": da.get("gradient_tier", "unchanged"),
    }


def load(root: str) -> dict:
    out = {}
    for p in sorted(glob.glob(f"{root}/runs/*/*/seed_*/episode_0000/result.json")):
        e = _episode(p)
        if e is not None:
            out[(e["task"], e["seed"])] = e
    return out


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """b 相对 a 的配对翻转检验（双侧精确二项）。"""
    win = int(((b == 1) & (a == 0)).sum())
    loss = int(((a == 1) & (b == 0)).sum())
    n = win + loss
    if n == 0:
        return win, loss, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(max(win, loss), n + 1)) / 2**n
    return win, loss, min(p, 1.0)


def sign_flip(diff: np.ndarray, iters: int = 20000, seed: int = 0) -> float:
    """配对连续量的符号翻转置换检验（双侧）。"""
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return 1.0
    obs = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(iters, diff.size))
    null = np.abs((signs * diff).mean(axis=1))
    return float((null >= obs - 1e-12).mean())


def poisson_binomial_pvalue(probs: list[float], observed: int) -> tuple[float, float]:
    """逐场景随机基线下 P(命中 >= observed) 的精确值。"""
    dist = np.array([1.0])
    for q in probs:
        nd = np.zeros(len(dist) + 1)
        nd[:-1] += dist * (1 - q)
        nd[1:] += dist * q
        dist = nd
    return float(sum(probs)), float(dist[observed:].sum())


def _fmt_binary(name: str, a: np.ndarray, b: np.ndarray, la: str, lb: str) -> str:
    w, l, p = mcnemar(a, b)
    star = " ★" if p < 0.05 else ""
    return (
        f"  {name:16s} {la} {100 * a.mean():5.1f}%  {lb} {100 * b.mean():5.1f}%"
        f"  Δ{100 * (b.mean() - a.mean()):+5.1f}   翻转 {w}/{l}   p={p:.4f}{star}"
    )


def _fmt_cont(name: str, a: np.ndarray, b: np.ndarray, la: str, lb: str) -> str:
    d = b - a
    p = sign_flip(d)
    star = " ★" if p < 0.05 else ""
    return (
        f"  {name:16s} {la} {a.mean():.4f}±{a.std(ddof=1):.4f}  {lb} {b.mean():.4f}±{b.std(ddof=1):.4f}"
        f"  Δ{d.mean():+.4f}   p={p:.4f}{star}"
    )


def compare(A: dict, B: dict, la: str, lb: str, title: str) -> None:
    shared = sorted(set(A) & set(B))
    print(f"\n{'=' * 78}\n{title}   配对 n={len(shared)}\n{'=' * 78}")

    print("[二元指标]")
    for key in ("success", "reach"):
        a = np.array([A[k][key] for k in shared])
        b = np.array([B[k][key] for k in shared])
        print(_fmt_binary(key, a, b, la, lb))

    tte_keys = [k for k in shared if A[k]["tte"] is not None and B[k]["tte"] is not None]
    if tte_keys:
        a = np.array([A[k]["tte"] for k in tte_keys])
        b = np.array([B[k]["tte"] for k in tte_keys])
        print(_fmt_binary(f"tte (n={len(tte_keys)})", a, b, la, lb))
        probs = [1 / A[k]["n_obj"] for k in tte_keys]
        exp, _ = poisson_binomial_pvalue(probs, 0)
        print(f"    随机基线 {100 * exp / len(tte_keys):.1f}%（逐场景 1/物体数，泊松二项）")
        for lab, arr in ((la, a), (lb, b)):
            _, pv = poisson_binomial_pvalue(probs, int(arr.sum()))
            tag = "★显著优于随机" if pv < 0.05 else "与随机无显著差异"
            print(f"      {lab:8s} {int(arr.sum())}/{len(tte_keys)}  P(随机≥观测)={pv:.4f}  {tag}")

    print("[连续指标]")
    for key in ("sge_xy_best", "sge_xy_final", "min_goal_dist", "sge_3d_final"):
        a = np.array([A[k][key] for k in shared], float)
        b = np.array([B[k][key] for k in shared], float)
        print(_fmt_cont(key, a, b, la, lb))


def prior_collapse(R: dict, label: str) -> None:
    by_task = defaultdict(list)
    for e in R.values():
        if np.all(np.isfinite(e["net_dir"])):
            by_task[e["task"]].append(e["net_dir"])
    means = {t: np.mean(v, axis=0) / (np.linalg.norm(np.mean(v, axis=0)) + 1e-12) for t, v in by_task.items()}
    g = np.mean(list(means.values()), axis=0)
    g = g / (np.linalg.norm(g) + 1e-12)
    cos = np.array([float(m @ g) for m in means.values()])
    print(
        f"  {label:14s} 任务数 {len(means):2d}  与全局平均方向 cos 中位 {np.median(cos):+.3f}"
        f"  最小 {cos.min():+.3f}  偏离(<0.9) {int((cos < 0.9).sum())}/{len(cos)}"
        f"  全局方向 ({g[0]:+.3f},{g[1]:+.3f})"
    )


def main() -> None:
    R = {k: load(v) for k, v in RUNS.items()}
    for k, v in R.items():
        print(f"{k:12s} {RUNS[k]:28s} episode {len(v)}")

    compare(R["pi05_base"], R["pi05_direct"], "base", "direct", "消融 base → direct（π0.5 + 镜像修正）")
    compare(R["pi0_base"], R["pi05_base"], "pi0 ", "pi05", "模型 π0 → π0.5+镜像（均为 base）")
    if R.get("scripted"):
        compare(R["pi05_base"], R["scripted"], "pi05", "oracle", "控制上界 π0.5 base → oracle 直线控制器")

    print(f"\n{'=' * 78}\n先验坍缩检验（各任务净移动方向是否被同一个固定方向支配）\n{'=' * 78}")
    for k in ("pi0_base", "pi05_base", "pi05_direct"):
        prior_collapse(R[k], k)

    print(f"\n{'=' * 78}\nbase → direct 逐任务族 reach / tte\n{'=' * 78}")
    shared = sorted(set(R["pi05_base"]) & set(R["pi05_direct"]))
    agg = defaultdict(lambda: Counter())
    for k in shared:
        f = R["pi05_base"][k]["family"]
        agg[f]["n"] += 1
        agg[f]["rb"] += R["pi05_base"][k]["reach"]
        agg[f]["rd"] += R["pi05_direct"][k]["reach"]
        if R["pi05_base"][k]["tte"] is not None:
            agg[f]["tn"] += 1
            agg[f]["tb"] += R["pi05_base"][k]["tte"]
            agg[f]["td"] += R["pi05_direct"][k]["tte"]
    print(f"  {'family':12s} {'n':>3s}  {'reach base→direct':>22s}   {'tte base→direct':>22s}")
    for f, c in sorted(agg.items()):
        tt = (
            f"{100 * c['tb'] / c['tn']:5.1f}% → {100 * c['td'] / c['tn']:5.1f}%  (n={c['tn']})"
            if c["tn"]
            else "        n/a          "
        )
        print(
            f"  {f:12s} {c['n']:3d}  {100 * c['rb'] / c['n']:5.1f}% → {100 * c['rd'] / c['n']:5.1f}%"
            f"  (n={c['n']})   {tt}"
        )


if __name__ == "__main__":
    main()
