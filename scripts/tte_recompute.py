"""TTE 重算：四个版本 × {3d, xy}。目标点 g = eef + goal_error（环境真值 goal，与 SGE/reach 同源）。"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

RUNS = {
    "openvla": ("run_eval_openvla", "base"),
    "internvla": ("run_eval_internvla", "base"),
    "openpi_0_old": ("run_eval_openpi_0", "base"),
    "spatialvla": ("run_eval_spatialvla", "base"),
    "spatialforcing": ("run_eval_spatialforcing", "base"),
    "pi0_v3": ("run_eval_openpi_05_v3", "base"),
    "pi05_v4": ("run_eval_pi05_v4", "base"),
    "pi05_v4_direct": ("run_eval_pi05_v4_direct", "direct"),
}
PATH_RE = re.compile(r"/runs/([^/]+)/([^/]+)/seed_(\d+)/")
RESAMPLE_N = 50
FAMILY_MAP = [
    ("Binding", "Target Identification"),
    ("Relation", "Relational Grounding"),
    ("Constraint", "Constraint Composition"),
    ("Geometric", "Geometric Inference"),
    ("Reindex", "Scene Updating"),
    ("Track", "Identity Tracking"),
]


def family_of(task: str) -> str:
    for pre, name in FAMILY_MAP:
        if task.startswith(pre):
            return name
    return "Unknown"


def probe() -> None:
    print("=" * 78)
    print("[probe] goal_error 维度与跨 run 可用性")
    print("=" * 78)
    for model, (root, _) in RUNS.items():
        ps = sorted(glob.glob(f"{root}/runs/*/*/seed_*/episode_0000/trace.json"))
        if not ps:
            print(f"  {model:16s} 无 trace")
            continue
        tr = json.load(open(ps[0]))
        pol = tr[0]["agent_obs"]["state"]["native_obs"]["policy"]
        ge = pol.get("goal_error")
        n_ok = 0
        for p in ps[:60]:
            try:
                t = json.load(open(p))
                if "goal_error" in t[0]["agent_obs"]["state"]["native_obs"]["policy"]:
                    n_ok += 1
            except Exception:
                pass
        print(
            f"  {model:16s} n={len(ps):4d}  goal_error={'len=' + str(len(ge)) if isinstance(ge, list) else ge!r}"
            f"   前60条可用 {n_ok}/60"
        )


def resample(d: np.ndarray, n: int = RESAMPLE_N) -> np.ndarray:
    if d.size == 1:
        return np.repeat(d, n)
    t_src = np.linspace(0.0, 1.0, d.size)
    t_dst = np.linspace(0.0, 1.0, n)
    return np.interp(t_dst, t_src, d)


def episode_row(path: str, model: str, variant: str) -> dict | None:
    m = PATH_RE.search(path)
    if m is None:
        return None
    d = json.load(open(path))
    try:
        trace = json.load(open(path.replace("result.json", "trace.json")))
    except Exception:
        return None
    if not trace:
        return None
    # d_xy 直接用环境真值 goal_error（xy-only，全 goal_kind 通用）。
    # d_3d 额外加上 z 分量：g_z 取离 goal 点最近物体的中心 z；物体位姿不可得时为 NaN。
    # 同时按论文 metrics_trajectory.py 的口径算 target_closest_rate（逐步平均）与末步版。
    n_obj = int((d.get("execution_binding") or {}).get("runtime_object_count") or 0)
    is_move_to = (d.get("execution_binding") or {}).get("goal_kind") == "move_to" and n_obj >= 2
    dxy_l, dz_l, hits = [], [], []
    for s in trace:
        p = s.get("agent_obs", {}).get("state", {}).get("native_obs", {}).get("policy", {})
        g = p.get("goal_error")
        eef = p.get("eef_pos")
        if not isinstance(g, list) or len(g) < 2 or not isinstance(eef, list) or len(eef) < 3:
            return None
        gx, gy = float(g[0]), float(g[1])
        dxy_l.append(math.hypot(gx, gy))
        raw = p.get("object")
        gz = float("nan")
        if isinstance(raw, list) and len(raw) >= 7:
            v = np.asarray(raw, float)
            P = np.array([v[i * 7 : i * 7 + 3] for i in range(len(v) // 7)])
            goal_xy = np.array([float(eef[0]) + gx, float(eef[1]) + gy])
            real = P[P[:, 2] > 0.5]  # 剔除填充槽位
            if real.size:
                gz = float(real[int(np.argmin(np.linalg.norm(real[:, :2] - goal_xy, axis=1))), 2])
            if is_move_to and len(P) >= n_obj:
                Q = P[:n_obj, :2]
                gt_i = int(np.argmin(np.linalg.norm(Q - goal_xy, axis=1)))
                pr_i = int(np.argmin(np.linalg.norm(Q - np.asarray(eef[:2], float), axis=1)))
                hits.append(int(gt_i == pr_i))
        dz_l.append(float(eef[2]) - gz)
    dxy = np.asarray(dxy_l, float)
    d3 = np.sqrt(dxy**2 + np.asarray(dz_l, float) ** 2)
    T = len(dxy)
    k = max(1, int(math.ceil(T * 0.2)))
    ev = d.get("evaluation") or {}
    eb = d.get("execution_binding") or {}
    task = m.group(1)
    out = {
        "episode_id": f"{model}|{task}|{m.group(2)}|seed_{m.group(3)}",
        "model": model,
        "variant": variant,
        "task": task,
        "family": family_of(task),
        "level": m.group(2),
        "seed": m.group(3),
        "goal_kind": eb.get("goal_kind") or "",
        "n_steps": T,
        "success": int(bool(ev.get("success"))),
        "reach_success": int(bool(ev.get("reach_success"))),
        "n_obj": n_obj,
        "target_closest_rate": float(np.mean(hits)) if len(hits) == T and T > 0 else float("nan"),
        "target_closest_final": float(hits[-1]) if len(hits) == T and T > 0 else float("nan"),
    }
    for suf, arr in (("3d", d3), ("xy", dxy)):
        if not np.isfinite(arr).all():
            for v in ("mean_raw", "mean_resampled", "final20", "min"):
                out[f"tte_{v}_{suf}"] = float("nan")
            continue
        out[f"tte_mean_raw_{suf}"] = float(arr.mean())
        out[f"tte_mean_resampled_{suf}"] = float(resample(arr).mean())
        out[f"tte_final20_{suf}"] = float(arr[-k:].mean())
        out[f"tte_min_{suf}"] = float(arr.min())
    return out


def sign_flip(diff: np.ndarray, iters: int = 20000, seed: int = 0) -> float:
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return 1.0
    obs = abs(diff.mean())
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1.0, 1.0], size=(iters, diff.size)) * diff).mean(axis=1))
    return float((null >= obs - 1e-12).mean())


VARIANTS = [
    f"tte_{v}_{s}" for v in ("mean_raw", "mean_resampled", "final20", "min") for s in ("3d", "xy")
] + ["target_closest_rate", "target_closest_final"]


def quant(rows: list[dict], col: str) -> str:
    v = np.array([r[col] for r in rows], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n=0（该版本不可得）"
    return (
        f"n={v.size:4d}  均值={v.mean():.4f}  中位={np.median(v):.4f}  "
        f"p25={np.percentile(v, 25):.4f}  p75={np.percentile(v, 75):.4f}"
    )


def main() -> None:
    probe()
    all_rows: list[dict] = []
    for model, (root, variant) in RUNS.items():
        got = 0
        for p in sorted(glob.glob(f"{root}/runs/*/*/seed_*/episode_0000/result.json")):
            r = episode_row(p, model, variant)
            if r is not None:
                all_rows.append(r)
                got += 1
        print(f"  载入 {model:16s} {got} 条")

    out_csv = "analysis_outputs/tte_recompute_episodes.csv"
    os.makedirs("analysis_outputs", exist_ok=True)
    cols = [
        "episode_id", "variant", "family", "task", "goal_kind", "model", "n_steps",
        "tte_mean_raw_3d", "tte_mean_raw_xy", "tte_mean_resampled_3d", "tte_mean_resampled_xy",
        "tte_final20_3d", "tte_final20_xy", "tte_min_3d", "tte_min_xy",
        "target_closest_rate", "target_closest_final", "n_obj", "success", "reach_success",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n逐 episode 明细已写出: {out_csv}  ({len(all_rows)} 行)")

    by_model = defaultdict(list)
    for r in all_rows:
        by_model[r["model"]].append(r)

    print("\n" + "=" * 78)
    print("[汇总 1] 各 run × 各版本（Track 族已剔除，单独见汇总 3）")
    print("=" * 78)
    for model in RUNS:
        rows = [r for r in by_model.get(model, []) if r["family"] != "Identity Tracking"]
        if not rows:
            continue
        print(f"\n  {model}   步数分布 {dict(sorted(Counter(r['n_steps'] for r in rows).items()))}")
        for c in VARIANTS:
            print(f"    {c:26s} {quant(rows, c)}")

    print("\n" + "=" * 78)
    print("[汇总 2] 成功 / 失败分报（pi05_v4，剔除 Track）")
    print("=" * 78)
    rows = [r for r in by_model.get("pi05_v4", []) if r["family"] != "Identity Tracking"]
    for lab, sub in (("reach 成功", [r for r in rows if r["reach_success"]]),
                     ("reach 失败", [r for r in rows if not r["reach_success"]])):
        print(f"  {lab} n={len(sub)}")
        for c in VARIANTS:
            print(f"    {c:26s} {quant(sub, c)}")

    print("\n" + "=" * 78)
    print("[汇总 3] Track (Identity Tracking) 族单列")
    print("=" * 78)
    for model in RUNS:
        rows = [r for r in by_model.get(model, []) if r["family"] == "Identity Tracking"]
        if not rows:
            continue
        print(f"  {model:16s} n={len(rows):4d}  "
              f"mean_raw_xy={np.mean([r['tte_mean_raw_xy'] for r in rows]):.4f}  "
              f"final20_xy={np.mean([r['tte_final20_xy'] for r in rows]):.4f}  "
              f"min_xy={np.mean([r['tte_min_xy'] for r in rows]):.4f}")

    print("\n" + "=" * 78)
    print("[检验] base vs direct 配对 sign-flip permutation (20000)，全部版本都报")
    print("=" * 78)
    B = {(r["task"], r["seed"]): r for r in by_model.get("pi05_v4", [])}
    D = {(r["task"], r["seed"]): r for r in by_model.get("pi05_v4_direct", [])}
    for scope, keep in (("全部任务", lambda r: True),
                        ("剔除 Track", lambda r: r["family"] != "Identity Tracking"),
                        ("仅 Track", lambda r: r["family"] == "Identity Tracking")):
        sh = sorted(k for k in set(B) & set(D) if keep(B[k]))
        print(f"\n  {scope}  配对 n={len(sh)}")
        for c in VARIANTS:
            a = np.array([B[k][c] for k in sh], float)
            b = np.array([D[k][c] for k in sh], float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            if a.size == 0:
                print(f"    {c:26s} 不可得")
                continue
            p = sign_flip(b - a)
            print(f"    {c:26s} n={a.size:3d}  base={a.mean():.4f}  direct={b.mean():.4f}  "
                  f"Δ={b.mean() - a.mean():+.4f}  p={p:.4f}{'  ★' if p < 0.05 else ''}")

    print("\n" + "=" * 78)
    print("[要保住的结论] tte_final20 > tte_mean_raw（轨迹后段反而更远）")
    print("=" * 78)
    for model in RUNS:
        rows = by_model.get(model, [])
        if not rows:
            continue
        line = f"  {model:16s}"
        for suf in ("3d", "xy"):
            sub = [r for r in rows if np.isfinite(r[f"tte_final20_{suf}"]) and np.isfinite(r[f"tte_mean_raw_{suf}"])]
            if not sub:
                line += f"   {suf}: 不可得"
                continue
            frac = np.mean([r[f"tte_final20_{suf}"] > r[f"tte_mean_raw_{suf}"] for r in sub])
            dm = np.mean([r[f"tte_final20_{suf}"] - r[f"tte_mean_raw_{suf}"] for r in sub])
            line += f"   {suf}(n={len(sub):4d}): 占比 {100 * frac:5.1f}%  平均差 {dm:+.4f}"
        print(line)

    print("\n  按任务族（pi05_v4，xy 版本）:")
    for fam in sorted({r["family"] for r in by_model.get("pi05_v4", [])}):
        rows = [r for r in by_model["pi05_v4"] if r["family"] == fam]
        frac = np.mean([r["tte_final20_xy"] > r["tte_mean_raw_xy"] for r in rows])
        print(f"    {fam:24s} n={len(rows):3d}  占比 {100 * frac:5.1f}%")


if __name__ == "__main__":
    main()
