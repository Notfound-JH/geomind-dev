"""pi0.5 base 与 direct 消融：逐族重算 SGE 与 TTE。

口径遵循 paper/rebuttal_metrics.md §1.3——所有距离都用环境自己的真值 goal
``g_t = eef_t + goal_error_t``，SGE 与 TTE 同源。这样可以绕开
``scripts/metrics_trajectory.py`` 的名义布局坐标 + 固定桌面 z（DEFAULT_TABLETOP_Z_M
= 0.784，实测中位 0.9543）带来的 3~4 倍膨胀。

SGE（Spatial Goal Error，单位 m）
  sge_xy_best  = min_t ||goal_error_t||_xy   主指标（与 tte_min_xy 同值，同源不同名）
  sge_xy_final = ||goal_error_T||_xy         末步版，已被 §1.2 弃用（受到达后下压污染）
  sge_xy_mean  = mean_t ||goal_error_t||_xy
  *_3d 为含 z 的对照，g_z 取离 goal 点最近的真实物体中心 z；§1.2 已弃用。

TTE（Trajectory Tracking Error，单位 m）四个版本 × {xy, 3d}
  mean_raw / mean_resampled(50 点) / final20(末 20% 步) / min

TTE-pre（截断版，主推）
  只在 t ∈ [0, t_best] 上积分，t_best = argmin_t ||goal_error_t||_xy 即 sge_best 的时刻。
  全程版把「到达后被 hold_initial 夹爪带着漂走 / 下压」的那段也算进去了，度量的是
  「趋近 + 离开」的混合物；截断版只度量趋近过程本身，语义干净。
  副产品 t_best_frac = t_best / (T-1) 说明最近点出现在轨迹的什么位置。
  注意截断段长度因 episode 而异，所以这里 mean_resampled 不再等同于 mean_raw。

用法
  /home/elm/miniconda3/envs/isaac_curobo/bin/python scripts/pi05_direct_family_sge_tte.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

RUNS = {
    "base": "run_eval_pi05_v4",
    "direct": "run_eval_pi05_v4_direct",
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
SGE_COLS = ["sge_xy_best", "sge_xy_final", "sge_xy_mean", "sge_3d_best", "sge_3d_final"]
TTE_COLS = [
    f"tte_{v}_{s}"
    for v in ("mean_raw", "mean_resampled", "final20", "min")
    for s in ("xy", "3d")
]
PRE_COLS = [
    f"tte_pre_{v}_{s}"
    for v in ("mean_raw", "mean_resampled", "final20")
    for s in ("xy", "3d")
] + ["t_best", "t_best_frac", "n_steps_pre"]


def family_of(task: str) -> str:
    for prefix, name in FAMILY_MAP:
        if task.startswith(prefix):
            return name
    return "Unknown"


def resample(d: np.ndarray, n: int = RESAMPLE_N) -> np.ndarray:
    if d.size == 1:
        return np.repeat(d, n)
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, d.size), d)


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

    dxy_list: list[float] = []
    dz_list: list[float] = []
    for step in trace:
        pol = step.get("agent_obs", {}).get("state", {}).get("native_obs", {}).get("policy", {})
        goal_error = pol.get("goal_error")
        eef = pol.get("eef_pos")
        if not isinstance(goal_error, list) or len(goal_error) < 2:
            return None
        if not isinstance(eef, list) or len(eef) < 3:
            return None
        gx, gy = float(goal_error[0]), float(goal_error[1])
        dxy_list.append(math.hypot(gx, gy))

        # g_z 取离 goal 点最近的真实物体中心 z；z<=0.5 的槽位是填充，剔除。
        gz = float("nan")
        raw = pol.get("object")
        if isinstance(raw, list) and len(raw) >= 7:
            flat = np.asarray(raw, float)
            poses = np.array([flat[i * 7 : i * 7 + 3] for i in range(len(flat) // 7)])
            real = poses[poses[:, 2] > 0.5]
            if real.size:
                goal_xy = np.array([float(eef[0]) + gx, float(eef[1]) + gy])
                nearest = int(np.argmin(np.linalg.norm(real[:, :2] - goal_xy, axis=1)))
                gz = float(real[nearest, 2])
        dz_list.append(float(eef[2]) - gz)

    dxy = np.asarray(dxy_list, float)
    d3d = np.sqrt(dxy**2 + np.asarray(dz_list, float) ** 2)
    n_steps = dxy.size
    tail = max(1, int(math.ceil(n_steps * 0.2)))

    evaluation = result.get("evaluation") or {}
    binding = result.get("execution_binding") or {}
    task = m.group(1)
    row: dict = {
        "episode_id": f"{variant}|{task}|{m.group(2)}|seed_{m.group(3)}",
        "variant": variant,
        "family": family_of(task),
        "task": task,
        "level": m.group(2),
        "seed": m.group(3),
        "goal_kind": binding.get("goal_kind") or "",
        "n_steps": n_steps,
        "success": int(bool(evaluation.get("success"))),
        "reach_success": int(bool(evaluation.get("reach_success"))),
        "sge_xy_best": float(dxy.min()),
        "sge_xy_final": float(dxy[-1]),
        "sge_xy_mean": float(dxy.mean()),
    }
    if np.isfinite(d3d).all():
        row["sge_3d_best"] = float(d3d.min())
        row["sge_3d_final"] = float(d3d[-1])
    else:
        row["sge_3d_best"] = float("nan")
        row["sge_3d_final"] = float("nan")

    for suffix, arr in (("xy", dxy), ("3d", d3d)):
        if not np.isfinite(arr).all():
            for name in ("mean_raw", "mean_resampled", "final20", "min"):
                row[f"tte_{name}_{suffix}"] = float("nan")
            continue
        row[f"tte_mean_raw_{suffix}"] = float(arr.mean())
        row[f"tte_mean_resampled_{suffix}"] = float(resample(arr).mean())
        row[f"tte_final20_{suffix}"] = float(arr[-tail:].mean())
        row[f"tte_min_{suffix}"] = float(arr.min())

    # 截断版：只保留到最接近目标的那一步为止，之后的漂移不计入。切点统一用 xy 的 argmin，
    # 这样 xy 与 3d 版本描述的是同一段轨迹。
    t_best = int(np.argmin(dxy))
    row["t_best"] = t_best
    row["t_best_frac"] = float(t_best / (n_steps - 1)) if n_steps > 1 else 0.0
    row["n_steps_pre"] = t_best + 1
    for suffix, arr in (("xy", dxy), ("3d", d3d)):
        segment = arr[: t_best + 1]
        if not np.isfinite(segment).all():
            for name in ("mean_raw", "mean_resampled", "final20"):
                row[f"tte_pre_{name}_{suffix}"] = float("nan")
            continue
        seg_tail = max(1, int(math.ceil(segment.size * 0.2)))
        row[f"tte_pre_mean_raw_{suffix}"] = float(segment.mean())
        row[f"tte_pre_mean_resampled_{suffix}"] = float(resample(segment).mean())
        row[f"tte_pre_final20_{suffix}"] = float(segment[-seg_tail:].mean())
    return row


def sign_flip(diff: np.ndarray, iters: int = 20000, seed: int = 0) -> float:
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1.0, 1.0], size=(iters, diff.size)) * diff).mean(axis=1))
    return float((null >= observed - 1e-12).mean())


def stats(rows: list[dict], col: str) -> tuple[int, float, float, float]:
    values = np.array([r[col] for r in rows], float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0, float("nan"), float("nan"), float("nan")
    return values.size, float(values.mean()), float(values.std(ddof=0)), float(np.median(values))


def main() -> None:
    rows: list[dict] = []
    for variant, root in RUNS.items():
        got = 0
        for path in sorted(glob.glob(f"{root}/runs/*/*/seed_*/episode_0000/result.json")):
            row = episode_row(path, variant)
            if row is not None:
                rows.append(row)
                got += 1
        print(f"载入 {variant:7s} {root:26s} {got} 条")

    os.makedirs("analysis_outputs", exist_ok=True)
    detail_cols = [
        "episode_id", "variant", "family", "task", "level", "seed", "goal_kind",
        "n_steps", "success", "reach_success", *SGE_COLS, *TTE_COLS, *PRE_COLS,
    ]
    detail_csv = "analysis_outputs/pi05_direct_family_sge_tte_episodes.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"逐 episode 明细: {detail_csv}  ({len(rows)} 行)\n")

    base_all = [r for r in rows if r["variant"] == "base"]
    direct = [r for r in rows if r["variant"] == "direct"]
    families = [name for _, name in FAMILY_MAP]

    base_map = {(r["task"], r["level"], r["seed"]): r for r in base_all}
    direct_map = {(r["task"], r["level"], r["seed"]): r for r in direct}
    shared = sorted(set(base_map) & set(direct_map))
    base_paired = [base_map[k] for k in shared]

    # base 跑了 2 个 seed/level 而 direct 只有 1 个，所以额外给出 base 在配对子集上的值，
    # 让 base/direct 的对比不被样本构成差异干扰。
    groups = [
        ("base(全量)", base_all),
        ("base(配对)", base_paired),
        ("direct", direct),
    ]

    def print_family_table(title: str, cols: list[str]) -> None:
        print("\n" + "=" * (30 + 24 * len(cols)))
        print(title)
        print("=" * (30 + 24 * len(cols)))
        header = f"  {'族':<22s} {'变体':<11s} {'n':>4s}"
        for col in cols:
            header += f" {col:>23s}"
        print(header)
        for family in families + ["__ALL__"]:
            label_family = "全部（27 任务）" if family == "__ALL__" else family
            for gi, (variant_label, variant_rows) in enumerate(groups):
                sub = (
                    variant_rows
                    if family == "__ALL__"
                    else [r for r in variant_rows if r["family"] == family]
                )
                if not sub:
                    continue
                shown = label_family if gi == 0 else ""
                line = f"  {shown:<22s} {variant_label:<11s} {len(sub):>4d}"
                for col in cols:
                    _, mean, std, median = stats(sub, col)
                    line += f" {mean:>8.4f}±{std:<6.4f}/{median:<6.4f}"
                print(line)
            print()

    print_family_table("[表 1] 逐族 SGE（单位 m，均值±std / 中位）", SGE_COLS)
    print_family_table(
        "[表 2] 逐族 TTE（xy，单位 m，均值±std / 中位）",
        [c for c in TTE_COLS if c.endswith("_xy")],
    )
    print_family_table(
        "[表 3] 逐族 TTE（3d 对照，rebuttal_metrics.md §1.2 已弃用）",
        [c for c in TTE_COLS if c.endswith("_3d")],
    )
    print_family_table(
        "[表 3b] 逐族 TTE-pre（截断到 sge_best 时刻，xy）+ 最近点位置",
        ["tte_pre_mean_raw_xy", "tte_pre_mean_resampled_xy", "tte_pre_final20_xy",
         "t_best", "t_best_frac"],
    )


    print("\n" + "=" * 120)
    print("[表 4] 逐族 base → direct 配对差（Δ = direct - base，负号表示 direct 更好）")
    print(f"       sign-flip permutation 20000 次，配对总数 n={len(shared)}")
    print("=" * 120)
    key_cols = [
        "sge_xy_best",
        "tte_pre_mean_raw_xy",
        "tte_pre_mean_resampled_xy",
        "tte_mean_raw_xy",
        "sge_xy_final",
        "tte_final20_xy",
    ]
    header = f"  {'族':<22s} {'n':>4s}"
    for col in key_cols:
        header += f" {col:>30s}"
    print(header)
    for family in families + ["__ALL__"]:
        keys = [k for k in shared if family == "__ALL__" or base_map[k]["family"] == family]
        if not keys:
            continue
        label = "全部（27 任务）" if family == "__ALL__" else family
        line = f"  {label:<22s} {len(keys):>4d}"
        for col in key_cols:
            a = np.array([base_map[k][col] for k in keys], float)
            b = np.array([direct_map[k][col] for k in keys], float)
            ok = np.isfinite(a) & np.isfinite(b)
            if not ok.any():
                line += f" {'不可得':>30s}"
                continue
            delta = (b[ok] - a[ok]).mean()
            p = sign_flip(b[ok] - a[ok])
            mark = "*" if p < 0.05 else " "
            line += f"  {a[ok].mean():.4f}->{b[ok].mean():.4f} Δ{delta:+.4f} p={p:.4f}{mark}"
        print(line)

    summary_csv = "analysis_outputs/pi05_direct_family_sge_tte_summary.csv"
    summary_rows: list[dict] = []
    for family in families + ["__ALL__"]:
        keys = [k for k in shared if family == "__ALL__" or base_map[k]["family"] == family]
        for variant_label, variant_rows in groups:
            sub = (
                variant_rows
                if family == "__ALL__"
                else [r for r in variant_rows if r["family"] == family]
            )
            if not sub:
                continue
            record = {
                "family": "ALL" if family == "__ALL__" else family,
                "variant": variant_label,
                "n_episodes": len(sub),
            }
            for col in SGE_COLS + TTE_COLS + PRE_COLS:
                _, mean, std, median = stats(sub, col)
                record[f"{col}_mean"] = round(mean, 6)
                record[f"{col}_std"] = round(std, 6)
                record[f"{col}_median"] = round(median, 6)
            # 配对差只对 direct 行有意义，base 行留空避免重复读数。
            for col in key_cols:
                delta = p_value = ""
                if variant_label == "direct" and keys:
                    a = np.array([base_map[k][col] for k in keys], float)
                    b = np.array([direct_map[k][col] for k in keys], float)
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.any():
                        delta = round(float((b[ok] - a[ok]).mean()), 6)
                        p_value = round(sign_flip(b[ok] - a[ok]), 4)
                record[f"{col}_delta_vs_base"] = delta
                record[f"{col}_p_signflip"] = p_value
            summary_rows.append(record)
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n逐族汇总: {summary_csv}  ({len(summary_rows)} 行)")

    print("\n" + "=" * 120)
    print("[表 5] 逐任务 base / direct 对照（配对子集，每任务 3 个 level）")
    print("=" * 120)
    print(
        f"  {'族':<22s} {'任务':<44s} {'n':>2s}  "
        f"{'sge_best b->d':<22s} {'tte_pre_raw b->d':<22s} {'tte_raw(全程) b->d':<22s} {'reach b/d'}"
    )
    by_task: dict[tuple[str, str], list] = defaultdict(list)
    for key in shared:
        row = direct_map[key]
        by_task[(row["family"], row["task"])].append(key)
    for family, task in sorted(by_task):
        keys = by_task[(family, task)]
        base_sub = [base_map[k] for k in keys]
        direct_sub = [direct_map[k] for k in keys]
        cells = ""
        for col in ("sge_xy_best", "tte_pre_mean_raw_xy", "tte_mean_raw_xy"):
            _, base_mean, _, _ = stats(base_sub, col)
            _, direct_mean, _, _ = stats(direct_sub, col)
            cells += f"{base_mean:.4f}->{direct_mean:.4f} ({direct_mean - base_mean:+.4f})  "
        reach_base = sum(r["reach_success"] for r in base_sub)
        reach_direct = sum(r["reach_success"] for r in direct_sub)
        print(
            f"  {family:<22s} {task:<44s} {len(keys):>2d}  {cells}"
            f"{reach_base}/{reach_direct} of {len(keys)}"
        )


if __name__ == "__main__":
    main()
