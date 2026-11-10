"""Q4 细粒度分析：chance-corrected grounding + direct 消融配对检验。

回应审稿意见「aggregate metrics 不足以支撑 core difficulty 在身份绑定」。

产出两张表：
  R1  analysis_outputs/q4_grounding_chance_corrected.csv
      逐 task group 的 grounding accuracy 与 per-scene 随机基线 1/n_obj 的对比。
  R2  analysis_outputs/q4_direct_ablation_paired.csv
      base vs direct 在同 task|level|seed 上的配对差异（grounding vs execution 分离）。

输入：analysis_outputs/tte_recompute_episodes.csv（由 scripts/tte_recompute.py 生成，
      逐步读取实时物体位姿，不使用 legacy_layouts.json 的标称位置）。
      初始目标距离 d0 直接从各 run 的 trace.json 首步 goal_error 读取。

注意：
  * `target_closest_rate` 仅在 trace 中含物体位姿的 run 上可用
    （pi0_v3 / pi05_v4 / pi05_v4_direct），其余 run 该列为 NaN。
  * R1 的 `ReindexInsert` 行是退化样本：`tte_min_xy == d0_xy == 0.2197`，
    机械臂全程未移动，目标恰好是起始位形下最近的物体，其 GA=0.921 不可解读为 grounding。
  * R1 的 `TrackMulti` 行 d0_xy=0.183，约为其余组的 3.4 倍，其 GA=0 含几何成分；
    族内对照应使用等距的 TrackSingle vs TrackClutter。

用法：python scripts/analysis_q4_grounding.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import random
import statistics as st

EPISODES_CSV = "analysis_outputs/tte_recompute_episodes.csv"
OUT_R1 = "analysis_outputs/q4_grounding_chance_corrected.csv"
OUT_R2 = "analysis_outputs/q4_direct_ablation_paired.csv"

# 只有这些 run 的 trace 含物体位姿，target_closest_rate 才有定义
GROUNDING_RUNS = {
    "pi0_v3": "run_eval_openpi_05_v3",
    "pi05_v4": "run_eval_pi05_v4",
    "pi05_v4_direct": "run_eval_pi05_v4_direct",
}
R1_MODELS = ("pi0_v3", "pi05_v4")

# R1 的 task group 划分：(family, 标签, task 名匹配子串)
R1_GROUPS = [
    ("Target Identification", "AnchorDistance", "AnchorDistance"),
    ("Target Identification", "AnchorDirection", "AnchorDirection"),
    ("Identity Tracking", "TrackSingle", "TrackSingle"),
    ("Identity Tracking", "TrackMulti", "TrackMulti"),
    ("Identity Tracking", "TrackClutter", "TrackClutter"),
    ("Scene Updating", "ReindexInsert", "ReindexInsert"),
    ("Scene Updating", "ReindexExtreme", "ReindexExtreme"),
    ("Scene Updating", "ReindexNeighbor", "ReindexNeighbor"),
    ("Scene Updating", "ReindexSubset", "ReindexSubset"),
    ("Scene Updating", "OrdinalInsertStatic", "OrdinalInsertStatic"),
    ("Scene Updating", "GridRowcolGrounding", "GridRowcolGrounding"),
]

R2_FAMILIES = [
    "Identity Tracking",
    "Scene Updating",
    "Target Identification",
    "Constraint Composition",
    "Geometric Inference",
]
R2_METRICS = [
    ("target_closest_rate", "grounding"),
    ("reach_success", "execution"),
    ("tte_min_xy", "closest_approach_error"),
]

N_PERM = 20000
N_BOOT = 10000


def fnum(v):
    """CSV 字段转 float，空串/NaN 返回 None。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def load_d0() -> dict[str, float]:
    """从 trace.json 首步 goal_error 取初始目标距离（xy 平面，米）。"""
    d0: dict[str, float] = {}
    for model, root in GROUNDING_RUNS.items():
        for path in glob.glob(f"{root}/runs/*/*/seed_*/episode_0000/trace.json"):
            parts = path.split("/runs/")[1].split("/")
            if len(parts) < 3:
                continue
            task, level, seed = parts[0], parts[1], parts[2]
            try:
                with open(path) as fh:
                    trace = json.load(fh)
                ge = trace[0]["agent_obs"]["state"]["native_obs"]["policy"]["goal_error"]
            except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
            d0[f"{model}|{task}|{level}|{seed}"] = math.hypot(float(ge[0]), float(ge[1]))
    return d0


def perm_test(diffs: list[float], seed: int = 0) -> float:
    """对配对差值做符号翻转置换检验，返回双侧 p。"""
    if not diffs:
        return float("nan")
    rnd = random.Random(seed)
    obs = abs(st.mean(diffs))
    hits = sum(
        abs(st.mean([d if rnd.random() < 0.5 else -d for d in diffs])) >= obs - 1e-12
        for _ in range(N_PERM)
    )
    return (hits + 1) / (N_PERM + 1)


def boot_ci(vals: list[float], seed: int = 1) -> tuple[float, float]:
    """均值的 bootstrap 95% 置信区间。"""
    if len(vals) < 2:
        return float("nan"), float("nan")
    rnd = random.Random(seed)
    means = sorted(st.mean([rnd.choice(vals) for _ in vals]) for _ in range(N_BOOT))
    return means[int(0.025 * N_BOOT)], means[int(0.975 * N_BOOT)]


def build_r1(rows: list[dict]) -> list[dict]:
    pool = [
        r
        for r in rows
        if r["model"] in R1_MODELS
        and fnum(r["target_closest_rate"]) is not None
        and fnum(r["n_obj"])
    ]
    out = []
    for family, label, key in R1_GROUPS:
        grp = [r for r in pool if r["family"] == family and key in r["task"]]
        if not grp:
            continue
        ga = [fnum(r["target_closest_rate"]) for r in grp]
        chance = [1.0 / fnum(r["n_obj"]) for r in grp]
        diffs = [a - c for a, c in zip(ga, chance)]
        lo, hi = boot_ci(ga)
        d0s = [r["d0"] for r in grp if r["d0"] is not None]
        out.append(
            {
                "family": family,
                "task_group": label,
                "n_episodes": len(grp),
                "n_obj": round(st.mean([fnum(r["n_obj"]) for r in grp]), 2),
                "chance": round(st.mean(chance), 4),
                "grounding_accuracy": round(st.mean(ga), 4),
                "ci95_low": round(lo, 4),
                "ci95_high": round(hi, 4),
                "lift_over_chance": round(st.mean(diffs), 4),
                "p_perm": round(perm_test(diffs, seed=2), 4),
                "reach_success": round(
                    st.mean([fnum(r["reach_success"]) or 0.0 for r in grp]), 4
                ),
                "tte_min_xy": round(st.mean([fnum(r["tte_min_xy"]) for r in grp]), 4),
                "d0_xy": round(st.mean(d0s), 4) if d0s else "",
            }
        )
    return out


def build_r2(rows: list[dict]) -> list[dict]:
    def key_of(r: dict) -> tuple[str, str, str]:
        parts = r["episode_id"].split("|")
        return (r["task"], parts[2], parts[3])

    base = {key_of(r): r for r in rows if r["model"] == "pi05_v4"}
    pairs = [
        (base[key_of(r)], r)
        for r in rows
        if r["model"] == "pi05_v4_direct" and key_of(r) in base
    ]
    out = []
    for metric, kind in R2_METRICS:
        for family in R2_FAMILIES:
            vals = [
                (fnum(b[metric]), fnum(d[metric]))
                for b, d in pairs
                if b["family"] == family
                and fnum(b[metric]) is not None
                and fnum(d[metric]) is not None
            ]
            if len(vals) < 3:
                continue
            diffs = [d - b for b, d in vals]
            out.append(
                {
                    "metric": metric,
                    "metric_kind": kind,
                    "family": family,
                    "n_pairs": len(vals),
                    "base": round(st.mean([b for b, _ in vals]), 4),
                    "direct": round(st.mean([d for _, d in vals]), 4),
                    "delta": round(st.mean(diffs), 4),
                    "p_perm": round(perm_test(diffs, seed=3), 4),
                }
            )
    return out


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def show(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    cols = list(rows[0].keys())
    widths = [
        max(len(c), max(len(str(r[c])) for r in rows)) + 2 for c in cols
    ]
    print("".join(c.rjust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("".join(str(r[c]).rjust(w) for c, w in zip(cols, widths)))


def main() -> None:
    with open(EPISODES_CSV) as fh:
        rows = list(csv.DictReader(fh))
    d0 = load_d0()
    for r in rows:
        r["d0"] = d0.get(r["episode_id"])

    r1 = build_r1(rows)
    r2 = build_r2(rows)
    write_csv(OUT_R1, r1)
    write_csv(OUT_R2, r2)
    show("R1  chance-corrected grounding", r1)
    show("R2  direct 消融配对检验", r2)
    print(f"\n已写出 {OUT_R1}\n已写出 {OUT_R2}")


if __name__ == "__main__":
    main()
