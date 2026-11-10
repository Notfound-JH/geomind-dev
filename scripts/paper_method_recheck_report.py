"""按论文原口径（metrics_trajectory.py + geomind_eval_diagnostics.py）汇总 pi05 base/direct 的 TTE 与 SGE。"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis_outputs" / "paper_method_recheck"
RUNS = {"base": "pi05_v4", "direct": "pi05_v4_direct"}
RNG = np.random.default_rng(0)


def fnum(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def load_traj(run: str) -> list[dict]:
    return [json.loads(l) for l in (OUT / f"{run}_traj.jsonl").read_text().splitlines() if l.strip()]


def load_diag(run: str) -> list[dict]:
    with (OUT / f"diag_{run}" / "per_episode_diagnostics.csv").open() as fh:
        return list(csv.DictReader(fh))


def epi_key(episode_dir: str) -> tuple[str, str]:
    p = Path(episode_dir)
    parts = p.parts
    task = next((x for x in parts if "Vla_v0" in x or "_v0" in x), p.name)
    seed = next((x for x in parts if x.startswith("seed_")), p.name)
    return (task, seed)


def stats(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
                lo=float(a.min()), hi=float(a.max()))


def sign_flip(d, it=20000):
    d = np.asarray(d, dtype=float)
    obs = abs(d.mean())
    s = RNG.choice([-1.0, 1.0], size=(it, d.size))
    return float((np.abs((s * d).mean(1)) >= obs - 1e-12).mean())


def fmt(s):
    return "n/a" if s is None else (f"n={s['n']:3d}  mean={s['mean']:.4f}  median={s['median']:.4f}  "
                                    f"[p25={s['p25']:.4f} p75={s['p75']:.4f}]  range=[{s['lo']:.4f},{s['hi']:.4f}]")


TRAJ_COLS = ["mean_target_distance", "final_target_distance", "min_target_distance",
             "target_closest_rate", "mean_target_margin", "target_near_fraction"]


def main() -> None:
    traj = {k: load_traj(v) for k, v in RUNS.items()}
    diag = {k: load_diag(v) for k, v in RUNS.items()}

    print("=" * 92)
    print("A. 论文口径 TTE（metrics_trajectory.py，未做任何修改）")
    print("=" * 92)
    for cond in RUNS:
        rows = traj[cond]
        print(f"\n--- {cond} (run_eval_{RUNS[cond]}, 共 {len(rows)} 条) ---")
        src = defaultdict(int)
        for r in rows:
            src[str(r.get("object_pos_source"))] += 1
        print(f"  object_pos_source: {dict(src)}   static_layout_proxy={sum(1 for r in rows if r.get('static_layout_proxy'))}")
        xo = defaultdict(int)
        for r in rows:
            xo[str(r.get("xy_only"))] += 1
        print(f"  xy_only: {dict(xo)}")
        for c in TRAJ_COLS:
            print(f"  {c:24s} {fmt(stats([fnum(r.get(c)) for r in rows]))}")

    print("\n" + "=" * 92)
    print("B. 论文口径 SGE（geomind_eval_diagnostics.py，未做任何修改）")
    print("=" * 92)
    for cond in RUNS:
        rows = diag[cond]
        print(f"\n--- {cond} (共 {len(rows)} 条) ---")
        print(f"  {'sge':24s} {fmt(stats([fnum(r.get('sge')) for r in rows]))}")
        w = defaultdict(int)
        for r in rows:
            for tok in str(r.get("sge_warning") or r.get("warnings") or "").split(";"):
                if tok.strip():
                    w[tok.strip()] += 1
        print(f"  sge 警告: {dict(w) if w else '(无)'}")
        for extra in ("mta", "scs", "success", "reach_success"):
            if rows and extra in rows[0]:
                s = stats([fnum(r.get(extra)) for r in rows])
                if s:
                    print(f"  {extra:24s} {fmt(s)}")

    print("\n" + "=" * 92)
    print("C. base → direct 配对比较（同种子；符号翻转置换检验 20000 次，双侧）")
    print("=" * 92)
    A = {epi_key(r["episode_dir"]): r for r in traj["base"] if r.get("episode_dir")}
    B = {epi_key(r["episode_dir"]): r for r in traj["direct"] if r.get("episode_dir")}
    shared = sorted(set(A) & set(B))
    print(f"配对 n = {len(shared)}（base {len(A)} / direct {len(B)}）\n")
    for c in TRAJ_COLS:
        pa, pb = [], []
        for k in shared:
            va, vb = fnum(A[k].get(c)), fnum(B[k].get(c))
            if va is not None and vb is not None:
                pa.append(va)
                pb.append(vb)
        if not pa:
            print(f"  {c:24s} 无可配对样本")
            continue
        a, b = np.array(pa), np.array(pb)
        p = sign_flip(b - a)
        print(f"  {c:24s} n={a.size:3d}  base={a.mean():.4f}  direct={b.mean():.4f}  "
              f"Δ={b.mean() - a.mean():+.4f}  p={p:.4f}{'  ★' if p < 0.05 else ''}")

    dA = {epi_key(r.get("result_json", "")): r for r in diag["base"] if r.get("result_json")}
    dB = {epi_key(r.get("result_json", "")): r for r in diag["direct"] if r.get("result_json")}
    ds = sorted(set(dA) & set(dB))
    pa, pb = [], []
    for k in ds:
        va, vb = fnum(dA[k].get("sge")), fnum(dB[k].get("sge"))
        if va is not None and vb is not None:
            pa.append(va)
            pb.append(vb)
    if pa:
        a, b = np.array(pa), np.array(pb)
        p = sign_flip(b - a)
        print(f"  {'sge':24s} n={a.size:3d}  base={a.mean():.4f}  direct={b.mean():.4f}  "
              f"Δ={b.mean() - a.mean():+.4f}  p={p:.4f}{'  ★' if p < 0.05 else ''}")
    else:
        print(f"  {'sge':24s} 无可配对样本（配对键 {len(ds)}）")

    print("\n" + "=" * 92)
    print("D. 逐任务族（论文口径 TTE = mean_target_distance；SGE），按 task_id 前缀统一分族")
    print("=" * 92)
    fam_map = {"Binding": "Target Identification", "Relation": "Relational Grounding",
               "Constraint": "Constraint Composition", "Geometric": "Geometric Inference",
               "Reindex": "Scene Updating", "Track": "Identity Tracking"}

    def fam_of_task(tid: str) -> str:
        for pre, name in fam_map.items():
            if str(tid).startswith(pre):
                return name
        return "?"

    byfam: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for cond in RUNS:
        for r in traj[cond]:
            v = fnum(r.get("mean_target_distance"))
            if v is not None:
                byfam[fam_of_task(r.get("task_id", ""))][f"tte_{cond}"].append(v)
        for r in diag[cond]:
            v = fnum(r.get("sge"))
            if v is not None:
                byfam[fam_of_task(r.get("task_id", ""))][f"sge_{cond}"].append(v)
    print(f"{'family':26s} {'TTE base':>10s} {'TTE direct':>11s} {'ΔTTE':>9s} "
          f"{'SGE base':>10s} {'SGE direct':>11s} {'ΔSGE':>9s}")
    for fam in sorted(byfam):
        d = byfam[fam]

        def m(key):
            return np.mean(d[key]) if d.get(key) else None

        tb, td, sb_, sd_ = m("tte_base"), m("tte_direct"), m("sge_base"), m("sge_direct")

        def s(x):
            return "—" if x is None else f"{x:.4f}"

        def dl(x, y):
            return "—" if (x is None or y is None) else f"{y - x:+.4f}"

        print(f"{fam:26s} {s(tb):>10s} {s(td):>11s} {dl(tb, td):>9s} "
              f"{s(sb_):>10s} {s(sd_):>11s} {dl(sb_, sd_):>9s}")

    print("\n" + "=" * 92)
    print("E. 与论文归档值对照")
    print("=" * 92)
    arch = ROOT / "analysis_outputs" / "openpi_05_metrics_trajectory.csv"
    if arch.is_file():
        with arch.open() as fh:
            rows = list(csv.DictReader(fh))
        for c in ("mean_target_distance", "final_target_distance", "min_target_distance", "target_closest_rate"):
            print(f"  归档 openpi_05 {c:24s} {fmt(stats([fnum(r.get(c)) for r in rows]))}")


if __name__ == "__main__":
    main()
