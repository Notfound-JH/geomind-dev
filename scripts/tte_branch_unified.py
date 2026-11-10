"""
按真实张量结构统一解析 policy.object，重算论文口径 TTE。

背景（诊断结论）：
  trace 里的 policy.object 结构为
      [pos3+quat4] * n   +   相对eef位移 3*n   +   两两相对位置 3*C(n,2)
      总长 = 10n + 3n(n-1)/2
  原脚本 _parse_policy_object_tensor 假设它是规则的 dpo×n 数组，
  仅当 len(raw) % n == 0 时才走 "dynamic"，且按 dpo=len//n 切分 —— 索引 >=1 的
  物体全部错位（读到四元数/相对量）。整除性随 n 变化，导致 base/direct 分支跳变。

本脚本把解析改成按真实结构取每物体前 3 维（真值），使两批走同一分支且坐标正确。
"""

import csv
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/elm/projects/geomind_lab/scripts")
from rebuttal_metrics import sign_flip  # noqa: E402

ROOT = Path("/home/elm/projects/geomind_lab")
OUT = ROOT / "analysis_outputs/tte_branch_unified"
OUT.mkdir(parents=True, exist_ok=True)

PATCH = r'''
import sys
sys.path.insert(0, "/home/elm/projects/geomind_lab/scripts")
import metrics_trajectory as mt


def _fixed(pol, ordered_names):
    raw = pol.get("object")
    if isinstance(raw, dict) and raw.get("omitted"):
        return None
    if not isinstance(raw, list) or not raw or not ordered_names:
        return None
    n = len(ordered_names)
    if len(raw) != 10 * n + 3 * n * (n - 1) // 2:
        return None
    try:
        return {
            name: (float(raw[i * 7]), float(raw[i * 7 + 1]), float(raw[i * 7 + 2]))
            for i, name in enumerate(ordered_names)
        }
    except (TypeError, ValueError, IndexError):
        return None


mt._parse_policy_object_tensor = _fixed
sys.argv = ["metrics_trajectory.py"] + sys.argv[1:]
raise SystemExit(mt.main())
'''
runner = OUT / "_patched_runner.py"
runner.write_text(PATCH, encoding="utf-8")

LAYOUTS = [
    "geomind/configs/legacy/layouts/standard.json",
    "geomind/configs/legacy/layouts/vla.json",
]
RUNS = {"base": "run_eval_pi05_v4", "direct": "run_eval_pi05_v4_direct"}

for tag, run in RUNS.items():
    cmd = [
        sys.executable, str(runner),
        "--load-trace-eef",
        "--results", str(ROOT / run / "trial_results.jsonl"),
        "--out", str(OUT / f"{tag}.jsonl"),
        "--csv", str(OUT / f"{tag}.csv"),
    ]
    for lay in LAYOUTS:
        cmd += ["--layouts", str(ROOT / lay)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[{tag}] exit={r.returncode} {r.stderr.strip()[-200:]}")


def load(tag: str) -> dict:
    out = {}
    for r in csv.DictReader(open(OUT / f"{tag}.csv")):
        parts = Path(r["result_json"]).parts
        task = next((p for p in parts if "Vla_v0" in p), r.get("task_id", ""))
        seed = next((p for p in parts if p.startswith("seed_")), "")
        out[(task, seed)] = r
    return out


def f(r, k):
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


base, direct = load("base"), load("direct")
print()
print("=" * 78)
print("0. 分支一致性检查（修复后）")
print("=" * 78)
for tag, d in (("base", base), ("direct", direct)):
    c: dict = {}
    for r in d.values():
        c[r["object_pos_source"]] = c.get(r["object_pos_source"], 0) + 1
    print(f"  {tag:7s} n={len(d):4d}  object_pos_source={c}")

common = sorted(set(base) & set(direct))
KEYS = ["mean_target_distance", "final_target_distance", "min_target_distance",
        "target_closest_rate", "mean_target_margin", "target_near_fraction"]


def preloaded(r) -> bool:
    return str(r.get("target_preloaded_in_gripper", "")).lower() == "true"


SUBSETS = {
    "全部": [c for c in common],
    "排除 held_target_block(预装在夹爪)": [c for c in common if not (preloaded(base[c]) or preloaded(direct[c]))],
}

for name, keys in SUBSETS.items():
    print()
    print("=" * 78)
    print(f"{name}   n={len(keys)}   (配对 sign-flip 20000)")
    print("=" * 78)
    print(f"{'指标':24s} {'base':>9s} {'direct':>9s} {'Δ':>9s} {'p':>8s}")
    for k in KEYS:
        a = np.array([f(base[c], k) for c in keys])
        b = np.array([f(direct[c], k) for c in keys])
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        if a.size == 0:
            continue
        p = sign_flip(b - a)
        print(f"{k:24s} {a.mean():9.4f} {b.mean():9.4f} {b.mean() - a.mean():+9.4f} {p:8.4f}{' *' if p < 0.05 else ''}")

FAM = {
    "Binding": "Target Identification", "Constraint": "Constraint Composition",
    "Geometric": "Geometric Inference", "Relation": "Relational Grounding",
    "Reindex": "Scene Updating", "Track": "Identity Tracking",
}


def fam(task: str) -> str:
    for p, n in FAM.items():
        if task.startswith(p):
            return n
    return "?"


print()
print("=" * 78)
print("逐族 mean_target_distance（配对，全部回合）")
print("=" * 78)
print(f"{'family':24s} {'n':>3s} {'base':>9s} {'direct':>9s} {'Δ':>9s}  {'预装夹爪':>8s}")
groups: dict = {}
for c in common:
    groups.setdefault(fam(c[0]), []).append(c)
for name in sorted(groups):
    ks = groups[name]
    a = np.array([f(base[c], "mean_target_distance") for c in ks])
    b = np.array([f(direct[c], "mean_target_distance") for c in ks])
    npre = sum(1 for c in ks if preloaded(base[c]) or preloaded(direct[c]))
    print(f"{name:24s} {len(ks):3d} {np.nanmean(a):9.4f} {np.nanmean(b):9.4f} "
          f"{np.nanmean(b) - np.nanmean(a):+9.4f}  {npre:8d}")

print()
print("=" * 78)
print("全体均值（非配对）")
print("=" * 78)
for tag, d in (("base", base), ("direct", direct)):
    v = [f(r, "mean_target_distance") for r in d.values()]
    v = [x for x in v if np.isfinite(x)]
    print(f"  {tag:7s} n={len(v):4d} mean={st.mean(v):.4f} median={st.median(v):.4f} "
          f"p25={np.percentile(v, 25):.4f} p75={np.percentile(v, 75):.4f} "
          f"range=[{min(v):.4f},{max(v):.4f}]")

json.dump({"note": "policy.object 按 10n+3C(n,2) 真实结构解析后的论文口径 TTE"},
          open(OUT / "README.json", "w"), ensure_ascii=False)
