#!/usr/bin/env python3
"""Appendix figures for GeoMind paper: dashboard, Base heatmap, Aug drop bars."""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths & CSV parsing (TODO placeholders -> 0.0 for draft figures)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def parse_float_series(series: pd.Series, *, name: str) -> pd.Series:
    out: list[float] = []
    warned = False
    for x in series:
        if pd.isna(x):
            out.append(0.0)
            continue
        s = str(x).strip().lstrip("+")
        if s.upper() == "TODO":
            if not warned:
                print(
                    f"[plot_appendix_dashboard] {name}: placeholder TODO treated as 0.0 for plotting.",
                    file=sys.stderr,
                )
                warned = True
            out.append(0.0)
        else:
            out.append(float(s))
    return pd.Series(out, index=series.index)


def plot_dashboard(data_dir: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    base = pd.read_csv(data_dir / "base_family_results.csv")
    aug = pd.read_csv(data_dir / "aug_results.csv")
    drops = pd.read_csv(data_dir / "aug_drop_results.csv")
    out = out_dir / "app_result_dashboard.pdf"
    out_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))

    methods = base["method"].tolist()
    vals = parse_float_series(base["overall_avg"], name="base_family_results.overall_avg")
    ax = axes[0, 0]
    ax.bar(np.arange(len(methods)), vals, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_xticks(np.arange(len(methods)), methods, rotation=35, ha="right")
    ax.set_ylabel("Success (%)")
    ax.set_title("(a) GeoMind-Base overall")

    methods_b = aug["method"].tolist()
    vals_b = parse_float_series(aug["aug_avg"], name="aug_results.aug_avg")
    ax = axes[0, 1]
    ax.bar(np.arange(len(methods_b)), vals_b, color="#55A868", edgecolor="white", linewidth=0.4)
    ax.set_xticks(np.arange(len(methods_b)), methods_b, rotation=35, ha="right")
    ax.set_ylabel("Success (%)")
    ax.set_title("(b) GeoMind-Aug overall")

    base_no_oracle = base[base["paradigm"].astype(str).str.lower() != "oracle"].copy()
    base_no_oracle["_overall_avg_num"] = parse_float_series(
        base_no_oracle["overall_avg"], name="base_family_results.overall_avg"
    )
    g = base_no_oracle.groupby("paradigm", sort=False)["_overall_avg_num"].mean()
    ax = axes[1, 0]
    ax.bar(np.arange(len(g)), g.values.astype(float), color="#8172B3", edgecolor="white", linewidth=0.4)
    ax.set_xticks(np.arange(len(g)), list(g.index), rotation=20, ha="right")
    ax.set_ylabel("Mean success (%)")
    ax.set_title("(c) Mean Base success by paradigm")

    x = np.arange(len(drops))
    w = 0.25
    sem = parse_float_series(drops["semantic_drop"], name="aug_drop.semantic_drop")
    vis = parse_float_series(drops["visual_drop"], name="aug_drop.visual_drop")
    exe = parse_float_series(drops["execution_drop"], name="aug_drop.execution_drop")
    ax = axes[1, 1]
    ax.bar(x - w, sem, width=w, label=r"Semantic $\Delta$", color="#CC613E", edgecolor="white", linewidth=0.3)
    ax.bar(x, vis, width=w, label=r"Visual $\Delta$", color="#E59C52", edgecolor="white", linewidth=0.3)
    ax.bar(x + w, exe, width=w, label=r"Execution $\Delta$", color="#6BAED6", edgecolor="white", linewidth=0.3)
    ax.set_xticks(x, drops["method"].tolist(), rotation=35, ha="right")
    ax.set_ylabel("Positive degradation (p.p.)")
    ax.set_title(r"(d) Augmentation fragility (Base$-$Aug)")
    ax.legend(frameon=False, ncol=3, loc="upper right")

    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_base_heatmap(data_dir: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    family_cols = [
        "target_identification",
        "relational_grounding",
        "constraint_composition",
        "geometric_inference",
        "scene_updating",
        "identity_tracking",
    ]
    family_labels = [
        "Target Id.",
        "Rel. Grounding",
        "Constr. Comp.",
        "Geom. Infer.",
        "Scene Upd.",
        "Id. Tracking",
    ]
    csv_path = data_dir / "base_family_results.csv"
    out = out_dir / "app_base_heatmap.pdf"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    mat = np.zeros((len(df), len(family_cols)), dtype=float)
    for j, c in enumerate(family_cols):
        mat[:, j] = parse_float_series(df[c], name=f"heatmap.{c}").to_numpy()

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, max(2.8, 0.35 * len(df))))
    cmap = mpl.colormaps["YlGnBu"]
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=100.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Success Rate (%)")

    ax.set_xticks(np.arange(len(family_cols)), family_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(df)), df["method"].tolist())
    ax.set_xlabel("Task family")
    ax.set_ylabel("Method")

    raw = df[family_cols].copy()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            cell = raw.iloc[i, j]
            label = "TODO" if isinstance(cell, str) and str(cell).strip().upper() == "TODO" else f"{mat[i, j]:.1f}"
            color = "0.35" if label == "TODO" else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_aug_drop_bar(data_dir: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    csv_path = data_dir / "aug_drop_results.csv"
    out = out_dir / "app_aug_drop_bar.pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    x = np.arange(len(df))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(
        x - w,
        parse_float_series(df["semantic_drop"], name="aug_drop.semantic_drop"),
        width=w,
        label="Semantic",
        color="#CC613E",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.bar(
        x,
        parse_float_series(df["visual_drop"], name="aug_drop.visual_drop"),
        width=w,
        label="Visual",
        color="#E59C52",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.bar(
        x + w,
        parse_float_series(df["execution_drop"], name="aug_drop.execution_drop"),
        width=w,
        label="Execution",
        color="#6BAED6",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xticks(x, df["method"].tolist(), rotation=35, ha="right")
    ax.set_ylabel("Base $-$ Aug (p.p.)")
    ax.set_title("Augmentation-induced degradation (higher is more fragile)")
    ax.legend(frameon=False, ncol=3)
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate appendix dashboard PDF figures.")
    p.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=None,
        help="Directory with CSVs (default: paper/data/appendix under repo root).",
    )
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for PDFs (default: paper/figures).",
    )
    p.add_argument(
        "--only",
        choices=("dashboard", "heatmap", "augdrop", "all"),
        default="all",
    )
    args = p.parse_args()
    root = repo_root()
    data_dir = args.data_dir or (root / "paper" / "data" / "appendix")
    out_dir = args.out_dir or (root / "paper" / "figures")
    if not data_dir.is_dir():
        warnings.warn(f"Data directory not found: {data_dir}", stacklevel=1)

    if args.only in ("dashboard", "all"):
        path = plot_dashboard(data_dir, out_dir)
        print(path)
    if args.only in ("heatmap", "all"):
        path = plot_base_heatmap(data_dir, out_dir)
        print(path)
    if args.only in ("augdrop", "all"):
        path = plot_aug_drop_bar(data_dir, out_dir)
        print(path)


if __name__ == "__main__":
    main()
