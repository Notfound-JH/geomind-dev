#!/usr/bin/env python3
"""从 paper/data/appendix/*.csv 生成 LaTeX 行片段，供 appendix_results_dashboard.tex \\input。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PARADIGM_COLOR = {
    "oracle": "OracleFill",
    "vla": "VLAFill",
    "spatial_vla": "SpatialVLAFill",
    "modular": "ModularFill",
}

METHOD_TEX = {
    "Oracle": "Oracle",
    "OpenVLA": "OpenVLA",
    "pi_0": r"$\pi_0$",
    "pi_0_5": r"$\pi_{0.5}$",
    "SpatialVLA": "SpatialVLA",
    "InternVLA-M1": "InternVLA-M1",
    "Spatial-Forcing": "Spatial-Forcing",
    "COPA-GPT-4o": "COPA-GPT-4o",
    "COPA-Gemini": "COPA-Gemini",
    "COPA-Claude": "COPA-Claude",
    "MOKA-GPT-4o": "MOKA-GPT-4o",
    "MOKA-Gemini": "MOKA-Gemini",
    "MOKA-Claude": "MOKA-Claude",
}

MODULAR_NAMES = [
    "COPA-GPT-4o",
    "COPA-Gemini",
    "COPA-Claude",
    "MOKA-GPT-4o",
    "MOKA-Gemini",
    "MOKA-Claude",
]

FAMILY_KEYS = [
    "target_identification",
    "relational_grounding",
    "constraint_composition",
    "geometric_inference",
    "scene_updating",
    "identity_tracking",
]


def _cell(v: str) -> str:
    s = (v or "").strip()
    if not s:
        return r"\textemdash{}"
    return s


def _textbf(s: str) -> str:
    return rf"\textbf{{{s}}}"


def render_base_body(csv_path: Path, out: Path) -> None:
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lines: list[str] = []
    for row in rows:
        method = row["method"]
        paradigm = row.get("paradigm", "vla")
        color = PARADIGM_COLOR.get(paradigm, "HeaderGray")
        mtex = METHOD_TEX.get(method, method)
        fam_cells = [_cell(row.get(k, "")) for k in FAMILY_KEYS]
        sr = _textbf(_cell(row.get("spatial_reasoning_avg", "")))
        sm = _textbf(_cell(row.get("spatial_memory_avg", "")))
        ov = _textbf(_cell(row.get("overall_avg", "")))
        body = " & ".join(fam_cells + [sr, sm, ov])
        lines.append(f"    \\rowcolor{{{color}}}")
        lines.append(f"    {mtex}")
        lines.append(f"      & {body} \\\\")
        if method == "Oracle":
            lines.append(r"    \midrule")
        if method == "pi_0_5":
            lines.append(r"    \midrule")
        if method == "Spatial-Forcing":
            lines.append(r"    \midrule")
    for mn in MODULAR_NAMES:
        mtex = METHOD_TEX[mn]
        em = r"\textemdash{}"
        b = rf"\textbf{{{em}}}"
        z = " & ".join([em] * 6 + [b, b, b])
        lines.append(f"    \\rowcolor{{{PARADIGM_COLOR['modular']}}}")
        lines.append(f"    {mtex}")
        lines.append(f"      & {z} \\\\")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_aug_body(csv_path: Path, out: Path) -> None:
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lines: list[str] = []

    def augcell(succ: str, drop: str) -> str:
        su = (succ or "").strip()
        dr = (drop or "").strip()
        if not su:
            return r"\textemdash{}"
        if not dr:
            return rf"\augcell{{{su}}}{{}}"
        if not dr.startswith(("-", "+")):
            try:
                drn = float(dr)
                dr = f"+{drn:.1f}" if drn > 0 else f"{drn:.1f}"
            except ValueError:
                pass
        return rf"\augcell{{{su}}}{{{dr}}}"

    for row in rows:
        method = row["method"]
        paradigm = row.get("paradigm", "vla")
        color = PARADIGM_COLOR.get(paradigm, "HeaderGray")
        mtex = METHOD_TEX.get(method, method)
        c0 = _cell(row.get("base_avg", ""))
        c1 = augcell(row.get("semantic_success", ""), row.get("semantic_drop", ""))
        c2 = augcell(row.get("visual_success", ""), row.get("visual_drop", ""))
        c3 = augcell(row.get("execution_success", ""), row.get("execution_drop", ""))
        c4 = _cell(row.get("aug_avg", ""))
        w = (row.get("worst_drop") or "").strip()
        c5 = (r"\cellcolor{DropLight}" + _cell(w)) if w else r"\cellcolor{DropLight}\textemdash{}"
        lines.append(f"    \\rowcolor{{{color}}}")
        lines.append(f"    {mtex}")
        lines.append(f"      & {c0} & {c1} & {c2} & {c3} & {c4} & {c5} \\\\")
        if method == "pi_0_5":
            lines.append(r"    \midrule")
        if method == "Spatial-Forcing":
            lines.append(r"    \midrule")
    for mn in MODULAR_NAMES:
        mtex = METHOD_TEX[mn]
        em = r"\textemdash{}"
        lines.append(f"    \\rowcolor{{{PARADIGM_COLOR['modular']}}}")
        lines.append(f"    {mtex}")
        lines.append(
            f"      & {em} & {em} & {em} & {em} & {em} & \\cellcolor{{DropLight}}{em} \\\\"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


DEMAND_COLOR = {"Low": "DemandLow", "Medium": "DemandMed", "High": "DemandHigh"}


def render_capability_body(csv_path: Path, out: Path) -> None:
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lines: list[str] = []
    for row in rows:
        fam = row["family"]
        mem = row["memory_demand"]
        reas = row["reasoning_demand"]
        mc = DEMAND_COLOR.get(mem, "HeaderGray")
        rc = DEMAND_COLOR.get(reas, "HeaderGray")
        bm = _cell(row.get("best_modular", ""))
        rest = " & ".join(
            [
                rf"\cellcolor{{{mc}}}{mem}",
                rf"\cellcolor{{{rc}}}{reas}",
                _cell(row.get("oracle", "")),
                _cell(row.get("best_vla", "")),
                _cell(row.get("best_spatial_vla", "")),
                bm,
                _cell(row.get("overall_avg", "")),
            ]
        )
        lines.append(f"    {fam}\n      & {rest} \\\\")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_diagnostics_body(csv_path: Path, out: Path) -> None:
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lines: list[str] = []
    for row in rows:
        fam = row["family"]
        dim = row["capability_dimension"]
        fill = "ReasoningFill" if "Reasoning" in dim else "MemoryFill"
        fr = _cell(row.get("failure_rate", ""))
        sge = _cell(row.get("sge", ""))
        tte = _cell(row.get("tte", ""))
        mft = row.get("main_failure_type", "")
        lines.append(
            f"    {fam} & \\cellcolor{{{fill}}}{dim} & {fr} & {sge} & {tte} & {mft} \\\\"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=REPO)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    data = root / "paper" / "data" / "appendix"
    gen = root / "paper" / "generated"
    render_base_body(data / "base_family_results.csv", gen / "app_base_table_body.tex")
    render_aug_body(data / "aug_results.csv", gen / "app_aug_table_body.tex")
    render_capability_body(data / "capability_demand_results.csv", gen / "app_capability_table_body.tex")
    render_diagnostics_body(data / "diagnostics_results.csv", gen / "app_diagnostics_table_body.tex")
    print(f"Wrote {gen / 'app_base_table_body.tex'}")
    print(f"Wrote {gen / 'app_aug_table_body.tex'}")
    print(f"Wrote {gen / 'app_capability_table_body.tex'}")
    print(f"Wrote {gen / 'app_diagnostics_table_body.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
