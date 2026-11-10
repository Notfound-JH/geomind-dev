# Paper appendix assets

## Appendix result dashboard (LaTeX + figures)

- `appendix_results_dashboard_preamble.tex` — packages, colors, and `\augcell`; `\input{}` from your main preamble.
- `appendix_results_dashboard.tex` — full `\section{Complete Experimental Results}` body; numeric tables `\input{}` fragments from `generated/`.
- `data/appendix/*.csv` — **generated** from repo `run_eval_*` outputs (do not hand-edit for routine updates).
- `generated/*.tex` — table row bodies produced by `scripts/render_appendix_tex_tables.py`.
- `figures/` — PDFs from `scripts/plot_appendix_dashboard.py`.

### Refresh numbers from eval runs (recommended)

After new `run_eval_*` summaries or an updated `run_eval_models_task_level_matrix.csv`:

```bash
python scripts/aggregate_run_eval_matrix.py   # optional: refresh root matrix from run_eval_*
python scripts/export_paper_appendix_data.py  # writes paper/data/appendix/*.csv
python scripts/render_appendix_tex_tables.py  # writes paper/generated/app_*_table_body.tex
python scripts/plot_appendix_dashboard.py     # writes paper/figures/app_*.pdf
```

Or only figures:

```bash
python scripts/plot_appendix_dashboard.py
python scripts/plot_app_result_dashboard.py   # wrappers: dashboard / heatmap / augdrop only
```

**Data sources:** `statistics/analysis_outputs/matrix_radar/run_eval_models_task_level_matrix_family_by_model_long.csv`, `run_eval_models_task_level_matrix.csv`, and `analysis_outputs/eval_diagnostics/run_eval_<slug>/aggregate_diagnostics_by_*.csv`. Oracle defaults to `100.0` in CSVs (nominal ceiling); override with `python scripts/export_paper_appendix_data.py --oracle-percent 95` if needed.

### LaTeX

Set `\graphicspath{{paper/figures/}{figures/}}` or adjust `\includegraphics{...}` paths in `appendix_results_dashboard.tex` to match your project layout. Ensure `paper/generated/app_*_table_body.tex` exists (run `render_appendix_tex_tables.py`) before compiling.
