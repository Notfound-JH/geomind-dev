#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/configs/geomind_benchmark_smoke.json}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/protocol_smoke_regression/${TIMESTAMP}}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "GeoMind Protocol Smoke Regression"
echo "========================================================================"
echo "manifest=${MANIFEST_PATH}"
echo "output_dir=${OUTPUT_DIR}"
echo "dry_run=${DRY_RUN}"
echo "========================================================================"

AGENT_NAMES_OVERRIDE="${AGENT_NAMES_OVERRIDE:-oracle}" \
STATE_REFRESH_AGENT_EPISODES="${STATE_REFRESH_AGENT_EPISODES:-1}" \
GEOMETRIC_STRUCTURE_AGENT_EPISODES="${GEOMETRIC_STRUCTURE_AGENT_EPISODES:-1}" \
HEADLESS="${HEADLESS:-1}" \
DRY_RUN="${DRY_RUN}" \
bash "${SCRIPT_DIR}/run_geomind_benchmark.sh" "${MANIFEST_PATH}" "${OUTPUT_DIR}"

cp "${OUTPUT_DIR}/run_manifest.tsv" "${OUTPUT_DIR}/smoke_regression_manifest.tsv"

python - "${OUTPUT_DIR}" "${DRY_RUN}" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

output_dir = Path(sys.argv[1]).resolve()
dry_run = bool(int(sys.argv[2]))
report_path = output_dir / "smoke_regression_report.txt"


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


lines: list[str] = []
lines.append(f"output_dir={output_dir}")
lines.append(f"dry_run={int(dry_run)}")

benchmark_info = output_dir / "benchmark_info.json"
run_manifest = output_dir / "run_manifest.tsv"
smoke_manifest = output_dir / "smoke_regression_manifest.tsv"
expect(benchmark_info.is_file(), "benchmark_info.json was not written")
expect(run_manifest.is_file(), "run_manifest.tsv was not written")
expect(smoke_manifest.is_file(), "smoke_regression_manifest.tsv was not written")
lines.append("check: benchmark manifest parsing -> ok")
lines.append("check: benchmark run manifest -> ok")

if dry_run:
    lines.append("check: runtime rollout execution -> skipped (dry_run=1)")
    lines.append("check: summary field validation -> skipped (dry_run=1)")
else:
    aggregate_state_refresh = output_dir / "aggregate_state_refresh.csv"
    aggregate_geometric_structure = output_dir / "aggregate_geometric_structure.csv"
    joint_summary = output_dir / "joint_summary.csv"
    expect(aggregate_state_refresh.is_file(), "aggregate_state_refresh.csv was not written")
    expect(aggregate_geometric_structure.is_file(), "aggregate_geometric_structure.csv was not written")
    expect(joint_summary.is_file(), "joint_summary.csv was not written")
    lines.append("check: family aggregate outputs -> ok")
    lines.append("check: joint summary output -> ok")

    state_refresh_summaries = sorted(output_dir.glob("state_refresh/**/*.summary.json"))
    geometric_structure_summaries = sorted(output_dir.glob("geometric_structure/**/*.summary.json"))
    expect(state_refresh_summaries, "No state_refresh summary JSON files were produced")
    expect(geometric_structure_summaries, "No geometric_structure summary JSON files were produced")

    sr_payload = json.loads(state_refresh_summaries[0].read_text(encoding="utf-8"))
    gs_payload = json.loads(geometric_structure_summaries[0].read_text(encoding="utf-8"))
    for field in ("task_name", "family", "agent_name", "episode_count", "valid_rate", "success_rate"):
        expect(field in sr_payload, f"state_refresh summary missing field: {field}")
        expect(field in gs_payload, f"geometric_structure summary missing field: {field}")
    expect("intervention_success_rate" in sr_payload, "state_refresh summary missing intervention_success_rate")
    lines.append("check: per-task summary fields -> ok")

    with joint_summary.open("r", encoding="utf-8", newline="") as file:
        joint_rows = list(csv.DictReader(file))
    expect(joint_rows, "joint_summary.csv is empty")
    required_joint_fields = {"family", "variant", "task_name", "agent_name", "episode_count", "valid_rate", "success_rate", "task_type"}
    expect(required_joint_fields.issubset(set(joint_rows[0].keys())), "joint_summary.csv missing common benchmark columns")
    lines.append("check: joint summary common fields -> ok")

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report_path.read_text(encoding="utf-8"), end="")
PY
