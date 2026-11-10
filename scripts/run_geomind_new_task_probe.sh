#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_geomind_new_task_probe.sh [task_id ...] [options]

Default task_ids:
  square_completion
  place_directional

Purpose:
  Exercise the current GeoMind new-task path through metadata, suite selection,
  runner dispatch, and benchmark reporting. If a selected task already has an
  implemented runner family, the script will launch that backend as well.

Options:
  --suite NAME              phase1 | main | phase2-reserve. Default: phase1
  --output DIR              Override output directory.
  --task-version VALUES     Space- or comma-separated version filter.
  --phase VALUES            Space- or comma-separated phase filter.
  --agents VALUES           Optional agent override.
  --strict                  Use strict dispatch mode.
  --permissive              Use permissive dispatch mode. Default.
  --dry-run                 Resolve and report without launching child runners.
  --headless                Export HEADLESS=1. Default.
  --disable-fabric          Export DISABLE_FABRIC=1.
  -h, --help                Show this help.

Examples:
  bash scripts/run_geomind_new_task_probe.sh
  bash scripts/run_geomind_new_task_probe.sh square_completion --task-version skill
  bash scripts/run_geomind_new_task_probe.sh place_directional square_completion --strict
  bash scripts/run_geomind_new_task_probe.sh reindex_extreme perfect_midpoint
EOF
}

normalize_list_arg() {
  local raw="$1"
  raw="${raw//,/ }"
  echo "${raw}"
}

suite="phase1"
output_dir=""
runner_mode="permissive"
dry_run="0"
headless="1"
disable_fabric=""
task_version_filter=""
phase_filter=""
agent_names_override=""
declare -a task_ids=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      suite="${2:-}"
      shift 2
      ;;
    --output)
      output_dir="${2:-}"
      shift 2
      ;;
    --task-version)
      task_version_filter="$(normalize_list_arg "${2:-}")"
      shift 2
      ;;
    --phase)
      phase_filter="$(normalize_list_arg "${2:-}")"
      shift 2
      ;;
    --agents)
      agent_names_override="$(normalize_list_arg "${2:-}")"
      shift 2
      ;;
    --strict)
      runner_mode="strict"
      shift
      ;;
    --permissive)
      runner_mode="permissive"
      shift
      ;;
    --dry-run)
      dry_run="1"
      shift
      ;;
    --headless)
      headless="1"
      shift
      ;;
    --disable-fabric)
      disable_fabric="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      task_ids+=("$1")
      shift
      ;;
  esac
done

if [[ ${#task_ids[@]} -eq 0 ]]; then
  task_ids=("square_completion" "place_directional")
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${output_dir}" ]]; then
  output_dir="${REPO_ROOT}/logs/geomind_benchmark/${timestamp}_geomind_new_task_probe"
fi

task_arg="$(printf '%s ' "${task_ids[@]}")"
task_arg="${task_arg% }"

echo "probe_output=${output_dir}"
echo "suite=${suite}"
echo "runner_mode=${runner_mode}"
echo "dry_run=${dry_run}"
echo "headless=${headless}"
echo "task_ids=${task_arg}"
if [[ -n "${task_version_filter}" ]]; then
  echo "task_version_filter=${task_version_filter}"
fi
if [[ -n "${phase_filter}" ]]; then
  echo "phase_filter=${phase_filter}"
fi
if [[ -n "${agent_names_override}" ]]; then
  echo "agents_override=${agent_names_override}"
fi

suite_args=(
  "${suite}"
  --runner-mode "${runner_mode}"
  --tasks "${task_arg}"
  --output "${output_dir}"
)

if [[ "${dry_run}" == "1" ]]; then
  suite_args+=(--dry-run)
fi
if [[ -n "${task_version_filter}" ]]; then
  suite_args+=(--task-version "${task_version_filter}")
fi
if [[ -n "${phase_filter}" ]]; then
  suite_args+=(--phase "${phase_filter}")
fi
if [[ -n "${agent_names_override}" ]]; then
  suite_args+=(--agents "${agent_names_override}")
fi
if [[ "${headless}" == "1" ]]; then
  suite_args+=(--headless)
fi
if [[ -n "${disable_fabric}" ]]; then
  suite_args+=(--disable-fabric)
fi

env \
  AGENT_EPISODES=1 \
  STATE_REFRESH_AGENT_EPISODES=1 \
  GEOMETRIC_STRUCTURE_AGENT_EPISODES=1 \
  bash "${SCRIPT_DIR}/run_geomind_suite.sh" "${suite_args[@]}"

report_path="${output_dir}/runner_dispatch_report.json"
info_path="${output_dir}/benchmark_info.json"

if [[ -f "${report_path}" ]]; then
  python - "${report_path}" "${info_path}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
info_path = Path(sys.argv[2])

report = json.loads(report_path.read_text(encoding="utf-8"))
info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}

selected = list(report.get("selected_task_descriptors", []))
runnable = list(report.get("runnable_task_descriptors", []))
skipped = list(report.get("skipped_task_descriptors", []))

print("=" * 72)
print("new_task_probe_summary")
print("=" * 72)
print(f"dispatch_mode:  {report.get('runner_dispatch_mode', '')}")
print(f"selected_count: {len(selected)}")
print(f"runnable_count: {len(runnable)}")
print(f"skipped_count:  {len(skipped)}")
print(f"output_dir:     {info.get('output_dir', report_path.parent)}")

for entry in selected:
    task_id = str(entry.get("task_id", ""))
    task_version = str(entry.get("task_version", ""))
    phase = str(entry.get("phase", ""))
    runner_family = str(entry.get("runner_family", ""))
    dispatch_status = str(entry.get("dispatch_status", ""))
    reason = str(entry.get("dispatch_reason", ""))
    print("-" * 72)
    print(f"task_id:         {task_id}")
    print(f"task_version:    {task_version}")
    print(f"phase:           {phase}")
    print(f"runner_family:   {runner_family}")
    print(f"dispatch_status: {dispatch_status}")
    if reason:
        print(f"dispatch_reason: {reason}")
PY
fi

echo "benchmark_info=${info_path}"
echo "dispatch_report=${report_path}"
