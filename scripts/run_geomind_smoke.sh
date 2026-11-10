#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_geomind_smoke.sh [--output DIR] [--permissive]

Runs a minimal real benchmark smoke with:
  - one runnable state-refresh family task
  - one runnable geometric-structure family task
  - one agent per family
  - one episode per agent

Defaults:
  runner mode: strict
  headless:    1

Environment overrides:
  STATE_REFRESH_SMOKE_TASK      Default: reindex_extreme
  GEOMETRIC_STRUCTURE_SMOKE_TASK      Default: perfect_midpoint
  STATE_REFRESH_SMOKE_AGENT     Default: state_refresh_core_oracle
  GEOMETRIC_STRUCTURE_SMOKE_AGENT     Default: explicit_midpoint_oracle
  CONDA_ENV_NAME     Default: isaac_curobo
  ANNOTATIONS_JSON   Optional custom annotations file
EOF
}

output_dir=""
runner_mode="strict"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      output_dir="${2:-}"
      shift 2
      ;;
    --permissive)
      runner_mode="permissive"
      shift
      ;;
    --strict)
      runner_mode="strict"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${output_dir}" ]]; then
  output_dir="${REPO_ROOT}/logs/geomind_benchmark/${timestamp}_geomind_smoke"
fi

state_refresh_task="${STATE_REFRESH_SMOKE_TASK:-reindex_extreme}"
geometric_structure_task="${GEOMETRIC_STRUCTURE_SMOKE_TASK:-perfect_midpoint}"
state_refresh_agent="${STATE_REFRESH_SMOKE_AGENT:-state_refresh_core_oracle}"
geometric_structure_agent="${GEOMETRIC_STRUCTURE_SMOKE_AGENT:-explicit_midpoint_oracle}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
export OMNI_USER_DIR="${OMNI_USER_DIR:-/tmp/omni-user}"
export OMNI_CACHE_ROOT="${OMNI_CACHE_ROOT:-/tmp/omni-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OMNI_USER_DIR}" "${OMNI_CACHE_ROOT}"

echo "smoke_output=${output_dir}"
echo "runner_mode=${runner_mode}"
echo "state_refresh_task=${state_refresh_task}"
echo "geometric_structure_task=${geometric_structure_task}"
echo "state_refresh_agent=${state_refresh_agent}"
echo "geometric_structure_agent=${geometric_structure_agent}"

env \
  AGENT_EPISODES=1 \
  STATE_REFRESH_AGENT_EPISODES=1 \
  GEOMETRIC_STRUCTURE_AGENT_EPISODES=1 \
  HEADLESS=1 \
  bash "${SCRIPT_DIR}/run_geomind_suite.sh" phase1 \
    --runner-mode "${runner_mode}" \
    --tasks "${state_refresh_task} ${geometric_structure_task}" \
    --agents "${state_refresh_agent} ${geometric_structure_agent}" \
    --output "${output_dir}"
