#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_geometric_explicit_baselines.sh"

# Minimal explicit geometric runner.
# Supported environment variables:
#   RUN_MODE=single|suite
#   TASK_LABEL=midpoint|extension|symmetry|perfect_midpoint|isometric_extension|point_symmetry|<task_id>|<layout_key>|<layout_key:task_index>
#   AGENT=oracle|random|explicit_midpoint_oracle|explicit_extension_oracle|explicit_random
#   AGENT_SEED=<int>
#   AGENT_EPISODES=<int>
#
# Examples:
#   bash scripts/run_geometric_explicit_minimal.sh
#   RUN_MODE=single TASK_LABEL=midpoint AGENT=oracle bash scripts/run_geometric_explicit_minimal.sh
#   RUN_MODE=single TASK_LABEL=isometric_extension AGENT=explicit_extension_oracle bash scripts/run_geometric_explicit_minimal.sh
#   RUN_MODE=single TASK_LABEL=extension AGENT=random AGENT_EPISODES=5 bash scripts/run_geometric_explicit_minimal.sh
#   RUN_MODE=suite AGENT=oracle bash scripts/run_geometric_explicit_minimal.sh
#   RUN_MODE=suite AGENT=random AGENT_SEED=11 AGENT_EPISODES=3 HEADLESS=1 bash scripts/run_geometric_explicit_minimal.sh

RUN_MODE="${RUN_MODE:-single}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  RUN_MODE="${1}"
  shift
fi

TASK_LABEL="${TASK_LABEL:-midpoint}"
AGENT="${AGENT:-oracle}"
AGENT_SEED="${AGENT_SEED:-0}"
AGENT_EPISODES="${AGENT_EPISODES:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaac_curobo}"
HEADLESS="${HEADLESS:-0}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
EPSILON_POS="${EPSILON_POS:-0.05}"
TAU_LINE="${TAU_LINE:-0.02}"
POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS:-24}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/geometric_explicit_minimal/${TIMESTAMP}}"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "GeoMind Explicit Geometric Minimal Runner"
echo "========================================================================"
echo "run_mode=${RUN_MODE}"
echo "task_label=${TASK_LABEL}"
echo "agent=${AGENT}"
echo "agent_seed=${AGENT_SEED}"
echo "agent_episodes=${AGENT_EPISODES}"
echo "output_dir=${OUTPUT_DIR}"
echo "========================================================================"

RUN_MODE="${RUN_MODE}" \
TASK_LABEL="${TASK_LABEL}" \
AGENT="${AGENT}" \
AGENT_SEED="${AGENT_SEED}" \
AGENT_EPISODES="${AGENT_EPISODES}" \
CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
HEADLESS="${HEADLESS}" \
DISABLE_FABRIC="${DISABLE_FABRIC}" \
EPSILON_POS="${EPSILON_POS}" \
TAU_LINE="${TAU_LINE}" \
POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${BASE_RUNNER}" "$@"
