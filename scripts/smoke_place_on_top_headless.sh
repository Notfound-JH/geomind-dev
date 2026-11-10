#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Minimal headless smoke once GPU/CUDA and writable Omni cache issues are resolved.
# Default: skill only.
# Optional: RUN_STRESS=1 bash scripts/smoke_place_on_top_headless.sh

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
HEADLESS="${HEADLESS:-1}"
DISABLE_FABRIC="${DISABLE_FABRIC:-1}"
AGENT_EPISODES="${AGENT_EPISODES:-1}"
RESET_SETTLE_STEPS="${RESET_SETTLE_STEPS:-24}"
POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS:-24}"
RUN_STRESS="${RUN_STRESS:-0}"

run_case() {
  local scene_id="$1"
  local layout_key="$2"
  local label="$3"
  local output_dir="${REPO_ROOT}/logs/spatial_placement_rollouts/place_on_top/smoke/${label}/${TIMESTAMP}"

  echo "========================================================================"
  echo "place_on_top smoke: ${label}"
  echo "scene_id=${scene_id}"
  echo "layout_key=${layout_key}"
  echo "output_dir=${output_dir}"
  echo "========================================================================"

  TASK_ID=place_on_top \
  SCENE_ID="${scene_id}" \
  LAYOUT_KEY="${layout_key}" \
  AGENT_EPISODES="${AGENT_EPISODES}" \
  RESET_SETTLE_STEPS="${RESET_SETTLE_STEPS}" \
  POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS}" \
  HEADLESS="${HEADLESS}" \
  DISABLE_FABRIC="${DISABLE_FABRIC}" \
  OUTPUT_DIR="${output_dir}" \
  bash "${SCRIPT_DIR}/debug_protocol_spatial_placement.sh"
}

run_case 65 place_on_top_scene65_skill skill

if [[ "${RUN_STRESS}" == "1" ]]; then
  run_case 17 place_on_top_scene17_stress stress
fi
