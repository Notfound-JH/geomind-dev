#!/usr/bin/env bash
set -euo pipefail

GEOMIND_DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
if [[ -z "${MDL_SYSTEM_PATH:-}" ]]; then
  if [[ -d "${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}" ]]; then
    export MDL_SYSTEM_PATH="${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}"
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Run:
#   bash scripts/run_place_on_top.sh
#   MODE=stress bash scripts/run_place_on_top.sh
#   MODE=both bash scripts/run_place_on_top.sh

traceback() {
  local exit_code="$1"
  local line_no="$2"
  local failed_command="$3"
  echo "========================================================================" >&2
  echo "run_place_on_top.sh failure" >&2
  echo "========================================================================" >&2
  echo "exit_code=${exit_code}" >&2
  echo "line=${line_no}" >&2
  echo "command=${failed_command}" >&2
  echo "mode=${MODE:-}" >&2
  echo "task_version=${TASK_VERSION:-}" >&2
  echo "scene_id=${SCENE_ID:-}" >&2
  echo "layout_key=${LAYOUT_KEY:-}" >&2
  echo "headless=${HEADLESS:-}" >&2
  echo "disable_fabric=${DISABLE_FABRIC:-}" >&2
  echo "reset_settle_steps=${RESET_SETTLE_STEPS:-}" >&2
  echo "post_place_settle_steps=${POST_PLACE_SETTLE_STEPS:-}" >&2
  echo "call_stack:" >&2
  local frame=0
  while caller "${frame}" >&2; do
    frame=$((frame + 1))
  done
  echo "========================================================================" >&2
}

trap 'traceback $? ${LINENO} "$BASH_COMMAND"' ERR

MODE="${MODE:-skill}"          # skill | stress | both
HEADLESS="${HEADLESS:-0}"
DISABLE_FABRIC="${DISABLE_FABRIC:-1}"
AGENT_EPISODES="${AGENT_EPISODES:-1}"
RESET_SETTLE_STEPS="${RESET_SETTLE_STEPS:-40}"
POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS:-24}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

run_case() {
  local label="$1"
  local scene_id="$2"
  local layout_key="$3"
  local task_version="$4"
  shift 4
  local extra_args=("$@")
  local output_dir="${REPO_ROOT}/logs/spatial_placement_rollouts/place_on_top/${label}/${TIMESTAMP}"

  echo "========================================================================"
  echo "place_on_top run: ${label}"
  echo "task_version=${task_version}"
  echo "scene_id=${scene_id}"
  echo "layout_key=${layout_key}"
  echo "headless=${HEADLESS}"
  echo "disable_fabric=${DISABLE_FABRIC}"
  echo "reset_settle_steps=${RESET_SETTLE_STEPS}"
  echo "post_place_settle_steps=${POST_PLACE_SETTLE_STEPS}"
  echo "output_dir=${output_dir}"
  echo "========================================================================"

  TASK_VERSION="${task_version}" \
  SCENE_ID="${scene_id}" \
  LAYOUT_KEY="${layout_key}" \
  TASK_INDEX=0 \
  AGENT_EPISODES="${AGENT_EPISODES}" \
  RESET_SETTLE_STEPS="${RESET_SETTLE_STEPS}" \
  POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS}" \
  HEADLESS="${HEADLESS}" \
  DISABLE_FABRIC="${DISABLE_FABRIC}" \
  OUTPUT_DIR="${output_dir}" \
  bash "${SCRIPT_DIR}/debug_place_on_top.sh" "${extra_args[@]}"
}

case "${MODE}" in
  skill)
    run_case skill 65 place_on_top_scene65_skill skill "$@"
    ;;
  stress)
    run_case stress 17 place_on_top_scene17_stress stress "$@"
    ;;
  both)
    run_case skill 65 place_on_top_scene65_skill skill "$@"
    run_case stress 17 place_on_top_scene17_stress stress "$@"
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Expected skill, stress, or both." >&2
    exit 1
    ;;
esac
