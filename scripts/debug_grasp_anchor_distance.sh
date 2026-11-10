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
source "${SCRIPT_DIR}/lib/geomind_debug_task_common.sh"

export TASK_ID="${TASK_ID:-grasp_anchor_distance}"
export TASK_VERSION="${TASK_VERSION:-base}"
geomind_debug_scan_request_args 0 "$@"
geomind_debug_sync_task_version_to_variant
geomind_resolve_task_context "${REPO_ROOT}" "${TASK_ID}"

export TASK_INDEX="${TASK_INDEX:-0}"
export NUM_ENVS="${NUM_ENVS:-1}"
export AGENT="${AGENT:-oracle}"
export AGENT_SEED="${AGENT_SEED:-0}"
export AGENT_EPISODES="${AGENT_EPISODES:-1}"
export RESET_SETTLE_STEPS="${RESET_SETTLE_STEPS:-40}"
export POST_GRASP_HOLD_STEPS="${POST_GRASP_HOLD_STEPS:-8}"
export HEADLESS="${HEADLESS:-0}"
export DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
export DEVICE="${DEVICE:-}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
geomind_default_output_dir "${REPO_ROOT}" "${TIMESTAMP}"
export SUMMARY_JSON_OUT="${SUMMARY_JSON_OUT:-${OUTPUT_DIR}/summary.json}"
export TRAJECTORY_JSON_OUT="${TRAJECTORY_JSON_OUT:-${OUTPUT_DIR}/trajectory.json}"

if [[ "${HEADLESS}" == "1" ]]; then
  export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
fi

geomind_debug_print_core_context "GeoMind Grasp Anchor Distance"

exec bash "${SCRIPT_DIR}/debug_protocol_anchor_reference.sh" "${GEOMIND_FORWARD_ARGS[@]}" "$@"
