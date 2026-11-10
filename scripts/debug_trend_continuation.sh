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

export TASK_ID="${TASK_ID:-trend_continuation}"
export PRESET="${PRESET:-standard}"
geomind_debug_scan_request_args 1 "$@"
if [[ -z "${GEOMIND_REQUEST_VARIANT:-}" ]]; then
  geomind_debug_sync_task_version_to_variant
fi
if [[ -z "${GEOMIND_REQUEST_VARIANT:-}" && -z "${TASK_VARIANT:-}" && -z "${VARIANT:-}" ]]; then
  geomind_debug_apply_preset_as_variant
fi
geomind_resolve_task_context "${REPO_ROOT}" "${TASK_ID}"

export TASK_INDEX="${TASK_INDEX:-0}"
export AGENT="${AGENT:-implicit_trend_oracle}"
export AGENT_SEED="${AGENT_SEED:-0}"
export AGENT_EPISODES="${AGENT_EPISODES:-1}"
export EPSILON_POS="${EPSILON_POS:-0.05}"
export TAU_LINE="${TAU_LINE:-0.02}"
export TAU_TREND="${TAU_TREND:-0.03}"
export TAU_YAW="${TAU_YAW:-0.35}"
export POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS:-24}"
export HEADLESS="${HEADLESS:-0}"
export DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
geomind_default_output_dir "${REPO_ROOT}" "${TIMESTAMP}"

if [[ "${HEADLESS}" == "1" ]]; then
  export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
fi

geomind_debug_print_core_context "GeoMind Trend Continuation"

exec bash "${SCRIPT_DIR}/debug_protocol_geometric_inference.sh" "${GEOMIND_FORWARD_ARGS[@]}" "$@"
