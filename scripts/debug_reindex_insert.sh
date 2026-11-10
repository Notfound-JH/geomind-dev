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

DEFAULT_TASK_ID="reindex_insert"
export PRESET="${PRESET:-standard}"
geomind_debug_scan_request_args 1 "$@"
if [[ -z "${GEOMIND_REQUEST_VARIANT:-}" ]]; then
  geomind_debug_sync_task_version_to_variant
fi
if [[ -z "${GEOMIND_REQUEST_VARIANT:-}" && -z "${TASK_VARIANT:-}" && -z "${VARIANT:-}" ]]; then
  geomind_debug_apply_preset_as_variant
fi
geomind_resolve_task_context "${REPO_ROOT}" "${DEFAULT_TASK_ID}"

export TASK_INDEX="${TASK_INDEX:-0}"
AGENT="${AGENT:-oracle}"
AGENT_SEED="${AGENT_SEED:-0}"
# Align with debug_reindex_extreme default rollout behavior.
AGENT_EPISODES="${AGENT_EPISODES:-1}"
# Insert debug defaults: enable smooth tabletop sliding while preserving authored object colors.
# GT and task instruction are preserved; only runtime motion/physics are patched by default.
export GEOMIND_INSERT_DEBUG="${GEOMIND_INSERT_DEBUG:-1}"
export GEOMIND_INSERT_COLOR_TAGS="${GEOMIND_INSERT_COLOR_TAGS:-0}"
# The insert slide is intentionally short; this leaves enough margin for the smooth path to finish
# before the agent selects/grabs without making the debug wrapper feel sluggish.
export POST_INTERVENTION_SETTLE_STEPS="${POST_INTERVENTION_SETTLE_STEPS:-80}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
SUMMARY_JSON_OUT="${SUMMARY_JSON_OUT:-}"
CONTROL_MODE="${CONTROL_MODE:-}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
geomind_default_output_dir "${REPO_ROOT}" "${TIMESTAMP}"

if [[ "${HEADLESS:-0}" == "1" ]]; then
  export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
fi

geomind_debug_print_core_context "GeoMind Reindex Insert"
echo "insert_debug=${GEOMIND_INSERT_DEBUG}"
echo "insert_color_tags=${GEOMIND_INSERT_COLOR_TAGS}"
echo "post_intervention_settle_steps=${POST_INTERVENTION_SETTLE_STEPS}"
echo "========================================================================"

runner_args=(
  --agent "${AGENT}"
  --agent-seed "${AGENT_SEED}"
  --agent-episodes "${AGENT_EPISODES}"
)
[[ "${SUMMARY_ONLY}" == "1" ]] && runner_args+=(--summary-only)
[[ -n "${SUMMARY_JSON_OUT}" ]] && runner_args+=(--summary-json-out "${SUMMARY_JSON_OUT}")
[[ -n "${CONTROL_MODE}" ]] && runner_args+=(--control-mode "${CONTROL_MODE}")
[[ "${DISABLE_FABRIC}" == "1" ]] && runner_args+=(--disable-fabric)

exec bash "${SCRIPT_DIR}/debug_protocol_state_reindex.sh" "${runner_args[@]}" "${GEOMIND_FORWARD_ARGS[@]}"
