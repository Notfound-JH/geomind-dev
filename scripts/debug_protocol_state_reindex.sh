#!/usr/bin/env bash
set -euo pipefail
GEOMIND_DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
if [[ -z "${MDL_SYSTEM_PATH:-}" ]]; then
  if [[ -d "${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}" ]]; then
    export MDL_SYSTEM_PATH="${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}"
  fi
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
export OMNI_USER_DIR="${OMNI_USER_DIR:-/tmp/omni-user}"
export OMNI_CACHE_ROOT="${OMNI_CACHE_ROOT:-/tmp/omni-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OMNI_USER_DIR}" "${OMNI_CACHE_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run:
#   bash scripts/debug_protocol_state_reindex.sh
#
# Override any of these values inline if needed, for example:
#   TASK_ID=reindex_extreme SCENE_ID=17 TASK_INDEX=0 HEADLESS=1 bash scripts/debug_protocol_state_reindex.sh
TASK_ID="${TASK_ID:-reindex_extreme}"
SCENE_ID="${SCENE_ID:-17}"
LAYOUT_KEY="${LAYOUT_KEY:-}"
TASK_INDEX="${TASK_INDEX:-0}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
PROTOCOL_DEBUG_INTERVAL="${PROTOCOL_DEBUG_INTERVAL:-30}"
POST_INTERVENTION_SETTLE_STEPS="${POST_INTERVENTION_SETTLE_STEPS:-60}"
PROTOCOL_DISPLACEMENT_TOL="${PROTOCOL_DISPLACEMENT_TOL:-}"
PROTOCOL_MIN_REORDER_GAP="${PROTOCOL_MIN_REORDER_GAP:-}"
SCRIPTED_GRASP_QUERY="${SCRIPTED_GRASP_QUERY:-0}"
GRASP_READOUT_WINDOW="${GRASP_READOUT_WINDOW:-12}"
POST_RELEASE_SETTLE_STEPS="${POST_RELEASE_SETTLE_STEPS:-}"

# Set to 1 for headless smoke runs.
HEADLESS="${HEADLESS:-0}"
# With --agent and GUI: after rollout the Python script keeps the viewer open (zero actions) until you
# close the window. Set VIEWER_IDLE_AFTER_AGENT=0 to exit immediately after the agent summary.

# Viewer camera.
VIEWER_EYE=(2.2 2.2 1.8)
VIEWER_LOOKAT=(0.3 0.0 0.8)

source "${SCRIPT_DIR}/lib/geomind_conda.sh"
geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1

ARGS=(
  --task-id "${TASK_ID}"
  --task-index "${TASK_INDEX}"
  --num-envs "${NUM_ENVS}"
  --max-steps "${MAX_STEPS}"
  --protocol-debug-interval "${PROTOCOL_DEBUG_INTERVAL}"
  --post-intervention-settle-steps "${POST_INTERVENTION_SETTLE_STEPS}"
  --viewer-eye "${VIEWER_EYE[@]}"
  --viewer-lookat "${VIEWER_LOOKAT[@]}"
)

if [[ -n "${SCENE_ID}" ]]; then
  ARGS+=(--scene-id "${SCENE_ID}")
fi

if [[ -n "${SCENE:-}" ]]; then
  ARGS+=(--scene "${SCENE}")
fi

if [[ -n "${TASK_VARIANT:-${VARIANT:-}}" ]]; then
  ARGS+=(--variant "${TASK_VARIANT:-${VARIANT:-}}")
fi

if [[ -n "${OBJECT_MODE:-}" ]]; then
  ARGS+=(--object-mode "${OBJECT_MODE}")
fi

if [[ -n "${LAYOUT_KEY}" ]]; then
  ARGS+=(--layout-key "${LAYOUT_KEY}")
fi

if [[ -n "${PROTOCOL_DISPLACEMENT_TOL}" ]]; then
  ARGS+=(--protocol-displacement-tol "${PROTOCOL_DISPLACEMENT_TOL}")
fi

if [[ -n "${PROTOCOL_MIN_REORDER_GAP}" ]]; then
  ARGS+=(--protocol-min-reorder-gap "${PROTOCOL_MIN_REORDER_GAP}")
fi

if [[ "${SCRIPTED_GRASP_QUERY}" == "1" ]]; then
  ARGS+=(--scripted-grasp-query --grasp-readout-window "${GRASP_READOUT_WINDOW}")
fi

if [[ -n "${POST_RELEASE_SETTLE_STEPS}" ]]; then
  ARGS+=(--post-release-settle-steps "${POST_RELEASE_SETTLE_STEPS}")
fi

if [[ "${HEADLESS}" == "1" ]]; then
  ARGS+=(--headless)
  export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
fi

echo "========================================================================"
echo "GeoMind State Reindex Runner"
echo "========================================================================"
echo "task_id=${TASK_ID}"
echo "variant=${TASK_VARIANT:-${VARIANT:-}}"
echo "scene=${SCENE:-}"
echo "scene_id=${SCENE_ID}"
echo "object_mode=${OBJECT_MODE:-}"
echo "layout_key=${LAYOUT_KEY}"
echo "task_index=${TASK_INDEX}"
echo "headless=${HEADLESS}"
echo "headless_camera_video=${HEADLESS_CAMERA_VIDEO:-}"
echo "relocation_motion_mode=${GEOMIND_RELOCATION_MOTION_MODE:-<task-default>}"
echo "========================================================================"

python "${SCRIPT_DIR}/debug_protocol_state_reindex.py" "${ARGS[@]}" "$@"
