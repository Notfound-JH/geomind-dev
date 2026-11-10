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
#   bash scripts/debug_protocol_constraint_selection.sh
#
# Override inline if needed, for example:
#   HEADLESS=1 bash scripts/debug_protocol_constraint_selection.sh
#   AGENT=random AGENT_EPISODES=5 bash scripts/debug_protocol_constraint_selection.sh
#   LAYOUT_KEY=select_visual_spatial_scene17_stress SCENE_ID=17 bash scripts/debug_protocol_constraint_selection.sh
#   SUMMARY_JSON_OUT=/tmp/constraint.summary.json TRAJECTORY_JSON_OUT=/tmp/constraint.trajectory.json bash scripts/debug_protocol_constraint_selection.sh

TASK_ID="${TASK_ID:-select_visual_spatial}"
SCENE_ID="${SCENE_ID:-65}"
LAYOUT_KEY="${LAYOUT_KEY:-select_visual_spatial_scene65_base}"
TASK_INDEX="${TASK_INDEX:-0}"
NUM_ENVS="${NUM_ENVS:-1}"
AGENT="${AGENT:-oracle}"
AGENT_SEED="${AGENT_SEED:-0}"
AGENT_EPISODES="${AGENT_EPISODES:-1}"
RESET_SETTLE_STEPS="${RESET_SETTLE_STEPS:-40}"
POST_GRASP_HOLD_STEPS="${POST_GRASP_HOLD_STEPS:-8}"
HEADLESS="${HEADLESS:-0}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
DEVICE="${DEVICE:-}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/constraint_selection_rollouts/${TIMESTAMP}}"
SUMMARY_JSON_OUT="${SUMMARY_JSON_OUT:-${OUTPUT_DIR}/summary.json}"
TRAJECTORY_JSON_OUT="${TRAJECTORY_JSON_OUT:-${OUTPUT_DIR}/trajectory.json}"

VIEWER_EYE=(${VIEWER_EYE:-2.2 2.2 1.8})
VIEWER_LOOKAT=(${VIEWER_LOOKAT:-0.3 0.0 0.8})

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
export OMNI_USER_DIR="${OMNI_USER_DIR:-/tmp/omni-user}"
export OMNI_CACHE_ROOT="${OMNI_CACHE_ROOT:-/tmp/omni-cache}"
mkdir -p "${OUTPUT_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OMNI_USER_DIR}" "${OMNI_CACHE_ROOT}"

source "${SCRIPT_DIR}/lib/geomind_conda.sh"
geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1

ARGS=(
  --num-envs "${NUM_ENVS}"
  --agent "${AGENT}"
  --agent-seed "${AGENT_SEED}"
  --agent-episodes "${AGENT_EPISODES}"
  --reset-settle-steps "${RESET_SETTLE_STEPS}"
  --post-grasp-hold-steps "${POST_GRASP_HOLD_STEPS}"
  --summary-json-out "${SUMMARY_JSON_OUT}"
  --trajectory-json-out "${TRAJECTORY_JSON_OUT}"
  --viewer-eye "${VIEWER_EYE[@]}"
  --viewer-lookat "${VIEWER_LOOKAT[@]}"
)

if [[ -n "${TASK_ID}" ]]; then
  ARGS+=(--task-id "${TASK_ID}")
fi

if [[ -n "${SCENE_ID}" ]]; then
  ARGS+=(--scene-id "${SCENE_ID}")
fi

if [[ -n "${LAYOUT_KEY}" ]]; then
  ARGS+=(--layout-key "${LAYOUT_KEY}")
fi

if [[ -n "${TASK_INDEX}" ]]; then
  ARGS+=(--task-index "${TASK_INDEX}")
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

if [[ -n "${DEVICE}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi

if [[ "${HEADLESS}" == "1" ]]; then
  ARGS+=(--headless)
  export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
fi

if [[ "${DISABLE_FABRIC}" == "1" ]]; then
  ARGS+=(--disable-fabric)
fi

echo "========================================================================"
echo "GeoMind Constraint Selection Runner"
echo "========================================================================"
echo "task_id=${TASK_ID}"
echo "variant=${TASK_VARIANT:-${VARIANT:-}}"
echo "scene=${SCENE:-}"
echo "scene_id=${SCENE_ID}"
echo "object_mode=${OBJECT_MODE:-}"
echo "layout_key=${LAYOUT_KEY}"
echo "task_index=${TASK_INDEX}"
echo "agent=${AGENT}"
echo "agent_episodes=${AGENT_EPISODES}"
echo "headless=${HEADLESS}"
echo "headless_camera_video=${HEADLESS_CAMERA_VIDEO:-}"
echo "summary_json=${SUMMARY_JSON_OUT}"
echo "trajectory_json=${TRAJECTORY_JSON_OUT}"
echo "========================================================================"

python "${SCRIPT_DIR}/debug_protocol_constraint_selection.py" "${ARGS[@]}" "$@"
