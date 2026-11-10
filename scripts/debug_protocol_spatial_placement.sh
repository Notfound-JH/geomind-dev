#!/usr/bin/env bash
set -euo pipefail
export MDL_SYSTEM_PATH="${MDL_SYSTEM_PATH:-${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Run:
#   bash scripts/debug_protocol_spatial_placement.sh
#
# Override inline if needed, for example:
#   HEADLESS=1 bash scripts/debug_protocol_spatial_placement.sh
#   TASK_ID=place_directional SCENE_ID=10 LAYOUT_KEY=place_directional_scene10_skill bash scripts/debug_protocol_spatial_placement.sh
#   TASK_ID=place_on_top SCENE_ID=65 LAYOUT_KEY=place_on_top_scene65_skill bash scripts/debug_protocol_spatial_placement.sh

TASK_ID="${TASK_ID:-place_directional}"
SCENE_ID="${SCENE_ID:-65}"
LAYOUT_KEY="${LAYOUT_KEY:-place_directional_scene65_base}"
TASK_INDEX="${TASK_INDEX:-0}"
AGENT="${AGENT:-oracle}"
AGENT_EPISODES="${AGENT_EPISODES:-1}"
RESET_SETTLE_STEPS="${RESET_SETTLE_STEPS:-24}"
POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS:-24}"
HEADLESS="${HEADLESS:-0}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
DEVICE="${DEVICE:-}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/spatial_placement_rollouts/${TIMESTAMP}}"
SUMMARY_JSON_OUT="${SUMMARY_JSON_OUT:-${OUTPUT_DIR}/summary.json}"
TRAJECTORY_H5_OUT="${TRAJECTORY_H5_OUT:-}"
TRAJECTORY_SAMPLE_HZ="${TRAJECTORY_SAMPLE_HZ:-10}"
TRAJECTORY_SEGMENT_SOURCE="${TRAJECTORY_SEGMENT_SOURCE:-stage_v1}"

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
  --task-id "${TASK_ID}"
  --scene-id "${SCENE_ID}"
  --layout-key "${LAYOUT_KEY}"
  --task-index "${TASK_INDEX}"
  --agent "${AGENT}"
  --agent-episodes "${AGENT_EPISODES}"
  --reset-settle-steps "${RESET_SETTLE_STEPS}"
  --post-place-settle-steps "${POST_PLACE_SETTLE_STEPS}"
  --summary-json-out "${SUMMARY_JSON_OUT}"
  --trajectory-sample-hz "${TRAJECTORY_SAMPLE_HZ}"
  --trajectory-segment-source "${TRAJECTORY_SEGMENT_SOURCE}"
  --viewer-eye "${VIEWER_EYE[@]}"
  --viewer-lookat "${VIEWER_LOOKAT[@]}"
)

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

if [[ -n "${TRAJECTORY_H5_OUT}" ]]; then
  ARGS+=(--trajectory-h5-out "${TRAJECTORY_H5_OUT}")
fi

if [[ "${HEADLESS}" == "1" ]]; then
  ARGS+=(--headless)
  export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
fi

if [[ "${DISABLE_FABRIC}" == "1" ]]; then
  ARGS+=(--disable-fabric)
fi

echo "========================================================================"
echo "GeoMind Spatial Placement Runner"
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
echo "trajectory_h5=${TRAJECTORY_H5_OUT:-<disabled>}"
echo "trajectory_segment_source=${TRAJECTORY_SEGMENT_SOURCE}"
echo "========================================================================"

python "${SCRIPT_DIR}/debug_protocol_spatial_placement.py" "${ARGS[@]}" "$@"
