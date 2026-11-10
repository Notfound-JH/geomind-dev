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
#   bash scripts/debug_reindex_subset_control.sh
#
#   PRESET=standard HEADLESS=1 bash scripts/debug_reindex_subset_control.sh
#   PRESET=stress SCENE_ID=10 bash scripts/debug_reindex_subset_control.sh

PRESET="${PRESET:-standard}"
TASK_ID="${TASK_ID:-reindex_subset_control}"
TASK_INDEX="${TASK_INDEX:-0}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

case "${PRESET}" in
  standard)
    DEFAULT_SCENE_ID="65"
    DEFAULT_LAYOUT_KEY="reindex_subset_control_scene65_control"
    PRESET_NAME="standard"
    ;;
  stress)
    DEFAULT_SCENE_ID="10"
    DEFAULT_LAYOUT_KEY="reindex_subset_control_scene10_stress"
    PRESET_NAME="stress"
    ;;
  *)
    echo "Unsupported PRESET=${PRESET}. Expected 'standard' or 'stress'." >&2
    exit 1
    ;;
esac

export TASK_ID
export SCENE_ID="${SCENE_ID:-${DEFAULT_SCENE_ID}}"
export LAYOUT_KEY="${LAYOUT_KEY:-${DEFAULT_LAYOUT_KEY}}"
export TASK_INDEX
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/state_reindex_rollouts/reindex_subset_control/${PRESET_NAME}/${TIMESTAMP}}"

if [[ "${HEADLESS:-0}" == "1" ]]; then
  export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
fi

echo "========================================================================"
echo "GeoMind Reindex Subset (paired control)"
echo "========================================================================"
echo "task_id=${TASK_ID}"
echo "preset=${PRESET}"
echo "scene_id=${SCENE_ID}"
echo "layout_key=${LAYOUT_KEY}"
echo "task_index=${TASK_INDEX}"
echo "output_dir=${OUTPUT_DIR}"
echo "headless_camera_video=${HEADLESS_CAMERA_VIDEO:-}"
echo "========================================================================"

exec bash "${SCRIPT_DIR}/debug_protocol_state_reindex.sh" "$@"
