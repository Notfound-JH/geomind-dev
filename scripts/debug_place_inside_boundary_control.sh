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
#   bash scripts/debug_place_inside_boundary_control.sh
#
#   TASK_VERSION=control HEADLESS=1 bash scripts/debug_place_inside_boundary_control.sh
#   TASK_VERSION=stress SCENE_ID=17 bash scripts/debug_place_inside_boundary_control.sh

TASK_VERSION="${TASK_VERSION:-control}"
TASK_ID="${TASK_ID:-place_inside_boundary_control}"
TASK_INDEX="${TASK_INDEX:-0}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

case "${TASK_VERSION}" in
  control)
    DEFAULT_SCENE_ID="65"
    DEFAULT_LAYOUT_KEY="place_inside_boundary_control_scene65_control"
    ;;
  stress)
    DEFAULT_SCENE_ID="17"
    DEFAULT_LAYOUT_KEY="place_inside_boundary_control_scene17_stress"
    ;;
  *)
    echo "Unsupported TASK_VERSION=${TASK_VERSION}. Expected 'control' or 'stress'." >&2
    exit 1
    ;;
esac

export TASK_ID
export SCENE_ID="${SCENE_ID:-${DEFAULT_SCENE_ID}}"
export LAYOUT_KEY="${LAYOUT_KEY:-${DEFAULT_LAYOUT_KEY}}"
export TASK_INDEX
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/spatial_placement_rollouts/place_inside_boundary_control/${TASK_VERSION}/${TIMESTAMP}}"

if [[ "${HEADLESS:-0}" == "1" ]]; then
  export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
fi

echo "========================================================================"
echo "GeoMind Place Inside Boundary (paired control)"
echo "========================================================================"
echo "task_id=${TASK_ID}"
echo "task_version=${TASK_VERSION}"
echo "scene_id=${SCENE_ID}"
echo "layout_key=${LAYOUT_KEY}"
echo "task_index=${TASK_INDEX}"
echo "output_dir=${OUTPUT_DIR}"
echo "headless_camera_video=${HEADLESS_CAMERA_VIDEO:-}"
echo "========================================================================"

exec bash "${SCRIPT_DIR}/debug_protocol_spatial_placement.sh" "$@"
