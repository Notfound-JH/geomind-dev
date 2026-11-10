#!/usr/bin/env bash
set -euo pipefail
GEOMIND_DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
if [[ -z "${MDL_SYSTEM_PATH:-}" ]]; then
  if [[ -d "${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}" ]]; then
    export MDL_SYSTEM_PATH="${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}"
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Leave empty to capture all scenes.
# Examples:
#   SCENE_ID="19"
#   SCENE_ID="[1, 19, 27]"
SCENE_ID="1"
SETTLE_STEPS=60
RESOLUTION=1024
LIGHTING_MODE="natural"
OUTPUT_DIR="${SCRIPT_DIR}/../logs/scene_check_images_id_1"

source "${SCRIPT_DIR}/lib/geomind_conda.sh"
geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1

ARGS=(
  --settle-steps "${SETTLE_STEPS}"
  --resolution "${RESOLUTION}"
  --lighting-mode "${LIGHTING_MODE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${SCENE_ID}" ]]; then
  ARGS+=(--scene-id "${SCENE_ID}")
fi

python "${SCRIPT_DIR}/capture_geomind_scene_check_images.py" "${ARGS[@]}" "$@"
