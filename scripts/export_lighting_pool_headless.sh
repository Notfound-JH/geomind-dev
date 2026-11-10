#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCENE_ID="${SCENE_ID:-1}"
RESOLUTION="${RESOLUTION:-1024}"
SETTLE_STEPS="${SETTLE_STEPS:-30}"
BASE_LIGHTING_MODE="${BASE_LIGHTING_MODE:-natural}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/lighting_pool_scene_cam_v6}"

# Avoid remote material/texture failures during preview capture.
export GEOMIND_DISABLE_GROUND_MDL="${GEOMIND_DISABLE_GROUND_MDL:-1}"

cd "${REPO_ROOT}"
python scripts/capture_scene_cam_lighting_pool.py \
  --scene-id "${SCENE_ID}" \
  --resolution "${RESOLUTION}" \
  --settle-steps "${SETTLE_STEPS}" \
  --base-lighting-mode "${BASE_LIGHTING_MODE}" \
  --output-dir "${OUTPUT_DIR}" \
  --headless
