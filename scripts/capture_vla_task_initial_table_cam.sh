#!/usr/bin/env bash
set -euo pipefail

GEOMIND_DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
if [[ -z "${MDL_SYSTEM_PATH:-}" ]]; then
  if [[ -d "${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}" ]]; then
    export MDL_SYSTEM_PATH="${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}"
  fi
fi
export GEOMIND_ACTIVE_CAMERAS="table_cam"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TASK_DIR="${TASK_DIR:-${REPO_ROOT}/geomind/configs/vla_tasks}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/vla_task_initial_table_cam}"
RESOLUTION="${RESOLUTION:-1024}"
SETTLE_STEPS="${SETTLE_STEPS:-0}"
LIGHTING_MODE="${LIGHTING_MODE:-natural}"
TASK_ID="${TASK_ID:-}"

source "${SCRIPT_DIR}/lib/geomind_conda.sh"
geomind_activate_conda "${CONDA_ENV:-${CONDA_ENV_NAME:-isaac_curobo}}" || exit 1

ARGS=(
  --task-dir "${TASK_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --resolution "${RESOLUTION}"
  --settle-steps "${SETTLE_STEPS}"
  --lighting-mode "${LIGHTING_MODE}"
)

if [[ -n "${TASK_ID}" ]]; then
  ARGS+=(--task-id "${TASK_ID}")
fi

python "${SCRIPT_DIR}/capture_vla_task_initial_table_cam.py" "${ARGS[@]}" "$@"
