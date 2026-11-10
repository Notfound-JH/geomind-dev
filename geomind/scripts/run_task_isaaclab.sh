#!/usr/bin/env bash
set -euo pipefail

GEOMIND_DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
if [[ -z "${MDL_SYSTEM_PATH:-}" ]]; then
  if [[ -d "${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}" ]]; then
    export MDL_SYSTEM_PATH="${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}"
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK_ID="${TASK_ID:-SelectByDirection}"
MAX_STEPS="${MAX_STEPS:-100}"
HEADLESS="${HEADLESS:-1}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
VIEWER_EYE="${VIEWER_EYE:-2.2 2.2 1.8}"
VIEWER_LOOKAT="${VIEWER_LOOKAT:-0.3 0.0 0.8}"
INTERACTIVE="${INTERACTIVE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
export OMNI_USER_DIR="${OMNI_USER_DIR:-/tmp/omni-user}"
export OMNI_CACHE_ROOT="${OMNI_CACHE_ROOT:-/tmp/omni-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OMNI_USER_DIR}" "${OMNI_CACHE_ROOT}"

source "${SCRIPT_DIR}/../../scripts/lib/geomind_conda.sh"
geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1

ARGS=(
  --task-id "${TASK_ID}"
  --max-steps "${MAX_STEPS}"
  --backend isaaclab
)

if [[ "${INTERACTIVE}" == "1" ]]; then
  ARGS+=(--interactive)
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

echo "========================================================================"
echo "GeoMind Run Task (IsaacLab)"
echo "========================================================================"
echo "task_id=${TASK_ID}"
echo "max_steps=${MAX_STEPS}"
echo "backend=isaaclab"
echo "interactive=${INTERACTIVE}"
echo "headless=${HEADLESS}"
echo "disable_fabric=${DISABLE_FABRIC}"
echo "enable_cameras=${ENABLE_CAMERAS}"
echo "viewer_eye=${VIEWER_EYE}"
echo "viewer_lookat=${VIEWER_LOOKAT}"
echo "output_dir=${OUTPUT_DIR}"
echo "========================================================================"

cd "${REPO_ROOT}"
HEADLESS="${HEADLESS}" \
DISABLE_FABRIC="${DISABLE_FABRIC}" \
ENABLE_CAMERAS="${ENABLE_CAMERAS}" \
VIEWER_EYE="${VIEWER_EYE}" \
VIEWER_LOOKAT="${VIEWER_LOOKAT}" \
python -m geomind.runners.run_task "${ARGS[@]}" "$@"
