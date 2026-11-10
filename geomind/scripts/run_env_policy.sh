#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GEOMIND_DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
if [[ -z "${MDL_SYSTEM_PATH:-}" ]]; then
  if [[ -d "${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}" ]]; then
    export MDL_SYSTEM_PATH="${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}"
  fi
fi

TASK_ID_ENV="${TASK_ID_ENV:-SelectByDirection}"
POLICY="${POLICY:-oracle}"
MAX_STEPS="${MAX_STEPS:-100}"
BACKEND="${BACKEND:-isaaclab}"
SEED="${SEED:-0}"
INTERACTIVE="${INTERACTIVE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

ARGS=(
  --task-id-env "${TASK_ID_ENV}"
  --policy "${POLICY}"
  --max-steps "${MAX_STEPS}"
  --backend "${BACKEND}"
  --seed "${SEED}"
)

if [[ "${INTERACTIVE}" == "1" ]]; then
  ARGS+=(--interactive)
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

echo "========================================================================"
echo "GeoMind Env-Policy Runner"
echo "========================================================================"
echo "task_id_env=${TASK_ID_ENV}"
echo "policy=${POLICY}"
echo "max_steps=${MAX_STEPS}"
echo "backend=${BACKEND}"
echo "seed=${SEED}"
echo "interactive=${INTERACTIVE}"
echo "output_dir=${OUTPUT_DIR}"
echo "mdl_system_path=${MDL_SYSTEM_PATH:-}"
echo "========================================================================"

cd "${REPO_ROOT}"
python -m geomind.runners.run_env_policy "${ARGS[@]}" "$@"
