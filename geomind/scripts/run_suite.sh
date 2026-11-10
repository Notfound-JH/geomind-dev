#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GROUP="${GROUP:-base}"
POLICY="${POLICY:-oracle}"
MAX_STEPS="${MAX_STEPS:-100}"
BACKEND="${BACKEND:-isaaclab}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
INTERACTIVE="${INTERACTIVE:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TASK_IDS="${TASK_IDS:-}"
ISAAC_RENDER="${ISAAC_RENDER:-0}"

# IsaacLab rendering switch:
# - ISAAC_RENDER=0 (default): headless mode
# - ISAAC_RENDER=1: open Isaac rendering window for live inspection
if [[ "${BACKEND}" == "isaaclab" ]]; then
  if [[ "${ISAAC_RENDER}" == "1" ]]; then
    export HEADLESS="${HEADLESS:-0}"
    export ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
  else
    export HEADLESS="${HEADLESS:-1}"
    export ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
  fi
fi

if [[ "${BACKEND}" == "isaaclab" && -z "${MDL_SYSTEM_PATH:-}" ]]; then
  DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
  if [[ -d "${DEFAULT_MDL_SYSTEM_PATH}" ]]; then
    export MDL_SYSTEM_PATH="${DEFAULT_MDL_SYSTEM_PATH}"
  fi
fi

ARGS=(
  --group "${GROUP}"
  --policy "${POLICY}"
  --max-steps "${MAX_STEPS}"
  --backend "${BACKEND}"
  --episodes-per-task "${EPISODES_PER_TASK}"
  --seed "${SEED}"
)

if [[ "${INTERACTIVE}" == "1" ]]; then
  ARGS+=(--interactive)
fi
if [[ "${SAVE_VIDEO}" == "1" ]]; then
  ARGS+=(--save-video)
else
  ARGS+=(--no-save-video)
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${TASK_IDS}" ]]; then
  # TASK_IDS format example:
  #   TASK_IDS="SelectByDirection SelectByDistance_v0"
  read -r -a TASK_ID_ARRAY <<< "${TASK_IDS}"
  ARGS+=(--task-ids "${TASK_ID_ARRAY[@]}")
fi

echo "========================================================================"
echo "GeoMind Run Suite"
echo "========================================================================"
echo "group=${GROUP}"
echo "policy=${POLICY}"
echo "max_steps=${MAX_STEPS}"
echo "backend=${BACKEND}"
echo "episodes_per_task=${EPISODES_PER_TASK}"
echo "interactive=${INTERACTIVE}"
echo "save_video=${SAVE_VIDEO}"
echo "seed=${SEED}"
echo "task_ids=${TASK_IDS}"
echo "output_dir=${OUTPUT_DIR}"
echo "isaac_render=${ISAAC_RENDER}"
if [[ "${BACKEND}" == "isaaclab" ]]; then
  echo "headless=${HEADLESS:-}"
  echo "enable_cameras=${ENABLE_CAMERAS:-}"
fi
echo "========================================================================"

cd "${REPO_ROOT}"
python -m geomind.runners.run_suite "${ARGS[@]}" "$@"
