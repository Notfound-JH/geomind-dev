#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK_ID_ENV="${TASK_ID_ENV:-SelectByDirection}"
BACKEND="${BACKEND:-isaaclab}"
MAX_STEPS="${MAX_STEPS:-100}"
EPISODES="${EPISODES:-5}"
SEED="${SEED:-0}"
ACTION_NOISE_SCALE="${ACTION_NOISE_SCALE:-0.05}"
OBJECT_PERTURB_RANGE="${OBJECT_PERTURB_RANGE:-0.02}"
NATIVE_OBJECT_POSE_PERTURB_RANGE="${NATIVE_OBJECT_POSE_PERTURB_RANGE:-0.0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
ISAAC_RENDER="${ISAAC_RENDER:-}"

if [[ "${BACKEND}" == "isaaclab" ]]; then
  if [[ -z "${MDL_SYSTEM_PATH:-}" ]]; then
    DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
    if [[ -d "${DEFAULT_MDL_SYSTEM_PATH}" ]]; then
      export MDL_SYSTEM_PATH="${DEFAULT_MDL_SYSTEM_PATH}"
    fi
  fi

  source "${SCRIPT_DIR}/../../scripts/lib/geomind_conda.sh"
  geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1

  if ! python -c "import isaacsim" >/dev/null 2>&1; then
    echo "[collect_data] ERROR: current python cannot import 'isaacsim'." >&2
    echo "[collect_data] Hint: activate Isaac environment first, e.g. CONDA_ENV_NAME=isaac_curobo." >&2
    echo "[collect_data] Current python: $(command -v python || true)" >&2
    exit 1
  fi
fi

# IsaacLab rendering switch for data collection:
# - ISAAC_RENDER=0: headless mode
# - ISAAC_RENDER=1: open Isaac rendering window for live inspection
# - unset: keep externally provided HEADLESS/ENABLE_CAMERAS behavior
if [[ "${BACKEND}" == "isaaclab" && -n "${ISAAC_RENDER}" ]]; then
  if [[ "${ISAAC_RENDER}" == "1" ]]; then
    export HEADLESS="${HEADLESS:-0}"
    export ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
  else
    export HEADLESS="${HEADLESS:-1}"
    export ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
  fi
fi

ARGS=(
  --task-id-env "${TASK_ID_ENV}"
  --backend "${BACKEND}"
  --max-steps "${MAX_STEPS}"
  --episodes "${EPISODES}"
  --seed "${SEED}"
  --action-noise-scale "${ACTION_NOISE_SCALE}"
  --object-perturb-range "${OBJECT_PERTURB_RANGE}"
  --native-object-pose-perturb-range "${NATIVE_OBJECT_POSE_PERTURB_RANGE}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

echo "========================================================================"
echo "GeoMind Data Collection"
echo "========================================================================"
echo "task_id_env=${TASK_ID_ENV}"
echo "backend=${BACKEND}"
echo "max_steps=${MAX_STEPS}"
echo "episodes=${EPISODES}"
echo "seed=${SEED}"
echo "action_noise_scale=${ACTION_NOISE_SCALE}"
echo "object_perturb_range=${OBJECT_PERTURB_RANGE}"
echo "native_object_pose_perturb_range=${NATIVE_OBJECT_POSE_PERTURB_RANGE}"
echo "output_dir=${OUTPUT_DIR}"
echo "isaac_render=${ISAAC_RENDER}"
if [[ "${BACKEND}" == "isaaclab" ]]; then
  echo "conda_env_name=${CONDA_ENV_NAME:-isaac_curobo}"
  echo "python_bin=$(command -v python || true)"
  echo "mdl_system_path=${MDL_SYSTEM_PATH:-}"
  echo "headless=${HEADLESS:-}"
  echo "enable_cameras=${ENABLE_CAMERAS:-}"
fi
echo "========================================================================"

cd "${REPO_ROOT}"
python -m geomind.runners.collect_data "${ARGS[@]}" "$@"
