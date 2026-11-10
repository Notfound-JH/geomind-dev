#!/usr/bin/env bash
# Remote-policy rollout against GeoMind *VLA* task configs (geomind/configs/vla_tasks/*.json).
# Same runner as run_remote_policy_isaaclab.sh; defaults point at VLA task ids + separate output tree.
#
# Example:
#   MAX_STEPS=200 TASK_ID_ENV=SelectByDirectionVla_v0 bash geomind/scripts/run_remote_policy_isaaclab_vla.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default: preloaded geometric VLA smoke test; its target_object starts in the gripper.
TASK_ID_ENV="${TASK_ID_ENV:-ExtendEqualSpacingVla_v0}"
POLICY="${POLICY:-model}"
MAX_STEPS="${MAX_STEPS:-1}"
BACKEND="${BACKEND:-isaaclab}"
SEED="${SEED:-0}"
ABLATION_ID="${ABLATION_ID:-base}"
PAIR_ID="${PAIR_ID:-}"
NATIVE_OBJECT_POSE_PERTURB_RANGE="${NATIVE_OBJECT_POSE_PERTURB_RANGE:-0}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaac_curobo}"

VLA_EVAL_CONFIG_DIR="${VLA_EVAL_CONFIG_DIR:-${REPO_ROOT}/geomind/configs/vla_eval_models}"
REMOTE_POLICY_PORT_MODEL_MAP="${REMOTE_POLICY_PORT_MODEL_MAP:-8000=spatialforcing,8001=spatialvla,8002=internvla}"
eval_config_value() {
  local config_path="$1"
  local key="$2"
  python - "${config_path}" "${key}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.is_file():
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
value = payload.get(key, "")
if value is None:
    value = ""
print(value)
PY
}

remote_policy_model_from_url() {
  local url="$1"
  local port=""
  if [[ "${url}" =~ :([0-9]+)(/|$) ]]; then
    port="${BASH_REMATCH[1]}"
  fi
  local pair
  for pair in ${REMOTE_POLICY_PORT_MODEL_MAP//,/ }; do
    if [[ "${pair}" == "${port}="* ]]; then
      echo "${pair#*=}"
      return
    fi
  done
  case "${port}" in
    8000) echo "spatialforcing" ;;
    8001) echo "spatialvla" ;;
    8002) echo "internvla" ;;
    *) echo "remote_${port:-unknown}" ;;
  esac
}
REMOTE_POLICY_MODEL="${VLA_EVAL_MODEL:-${REMOTE_POLICY_MODEL:-}}"
REMOTE_POLICY_URL="${REMOTE_POLICY_URL:-}"
if [[ -z "${REMOTE_POLICY_MODEL}" && -n "${REMOTE_POLICY_URL}" ]]; then
  REMOTE_POLICY_MODEL="$(remote_policy_model_from_url "${REMOTE_POLICY_URL}")"
fi
REMOTE_POLICY_MODEL="${REMOTE_POLICY_MODEL:-spatialvla}"
VLA_EVAL_CONFIG_PATH="${VLA_EVAL_CONFIG_PATH:-${VLA_EVAL_CONFIG_DIR}/${REMOTE_POLICY_MODEL}.json}"
CONFIG_REMOTE_POLICY_URL="$(eval_config_value "${VLA_EVAL_CONFIG_PATH}" "remote_policy_url")"
CONFIG_REMOTE_POLICY_ACTION_SCALE="$(eval_config_value "${VLA_EVAL_CONFIG_PATH}" "remote_policy_action_scale")"
CONFIG_REMOTE_POLICY_DISABLE_ROTATION="$(eval_config_value "${VLA_EVAL_CONFIG_PATH}" "remote_policy_disable_rotation")"
CONFIG_REMOTE_POLICY_GRIPPER_MODE="$(eval_config_value "${VLA_EVAL_CONFIG_PATH}" "remote_policy_gripper_mode")"
CONFIG_REMOTE_POLICY_OVERSHOOT_GUARD="$(eval_config_value "${VLA_EVAL_CONFIG_PATH}" "remote_policy_overshoot_guard")"
CONFIG_OUTPUT_DIR_NAME="$(eval_config_value "${VLA_EVAL_CONFIG_PATH}" "output_dir_name")"
REMOTE_POLICY_URL="${REMOTE_POLICY_URL_OVERRIDE:-${REMOTE_POLICY_URL:-${CONFIG_REMOTE_POLICY_URL:-http://127.0.0.1:8001/infer}}}"
REMOTE_POLICY_TIMEOUT_S="${REMOTE_POLICY_TIMEOUT_S:-20}"
REMOTE_POLICY_TRACE_TAIL="${REMOTE_POLICY_TRACE_TAIL:-0}"
REMOTE_POLICY_CAMERA_NAME="${REMOTE_POLICY_CAMERA_NAME:-table_cam,wrist_cam}"
REMOTE_POLICY_DEBUG="${REMOTE_POLICY_DEBUG:-1}"
REMOTE_POLICY_TIMING="${REMOTE_POLICY_TIMING:-1}"
REMOTE_POLICY_USE_ACTION_CHUNK="${REMOTE_POLICY_USE_ACTION_CHUNK:-1}"
REMOTE_POLICY_DISABLE_ACTION_POSTPROCESS="${REMOTE_POLICY_DISABLE_ACTION_POSTPROCESS:-0}"
REMOTE_POLICY_ACTION_SCALE="${REMOTE_POLICY_ACTION_SCALE_OVERRIDE:-${CONFIG_REMOTE_POLICY_ACTION_SCALE:-${REMOTE_POLICY_ACTION_SCALE:-1}}}"
REMOTE_POLICY_DISABLE_ROTATION="${REMOTE_POLICY_DISABLE_ROTATION_OVERRIDE:-${CONFIG_REMOTE_POLICY_DISABLE_ROTATION:-${REMOTE_POLICY_DISABLE_ROTATION:-0}}}"
REMOTE_POLICY_GRIPPER_MODE="${REMOTE_POLICY_GRIPPER_MODE_OVERRIDE:-${CONFIG_REMOTE_POLICY_GRIPPER_MODE:-${REMOTE_POLICY_GRIPPER_MODE:-isaac}}}"
REMOTE_POLICY_OVERSHOOT_GUARD="${REMOTE_POLICY_OVERSHOOT_GUARD_OVERRIDE:-${CONFIG_REMOTE_POLICY_OVERSHOOT_GUARD:-${REMOTE_POLICY_OVERSHOOT_GUARD:-0}}}"
GEOMIND_MODEL_POLICY_HOOK="${GEOMIND_MODEL_POLICY_HOOK:-geomind_client:policy}"
GEOMIND_ACTIVE_CAMERAS="${GEOMIND_ACTIVE_CAMERAS:-${REMOTE_POLICY_CAMERA_NAME}}"

HEADLESS="${HEADLESS:-1}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
GEOMIND_DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
if [[ -z "${MDL_SYSTEM_PATH:-}" ]]; then
  if [[ -d "${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}" ]]; then
    export MDL_SYSTEM_PATH="${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}"
  fi
fi

SAVE_CAMERA_VIDEO="${SAVE_CAMERA_VIDEO:-1}"
CAMERA_VIDEO_FPS="${CAMERA_VIDEO_FPS:-60}"
CAMERA_VIDEO_NAMES="${CAMERA_VIDEO_NAMES:-${GEOMIND_ACTIVE_CAMERAS}}"
GEOMIND_TRACE_MODE="${GEOMIND_TRACE_MODE:-compact}"
GEOMIND_DATASET_ARTIFACTS="${GEOMIND_DATASET_ARTIFACTS:-1}"
# Roll out full MAX_STEPS even if IsaacLab success termination fires (optional mask at runner).
GEOMIND_IGNORE_SUCCESS_TERMINATION="${GEOMIND_IGNORE_SUCCESS_TERMINATION:-1}"
# Clamp legacy VLA "near object" radii; 10 cm is too loose for 5-6 cm objects.
GEOMIND_MOVE_TO_SUCCESS_RADIUS_MAX="${GEOMIND_MOVE_TO_SUCCESS_RADIUS_MAX:-0.045}"
# VLA tracking videos should show the scripted object swap inside short smoke rollouts.
GEOMIND_RESET_STABILIZATION_STEPS="${GEOMIND_RESET_STABILIZATION_STEPS:-0}"
GEOMIND_POST_INTERVENTION_SETTLE_STEPS="${GEOMIND_POST_INTERVENTION_SETTLE_STEPS:-0}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_${REMOTE_POLICY_MODEL}_vla_${TASK_ID_ENV}_${POLICY}_remote}"
REMOTE_POLICY_OUTPUT_DIR_NAME="${REMOTE_POLICY_OUTPUT_DIR_NAME:-${CONFIG_OUTPUT_DIR_NAME:-run_eval_${REMOTE_POLICY_MODEL}}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-geomind/outputs/${REMOTE_POLICY_OUTPUT_DIR_NAME}/${RUN_NAME}}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
EPISODE_DIR_NAME="$(printf 'episode_%04d' "${EPISODE_INDEX}")"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EPISODE_DIR_NAME}}"

export REMOTE_POLICY_URL
export REMOTE_POLICY_PORT_MODEL_MAP
export REMOTE_POLICY_MODEL
export VLA_EVAL_CONFIG_PATH
export REMOTE_POLICY_TIMEOUT_S
export REMOTE_POLICY_TRACE_TAIL
export REMOTE_POLICY_CAMERA_NAME
export REMOTE_POLICY_DEBUG
export REMOTE_POLICY_TIMING
export REMOTE_POLICY_USE_ACTION_CHUNK
export REMOTE_POLICY_DISABLE_ACTION_POSTPROCESS
export REMOTE_POLICY_ACTION_SCALE
export REMOTE_POLICY_DISABLE_ROTATION
export REMOTE_POLICY_GRIPPER_MODE
export REMOTE_POLICY_OVERSHOOT_GUARD
export GEOMIND_MODEL_POLICY_HOOK
export GEOMIND_ACTIVE_CAMERAS
export GEOMIND_TRACE_MODE
export GEOMIND_DATASET_ARTIFACTS
export GEOMIND_IGNORE_SUCCESS_TERMINATION
export GEOMIND_MOVE_TO_SUCCESS_RADIUS_MAX
export GEOMIND_RESET_STABILIZATION_STEPS
export GEOMIND_POST_INTERVENTION_SETTLE_STEPS
export GEOMIND_SAVE_CAMERA_VIDEO="${SAVE_CAMERA_VIDEO}"
export GEOMIND_CAMERA_VIDEO_CAMERA="${CAMERA_VIDEO_NAMES}"
export GEOMIND_CAMERA_VIDEO_FPS="${CAMERA_VIDEO_FPS}"
export HEADLESS
export ENABLE_CAMERAS
export DISABLE_FABRIC
export MDL_SYSTEM_PATH

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
export OMNI_USER_DIR="${OMNI_USER_DIR:-/tmp/omni-user}"
export OMNI_CACHE_ROOT="${OMNI_CACHE_ROOT:-/tmp/omni-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OMNI_USER_DIR}" "${OMNI_CACHE_ROOT}"

source "${SCRIPT_DIR}/../../scripts/lib/geomind_conda.sh"
geomind_activate_conda "${CONDA_ENV_NAME}" || exit 1

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}"
LOG_PATH="${OUTPUT_ROOT}/run.log"

# Before conda / IsaacLab: optional skip when batch eval passes --resume and a prior result exists.
if [[ "${VLA_EVAL_SKIP_IF_RESULT_JSON:-0}" == "1" && -f "${OUTPUT_DIR}/result.json" ]]; then
  echo "========================================================================" | tee "${LOG_PATH}"
  echo "Skipping IsaacLab rollout (existing result.json)" | tee -a "${LOG_PATH}"
  echo "output_dir=${OUTPUT_DIR}" | tee -a "${LOG_PATH}"
  echo "result_json=${OUTPUT_DIR}/result.json" | tee -a "${LOG_PATH}"
  echo "========================================================================" | tee -a "${LOG_PATH}"
  exit 0
fi

echo "========================================================================" | tee "${LOG_PATH}"
echo "GeoMind Remote Policy IsaacLab — VLA tasks (configs/vla_tasks)" | tee -a "${LOG_PATH}"
echo "========================================================================" | tee -a "${LOG_PATH}"
echo "task_id_env=${TASK_ID_ENV}" | tee -a "${LOG_PATH}"
echo "policy=${POLICY}" | tee -a "${LOG_PATH}"
echo "max_steps=${MAX_STEPS}" | tee -a "${LOG_PATH}"
echo "backend=${BACKEND}" | tee -a "${LOG_PATH}"
echo "seed=${SEED}" | tee -a "${LOG_PATH}"
echo "native_object_pose_perturb_range=${NATIVE_OBJECT_POSE_PERTURB_RANGE}" | tee -a "${LOG_PATH}"
echo "conda_env=${CONDA_ENV_NAME}" | tee -a "${LOG_PATH}"
echo "remote_policy_url=${REMOTE_POLICY_URL}" | tee -a "${LOG_PATH}"
echo "remote_policy_model=${REMOTE_POLICY_MODEL}" | tee -a "${LOG_PATH}"
echo "remote_policy_action_scale=${REMOTE_POLICY_ACTION_SCALE}" | tee -a "${LOG_PATH}"
echo "remote_policy_disable_rotation=${REMOTE_POLICY_DISABLE_ROTATION}" | tee -a "${LOG_PATH}"
echo "remote_policy_gripper_mode=${REMOTE_POLICY_GRIPPER_MODE}" | tee -a "${LOG_PATH}"
echo "remote_policy_overshoot_guard=${REMOTE_POLICY_OVERSHOOT_GUARD}" | tee -a "${LOG_PATH}"
echo "vla_eval_config_path=${VLA_EVAL_CONFIG_PATH}" | tee -a "${LOG_PATH}"
echo "remote_policy_port_model_map=${REMOTE_POLICY_PORT_MODEL_MAP}" | tee -a "${LOG_PATH}"
echo "remote_policy_camera_name=${REMOTE_POLICY_CAMERA_NAME}" | tee -a "${LOG_PATH}"
echo "geomind_active_cameras=${GEOMIND_ACTIVE_CAMERAS}" | tee -a "${LOG_PATH}"
echo "geomind_ignore_success_termination=${GEOMIND_IGNORE_SUCCESS_TERMINATION}" | tee -a "${LOG_PATH}"
echo "geomind_move_to_success_radius_max=${GEOMIND_MOVE_TO_SUCCESS_RADIUS_MAX}" | tee -a "${LOG_PATH}"
echo "geomind_reset_stabilization_steps=${GEOMIND_RESET_STABILIZATION_STEPS}" | tee -a "${LOG_PATH}"
echo "geomind_post_intervention_settle_steps=${GEOMIND_POST_INTERVENTION_SETTLE_STEPS}" | tee -a "${LOG_PATH}"
echo "mdl_system_path=${MDL_SYSTEM_PATH}" | tee -a "${LOG_PATH}"
echo "headless=${HEADLESS}" | tee -a "${LOG_PATH}"
echo "enable_cameras=${ENABLE_CAMERAS}" | tee -a "${LOG_PATH}"
echo "output_dir=${OUTPUT_DIR}" | tee -a "${LOG_PATH}"
echo "output_root=${OUTPUT_ROOT}" | tee -a "${LOG_PATH}"
echo "========================================================================" | tee -a "${LOG_PATH}"

# python | tee 时 Ctrl+C 往往只打断 tee，Isaac/Omniverse 子进程会继续跑。对整个进程组发信号以便真正退出。
trap 'echo "" | tee -a "${LOG_PATH}"; echo "[run_remote_policy_isaaclab_vla] SIGINT: stopping process group (IsaacLab)..." | tee -a "${LOG_PATH}"; kill -- -$$ 2>/dev/null || true; exit 130' INT

set +e
python -m geomind.runners.run_env_policy \
  --task-id-env "${TASK_ID_ENV}" \
  --policy "${POLICY}" \
  --max-steps "${MAX_STEPS}" \
  --backend "${BACKEND}" \
  --seed "${SEED}" \
  --native-object-pose-perturb-range "${NATIVE_OBJECT_POSE_PERTURB_RANGE}" \
  --output-dir "${OUTPUT_DIR}" \
  --ablation-id "${ABLATION_ID}" \
  --pair-id "${PAIR_ID}" \
  "$@" 2>&1 | tee -a "${LOG_PATH}"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [[ "${RUN_STATUS}" != "0" ]]; then
  echo "========================================================================" | tee -a "${LOG_PATH}"
  echo "run_env_policy_failed_exit_code=${RUN_STATUS}" | tee -a "${LOG_PATH}"
  echo "run_log=${LOG_PATH}" | tee -a "${LOG_PATH}"
  echo "========================================================================" | tee -a "${LOG_PATH}"
  exit "${RUN_STATUS}"
fi

MANIFEST_PATH="${OUTPUT_ROOT}/manifest.json"
python - <<PY
import json
from pathlib import Path
manifest = {
    "suite": "vla_tasks",
    "task_id_env": "${TASK_ID_ENV}",
    "ablation_id": "${ABLATION_ID}",
    "pair_id": "${PAIR_ID}",
    "backend": "${BACKEND}",
    "policy": "${POLICY}",
    "remote_policy_url": "${REMOTE_POLICY_URL}",
    "remote_policy_model": "${REMOTE_POLICY_MODEL}",
    "remote_policy_action_scale": float("${REMOTE_POLICY_ACTION_SCALE}"),
    "remote_policy_disable_rotation": "${REMOTE_POLICY_DISABLE_ROTATION}",
    "remote_policy_gripper_mode": "${REMOTE_POLICY_GRIPPER_MODE}",
    "remote_policy_overshoot_guard": "${REMOTE_POLICY_OVERSHOOT_GUARD}",
    "vla_eval_config_path": "${VLA_EVAL_CONFIG_PATH}",
    "remote_policy_port_model_map": "${REMOTE_POLICY_PORT_MODEL_MAP}",
    "remote_policy_camera_name": "${REMOTE_POLICY_CAMERA_NAME}",
    "max_steps": int("${MAX_STEPS}"),
    "episodes": [
        {
            "episode_index": int("${EPISODE_INDEX}"),
            "path": "${OUTPUT_DIR}",
            "result_json": "${OUTPUT_DIR}/result.json",
            "trace_json": "${OUTPUT_DIR}/trace.json",
        }
    ],
}
Path("${MANIFEST_PATH}").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY

echo "========================================================================" | tee -a "${LOG_PATH}"
echo "result_json=${OUTPUT_DIR}/result.json" | tee -a "${LOG_PATH}"
echo "trace_json=${OUTPUT_DIR}/trace.json" | tee -a "${LOG_PATH}"
echo "manifest_json=${MANIFEST_PATH}" | tee -a "${LOG_PATH}"
echo "run_log=${LOG_PATH}" | tee -a "${LOG_PATH}"
echo "========================================================================" | tee -a "${LOG_PATH}"
