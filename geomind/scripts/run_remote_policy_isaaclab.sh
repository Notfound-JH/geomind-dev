#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK_ID_ENV="${TASK_ID_ENV:-CompleteSquare_v0}"
POLICY="${POLICY:-model}"
MAX_STEPS="${MAX_STEPS:-10}"
BACKEND="${BACKEND:-isaaclab}"
SEED="${SEED:-0}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaac_curobo}"

REMOTE_POLICY_URL="${REMOTE_POLICY_URL:-http://127.0.0.1:8000/infer}"
REMOTE_POLICY_TIMEOUT_S="${REMOTE_POLICY_TIMEOUT_S:-20}"
REMOTE_POLICY_TRACE_TAIL="${REMOTE_POLICY_TRACE_TAIL:-0}"
# 桌面相机 + 手腕相机，与 payload["images"] / image_b64_by_camera 一致（两路 PNG Base64）
REMOTE_POLICY_CAMERA_NAME="${REMOTE_POLICY_CAMERA_NAME:-table_cam,wrist_cam}"
REMOTE_POLICY_DEBUG="${REMOTE_POLICY_DEBUG:-1}"
REMOTE_POLICY_TIMING="${REMOTE_POLICY_TIMING:-1}"
REMOTE_POLICY_USE_ACTION_CHUNK="${REMOTE_POLICY_USE_ACTION_CHUNK:-1}"
# 为 1 时跳过 geomind_client 工作空间裁剪 / xy 约束 / gripper 与旋转等后处理，仅 _coerce + clip 到 [-1,1]
REMOTE_POLICY_DISABLE_ACTION_POSTPROCESS="${REMOTE_POLICY_DISABLE_ACTION_POSTPROCESS:-1}"
GEOMIND_MODEL_POLICY_HOOK="${GEOMIND_MODEL_POLICY_HOOK:-geomind_client:policy}"
# 与 REMOTE_POLICY_CAMERA_NAME 对齐，否则观测里只有第一路相机，第二路无法打包进请求
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
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_${TASK_ID_ENV}_${POLICY}_remote}"
OUTPUT_ROOT="${OUTPUT_ROOT:-geomind/outputs/remote_policy/${RUN_NAME}}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
EPISODE_DIR_NAME="$(printf 'episode_%04d' "${EPISODE_INDEX}")"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EPISODE_DIR_NAME}}"

export REMOTE_POLICY_URL
export REMOTE_POLICY_TIMEOUT_S
export REMOTE_POLICY_TRACE_TAIL
export REMOTE_POLICY_CAMERA_NAME
export REMOTE_POLICY_DEBUG
export REMOTE_POLICY_TIMING
export REMOTE_POLICY_USE_ACTION_CHUNK
export REMOTE_POLICY_DISABLE_ACTION_POSTPROCESS
export GEOMIND_MODEL_POLICY_HOOK
export GEOMIND_ACTIVE_CAMERAS
export GEOMIND_TRACE_MODE
export GEOMIND_DATASET_ARTIFACTS
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

echo "========================================================================" | tee "${LOG_PATH}"
echo "GeoMind Remote Policy IsaacLab Smoke" | tee -a "${LOG_PATH}"
echo "========================================================================" | tee -a "${LOG_PATH}"
echo "task_id_env=${TASK_ID_ENV}" | tee -a "${LOG_PATH}"
echo "policy=${POLICY}" | tee -a "${LOG_PATH}"
echo "max_steps=${MAX_STEPS}" | tee -a "${LOG_PATH}"
echo "backend=${BACKEND}" | tee -a "${LOG_PATH}"
echo "seed=${SEED}" | tee -a "${LOG_PATH}"
echo "conda_env=${CONDA_ENV_NAME}" | tee -a "${LOG_PATH}"
echo "remote_policy_url=${REMOTE_POLICY_URL}" | tee -a "${LOG_PATH}"
echo "remote_policy_camera_name=${REMOTE_POLICY_CAMERA_NAME}" | tee -a "${LOG_PATH}"
echo "remote_policy_trace_tail=${REMOTE_POLICY_TRACE_TAIL}" | tee -a "${LOG_PATH}"
echo "remote_policy_use_action_chunk=${REMOTE_POLICY_USE_ACTION_CHUNK}" | tee -a "${LOG_PATH}"
echo "remote_policy_disable_action_postprocess=${REMOTE_POLICY_DISABLE_ACTION_POSTPROCESS}" | tee -a "${LOG_PATH}"
echo "geomind_active_cameras=${GEOMIND_ACTIVE_CAMERAS}" | tee -a "${LOG_PATH}"
echo "mdl_system_path=${MDL_SYSTEM_PATH}" | tee -a "${LOG_PATH}"
echo "headless=${HEADLESS}" | tee -a "${LOG_PATH}"
echo "enable_cameras=${ENABLE_CAMERAS}" | tee -a "${LOG_PATH}"
echo "output_dir=${OUTPUT_DIR}" | tee -a "${LOG_PATH}"
echo "output_root=${OUTPUT_ROOT}" | tee -a "${LOG_PATH}"
echo "episode_dir=${OUTPUT_DIR}" | tee -a "${LOG_PATH}"
echo "save_camera_video=${SAVE_CAMERA_VIDEO}" | tee -a "${LOG_PATH}"
echo "camera_video_names=${CAMERA_VIDEO_NAMES}" | tee -a "${LOG_PATH}"
echo "geomind_trace_mode=${GEOMIND_TRACE_MODE}" | tee -a "${LOG_PATH}"
echo "geomind_dataset_artifacts=${GEOMIND_DATASET_ARTIFACTS}" | tee -a "${LOG_PATH}"
if command -v free >/dev/null 2>&1; then
  free -h | tee -a "${LOG_PATH}" || true
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee -a "${LOG_PATH}" || true
fi
echo "========================================================================" | tee -a "${LOG_PATH}"

trap 'echo "" | tee -a "${LOG_PATH}"; echo "[run_remote_policy_isaaclab] SIGINT: stopping process group (IsaacLab)..." | tee -a "${LOG_PATH}"; kill -- -$$ 2>/dev/null || true; exit 130' INT

set +e
python -m geomind.runners.run_env_policy \
  --task-id-env "${TASK_ID_ENV}" \
  --policy "${POLICY}" \
  --max-steps "${MAX_STEPS}" \
  --backend "${BACKEND}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@" 2>&1 | tee -a "${LOG_PATH}"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [[ "${RUN_STATUS}" != "0" ]]; then
  echo "========================================================================" | tee -a "${LOG_PATH}"
  echo "run_env_policy_failed_exit_code=${RUN_STATUS}" | tee -a "${LOG_PATH}"
  if [[ "${RUN_STATUS}" == "137" || "${RUN_STATUS}" == "9" ]]; then
    echo "process_was_killed=true" | tee -a "${LOG_PATH}"
    echo "hint=SIGKILL usually means OOM killer, GPU/driver reset, or an external kill signal; check dmesg and nvidia-smi." | tee -a "${LOG_PATH}"
  fi
  echo "run_log=${LOG_PATH}" | tee -a "${LOG_PATH}"
  echo "========================================================================" | tee -a "${LOG_PATH}"
  exit "${RUN_STATUS}"
fi

MANIFEST_PATH="${OUTPUT_ROOT}/manifest.json"
python - <<PY
import json
from pathlib import Path
manifest = {
    "task_id_env": "${TASK_ID_ENV}",
    "backend": "${BACKEND}",
    "policy": "${POLICY}",
    "remote_policy_url": "${REMOTE_POLICY_URL}",
    "remote_policy_camera_name": "${REMOTE_POLICY_CAMERA_NAME}",
    "max_steps": int("${MAX_STEPS}"),
    "episodes": [
        {
            "episode_index": int("${EPISODE_INDEX}"),
            "path": "${OUTPUT_DIR}",
            "result_json": "${OUTPUT_DIR}/result.json",
            "trace_json": "${OUTPUT_DIR}/trace.json",
            "trajectory_json": "${OUTPUT_DIR}/trajectory.json",
            "observation_info_json": "${OUTPUT_DIR}/observation_info.json",
            "camera_videos_dir": "${OUTPUT_DIR}/camera_videos",
        }
    ],
}
Path("${MANIFEST_PATH}").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY

echo "========================================================================" | tee -a "${LOG_PATH}"
echo "result_json=${OUTPUT_DIR}/result.json" | tee -a "${LOG_PATH}"
echo "trace_json=${OUTPUT_DIR}/trace.json" | tee -a "${LOG_PATH}"
echo "trajectory_json=${OUTPUT_DIR}/trajectory.json" | tee -a "${LOG_PATH}"
echo "observation_info_json=${OUTPUT_DIR}/observation_info.json" | tee -a "${LOG_PATH}"
echo "camera_videos_dir=${OUTPUT_DIR}/camera_videos" | tee -a "${LOG_PATH}"
echo "manifest_json=${MANIFEST_PATH}" | tee -a "${LOG_PATH}"
echo "run_log=${LOG_PATH}" | tee -a "${LOG_PATH}"
echo "========================================================================" | tee -a "${LOG_PATH}"
