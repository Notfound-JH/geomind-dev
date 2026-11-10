#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/capture_geometric_randomized_scene_cam.py"

SESSION_NAME="${TMUX_SESSION_NAME:-geomind_scene_cam_capture}"
DEFAULT_LOG_DIR="${REPO_ROOT}/logs/geometric_randomized_scene_cam"
LOG_FILE="${GEOMIND_CAPTURE_LOG:-${DEFAULT_LOG_DIR}/tmux_run.log}"

mkdir -p "${DEFAULT_LOG_DIR}"

run_inner() {
  exec > >(tee -a "${LOG_FILE}") 2>&1

  source "${SCRIPT_DIR}/lib/geomind_conda.sh"
  geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1
  cd "${REPO_ROOT}"

  export MDL_SYSTEM_PATH="${MDL_SYSTEM_PATH:-${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials}"

  echo "========================================================================"
  echo "GeoMind scene_cam randomized capture (tmux enforced)"
  echo "session=${SESSION_NAME}"
  echo "cwd=${REPO_ROOT}"
  echo "script=${PY_SCRIPT}"
  echo "log_file=${LOG_FILE}"
  echo "MDL_SYSTEM_PATH=${MDL_SYSTEM_PATH}"
  echo "========================================================================"

  python "${PY_SCRIPT}" "$@"
}

if [[ "${1:-}" == "--inner" ]]; then
  shift
  run_inner "$@"
  exit $?
fi

if [[ -n "${TMUX:-}" ]]; then
  run_inner "$@"
  exit $?
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "error: tmux is required but not found in PATH." >&2
  exit 1
fi

quoted_args=()
for arg in "$@"; do
  quoted_args+=("$(printf '%q' "$arg")")
done
joined_args="${quoted_args[*]:-}"
printf -v quoted_repo '%q' "${REPO_ROOT}"
printf -v quoted_script '%q' "${SCRIPT_DIR}/run_capture_geometric_randomized_scene_cam_tmux.sh"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux send-keys -t "${SESSION_NAME}" "cd ${quoted_repo} && bash ${quoted_script} --inner ${joined_args}" C-m
  echo "Attached command to existing tmux session: ${SESSION_NAME}"
  echo "Use: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" "cd ${quoted_repo} && bash ${quoted_script} --inner ${joined_args}"
echo "Started tmux session: ${SESSION_NAME}"
echo "Use: tmux attach -t ${SESSION_NAME}"
