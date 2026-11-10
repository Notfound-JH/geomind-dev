#!/usr/bin/env bash
# Run every task-level debug wrapper once and keep its scene_cam clip and goal together.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HEADLESS="${HEADLESS:-1}"
export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/debug_scene_cam_$(date +%Y%m%d_%H%M%S)}"
TASK_IDS="${TASK_IDS:-}"

mkdir -p "${RUN_ROOT}"
SUMMARY_FILE="${RUN_ROOT}/batch_summary.txt"
FAILED_FILE="${RUN_ROOT}/failed.txt"
: >"${SUMMARY_FILE}"
: >"${FAILED_FILE}"

if [[ -n "${TASK_IDS}" ]]; then
  read -r -a TASK_ID_ARRAY <<<"${TASK_IDS}"
else
  mapfile -t TASK_ID_ARRAY < <(
    python3 - "${SCRIPT_DIR}/debug_task_registry.json" <<'PY'
import json
import sys

registry = json.load(open(sys.argv[1], encoding="utf-8"))
for task in registry["tasks"]:
    if task.get("category") != "protocol_backend":
        print(task["id"])
PY
  )
fi

echo "run_root=${RUN_ROOT}" | tee -a "${SUMMARY_FILE}"
echo "headless=${HEADLESS} headless_camera_video=${HEADLESS_CAMERA_VIDEO}" | tee -a "${SUMMARY_FILE}"
echo "task_count=${#TASK_ID_ARRAY[@]}" | tee -a "${SUMMARY_FILE}"

failed=0
for index in "${!TASK_ID_ARRAY[@]}"; do
  task_id="${TASK_ID_ARRAY[${index}]}"
  task_dir="${RUN_ROOT}/${task_id}"
  log_file="${task_dir}/run.log"
  mkdir -p "${task_dir}"

  if ! python3 "${SCRIPT_DIR}/write_task_info.py" --task-id "${task_id}" --output "${task_dir}/task_info.txt" >"${task_dir}/task_info.log" 2>&1; then
    printf 'Task name: %s\nTask id: %s\nGoal: (task metadata resolution failed; see task_info.log)\n' "${task_id}" "${task_id}" >"${task_dir}/task_info.txt"
  fi

  echo "[$((index + 1))/${#TASK_ID_ARRAY[@]}] START ${task_id}" | tee -a "${SUMMARY_FILE}"
  (cd "${REPO_ROOT}" && env \
    HEADLESS="${HEADLESS}" \
    HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO}" \
    OUTPUT_DIR="${task_dir}" \
    bash "${SCRIPT_DIR}/debug_task.sh" "${task_id}") >"${log_file}" 2>&1
  run_rc=$?

  scene_cam="$(find "${task_dir}/camera_videos" -type f -name 'scene_cam*.mp4' -print -quit 2>/dev/null || true)"
  if [[ -n "${scene_cam}" ]]; then
    cp -f "${scene_cam}" "${task_dir}/scene_cam.mp4"
    video_note="scene_cam.mp4"
  else
    video_note="missing"
  fi

  printf 'run_exit_code=%s\nscene_cam=%s\nsource=%s\n' "${run_rc}" "${video_note}" "${scene_cam:-}" >"${task_dir}/export_status.txt"
  if [[ "${run_rc}" -eq 0 && -n "${scene_cam}" ]]; then
    echo "[$((index + 1))/${#TASK_ID_ARRAY[@]}] OK ${task_id} video=${video_note}" | tee -a "${SUMMARY_FILE}"
    continue
  fi

  echo "[$((index + 1))/${#TASK_ID_ARRAY[@]}] FAIL ${task_id} exit=${run_rc} video=${video_note}" | tee -a "${SUMMARY_FILE}"
  printf '%s exit=%s video=%s log=%s\n' "${task_id}" "${run_rc}" "${video_note}" "${log_file}" >>"${FAILED_FILE}"
  failed=$((failed + 1))
  if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
    exit "${run_rc:-1}"
  fi
done

echo "finished failed=${failed}/${#TASK_ID_ARRAY[@]}" | tee -a "${SUMMARY_FILE}"
[[ "${failed}" -eq 0 ]]