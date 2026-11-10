#!/usr/bin/env bash
# 依次以「全新子进程」运行各 debug 包装脚本：每次 bash 退出后下一次再执行，
# 会重新 conda activate 并启动新的 Python / Isaac Lab 进程（非单进程内连续跑多个任务）。
#
# 用法:
#   bash scripts/run_debug_wrappers_headless_video_batch.sh
#   HEADLESS=1 bash scripts/run_debug_wrappers_headless_video_batch.sh   # 默认已 HEADLESS=1
#   CONTINUE_ON_ERROR=1 BETWEEN_RUNS_SLEEP=5 bash scripts/run_debug_wrappers_headless_video_batch.sh
#
# 视频: 各子脚本在 HEADLESS=1 下会设置 HEADLESS_CAMERA_VIDEO=1，录像一般在各 OUTPUT_DIR 下
#       （如 camera_videos/）；部分脚本用自身 OUTPUT_DIR 逻辑，见各 wrapper 注释。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HEADLESS="${HEADLESS:-1}"
export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
# 每次运行使用新时间戳，避免各任务 OUTPUT_DIR 冲突（依赖子脚本内 TIMESTAMP）
unset TIMESTAMP 2>/dev/null || true

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
BETWEEN_RUNS_SLEEP="${BETWEEN_RUNS_SLEEP:-3}"

# 顺序与仓库内 scripts/debug_*.sh 一致（已修正原列表中的 cripts/ 与 .s 笔误）
WRAPPERS=(
  debug_protocol_anchor_reference.sh
  debug_grasp_anchor_distance.sh
  debug_place_directional.sh
  debug_place_on_top.sh
  debug_place_inside.sh
  debug_select_visual_spatial.sh
  debug_select_exclusion_unique.sh
  debug_reindex_extreme.sh
  debug_reindex_insert.sh
  debug_grid_rowcol_grounding.sh
  debug_reindex_subset.sh
  debug_reindex_neighbor.sh
  debug_track_single.sh
  debug_track_multi.sh
  debug_track_clutter.sh
  debug_perfect_midpoint.sh
  debug_isometric_extension.sh
  debug_point_symmetry.sh
  debug_trend_continuation.sh
  debug_weak_destruction_control.sh
  debug_square_completion.sh
)

RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_ROOT}/logs/batch_debug_headless_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_LOG_DIR}"
SUMMARY_FILE="${RUN_LOG_DIR}/batch_summary.txt"
FAILED_FILE="${RUN_LOG_DIR}/failed.txt"
: >"${SUMMARY_FILE}"
: >"${FAILED_FILE}"

echo "batch_run_root=${RUN_LOG_DIR}"
echo "HEADLESS=${HEADLESS} HEADLESS_CAMERA_VIDEO=${HEADLESS_CAMERA_VIDEO}"
echo "continue_on_error=${CONTINUE_ON_ERROR} between_runs_sleep=${BETWEEN_RUNS_SLEEP}"
echo "----------------------------------------------------------------"

idx=0
fail_count=0
for name in "${WRAPPERS[@]}"; do
  idx=$((idx + 1))
  path="${SCRIPT_DIR}/${name}"
  log="${RUN_LOG_DIR}/$(printf '%02d' "${idx}")_${name%.sh}.log"

  if [[ ! -f "${path}" ]]; then
    echo "[${idx}/${#WRAPPERS[@]}] SKIP missing: ${path}" | tee -a "${SUMMARY_FILE}"
    echo "${name} MISSING" >>"${FAILED_FILE}"
    fail_count=$((fail_count + 1))
    continue
  fi

  echo "[${idx}/${#WRAPPERS[@]}] START ${name}" | tee -a "${SUMMARY_FILE}"
  start_ts=$(date -Iseconds 2>/dev/null || date)

  # 子 shell：每次均为新 bash → 新 Python / Isaac Lab 进程
  (cd "${REPO_ROOT}" && env HEADLESS="${HEADLESS}" HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO}" bash "${path}") \
    >"${log}" 2>&1
  rc=$?

  end_ts=$(date -Iseconds 2>/dev/null || date)
  if [[ "${rc}" -eq 0 ]]; then
    echo "[${idx}/${#WRAPPERS[@]}] OK   ${name}  (${start_ts} → ${end_ts})  log=${log}" | tee -a "${SUMMARY_FILE}"
  else
    echo "[${idx}/${#WRAPPERS[@]}] FAIL ${name}  exit=${rc}  log=${log}" | tee -a "${SUMMARY_FILE}"
    echo "${name} exit=${rc} log=${log}" >>"${FAILED_FILE}"
    fail_count=$((fail_count + 1))
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      echo "Stopping because CONTINUE_ON_ERROR!=1" | tee -a "${SUMMARY_FILE}"
      exit "${rc}"
    fi
  fi

  if [[ "${BETWEEN_RUNS_SLEEP}" =~ ^[0-9]+$ ]] && [[ "${BETWEEN_RUNS_SLEEP}" -gt 0 ]]; then
    sleep "${BETWEEN_RUNS_SLEEP}"
  fi
done

echo "----------------------------------------------------------------"
echo "done. failed_count=${fail_count}/${#WRAPPERS[@]}"
echo "summary=${SUMMARY_FILE}"
echo "failed_list=${FAILED_FILE}"

if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
exit 0
