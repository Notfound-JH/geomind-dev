#!/usr/bin/env bash
set -euo pipefail

GEOMIND_DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
if [[ -z "${MDL_SYSTEM_PATH:-}" ]]; then
  if [[ -d "${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}" ]]; then
    export MDL_SYSTEM_PATH="${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}"
  fi
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
export OMNI_USER_DIR="${OMNI_USER_DIR:-/tmp/omni-user}"
export OMNI_CACHE_ROOT="${OMNI_CACHE_ROOT:-/tmp/omni-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OMNI_USER_DIR}" "${OMNI_CACHE_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 场景模板（默认复刻 reindex_extreme 的 line_4 摆放语义）
ANNOTATION_LAYOUT_KEY="${ANNOTATION_LAYOUT_KEY:-state_reindex_blocks_line_4}"
ANNOTATION_PATH="${ANNOTATION_PATH:-${REPO_ROOT}/assets/annotations/annotated_tasks.json}"
SCENE_ID="${SCENE_ID:-}"

# 摆放接口：line / matrix / t_shape / circle / basket_center / stack_center
ARRANGEMENT="${ARRANGEMENT:-line}"
NUM_OBJECTS="${NUM_OBJECTS:-4}"
NUM_OBJECTS_MIN="${NUM_OBJECTS_MIN:-}"
NUM_OBJECTS_MAX="${NUM_OBJECTS_MAX:-}"
# Backward-compat aliases to avoid silent fallback when users type OBJECTS_MIN/OBJECTS_MAX.
if [[ -z "${NUM_OBJECTS_MIN}" && -n "${OBJECTS_MIN:-}" ]]; then
  NUM_OBJECTS_MIN="${OBJECTS_MIN}"
fi
if [[ -z "${NUM_OBJECTS_MAX}" && -n "${OBJECTS_MAX:-}" ]]; then
  NUM_OBJECTS_MAX="${OBJECTS_MAX}"
fi
RANDOM_SEED="${RANDOM_SEED:-}"
SPACING_JITTER="${SPACING_JITTER:-0}"
XY_JITTER="${XY_JITTER:-0}"
BLOCK_COLOR_MODE="${BLOCK_COLOR_MODE:-distinct}"
EXPORT_MEDIA="${EXPORT_MEDIA:-none}"
EXPORT_DIR="${EXPORT_DIR:-}"
EXPORT_CAMERAS="${EXPORT_CAMERAS:-}"
EXPORT_VIDEO_FRAMES="${EXPORT_VIDEO_FRAMES:-}"
EXPORT_RUN_TAG="${EXPORT_RUN_TAG:-}"
LINE_AXIS="${LINE_AXIS:-y}"
SPACING="${SPACING:-}"
MATRIX_COLS="${MATRIX_COLS:-3}"
CIRCLE_RADIUS="${CIRCLE_RADIUS:-}"
CUBE_SIZE="${CUBE_SIZE:-0.045}"
# stack_center：顶层相对底层边长比例（与 tests/render_static_arrangement_scene.py --stack-top-ratio 一致）
STACK_TOP_RATIO="${STACK_TOP_RATIO:-}"
CENTER_XY="${CENTER_XY:-}"

SETTLE_STEPS="${SETTLE_STEPS:-60}"
# 0 表示一直渲染到你手动关闭窗口
HOLD_STEPS="${HOLD_STEPS:-0}"
# GUI 下启动阶段 is_running 可能短暂为假，低于该步数时不据此退出（默认与 Python 一致）
MIN_LIVE_STEPS="${MIN_LIVE_STEPS:-512}"
LIGHTING_MODE="${LIGHTING_MODE:-natural}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
HEADLESS="${HEADLESS:-0}"
VIEWER_EYE="${VIEWER_EYE:-2.2 2.2 1.8}"
VIEWER_LOOKAT="${VIEWER_LOOKAT:-0.35 0.0 0.75}"

source "${SCRIPT_DIR}/lib/geomind_conda.sh"
geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1

ARGS=(
  --annotation-layout-key "${ANNOTATION_LAYOUT_KEY}"
  --annotation-path "${ANNOTATION_PATH}"
  --arrangement "${ARRANGEMENT}"
  --line-axis "${LINE_AXIS}"
  --matrix-cols "${MATRIX_COLS}"
  --cube-size "${CUBE_SIZE}"
  --settle-steps "${SETTLE_STEPS}"
  --hold-steps "${HOLD_STEPS}"
  --min-live-steps "${MIN_LIVE_STEPS}"
  --lighting-mode "${LIGHTING_MODE}"
  --block-color-mode "${BLOCK_COLOR_MODE}"
  --spacing-jitter "${SPACING_JITTER}"
  --xy-jitter "${XY_JITTER}"
  --export-media "${EXPORT_MEDIA}"
  --viewer-eye ${VIEWER_EYE}
  --viewer-lookat ${VIEWER_LOOKAT}
)

if [[ -n "${NUM_OBJECTS_MIN}" && -n "${NUM_OBJECTS_MAX}" ]]; then
  ARGS+=(--num-objects-min "${NUM_OBJECTS_MIN}" --num-objects-max "${NUM_OBJECTS_MAX}")
else
  ARGS+=(--num-objects "${NUM_OBJECTS}")
fi

if [[ -n "${SCENE_ID}" ]]; then
  ARGS+=(--scene-id "${SCENE_ID}")
fi
if [[ -n "${SPACING}" ]]; then
  ARGS+=(--spacing "${SPACING}")
fi
if [[ -n "${CIRCLE_RADIUS}" ]]; then
  ARGS+=(--circle-radius "${CIRCLE_RADIUS}")
fi
if [[ -n "${CENTER_XY}" ]]; then
  # 例如：CENTER_XY="0.42 0.0"
  ARGS+=(--center-xy ${CENTER_XY})
fi
if [[ "${DISABLE_FABRIC}" == "1" ]]; then
  ARGS+=(--disable-fabric)
fi
if [[ "${HEADLESS}" == "1" ]]; then
  ARGS+=(--headless)
fi
if [[ -n "${STACK_TOP_RATIO}" ]]; then
  ARGS+=(--stack-top-ratio "${STACK_TOP_RATIO}")
fi
if [[ -n "${RANDOM_SEED}" ]]; then
  ARGS+=(--random-seed "${RANDOM_SEED}")
fi
if [[ -n "${EXPORT_DIR}" ]]; then
  ARGS+=(--export-dir "${EXPORT_DIR}")
fi
if [[ -n "${EXPORT_CAMERAS}" ]]; then
  ARGS+=(--export-cameras "${EXPORT_CAMERAS}")
fi
if [[ -n "${EXPORT_VIDEO_FRAMES}" ]]; then
  ARGS+=(--export-video-frames "${EXPORT_VIDEO_FRAMES}")
fi
if [[ -n "${EXPORT_RUN_TAG}" ]]; then
  ARGS+=(--export-run-tag "${EXPORT_RUN_TAG}")
fi

echo "========================================================================"
echo "GeoMind Static Arrangement Scene"
echo "========================================================================"
echo "annotation_layout_key=${ANNOTATION_LAYOUT_KEY}"
echo "annotation_path=${ANNOTATION_PATH}"
echo "scene_id=${SCENE_ID:-<from-annotation>}"
echo "arrangement=${ARRANGEMENT}"
if [[ -n "${NUM_OBJECTS_MIN}" && -n "${NUM_OBJECTS_MAX}" ]]; then
  echo "num_objects_range=${NUM_OBJECTS_MIN}..${NUM_OBJECTS_MAX}"
else
  echo "num_objects=${NUM_OBJECTS}"
fi
echo "random_seed=${RANDOM_SEED:-<none>}"
echo "spacing_jitter=${SPACING_JITTER}"
echo "xy_jitter=${XY_JITTER}"
echo "block_color_mode=${BLOCK_COLOR_MODE}"
echo "export_media=${EXPORT_MEDIA}"
echo "line_axis=${LINE_AXIS}"
echo "spacing=${SPACING:-<template-derived>}"
echo "matrix_cols=${MATRIX_COLS}"
echo "circle_radius=${CIRCLE_RADIUS:-<auto>}"
echo "cube_size=${CUBE_SIZE}"
echo "stack_top_ratio=${STACK_TOP_RATIO:-<default 0.78>}"
echo "center_xy=${CENTER_XY:-<template-center>}"
echo "settle_steps=${SETTLE_STEPS}"
echo "hold_steps=${HOLD_STEPS}"
echo "min_live_steps=${MIN_LIVE_STEPS}"
echo "headless=${HEADLESS}"
echo "disable_fabric=${DISABLE_FABRIC}"
echo "========================================================================"
echo "若日志中出现 omni.kvdb / kvdb 被占用，请先关闭其它 Isaac Sim 或 Kit 进程再运行本脚本。"
echo "========================================================================"

export PYTHONUNBUFFERED=1
exec python -u "${REPO_ROOT}/tests/render_static_arrangement_scene.py" "${ARGS[@]}" "$@"
