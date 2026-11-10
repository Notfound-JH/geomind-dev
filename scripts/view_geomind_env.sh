#!/usr/bin/env bash
set -euo pipefail
GEOMIND_DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
if [[ -z "${MDL_SYSTEM_PATH:-}" ]]; then
  if [[ -d "${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}" ]]; then
    export MDL_SYSTEM_PATH="${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}"
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Edit these values, then run:
#   bash scripts/view_geomind_env.sh
SCENE_ID=10
LAYOUT_KEY="line_3"
TASK_INDEX=0
CFG_KIND="visuomotor"
#CFG_KIND="ik-rel"
TASK_MODE=""
LIGHTING_MODE="natural"
# Example: '{"1":"table_cam","2":"wrist_cam"}'
UI_VIEWPORT_CAMERA_MAP='{"1":"table_cam","2":"wrist_cam"}'
NUM_ENVS=1
ACTION="zero"
MAX_STEPS=0
#帮我修改：帮我修改一下每次的相机位置和lookat。目前的位置是固定的，但实际上每个桌子的高度不一样，请帮我实现，
# Viewer camera.
VIEWER_EYE=(1.1 1.1 1.8)
VIEWER_LOOKAT=(0.3 0.0 0.8)

# Set to 1 if you want headless smoke tests instead of the interactive viewer.
HEADLESS=0

# Set to 1 to enable the high-quality render preset from SimulationCfg.render.
# This increases reflections, GI, AO, shadow quality, and direct-light sampling cost.
HIGH_QUALITY_RENDER=1

source "${SCRIPT_DIR}/lib/geomind_conda.sh"
geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1

ARGS=(
  --scene-id "${SCENE_ID}"
  --layout-key "${LAYOUT_KEY}"
  --task-index "${TASK_INDEX}"
  --cfg-kind "${CFG_KIND}"
  --lighting-mode "${LIGHTING_MODE}"
  --num-envs "${NUM_ENVS}"
  --action "${ACTION}"
  --max-steps "${MAX_STEPS}"
  --viewer-eye "${VIEWER_EYE[@]}"
  --viewer-lookat "${VIEWER_LOOKAT[@]}"
)

if [[ -n "${TASK_MODE}" ]]; then
  ARGS+=(--task-mode "${TASK_MODE}")
fi
if [[ -n "${UI_VIEWPORT_CAMERA_MAP}" ]]; then
  ARGS+=(--ui-viewport-camera-map "${UI_VIEWPORT_CAMERA_MAP}")
fi
if [[ "${HEADLESS}" == "1" ]]; then
  ARGS+=(--headless)
fi
if [[ "${HIGH_QUALITY_RENDER}" == "1" ]]; then
  ARGS+=(--high-quality-render)
fi

python "${SCRIPT_DIR}/view_geomind_env.py" "${ARGS[@]}" "$@"
