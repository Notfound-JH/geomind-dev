#!/usr/bin/env bash

geomind_debug_scan_request_args() {
  local consume_positional_variant="${1:-0}"
  shift || true

  GEOMIND_FORWARD_ARGS=()
  GEOMIND_REQUEST_TASK_ID=""
  GEOMIND_REQUEST_VARIANT=""
  GEOMIND_REQUEST_SCENE=""
  GEOMIND_REQUEST_SCENE_ID=""
  GEOMIND_REQUEST_LAYOUT_KEY=""
  GEOMIND_REQUEST_TASK_INDEX=""
  GEOMIND_REQUEST_OBJECT_MODE=""

  local positional_consumed=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --task-id)
        GEOMIND_REQUEST_TASK_ID="${2:-}"
        GEOMIND_FORWARD_ARGS+=("$1" "${2:-}")
        shift 2
        ;;
      --variant)
        GEOMIND_REQUEST_VARIANT="${2:-}"
        GEOMIND_FORWARD_ARGS+=("$1" "${2:-}")
        shift 2
        ;;
      --scene)
        GEOMIND_REQUEST_SCENE="${2:-}"
        GEOMIND_FORWARD_ARGS+=("$1" "${2:-}")
        shift 2
        ;;
      --scene-id)
        GEOMIND_REQUEST_SCENE_ID="${2:-}"
        GEOMIND_FORWARD_ARGS+=("$1" "${2:-}")
        shift 2
        ;;
      --layout-key)
        GEOMIND_REQUEST_LAYOUT_KEY="${2:-}"
        GEOMIND_FORWARD_ARGS+=("$1" "${2:-}")
        shift 2
        ;;
      --task-index)
        GEOMIND_REQUEST_TASK_INDEX="${2:-}"
        GEOMIND_FORWARD_ARGS+=("$1" "${2:-}")
        shift 2
        ;;
      --object-mode|--object_mode)
        GEOMIND_REQUEST_OBJECT_MODE="${2:-}"
        GEOMIND_FORWARD_ARGS+=("--object-mode" "${2:-}")
        shift 2
        ;;
      --)
        GEOMIND_FORWARD_ARGS+=("$@")
        break
        ;;
      *)
        if [[ "${consume_positional_variant}" == "1" && "${positional_consumed}" == "0" && -z "${GEOMIND_REQUEST_VARIANT:-}" && "$1" != --* ]]; then
          GEOMIND_REQUEST_VARIANT="$1"
          positional_consumed=1
          shift
          continue
        fi
        GEOMIND_FORWARD_ARGS+=("$1")
        shift
        ;;
    esac
  done
}


geomind_resolve_task_context() {
  local repo_root="$1"
  local default_task_id="$2"

  local task_id="${GEOMIND_REQUEST_TASK_ID:-${TASK_ID:-${default_task_id}}}"
  local variant="${GEOMIND_REQUEST_VARIANT:-${TASK_VARIANT:-${VARIANT:-${PRESET:-${TASK_VERSION:-}}}}}"
  local scene="${GEOMIND_REQUEST_SCENE:-${SCENE:-}}"
  local scene_id="${GEOMIND_REQUEST_SCENE_ID:-${SCENE_ID:-}}"
  local layout_key="${GEOMIND_REQUEST_LAYOUT_KEY:-${LAYOUT_KEY:-}}"
  local task_index="${GEOMIND_REQUEST_TASK_INDEX:-${TASK_INDEX:-}}"
  local object_mode="${GEOMIND_REQUEST_OBJECT_MODE:-${OBJECT_MODE:-}}"

  local resolved_exports
  local resolve_status
  resolved_exports="$(
    python - "$repo_root" "$task_id" "$variant" "$scene" "$scene_id" "$layout_key" "$task_index" "$object_mode" <<'PY'
import os
import shlex
import sys

repo_root, task_id, variant, scene, scene_id, layout_key, task_index, object_mode = sys.argv[1:9]
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from geomind_lab.tasks.manager_based.manipulation.geomind.task_profiles import (
    resolve_task_env_defaults,
    resolve_task_log_root,
)
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import resolve_task_request

request = resolve_task_request(
    task_id=task_id or None,
    variant=variant or None,
    scene=scene or None,
    scene_id=scene_id or None,
    layout_key=layout_key or None,
    task_index=task_index or None,
    object_mode=object_mode or None,
)
env_defaults = resolve_task_env_defaults(request.task_id, request.variant)
exports = {
    "TASK_ID": request.task_id,
    "TASK_VARIANT": request.variant,
    "VARIANT": request.variant,
    "SCENE": request.scene,
    "SCENE_ID": str(request.scene_id),
    "LAYOUT_KEY": request.layout_key,
    "TASK_INDEX": str(request.task_index),
    "OBJECT_MODE": request.object_mode,
    "TASK_BACKEND": request.backend,
    "TASK_LOG_ROOT": resolve_task_log_root(request.task_id),
}

for key, value in exports.items():
    print(f"export {key}={shlex.quote(str(value))}")

for key, value in env_defaults.items():
    text_key = str(key).strip()
    if not text_key or os.environ.get(text_key):
        continue
    print(f"export {text_key}={shlex.quote(str(value))}")
PY
  )"
  resolve_status=$?

  if [[ ${resolve_status} -ne 0 || -z "${resolved_exports}" ]]; then
    echo "geomind_resolve_task_context: 任务解析失败 (exit=${resolve_status})。" >&2
    echo "  task_id='${task_id}' variant='${variant}' scene='${scene}' scene_id='${scene_id}' layout_key='${layout_key}' task_index='${task_index}' object_mode='${object_mode}'" >&2
    echo "  上方 Python traceback 即为根因；请勿忽略后续报错。" >&2
    exit 1
  fi

  eval "${resolved_exports}"
}


geomind_default_output_dir() {
  local repo_root="$1"
  local timestamp="$2"
  if [[ -n "${OUTPUT_DIR:-}" ]]; then
    return
  fi

  local log_root="${TASK_LOG_ROOT:-debug_rollouts}"
  local task_id="${TASK_ID:-task}"
  local variant="${TASK_VARIANT:-default}"
  local object_mode="${OBJECT_MODE:-default}"
  export OUTPUT_DIR="${repo_root}/logs/${log_root}/${task_id}/${variant}/${object_mode}/${timestamp}"
}


# Map legacy TASK_VERSION to TASK_VARIANT / VARIANT when the unified resolver expects variant names.
geomind_debug_sync_task_version_to_variant() {
  if [[ -z "${TASK_VERSION:-}" ]]; then
    return 0
  fi
  if [[ -n "${GEOMIND_REQUEST_VARIANT:-}" || -n "${TASK_VARIANT:-}" || -n "${VARIANT:-}" ]]; then
    return 0
  fi
  export TASK_VARIANT="${TASK_VERSION}"
  export VARIANT="${TASK_VERSION}"
}


# When wrappers still expose PRESET but the user did not pass --variant, treat PRESET as the task variant.
geomind_debug_apply_preset_as_variant() {
  if [[ -z "${PRESET:-}" ]]; then
    return 0
  fi
  if [[ -n "${GEOMIND_REQUEST_VARIANT:-}" || -n "${TASK_VARIANT:-}" || -n "${VARIANT:-}" ]]; then
    return 0
  fi
  export TASK_VARIANT="${PRESET}"
  export VARIANT="${PRESET}"
}


geomind_debug_print_core_context() {
  local title="${1:-GeoMind debug}"
  echo "========================================================================"
  echo "${title}"
  echo "========================================================================"
  echo "task_id=${TASK_ID:-}"
  echo "variant=${TASK_VARIANT:-${VARIANT:-}}"
  echo "scene=${SCENE:-} scene_id=${SCENE_ID:-}"
  echo "object_mode=${OBJECT_MODE:-}"
  echo "layout_key=${LAYOUT_KEY:-}"
  echo "task_index=${TASK_INDEX:-}"
  echo "output_dir=${OUTPUT_DIR:-}"
  echo "headless_camera_video=${HEADLESS_CAMERA_VIDEO:-}"
  echo "========================================================================"
}
