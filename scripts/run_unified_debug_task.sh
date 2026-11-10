#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_unified_debug_task.sh --task-id TASK_ID [OPTIONS] [-- EXTRA_ARGS...]

Unified selectors:
  --task-id ID            Canonical task id, e.g. track_single, place_inside, square_completion.
  --variant NAME          Difficulty/instance variant, e.g. standard, stress, skill, grid.
  --scene NAME_OR_ID      Scene alias/id, e.g. kitchen_table, lab_table, clutter_table, scene_65, 10.
  --object-mode MODE      concrete or geometric. --object_mode is also accepted.

Runtime options:
  --headless              Run headless and enable camera video by default.
  --no-video              Disable HEADLESS_CAMERA_VIDEO even in headless mode.
  --episodes N            Set AGENT_EPISODES.
  --device DEVICE         Forward device to wrappers, e.g. cuda:0.
  --output DIR            Set OUTPUT_DIR.
  --dry-run               Print the resolved command without executing it.
  -h, --help              Show this help.

Examples:
  bash scripts/run_unified_debug_task.sh --task-id track_single --scene kitchen_table --variant standard --object-mode concrete --headless
  bash scripts/run_unified_debug_task.sh --task-id place_inside --scene lab_table --variant standard --object-mode geometric --headless
  bash scripts/run_unified_debug_task.sh --task-id square_completion --scene lab_table --variant stress --object-mode geometric --headless --episodes 1

Extra args after -- are passed through to the task-specific debug wrapper.
EOF
}

task_id=""
variant=""
scene=""
object_mode=""
episodes=""
device=""
output_dir=""
headless="${HEADLESS:-0}"
video=""
dry_run=0
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-id|--task)
      task_id="${2:-}"
      shift 2
      ;;
    --variant)
      variant="${2:-}"
      shift 2
      ;;
    --scene)
      scene="${2:-}"
      shift 2
      ;;
    --object-mode|--object_mode)
      object_mode="${2:-}"
      shift 2
      ;;
    --headless)
      headless=1
      shift
      ;;
    --no-video)
      video=0
      shift
      ;;
    --episodes)
      episodes="${2:-}"
      shift 2
      ;;
    --device)
      device="${2:-}"
      shift 2
      ;;
    --output)
      output_dir="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${task_id}" ]]; then
  echo "error: --task-id is required" >&2
  usage >&2
  exit 2
fi

debug_args=(--task "${task_id}" --)
if [[ -n "${variant}" ]]; then
  debug_args+=(--variant "${variant}")
fi
if [[ -n "${scene}" ]]; then
  debug_args+=(--scene "${scene}")
fi
if [[ -n "${object_mode}" ]]; then
  debug_args+=(--object-mode "${object_mode}")
fi
debug_args+=("${extra_args[@]}")

export HEADLESS="${headless}"
if [[ -n "${episodes}" ]]; then
  export AGENT_EPISODES="${episodes}"
fi
if [[ -n "${device}" ]]; then
  export DEVICE="${device}"
fi
if [[ -n "${output_dir}" ]]; then
  export OUTPUT_DIR="${output_dir}"
fi
if [[ "${HEADLESS}" == "1" ]]; then
  export HEADLESS_CAMERA_VIDEO="${video:-${HEADLESS_CAMERA_VIDEO:-1}}"
elif [[ -n "${video}" ]]; then
  export HEADLESS_CAMERA_VIDEO="${video}"
fi

cmd=(bash "${SCRIPT_DIR}/debug_task.sh" "${debug_args[@]}")

echo "repo_root=${REPO_ROOT}"
echo "task_id=${task_id}"
echo "variant=${variant:-<default>}"
echo "scene=${scene:-<task-default>}"
echo "object_mode=${object_mode:-<task-default>}"
echo "headless=${HEADLESS}"
echo "headless_camera_video=${HEADLESS_CAMERA_VIDEO:-}"
echo "command=${cmd[*]}"

if [[ "${dry_run}" == "1" ]]; then
  exit 0
fi

exec "${cmd[@]}"
