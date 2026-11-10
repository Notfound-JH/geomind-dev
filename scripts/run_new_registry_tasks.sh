#!/usr/bin/env bash
set -euo pipefail

GEOMIND_DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
if [[ -z "${MDL_SYSTEM_PATH:-}" ]]; then
  if [[ -d "${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}" ]]; then
    export MDL_SYSTEM_PATH="${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}"
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_RUNNER="${SCRIPT_DIR}/debug_new_registry_tasks.py"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_new_registry_tasks.sh [task_id ...] [options]

Default task_ids:
  reindex_extreme
  reindex_insert
  track_single
  track_multi

Options:
  --scene-id ID          Scene id to test. Default: 10
  --task-version VALUE   Task version. Default: base
  --output DIR           Output directory for summary JSON
  --dry-run              Only build task plans; do not launch IsaacLab
  --headless             Run IsaacLab headless
  --disable-fabric       Disable fabric
  --device DEVICE        Runtime device. Default: cuda:0
  -h, --help             Show this help

Examples:
  bash scripts/run_new_registry_tasks.sh --dry-run
  bash scripts/run_new_registry_tasks.sh reindex_insert track_single --headless
EOF
}

activate_conda_env() {
  source "${SCRIPT_DIR}/lib/geomind_conda.sh"
  geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1
}

scene_id="10"
task_version="base"
output_dir=""
dry_run="0"
headless="0"
disable_fabric="0"
device="cuda:0"
declare -a task_ids=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-id)
      scene_id="${2:-}"
      shift 2
      ;;
    --task-version)
      task_version="${2:-}"
      shift 2
      ;;
    --output)
      output_dir="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run="1"
      shift
      ;;
    --headless)
      headless="1"
      shift
      ;;
    --disable-fabric)
      disable_fabric="1"
      shift
      ;;
    --device)
      device="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      task_ids+=("$1")
      shift
      ;;
  esac
done

if [[ ${#task_ids[@]} -eq 0 ]]; then
  task_ids=("reindex_extreme" "reindex_insert" "track_single" "track_multi")
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${output_dir}" ]]; then
  output_dir="${REPO_ROOT}/logs/new_registry_tasks/${timestamp}_scene${scene_id}"
fi
mkdir -p "${output_dir}"

summary_json="${output_dir}/summary.json"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
export OMNI_USER_DIR="${OMNI_USER_DIR:-/tmp/omni-user}"
export OMNI_CACHE_ROOT="${OMNI_CACHE_ROOT:-/tmp/omni-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OMNI_USER_DIR}" "${OMNI_CACHE_ROOT}"

activate_conda_env

cmd=(
  python "${PYTHON_RUNNER}"
  --scene-id "${scene_id}"
  --task-version "${task_version}"
  --device "${device}"
  --summary-json-out "${summary_json}"
)

if [[ "${dry_run}" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ "${headless}" == "1" ]]; then
  cmd+=(--headless)
fi
if [[ "${disable_fabric}" == "1" ]]; then
  cmd+=(--disable-fabric)
fi

for task_id in "${task_ids[@]}"; do
  cmd+=("${task_id}")
done

echo "scene_id=${scene_id}"
echo "task_version=${task_version}"
echo "dry_run=${dry_run}"
echo "headless=${headless}"
echo "task_ids=${task_ids[*]}"
echo "summary_json=${summary_json}"

"${cmd[@]}"
