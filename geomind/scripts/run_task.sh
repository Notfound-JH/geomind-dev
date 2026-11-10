#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK_ID="${TASK_ID:-SelectByDirection}"
MAX_STEPS="${MAX_STEPS:-100}"
BACKEND="${BACKEND:-isaaclab}"
INTERACTIVE="${INTERACTIVE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

ARGS=(
  --task-id "${TASK_ID}"
  --max-steps "${MAX_STEPS}"
  --backend "${BACKEND}"
)

if [[ "${INTERACTIVE}" == "1" ]]; then
  ARGS+=(--interactive)
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

echo "========================================================================"
echo "GeoMind Run Task"
echo "========================================================================"
echo "task_id=${TASK_ID}"
echo "max_steps=${MAX_STEPS}"
echo "backend=${BACKEND}"
echo "interactive=${INTERACTIVE}"
echo "output_dir=${OUTPUT_DIR}"
echo "========================================================================"

cd "${REPO_ROOT}"
python -m geomind.runners.run_task "${ARGS[@]}" "$@"
