#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
README_PATH="${REPO_ROOT}/geomind/scripts/README.md"
RUN_SUITE_SCRIPT="${REPO_ROOT}/geomind/scripts/run_suite.sh"

if [[ ! -f "${README_PATH}" ]]; then
  echo "[ERROR] README not found: ${README_PATH}" >&2
  exit 1
fi
if [[ ! -f "${RUN_SUITE_SCRIPT}" ]]; then
  echo "[ERROR] run_suite.sh not found: ${RUN_SUITE_SCRIPT}" >&2
  exit 1
fi

# Defaults aligned with the command provided by user.
BACKEND="${BACKEND:-isaaclab}"
POLICY="${POLICY:-oracle}"
ISAAC_RENDER="${ISAAC_RENDER:-1}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"

mapfile -t BASE_TASK_IDS < <(
  python - "${README_PATH}" <<'PY'
import re
import sys

readme_path = sys.argv[1]
text = open(readme_path, "r", encoding="utf-8").read().splitlines()

in_base = False
task_ids = []
for line in text:
    if line.strip().startswith("### Base 任务"):
        in_base = True
        continue
    if in_base and line.strip().startswith("### Aug 任务"):
        break
    if not in_base:
        continue
    # Match markdown table rows like:
    # | `ReindexInsert_v0` | `ReindexInsert_v0` |
    m = re.match(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|", line.strip())
    if m:
        registered_id = m.group(2).strip()
        if registered_id:
            task_ids.append(registered_id)

for task_id in task_ids:
    print(task_id)
PY
)

if [[ "${#BASE_TASK_IDS[@]}" -eq 0 ]]; then
  echo "[ERROR] No Base task IDs parsed from ${README_PATH}" >&2
  exit 1
fi

echo "========================================================================"
echo "Run All Base Tasks Once"
echo "========================================================================"
echo "backend=${BACKEND}"
echo "policy=${POLICY}"
echo "isaac_render=${ISAAC_RENDER}"
echo "episodes_per_task=${EPISODES_PER_TASK}"
echo "save_video=${SAVE_VIDEO}"
echo "task_count=${#BASE_TASK_IDS[@]}"
echo "========================================================================"

success_count=0
failed_tasks=()

for task_id in "${BASE_TASK_IDS[@]}"; do
  echo
  echo "------------------------------------------------------------------------"
  echo "[RUN] task_id=${task_id}"
  echo "------------------------------------------------------------------------"
  if (
    cd "${REPO_ROOT}" && \
    TASK_IDS="${task_id}" \
    BACKEND="${BACKEND}" \
    POLICY="${POLICY}" \
    ISAAC_RENDER="${ISAAC_RENDER}" \
    EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
    SAVE_VIDEO="${SAVE_VIDEO}" \
    bash "${RUN_SUITE_SCRIPT}"
  ); then
    success_count=$((success_count + 1))
    echo "[OK]  ${task_id}"
  else
    failed_tasks+=("${task_id}")
    echo "[FAIL] ${task_id}" >&2
  fi
done

echo
echo "========================================================================"
echo "Run Summary"
echo "========================================================================"
echo "total=${#BASE_TASK_IDS[@]}"
echo "success=${success_count}"
echo "failed=${#failed_tasks[@]}"
if [[ "${#failed_tasks[@]}" -gt 0 ]]; then
  printf 'failed_task_ids=%s\n' "${failed_tasks[*]}"
fi
echo "========================================================================"

if [[ "${#failed_tasks[@]}" -gt 0 ]]; then
  exit 1
fi
