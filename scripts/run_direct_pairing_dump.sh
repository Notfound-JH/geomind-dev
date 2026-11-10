#!/usr/bin/env bash
# Driver for verification #1/#2/#4: dump one (task, level, trial) per process.
#
# Building a second IsaacLab env inside an already-booted process is slower than a cold
# start and has no reuse benefit, so this driver spawns one process per task and lets
# check_direct_pairing.py --resume accumulate into a single JSON. It also kills leftover
# Isaac Sim processes between runs, because they do not always exit on their own and a
# lingering instance steals the GPU from the next run.
#
#   bash scripts/run_direct_pairing_dump.sh base
#   bash scripts/run_direct_pairing_dump.sh direct
#
# Env overrides: PYTHON_BIN LEVELS TRIALS BASE_SEED OUT_JSON FRAMES_DIR LOG_DIR TASKS

set -uo pipefail

ABLATION="${1:-}"
if [[ "${ABLATION}" != "base" && "${ABLATION}" != "direct" ]]; then
  echo "usage: $0 {base|direct}" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  source "${REPO_ROOT}/scripts/lib/geomind_conda.sh"
  geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1
  PYTHON_BIN="$(command -v python)"
fi
LEVELS="${LEVELS:-loose}"
TRIALS="${TRIALS:-0}"
BASE_SEED="${BASE_SEED:-20260502}"
OUT_JSON="${OUT_JSON:-docs/direct_ablation/pairing_${ABLATION}.json}"
FRAMES_DIR="${FRAMES_DIR:-docs/direct_ablation/frames/${ABLATION}}"
LOG_DIR="${LOG_DIR:-docs/direct_ablation/logs/${ABLATION}}"
PER_TASK_TIMEOUT="${PER_TASK_TIMEOUT:-1800}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[dump] python interpreter not found: ${PYTHON_BIN}" >&2
  echo "[dump] set PYTHON_BIN to the Isaac Sim interpreter (the one used by debug scripts)" >&2
  exit 2
fi

if [[ -n "${TASKS:-}" ]]; then
  IFS=',' read -r -a TASK_LIST <<<"${TASKS}"
else
  mapfile -t TASK_LIST < <(
    find geomind/configs/vla_tasks -maxdepth 1 -name '*.json' ! -name 'legacy_layouts.json' \
      -printf '%f\n' | sed 's/\.json$//' | sort
  )
fi

mkdir -p "${LOG_DIR}" "$(dirname "${OUT_JSON}")"

kill_stale_sims() {
  # Isaac Sim frequently leaves the python process alive after the script returns; a
  # leftover instance holds ~3 GB of VRAM and starves the next run.
  local pids
  pids="$(pgrep -f 'check_direct_pairing\.py' || true)"
  if [[ -n "${pids}" ]]; then
    echo "[dump] killing leftover sim pids: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
    sleep 5
  fi
}

total="${#TASK_LIST[@]}"
index=0
failed=()

for task_id in "${TASK_LIST[@]}"; do
  index=$((index + 1))
  kill_stale_sims
  log_file="${LOG_DIR}/${task_id}.log"
  echo "[dump] (${index}/${total}) ${ABLATION} ${task_id} -> ${log_file}"
  HEADLESS=1 ENABLE_CAMERAS=1 timeout "${PER_TASK_TIMEOUT}" "${PYTHON_BIN}" \
    scripts/check_direct_pairing.py \
    --mode dump \
    --ablation "${ABLATION}" \
    --tasks "${task_id}" \
    --levels "${LEVELS}" \
    --trials "${TRIALS}" \
    --base-seed "${BASE_SEED}" \
    --out "${OUT_JSON}" \
    --frames-dir "${FRAMES_DIR}" \
    --resume \
    >"${log_file}" 2>&1
  status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "[dump]   exit=${status} (see ${log_file})"
  fi
  if ! grep -q "^\[dump\] ${task_id}|" "${log_file}"; then
    echo "[dump]   FAILED: no entry recorded for ${task_id}"
    failed+=("${task_id}")
  else
    grep "^\[dump\] ${task_id}|" "${log_file}" | sed 's/^/[dump]   /'
  fi
done

kill_stale_sims

echo
echo "[dump] ablation=${ABLATION} tasks=${total} failed=${#failed[@]}"
for task_id in "${failed[@]:-}"; do
  [[ -n "${task_id}" ]] && echo "[dump]   FAIL ${task_id}"
done
echo "[dump] out=${OUT_JSON}"
[[ ${#failed[@]} -eq 0 ]]
