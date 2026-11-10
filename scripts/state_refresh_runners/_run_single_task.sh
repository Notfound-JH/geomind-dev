#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEBUG_WRAPPER="${REPO_ROOT}/scripts/debug_protocol_state_reindex.sh"

TASK_ID="${TASK_ID:?TASK_ID is required}"
SCENE_ID="${SCENE_ID:?SCENE_ID is required}"
LAYOUT_KEY="${LAYOUT_KEY:-}"
TASK_INDEX="${TASK_INDEX:-}"

AGENT="${AGENT:-${AGENT_DEFAULT:-oracle}}"
CONTROL_MODE="${CONTROL_MODE:-${CONTROL_MODE_DEFAULT:-}}"
HEADLESS="${HEADLESS:-1}"
AGENT_EPISODES="${AGENT_EPISODES:-5}"
POST_INTERVENTION_SETTLE_STEPS="${POST_INTERVENTION_SETTLE_STEPS:-60}"
PROTOCOL_DEBUG_INTERVAL="${PROTOCOL_DEBUG_INTERVAL:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-1}"
ANNOTATIONS_JSON="${ANNOTATIONS_JSON:-}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/state_refresh_tasks/${TIMESTAMP}}"

resolve_task_locator_from_task_id() {
  python - "${REPO_ROOT}" "${ANNOTATIONS_JSON}" "${TASK_ID}" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
annotations_path = Path(sys.argv[2]) if sys.argv[2].strip() else None
task_id = sys.argv[3]

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import resolve_task_locator  # noqa: E402

locator = resolve_task_locator(task_id, annotations_path)
print(f"{locator['layout_key']}\t{locator['task_index']}")
PY
}

if [[ -z "${LAYOUT_KEY}" || -z "${TASK_INDEX}" ]]; then
  IFS=$'\t' read -r resolved_layout_key resolved_task_index < <(resolve_task_locator_from_task_id)
  if [[ -z "${LAYOUT_KEY}" ]]; then
    LAYOUT_KEY="${resolved_layout_key}"
  fi
  if [[ -z "${TASK_INDEX}" ]]; then
    TASK_INDEX="${resolved_task_index}"
  fi
fi

sanitize_slug() {
  printf '%s' "$1" | tr ' /:' '_' | tr -cd '[:alnum:]_.-'
}

run_label="${TASK_ID}__${AGENT}"
if [[ -n "${CONTROL_MODE}" ]]; then
  run_label="${run_label}__${CONTROL_MODE}"
fi

run_slug="$(sanitize_slug "${run_label}")"
summary_path="${OUTPUT_DIR}/${run_slug}.summary.json"
log_path="${OUTPUT_DIR}/${run_slug}.log"

mkdir -p "${OUTPUT_DIR}"

wrapper_args=(
  --agent "${AGENT}"
  --agent-episodes "${AGENT_EPISODES}"
  --summary-json-out "${summary_path}"
)

if [[ "${SUMMARY_ONLY}" == "1" ]]; then
  wrapper_args+=(--summary-only)
fi

if [[ -n "${CONTROL_MODE}" ]]; then
  wrapper_args+=(--control-mode "${CONTROL_MODE}")
fi

echo "========================================================================"
echo "GeoMind State Refresh Single Task Runner"
echo "========================================================================"
echo "task_id=${TASK_ID}"
echo "scene_id=${SCENE_ID}"
echo "layout_key=${LAYOUT_KEY}"
echo "task_index=${TASK_INDEX}"
echo "agent=${AGENT}"
echo "control_mode=${CONTROL_MODE:-none}"
echo "output_dir=${OUTPUT_DIR}"
echo "summary_json=${summary_path}"
echo "task_log=${log_path}"
echo "========================================================================"

env_args=(
  "HEADLESS=${HEADLESS}"
  "SCENE_ID=${SCENE_ID}"
  "LAYOUT_KEY=${LAYOUT_KEY}"
  "TASK_INDEX=${TASK_INDEX}"
  "POST_INTERVENTION_SETTLE_STEPS=${POST_INTERVENTION_SETTLE_STEPS}"
  "PROTOCOL_DEBUG_INTERVAL=${PROTOCOL_DEBUG_INTERVAL}"
)

if [[ -n "${ANNOTATIONS_JSON}" ]]; then
  env_args+=("GEOMIND_LAB_TASK_SPECS_PATH=${ANNOTATIONS_JSON}")
fi

env "${env_args[@]}" \
  bash "${DEBUG_WRAPPER}" "${wrapper_args[@]}" "$@" | tee "${log_path}"

python - "${summary_path}" "${TASK_ID}" "${SCENE_ID}" "${LAYOUT_KEY}" "${TASK_INDEX}" "${AGENT}" "${CONTROL_MODE}" "${log_path}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
payload = json.loads(summary_path.read_text(encoding="utf-8"))
payload["task_family"] = "state_refresh"
payload["task_id"] = sys.argv[2]
payload["scene_id"] = int(sys.argv[3])
payload["layout_key"] = sys.argv[4]
payload["task_index"] = int(sys.argv[5])
payload["agent_name"] = sys.argv[6]
payload["control_mode"] = sys.argv[7] or None
payload["summary_json"] = str(summary_path)
payload["task_log"] = sys.argv[8]
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
PY

echo "done"
echo "summary_json=${summary_path}"
echo "task_log=${log_path}"
