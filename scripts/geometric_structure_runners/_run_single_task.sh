#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEBUG_WRAPPER="${REPO_ROOT}/scripts/debug_protocol_geometric_inference.sh"

TASK_ID="${TASK_ID:?TASK_ID is required}"
SCENE_ID="${SCENE_ID:?SCENE_ID is required}"
LAYOUT_KEY="${LAYOUT_KEY:?LAYOUT_KEY is required}"
TASK_INDEX="${TASK_INDEX:?TASK_INDEX is required}"

AGENT="${AGENT:-${AGENT_DEFAULT:-oracle}}"
AGENT_SEED="${AGENT_SEED:-0}"
AGENT_EPISODES="${AGENT_EPISODES:-1}"
EPSILON_POS="${EPSILON_POS:-0.05}"
TAU_LINE="${TAU_LINE:-0.02}"
TAU_TREND="${TAU_TREND:-0.03}"
TAU_YAW="${TAU_YAW:-0.35}"
POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS:-24}"
HEADLESS="${HEADLESS:-0}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
ANNOTATIONS_JSON="${ANNOTATIONS_JSON:-}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/geometric_structure_tasks/${TIMESTAMP}}"

sanitize_slug() {
  printf '%s' "$1" | tr ' /:' '_' | tr -cd '[:alnum:]_.-'
}

run_label="${TASK_ID}__${AGENT}"
run_slug="$(sanitize_slug "${run_label}")"
summary_path="${OUTPUT_DIR}/${run_slug}.summary.json"
log_path="${OUTPUT_DIR}/${run_slug}.log"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "GeoMind Geometric Structure Single Task Runner"
echo "========================================================================"
echo "task_id=${TASK_ID}"
echo "scene_id=${SCENE_ID}"
echo "layout_key=${LAYOUT_KEY}"
echo "task_index=${TASK_INDEX}"
echo "agent=${AGENT}"
echo "agent_seed=${AGENT_SEED}"
echo "agent_episodes=${AGENT_EPISODES}"
echo "tau_trend=${TAU_TREND}"
echo "tau_yaw=${TAU_YAW}"
echo "output_dir=${OUTPUT_DIR}"
echo "summary_json=${summary_path}"
echo "task_log=${log_path}"
echo "========================================================================"

env_args=(
  "CASE=single"
  "SCENE_ID=${SCENE_ID}"
  "LAYOUT_KEY=${LAYOUT_KEY}"
  "TASK_INDEX=${TASK_INDEX}"
  "AGENT=${AGENT}"
  "AGENT_SEED=${AGENT_SEED}"
  "AGENT_EPISODES=${AGENT_EPISODES}"
  "EPSILON_POS=${EPSILON_POS}"
  "TAU_LINE=${TAU_LINE}"
  "TAU_TREND=${TAU_TREND}"
  "TAU_YAW=${TAU_YAW}"
  "POST_PLACE_SETTLE_STEPS=${POST_PLACE_SETTLE_STEPS}"
  "HEADLESS=${HEADLESS}"
  "DISABLE_FABRIC=${DISABLE_FABRIC}"
  "SUMMARY_JSON_OUT=${summary_path}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
)

if [[ -n "${ANNOTATIONS_JSON}" ]]; then
  env_args+=("GEOMIND_LAB_TASK_SPECS_PATH=${ANNOTATIONS_JSON}")
fi

env "${env_args[@]}" \
  bash "${DEBUG_WRAPPER}" "$@" | tee "${log_path}"

python - "${summary_path}" "${TASK_ID}" "${SCENE_ID}" "${LAYOUT_KEY}" "${TASK_INDEX}" "${AGENT}" "${AGENT_SEED}" "${log_path}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
payload = json.loads(summary_path.read_text(encoding="utf-8"))
payload["task_family"] = "geometric_structure"
payload["task_id"] = sys.argv[2]
payload["scene_id"] = int(sys.argv[3])
payload["layout_key"] = sys.argv[4]
payload["task_index"] = int(sys.argv[5])
payload["agent_name"] = sys.argv[6]
payload["task_agent_seed"] = int(sys.argv[7])
payload["summary_json"] = str(summary_path)
payload["task_log"] = sys.argv[8]
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
PY

echo "done"
echo "summary_json=${summary_path}"
echo "task_log=${log_path}"
