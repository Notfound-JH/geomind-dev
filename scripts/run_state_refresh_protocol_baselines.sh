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
DEBUG_WRAPPER="${SCRIPT_DIR}/debug_protocol_state_reindex.sh"

CASE="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

HEADLESS="${HEADLESS:-1}"
AGENT_EPISODES="${AGENT_EPISODES:-5}"
POST_INTERVENTION_SETTLE_STEPS="${POST_INTERVENTION_SETTLE_STEPS:-60}"
PROTOCOL_DEBUG_INTERVAL="${PROTOCOL_DEBUG_INTERVAL:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/state_refresh_protocol_baselines/${TIMESTAMP}}"
SUITE_SUMMARY_PATH="${OUTPUT_DIR}/task_suite_summary.json"

mkdir -p "${OUTPUT_DIR}"

SUMMARY_FILES=()

print_suite_header() {
  echo "========================================================================"
  echo "GeoMind State Refresh Protocol Baseline Runner"
  echo "========================================================================"
  echo "case=${CASE}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "agent_episodes=${AGENT_EPISODES}"
  echo "headless=${HEADLESS}"
  echo "summary_only=${SUMMARY_ONLY}"
  echo "========================================================================"
}

build_wrapper_args() {
  local args=(
    --agent-episodes "${AGENT_EPISODES}"
  )
  if [[ "${SUMMARY_ONLY}" == "1" ]]; then
    args+=(--summary-only)
  fi
  printf '%s\n' "${args[@]}"
}

run_task() {
  local case_name="$1"
  local task_label="$2"
  local scene_id="$3"
  local layout_key="$4"
  local task_index="$5"
  shift 5

  local wrapper_args=()
  local task_slug
  local summary_path
  local log_path
  while IFS= read -r arg; do
    wrapper_args+=("${arg}")
  done < <(build_wrapper_args)
  task_slug="$(printf '%s__%s__scene%s__task%s' "${case_name}" "${task_label}" "${scene_id}" "${task_index}" | tr ' /' '_' | tr -cd '[:alnum:]_.-')"
  summary_path="${OUTPUT_DIR}/${task_slug}.summary.json"
  log_path="${OUTPUT_DIR}/${task_slug}.log"

  echo
  echo "------------------------------------------------------------------------"
  echo "Case: ${case_name}"
  echo "Task: ${task_label}"
  echo "scene_id=${scene_id} layout_key=${layout_key} task_index=${task_index}"
  echo "summary_json=${summary_path}"
  echo "task_log=${log_path}"
  echo "------------------------------------------------------------------------"

  HEADLESS="${HEADLESS}" \
  POST_INTERVENTION_SETTLE_STEPS="${POST_INTERVENTION_SETTLE_STEPS}" \
  PROTOCOL_DEBUG_INTERVAL="${PROTOCOL_DEBUG_INTERVAL}" \
  bash "${DEBUG_WRAPPER}" \
    "${wrapper_args[@]}" \
    --summary-json-out "${summary_path}" \
    --scene-id "${scene_id}" \
    --layout-key "${layout_key}" \
    --task-index "${task_index}" \
    "$@" | tee "${log_path}"

  python - "${summary_path}" "${case_name}" "${task_label}" "${log_path}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
case_name = sys.argv[2]
task_label = sys.argv[3]
log_path = sys.argv[4]

payload = json.loads(summary_path.read_text(encoding="utf-8"))
payload["case_name"] = case_name
payload["task_label"] = task_label
payload["summary_json"] = str(summary_path)
payload["task_log"] = log_path
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

  SUMMARY_FILES+=("${summary_path}")
}

write_suite_summary() {
  python - "${SUITE_SUMMARY_PATH}" "${OUTPUT_DIR}" "${CASE}" "${SUMMARY_FILES[@]}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

suite_summary_path = Path(sys.argv[1])
output_dir = sys.argv[2]
suite_case = sys.argv[3]
summary_paths = [Path(arg) for arg in sys.argv[4:]]

def rate(count: int, total: int) -> float:
    return float(count) / float(total) if total > 0 else 0.0

def aggregate(items):
    total_episodes = sum(int(item.get("num_episodes", 0)) for item in items)
    total_valid = sum(int(item.get("valid_episode_count", 0)) for item in items)
    total_success = sum(int(item.get("success_count", 0)) for item in items)
    total_selection_match = sum(int(item.get("selection_match_count", 0)) for item in items)
    total_grasp_detected = sum(int(item.get("grasp_detected_count", 0)) for item in items)
    total_intervention_success = sum(int(item.get("intervention_success_count", 0)) for item in items)
    total_wrong_selection = sum(int(item.get("wrong_selection_count", 0)) for item in items)
    total_no_grasp = sum(int(item.get("no_grasp_count", 0)) for item in items)
    total_intervention_failed = sum(int(item.get("intervention_failed_count", 0)) for item in items)
    total_episode_invalid = sum(int(item.get("episode_invalid_count", 0)) for item in items)
    total_grasped_wrong_object = sum(int(item.get("grasped_wrong_object_count", 0)) for item in items)
    return {
        "task_count": len(items),
        "num_episodes": total_episodes,
        "valid_episode_count": total_valid,
        "success_count": total_success,
        "selection_match_count": total_selection_match,
        "grasp_detected_count": total_grasp_detected,
        "intervention_success_count": total_intervention_success,
        "success_rate": rate(total_success, total_episodes),
        "valid_episode_rate": rate(total_valid, total_episodes),
        "selection_accuracy": rate(total_selection_match, total_episodes),
        "grasp_detection_rate": rate(total_grasp_detected, total_episodes),
        "wrong_selection_count": total_wrong_selection,
        "wrong_selection_rate": rate(total_wrong_selection, total_episodes),
        "no_grasp_count": total_no_grasp,
        "no_grasp_rate": rate(total_no_grasp, total_episodes),
        "intervention_failed_count": total_intervention_failed,
        "intervention_failed_rate": rate(total_intervention_failed, total_episodes),
        "episode_invalid_count": total_episode_invalid,
        "episode_invalid_rate": rate(total_episode_invalid, total_episodes),
        "grasped_wrong_object_count": total_grasped_wrong_object,
        "grasped_wrong_object_rate": rate(total_grasped_wrong_object, total_episodes),
    }

task_summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
case_buckets = defaultdict(list)
for item in task_summaries:
    case_buckets[str(item.get("case_name", "unknown"))].append(item)

case_summaries = {}
for case_name, items in sorted(case_buckets.items()):
    case_summary = aggregate(items)
    case_summary["case_name"] = case_name
    case_summary["task_refs"] = [
        {
            "task_label": item.get("task_label"),
            "layout_key": item.get("layout_key"),
            "task_index": item.get("task_index"),
            "summary_json": item.get("summary_json"),
        }
        for item in items
    ]
    case_summaries[case_name] = case_summary

suite_summary = aggregate(task_summaries)
suite_summary["suite_case"] = suite_case
suite_summary["output_dir"] = output_dir
suite_summary["summary_json"] = str(suite_summary_path)
suite_summary["task_summaries"] = task_summaries
suite_summary["case_summaries"] = case_summaries

suite_summary_path.parent.mkdir(parents=True, exist_ok=True)
suite_summary_path.write_text(json.dumps(suite_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(suite_summary, indent=2, sort_keys=True))
PY
}

run_case_state_refresh_core_oracle() {
  run_task "state_refresh_core_oracle" "current_leftmost_after_left_shift" 4 state_reindex_blocks_line_4 0 --agent oracle "$@"
  run_task "state_refresh_core_oracle" "current_rightmost_after_left_shift" 4 state_reindex_blocks_line_4 1 --agent oracle "$@"
  run_task "state_refresh_core_oracle" "current_leftmost_after_right_shift" 4 state_reindex_blocks_line_4 2 --agent oracle "$@"
}

run_case_state_refresh_core_random() {
  run_task "state_refresh_core_random" "current_leftmost_after_left_shift" 4 state_reindex_blocks_line_4 0 --agent random "$@"
  run_task "state_refresh_core_random" "current_rightmost_after_left_shift" 4 state_reindex_blocks_line_4 1 --agent random "$@"
  run_task "state_refresh_core_random" "current_leftmost_after_right_shift" 4 state_reindex_blocks_line_4 2 --agent random "$@"
}

run_case_historical_oracle() {
  run_task "historical_oracle" "original_middle" 1 historical_identity_line_3 0 --agent oracle "$@"
}

run_case_historical_final_frame_only() {
  run_task "historical_final_frame_only" "original_middle" 1 historical_identity_line_3 0 --agent random --control-mode final_frame_only "$@"
}

print_suite_header

case "${CASE}" in
  state_refresh_core_oracle)
    run_case_state_refresh_core_oracle "$@"
    ;;
  state_refresh_core_random)
    run_case_state_refresh_core_random "$@"
    ;;
  historical_oracle)
    run_case_historical_oracle "$@"
    ;;
  historical_final_frame_only)
    run_case_historical_final_frame_only "$@"
    ;;
  all)
    run_case_state_refresh_core_oracle "$@"
    run_case_state_refresh_core_random "$@"
    run_case_historical_oracle "$@"
    run_case_historical_final_frame_only "$@"
    ;;
  *)
    echo "Unknown case: ${CASE}" >&2
    echo "Usage: bash scripts/run_state_refresh_protocol_baselines.sh [all|state_refresh_core_oracle|state_refresh_core_random|historical_oracle|historical_final_frame_only] [extra args...]" >&2
    exit 1
    ;;
esac

echo
echo "========================================================================"
echo "task_suite_summary"
echo "========================================================================"
write_suite_summary
echo "========================================================================"
echo "suite_summary_json=${SUITE_SUMMARY_PATH}"
echo "========================================================================"
