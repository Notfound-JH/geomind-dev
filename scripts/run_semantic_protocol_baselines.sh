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

CASE="${1:-semantic_oracle}"
if [[ $# -gt 0 ]]; then
  shift
fi

HEADLESS="${HEADLESS:-1}"
AGENT_EPISODES="${AGENT_EPISODES:-5}"
POST_INTERVENTION_SETTLE_STEPS="${POST_INTERVENTION_SETTLE_STEPS:-60}"
PROTOCOL_DEBUG_INTERVAL="${PROTOCOL_DEBUG_INTERVAL:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/semantic_protocol_baselines/${TIMESTAMP}}"

mkdir -p "${OUTPUT_DIR}"

print_header() {
  echo "========================================================================"
  echo "GeoMind Semantic Protocol Runner"
  echo "========================================================================"
  echo "case=${CASE}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "agent_episodes=${AGENT_EPISODES}"
  echo "headless=${HEADLESS}"
  echo "summary_only=${SUMMARY_ONLY}"
  echo "========================================================================"
}

run_suite() {
  local case_name="$1"
  local eval_suite="$2"
  local agent_name="$3"
  shift 3

  local summary_path="${OUTPUT_DIR}/${case_name}.summary.json"
  local log_path="${OUTPUT_DIR}/${case_name}.log"
  local wrapper_args=(
    --agent "${agent_name}"
    --eval-suite "${eval_suite}"
    --agent-episodes "${AGENT_EPISODES}"
    --summary-json-out "${summary_path}"
  )

  if [[ "${SUMMARY_ONLY}" == "1" ]]; then
    wrapper_args+=(--summary-only)
  fi

  echo
  echo "------------------------------------------------------------------------"
  echo "Case: ${case_name}"
  echo "Eval suite: ${eval_suite}"
  echo "Agent: ${agent_name}"
  echo "summary_json=${summary_path}"
  echo "task_log=${log_path}"
  echo "------------------------------------------------------------------------"

  HEADLESS="${HEADLESS}" \
  POST_INTERVENTION_SETTLE_STEPS="${POST_INTERVENTION_SETTLE_STEPS}" \
  PROTOCOL_DEBUG_INTERVAL="${PROTOCOL_DEBUG_INTERVAL}" \
  bash "${DEBUG_WRAPPER}" \
    "${wrapper_args[@]}" \
    "$@" | tee "${log_path}"
}

run_task() {
  local case_name="$1"
  local scene_id="$2"
  local layout_key="$3"
  local task_index="$4"
  local agent_name="$5"
  shift 5

  local summary_path="${OUTPUT_DIR}/${case_name}.summary.json"
  local log_path="${OUTPUT_DIR}/${case_name}.log"
  local wrapper_args=(
    --scene-id "${scene_id}"
    --layout-key "${layout_key}"
    --task-index "${task_index}"
    --agent "${agent_name}"
    --agent-episodes "${AGENT_EPISODES}"
    --summary-json-out "${summary_path}"
  )

  if [[ "${SUMMARY_ONLY}" == "1" ]]; then
    wrapper_args+=(--summary-only)
  fi

  echo
  echo "------------------------------------------------------------------------"
  echo "Case: ${case_name}"
  echo "Scene: ${scene_id}"
  echo "Layout: ${layout_key}"
  echo "Task index: ${task_index}"
  echo "Agent: ${agent_name}"
  echo "summary_json=${summary_path}"
  echo "task_log=${log_path}"
  echo "------------------------------------------------------------------------"

  HEADLESS="${HEADLESS}" \
  POST_INTERVENTION_SETTLE_STEPS="${POST_INTERVENTION_SETTLE_STEPS}" \
  PROTOCOL_DEBUG_INTERVAL="${PROTOCOL_DEBUG_INTERVAL}" \
  bash "${DEBUG_WRAPPER}" \
    "${wrapper_args[@]}" \
    "$@" | tee "${log_path}"
}

print_header

case "${CASE}" in
  semantic_oracle)
    run_suite "semantic_oracle" "semantic_protocol_minimal" "oracle" "$@"
    ;;
  semantic_random)
    run_suite "semantic_random" "semantic_protocol_minimal" "random" "$@"
    ;;
  position_oracle)
    run_suite "position_oracle" "object_position_update_minimal" "oracle" "$@"
    ;;
  identity_oracle)
    run_suite "identity_oracle" "object_identity_tracking_minimal" "oracle" "$@"
    ;;
  identity_final_frame_only)
    run_suite "identity_final_frame_only" "object_identity_tracking_minimal" "random" --control-mode final_frame_only "$@"
    ;;
  distractor_oracle)
    run_suite "distractor_oracle" "reorder_with_distractors_minimal" "oracle" "$@"
    ;;
  all)
    echo "all mode runs one task per IsaacLab process." 

    run_task "semantic_oracle_object_position_update_edge_line_3" 17 "object_position_update_edge_line_3" 0 "oracle" "$@"
    run_task "semantic_oracle_object_position_update_insert_line_4" 1 "object_position_update_insert_line_4" 0 "oracle" "$@"
    run_task "semantic_oracle_object_identity_tracking_single_step_line_3" 17 "object_identity_tracking_single_step_line_3" 0 "oracle" "$@"
    run_task "semantic_oracle_object_identity_tracking_multistep_line_5" 23 "object_identity_tracking_multistep_line_5" 0 "oracle" "$@"
    run_task "semantic_oracle_reorder_with_distractors_filter_line_6" 21 "reorder_with_distractors_filter_line_6" 0 "oracle" "$@"
    run_task "semantic_oracle_reorder_with_distractors_subset_line_5" 26 "reorder_with_distractors_subset_line_5" 0 "oracle" "$@"

    run_task "semantic_random_object_position_update_edge_line_3" 17 "object_position_update_edge_line_3" 0 "random" "$@"
    run_task "semantic_random_object_position_update_insert_line_4" 1 "object_position_update_insert_line_4" 0 "random" "$@"
    run_task "semantic_random_object_identity_tracking_single_step_line_3" 17 "object_identity_tracking_single_step_line_3" 0 "random" "$@"
    run_task "semantic_random_object_identity_tracking_multistep_line_5" 23 "object_identity_tracking_multistep_line_5" 0 "random" "$@"
    run_task "semantic_random_reorder_with_distractors_filter_line_6" 21 "reorder_with_distractors_filter_line_6" 0 "random" "$@"
    run_task "semantic_random_reorder_with_distractors_subset_line_5" 26 "reorder_with_distractors_subset_line_5" 0 "random" "$@"

    run_task "identity_final_frame_only_object_identity_tracking_single_step_line_3" 17 "object_identity_tracking_single_step_line_3" 0 "random" --control-mode final_frame_only "$@"
    run_task "identity_final_frame_only_object_identity_tracking_multistep_line_5" 23 "object_identity_tracking_multistep_line_5" 0 "random" --control-mode final_frame_only "$@"
    ;;
  *)
    echo "Unknown case: ${CASE}" >&2
    echo "Usage: bash scripts/run_semantic_protocol_baselines.sh [semantic_oracle|semantic_random|position_oracle|identity_oracle|identity_final_frame_only|distractor_oracle|all] [extra args...]" >&2
    exit 1
    ;;
esac

echo
echo "========================================================================"
echo "done"
echo "output_dir=${OUTPUT_DIR}"
echo "========================================================================"
