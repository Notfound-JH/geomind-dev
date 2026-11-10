#!/usr/bin/env bash
set -euo pipefail
export MDL_SYSTEM_PATH="${MDL_SYSTEM_PATH:-${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_SCRIPT="${SCRIPT_DIR}/run_geometric_structure_protocol_baselines.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Run:
#   bash scripts/debug_protocol_geometric_inference.sh
#   bash scripts/run_geometric_structure_protocol_baselines.sh geometric_inference_primary_oracle
#
# Override inline if needed, for example:
#   AGENT=random AGENT_SEED=11 AGENT_EPISODES=5 HEADLESS=1 bash scripts/debug_protocol_geometric_inference.sh geometric_inference_primary_random
#   bash scripts/debug_protocol_geometric_inference.sh --variant standard --scene kitchen_table --object-mode geometric

CASE="${CASE:-single}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  CASE="${1}"
  shift
fi

case "${CASE}" in
  geometric_inference_primary_oracle|geometric_inference_primary_random|all)
    bash "${RUNNER_SCRIPT}" "${CASE}" "$@"
    exit 0
    ;;
esac

SCENE_ID="${SCENE_ID:-}"
TASK_ID="${TASK_ID:-perfect_midpoint}"
LAYOUT_KEY="${LAYOUT_KEY:-}"
TASK_INDEX="${TASK_INDEX:-0}"
EVAL_SUITE="${EVAL_SUITE:-}"
AGENT="${AGENT:-oracle}"
AGENT_SEED="${AGENT_SEED:-0}"
AGENT_EPISODES="${AGENT_EPISODES:-1}"
EPSILON_POS="${EPSILON_POS:-0.05}"
TAU_LINE="${TAU_LINE:-0.02}"
TAU_TREND="${TAU_TREND:-0.03}"
TAU_YAW="${TAU_YAW:-0.35}"
POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS:-24}"
SUMMARY_JSON_OUT="${SUMMARY_JSON_OUT:-}"
TRAJECTORY_H5_OUT="${TRAJECTORY_H5_OUT:-}"
TRAJECTORY_SAMPLE_HZ="${TRAJECTORY_SAMPLE_HZ:-10}"
TRAJECTORY_SEGMENT_SOURCE="${TRAJECTORY_SEGMENT_SOURCE:-stage_v1}"
HEADLESS="${HEADLESS:-0}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/geometric_structure_protocol_baselines/${TIMESTAMP}}"

if [[ "${HEADLESS}" == "1" ]]; then
  export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
fi

VIEWER_EYE=(${VIEWER_EYE:-2.2 2.2 1.8})
VIEWER_LOOKAT=(${VIEWER_LOOKAT:-0.3 0.0 0.8})

source "${SCRIPT_DIR}/lib/geomind_conda.sh"
geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1

build_base_args() {
  local args=(
    --task-id "${TASK_ID}"
    --task-index "${TASK_INDEX}"
    --agent "${AGENT}"
    --agent-seed "${AGENT_SEED}"
    --agent-episodes "${AGENT_EPISODES}"
    --epsilon-pos "${EPSILON_POS}"
    --tau-line "${TAU_LINE}"
    --tau-trend "${TAU_TREND}"
    --tau-yaw "${TAU_YAW}"
    --post-place-settle-steps "${POST_PLACE_SETTLE_STEPS}"
    --trajectory-sample-hz "${TRAJECTORY_SAMPLE_HZ}"
    --trajectory-segment-source "${TRAJECTORY_SEGMENT_SOURCE}"
    --viewer-eye "${VIEWER_EYE[@]}"
    --viewer-lookat "${VIEWER_LOOKAT[@]}"
  )

  if [[ -n "${SCENE_ID}" ]]; then
    args+=(--scene-id "${SCENE_ID}")
  fi

  if [[ -n "${SCENE:-}" ]]; then
    args+=(--scene "${SCENE}")
  fi

  if [[ -n "${TASK_VARIANT:-${VARIANT:-}}" ]]; then
    args+=(--variant "${TASK_VARIANT:-${VARIANT:-}}")
  fi

  if [[ -n "${OBJECT_MODE:-}" ]]; then
    args+=(--object-mode "${OBJECT_MODE}")
  fi

  if [[ -n "${LAYOUT_KEY}" ]]; then
    args+=(--layout-key "${LAYOUT_KEY}")
  fi

  if [[ -n "${EVAL_SUITE}" ]]; then
    args+=(--eval-suite "${EVAL_SUITE}")
  fi

  if [[ -n "${SUMMARY_JSON_OUT}" ]]; then
    args+=(--summary-json-out "${SUMMARY_JSON_OUT}")
  fi

  if [[ -n "${TRAJECTORY_H5_OUT}" ]]; then
    args+=(--trajectory-h5-out "${TRAJECTORY_H5_OUT}")
  fi

  if [[ "${HEADLESS}" == "1" ]]; then
    args+=(--headless)
  fi

  if [[ "${DISABLE_FABRIC}" == "1" ]]; then
    args+=(--disable-fabric)
  fi

  printf '%s\n' "${args[@]}"
}

run_python() {
  local args=()
  while IFS= read -r arg; do
    args+=("${arg}")
  done < <(build_base_args)
  python "${SCRIPT_DIR}/debug_protocol_geometric_inference.py" "${args[@]}" "$@"
}

run_suite_case() {
  local suite_agent="$1"
  shift
  local suite_name="geometric_inference_primary_${suite_agent}"
  local suite_summary_path="${SUMMARY_JSON_OUT}"
  mkdir -p "${OUTPUT_DIR}"

  if [[ -z "${suite_summary_path}" ]]; then
    suite_summary_path="${OUTPUT_DIR}/${suite_name}.summary.json"
  fi

  echo "========================================================================"
  echo "GeoMind Geometric Structure Baseline Runner"
  echo "========================================================================"
  echo "case=${suite_name}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "agent_episodes=${AGENT_EPISODES}"
  echo "headless=${HEADLESS}"
  echo "headless_camera_video=${HEADLESS_CAMERA_VIDEO:-}"
  echo "summary_json=${suite_summary_path}"
  echo "========================================================================"

  AGENT="${suite_agent}" \
  EVAL_SUITE="geometric_inference_primary" \
  SUMMARY_JSON_OUT="${suite_summary_path}" \
  run_python "$@"
}

case "${CASE}" in
  single)
    run_python "$@"
    ;;
  geometric_inference_primary_oracle)
    run_suite_case oracle "$@"
    ;;
  geometric_inference_primary_random)
    run_suite_case random "$@"
    ;;
  all)
    mkdir -p "${OUTPUT_DIR}"
    oracle_summary="${OUTPUT_DIR}/geometric_inference_primary_oracle.summary.json"
    random_summary="${OUTPUT_DIR}/geometric_inference_primary_random.summary.json"

    SUMMARY_JSON_OUT="${oracle_summary}" run_suite_case oracle "$@"
    SUMMARY_JSON_OUT="${random_summary}" run_suite_case random "$@"

    python - "${oracle_summary}" "${random_summary}" <<'PY'
import json
import sys
from pathlib import Path

oracle = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
random = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

delta = {
    "oracle_success_rate_minus_random": float(oracle.get("success_rate", 0.0)) - float(random.get("success_rate", 0.0)),
    "oracle_l2_error_mean_minus_random": (
        None
        if oracle.get("l2_error_mean") is None or random.get("l2_error_mean") is None
        else float(oracle.get("l2_error_mean")) - float(random.get("l2_error_mean"))
    ),
    "oracle_rule_satisfied_rate_minus_random": float(oracle.get("rule_satisfied_rate", 0.0)) - float(random.get("rule_satisfied_rate", 0.0)),
}

print("========================================================================")
print("geometric_inference_primary_comparison")
print("========================================================================")
print(json.dumps(
    {
        "oracle_summary_json": sys.argv[1],
        "random_summary_json": sys.argv[2],
        "delta": delta,
    },
    indent=2,
    sort_keys=True,
))
PY
    ;;
  *)
    echo "Unknown case: ${CASE}" >&2
    echo "Usage: bash scripts/debug_protocol_geometric_inference.sh [single|geometric_inference_primary_oracle|geometric_inference_primary_random|all] [extra args...]" >&2
    exit 1
    ;;
esac
