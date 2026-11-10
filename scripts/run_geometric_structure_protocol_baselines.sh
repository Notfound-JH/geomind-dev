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
DEBUG_WRAPPER="${SCRIPT_DIR}/debug_protocol_geometric_inference.sh"

CASE="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

HEADLESS="${HEADLESS:-0}"
AGENT_EPISODES="${AGENT_EPISODES:-5}"
AGENT_SEED="${AGENT_SEED:-0}"
EPSILON_POS="${EPSILON_POS:-0.05}"
TAU_LINE="${TAU_LINE:-0.02}"
POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS:-24}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/geometric_structure_protocol_baselines/${TIMESTAMP}}"

mkdir -p "${OUTPUT_DIR}"

SUMMARY_FILES=()

print_header() {
  echo "========================================================================"
  echo "GeoMind Geometric Structure Protocol Baseline Runner"
  echo "========================================================================"
  echo "case=${CASE}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "agent_episodes=${AGENT_EPISODES}"
  echo "agent_seed=${AGENT_SEED}"
  echo "headless=${HEADLESS}"
  echo "========================================================================"
}

resolve_geometric_inference_primary_tasks() {
  python - "${REPO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
payload = json.loads((repo_root / "assets" / "annotations" / "annotated_tasks.json").read_text(encoding="utf-8"))

for layout_key, layout in sorted(payload.items()):
    if not isinstance(layout, dict):
        continue
    scene_id = int(layout.get("scene_id", 1))
    tasks = layout.get("tasks", [])
    if not isinstance(tasks, list):
        continue
    for task_index, task_payload in enumerate(tasks):
        if not isinstance(task_payload, dict):
            continue
        annotation = task_payload.get("annotation", {})
        if not isinstance(annotation, dict):
            continue
        protocol = annotation.get("protocol", {})
        if not isinstance(protocol, dict):
            continue
        gt_spec = protocol.get("gt_spec", {})
        if (
            str(protocol.get("module_name", "")) == "geometric_inference"
            and str(protocol.get("track_name", "")) == "explicit_rule_extrapolation"
            and str(protocol.get("query_family", "")) == "explicit_rule_query"
            and isinstance(gt_spec, dict)
            and str(gt_spec.get("rule_type", "")) == "extension"
        ):
            label = annotation.get("task_id") or task_payload.get("instruction") or f"{layout_key}__task_{task_index}"
            print(f"{scene_id}\t{layout_key}\t{task_index}\t{label}")
PY
}

run_task() {
  local case_name="$1"
  local task_label="$2"
  local scene_id="$3"
  local layout_key="$4"
  local task_index="$5"
  local agent_name="$6"
  local task_ordinal="$7"
  shift 7

  local task_agent_seed=$((AGENT_SEED + task_ordinal * 100003))
  local task_slug
  local summary_path
  local log_path

  task_slug="$(printf '%s__%s__scene%s__task%s' "${case_name}" "${task_label}" "${scene_id}" "${task_index}" | tr ' /' '_' | tr -cd '[:alnum:]_.-')"
  summary_path="${OUTPUT_DIR}/${task_slug}.summary.json"
  log_path="${OUTPUT_DIR}/${task_slug}.log"

  echo
  echo "------------------------------------------------------------------------"
  echo "Case: ${case_name}"
  echo "Task: ${task_label}"
  echo "scene_id=${scene_id} layout_key=${layout_key} task_index=${task_index}"
  echo "agent=${agent_name} task_agent_seed=${task_agent_seed}"
  echo "summary_json=${summary_path}"
  echo "task_log=${log_path}"
  echo "------------------------------------------------------------------------"

  HEADLESS="${HEADLESS}" \
  AGENT_EPISODES="${AGENT_EPISODES}" \
  AGENT_SEED="${task_agent_seed}" \
  EPSILON_POS="${EPSILON_POS}" \
  TAU_LINE="${TAU_LINE}" \
  POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS}" \
  DISABLE_FABRIC="${DISABLE_FABRIC}" \
  SUMMARY_JSON_OUT="${summary_path}" \
  bash "${DEBUG_WRAPPER}" \
    --scene-id "${scene_id}" \
    --layout-key "${layout_key}" \
    --task-index "${task_index}" \
    --agent "${agent_name}" \
    "$@" | tee "${log_path}"

  python - "${summary_path}" "${case_name}" "${task_label}" "${task_agent_seed}" "${log_path}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
case_name = sys.argv[2]
task_label = sys.argv[3]
task_agent_seed = int(sys.argv[4])
log_path = sys.argv[5]

payload = json.loads(summary_path.read_text(encoding="utf-8"))
payload["case_name"] = case_name
payload["task_label"] = task_label
payload["task_agent_seed"] = task_agent_seed
payload["summary_json"] = str(summary_path)
payload["task_log"] = log_path
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
PY

  SUMMARY_FILES+=("${summary_path}")
}

write_suite_summary() {
  local case_name="$1"
  local suite_summary_path="${OUTPUT_DIR}/${case_name}.summary.json"

  python - "${suite_summary_path}" "${OUTPUT_DIR}" "${case_name}" "${SUMMARY_FILES[@]}" <<'PY'
import json
import sys
from pathlib import Path

suite_summary_path = Path(sys.argv[1])
output_dir = sys.argv[2]
case_name = sys.argv[3]
summary_paths = [Path(arg) for arg in sys.argv[4:]]

def rate(count: int, total: int) -> float:
    return float(count) / float(total) if total > 0 else 0.0

def weighted_mean(items, value_key: str, weight_key: str = "metric_episode_count"):
    numer = 0.0
    denom = 0
    for item in items:
      value = item.get(value_key)
      if value is None:
          continue
      weight = int(item.get(weight_key, 0))
      if weight <= 0:
          continue
      numer += float(value) * float(weight)
      denom += weight
    if denom <= 0:
        return None
    return numer / float(denom)

task_summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
total_episodes = sum(int(item.get("num_episodes", 0)) for item in task_summaries)
valid_episode_count = sum(int(item.get("valid_episode_count", 0)) for item in task_summaries)
success_count = sum(int(item.get("success_count", 0)) for item in task_summaries)
rule_satisfied_count = sum(int(item.get("rule_satisfied_count", 0)) for item in task_summaries)
grasp_execution_success_count = sum(int(item.get("grasp_execution_success_count", 0)) for item in task_summaries)
placement_execution_success_count = sum(int(item.get("placement_execution_success_count", 0)) for item in task_summaries)

suite_summary = {
    "suite_name": case_name,
    "agent_name": task_summaries[0].get("agent_name") if task_summaries else None,
    "num_tasks": len(task_summaries),
    "total_episodes": total_episodes,
    "valid_episode_count": valid_episode_count,
    "valid_episode_rate": rate(valid_episode_count, total_episodes),
    "success_count": success_count,
    "success_rate": rate(success_count, total_episodes),
    "metric_episode_count": sum(int(item.get("metric_episode_count", 0)) for item in task_summaries),
    "l2_error_mean": weighted_mean(task_summaries, "l2_error_mean"),
    "orthogonal_distance_mean": weighted_mean(task_summaries, "orthogonal_distance_mean"),
    "rule_satisfied_count": rule_satisfied_count,
    "rule_satisfied_rate": rate(rule_satisfied_count, total_episodes),
    "grasp_execution_success_count": grasp_execution_success_count,
    "grasp_execution_success_rate": rate(grasp_execution_success_count, total_episodes),
    "placement_execution_success_count": placement_execution_success_count,
    "placement_execution_success_rate": rate(placement_execution_success_count, total_episodes),
    "output_dir": output_dir,
    "summary_json": str(suite_summary_path),
    "task_results": task_summaries,
}

suite_summary_path.write_text(json.dumps(suite_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(suite_summary, indent=2, sort_keys=True, ensure_ascii=False))
PY
}

run_primary_case() {
  local case_name="$1"
  local agent_name="$2"
  shift 2

  SUMMARY_FILES=()
  local task_ordinal=0
  while IFS=$'\t' read -r scene_id layout_key task_index task_label; do
    if [[ -z "${scene_id}" ]]; then
      continue
    fi
    run_task "${case_name}" "${task_label}" "${scene_id}" "${layout_key}" "${task_index}" "${agent_name}" "${task_ordinal}" "$@"
    task_ordinal=$((task_ordinal + 1))
  done < <(resolve_geometric_inference_primary_tasks)

  echo
  echo "========================================================================"
  echo "suite_summary: ${case_name}"
  echo "========================================================================"
  write_suite_summary "${case_name}"
  echo "========================================================================"
  echo "suite_summary_json=${OUTPUT_DIR}/${case_name}.summary.json"
  echo "========================================================================"
}

print_comparison() {
  local oracle_summary="${OUTPUT_DIR}/geometric_inference_primary_oracle.summary.json"
  local random_summary="${OUTPUT_DIR}/geometric_inference_primary_random.summary.json"

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
    ensure_ascii=False,
))
PY
}

print_header

case "${CASE}" in
  geometric_inference_primary_oracle)
    run_primary_case "geometric_inference_primary_oracle" "oracle" "$@"
    ;;
  geometric_inference_primary_random)
    run_primary_case "geometric_inference_primary_random" "random" "$@"
    ;;
  all)
    run_primary_case "geometric_inference_primary_oracle" "oracle" "$@"
    run_primary_case "geometric_inference_primary_random" "random" "$@"
    print_comparison
    ;;
  *)
    echo "Unknown case: ${CASE}" >&2
    echo "Usage: bash scripts/run_geometric_structure_protocol_baselines.sh [all|geometric_inference_primary_oracle|geometric_inference_primary_random] [extra args...]" >&2
    exit 1
    ;;
esac

echo
echo "========================================================================"
echo "output_dir=${OUTPUT_DIR}"
echo "========================================================================"
