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
ANNOTATIONS_JSON="${REPO_ROOT}/assets/annotations/annotated_tasks.json"
DEBUG_RUNNER="${SCRIPT_DIR}/debug_protocol_geometric_inference.py"

RUN_MODE="${RUN_MODE:-single}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  RUN_MODE="${1}"
  shift
fi

AGENT="${AGENT:-oracle}"
AGENT_SEED="${AGENT_SEED:-0}"
AGENT_EPISODES="${AGENT_EPISODES:-1}"
TASK_LABEL="${TASK_LABEL:-midpoint}"
EPSILON_POS="${EPSILON_POS:-0.05}"
TAU_LINE="${TAU_LINE:-0.02}"
POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS:-24}"
HEADLESS="${HEADLESS:-0}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaac_curobo}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/geometric_explicit_baselines/${TIMESTAMP}}"
VIEWER_EYE=(${VIEWER_EYE:-2.2 2.2 1.8})
VIEWER_LOOKAT=(${VIEWER_LOOKAT:-0.3 0.0 0.8})

EXTRA_ARGS=("$@")
SUMMARY_FILES=()

activate_conda_env() {
  source "${SCRIPT_DIR}/lib/geomind_conda.sh"
  geomind_activate_conda "${CONDA_ENV_NAME}" || exit 1
}

sanitize_slug() {
  printf '%s' "$1" | tr ' /:' '_' | tr -cd '[:alnum:]_.-'
}

build_common_args() {
  local args=(
    --epsilon-pos "${EPSILON_POS}"
    --tau-line "${TAU_LINE}"
    --post-place-settle-steps "${POST_PLACE_SETTLE_STEPS}"
    --viewer-eye "${VIEWER_EYE[@]}"
    --viewer-lookat "${VIEWER_LOOKAT[@]}"
  )

  if [[ "${HEADLESS}" == "1" ]]; then
    args+=(--headless)
  fi

  if [[ "${DISABLE_FABRIC}" == "1" ]]; then
    args+=(--disable-fabric)
  fi

  printf '%s\n' "${args[@]}"
}

run_debug_runner() {
  local args=()
  while IFS= read -r arg; do
    args+=("${arg}")
  done < <(build_common_args)

  python "${DEBUG_RUNNER}" "${args[@]}" "$@"
}

resolve_task_descriptor() {
  python - "${REPO_ROOT}" "${ANNOTATIONS_JSON}" "$1" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
annotations_path = Path(sys.argv[2])
requested = str(sys.argv[3]).strip()

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from geomind_lab.mimic.utils.task_metadata_registry import resolve_task_id  # noqa: E402

def normalize_rule_type(value: str) -> str | None:
    mapping = {
        "midpoint": "midpoint_placement",
        "midpoint_placement": "midpoint_placement",
        "perfect_midpoint": "midpoint_placement",
        "extension": "extension_placement",
        "extension_placement": "extension_placement",
        "isometric_extension": "extension_placement",
        "symmetry": "point_symmetry",
        "point_symmetry": "point_symmetry",
    }
    return mapping.get(str(value).strip().lower())


payload = json.loads(annotations_path.read_text(encoding="utf-8"))
records = []
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
        if not isinstance(gt_spec, dict):
            continue
        rule_type = normalize_rule_type(gt_spec.get("rule_type", ""))
        if rule_type not in {"midpoint_placement", "extension_placement", "point_symmetry"}:
            continue
        raw_task_name = str(annotation.get("task_id") or f"{layout_key}__task_{task_index}")
        task_id = resolve_task_id(raw_task_name) or raw_task_name
        records.append(
            {
                "scene_id": scene_id,
                "layout_key": str(layout_key),
                "task_index": int(task_index),
                "task_id": task_id,
                "raw_task_name": raw_task_name,
                "rule_type": rule_type,
            }
        )

if not records:
    raise SystemExit("No explicit geometric tasks found in annotated_tasks.json.")

preferred_by_alias = {
    "midpoint": "perfect_midpoint",
    "midpoint_placement": "perfect_midpoint",
    "perfect_midpoint": "perfect_midpoint",
    "extension": "isometric_extension",
    "extension_placement": "isometric_extension",
    "isometric_extension": "isometric_extension",
    "symmetry": "point_symmetry",
    "point_symmetry": "point_symmetry",
}

if not requested:
    requested = "midpoint"

normalized_request = normalize_rule_type(requested)
if normalized_request is not None:
    preferred_task_id = preferred_by_alias.get(requested.strip().lower())
    if preferred_task_id:
        for record in records:
            if record["task_id"] == preferred_task_id:
                print(
                    f'{record["scene_id"]}\t{record["layout_key"]}\t{record["task_index"]}\t{record["task_id"]}'
                )
                raise SystemExit(0)
    for record in records:
        if record["rule_type"] == normalized_request:
            print(f'{record["scene_id"]}\t{record["layout_key"]}\t{record["task_index"]}\t{record["task_id"]}')
            raise SystemExit(0)

if ":" in requested:
    maybe_layout_key, maybe_task_index = requested.rsplit(":", 1)
    if maybe_task_index.isdigit():
        for record in records:
            if record["layout_key"] == maybe_layout_key and record["task_index"] == int(maybe_task_index):
                print(
                    f'{record["scene_id"]}\t{record["layout_key"]}\t{record["task_index"]}\t{record["task_id"]}'
                )
                raise SystemExit(0)

for record in records:
    if requested in {record["task_id"], record["raw_task_name"], record["layout_key"]}:
        print(f'{record["scene_id"]}\t{record["layout_key"]}\t{record["task_index"]}\t{record["task_id"]}')
        raise SystemExit(0)

raise SystemExit(f"Unable to resolve TASK_LABEL={requested!r}.")
PY
}

resolve_suite_descriptors() {
  python - "${REPO_ROOT}" "${ANNOTATIONS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
annotations_path = Path(sys.argv[2])

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from geomind_lab.mimic.utils.task_metadata_registry import resolve_task_id  # noqa: E402

def normalize_rule_type(value: str) -> str | None:
    mapping = {
        "midpoint": "midpoint_placement",
        "midpoint_placement": "midpoint_placement",
        "perfect_midpoint": "midpoint_placement",
        "extension": "extension_placement",
        "extension_placement": "extension_placement",
        "isometric_extension": "extension_placement",
        "symmetry": "point_symmetry",
        "point_symmetry": "point_symmetry",
    }
    return mapping.get(str(value).strip().lower())


payload = json.loads(annotations_path.read_text(encoding="utf-8"))
records = []
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
        if not isinstance(gt_spec, dict):
            continue
        rule_type = normalize_rule_type(gt_spec.get("rule_type", ""))
        if rule_type not in {"midpoint_placement", "extension_placement", "point_symmetry"}:
            continue
        records.append(
            {
                "scene_id": scene_id,
                "layout_key": str(layout_key),
                "task_index": int(task_index),
                "task_id": resolve_task_id(str(annotation.get("task_id") or f"{layout_key}__task_{task_index}"))
                or str(annotation.get("task_id") or f"{layout_key}__task_{task_index}"),
                "rule_type": rule_type,
            }
        )

preferred = {
    "midpoint_placement": "perfect_midpoint",
    "extension_placement": "isometric_extension",
}

for rule_type in ("midpoint_placement", "extension_placement"):
    chosen = None
    for record in records:
        if record["task_id"] == preferred[rule_type]:
            chosen = record
            break
    if chosen is None:
        for record in records:
            if record["rule_type"] == rule_type:
                chosen = record
                break
    if chosen is None:
        raise SystemExit(f"Unable to resolve suite task for rule_type={rule_type!r}.")
    print(f'{chosen["scene_id"]}\t{chosen["layout_key"]}\t{chosen["task_index"]}\t{chosen["task_id"]}')
PY
}

run_task() {
  local case_name="$1"
  local task_label="$2"
  local scene_id="$3"
  local layout_key="$4"
  local task_index="$5"
  local task_ordinal="$6"
  shift 6

  local task_agent_seed=$((AGENT_SEED + task_ordinal * 100003))
  local task_slug
  local summary_path
  local log_path

  task_slug="$(sanitize_slug "${case_name}__${task_label}__scene${scene_id}__task${task_index}")"
  summary_path="${OUTPUT_DIR}/${task_slug}.summary.json"
  log_path="${OUTPUT_DIR}/${task_slug}.log"

  echo
  echo "------------------------------------------------------------------------"
  echo "Case: ${case_name}"
  echo "Task: ${task_label}"
  echo "scene_id=${scene_id} layout_key=${layout_key} task_index=${task_index}"
  echo "agent=${AGENT} task_agent_seed=${task_agent_seed}"
  echo "summary_json=${summary_path}"
  echo "task_log=${log_path}"
  echo "------------------------------------------------------------------------"

  run_debug_runner \
    --scene-id "${scene_id}" \
    --layout-key "${layout_key}" \
    --task-index "${task_index}" \
    --agent "${AGENT}" \
    --agent-seed "${task_agent_seed}" \
    --agent-episodes "${AGENT_EPISODES}" \
    --summary-json-out "${summary_path}" \
    "${EXTRA_ARGS[@]}" | tee "${log_path}"

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
  local suite_name="$1"
  local suite_summary_path="${OUTPUT_DIR}/${suite_name}.summary.json"

  python - "${suite_summary_path}" "${OUTPUT_DIR}" "${suite_name}" "${SUMMARY_FILES[@]}" <<'PY'
import json
import sys
from pathlib import Path

suite_summary_path = Path(sys.argv[1])
output_dir = sys.argv[2]
suite_name = sys.argv[3]
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
placement_execution_success_count = sum(
    int(item.get("placement_execution_success_count", 0)) for item in task_summaries
)

suite_summary = {
    "suite_name": suite_name,
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

print_header() {
  echo "========================================================================"
  echo "GeoMind Explicit Geometric Baseline Runner"
  echo "========================================================================"
  echo "run_mode=${RUN_MODE}"
  echo "task_label=${TASK_LABEL}"
  echo "agent=${AGENT}"
  echo "agent_seed=${AGENT_SEED}"
  echo "agent_episodes=${AGENT_EPISODES}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "========================================================================"
}

main() {
  mkdir -p "${OUTPUT_DIR}"
  activate_conda_env
  print_header

  case "${RUN_MODE}" in
    single)
      local scene_id
      local layout_key
      local task_index
      local resolved_label
      IFS=$'\t' read -r scene_id layout_key task_index resolved_label < <(resolve_task_descriptor "${TASK_LABEL}")
      run_task "single_${AGENT}" "${resolved_label}" "${scene_id}" "${layout_key}" "${task_index}" 0
      ;;
    suite)
      local task_ordinal=0
      local scene_id
      local layout_key
      local task_index
      local resolved_label
      while IFS=$'\t' read -r scene_id layout_key task_index resolved_label; do
        [[ -z "${scene_id}" ]] && continue
        run_task "suite_${AGENT}" "${resolved_label}" "${scene_id}" "${layout_key}" "${task_index}" "${task_ordinal}"
        task_ordinal=$((task_ordinal + 1))
      done < <(resolve_suite_descriptors)
      echo
      echo "========================================================================"
      echo "suite_summary: suite_${AGENT}"
      echo "========================================================================"
      write_suite_summary "suite_${AGENT}"
      ;;
    *)
      echo "Unknown RUN_MODE=${RUN_MODE}" >&2
      echo "Supported RUN_MODE values: single, suite" >&2
      exit 1
      ;;
  esac

  echo
  echo "========================================================================"
  echo "output_dir=${OUTPUT_DIR}"
  echo "========================================================================"
}

main
