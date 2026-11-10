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
ANNOTATIONS_JSON="${ANNOTATIONS_JSON:-${REPO_ROOT}/assets/annotations/annotated_tasks.json}"
DEBUG_RUNNER="${SCRIPT_DIR}/debug_protocol_geometric_inference.py"
SUMMARIZER="${SCRIPT_DIR}/summarize_geometric_inference_results.py"

# Thin batch runner for the legacy geometric structure-inference backend.
# Supported environment variables:
#   TASK_LIST_FILE=<txt|json>   # task_id / layout_key / layout_key:task_index
#   AGENT_LIST_FILE=<txt>       # optional newline-delimited agent names
#   AGENTS="explicit_midpoint_oracle explicit_extension_oracle explicit_random implicit_trend_oracle implicit_trend_random"
#   AGENT_EPISODES=<int>
#   AGENT_SEED=<int>
#   EPSILON_POS=<float>
#   TAU_ORTHOGONAL=<float>
#   TAU_TREND=<float>
#   POST_PLACE_SETTLE_STEPS=<int>
#   HEADLESS=0|1
#   DISABLE_FABRIC=0|1
#   CONDA_ENV_NAME=<name>
#   OUTPUT_ROOT=<path>
#   OUTPUT_DIR=<path>           # legacy alias; wins over OUTPUT_ROOT when both are set
#   BENCHMARK_INFO_JSON=<path>
#   RUNNER_SELECTION_JSON=<path>
#   RUNNER_SKIPPED_SELECTION_JSON=<path>
#   SUMMARY_GROUP_BY_FILE=<path>
#   RUNNER_FAMILY=<name>
#   SUMMARIZE_AT_END=0|1
#
# Examples:
#   TASK_LIST_FILE=scripts/geometric_matrix_tasks.txt bash scripts/run_geometric_inference_matrix.sh
#   TASK_LIST_FILE=tmp/geometric_tasks.json AGENTS="explicit_midpoint_oracle explicit_random" HEADLESS=1 bash scripts/run_geometric_inference_matrix.sh

TASK_LIST_FILE="${TASK_LIST_FILE:-}"
if [[ -z "${TASK_LIST_FILE}" ]]; then
  echo "TASK_LIST_FILE is required." >&2
  exit 1
fi

AGENT_LIST_FILE="${AGENT_LIST_FILE:-}"
AGENTS="${AGENTS:-explicit_midpoint_oracle explicit_extension_oracle explicit_random implicit_trend_oracle implicit_trend_random}"
AGENT_EPISODES="${AGENT_EPISODES:-1}"
AGENT_SEED="${AGENT_SEED:-0}"
EPSILON_POS="${EPSILON_POS:-0.05}"
TAU_LINE="${TAU_LINE:-0.02}"
TAU_ORTHOGONAL="${TAU_ORTHOGONAL:-${TAU_LINE}}"
TAU_TREND="${TAU_TREND:-0.03}"
POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS:-24}"
HEADLESS="${HEADLESS:-0}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaac_curobo}"
SUMMARIZE_AT_END="${SUMMARIZE_AT_END:-1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-${REPO_ROOT}/logs/geometric_inference_matrix/${TIMESTAMP}}}"
BENCHMARK_INFO_JSON="${BENCHMARK_INFO_JSON:-}"
RUNNER_SELECTION_JSON="${RUNNER_SELECTION_JSON:-}"
RUNNER_SKIPPED_SELECTION_JSON="${RUNNER_SKIPPED_SELECTION_JSON:-}"
SUMMARY_GROUP_BY_FILE="${SUMMARY_GROUP_BY_FILE:-}"
RUNNER_FAMILY="${RUNNER_FAMILY:-geometric_structure}"
VIEWER_EYE=(${VIEWER_EYE:-2.2 2.2 1.8})
VIEWER_LOOKAT=(${VIEWER_LOOKAT:-0.3 0.0 0.8})

SUMMARY_FILES=()
MANIFEST_PATH="${OUTPUT_DIR}/run_manifest.tsv"

activate_conda_env() {
  source "${SCRIPT_DIR}/lib/geomind_conda.sh"
  geomind_activate_conda "${CONDA_ENV_NAME}" || exit 1
}

sanitize_slug() {
  printf '%s' "$1" | tr ' /:' '_' | tr -cd '[:alnum:]_.-'
}

print_header() {
  echo "========================================================================"
  echo "GeoMind Metadata-First Family Runner"
  echo "========================================================================"
  echo "runner_family=${RUNNER_FAMILY}"
  echo "backend_role=legacy geometric structure-inference backend"
  echo "task_list_file=${TASK_LIST_FILE}"
  if [[ -n "${AGENT_LIST_FILE}" ]]; then
    echo "agent_list_file=${AGENT_LIST_FILE}"
  fi
  echo "agents=${AGENTS}"
  echo "agent_episodes=${AGENT_EPISODES}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "annotations_json=${ANNOTATIONS_JSON}"
  if [[ -n "${BENCHMARK_INFO_JSON}" ]]; then
    echo "benchmark_info_json=${BENCHMARK_INFO_JSON}"
  fi
  if [[ -n "${RUNNER_SELECTION_JSON}" ]]; then
    echo "runner_selection_json=${RUNNER_SELECTION_JSON}"
  fi
  echo "========================================================================"
}

load_agents() {
  if [[ -n "${AGENT_LIST_FILE}" && -s "${AGENT_LIST_FILE}" ]]; then
    paste -sd' ' "${AGENT_LIST_FILE}"
    return 0
  fi
  printf '%s\n' "${AGENTS}"
}

read_task_tokens() {
  python - "${TASK_LIST_FILE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

try:
    payload = json.loads(text)
except Exception:
    payload = None

if isinstance(payload, list):
    for item in payload:
        if isinstance(item, str) and item.strip():
            print(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        task_label = item.get("task_label") or item.get("task_id")
        if isinstance(task_label, str) and task_label.strip():
            print(task_label.strip())
            continue
        layout_key = item.get("layout_key")
        task_index = item.get("task_index", 0)
        if isinstance(layout_key, str) and layout_key.strip():
            print(f"{layout_key.strip()}:{int(task_index)}")
    raise SystemExit(0)

for line in text.splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    print(stripped)
PY
}

resolve_task_descriptor() {
  python - "${ANNOTATIONS_JSON}" "$1" <<'PY'
import json
import sys
from pathlib import Path

annotations_path = Path(sys.argv[1])
requested = str(sys.argv[2]).strip()
repo_root = annotations_path.parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from geomind_lab.mimic.utils.config_semantics_adapter import augment_record_with_semantics, normalize_annotation_entry  # noqa: E402

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
        if str(protocol.get("module_name", "")) != "geometric_inference":
            continue
        raw_task_name = str(annotation.get("task_id") or f"{layout_key}__task_{task_index}")
        semantic_view = normalize_annotation_entry(
            layout_key=layout_key,
            task_index=task_index,
            layout=layout,
            task_payload=task_payload,
        )
        enriched = augment_record_with_semantics(
            {
                "task_name": raw_task_name,
                "task_id": raw_task_name,
                "family": semantic_view.get("family"),
                "variant": semantic_view.get("variant"),
                "task_type": semantic_view.get("task_type"),
                "scene_id": scene_id,
                "layout_key": str(layout_key),
                "task_index": int(task_index),
            }
        )
        family = str(protocol.get("family", protocol.get("track_name", "")))
        variant = str(protocol.get("variant", protocol.get("subtask_name", "")))
        instruction = str(task_payload.get("instruction") or layout.get("instruction") or "").strip()
        instruction = " ".join(instruction.split())
        records.append(
            {
                "scene_id": scene_id,
                "layout_key": str(layout_key),
                "task_index": int(task_index),
                "raw_task_name": raw_task_name,
                "task_id": str(enriched.get("task_id", raw_task_name)),
                "task_version": str(enriched.get("task_version", "")),
                "phase": str(enriched.get("phase", "")),
                "semantic_module": str(enriched.get("semantic_module", "")),
                "skill_type": str(enriched.get("skill_type", "")),
                "parent_task_id": str(enriched.get("parent_task_id", "")),
                "paired_control_id": str(enriched.get("paired_control_id", "")),
                "instruction": instruction,
                "family": family,
                "variant": variant,
            }
        )

if ":" in requested:
    maybe_layout_key, maybe_task_index = requested.rsplit(":", 1)
    if maybe_task_index.isdigit():
        for record in records:
            if record["layout_key"] == maybe_layout_key and record["task_index"] == int(maybe_task_index):
                print(
                    f'{record["scene_id"]}\t{record["layout_key"]}\t{record["task_index"]}\t{record["raw_task_name"]}\t{record["task_id"]}\t{record["task_version"]}\t{record["phase"]}\t{record["semantic_module"]}\t{record["skill_type"]}\t{record["parent_task_id"]}\t{record["paired_control_id"]}\t{record["instruction"]}\t{record["family"]}\t{record["variant"]}'
                )
                raise SystemExit(0)

for record in records:
    if requested in {record["raw_task_name"], record["task_id"], record["layout_key"]}:
        print(
            f'{record["scene_id"]}\t{record["layout_key"]}\t{record["task_index"]}\t{record["raw_task_name"]}\t{record["task_id"]}\t{record["task_version"]}\t{record["phase"]}\t{record["semantic_module"]}\t{record["skill_type"]}\t{record["parent_task_id"]}\t{record["paired_control_id"]}\t{record["instruction"]}\t{record["family"]}\t{record["variant"]}'
        )
        raise SystemExit(0)

raise SystemExit(f"Unable to resolve geometric structure-inference backend task label: {requested!r}")
PY
}

run_task_agent_pair() {
  local task_token="$1"
  local agent_name="$2"
  local task_ordinal="$3"
  local agent_ordinal="$4"

  local scene_id
  local layout_key
  local task_index
  local raw_task_name
  local task_id
  local task_version
  local phase
  local semantic_module
  local skill_type
  local parent_task_id
  local paired_control_id
  local instruction
  local family
  local variant
  IFS=$'\t' read -r scene_id layout_key task_index raw_task_name task_id task_version phase semantic_module skill_type parent_task_id paired_control_id instruction family variant < <(resolve_task_descriptor "${task_token}")

  local task_slug
  local agent_slug
  local base_dir
  local summary_path
  local log_path
  local derived_seed
  local run_args=()

  task_slug="$(sanitize_slug "${raw_task_name}")"
  agent_slug="$(sanitize_slug "${agent_name}")"
  base_dir="${OUTPUT_DIR}/${family}/${variant}/${agent_slug}"
  summary_path="${base_dir}/${task_slug}.summary.json"
  log_path="${base_dir}/${task_slug}.log"
  derived_seed=$((AGENT_SEED + task_ordinal * 100003 + agent_ordinal * 7001))

  mkdir -p "${base_dir}"

  echo
  echo "------------------------------------------------------------------------"
  echo "task_id=${task_id}"
  echo "task_version=${task_version}"
  echo "phase=${phase}"
  echo "semantic_module=${semantic_module}"
  echo "skill_type=${skill_type}"
  if [[ -n "${paired_control_id}" ]]; then
    echo "paired_control_id=${paired_control_id}"
  fi
  echo "runner_family=${RUNNER_FAMILY}"
  echo "scene_id=${scene_id} layout_key=${layout_key} task_index=${task_index}"
  echo "instruction=${instruction}"
  echo "agent=${agent_name} seed=${derived_seed}"
  echo "legacy_task_name=${raw_task_name}"
  echo "legacy_backend=${family}/${variant}"
  echo "summary_json=${summary_path}"
  echo "task_log=${log_path}"
  echo "------------------------------------------------------------------------"

  run_args=(
    --scene-id "${scene_id}"
    --layout-key "${layout_key}"
    --task-index "${task_index}"
    --agent "${agent_name}"
    --agent-seed "${derived_seed}"
    --agent-episodes "${AGENT_EPISODES}"
    --epsilon-pos "${EPSILON_POS}"
    --tau-orthogonal "${TAU_ORTHOGONAL}"
    --tau-trend "${TAU_TREND}"
    --post-place-settle-steps "${POST_PLACE_SETTLE_STEPS}"
    --summary-json-out "${summary_path}"
    --viewer-eye "${VIEWER_EYE[@]}"
    --viewer-lookat "${VIEWER_LOOKAT[@]}"
  )
  if [[ "${HEADLESS}" == "1" ]]; then
    run_args+=(--headless)
  fi
  if [[ "${DISABLE_FABRIC}" == "1" ]]; then
    run_args+=(--disable-fabric)
  fi

  GEOMIND_LAB_TASK_SPECS_PATH="${ANNOTATIONS_JSON}" \
  python "${DEBUG_RUNNER}" "${run_args[@]}" \
    | tee "${log_path}"

  python - "${summary_path}" "${raw_task_name}" "${task_id}" "${task_version}" "${phase}" "${semantic_module}" "${skill_type}" "${parent_task_id}" "${paired_control_id}" "${RUNNER_FAMILY}" "${family}" "${variant}" "${log_path}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
raw_task_name = sys.argv[2]
task_id = sys.argv[3]
task_version = sys.argv[4]
phase = sys.argv[5]
semantic_module = sys.argv[6]
skill_type = sys.argv[7]
parent_task_id = sys.argv[8]
paired_control_id = sys.argv[9]
runner_family = sys.argv[10]
family = sys.argv[11]
variant = sys.argv[12]
log_path = sys.argv[13]

payload = json.loads(summary_path.read_text(encoding="utf-8"))
payload["task_id"] = task_id
payload["task_name"] = raw_task_name
payload["raw_task_name"] = raw_task_name
payload["legacy_task_name"] = raw_task_name
payload["task_version"] = task_version
payload["phase"] = phase
payload["semantic_module"] = semantic_module
payload["skill_type"] = skill_type
payload["parent_task_id"] = parent_task_id
payload["paired_control_id"] = paired_control_id
payload["runner_family"] = runner_family
payload["family"] = payload.get("family") or family
payload["variant"] = payload.get("variant") or variant
payload["task_log"] = log_path
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
PY

  SUMMARY_FILES+=("${summary_path}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${raw_task_name}" "${family}" "${variant}" "${agent_name}" "${scene_id}" "${layout_key}" "${summary_path}" "${log_path}" \
    >> "${MANIFEST_PATH}"
}

main() {
  mkdir -p "${OUTPUT_DIR}"
  : > "${MANIFEST_PATH}"
  activate_conda_env
  print_header

  mapfile -t task_tokens < <(read_task_tokens)
  if [[ "${#task_tokens[@]}" -eq 0 ]]; then
    echo "No task tokens were resolved from TASK_LIST_FILE=${TASK_LIST_FILE}" >&2
    exit 1
  fi

  local agents_value
  agents_value="$(load_agents)"
  read -r -a agent_list <<< "${agents_value}"
  if [[ "${#agent_list[@]}" -eq 0 ]]; then
    echo "No agents were configured." >&2
    exit 1
  fi

  local task_ordinal=0
  local task_token
  for task_token in "${task_tokens[@]}"; do
    local agent_ordinal=0
    local agent_name
    for agent_name in "${agent_list[@]}"; do
      run_task_agent_pair "${task_token}" "${agent_name}" "${task_ordinal}" "${agent_ordinal}"
      agent_ordinal=$((agent_ordinal + 1))
    done
    task_ordinal=$((task_ordinal + 1))
  done

  if [[ "${SUMMARIZE_AT_END}" == "1" && "${#SUMMARY_FILES[@]}" -gt 0 ]]; then
    echo
    echo "========================================================================"
    echo "aggregated_results"
    echo "========================================================================"
    python "${SUMMARIZER}" "${OUTPUT_DIR}" --annotations "${ANNOTATIONS_JSON}" --csv-out "${OUTPUT_DIR}/aggregate.csv" \
      | tee "${OUTPUT_DIR}/aggregate.txt"
  fi

  echo
  echo "========================================================================"
  echo "output_dir=${OUTPUT_DIR}"
  echo "manifest=${MANIFEST_PATH}"
  echo "========================================================================"
}

main
