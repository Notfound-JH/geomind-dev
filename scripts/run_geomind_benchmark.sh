#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ANNOTATIONS_JSON="${ANNOTATIONS_JSON:-${REPO_ROOT}/assets/annotations/annotated_tasks.json}"
MANIFEST_PATH="${1:-${MANIFEST_PATH:-}}"

if [[ -z "${MANIFEST_PATH}" ]]; then
  echo "Usage: bash scripts/run_geomind_benchmark.sh <benchmark_manifest.json> [output_dir]" >&2
  exit 1
fi

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-${REPO_ROOT}/logs/geomind_benchmark/${TIMESTAMP}}}"
TASK_NAMES_OVERRIDE="${TASK_NAMES_OVERRIDE:-}"
AGENT_NAMES_OVERRIDE="${AGENT_NAMES_OVERRIDE:-}"
TASK_VERSION_FILTER="${TASK_VERSION_FILTER:-}"
PHASE_FILTER="${PHASE_FILTER:-}"
RUNNER_DISPATCH_MODE="${RUNNER_DISPATCH_MODE:-}"
DRY_RUN="${DRY_RUN:-0}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaac_curobo}"
SUMMARIZE_AT_END="${SUMMARIZE_AT_END:-1}"
STATE_REFRESH_AGENT_EPISODES="${STATE_REFRESH_AGENT_EPISODES:-${AGENT_EPISODES:-5}}"
GEOMETRIC_STRUCTURE_AGENT_EPISODES="${GEOMETRIC_STRUCTURE_AGENT_EPISODES:-${AGENT_EPISODES:-1}}"
JOINT_SUMMARIZER="${SCRIPT_DIR}/summarize_geomind_benchmark_results.py"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/geomind_benchmark.XXXXXX")"
RUNNER_PLAN_JSON="${TMP_DIR}/runner_plan.json"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

print_header() {
  echo "========================================================================"
  echo "GeoMind Joint Benchmark Runner"
  echo "========================================================================"
  echo "manifest=${MANIFEST_PATH}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "annotations_json=${ANNOTATIONS_JSON}"
  if [[ -n "${TASK_NAMES_OVERRIDE}" ]]; then
    echo "task_names_override=${TASK_NAMES_OVERRIDE}"
  fi
  if [[ -n "${AGENT_NAMES_OVERRIDE}" ]]; then
    echo "agent_names_override=${AGENT_NAMES_OVERRIDE}"
  fi
  if [[ -n "${TASK_VERSION_FILTER}" ]]; then
    echo "task_version_filter=${TASK_VERSION_FILTER}"
  fi
  if [[ -n "${PHASE_FILTER}" ]]; then
    echo "phase_filter=${PHASE_FILTER}"
  fi
  if [[ -n "${RUNNER_DISPATCH_MODE}" ]]; then
    echo "runner_dispatch_mode_override=${RUNNER_DISPATCH_MODE}"
  fi
  echo "dry_run=${DRY_RUN}"
  echo "========================================================================"
}

prepare_manifest_selection() {
  python - "${REPO_ROOT}" "${MANIFEST_PATH}" "${ANNOTATIONS_JSON}" "${OUTPUT_DIR}" "${TMP_DIR}" "${TASK_NAMES_OVERRIDE}" "${AGENT_NAMES_OVERRIDE}" "${TASK_VERSION_FILTER}" "${PHASE_FILTER}" "${RUNNER_DISPATCH_MODE}" "${DRY_RUN}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys

repo_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2]).resolve()
annotations_path = Path(sys.argv[3]).resolve()
output_dir = Path(sys.argv[4]).resolve()
tmp_dir = Path(sys.argv[5]).resolve()
task_names_override = str(sys.argv[6]).strip()
agent_names_override = str(sys.argv[7]).strip()
task_version_filter_override = str(sys.argv[8]).strip()
phase_filter_override = str(sys.argv[9]).strip()
runner_dispatch_mode_override = str(sys.argv[10]).strip()
dry_run = bool(int(sys.argv[11]))

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from geomind_lab.mimic.utils.benchmark_manifest_utils import (  # noqa: E402
    DISPATCH_STATUS_RUNNABLE,
    RUNNER_DISPATCH_MODE_PERMISSIVE,
    RUNNER_DISPATCH_MODE_STRICT,
    UNKNOWN_RUNNER_KEY,
    load_benchmark_manifest,
    plan_runner_dispatch,
    registered_runner_families,
    runner_family_config,
    runner_key_for_family,
)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = str(raw_value).strip()
        if not value or value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _write_text_lines(path: Path, values: list[str]) -> None:
    path.write_text(
        "\n".join(str(value) for value in values) + ("\n" if values else ""),
        encoding="utf-8",
    )


manifest = load_benchmark_manifest(manifest_path)
task_names = manifest.get("task_ids", [])
if not isinstance(task_names, list):
    task_names = manifest.get("task_names", [])
if not isinstance(task_names, list):
    task_names = []
canonical_task_ids = manifest.get("canonical_task_ids", [])
if not isinstance(canonical_task_ids, list):
    canonical_task_ids = []
if not task_names and canonical_task_ids:
    task_names = canonical_task_ids
if task_names_override:
    task_names = shlex.split(task_names_override)
task_names = _dedupe_keep_order(task_names)

agent_names = manifest.get("agent_names", [])
if not isinstance(agent_names, list):
    agent_names = []
default_agent_groups = manifest.get("default_agent_groups", {})
if not agent_names and isinstance(default_agent_groups, dict):
    merged_agents: list[str] = []
    for runner_family in registered_runner_families():
        values = default_agent_groups.get(runner_family, [])
        if isinstance(values, list):
            merged_agents.extend(str(value) for value in values)
    agent_names = merged_agents
if agent_names_override:
    agent_names = shlex.split(agent_names_override)
agent_names = _dedupe_keep_order(agent_names)

task_version_filter = manifest.get("task_version_filter", [])
if not isinstance(task_version_filter, list):
    task_version_filter = []
if task_version_filter_override:
    task_version_filter = shlex.split(task_version_filter_override)
task_version_filter = _dedupe_keep_order(task_version_filter)

phase_filter = manifest.get("phase_filter", [])
if not isinstance(phase_filter, list):
    phase_filter = []
if phase_filter_override:
    phase_filter = shlex.split(phase_filter_override)
phase_filter = _dedupe_keep_order(phase_filter)

dispatch_mode = runner_dispatch_mode_override or str(
    manifest.get("runner_dispatch_mode", RUNNER_DISPATCH_MODE_PERMISSIVE)
).strip()
if dispatch_mode not in {RUNNER_DISPATCH_MODE_PERMISSIVE, RUNNER_DISPATCH_MODE_STRICT}:
    raise SystemExit(f"Unsupported runner dispatch mode: {dispatch_mode!r}")

task_families = manifest.get("task_families", [])
if not isinstance(task_families, list):
    task_families = []
unsupported_families = sorted(
    {
        str(family)
        for family in task_families
        if runner_key_for_family(str(family)) == UNKNOWN_RUNNER_KEY
    }
)

dispatch = plan_runner_dispatch(
    task_names,
    agent_names,
    annotations_path=annotations_path,
    task_versions=task_version_filter,
    phases=phase_filter,
    dispatch_mode=dispatch_mode,
)
task_groups = dispatch["task_groups"]
agent_groups = dispatch["agent_groups"]
unknown_tasks = _dedupe_keep_order(dispatch["unknown_tasks"])
unknown_agents = _dedupe_keep_order(dispatch["unknown_agents"])
selected_task_descriptors = dispatch["selected_task_descriptors"]
runnable_task_descriptors = dispatch["runnable_task_descriptors"]
skipped_task_descriptors = dispatch["skipped_task_descriptors"]

errors: list[str] = []
if unknown_tasks:
    errors.append(f"Unknown task names for annotations {annotations_path}: {unknown_tasks}")
if unknown_agents:
    errors.append(f"Unknown agent names for registered runner families: {unknown_agents}")
if dispatch_mode == RUNNER_DISPATCH_MODE_STRICT and skipped_task_descriptors:
    preview = [
        "  - "
        + f"task_id={entry.get('task_id', '')} task_version={entry.get('task_version', '')} "
        + f"runner_family={entry.get('runner_family', '')} status={entry.get('dispatch_status', '')}"
        for entry in skipped_task_descriptors
    ]
    errors.append(
        "Runner dispatch mode 'strict' does not allow declared but not runnable tasks:\n"
        + "\n".join(preview)
    )
if errors:
    raise SystemExit("\n".join(errors))

summary_preset = manifest.get("summary_preset", {})
if not isinstance(summary_preset, dict):
    summary_preset = {}
joint_group_by = summary_preset.get("joint_group_by", ["family", "variant", "task_name", "agent_name", "task_type"])
if not isinstance(joint_group_by, list):
    raise SystemExit(f"joint_group_by must be a list in benchmark manifest {manifest_path}")

group_by_by_runner: dict[str, list[str]] = {}
for runner_family in registered_runner_families():
    config = runner_family_config(runner_family)
    group_key = str(config.get("summary_group_by_key", "joint_group_by"))
    values = summary_preset.get(group_key, joint_group_by)
    if not isinstance(values, list):
        raise SystemExit(f"{group_key} must be a list in benchmark manifest {manifest_path}")
    group_by_by_runner[runner_family] = [str(value) for value in values]

selected_by_runner: dict[str, list[dict[str, object]]] = {runner_family: [] for runner_family in registered_runner_families()}
runnable_by_runner: dict[str, list[dict[str, object]]] = {runner_family: [] for runner_family in registered_runner_families()}
skipped_by_runner: dict[str, list[dict[str, object]]] = {runner_family: [] for runner_family in registered_runner_families()}
for entry in selected_task_descriptors:
    runner_family = str(entry.get("runner_family", UNKNOWN_RUNNER_KEY))
    if runner_family not in selected_by_runner:
        continue
    selected_by_runner[runner_family].append(entry)
    if str(entry.get("dispatch_status", "")) == DISPATCH_STATUS_RUNNABLE:
        runnable_by_runner[runner_family].append(entry)
    else:
        skipped_by_runner[runner_family].append(entry)

runner_plan: list[dict[str, object]] = []
for runner_family in registered_runner_families():
    config = runner_family_config(runner_family)
    task_list_file = tmp_dir / f"{runner_family}_tasks.json"
    agent_list_file = tmp_dir / f"{runner_family}_agents.txt"
    group_by_file = tmp_dir / f"{runner_family}_group_by.txt"
    selection_file = tmp_dir / f"{runner_family}_selection.json"
    skipped_file = tmp_dir / f"{runner_family}_skipped.json"

    task_list = list(task_groups.get(runner_family, []))
    agent_list = list(agent_groups.get(runner_family, []))
    group_by_values = group_by_by_runner.get(runner_family, list(joint_group_by))

    task_list_file.write_text(json.dumps(task_list, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_text_lines(agent_list_file, agent_list)
    _write_text_lines(group_by_file, group_by_values)
    selection_file.write_text(
        json.dumps(runnable_by_runner.get(runner_family, []), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    skipped_file.write_text(
        json.dumps(skipped_by_runner.get(runner_family, []), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not (
        selected_by_runner.get(runner_family)
        or task_list
        or agent_list
    ):
        continue

    runner_plan.append(
        {
            "runner_family": runner_family,
            "implemented": bool(config.get("implemented", False)),
            "child_runner_script_name": str(config.get("child_runner_script_name", "")),
            "summarizer_script_name": str(config.get("summarizer_script_name", "")),
            "output_subdir": str(config.get("output_subdir", runner_family)),
            "summary_group_by_key": str(config.get("summary_group_by_key", "joint_group_by")),
            "task_list_file": str(task_list_file),
            "agent_list_file": str(agent_list_file),
            "summary_group_by_file": str(group_by_file),
            "runner_selection_json": str(selection_file),
            "runner_skipped_json": str(skipped_file),
            "task_count": len(task_list),
            "agent_count": len(agent_list),
            "selected_descriptor_count": len(selected_by_runner.get(runner_family, [])),
            "skipped_descriptor_count": len(skipped_by_runner.get(runner_family, [])),
            "agent_episode_env": str(config.get("agent_episode_env", "AGENT_EPISODES")),
            "default_agent_episodes": int(config.get("default_agent_episodes", 1)),
            "notes": str(config.get("notes", "")),
        }
    )

(tmp_dir / "runner_plan.json").write_text(
    json.dumps(runner_plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

dispatch_warnings: list[str] = []
for entry in skipped_task_descriptors:
    dispatch_warnings.append(
        f"skip task_id={entry.get('task_id', '')} task_version={entry.get('task_version', '')} "
        f"runner_family={entry.get('runner_family', '')} status={entry.get('dispatch_status', '')}: "
        f"{entry.get('dispatch_reason', '')}"
    )
(output_dir / "dispatch_warnings.txt").write_text(
    "\n".join(dispatch_warnings) + ("\n" if dispatch_warnings else ""),
    encoding="utf-8",
)

dispatch_report = {
    "runner_dispatch_mode": dispatch_mode,
    "dry_run": dry_run,
    "unsupported_task_families": unsupported_families,
    "selected_task_descriptors": selected_task_descriptors,
    "runnable_task_descriptors": runnable_task_descriptors,
    "skipped_task_descriptors": skipped_task_descriptors,
    "selected_tasks_by_runner": {
        runner_family: list(task_groups.get(runner_family, []))
        for runner_family in registered_runner_families()
    },
    "selected_agents_by_runner": {
        runner_family: list(agent_groups.get(runner_family, []))
        for runner_family in registered_runner_families()
    },
    "runner_plan": runner_plan,
}
(output_dir / "runner_dispatch_report.json").write_text(
    json.dumps(dispatch_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

benchmark_info = dict(manifest)
benchmark_info.update(
    {
        "manifest_path": str(manifest_path),
        "annotations_json": str(annotations_path),
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "runner_dispatch_mode": dispatch_mode,
        "unsupported_task_families": unsupported_families,
        "selected_task_names": task_names,
        "selected_task_ids": [str(entry.get("task_id", "")) for entry in selected_task_descriptors],
        "selected_canonical_task_ids": canonical_task_ids,
        "selected_agent_names": agent_names,
        "task_version_filter": task_version_filter,
        "phase_filter": phase_filter,
        "selected_task_descriptors": selected_task_descriptors,
        "runnable_task_descriptors": runnable_task_descriptors,
        "skipped_task_descriptors": skipped_task_descriptors,
        "selected_tasks_by_runner": dispatch_report["selected_tasks_by_runner"],
        "selected_agents_by_runner": dispatch_report["selected_agents_by_runner"],
        "runner_plan": runner_plan,
    }
)
(output_dir / "benchmark_info.json").write_text(
    json.dumps(benchmark_info, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

join_space_separated() {
  local source_file="$1"
  if [[ ! -s "${source_file}" ]]; then
    return 0
  fi
  paste -sd' ' "${source_file}"
}

json_list_count() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if isinstance(payload, list):
    print(len(payload))
else:
    print(0)
PY
}

json_list_joined() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if isinstance(payload, list):
    print(" ".join(str(value) for value in payload))
else:
    print("")
PY
}

emit_runner_plan_rows() {
  python - "$1" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
for item in payload:
    delimiter = "|"
    print(
        delimiter.join(
            [
                str(item.get("runner_family", "")),
                "1" if bool(item.get("implemented", False)) else "0",
                str(item.get("child_runner_script_name", "")),
                str(item.get("summarizer_script_name", "")),
                str(item.get("output_subdir", "")),
                str(item.get("task_list_file", "")),
                str(item.get("agent_list_file", "")),
                str(item.get("summary_group_by_file", "")),
                str(item.get("runner_selection_json", "")),
                str(item.get("runner_skipped_json", "")),
                str(item.get("task_count", 0)),
                str(item.get("agent_count", 0)),
                str(item.get("selected_descriptor_count", 0)),
                str(item.get("skipped_descriptor_count", 0)),
                str(item.get("agent_episode_env", "AGENT_EPISODES")),
                str(item.get("default_agent_episodes", 1)),
                str(item.get("notes", "")),
            ]
        )
    )
PY
}

resolve_agent_episodes() {
  local agent_episode_env="$1"
  local default_agent_episodes="$2"
  local resolved="${default_agent_episodes}"
  if [[ -n "${agent_episode_env}" && -n "${!agent_episode_env:-}" ]]; then
    resolved="${!agent_episode_env}"
  fi
  printf '%s' "${resolved}"
}

maybe_append_optional_env() {
  local array_name="$1"
  local variable_name="$2"
  if [[ -n "${!variable_name+x}" ]]; then
    eval "${array_name}+=(\"${variable_name}=\${${variable_name}}\")"
  fi
}

run_child_runner() {
  local runner_family="$1"
  local implemented="$2"
  local runner_script="$3"
  local task_list_file="$4"
  local agent_list_file="$5"
  local output_subdir="$6"
  local agent_episodes="$7"
  local selection_json="$8"
  local skipped_json="$9"
  local summary_group_by_file="${10}"
  local notes="${11}"
  local selected_descriptor_count="${12}"
  local skipped_descriptor_count="${13}"

  local agent_names
  local task_count
  task_count="$(json_list_count "${task_list_file}")"
  agent_names="$(join_space_separated "${agent_list_file}")"

  echo
  echo "------------------------------------------------------------------------"
  echo "runner_family=${runner_family}"
  echo "implemented=${implemented}"
  echo "selected_descriptor_count=${selected_descriptor_count}"
  echo "skipped_descriptor_count=${skipped_descriptor_count}"
  echo "notes=${notes}"
  echo "------------------------------------------------------------------------"

  if [[ "${implemented}" != "1" ]]; then
    echo "skip runner_family=${runner_family}: child runner is not implemented"
    return 0
  fi
  if [[ -z "${runner_script}" || ! -f "${runner_script}" ]]; then
    echo "skip runner_family=${runner_family}: child runner script is unavailable (${runner_script})"
    return 0
  fi
  if [[ "${task_count}" == "0" ]]; then
    echo "skip runner_family=${runner_family}: no runnable tasks selected"
    return 0
  fi
  if [[ -z "${agent_names}" ]]; then
    echo "skip runner_family=${runner_family}: no agents selected"
    return 0
  fi

  mkdir -p "${output_subdir}"

  local env_args=(
    "ANNOTATIONS_JSON=${ANNOTATIONS_JSON}"
    "TASK_LIST_FILE=${task_list_file}"
    "AGENT_LIST_FILE=${agent_list_file}"
    "AGENTS=${agent_names}"
    "AGENT_EPISODES=${agent_episodes}"
    "OUTPUT_ROOT=${output_subdir}"
    "OUTPUT_DIR=${output_subdir}"
    "BENCHMARK_INFO_JSON=${OUTPUT_DIR}/benchmark_info.json"
    "RUNNER_SELECTION_JSON=${selection_json}"
    "RUNNER_SKIPPED_SELECTION_JSON=${skipped_json}"
    "SUMMARY_GROUP_BY_FILE=${summary_group_by_file}"
    "RUNNER_FAMILY=${runner_family}"
    "CONDA_ENV_NAME=${CONDA_ENV_NAME}"
    "SUMMARIZE_AT_END=${SUMMARIZE_AT_END}"
    "TIMESTAMP=${TIMESTAMP}"
  )
  maybe_append_optional_env env_args HEADLESS
  maybe_append_optional_env env_args DISABLE_FABRIC

  echo "tasks=$(json_list_joined "${task_list_file}")"
  echo "agents=${agent_names}"
  echo "output=${output_subdir}"
  echo "runner_script=${runner_script}"
  echo "------------------------------------------------------------------------"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'dry_run env %q' "${env_args[0]}"
    local env_item
    for env_item in "${env_args[@]:1}"; do
      printf ' %q' "${env_item}"
    done
    printf ' bash %q\n' "${runner_script}"
    return 0
  fi

  env "${env_args[@]}" bash "${runner_script}"
}

summarize_family_root() {
  local runner_family="$1"
  local input_dir="$2"
  local group_by_file="$3"
  local summarizer="$4"
  local csv_out="$5"
  local txt_out="$6"

  if [[ -z "${summarizer}" || ! -f "${summarizer}" ]]; then
    return 0
  fi
  if [[ ! -d "${input_dir}" ]]; then
    return 0
  fi

  local summary_count
  summary_count="$(find "${input_dir}" -type f -name '*.summary.json' | wc -l | tr -d ' ')"
  if [[ "${summary_count}" == "0" ]]; then
    return 0
  fi

  mapfile -t group_by < "${group_by_file}"
  python "${summarizer}" "${input_dir}" --annotations "${ANNOTATIONS_JSON}" --group-by "${group_by[@]}" --csv-out "${csv_out}" \
    | tee "${txt_out}"
}

combine_run_manifests() {
  python - "${OUTPUT_DIR}" "${RUNNER_PLAN_JSON}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runner_plan = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
output_path = root / "run_manifest.tsv"
rows = ["runner_key\ttask_name\tfamily\tvariant\tagent_name\tscene_id\tlayout_key\tsummary_json\ttask_log"]
for item in runner_plan:
    runner_family = str(item.get("runner_family", "")).strip()
    output_subdir = str(item.get("output_subdir", "")).strip()
    if not runner_family or not output_subdir:
        continue
    path = root / output_subdir / "run_manifest.tsv"
    if not path.is_file():
        continue
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        rows.append(f"{runner_family}\t{stripped}")
output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
PY
}

summarize_joint_root() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'dry_run: joint summary skipped\n' > "${OUTPUT_DIR}/joint_summary.txt"
    return 0
  fi

  python - "${OUTPUT_DIR}/benchmark_info.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
group_by = []
summary_preset = payload.get("summary_preset", {})
if isinstance(summary_preset, dict):
    values = summary_preset.get("joint_group_by", [])
    if isinstance(values, list):
        group_by = [str(value) for value in values]
print("\n".join(group_by))
PY
}

print_dispatch_warnings() {
  if [[ -s "${OUTPUT_DIR}/dispatch_warnings.txt" ]]; then
    echo
    echo "------------------------------------------------------------------------"
    echo "dispatch_warnings"
    echo "------------------------------------------------------------------------"
    cat "${OUTPUT_DIR}/dispatch_warnings.txt"
  fi
}

main() {
  mkdir -p "${OUTPUT_DIR}"
  print_header
  prepare_manifest_selection
  print_dispatch_warnings

  printf 'runner_key\ttask_name\tfamily\tvariant\tagent_name\tscene_id\tlayout_key\tsummary_json\ttask_log\n' \
    > "${OUTPUT_DIR}/run_manifest.tsv"

  while IFS='|' read -r runner_family implemented child_runner_script_name summarizer_script_name output_subdir_rel task_list_file agent_list_file group_by_file selection_json skipped_json task_count agent_count selected_descriptor_count skipped_descriptor_count agent_episode_env default_agent_episodes notes; do
    [[ -z "${runner_family}" ]] && continue
    local runner_script=""
    local summarizer=""
    local output_subdir="${OUTPUT_DIR}/${output_subdir_rel}"
    local agent_episodes
    agent_episodes="$(resolve_agent_episodes "${agent_episode_env}" "${default_agent_episodes}")"
    if [[ -n "${child_runner_script_name}" ]]; then
      runner_script="${SCRIPT_DIR}/${child_runner_script_name}"
    fi
    if [[ -n "${summarizer_script_name}" ]]; then
      summarizer="${SCRIPT_DIR}/${summarizer_script_name}"
    fi

    run_child_runner \
      "${runner_family}" \
      "${implemented}" \
      "${runner_script}" \
      "${task_list_file}" \
      "${agent_list_file}" \
      "${output_subdir}" \
      "${agent_episodes}" \
      "${selection_json}" \
      "${skipped_json}" \
      "${group_by_file}" \
      "${notes}" \
      "${selected_descriptor_count}" \
      "${skipped_descriptor_count}"

    if [[ "${DRY_RUN}" != "1" ]]; then
      summarize_family_root \
        "${runner_family}" \
        "${output_subdir}" \
        "${group_by_file}" \
        "${summarizer}" \
        "${OUTPUT_DIR}/aggregate_${runner_family}.csv" \
        "${OUTPUT_DIR}/aggregate_${runner_family}.txt"
    fi
  done < <(emit_runner_plan_rows "${RUNNER_PLAN_JSON}")

  combine_run_manifests

  if [[ "${DRY_RUN}" != "1" ]]; then
    mapfile -t joint_group_by < <(summarize_joint_root)
    python "${JOINT_SUMMARIZER}" "${OUTPUT_DIR}" --group-by "${joint_group_by[@]}" --csv-out "${OUTPUT_DIR}/joint_summary.csv" \
      | tee "${OUTPUT_DIR}/joint_summary.txt"
  else
    printf 'dry_run: joint summary skipped\n' > "${OUTPUT_DIR}/joint_summary.txt"
  fi

  echo
  echo "========================================================================"
  echo "benchmark_info=${OUTPUT_DIR}/benchmark_info.json"
  echo "dispatch_report=${OUTPUT_DIR}/runner_dispatch_report.json"
  echo "dispatch_warnings=${OUTPUT_DIR}/dispatch_warnings.txt"
  echo "run_manifest=${OUTPUT_DIR}/run_manifest.tsv"
  if [[ "${DRY_RUN}" != "1" ]]; then
    echo "joint_summary=${OUTPUT_DIR}/joint_summary.csv"
  fi
  echo "========================================================================"
}

main
