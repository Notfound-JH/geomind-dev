#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_geomind_suite.sh [suite|manifest.json] [options]

Suites:
  phase1           Run the recommended Phase 1 suite. Default.
  main             Run the main benchmark manifest.
  phase2-reserve   Run the Phase 2 reserve manifest.
  <path>.json      Run a custom benchmark manifest.

Options:
  --dry-run                  Resolve tasks/agents without launching child runners.
  --output DIR               Override output directory.
  --task-version VALUES      Space-separated or comma-separated version filter.
  --phase VALUES             Space-separated or comma-separated phase filter.
  --runner-mode MODE         Dispatch mode: permissive or strict.
  --strict                   Shortcut for --runner-mode strict.
  --permissive               Shortcut for --runner-mode permissive.
  --tasks VALUES             Override task selection passed to the benchmark runner.
  --agents VALUES            Override agent selection passed to the benchmark runner.
  --annotations PATH         Override annotated_tasks.json path.
  --conda-env NAME           Override CONDA_ENV_NAME. Default: isaac_curobo
  --headless                 Export HEADLESS=1 for child runners.
  --disable-fabric           Export DISABLE_FABRIC=1 for child runners.
  -h, --help                 Show this help.

Examples:
  bash scripts/run_geomind_suite.sh
  bash scripts/run_geomind_suite.sh phase1 --dry-run
  bash scripts/run_geomind_suite.sh main --task-version base
  bash scripts/run_geomind_suite.sh configs/geomind_benchmark_phase1.json --output /tmp/geomind_run
EOF
}

normalize_list_arg() {
  local raw="$1"
  raw="${raw//,/ }"
  echo "${raw}"
}

resolve_manifest_path() {
  local suite="$1"
  case "${suite}" in
    ""|"phase1")
      echo "${REPO_ROOT}/configs/geomind_benchmark_phase1.json"
      ;;
    "main")
      echo "${REPO_ROOT}/configs/geomind_benchmark_main.json"
      ;;
    "phase2-reserve"|"phase2_reserve")
      echo "${REPO_ROOT}/configs/geomind_benchmark_phase2_reserve.json"
      ;;
    *.json)
      if [[ "${suite}" = /* ]]; then
        echo "${suite}"
      else
        echo "${REPO_ROOT}/${suite}"
      fi
      ;;
    *)
      echo ""
      ;;
  esac
}

suite="phase1"
manifest_arg=""
output_dir=""
dry_run="0"
task_version_filter=""
phase_filter=""
runner_dispatch_mode="${RUNNER_DISPATCH_MODE:-}"
task_names_override=""
agent_names_override=""
annotations_json="${ANNOTATIONS_JSON:-}"
conda_env_name="${CONDA_ENV_NAME:-isaac_curobo}"
headless="0"
disable_fabric=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      dry_run="1"
      shift
      ;;
    --output)
      output_dir="${2:-}"
      shift 2
      ;;
    --task-version)
      task_version_filter="$(normalize_list_arg "${2:-}")"
      shift 2
      ;;
    --phase)
      phase_filter="$(normalize_list_arg "${2:-}")"
      shift 2
      ;;
    --tasks)
      task_names_override="$(normalize_list_arg "${2:-}")"
      shift 2
      ;;
    --runner-mode)
      runner_dispatch_mode="${2:-}"
      shift 2
      ;;
    --strict)
      runner_dispatch_mode="strict"
      shift
      ;;
    --permissive)
      runner_dispatch_mode="permissive"
      shift
      ;;
    --agents)
      agent_names_override="$(normalize_list_arg "${2:-}")"
      shift 2
      ;;
    --annotations)
      annotations_json="${2:-}"
      shift 2
      ;;
    --conda-env)
      conda_env_name="${2:-}"
      shift 2
      ;;
    --headless)
      headless="1"
      shift
      ;;
    --disable-fabric)
      disable_fabric="1"
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -z "${manifest_arg}" ]]; then
        manifest_arg="$1"
        suite="$1"
      else
        echo "Unexpected positional argument: $1" >&2
        usage >&2
        exit 2
      fi
      shift
      ;;
  esac
done

manifest_path="$(resolve_manifest_path "${suite}")"
if [[ -z "${manifest_path}" ]]; then
  echo "Unsupported suite: ${suite}" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "${manifest_path}" ]]; then
  echo "Manifest not found: ${manifest_path}" >&2
  exit 2
fi

if [[ -z "${output_dir}" ]]; then
  timestamp="$(date +%Y%m%d_%H%M%S)"
  suite_slug="$(basename "${manifest_path}" .json)"
  output_dir="${REPO_ROOT}/logs/geomind_benchmark/${timestamp}_${suite_slug}"
fi

env_args=(
  "CONDA_ENV_NAME=${conda_env_name}"
  "DRY_RUN=${dry_run}"
)

if [[ -n "${annotations_json}" ]]; then
  env_args+=("ANNOTATIONS_JSON=${annotations_json}")
fi
if [[ -n "${task_version_filter}" ]]; then
  env_args+=("TASK_VERSION_FILTER=${task_version_filter}")
fi
if [[ -n "${phase_filter}" ]]; then
  env_args+=("PHASE_FILTER=${phase_filter}")
fi
if [[ -n "${task_names_override}" ]]; then
  env_args+=("TASK_NAMES_OVERRIDE=${task_names_override}")
fi
if [[ -n "${runner_dispatch_mode}" ]]; then
  env_args+=("RUNNER_DISPATCH_MODE=${runner_dispatch_mode}")
fi
if [[ -n "${agent_names_override}" ]]; then
  env_args+=("AGENT_NAMES_OVERRIDE=${agent_names_override}")
fi
if [[ -n "${headless}" ]]; then
  env_args+=("HEADLESS=${headless}")
fi
if [[ -n "${disable_fabric}" ]]; then
  env_args+=("DISABLE_FABRIC=${disable_fabric}")
fi

echo "suite_manifest=${manifest_path}"
echo "output_dir=${output_dir}"
if [[ -n "${task_version_filter}" ]]; then
  echo "task_version_filter=${task_version_filter}"
fi
if [[ -n "${phase_filter}" ]]; then
  echo "phase_filter=${phase_filter}"
fi
if [[ -n "${task_names_override}" ]]; then
  echo "tasks_override=${task_names_override}"
fi
if [[ -n "${runner_dispatch_mode}" ]]; then
  echo "runner_dispatch_mode=${runner_dispatch_mode}"
fi
if [[ -n "${agent_names_override}" ]]; then
  echo "agents_override=${agent_names_override}"
fi

env "${env_args[@]}" bash "${SCRIPT_DIR}/run_geomind_benchmark.sh" "${manifest_path}" "${output_dir}"
