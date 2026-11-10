#!/usr/bin/env bash
# Sequentially run GeoMind VLA batch eval for every model JSON under geomind/configs/vla_eval_models.
# Extra CLI args are forwarded to run_vla_eval.py (e.g. --tasks T1,T2 --dry-run).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults for every model in this suite (override with MAX_STEPS / VLA_EVAL_RUNS_PER_LEVEL in the environment).
export MAX_STEPS="${MAX_STEPS:-40}"
export VLA_EVAL_RUNS_PER_LEVEL="${VLA_EVAL_RUNS_PER_LEVEL:-8}"

VLA_EVAL_MODELS_DIR="${VLA_EVAL_MODELS_DIR:-${REPO_ROOT}/geomind/configs/vla_eval_models}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${SCRIPT_DIR}/run_vla_eval.py}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_LOG="${SUITE_LOG:-${REPO_ROOT}/run_eval_all_models_suites/suite_${TIMESTAMP}.log}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
SKIP_MODELS="${SKIP_MODELS:-}"
ONLY_MODELS="${ONLY_MODELS:-}"

skip_contains() {
  local needle="$1"
  local token
  for token in ${SKIP_MODELS//,/ }; do
    [[ -z "${token}" ]] && continue
    if [[ "${needle}" == "${token}" ]]; then
      return 0
    fi
  done
  return 1
}

only_passes() {
  [[ -z "${ONLY_MODELS// /}" ]] && return 0
  local needle="$1"
  local token
  for token in ${ONLY_MODELS//,/ }; do
    [[ -z "${token}" ]] && continue
    if [[ "${needle}" == "${token}" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ ! -d "${VLA_EVAL_MODELS_DIR}" ]]; then
  echo "error: VLA_EVAL_MODELS_DIR is not a directory: ${VLA_EVAL_MODELS_DIR}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "error: EVAL_SCRIPT not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${SUITE_LOG}")"

log() {
  echo "$@" | tee -a "${SUITE_LOG}"
}

log "========================================================================"
log "GeoMind VLA eval — all models in ${VLA_EVAL_MODELS_DIR}"
log "timestamp=${TIMESTAMP}"
log "suite_log=${SUITE_LOG}"
log "eval_script=${EVAL_SCRIPT}"
log "stop_on_failure=${STOP_ON_FAILURE}"
log "skip_models=${SKIP_MODELS:-<none>}"
log "only_models=${ONLY_MODELS:-<all>}"
log "max_steps=${MAX_STEPS}"
log "runs_per_level(VLA_EVAL_RUNS_PER_LEVEL)=${VLA_EVAL_RUNS_PER_LEVEL}"
log "forwarded_args=$*"
log "========================================================================"

shopt -s nullglob
mapfile -t json_paths < <(find "${VLA_EVAL_MODELS_DIR}" -maxdepth 1 -name '*.json' -type f | LC_ALL=C sort)
shopt -u nullglob

if [[ "${#json_paths[@]}" -eq 0 ]]; then
  log "error: no *.json under ${VLA_EVAL_MODELS_DIR}" >&2
  exit 1
fi

failures=0
ran=0
for path in "${json_paths[@]}"; do
  stem="$(basename "${path}" .json)"
  if skip_contains "${stem}"; then
    log "[skip] model=${stem} (SKIP_MODELS)"
    continue
  fi
  if ! only_passes "${stem}"; then
    log "[skip] model=${stem} (not in ONLY_MODELS)"
    continue
  fi

  ran=$((ran + 1))
  log "------------------------------------------------------------------------"
  log "[${ran}] VLA_EVAL_MODEL=${stem}"
  log "------------------------------------------------------------------------"

  set +e
  VLA_EVAL_MODEL="${stem}" python3 "${EVAL_SCRIPT}" "$@" 2>&1 | tee -a "${SUITE_LOG}"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 ]]; then
    failures=$((failures + 1))
    log "[fail] model=${stem} exit=${status}"
    if [[ "${STOP_ON_FAILURE}" == "1" ]]; then
      log "stopping_on_failure model=${stem}"
      break
    fi
  else
    log "[ok] model=${stem}"
  fi
done

log "========================================================================"
log "done ran=${ran} failures=${failures} suite_log=${SUITE_LOG}"
log "========================================================================"

if [[ "${failures}" -gt 0 ]]; then
  exit 1
fi
exit 0
