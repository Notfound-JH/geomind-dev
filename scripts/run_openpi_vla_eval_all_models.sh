#!/usr/bin/env bash
# Full GeoMind VLA batch eval for every OpenPI model JSON (same loop as run_vla_eval_all_models.sh).
# Sets VLA_EVAL_CONFIG_DIR so run_vla_eval.py loads geomind/configs/openpi_eval_models/<name>.json
# (not vla_eval_models/). Extra CLI args are forwarded to run_vla_eval.py (e.g. --tasks T1,T2 --dry-run).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MAX_STEPS="${MAX_STEPS:-50}"
export VLA_EVAL_RUNS_PER_LEVEL="${VLA_EVAL_RUNS_PER_LEVEL:-4}"

OPENPI_EVAL_MODELS_DIR="${OPENPI_EVAL_MODELS_DIR:-${REPO_ROOT}/geomind/configs/openpi_eval_models}"
if [[ ! -d "${OPENPI_EVAL_MODELS_DIR}" && -d "${OPENPI_EVAL_MODELS_DIR} " ]]; then
  OPENPI_EVAL_MODELS_DIR="${OPENPI_EVAL_MODELS_DIR} "
fi

export VLA_EVAL_CONFIG_DIR="${VLA_EVAL_CONFIG_DIR:-${OPENPI_EVAL_MODELS_DIR}}"

# geomind_client OpenPI websocket: default wait OPENPI_WS_CONNECT_MAX_WAIT_S=120 (0 = stock infinite wait).
# OPENPI_WS_CONNECT_RETRY_S, OPENPI_WS_OPEN_TIMEOUT_S; OPENPI_WS_CONNECT_LOG=0 silences connect prints.

EVAL_SCRIPT="${EVAL_SCRIPT:-${SCRIPT_DIR}/run_vla_eval.py}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_LOG="${SUITE_LOG:-${REPO_ROOT}/run_eval_all_models_suites/openpi_vla_suite_${TIMESTAMP}.log}"
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

if [[ ! -d "${OPENPI_EVAL_MODELS_DIR}" ]]; then
  echo "error: OPENPI_EVAL_MODELS_DIR is not a directory: ${OPENPI_EVAL_MODELS_DIR}" >&2
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
log "GeoMind OpenPI VLA eval — all models in ${OPENPI_EVAL_MODELS_DIR}"
log "VLA_EVAL_CONFIG_DIR=${VLA_EVAL_CONFIG_DIR}"
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
mapfile -t json_paths < <(find "${OPENPI_EVAL_MODELS_DIR}" -maxdepth 1 -name '*.json' -type f | LC_ALL=C sort)
shopt -u nullglob

if [[ "${#json_paths[@]}" -eq 0 ]]; then
  log "error: no *.json under ${OPENPI_EVAL_MODELS_DIR}" >&2
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
  log "[${ran}] OPENPI_VLA_MODEL=${stem}"
  log "config=${path}"
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
