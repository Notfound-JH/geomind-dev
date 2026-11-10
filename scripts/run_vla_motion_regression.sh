#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
VLA_EVAL_CONFIG_DIR="${VLA_EVAL_CONFIG_DIR:-${REPO_ROOT}/geomind/configs/vla_eval_models}"
REMOTE_POLICY_PORT_MODEL_MAP="${REMOTE_POLICY_PORT_MODEL_MAP:-8000=spatialforcing,8001=spatialvla,8002=internvla}"

eval_config_value() {
  local config_path="$1"
  local key="$2"
  python - "${config_path}" "${key}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.is_file():
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
value = payload.get(key, "")
if value is None:
    value = ""
print(value)
PY
}

remote_policy_model_from_url() {
  local url="$1"
  local port=""
  if [[ "${url}" =~ :([0-9]+)(/|$) ]]; then
    port="${BASH_REMATCH[1]}"
  fi
  local pair
  for pair in ${REMOTE_POLICY_PORT_MODEL_MAP//,/ }; do
    if [[ "${pair}" == "${port}="* ]]; then
      echo "${pair#*=}"
      return
    fi
  done
  case "${port}" in
    8000) echo "spatialforcing" ;;
    8001) echo "spatialvla" ;;
    8002) echo "internvla" ;;
    *) echo "remote_${port:-unknown}" ;;
  esac
}

REMOTE_POLICY_MODEL="${VLA_EVAL_MODEL:-${REMOTE_POLICY_MODEL:-}}"
REMOTE_POLICY_URL="${REMOTE_POLICY_URL:-}"
if [[ -z "${REMOTE_POLICY_MODEL}" && -n "${REMOTE_POLICY_URL}" ]]; then
  REMOTE_POLICY_MODEL="$(remote_policy_model_from_url "${REMOTE_POLICY_URL}")"
fi
REMOTE_POLICY_MODEL="${REMOTE_POLICY_MODEL:-spatialvla}"
VLA_EVAL_CONFIG_PATH="${VLA_EVAL_CONFIG_PATH:-${VLA_EVAL_CONFIG_DIR}/${REMOTE_POLICY_MODEL}.json}"
CONFIG_REMOTE_POLICY_URL="$(eval_config_value "${VLA_EVAL_CONFIG_PATH}" "remote_policy_url")"
CONFIG_REMOTE_POLICY_ACTION_SCALE="$(eval_config_value "${VLA_EVAL_CONFIG_PATH}" "remote_policy_action_scale")"
CONFIG_REMOTE_POLICY_DISABLE_ROTATION="$(eval_config_value "${VLA_EVAL_CONFIG_PATH}" "remote_policy_disable_rotation")"
CONFIG_REMOTE_POLICY_OVERSHOOT_GUARD="$(eval_config_value "${VLA_EVAL_CONFIG_PATH}" "remote_policy_overshoot_guard")"
CONFIG_OUTPUT_DIR_NAME="$(eval_config_value "${VLA_EVAL_CONFIG_PATH}" "output_dir_name")"
REMOTE_POLICY_URL="${REMOTE_POLICY_URL_OVERRIDE:-${REMOTE_POLICY_URL:-${CONFIG_REMOTE_POLICY_URL:-http://127.0.0.1:8001/infer}}}"
REMOTE_POLICY_ACTION_SCALE="${REMOTE_POLICY_ACTION_SCALE_OVERRIDE:-${CONFIG_REMOTE_POLICY_ACTION_SCALE:-${REMOTE_POLICY_ACTION_SCALE:-1}}}"
REMOTE_POLICY_DISABLE_ROTATION="${REMOTE_POLICY_DISABLE_ROTATION_OVERRIDE:-${CONFIG_REMOTE_POLICY_DISABLE_ROTATION:-${REMOTE_POLICY_DISABLE_ROTATION:-0}}}"
REMOTE_POLICY_OVERSHOOT_GUARD="${REMOTE_POLICY_OVERSHOOT_GUARD_OVERRIDE:-${CONFIG_REMOTE_POLICY_OVERSHOOT_GUARD:-${REMOTE_POLICY_OVERSHOOT_GUARD:-0}}}"
REMOTE_POLICY_OUTPUT_DIR_NAME="${REMOTE_POLICY_OUTPUT_DIR_NAME:-${CONFIG_OUTPUT_DIR_NAME:-run_eval_${REMOTE_POLICY_MODEL}}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/${REMOTE_POLICY_OUTPUT_DIR_NAME}/motion_regression/${TIMESTAMP}}"
MAX_STEPS="${MAX_STEPS:-10}"
BASE_SEED="${BASE_SEED:-20260502}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
TASKS="${TASKS:-}"

if [[ -z "${TASKS}" ]]; then
  TASKS="\
SelectByGridIndexVla_v0 \
SelectByOrderSecondVla_v0 \
SelectByOrderFirstVla_v0 \
TrackObjectOriginClutterVla_v0 \
SelectByAdjacencyVla_v0 \
SelectWithinSubsetVla_v0"
fi

mkdir -p "${OUTPUT_ROOT}"
MANIFEST_TSV="${OUTPUT_ROOT}/manifest.tsv"
SUMMARY_JSON="${OUTPUT_ROOT}/summary.json"
SUMMARY_MD="${OUTPUT_ROOT}/summary.md"
SUITE_LOG="${OUTPUT_ROOT}/suite.log"

printf "task_index\ttask_id\tseed\tstatus\toutput_root\tresult_json\tlog\n" > "${MANIFEST_TSV}"

echo "========================================================================" | tee "${SUITE_LOG}"
echo "GeoMind VLA motion regression" | tee -a "${SUITE_LOG}"
echo "========================================================================" | tee -a "${SUITE_LOG}"
echo "output_root=${OUTPUT_ROOT}" | tee -a "${SUITE_LOG}"
echo "remote_policy_url=${REMOTE_POLICY_URL}" | tee -a "${SUITE_LOG}"
echo "remote_policy_model=${REMOTE_POLICY_MODEL}" | tee -a "${SUITE_LOG}"
echo "remote_policy_action_scale=${REMOTE_POLICY_ACTION_SCALE}" | tee -a "${SUITE_LOG}"
echo "remote_policy_disable_rotation=${REMOTE_POLICY_DISABLE_ROTATION}" | tee -a "${SUITE_LOG}"
echo "remote_policy_overshoot_guard=${REMOTE_POLICY_OVERSHOOT_GUARD}" | tee -a "${SUITE_LOG}"
echo "vla_eval_config_path=${VLA_EVAL_CONFIG_PATH}" | tee -a "${SUITE_LOG}"
echo "remote_policy_port_model_map=${REMOTE_POLICY_PORT_MODEL_MAP}" | tee -a "${SUITE_LOG}"
echo "max_steps=${MAX_STEPS}" | tee -a "${SUITE_LOG}"
echo "base_seed=${BASE_SEED}" | tee -a "${SUITE_LOG}"
echo "stop_on_failure=${STOP_ON_FAILURE}" | tee -a "${SUITE_LOG}"
echo "tasks=${TASKS}" | tee -a "${SUITE_LOG}"
echo "========================================================================" | tee -a "${SUITE_LOG}"

index=0
for task_id in ${TASKS}; do
  seed=$((BASE_SEED + index * 1009))
  task_root="${OUTPUT_ROOT}/${task_id}/seed_${seed}"
  episode_dir="${task_root}/episode_0000"
  task_log="${task_root}/run.log"

  mkdir -p "${task_root}" "${episode_dir}"
  echo "[${index}] task=${task_id} seed=${seed}" | tee -a "${SUITE_LOG}"

  set +e
  TASK_ID_ENV="${task_id}" \
  REMOTE_POLICY_URL="${REMOTE_POLICY_URL}" \
  REMOTE_POLICY_MODEL="${REMOTE_POLICY_MODEL}" \
  REMOTE_POLICY_ACTION_SCALE="${REMOTE_POLICY_ACTION_SCALE}" \
  REMOTE_POLICY_DISABLE_ROTATION="${REMOTE_POLICY_DISABLE_ROTATION}" \
  REMOTE_POLICY_OVERSHOOT_GUARD="${REMOTE_POLICY_OVERSHOOT_GUARD}" \
  VLA_EVAL_CONFIG_PATH="${VLA_EVAL_CONFIG_PATH}" \
  REMOTE_POLICY_PORT_MODEL_MAP="${REMOTE_POLICY_PORT_MODEL_MAP}" \
  SEED="${seed}" \
  MAX_STEPS="${MAX_STEPS}" \
  RUN_NAME="${REMOTE_POLICY_MODEL}_motion_regression_${task_id}_seed_${seed}" \
  OUTPUT_ROOT="${task_root}" \
  OUTPUT_DIR="${episode_dir}" \
  bash "${REPO_ROOT}/scripts/run_remote_policy_isaaclab_vla.sh" 2>&1 | tee "${task_log}"
  status=${PIPESTATUS[0]}
  set -e

  result_json="${episode_dir}/result.json"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${index}" "${task_id}" "${seed}" "${status}" "${task_root}" "${result_json}" "${task_log}" \
    >> "${MANIFEST_TSV}"

  if [[ "${status}" != "0" && "${STOP_ON_FAILURE}" == "1" ]]; then
    echo "stopping_on_failure task=${task_id} status=${status}" | tee -a "${SUITE_LOG}"
    break
  fi

  index=$((index + 1))
done

python - "${MANIFEST_TSV}" "${SUMMARY_JSON}" "${SUMMARY_MD}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def pick_success(payload: dict[str, Any]) -> Any:
    for key in ("success", "episode_success", "task_success"):
        if key in payload:
            return payload[key]
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for key in ("success", "episode_success", "task_success"):
            if key in metrics:
                return metrics[key]
    extras = payload.get("extras")
    if isinstance(extras, dict):
        for key in ("success", "episode_success", "task_success"):
            if key in extras:
                return extras[key]
        protocol = extras.get("protocol")
        if isinstance(protocol, dict):
            for key in ("success", "valid_episode", "intervention_success"):
                if key in protocol:
                    return protocol[key]
    return None


manifest_path = Path(sys.argv[1])
summary_json_path = Path(sys.argv[2])
summary_md_path = Path(sys.argv[3])

rows = list(csv.DictReader(manifest_path.open(encoding="utf-8"), delimiter="\t"))
items = []
for row in rows:
    result_path = Path(row["result_json"])
    payload = load_json(result_path)
    items.append(
        {
            "task_index": int(row["task_index"]),
            "task_id": row["task_id"],
            "seed": int(row["seed"]),
            "returncode": int(row["status"]),
            "ok": int(row["status"]) == 0,
            "success": pick_success(payload),
            "output_root": row["output_root"],
            "result_json": row["result_json"],
            "log": row["log"],
            "result_load_error": payload.get("_load_error"),
        }
    )

summary = {
    "num_tasks": len(items),
    "num_ok": sum(1 for item in items if item["ok"]),
    "num_failed": sum(1 for item in items if not item["ok"]),
    "items": items,
}
summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# VLA motion regression summary",
    "",
    f"- num_tasks: {summary['num_tasks']}",
    f"- num_ok: {summary['num_ok']}",
    f"- num_failed: {summary['num_failed']}",
    "",
    "| task | seed | returncode | success | output |",
    "|---|---:|---:|---|---|",
]
for item in items:
    lines.append(
        f"| {item['task_id']} | {item['seed']} | {item['returncode']} | "
        f"{item['success']} | `{item['output_root']}` |"
    )
summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "========================================================================" | tee -a "${SUITE_LOG}"
echo "summary_json=${SUMMARY_JSON}" | tee -a "${SUITE_LOG}"
echo "summary_md=${SUMMARY_MD}" | tee -a "${SUITE_LOG}"
echo "manifest_tsv=${MANIFEST_TSV}" | tee -a "${SUITE_LOG}"
echo "suite_log=${SUITE_LOG}" | tee -a "${SUITE_LOG}"
echo "========================================================================" | tee -a "${SUITE_LOG}"
