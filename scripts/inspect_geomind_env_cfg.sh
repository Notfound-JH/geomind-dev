#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_ROOT="${GEOMIND_ASSETS_ROOT:-/home/elm/projects/geomind_assets}"
STAGE_ROOT="${ASSETS_ROOT}/stage"

# Run:
#   bash scripts/inspect_geomind_env_cfg.sh
#
# If SCENE_IDS is empty, the script auto-discovers all scene ids under STAGE_ROOT.
# If you only want a subset, fill it manually, for example:
#   SCENE_IDS=(1 2 5 10)
#   SCENE_IDS=(10 12 15)
SCENE_IDS=()

LOG_FILE="${SCRIPT_DIR}/inspect_geomind_env_cfg.log"

if [[ ! -d "${STAGE_ROOT}" ]]; then
  echo "Stage root does not exist: ${STAGE_ROOT}" >&2
  exit 1
fi

if [[ "${#SCENE_IDS[@]}" -eq 0 ]]; then
  mapfile -t SCENE_IDS < <(find "${STAGE_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n)
fi

{
  printf '===== %s scene_count=%s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${#SCENE_IDS[@]}"
  printf 'stage_root=%s\n' "${STAGE_ROOT}"
  printf 'scene_ids=%s\n\n' "${SCENE_IDS[*]}"

  for SCENE_ID in "${SCENE_IDS[@]}"; do
    python "${SCRIPT_DIR}/inspect_geomind_env_cfg.py" "${SCENE_ID}"
    printf '\n'
  done
} 2>&1 | tee -a "${LOG_FILE}"
