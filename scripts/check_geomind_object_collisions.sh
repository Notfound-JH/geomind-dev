#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Leave SCENE_IDS empty to check every scene under GEOMIND_ASSETS_ROOT/stage.
SCENE_IDS=()
LOG_DIR="${SCRIPT_DIR}/../logs"
COLLISION_EPS="1e-6"

source "${SCRIPT_DIR}/lib/geomind_conda.sh"
geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1

ARGS=(
  --log-dir "${LOG_DIR}"
  --collision-eps "${COLLISION_EPS}"
)

if [[ "${#SCENE_IDS[@]}" -gt 0 ]]; then
  ARGS+=(--scene-ids "${SCENE_IDS[@]}")
fi

python "${SCRIPT_DIR}/check_geomind_object_collisions.py" "${ARGS[@]}" "$@"
