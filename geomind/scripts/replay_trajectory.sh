#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TRACE_JSON="${TRACE_JSON:-}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-}"
FPS="${FPS:-10}"

if [[ -z "${TRACE_JSON}" ]]; then
  echo "TRACE_JSON is required, e.g.:"
  echo "TRACE_JSON=geomind/outputs/env_policy/<run>/trace.json bash geomind/scripts/replay_trajectory.sh"
  exit 1
fi

ARGS=(
  --trace-json "${TRACE_JSON}"
  --fps "${FPS}"
)
if [[ -n "${OUTPUT_VIDEO}" ]]; then
  ARGS+=(--output-video "${OUTPUT_VIDEO}")
fi

echo "========================================================================"
echo "GeoMind Trajectory Replay"
echo "========================================================================"
echo "trace_json=${TRACE_JSON}"
echo "output_video=${OUTPUT_VIDEO}"
echo "fps=${FPS}"
echo "========================================================================"

cd "${REPO_ROOT}"
python -m geomind.tools.replay_trajectory "${ARGS[@]}" "$@"
