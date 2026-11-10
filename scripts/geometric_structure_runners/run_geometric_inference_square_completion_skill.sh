#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="square_completion" SCENE_ID="65" LAYOUT_KEY="geometric_inference_square_completion_corner_3_skill" TASK_INDEX="0" AGENT_DEFAULT="oracle" TAU_YAW="${TAU_YAW:-0.35}" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
