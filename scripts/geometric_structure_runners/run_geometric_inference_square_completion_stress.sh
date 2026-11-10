#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="square_completion" SCENE_ID="10" LAYOUT_KEY="geometric_inference_square_completion_tight_obstacle" TASK_INDEX="0" AGENT_DEFAULT="oracle" TAU_LINE="${TAU_LINE:-0.015}" TAU_YAW="${TAU_YAW:-0.18}" EPSILON_POS="${EPSILON_POS:-0.04}" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
