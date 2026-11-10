#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="perfect_midpoint" SCENE_ID="10" LAYOUT_KEY="geometric_inference_midpoint_grid_2_2_top_row" TASK_INDEX="0" AGENT_DEFAULT="explicit_midpoint_oracle" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
