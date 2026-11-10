#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="perfect_midpoint" SCENE_ID="65" LAYOUT_KEY="geometric_inference_perfect_midpoint_line_2" TASK_INDEX="0" AGENT_DEFAULT="explicit_midpoint_oracle" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
