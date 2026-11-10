#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="trend_continuation" SCENE_ID="10" LAYOUT_KEY="geometric_inference_implicit_trend_tight_obstacle" TASK_INDEX="0" AGENT_DEFAULT="implicit_trend_oracle" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
