#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="weak_destruction_control" SCENE_ID="10" LAYOUT_KEY="geometric_inference_implicit_trend_weak_destruction_tight_obstacle" TASK_INDEX="0" AGENT_DEFAULT="implicit_trend_oracle" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
