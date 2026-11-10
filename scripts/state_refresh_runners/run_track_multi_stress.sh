#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POST_INTERVENTION_SETTLE_STEPS="120" TASK_ID="track_multi" SCENE_ID="10" LAYOUT_KEY="historical_identity_multistep_blocks_line_5_tight" TASK_INDEX="0" AGENT_DEFAULT="oracle" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
