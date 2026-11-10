#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="track_clutter_stress" SCENE_ID="10" LAYOUT_KEY="clutter_robust_identity_blocks_line_3_tight" TASK_INDEX="0" AGENT_DEFAULT="oracle" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
