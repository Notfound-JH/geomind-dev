#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="reindex_subset" SCENE_ID="10" LAYOUT_KEY="state_reindex_red_subset_blocks_line_6_tight" TASK_INDEX="0" AGENT_DEFAULT="oracle" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
