#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="reindex_neighbor" SCENE_ID="10" LAYOUT_KEY="state_reindex_neighbor_red_left_blocks_6_tight" TASK_INDEX="0" AGENT_DEFAULT="oracle" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
