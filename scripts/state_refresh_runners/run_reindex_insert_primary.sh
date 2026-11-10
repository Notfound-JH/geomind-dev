#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="reindex_insert" SCENE_ID="65" LAYOUT_KEY="state_reindex_insert_blocks_line_4" TASK_INDEX="0" AGENT_DEFAULT="oracle" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
