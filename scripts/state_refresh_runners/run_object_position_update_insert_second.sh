#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="reindex_insert" SCENE_ID="1" LAYOUT_KEY="object_position_update_insert_line_4" TASK_INDEX="0" AGENT_DEFAULT="oracle" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
