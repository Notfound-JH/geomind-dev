#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="isometric_extension" SCENE_ID="1" LAYOUT_KEY="geometric_inference_extension_vertical_line_3" TASK_INDEX="0" AGENT_DEFAULT="explicit_extension_oracle" exec bash "${SCRIPT_DIR}/_run_single_task.sh" "$@"
