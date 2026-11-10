#!/usr/bin/env bash
# 对 run_eval_openvla 全量人工复核：不抽样、不把 result.json 成功自动标过（每条都进队）。
# 用法：在仓库根目录执行 ./scripts/review_run_eval_openvla_all.sh
# 可选附加参数传给 human_review_run_videos.py，例如：--player line
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/scripts/human_review_run_videos.py" "$ROOT/run_eval_openvla" \
  --max-human-per-task 0 \
  --no-auto-rule-success \
  "$@"
