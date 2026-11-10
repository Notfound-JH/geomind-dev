#!/usr/bin/env bash
# OpenPI 全量 VLA 评测（所有任务）已迁至 run_openpi_vla_eval_all_models.sh。
# 本脚本保留为兼容入口，直接转发；原 openpi_client 冒烟见 scripts/openpi_client_all_models_main.py（需手动调用）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_openpi_vla_eval_all_models.sh" "$@"
