# Geometric structure task runners

每个 `run_*.sh` 都对应一个单独的几何结构推理任务，方便逐任务调参和排查。

常用覆写方式：

- `AGENT=explicit_midpoint_oracle|explicit_extension_oracle|explicit_random|implicit_trend_oracle|implicit_trend_random|oracle|random`
- `AGENT_EPISODES=<int>`
- `AGENT_SEED=<int>`
- `HEADLESS=0|1`
- `OUTPUT_DIR=<path>`
- `ANNOTATIONS_JSON=<path>`

旧的聚合脚本仍保留在上层 `scripts/` 目录。
