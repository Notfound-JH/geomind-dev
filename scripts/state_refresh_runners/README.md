# State refresh task runners

每个 `run_*.sh` 都对应一个单独的状态刷新（reindexing / identity tracking）协议任务，方便逐任务精细微调。

常用覆写方式：

- `AGENT=oracle|random`
- `CONTROL_MODE=final_frame_only`
- `AGENT_EPISODES=<int>`
- `HEADLESS=0|1`
- `OUTPUT_DIR=<path>`
- `ANNOTATIONS_JSON=<path>`

旧的聚合脚本仍保留在上层 `scripts/` 目录。
