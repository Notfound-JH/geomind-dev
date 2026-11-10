# Scripts Index

当前脚本入口按任务族拆成两层：

- `scripts/state_refresh_runners/`: 单个状态刷新（state reindexing / identity tracking）协议任务的独立 `run_*.sh`
- `scripts/geometric_structure_runners/`: 单个几何结构推理任务的独立 `run_*.sh`

兼容保留：

- 顶层 `scripts/run_state_refresh_protocol_baselines.sh`
- 顶层 `scripts/run_geometric_structure_protocol_baselines.sh`
- 其他 matrix / benchmark / debug 脚本

建议后续做逐任务精调时，优先从 `scripts/state_refresh_runners/` 和 `scripts/geometric_structure_runners/` 进入。

全量 debug 脚本清单见：`scripts/DEBUG_SCRIPTS.md`

## Debug 任务统一入口 (`debug_task.sh`)

本地调试各任务封装脚本时，可用统一入口（需 `python3` 读取 registry）：

- **入口脚本**：`scripts/debug_task.sh`
- **Registry**：`scripts/debug_task_registry.json`（字段：`id`、`category`、`entry`、`backend`）

常用命令：

```bash
# 列出全部已注册的 task_id（制表符分隔：id、category、entry）
bash scripts/debug_task.sh --list

# 短别名 -> canonical task_id（如 place1 -> place_directional）
bash scripts/debug_task.sh --list-aliases

# 按类别查看
bash scripts/debug_task.sh --list-categories

# 运行某个任务：可使用 registry 中的 task_id，或使用短别名（如 place2、structure6）
bash scripts/debug_task.sh place_directional
bash scripts/debug_task.sh place1
HEADLESS=1 bash scripts/debug_task.sh reindex_extreme
TASK_VERSION=stress SCENE_ID=10 bash scripts/debug_task.sh grasp_anchor_distance

# 需要给底层封装脚本传位置参数时，使用 --
bash scripts/debug_task.sh track_clutter -- stress
```

启动前会在 stdout 打印一块 `[Task Loaded]`（含 `alias`、`task_id`、`semantic_module`、`task_version`、`phase`、`instruction` 及环境变量中的 `scene_id` / `layout_key`；未设置时显示 `<unset>`，元数据不可用时显示 `<unknown>`。

**兼容性**：原有的 `scripts/debug_*.sh` 路径与用法不变，仍可直接执行；`debug_task.sh` 只是薄分发层。

**参数透传**：`HEADLESS`、`TASK_VERSION`、`SCENE_ID`、`LAYOUT_KEY`、`PRESET`、`AGENT` 等由各封装脚本自行从环境变量读取，统一入口不会拦截——在命令前导出即可。

**排查 `task_id` 找不到**：先 `bash scripts/debug_task.sh --list` 核对 id；若仍失败，检查 `debug_task_registry.json` 是否包含对应条目，以及 `entry` 指向的文件是否存在于 `scripts/`。

**协议层 / 后端脚本**：`debug_protocol_*.sh` 为各任务族的低层协议入口（Python + AppLauncher），通常由上层 `debug_*.sh` 包装后调用；registry 中 `category` 为 `protocol_backend` 的条目可直接调协议层（需自行设置 `TASK_ID` / `LAYOUT_KEY` 等）。

## Runner Contract

当前 benchmark runner 采用“统一总控 + family runner dispatch”结构：

- suite 入口：`scripts/run_geomind_suite.sh`
- 总控：`scripts/run_geomind_benchmark.sh`
- family registry / dispatch：`geomind_lab/mimic/utils/family_runner_registry.py`
- child runner：例如 `scripts/run_state_reindexing_matrix.sh`、`scripts/run_geometric_inference_matrix.sh`

职责边界：

- 总控负责 manifest 解析、task/agent 过滤、按 `runner_family` 分桶、strict/permissive 校验、结果汇总。
- child runner 只负责自己 family 的真实执行：读取任务列表，逐任务逐 agent 调对应 debug python，写 summary/log，再在 family 内汇总。

child runner 最小输入接口：

- `TASK_LIST_FILE`
- `AGENT_LIST_FILE`
- `OUTPUT_ROOT` 或 `OUTPUT_DIR`
- `BENCHMARK_INFO_JSON`
- `RUNNER_SELECTION_JSON`
- `RUNNER_SKIPPED_SELECTION_JSON`
- `SUMMARY_GROUP_BY_FILE`
- `RUNNER_FAMILY`
- `AGENT_EPISODES`

child runner 最少应产出：

- `<output_root>/run_manifest.tsv`
- `*.summary.json`
- `*.log`
- family 内汇总：`aggregate.csv` 和 `aggregate.txt`

dispatch 规则：

- `strict`：只要选中的任务里有未实现 family、未知 family、或 declared-but-not-runnable 任务，直接退出。
- `permissive`：允许继续执行已实现部分，但总控必须在 `runner_dispatch_report.json` 和 `dispatch_warnings.txt` 里显式记录被跳过的任务。

当前 state refresh / geometric structure runner 已视作符合该接口的后端实现：

- `scripts/run_state_reindexing_matrix.sh` 对应 `state_refresh`
- `scripts/run_geometric_inference_matrix.sh` 对应 `geometric_structure`

后续新增 family runner 时，优先做三件事：

1. 在 `family_runner_registry.py` 里注册新的 `runner_family`
2. 提供一个遵守上面输入/输出约定的 child runner 脚本
3. 不要再修改总控主结构

## Real Smoke Run

最小真实 smoke 建议直接跑：

```bash
timeout 600 bash scripts/run_geomind_smoke.sh
```

这会只选 1 个 state_refresh runnable task 和 1 个 geometric_structure runnable task：

- state_refresh: `reindex_extreme`
- geometric_structure: `perfect_midpoint`

默认使用 `strict` 模式、`headless=1`、每个 family 只跑 1 个 agent / 1 个 episode，目标是验证整条链：

- suite / manifest 解析
- family dispatch
- child runner
- debug python
- per-task summary
- family summary
- joint summary

注意：这是真实运行，不是 dry-run。本机如果没有可用的 Isaac Sim / CUDA / 图形设备，链路会在 child runner 调起 debug python 后失败，此时应先排查运行环境，而不是 dispatch 配置。

成功后应至少看到：

- `benchmark_info.json`
- `runner_dispatch_report.json`
- `run_manifest.tsv`
- `aggregate_state_refresh.csv`
- `aggregate_geometric_structure.csv`
- `joint_summary.csv`

如果 smoke 只跑默认这 2 个任务，那么通常应有：

- `run_manifest.tsv`：1 行表头 + 2 行任务
- `state_refresh/` 下 1 个 `*.summary.json`
- `geometric_structure/` 下 1 个 `*.summary.json`
- `runner_dispatch_report.json` 里 `skipped_task_descriptors` 为空

## New Task Probe

如果你要测试新的 canonical `task_id`，优先用：

```bash
bash scripts/run_geomind_new_task_probe.sh
```

默认会测试：

- `square_completion`
- `place_directional`

这个脚本会走现有 suite / dispatch 主链，并在结束后直接打印每个任务的：

- `task_id`
- `task_version`
- `phase`
- `runner_family`
- `dispatch_status`

如果某个新任务已经有实现好的 child runner，它会真实启动该 backend；如果还没有实现，则会在 `runner_dispatch_report.json` 里明确显示 `declared_runner_unimplemented`，不会静默消失。

常用例子：

```bash
bash scripts/run_geomind_new_task_probe.sh square_completion --task-version skill
bash scripts/run_geomind_new_task_probe.sh place_directional square_completion --strict
bash scripts/run_geomind_new_task_probe.sh reindex_extreme perfect_midpoint
```

## Task-Specific Debug Wrappers

显式几何任务现在有三个便于手工排查的 wrapper：

- `bash scripts/debug_perfect_midpoint.sh`
- `bash scripts/debug_isometric_extension.sh`
- `bash scripts/debug_point_symmetry.sh`

`debug_isometric_extension.sh` 内置了几组常用场景预设：

- `PRESET=standard`：`scene 65 / geometric_inference_isometric_extension_line_2`，只用 runtime-injected 方块的标准版
- `PRESET=grid`：`scene 10 / geometric_inference_extension_grid_2_2_top_row`，只用 runtime-injected 方块并带下排干扰块
- `PRESET=stress`：同 `scene 10` 方块场景，但默认容差更紧

`debug_point_symmetry.sh` 内置两组常用场景预设：

- `PRESET=standard`：`scene 65 / geometric_inference_point_symmetry_line_2`，红块为唯一对称中心、蓝块为参考点，镜像落点无遮挡
- `PRESET=stress`：`scene 10 / geometric_inference_point_symmetry_tight_obstacle`，镜像落点右侧紧邻静态障碍方块，并带一个额外干扰方块
