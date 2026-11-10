# GeoMind 运行脚本

当前配置采用 v3：纯任务定义在 `configs/task_definitions/`，原生实例在 `configs/episodes/`。
场景、布局和记忆协议独立。`configs/tasks/` 与 `configs/vla_tasks/` 为保留旧 CLI 路径的兼容实例目录，
内容已改为精简 v3；下面的 runtime 示例描述适配器内部的旧执行视图。
完整字段、生成器与 GT 读取路径见 [v3 契约](../../docs/episode_contract_v3_zh.md)。

最新合并、参数、variant 与历史任务暂缓状态见 [任务迁移状态](../../docs/task_evolution_zh.md)。
最终 27 个 Canonical Task 的 Family 分组见 [分类清单](../../docs/canonical_task_families_zh.md)。

## 任务名称迁移

任务定义使用独立字段，`task_id` 只保留任务名称：

```json
{
  "task_id": "SelectByDirection",
  "family": "RelationalGrounding",
  "condition": "Base",
  "layout": "Cross",
  "version": 0
}
```

`condition` 可为 `Base`、`Semantic`、`Visual`、`Execution`。`layout` 是几何布局类别，具体布局实例仍由 `runtime.annotation_layout_key` 选择；SelectByDirection 的现有物体摆放是十字形，因此标记为 Cross。

场景、物理配置、指令、GT/oracle 分发与 Standard/Vla 接口放在 `runtime`。顶层 `family` 是任务分类；`runtime.protocol_family` 是执行协议族，二者职责不同。`runtime.registration_id` 保留唯一的兼容运行键，例如 `SelectByDirection_v0`。

裸任务名默认选择 Standard 接口、可用的 Base 条件和最新版本；多个布局仍匹配时会报歧义，需显式选择。其他条件用独立参数选择：

```bash
TASK_ID=SelectByDirection HEADLESS=0 bash geomind/scripts/run_task_isaaclab.sh --condition Base --layout Cross --version 0
```

旧的版本化 ID 和类别前缀 ID 保留为 registry 别名。新增任务脚手架和 VLA 生成器均输出同一结构；VLA 任务复用相同的裸 task_id，通过 runtime.interface 区分。

| 原任务名 | 新任务名 |
|---|---|
| `AnchorDirection` | `SelectByDirection` |
| `SelectExclusionUnique` | `SelectAfterExclusion` |
| `IsometricExtension` | `ExtendEqualSpacing` |
| `PerfectMidpoint` | `InferMidpoint` |
| `PointSymmetry` | `InferPointSymmetry` |
| `SquareCompletion` | `CompleteSquare` |
| `TrendContinuation` | `ContinueSpatialPattern` |
| `GridRowcolGrounding` | `SelectByGridIndex` |
| `OrdinalInsertStatic` | `SelectByOrder` |
| `ReindexSubset` | `SelectWithinSubset` |
| `TrackSingle` | `TrackMovingObject` |
| `TrackMulti` | `TrackObjectSwap` |

例如 `AnchorDirection_v0` 和 `BindingBaseBase_AnchorDirection_v0` 均可兼容解析到 SelectByDirection 的基础配置。

## 注册任务表（按 Base / Aug 分类）

下表列出独立任务字段与兼容运行 ID；Base / Aug 分组保留原有 runtime.split 分类，具体实验条件以 condition 字段为准。

### Base 任务

| Task ID | 兼容运行 ID | condition | layout | variant / parameters |
|---|---|---|---|---|
| `CompleteSquare` | `CompleteSquare_v0` | Base | Square | — |
| `ContinueSpatialPattern` | `ContinueSpatialPattern_v0` | Base | Line | — |
| `ExtendEqualSpacing` | `ExtendEqualSpacing_v0` | Base | Line | — |
| `InferMidpoint` | `InferMidpoint_v0` | Base | Line | — |
| `InferPointSymmetry` | `InferPointSymmetry_v0` | Base | Line | — |
| `PlaceByTopology` | `PlaceByTopologyInside_v0` | Base | Containment | Inside |
| `PlaceByTopology` | `PlaceByTopologyOnTop_v0` | Base | Stack | OnTop |
| `SelectAfterExclusion` | `SelectAfterExclusion_v0` | Base | Line | — |
| `SelectByAdjacency` | `SelectByAdjacency_v0` | Base | Line | {"relation":"Predecessor","reference_object_id":"red_anchor_block","structure":"DirectedChain"} |
| `SelectByDirection` | `SelectByDirection_v0` | Base | Cross | — |
| `SelectByGridIndex` | `SelectByGridIndex_v0` | Base | Grid | — |
| `SelectByOrder` | `SelectByOrderFirstSem_v0` | Semantic | Line | {"index":1,"from":"Left","reference_time":"Current"} |
| `SelectByOrder` | `SelectByOrderFirstVisual_v0` | Visual | Line | {"index":1,"from":"Left","reference_time":"Current"} |
| `SelectByOrder` | `SelectByOrderFirst_v0` | Base | Line | {"index":1,"from":"Left","reference_time":"Current"} |
| `SelectByOrder` | `SelectByOrderLast_v0` | Base | Line | {"index":1,"from":"Right","reference_time":"Current"} |
| `SelectByOrder` | `SelectByOrderSecond_v0` | Base | Line | {"index":2,"from":"Left","reference_time":"Current"} |
| `SelectWithVisualCue` | `SelectWithVisualCue_v0` | Base | Cross | — |
| `SelectWithinSubset` | `SelectWithinSubset_v0` | Base | Line | — |
| `TrackMovingObject` | `TrackMovingObjectSem_v0` | Semantic | Line | — |
| `TrackMovingObject` | `TrackMovingObjectVisual_v0` | Visual | Line | — |
| `TrackMovingObject` | `TrackMovingObject_v0` | Base | Line | — |
| `TrackObjectSwap` | `TrackObjectSwap_v0` | Base | Line | — |
| `UpdateObjectOrder` | `UpdateObjectOrder_v0` | Base | Line | {"index":2,"from":"Left","reference_time":"Current","history_mode":"Deferred"} |

### Aug 任务

| Task ID | 兼容运行 ID | condition | layout | variant / parameters |
|---|---|---|---|---|
| `SelectByDistance` | `SelectByDistance_v0` | Semantic | Scatter | — |

## 1) 单任务（无 policy 字段）

```bash
bash geomind/scripts/run_task.sh
```

常用覆盖：

```bash
TASK_ID=SelectByDirection \
MAX_STEPS=50 \
BACKEND=isaaclab \
INTERACTIVE=0 \
bash geomind/scripts/run_task.sh
```

说明：
- `run_task.sh` 只保留是否启用交互（`INTERACTIVE`），不再显式指定 policy。
- 交互模式用于接受 observation meta 中的 `policy_action`（若存在）。

## 2) Env-Policy 交互（单任务）

```bash
bash geomind/scripts/run_env_policy.sh
```

可选 policy：
- `oracle`（在 `BACKEND=isaaclab` 下会自动走协议级 oracle 分发；非 isaaclab 保持轻量 oracle）
- `oracle_protocol`（IsaacLab 下按任务族分发到协议级脚本化 oracle：spatial_placement / anchor_reference / constraint_selection / state_reindexing / historical_identity_tracking / geometric-structure 族）
- `random_gt`（在 `BACKEND=isaaclab` 下走协议级动作原语；`spatial_placement` 会随机采样合法 GT 放置目标，轨迹自然且保留随机 GT 特性）
- `random`（小噪声随机，通常触发最大步长）
- `agent`（预留接口）
- `model`（预留给 VLA 等模型）

远程模型接入请看：[Remote Policy 接入说明](../docs/remote_policy.md)。当前主链路是 `POLICY=model` + `GEOMIND_MODEL_POLICY_HOOK=geomind_client:policy`，client hook 内部再请求远程服务。

远程模型 + IsaacLab + 真实相机图像的快速测试脚本：

```bash
bash geomind/scripts/run_remote_policy_isaaclab.sh
```

默认输出 `result.json`、`trace.json`、`run.log` 和 `wrist_cam.gif` 到 `geomind/outputs/remote_policy/<run>/`。

`oracle` / `oracle_protocol` 在 `spatial_placement` 协议下的终止行为（用于避免收尾阶段被中断）：
- 默认会在协议脚本中禁用环境 `success` 终止项，确保 release/settle 检查跑完后再由脚本判定成功。
- 默认将 episode 时长上限提升到 `120s`，降低 timeout 打断概率。
- 可通过环境变量覆盖协议时长：`GEOMIND_PROTOCOL_EPISODE_LENGTH_S=180`（由 `run_suite` 透传给协议 runner）。
- `run_suite` 触发协议 runner 时会默认设置 `VIEWER_IDLE_AFTER_AGENT=0`，避免 agent 回合结束后卡在 viewer idle。

示例：

```bash
TASK_ID_ENV=SelectByDirection \
POLICY=oracle \
MAX_STEPS=100 \
BACKEND=isaaclab \
bash geomind/scripts/run_env_policy.sh
```

## 3) 数据收集（默认 oracle + 小扰动）

```bash
bash geomind/scripts/collect_data.sh
```

示例：

```bash
TASK_ID_ENV=SelectByDirection \
EPISODES=5 \
ACTION_NOISE_SCALE=0.05 \
OBJECT_PERTURB_RANGE=0.02 \
bash geomind/scripts/collect_data.sh
```

说明：当 `BACKEND=isaaclab` 时，脚本会尝试自动激活 `CONDA_ENV_NAME`（默认 `isaac_curobo`），并在启动前检查 `python -c "import isaacsim"`。

可视化渲染开关（仅 `BACKEND=isaaclab` 生效）：
- `ISAAC_RENDER=0`：headless（不弹窗，适合批量）
- `ISAAC_RENDER=1`：打开 Isaac 渲染窗口（便于观察执行阶段和异常行为）

示例（开启可视化）：

```bash
TASK_ID_ENV=SelectByDirection \
BACKEND=isaaclab \
ISAAC_RENDER=1 \
EPISODES=1 \
bash geomind/scripts/collect_data.sh
```

在真实 IsaacLab 后端中启用真实目标物体 pose 扰动（默认关闭）：

```bash
TASK_ID_ENV=SelectByDirection \
BACKEND=isaaclab \
NATIVE_OBJECT_POSE_PERTURB_RANGE=0.01 \
bash geomind/scripts/collect_data.sh
```

## 4) 批量任务（基于 Env-Policy）

```bash
bash geomind/scripts/run_suite.sh
```

说明：
- 默认每任务 `EPISODES_PER_TASK=5`
- 默认 `SAVE_VIDEO=1`（保存轨迹可视化 gif）
- `ISAAC_RENDER=0/1` 控制 IsaacLab 是否弹窗渲染：
  - `0`（默认）：headless 模式，适合长跑和批量执行
  - `1`：显示场景和机械臂动作，便于观察中间过程与定位问题

示例：

```bash
TASK_IDS="SelectByDirection SelectByDistance_v0" \
POLICY=random_gt \
EPISODES_PER_TASK=5 \
SAVE_VIDEO=1 \
bash geomind/scripts/run_suite.sh
```

开启 Isaac 可视化渲染（用于排查中间执行问题）：

```bash
TASK_IDS="SelectByDirection" \
POLICY=oracle \
EPISODES_PER_TASK=1 \
SAVE_VIDEO=0 \
ISAAC_RENDER=1 \
bash geomind/scripts/run_suite.sh
```

## 5) 单任务（IsaacLab 真实仿真后端）

```bash
bash geomind/scripts/run_task_isaaclab.sh
```

常用覆盖：

```bash
TASK_ID=SelectByDirection \
MAX_STEPS=80 \
HEADLESS=1 \
DISABLE_FABRIC=0 \
INTERACTIVE=0 \
bash geomind/scripts/run_task_isaaclab.sh
```

## 6) 轨迹 Replay（可视化）

```bash
TRACE_JSON=geomind/outputs/env_policy/<run>/trace.json \
bash geomind/scripts/replay_trajectory.sh
```
