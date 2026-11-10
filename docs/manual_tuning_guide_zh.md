# GeoMind 手调文件指南

## 适用范围

这份文档用于指导你在现有 GeoMind 状态刷新 / 几何结构体系上做手工配置调整。

默认前提是当前 protocol 主链保持不变：

`annotation.protocol -> task_spec["protocol"] -> cfg.task_protocol`

这份指南不打算做下面这些事情：

- 不新建 schema 系统
- 不新开 engine
- 不改 solver
- 不重写 scene/profile 主链
- 不做自动调参

目标只有一个：帮助你在正确的层级、正确的文件里做最小入侵的人工调整。

## 1. 现有配置入口总览

### Task 层入口

- `geomind_lab/mimic/utils/config_semantics_adapter.py`
  作用：维护 canonical task registry、旧 `task_id` 到 canonical task 的映射，以及 task/scene/instance 三层语义补层。
- `assets/annotations/annotated_tasks.json`
  作用：当前最主要的人工可编辑任务来源。里面保存 concrete layout、task entry、instruction 和 annotation。
- `geomind_lab/tasks/manager_based/manipulation/geomind/task_specs.py`
  作用：读取 `annotated_tasks.json`，通过 `layout_key + task_index` 取 task，抽取 `annotation.protocol`，并把对象 id remap 到当前运行 scene 的对象集合。

### Scene 层入口

- `geomind_lab/tasks/manager_based/manipulation/geomind/geomind_scene_cfg.py`
  作用：物理 scene 主链。`get_scene_profile(scene_id)` 解析 USD 场景，`apply_scene(scene_id)` 负责把桌子、篮子、椅子、灯等资产装载到环境里。
- `geomind_lab/tasks/manager_based/manipulation/geomind/geomind_env_cfg.py`
  作用：把 runtime `scene_id` 绑定到环境，然后再加载 task spec / protocol。
- `geomind_lab/mimic/utils/config_semantics_adapter.py`
  作用：提供 `scene_semantics`、capability tags、risk tags 和轻量 scene metadata，主要服务于选择、汇总和分析。

### Instance 层入口

- `assets/annotations/annotated_tasks.json`
  作用：每个 layout 的 `scene_structure`、`object_positions`、instruction、annotation payload 都在这里。
- `scripts/generate_state_reindexing_matrix.py`
  作用：状态刷新批量实例生成器。主要维护状态刷新 variant 默认项，例如 `script_name`、`target_rank`、`order_axis`、`visual_mode`、`intervention_type`、`intervention_steps`。
- `scripts/generate_geometric_inference_matrix.py`
  作用：几何结构批量实例生成器。主要维护 anchor 数、spacing、noise、trend/control 参数、成功阈值和生成 query。

### Benchmark manifest 入口

- `configs/geomind_benchmark_smoke.json`
- `configs/geomind_benchmark_main.json`
- `configs/geomind_benchmark_control.json`
- `configs/geomind_benchmark_stress.json`
  作用：suite 冻结配置。决定哪些任务和 agent 一起跑。可以继续保留旧 `task_names`，也可以额外写 `canonical_task_ids`。
- `geomind_lab/mimic/utils/benchmark_manifest_utils.py`
  作用：负责 raw task 和 canonical task 的解析、展开和 runner 分发。
- `scripts/run_geomind_benchmark.sh`
  作用：读取 manifest，解析 canonical task，记录最终选中的 task descriptor 到 benchmark info。

### Summary / analysis 入口

- `scripts/summarize_state_reindexing_results.py`
  作用：状态刷新结果汇总，并把 canonical task / task-scene-instance 语义补回结果行。
- `scripts/summarize_geometric_inference_results.py`
  作用：几何结构结果汇总，并补回 canonical task / task-scene-instance 语义。
- `geomind_lab/mimic/utils/joint_summary_utils.py`
  作用：状态刷新 / 几何结构 联合 summary 的公共归一化逻辑。
- `scripts/summarize_geomind_benchmark_results.py`
  作用：联合 benchmark 的最终 summary 输出和 CSV 导出。

## 2. 推荐的手调工作流

### 总原则

建议严格按下面顺序手调：

1. 先定 canonical task
2. 再选 scene carrier
3. 再调 instance
4. 最后才看 manifest / suite 是否要更新

如果顺序反过来，task / scene / instance 三层很容易重新混在一起。

### 第一步：先定 canonical task

把 `geomind_lab/mimic/utils/config_semantics_adapter.py` 当成任务语义层的参考中心。

当前 canonical tasks 包括：

- 状态刷新:
  `state_reindexing_extreme_after_edge_shift`
  `state_reindexing_internal_insertion`
  `state_reindexing_subset_middle_with_distractors`
  `historical_identity_tracking_single_step`
  `historical_identity_tracking_multistep`
- 几何结构:
  `perfect_midpoint`
  `isometric_extension`
  `trend_continuation`
  `weak_destruction_control`

这一层适合改的内容：

- `canonical_task_id`
- task-level query intent
- task-level `diagnostic_tags`
- task-level `difficulty_axes`

推荐修改位置：

- 语义命名、alias、canonical mapping：`geomind_lab/mimic/utils/config_semantics_adapter.py`
- 单个具体任务的文字：`assets/annotations/annotated_tasks.json`
- 某个家族的默认生成规则：对应 generator 脚本

这一层不要改的内容：

- 不要把 `layout_key` 当成 task name
- 不要把 `scene_id` 当成 task 语义
- 不要为了 task 改 solver

### 第二步：再选 scene carrier

这里要先想清楚你改的是哪种 scene 相关信息：

- 只是 scene metadata
- runtime 物理 scene id
- 真正底层 scene 资产链

推荐理解方式：

- `scene_semantics` 是轻量 metadata，主要服务于人工筛选和结果分析
- runtime `scene_id` 才是实际决定加载哪个物理场景的参数
- `geomind_scene_cfg.py` 是物理 scene 主链，除非你真的要引入或修改物理承载场景，否则不要随便碰

一个非常重要的运行时事实：

`geomind_env_cfg.py` 会先按 runtime `scene_id` 加载 scene，
然后 `task_specs.py` 再按 `layout_key + task_index` 加载 task。

这意味着 annotation 里的 `scene_id` 更像是默认值和记录值，
但 runner 完全可以覆盖真正运行时加载的 scene carrier。

这一层适合改的内容：

- `scene_id`
- `scene_semantics.capability_tags`
- `scene_semantics.risk_tags`
- scene notes

推荐修改位置：

- 某个 layout 自己的 scene metadata：在 annotation layout 或 generator 输出里补 `scene_semantics`
- 共享 scene metadata 默认值：`geomind_lab/mimic/utils/config_semantics_adapter.py`
- 真正的 scene 资产承载：`geomind_lab/tasks/manager_based/manipulation/geomind/geomind_scene_cfg.py`，但只在确有必要时动

这一层不要改的内容：

- 不要为了补偿错误的 scene 选择去改 task 语义
- 不要把 canonical task id 写成 scene 属性

### 第三步：再调 instance

这一层才是 `layout`、数量、spacing、noise、intervention、具体 query 文本该待的地方。

推荐用 `assets/annotations/annotated_tasks.json` 的情况：

- 只改一个任务或少量任务
- 想保留一个手工维护 layout
- 不想碰 generator 的家族级默认行为

推荐改 generator 的情况：

- 你在调整整个任务家族
- 你想批量生成一组结构化实例
- 你需要稳定复现的一致默认项

这一层适合改的内容：

- `layout`
- `object_count` / `anchor_count` / `distractor_count`
- `visual_mode`
- `spacing`
- `axis_variant`
- `noise_level` / `sigma_parallel` / `sigma_perpendicular`
- `intervention_type` / `intervention_steps`
- `query_template`
- `success_tolerance_l2` / `success_tolerance_orthogonal`
- `difficulty_factors`

推荐修改位置：

- 单条 layout：`assets/annotations/annotated_tasks.json`
- 状态刷新家族默认行为：`scripts/generate_state_reindexing_matrix.py`
- 几何结构家族默认行为：`scripts/generate_geometric_inference_matrix.py`

这一层不要改的内容：

- 不要把 canonical task id 改成 layout 名
- 不要用 scene metadata 表达 distractor 数量或 spacing
- 不要在 layout 层问题上去改 protocol engine

### 第四步：最后才更新 benchmark manifests

只有在 suite 定义真的要变时，才去改 manifest。

典型情况：

- 给 `main` 增加一个新任务
- 在 `control` 里换掉一个控制任务
- 在 `stress` 里增加某个 canonical task 家族

manifest 用于任务集合选择，不是用来编写 task 语义的地方。

适合改的内容：

- `task_names`
- `canonical_task_ids`
- `agent_names`
- `default_agent_groups`
- `summary_preset`
- suite-level `diagnostic_tags`

不适合改的内容：

- 不要把 scene geometry 或 protocol 变化直接塞进 manifest

### 一个实用的判断表

| 你想改什么 | 优先改哪个文件 |
| --- | --- |
| 单个任务 instruction / query | `assets/annotations/annotated_tasks.json` |
| canonical alias 或 task 语义标签 | `geomind_lab/mimic/utils/config_semantics_adapter.py` |
| 单个 layout 的位置或数量 | `assets/annotations/annotated_tasks.json` |
| 状态刷新全家族 intervention 默认值 | `scripts/generate_state_reindexing_matrix.py` |
| 几何结构全家族 spacing / noise 默认值 | `scripts/generate_geometric_inference_matrix.py` |
| scene metadata tags | `geomind_lab/mimic/utils/config_semantics_adapter.py` 或 layout `scene_semantics` |
| 真正的物理 scene 资产链 | `geomind_lab/tasks/manager_based/manipulation/geomind/geomind_scene_cfg.py` |
| smoke/main/control/stress 的任务集合 | benchmark manifest JSON |

## 3. 手调字段说明表

### Task 层字段

| 字段 | 作用 | 更适合在哪一层改 | 对哪些任务最关键 |
| --- | --- | --- | --- |
| `task_id` / `canonical_task_id` | `task_id` 是历史 raw label，`canonical_task_id` 是归一化后的任务原型。 | Task | 所有状态刷新/几何结构 任务。不要把 layout 或 scene 编进 canonical task。 |
| `family` | 表示高层任务家族，例如 `state_reindexing`、`historical_identity_tracking`、`explicit_rule_extrapolation`、`implicit_trend_fitting`。 | Task | 所有任务。会影响 generator、manifest、summary。 |
| `task_type` | 表示任务类型，例如 `discrete_selection` 或 `continuous_placement`。 | Task | 状态刷新 / 几何结构 区分和 summary 分组最关键。 |
| `core_skill` | 人类可读的能力标签。 | Task | 对分析、归档、跨 suite 比较很重要。 |
| `default_query_template` | canonical task 的默认 query 模板。 | Task | 新建实例和清理命名漂移时很重要。 |
| `diagnostic_tags` | 任务诊断标签。 | Task | 所有任务。对 suite 过滤和 summary slicing 很有用。 |
| `difficulty_axes` | 说明哪些 knob 真正影响难度。 | Task | 用来指导后续应该优先调哪些 instance 字段。 |
| `canonical_role` | 一般是 `task_prototype`，用于强调它是 canonical 任务原型而不是某个实例。 | Task | 主要用于语义整理和结果行检查。 |

### Scene 层字段

| 字段 | 作用 | 更适合在哪一层改 | 对哪些任务最关键 |
| --- | --- | --- | --- |
| `scene_id` | runtime 物理 scene carrier 选择器。最终环境实际加载的是它。 | Scene | 所有任务。 |
| `scene_name` | 人类可读的 scene 名称。 | Scene | 分析和人工 review。 |
| `capability_tags` | 表示这个 scene 适合承载哪类任务。 | Scene | task-scene 匹配、suite 设计。 |
| `risk_tags` | 表示 scene 的已知风险，比如 reflection、occlusion、edge risk、camera ambiguity。 | Scene | control/stress 设计和 failure analysis。 |
| `table_shape` | 支撑平面几何描述。 | Scene | 对几何结构任务和 edge-sensitive 状态刷新任务更关键。 |
| `table_material` | 桌面材质描述。 | Scene | 对 reflection / texture 敏感任务更关键。 |
| `background_level` | 背景复杂度粗粒度标记。 | Scene | distractor-heavy 状态刷新 和 perception-sensitive 几何结构任务。 |
| `reflection_level` | 反射风险粗粒度标记。 | Scene | 几何任务、identical-object 状态刷新变体。 |
| `texture_strength` | 强纹理或方向性纹理强度。 | Scene | trend task、subset filtering task。 |
| `workspace_margin` | 可用桌面边距。 | Scene | insertion、extension、distractor filtering、shell-game。 |
| `camera_profile` | 视角特征描述。 | Scene | 分析、failure clustering、任务匹配。 |

一个关键提醒：

`scene_semantics` 目前是 metadata，不会替代 `scene_id` 去加载物理资产。

### Instance 层字段

| 字段 | 作用 | 更适合在哪一层改 | 对哪些任务最关键 |
| --- | --- | --- | --- |
| `layout` | 具体空间组织方式，比如 `linear`、`grid`、`diagonal_line`。 | Instance | 所有任务。layout 必须留在这一层，不要重新塞回 task 名。 |
| `object_count` / `anchor_count` / `distractor_count` | 实例级对象数量。 | Instance | 状态刷新 主要看 `object_count` / `distractor_count`，几何结构主要看 `anchor_count`。 |
| `visual_mode` | 物体是否 distinct / identical。 | Instance | 状态刷新 reindexing 和 identity tracking。 |
| `spacing` | 物体或 anchor 间距。 | Instance | 几何结构 midpoint、extension、trend。 |
| `axis_variant` | 轴向或行列变体，例如 `x`、`y`、`vertical`、`top_row`、`left_column`。 | Instance | 状态刷新 order-axis 任务和 几何结构 extension 变体。 |
| `noise_level` | 粗粒度 noise preset。 | Instance | 几何结构 trend 和 weak destruction。 |
| `sigma_parallel` / `sigma_perpendicular` | 更细粒度的趋势扰动大小。 | Instance | 几何结构 trend 任务最关键。 |
| `intervention_type` / `intervention_steps` | 干预的类型和步数。 | Instance | 状态刷新 reindexing 和 historical tracking。 |
| `query_template` | 具体实例的 query 文本。 | Instance | 人工编写的 layout 最常用。 |
| `success_tolerance_l2` / `success_tolerance_orthogonal` | placement 成功阈值。 | Instance | 几何结构 continuous placement 任务。 |
| `difficulty_factors` | 当前实例难度因子的快照。 | Instance | 所有任务，summary 和后处理很有用。 |

## 4. 每个 canonical task 的手调重点

下面的建议综合了当前 canonical scene capability tags、generator 默认行为和现有 query 语义。

### `state_reindexing_extreme_after_edge_shift`

- 推荐 scene capability tags：`linear_friendly`
- 推荐场景特征：桌面干净、左右顺序清晰、无遮挡、极值位置容易判别
- 推荐 instance 布局：`linear`；3 到 5 个 target；可 distinct 也可 identical
- 不建议的场景/布局：grid 式歧义布局、边缘太紧、过重 distractor clutter
- 代表性 query：`Pick up the current leftmost object.`
- 最值得优先动的 3 个字段：
  - `intervention_type`
  - `intervention_steps`
  - `visual_mode`

### `state_reindexing_internal_insertion`

- 推荐 scene capability tags：`linear_friendly`、`insertion_friendly`
- 推荐场景特征：有足够桌面边距支持插入动作，排序轴清晰
- 推荐 instance 布局：`linear`；3 到 5 个 target；内部槽位清楚
- 不建议的场景/布局：窄桌面、强视觉杂乱、插入后顺序不清晰的布局
- 代表性 query：`Pick up the object that is currently second from the start.`
- 最值得优先动的 3 个字段：
  - `object_count`
  - `intervention_type`
  - `axis_variant`

### `state_reindexing_subset_middle_with_distractors`

- 推荐 scene capability tags：`linear_friendly`、`subset_filtering_friendly`
- 推荐场景特征：target subset 与 distractor 有明显分离、桌面边距足够、防止意外重叠
- 推荐 instance 布局：`linear`、`front_targets_back_distractors`，或其他能保留清晰 target subset 的布局
- 不建议的场景/布局：没有可靠 target cue 的 identical-object scene、过于拥挤的 workspace、target 和 distractor 过度交错
- 代表性 query：`Among the marked target objects, pick up the current leftmost one.`
- 最值得优先动的 3 个字段：
  - `distractor_count`
  - `visual_mode`
  - `layout`

### `historical_identity_tracking_single_step`

- 推荐 scene capability tags：`identity_tracking_friendly`、`linear_friendly`
- 推荐场景特征：顺序轴清楚、低遮挡、有足够空间完成一次明显移动、前后帧对比直观
- 推荐 instance 布局：`linear`；3 个 target；一步 scripted move
- 不建议的场景/布局：强 camera ambiguity、过重 distractor、一步任务却做成多步 shell-game
- 代表性 query：`Pick the object that was originally in the middle.`
- 最值得优先动的 3 个字段：
  - `intervention_steps`
  - `axis_variant`
  - `visual_mode`

### `historical_identity_tracking_multistep`

- 推荐 scene capability tags：`identity_tracking_friendly`
- 推荐场景特征：中间状态变化清楚、多次 relocation 有足够桌面边距、低遮挡
- 推荐 instance 布局：`linear`；5 个 target；多步 relocation
- 不建议的场景/布局：中间移动贴近篮子或家具边界、final-frame shortcut 太容易的布局
- 代表性 query：`Pick the object that was originally leftmost.`
- 最值得优先动的 3 个字段：
  - `intervention_steps`
  - `object_count`
  - `layout`

### `perfect_midpoint`

- 推荐 scene capability tags：`geometry_friendly`
- 推荐场景特征：视角稳定、低反射、背景简单、anchor pair 干净清楚
- 推荐 instance 布局：runtime-injected `blocks`；2 个彩色 anchor + 1 个 target；zero 或极低 noise
- 不建议的场景/布局：继续复用 scene 原生可抓物体、强透视歧义、anchor 附近强纹理、无意义的 distractor clutter
- 代表性 query：`Put the grasped object exactly at the midpoint between anchor A and anchor B.`
- 当前仓库里更适合直接 debug 的实例：
  - baseline：`scene_id=65`，`layout_key=geometric_inference_perfect_midpoint_line_2`
  - carry-over 变体：`scene_id=10`，`layout_key=geometric_inference_midpoint_grid_2_2_top_row`
- 推荐 debug 入口：
  - `bash scripts/debug_perfect_midpoint.sh`
  - `PRESET=grid_top_row HEADLESS=1 bash scripts/debug_perfect_midpoint.sh`
  - `bash scripts/geometric_structure_runners/run_geometric_inference_perfect_midpoint_primary.sh`
- 最值得优先动的 3 个字段：
  - `spacing`
  - `scene_id`
  - `success_tolerance_l2`

### `isometric_extension`

- 推荐 scene capability tags：`geometry_friendly`、`extension_friendly`
- 推荐场景特征：延长线方向清楚、anchor B 外侧有足够空间、低 edge risk
- 推荐 instance 布局：`linear`、`vertical`、`top_row`、`left_column`
- 不建议的场景/布局：延长结果落在桌边附近、行列方向不清晰的 grid 视图
- 代表性 query：`Put the object on the extension of the line from A to B with the same spacing.`
- 最值得优先动的 3 个字段：
  - `spacing`
  - `axis_variant`
  - `scene_id`

### `trend_continuation`

- 推荐 scene capability tags：`geometry_friendly`、`trend_friendly`
- 推荐场景特征：anchor 数量足够、低反射、相机稳定、趋势前方有可用空间
- 推荐 instance 布局：`linear` 或轻度 noisy line，3 到 5 个 anchor
- 不建议的场景/布局：grid-like 布局、过大的 perpendicular noise、workspace 太紧
- 代表性 query：`Continue the geometric trend by placing the grasped object at the next position.`
- 最值得优先动的 3 个字段：
  - `anchor_count`
  - `noise_level`
  - `sigma_perpendicular`

### `weak_destruction_control`

- 推荐 scene capability tags：`geometry_friendly`、`trend_friendly`
- 推荐场景特征：和 `trend_continuation` 基本一致，但要保证结构被轻度破坏后仍然可分析
- 推荐 instance 布局：带显式扰动的 noisy line，并有足够 anchor 展示结构退化
- 不建议的场景/布局：本身就已经很模糊的 scene，再叠加 weak destruction
- 代表性 query：`Continue the latent trend despite weak structural corruption.`
- 最值得优先动的 3 个字段：
  - `noise_level`
  - `sigma_perpendicular`
  - `anchor_count`

## 5. 手调模板文件建议

### 推荐形式

推荐采用：

- 一份 Markdown 指南
- 一份 JSON 草稿模板

原因：

- 现有仓库主要就是 JSON 驱动，annotation 和 manifest 都是 JSON
- JSON 模板和 generator / summary 输出结构更接近
- 便于 diff 和版本管理
- 不会给运行时引入新的依赖

当前模板文件：

- `configs/manual_task_scene_instance_template.json`

注意：

- 这个模板只是人工参考，不参与运行时加载
- 它不是新的 source of truth
- 它只是帮助你先想清楚 task / scene / instance 三层要怎么调

### 推荐使用方式

1. 先复制模板到你自己的 scratch 文件。
2. 先填 canonical task、scene metadata、instance knobs。
3. 再把最终值回写到真正会被运行时读取的文件：
   - annotation 文件
   - generator 脚本
   - manifest 文件
4. 最后跑现有 generator / summary 做校验。

## 6. 手调检查清单

每次手调后建议过一遍这份 checklist：

- `canonical_task_id` 仍然能在 `config_semantics_adapter.py` 中映射成功
- raw `task_id` alias 没有被误删
- `layout` 仍然明确属于 instance 层
- scene capability tags 和目标任务匹配
- scene risk tags 仍然诚实描述 scene 风险
- 你选的 `scene_id` 和 canonical task 语义仍然匹配
- 如果 runner 或 manifest 会 override `scene_id`，你已经显式考虑到了
- `annotation.protocol` 结构仍然合法
- query 文本和 canonical task 语义仍然一致
- 状态刷新 的 intervention 字段仍符合预期 scripted 行为
- 几何结构 的 spacing / noise 字段仍符合预期难度
- generator 生成结果仍然可解析
- summary 仍然能透传 `canonical_task_id`、`task_semantics`、`scene_semantics`、`instance_semantics`
- benchmark manifest 只有在 suite 集合变化时才需要同步更新
- main / control / stress 的改动只影响任务集合，不影响底层逻辑

## 7. 不建议你做的改动

除非你准备开启一个更大的迁移项目，否则不建议做下面这些事情：

- 不要把 `layout` 再塞回 canonical task 名
- 不要让某一个 scene 成为某类任务的语义宿主
- 不要在已有字段足够表达的情况下去改 solver
- 不要改 protocol 主链
- 不要把 manifest 当成 task 语义配置文件
- 不要把 task / scene / instance 三层重新揉回一个命名体系
- 不要为了 canonical 语义一次性全量迁移旧 annotation
- 不要删除现有日志、manifest、脚本仍在引用的 raw `task_id`
- 不要为了 scene metadata 去改 `geomind_scene_cfg.py`
- 不要把 `scene_semantics` 当成物理 scene loader
- 不要把 `canonical_task_id` 写成 scene id、layout family 或难度标签

## 8. 最安全的起步点

如果你想从最稳妥的地方开始，建议按下面优先级：

- 单个任务 / 单个 layout 微调：
  `assets/annotations/annotated_tasks.json`
- 状态刷新家族级调整：
  `scripts/generate_state_reindexing_matrix.py`
- 几何结构家族级调整：
  `scripts/generate_geometric_inference_matrix.py`
- canonical 语义整理：
  `geomind_lab/mimic/utils/config_semantics_adapter.py`
- 只改 suite 选题：
  benchmark manifest JSON

这条分工边界能最大程度避免 task / scene / instance 再次耦合在一起。
