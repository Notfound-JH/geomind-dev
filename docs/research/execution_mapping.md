# 任务类型、执行机制与设计到代码映射

核验日期：2026-11-07。这里的类型按“答案依赖哪些时刻”划分，不等于公共 family。注册存在、求解器存在、运行通过、模型能力成立是四个不同状态。

## 执行机制矩阵

| 类型/代表 | 状态如何到达查询 | 现有模块与入口 | 当前支持与证据边界 |
|---|---|---|---|
| 静态查询：ProjectOntoLine | reset即query/step0；不需要事件或历史 | [resolver.py](../../geomind/episodes/resolver.py) `EpisodeResolver.resolve/solve`；[runtime.py](../../geomind/episodes/runtime.py) `SemanticEpisode.reset/phase/observation/step/evaluate_answer`；[episode配置](../../geomind/configs/episodes/ProjectOntoLine_v0.json) | 原生CPU已运行；本轮当前配置+3对非零z诊断。静态选择 SelectAfterExclusion 则走旧 `constraint_selection_logic`，不能因都是静态查询就认为共用原生后端 |
| 更新后查询：UpdateObjectOrder | 插入/重排干预后，按当前排序选择；答案不必依赖参考历史 | [配置](../../geomind/configs/tasks/UpdateObjectOrder_v0.json) `reference_time=Current/history_mode=Deferred`；[task_evolution.py](../../geomind_lab/tasks/manager_based/manipulation/geomind/task_evolution.py) `apply_task_contract`；[state_reindexing_logic.py](../../geomind_lab/mimic/envs/state_reindexing_logic.py) `build_state_reindexing_intervention_plan/evaluate_state_reindexing_terminal_state` | 当前旧实现有更新机制，原生 `resolver.TASKS` 没有本任务；本轮只静态确认，未运行更新后的真实场景。历史语义明确暂缓，不能拿 DetectSceneChange 测试覆盖它 |
| 历史比较：DetectSceneChange | Observe→Delay隐蔽MoveObject→Respond；答案必须比较参考/终态 | [配置](../../geomind/configs/episodes/DetectSceneChange_v0.json)、[协议](../../geomind/configs/protocols/change_protocol_s0.json)；`SemanticEpisode.step`保存Observe.End，`solve`比较位移 | 原生CPU已运行；[6对报告](../../outputs/research/dsc_counterfactual_v1_20261107/reports/validation.json)为已完成证据。本轮历史回归测试通过。新增/移除定义迁移未决；未测模型历史利用 |
| 连续运动中的身份追踪：TrackMovingObject、TrackObjectSwap | 旧干预计划驱动物体逐步沿弧线/waypoint运动，末态读取初始身份/排序目标 | [TrackMovingObject配置](../../geomind/configs/tasks/TrackMovingObject_v0.json) `LegacyIsaacLab`；[task_profiles.py](../../geomind_lab/tasks/manager_based/manipulation/geomind/task_profiles.py) `track_single`；[post_action_relocation_motion.py](../../geomind_lab/mimic/envs/post_action_relocation_motion.py) `track_single_paired_arc_xy/track_multi_staggered_waypoint_xy`；[mimic环境](../../geomind_lab/mimic/envs/geomind_franka_ik_rel_mimic_env.py) `_update_state_reindexing_dynamic_motion/_refresh_state_reindexing_terminal_state/step`；`evaluate_historical_identity_tracking_terminal_state` | 有旧运动/末态判定源码，**本轮仅静态确认**，未运行PhysX。原生SemanticEpisode不是连续轨迹追踪后端；逐时刻跟踪精度/身份切换率等评估接口未在这条源码链确认，不能把连续运动+末态成功当作连续追踪能力证据 |
| 规划/占位边界 | 无可验收执行链 | [迁移说明](../task_evolution_zh.md)；`task_evolution.py` 对 TrackObjectOrigin 抛NotImplementedError；[双关系配置](../../geomind/configs/tasks/SelectByTwoRelations_v0.json)、[间接指代配置](../../geomind/configs/tasks/SelectByIndirectReference_v0.json) 为Unavailable | 来源历史、一般双关系/多跳仍缺；连续追踪行不能替这些任务补证据 |

旧 mock 的 `_build_rank_binding/_apply_intervention_positions` 可直接计算干预位置，但不是连续物理过程。其 generic fallback 更不等于任务语义。旧末端距离/dwell评分见 [simple.py:LayoutAwareEvaluator](../../geomind/tasks/evaluators/simple.py)，与原生点/ID回答评分不同。更早规划和研究意图仍参照 [GEOMIND_PROTOCOL](../../GEOMIND_PROTOCOL.md)，不把规划当完成。

## ProjectOntoLine：设计要求→实现→验收

| 设计要求 | 文件/函数 | 已有能力 | 缺口 | 验收证据 |
|---|---|---|---|---|
| [v3契约](../episode_contract_v3_zh.md) 无限三维直线投影，不截断到线段 | `resolver._validate_parameters/solve`，`geometry=InfiniteLine3D` | 正交投影、退化线拒绝 | 独立设立理论/视觉原点定位意图未确认 | `test_projection_uses_infinite_line_not_clamped_segment`；本轮±.14沿线样例超出±.1端点 |
| 静态观察即可求解 | `SemanticEpisode.reset/observation` | step0公开坐标、角色ID、frame，无历史 | 非纯视觉；角色ID省去语言绑定 | [当前配置完整观察](../../outputs/research/projection_diagnostic_20261107/reports/ProjectOntoLine_v0/result.json) |
| 连续点回答与明确精度 | `resolver.load_json`；`runtime.evaluate_answer/step` | JSON读入→必需字段/数值/frame检查→欧氏误差≤配置容差 | 不是模型自然语言解析器；无实际服务 | 每例正确、0.5τ、2τ、错点、错frame、缺字段、多字段、维度错、NaN、重复键、坏JSON，逐项保留 |
| 泛三维偏移不依赖q_z=p_z | [generate_spatial_suite.generate](../../geomind/tools/generate_spatial_suite.py)原偏移n_z=0；新[research_projection.generate/verify_case/verify](../../geomind/tools/research_projection.py)旋转法向量 | 本轮补齐3对n_z=0/非零诊断，独立构造q/残差检查 | 模型是否用捷径未测；正式生成器没有扩大分布 | [静态报告](../../outputs/research/projection_diagnostic_20261107/reports/validation.json)，B的z捷径误差>0.10m |

## DetectSceneChange：设计要求→实现→验收

| 设计要求 | 文件/函数 | 已有能力 | 缺口 | 验收证据 |
|---|---|---|---|---|
| 编码、隐藏、回答的时间边界 | `EpisodeResolver._validate_protocol`；`SemanticEpisode.phase/step` | 正持续时间、事件合法；观察末尾快照；隐藏objects=[] | Wait长度不等于长期记忆难度 | `test_memory_visibility_query_timing_and_event_application`及历史完整stream |
| 参考与查询属于同episode | `resolver._validate_snapshot/solve` | instance/frame/hash/参考步核对；reset清空历史 | 服务端缓存隔离仍需模型适配器保证 | 历史隔离/早答拒绝测试；不是远程服务验证 |
| 唯一超过位移阈值 | `solve` DetectSceneChange分支 | 恰一Δ>τ，所有|Δ−τ|>ε；对象ID回答 | 不支持早期增删事件 | [定义差异登记](decisions/detect_scene_change_semantics.md)、[历史失败样例](../../outputs/research/dsc_counterfactual_v1_20261107/reports/failure_cases/summary.json) |
| 最终状态不能唯一决定标签 | [research_pairs.generate/independent_case/validate_pair](../../geomind/tools/research_pairs.py) | 同终态、不同初态、不同GT；独立位移oracle与生产GT核对 | 没有模型full-history/final-only成绩 | 6对/12正答/24错答；固定规则不变量与真实模型分开 |

## 两者共享与分歧

共享：v3引用配置→resolver契约/快照→SemanticEpisode→同一公开observation形状→`step(answer)`→`evaluate_answer`→provenance。无需另建环境。数据导出可复用 `public_observation`；复现快照复用 `research_pairs.snapshot/check_provenance`，本轮仅扩展索引文件名和额外源码路径以支持静态实验，默认历史行为不变。

差异落在三个位置：①resolver按domain决定是否允许/必须有protocol；②runtime无protocol就在reset求解，有protocol则推进阶段并保存历史；③solve分别计算点投影与跨时位移，评分按Point（坐标/frame/容差）或Object（ID）分支。连续追踪旧实现与上述原生分支并非同一执行器，不能强行套进三阶段CPU协议。

## 本次完整请求与轨迹补齐（2026-11-07）

|设计要求|复用入口|发现缺口|最小修改/验收|
|---|---|---|---|
|类别→任务→场景→布局→GT|generate_spatial_suite.generate/verify；EpisodeResolver；SemanticEpisode|缺完整请求及条件分布记录|research_pipeline.parse_request/prepare；拒绝未知组合、保存解析请求，见本次实验报告|
|明确Aug与第二场景|现有geometry构造；registered_scene.scene_description/spawn_registered_scene|旧split/condition不可直接代表新Aug|只实现平移3cm，保存配对Base并重求GT；注册65重新加载真实场景|
|静态/历史物理闭环|run_episode_physics.run_case；IsaacLabEpisode.reset/step/execute_answer|原NPZ只有关节角/TCP位置/对象状态，缺动作和姿态|附加controls NPZ（不破坏旧trace列），保存公开流与计划/实测事件索引|
|固定GT并复核|既有独立几何残差/实测位移检查与物理评分|对象目标在执行后重新读取，难严格固定查询目标|冻结查询时目标；audit_physics离线重算关节限制/查询快照/GT/末端误差/姿态dwell|
|来源恢复|research_pairs.snapshot/check_provenance|仅dirty不足；物理模块延迟加载|显式补入执行器、场景加载器及工具源码，保留配置、skill和源码内容快照|

当前真实执行范围与失败记录见 [流水线实验](experiments/2026-11-07-pipeline-validation.md)。连续追踪仍缺统一原生流/身份指标与来源历史，不能用本次位移检测补齐。
