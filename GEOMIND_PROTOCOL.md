# GeoMind 工程协议
## 面向当前代码主线的低侵入 MVP 协议（v0.1）

本文档定义了在当前仓库结构下，GeoMind 的最小可执行协议。

当前仓库已经具备一条有效的运行时主链：

scene USD
→ scene profile
→ task spec
→ env cfg
→ Franka cfg
→ visuomotor / mimic env

因此，GeoMind 协议应当**附着在现有主链上**，而不是另外引入一套平行框架。

当前 MVP 仅聚焦两个模块：
- `post_action_relocation`（动作后重定向）
- `geometric_inference`（几何推断）

设计目标有两个：
1. **科学清晰性**：每个模块都明确给出任务语义、GT 生成、readout、对照和混淆源。
2. **工程低侵入性**：协议逻辑尽量注入到现有 `task_spec`、`env_cfg/runtime` 和 `mimic/env` 路径中，避免大规模重构。

---

## 0. 当前代码主线下的接入原则

### 0.1 协议语义应放在哪里
在当前代码结构下，协议信息建议按以下层次分布：

- **task_spec 层**
  - 任务语义
  - 模块名 / track 名
  - query family
  - query 模板
  - intervention 规格
  - GT 类型 / rule 类型
  - control mode

- **env runtime 层**
  - 运行时元数据
  - UUID 分配
  - 初始排序
  - 干预后状态
  - GT 解析
  - confound 日志

- **cfg 层**
  - 阈值
  - 开关
  - control 设置
  - intervention 默认参数

### 0.2 不应该做什么
协议**不应该**要求：

- 新建一整套环境框架
- 重写当前 env/task 注册系统
- 在现有 `scene profile -> env cfg` 之外再搞一套独立场景管线

更推荐的做法是：

- 扩展当前 task spec 载荷
- 在现有 env runtime 外围加 helper
- 复用 Franka IK / camera / Mimic 现有 plumbing
- 复用现有 extras / metrics 日志出口

---

## 1. 全局约定

### 1.1 工作空间与几何约定
- 所有协议评测都在**平面桌面工作空间**上进行
- 除非另有说明：
  - 几何推理只在 **XY 平面**进行
  - 放置高度 `z` 由桌面合法放置逻辑固定
  - 物体朝向 **不纳入 MVP 评测**
- 所有距离单位统一为 **米**

### 1.2 实体身份约定
- 每个生成出来的物体都必须绑定一个不可变的 `uuid`
- 所有离散 GT 都必须基于 `uuid` 解析，不能基于外观
- 历史类 query 无论物体后来怎么移动，都必须指向同一个 `uuid`

### 1.3 Episode 有效性约定
如果发生以下情况，episode 应标记为 **invalid**：
- intervention 执行失败
- 或非目标物体的副作用超出模块定义阈值
- 或任务特定的有效性检查失败

### 1.4 评测输出约定
每个 evaluator 都应该通过当前已有运行时日志通道（如 `extras`、`metrics` 或 task-local result dict）输出结构化结果：

- `success`
- `valid_episode`
- `gt`
- `prediction`
- `metrics`
- `confounds`

---

## 2. task_spec 扩展原则

协议应首先落在 `task_spec` 上。

### 2.1 task_spec 中建议支持的最小协议字段
每个支持协议的 task spec 至少应能承载：

- `module_name`
- `track_name`
- `query_family`
- `query_text`
- `intervention`
- `gt_spec`
- `control_mode`

这些字段对于旧任务可以缺省。  
若缺省，则保持当前 fallback 任务行为不变。

### 2.2 只应在 runtime 中保存的元数据
以下内容**不建议**静态写入 task_spec，因为它们是 episode 级别动态生成的：

- `initial_rank_to_uuid`
- `post_intervention_positions`
- `gt_uuid`
- `gt_pos`
- confound flags

这些应该在运行时生成和存储。

---

# 模块：post_action_relocation（动作后重定向）

## 3. 模块定义
**模块名**：`post_action_relocation`  
**任务类型**：`discrete_selection`  
**核心能力**：动作后的内部状态更新  
**核心科学目标**：剥离静态单帧识别捷径，测试智能体在物理干预后是否更新了内部世界模型。

---

## 4. 低侵入代码接入建议

### 4.1 推荐挂接点
本模块建议通过以下位置接入现有代码：

- `task_specs.py`
  - 保存协议语义：
    - 模块名
    - track 名
    - query family
    - intervention
    - control mode

- `geomind_env_cfg.py`
  - 将 task spec 中的协议信息带入运行时装配

- `geomind_franka_ik_rel_mimic_env.py`
  - 作为 scripted intervention 的第一落点
  - 执行运行时状态变换
  - 计算最终选中物体并记录 confound

### 4.2 复用优先原则
优先复用：
- 现有桌面场景生成
- 当前 rigid object 生成流程
- 现有 Franka IK 动作流
- 当前 picked object / lifted object 检测逻辑

仅新增：
- UUID 绑定
- 初始排序元数据
- intervention helper
- query resolver
- 基于 UUID 的 evaluator
- confound logger

---

## 5. 共享协议结构

### 5.1 状态假设
- 布局类型：`1d_linear`
- 物体数量：`N >= 3`
- 物体按严格递增的 `x` 坐标初始化
- 相邻物体满足：
  - `x[i+1] - x[i] >= d_min`
  - 且 `d_min > object_width`
- 初始排序由 `x` 升序定义

### 5.2 运行时需保存的元数据
在 reset 时保存：
- `uuid_list`
- `initial_rank_to_uuid`
- `initial_xy_positions`

在 scripted intervention 之后保存：
- `post_intervention_xy_positions`
- `intervention_success`

这些可以保存在现有 task-local runtime state 中，不需要新建大型存储系统。

### 5.3 标准 evaluator 输出
每个 evaluation 应记录：
- `success: bool`
- `valid_episode: bool`
- `gt_uuid: str`
- `grasped_uuid: Optional[str]`
- `query_family: str`
- `track_name: str`
- `metrics: Dict`
- `confounds: Dict`

---

## 6. Track：state_reindexing（状态重索引）

### 6.1 科学目标
测试智能体能否在场景被物理改变后，正确地重新建立空间索引关系。

### 6.2 状态定义
- 物体外观彼此可区分
- `feature_mode = distinct`
- 布局：`1d_linear`
- 初始排序唯一明确

### 6.3 干预定义
- 执行一个确定性的 scripted intervention，改变物体相对顺序
- 推荐 MVP 版 intervention：
  - 把当前最左边的物体放到当前最右边物体的右侧

实现说明：
- intervention 应以结构化数据表示，不应只写成自由文本

建议 task_spec 载荷：
- `module_name = post_action_relocation`
- `track_name = state_reindexing`
- `intervention = {"type": "scripted_reorder", "script_name": "leftmost_to_right_of_rightmost"}`

### 6.4 Query family
**Query family**：`state_reindex_query`

**标准 query 模板**
- “拿起当前最左边的方块。”
- “拿起当前最右边的方块。”

MVP 阶段先只实现 leftmost 也可以。

### 6.5 GT 生成
GT resolver 逻辑：
1. 读取干预结束后的仿真状态
2. 按 `x` 坐标对活跃物体重新排序
3. 根据最终状态谓词解析目标物体
4. 返回对应 `uuid`

### 6.6 Readout
**Readout 类型**：`discrete_id_match`

**成功判定**
- `grasped_uuid == gt_uuid`

实现说明：
- 若当前已有 picked/lifted object 检测逻辑，直接复用
- 只额外添加最后的 UUID 对比层

### 6.7 需记录的 confound
- `accidental_displacement: bool`
- `max_non_target_planar_displacement: float`
- `intervention_execution_failure: bool`

### 6.8 Episode 有效性规则
以下情况 episode 判 invalid：
- 任一非目标物体的平面位移超过 `displacement_tol`
- scripted intervention 未达到预期终态

默认：
- `displacement_tol = max(0.02, 0.2 * object_size)`

### 6.9 对照
该 track 不需要 history ablation，因为 query 本来就应当只根据最终状态可解。

---

## 7. Track：historical_identity_tracking（历史身份追踪）

### 7.1 科学目标
测试在去掉单帧视觉捷径后，智能体能否跨多个状态变化维持对象身份。

### 7.2 状态定义
- 物体外观完全同质
- `feature_mode = uniform`
- 布局：`1d_linear`
- `N >= 3`
- `N` 应为奇数，确保 middle 唯一
- 初始 middle 由 rank `(N - 1) / 2` 唯一定义

### 7.3 干预定义
- 执行一个 `K >= 2` 的确定性 scripted sequence
- 该 sequence 会制造空间混淆，但必须保持物理执行有效
- 推荐 MVP 版 sequence：
  1. 把当前中间物体放到最左侧
  2. 把当前最右侧物体放到最左侧

建议 task_spec 载荷：
- `module_name = post_action_relocation`
- `track_name = historical_identity_tracking`
- `intervention = {"type": "scripted_reorder", "script_name": "middle_then_rightmost_to_leftmost"}`

### 7.4 Query family
**Query family**：`historical_identity_query`

**标准 query 模板**
- “拿起最初位于中间的那个方块。”

MVP 阶段先只实现这一条即可。

### 7.5 GT 生成
GT resolver 逻辑：
1. 读取 reset 时保存的 `initial_rank_to_uuid`
2. 根据初始排序解析历史 query
3. 返回对应的 UUID

### 7.6 Readout
**Readout 类型**：`discrete_id_match`

**成功判定**
- `grasped_uuid == gt_uuid`

### 7.7 对照组
**Control 名称**：`final_frame_only`

**定义**
- 智能体只接收 intervention 后的最终观测
- 不提供动作历史和中间帧

**科学预期**
- 在理想同质设置下，性能应接近随机
- 若显著高于随机，应审计泄露线索

### 7.8 推荐的低成本泄露审计
MVP 阶段至少记录：
- `object_yaw_variance`
- `inter_object_spacing_variance`
- `feature_leakage_detected`

这样可以避免过早搭建复杂的 leakage 分析框架。

### 7.9 需记录的 confound
- `accidental_displacement: bool`
- `max_non_target_planar_displacement: float`
- `feature_leakage_detected: bool`
- `intervention_execution_failure: bool`

### 7.10 Episode 有效性规则
以下情况 episode 判 invalid：
- 任一非目标物体平面位移超过 `displacement_tol`
- intervention script 执行失败
- 在 uniform-object 设置下检测到严重持续泄露

默认：
- `displacement_tol = max(0.02, 0.2 * object_size)`

---

# 模块：geometric_inference（几何推断）

## 8. 模块定义
**模块名**：`geometric_inference`  
**任务类型**：`continuous_placement`  
**核心能力**：受几何结构约束的平面目标位推断  
**核心科学目标**：在固定目标身份与抓取获取阶段的前提下，评测连续空间中的目标位置推断能力。

**总原则**  
该模块只评测受控几何结构下的平面目标位推断，不评测目标身份选择，也不评测抓取获取。

---

## 9. 低侵入代码接入建议

### 9.1 推荐挂接点
本模块建议通过以下位置接入：

- `task_specs.py`
  - 保存：
    - 模块名
    - track 名
    - query family
    - query text
    - gt spec / rule type
    - control mode

- `geomind_env_cfg.py`
  - 把协议任务语义带入运行时装配

- `geomind_franka_ik_rel_mimic_env_cfg.py`
  - 可作为从 protocol-aware task spec 自动生成简单 place-style 子任务的入口

- runtime env / evaluator helper
  - 计算 GT 位置
  - 计算 dual-tier readout
  - 记录 placement confounds

### 9.2 复用优先原则
优先复用：
- 当前 scene construction
- 现有 anchor/object spawn utilities
- 当前 tabletop placement logic
- 当前 end-effector target command flow
- 当前 object pose extraction

仅新增：
- GT 几何求解器
- dual-tier evaluator
- 后续可选的 control 变体

---

## 10. 共享协议结构

### 10.1 工作空间与几何假设
- 工作空间：平面桌面
- 几何推理只在 XY 平面进行
- 目标物体在 episode 开始时已预先 attach 到 gripper
- 放置高度 `z` 由桌面放置逻辑固定
- MVP 阶段不评测 orientation

### 10.2 运行时需保存的元数据
在 reset 时保存：
- `anchor_positions_xy`
- `rule_type`
- `track_name`

在 GT 生成后保存：
- `gt_pos_xy`

在 episode 结束后保存：
- `pred_pos_xy`
- `l2_error`
- `orthogonal_distance`
- `rule_satisfied` 或 `trend_consistent`

### 10.3 标准 evaluator 输出
每个 evaluation 应记录：
- `success: bool`
- `valid_episode: bool`
- `gt_pos: List[float]`
- `pred_pos: List[float]`
- `track_name: str`
- `rule_type: str`
- `metrics: Dict`
- `confounds: Dict`

---

## 11. Track：explicit_rule_extrapolation（显式规则外推）

### 11.1 科学目标
测试智能体能否根据明确几何规则推断目标放置位置，并且该任务具有唯一数学 GT。

### 11.2 状态定义
- 目标物体已预先 attach 到 gripper
- anchor 数量：`M = 2`
- anchors 完全对齐、零噪声
- anchor 位置采样必须满足：
  - anchor 可达
  - GT 点可达
  - GT 区域无障碍

### 11.3 Track 参数
- `rule_type ∈ {extension, midpoint}`
- `anchor_noise_std = 0.0`
- `plane = xy`

为了低侵入 MVP：
- 先实现 `extension`
- 连续链路稳定后再加 `midpoint`

### 11.4 Query family
**Query family**：`explicit_rule_query`

**标准 query 模板**
- Extension：
  - “把方块放在 Anchor A 和 Anchor B 连线的延长线上，并保持相同间距。”
- Midpoint：
  - “把方块精确放在 Anchor A 和 Anchor B 的正中间。”

建议 task_spec 载荷：
- `module_name = geometric_inference`
- `track_name = explicit_rule_extrapolation`
- `query_family = explicit_rule_query`
- `gt_spec = {"rule_type": "extension"}`

### 11.5 GT 生成
#### rule_type: extension
设 anchor 坐标为 `P_A` 和 `P_B`：

- `P_GT = P_B + (P_B - P_A)`

#### rule_type: midpoint
设 anchor 坐标为 `P_A` 和 `P_B`：

- `P_GT = (P_A + P_B) / 2`

实现说明：
- GT 在 XY 平面中计算
- Z 使用当前桌面放置逻辑推导

### 11.6 Readout
**Readout 类型**：`dual_tier_geometry`

#### Tier 1：执行精度
- `l2_error = ||P_pred - P_GT||_2`
- 成功条件：
  - `l2_error < epsilon_pos`

默认：
- `epsilon_pos = 0.05`

#### Tier 2A：extension 的规则满足
- `orthogonal_distance = dist(P_pred, line(P_A, P_B))`
- `direction_correct = (P_pred - P_B) · (P_B - P_A) > 0`
- `rule_satisfied = (orthogonal_distance < tau_line) and direction_correct`

#### Tier 2B：midpoint 的弱规则满足
- `orthogonal_distance = dist(P_pred, line(P_A, P_B))`
- `between_score = ((P_pred - P_A) · (P_B - P_A)) / ||P_B - P_A||^2`
- `rule_satisfied = (orthogonal_distance < tau_line) and (0.0 <= between_score <= 1.0)`

说明：
- midpoint 的 Tier 2 故意设计成“弱结构一致性”，而不是“严格中点精度”

默认：
- `tau_line = 0.02`

### 11.7 对照
显式规则 MVP track 不需要结构破坏型 ablation。

可选 sanity checks：
- 交换 anchor 标签
- 镜像 anchor 布局
- 改变 anchor 间距但保持可达

### 11.8 需记录的 confound
- `ik_failure: bool`
- `ee_pose_error: float`
- `placement_contact_bounce: bool`

### 11.9 Episode 有效性规则
以下情况 episode 判 invalid：
- IK 严重失败
- GT 点不可达
- 目标放置区域被障碍占据

---

## 12. Track：implicit_trend_fitting（隐式趋势拟合）

### 12.1 科学目标
测试智能体能否在带噪线性阵列中提取主导几何趋势，并在一个明确操作化的 GT 协议下继续外推。

### 12.2 科学声明
该 track 使用的是**操作化 GT**，不是语言天然唯一 GT。  
所有隐式趋势 query 都统一按固定的 PCA-based continuation protocol 评测。

### 12.3 状态定义
- 目标物体预先 attach 到 gripper
- anchor 数量：`M >= 3`
- anchors 构成近似线性、但带噪的点列
- 噪声注入后仍需保留一个主导方向
- GT 点必须可达且无障碍

### 12.4 Track 参数
- `query_family = implicit_trend_query`
- `fit_method = pca_principal_axis`
- `noise_model = gaussian`
- `noise_std = sigma_anchor`
- `near_isometric_spacing = True`

### 12.5 Query family
**Query family**：`implicit_trend_query`

**标准 query 模板**
- “顺着这些方块的走势，把下一个位置补出来。”

建议 task_spec 载荷：
- `module_name = geometric_inference`
- `track_name = implicit_trend_fitting`
- `query_family = implicit_trend_query`
- `gt_spec = {"fit_method": "pca_principal_axis"}`

### 12.6 GT 生成
设 anchors 为 XY 平面上的点集 `A = {P_1, ..., P_M}`。

#### PCA-based operationalized GT protocol
1. 计算 anchor 中心点 `P_bar`
2. 计算协方差矩阵
3. 提取主特征向量 `u`
4. 对 anchors 做投影：
   - `s_i = (P_i - P_bar) · u`
5. 根据投影值定义两个端点：
   - `P_minproj = anchor with min(s_i)`
   - `P_maxproj = anchor with max(s_i)`
6. 定义生长方向：
   - `v_growth = P_maxproj - P_minproj`
7. 对齐主轴方向：
   - 若 `u · v_growth < 0`，则令 `u = -u`
8. 设置终端 anchor：
   - `P_term = P_maxproj`
9. 定义平均投影步长：
   - `d_bar = (max(s) - min(s)) / (M - 1)`
10. 生成 GT：
   - `P_GT = P_term + d_bar * u`

### 12.7 主方向显著性护栏
主 track episode 必须具有足够显著的主方向。

推荐判定：
- `lambda1 / lambda2 > r_min`

若不满足，则该样本不应视作合法 main-track episode。

### 12.8 Readout
**Readout 类型**：`dual_tier_geometry`

#### Tier 1：执行精度
- `l2_error = ||P_pred - P_GT||_2`
- 成功条件：
  - `l2_error < epsilon_pos`

#### Tier 2：趋势一致性
- `orthogonal_distance = dist(P_pred, PCA_axis)`
- `trend_consistent = orthogonal_distance < tau_trend`

默认：
- `epsilon_pos = 0.05`
- `tau_trend = 0.03`

### 12.9 Ablation controls

#### Control A：weak_destruction
目的：
- 作为核心 ablation，测试模型是否依赖趋势强度

定义：
- 保留整体主方向
- 保留近似端点顺序
- 增大相对于主轴的垂直噪声
- 在不让任务失去可解性的前提下削弱 trend salience

#### Control B：null_control
目的：
- 作为欠定几何 query 下的可靠性边界测试

定义：
- 彻底破坏线性趋势
- 在匹配工作空间范围内随机散落 anchors
- 不保证存在有意义的 continuation 结构

### 12.10 需记录的 confound
- `ik_failure: bool`
- `ee_pose_error: float`
- `placement_contact_bounce: bool`
- `anchor_generation_failure: bool`

### 12.11 Episode 有效性规则
以下情况 episode 判 invalid：
- anchor generation 不满足目标设置
- GT 点不可达
- 目标区域被遮挡
- 严重控制失败导致无法进行有意义评测

---

## 13. 当前代码中的推荐 Hook 点

为了尽量不打扰现有代码，协议逻辑建议挂在现有生命周期节点上。

### 推荐 hook 集
- scene construction / reset 时的 spawn logic
- task spec 的生成 / 加载
- env cfg 装配阶段
- 可选 scripted pre-action 阶段
- 状态稳定后生成 query + GT
- rollout 后评测
- 通过当前 metrics / extras 通道记录 confound

如果当前已有文件已经承担了类似职责，应优先在原处扩展，而不是新建平行子系统。

---

## 14. 推荐的最小 cfg 拆分

推荐在当前 env/task cfg 内增加轻量级 nested block，而不是新建全局大配置体系。

### Scene 相关 cfg
- object count
- anchor count
- layout type
- spacing min
- feature mode
- object size

### Protocol 相关 cfg
- module name
- track name
- query family
- rule type
- thresholds：
  - `epsilon_pos`
  - `tau_line`
  - `tau_trend`
  - `displacement_tol`
  - `r_min`

### Intervention 相关 cfg
- enabled
- scripted steps
- return-to-rest

### Control 相关 cfg
- final-frame-only
- weak-destruction
- null-control

---

## 15. 最小函数接口

这些函数不一定要做成独立环境类。  
若你当前结构更适合 helper-style，也完全可以。

- `reset_scene()`
  - 根据 scene/task protocol 生成状态

- `apply_intervention()`
  - 若开启，则执行 scripted pre-actions

- `generate_query_and_gt()`
  - 生成 query
  - 根据状态元数据或几何规则解析 GT

- `evaluate_episode()`
  - 计算 success / metrics / validity

- `log_confounds()`
  - 记录 invalidation 原因和协议级诊断信息

---

## 16. 推荐 MVP 实现顺序

1. `post_action_relocation / state_reindexing`
2. `post_action_relocation / historical_identity_tracking`
3. `geometric_inference / explicit_rule_extrapolation`
4. `geometric_inference / implicit_trend_fitting`

原因：
- `post_action_relocation` 提供最干净的离散 evaluator 路径
- `geometric_inference / explicit_rule_extrapolation` 提供最干净的连续 evaluator 路径
- `implicit_trend_fitting` 需要更复杂的生成与 control 逻辑