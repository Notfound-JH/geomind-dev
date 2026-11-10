# 27 种任务展示与十项占位补齐

2026-11-10。用户要求以现有任务质量和自然指令难度补齐剩余占位任务，并交付覆盖 27 种任务的网页。正在应用 [geomind-research](../../../.agents/skills/geomind-research/SKILL.md)；不把注册存在、CPU GT、到达诊断或历史成绩记为本轮新实例抓放通过。

**最终任务证据已完成：严格展示页为 27/27 种类型、30/30 个通过实例，其中原 20 个历史实例覆盖 17 类，本轮十个新实例补齐其余十类。** 新十例均通过实际正例、有效错误命令负控、运行时 GT、独立原始状态审计、保存来源及外部/腕部媒体门槛。13 个显式旧重试版本保留，最终选择始终取明确列出的最新目录，不回退挑成功。

展示文件位于 `outputs/research/canonical27_delivery_v1`，浏览器入口为 http://127.0.0.1:8781/outputs/research/canonical27_delivery_v1/index.html 。逐例来源和全部门槛见 [summary.json](../../../outputs/research/canonical27_delivery_v1/summary.json)，完整构建命令见 [final_delivery_command.json](../../../outputs/research/canonical27_delivery_v1/final_delivery_command.json)。本地网页/媒体/HTTP 检查单独记录，浏览器交互未验证。

## 范围与既有证据

27 是 canonical 类型数，不是视频实例数。先前 [20 项 Robotiq 交付](2026-11-10-registered-robotiq-regeneration.md)包含 17 种类型：First/Last/Second 是一种顺序选择的三例，Inside/OnTop 是一种拓扑放置的两例。此批旧结果各有保存版本、场景、控制器和审计边界；最终网页可以明确引用这些已验收历史实例，不声称它们在本次最新源码上一并重跑。

本轮补齐十种类型：`SelectByBranchPosition`、`SelectByRegion`、`SelectByPathDistance`、`SelectByTwoRelations`、`SelectByIndirectReference`、`ProjectOntoLine`、`ReflectAcrossLine`、`DetectSceneChange`、`UpdateEmptySlot`、`TrackObjectOrigin`。它们采用明确命名的 Base 变体和实际抓放动作；前三维原生任务已有部分独立历史实现，不重复把它们称为从零实现，也不把旧诊断替代本轮操作。

详细自然指令、固定量、关系/历史机制、干扰项、输入范围与独立检查见[十项任务实现卡](../tasks/canonical_extensions.md)。27 个 canonical 名称及 family 保持原分类。

## 选择与完成标准

场景优先沿用用户当前的 registered scene 65，使用已有空桌面源 USD、真实网格桌面和原墙体材质；不能因改 `scene_id` 就声称场景已加载。实际机器人沿用现有 Panda＋Robotiq 抓放链。布局需要与真实桌面、对象尺寸、支撑余量和机器人可达性一起验收。物体与布局角色为评估端数据，公开 observation 继续使用中性别名，不把求解答案写进指令。

选择类自然指令必须实际抓起正确对象并持稳，放置类必须抓取、搬运、松爪、退离和放稳；不能降级为坐标问答或 `ReachAboveTarget`。GT 从机器人运行时实测状态重算并固定，时序任务使用实测 `Observe.End` 与实际事件后的状态；事件计划不作为观测或答案。

物理判据沿用原 20 项同一执行器：抓起超过 60 mm；线速度 ≤0.01 m/s、角速度 ≤0.1 rad/s；参考位移 ≤5 mm、参考姿态差 ≤0.12 rad；桌面 Point 放置默认中心误差 ≤10 mm、支撑高度误差 ≤6 mm并通过实测桌面足迹检查；至少连续 `max(60 ticks, configured dwell, 0.5 s)`，实例更严格要求优先。动作阶段只写机器人关节目标，不允许用物体瞬移、吸附或外力替代抓放。

每个 worker ≤420 秒，GPU 串行，先单例再小批量；新目录保存每次失败及修复。负控不仅要被正确 GT 拒绝，还必须真实完成所命令的错误抓起/错误放置；放置对照需要退离、支撑和稳定，动作失败不算有效反证。

## 本轮实现与定义边界

| 实现模块 | 类型 | 本轮变化 |
| --- | --- | --- |
| [extended_static_tasks.py](../../../geomind/episodes/extended_static_tasks.py) | 分支、区域、路径距离、双关系、间接指代 | 显式拓扑/区域/参照配方，实际状态求解及独立反捷径检查，可见桌面 context；完整动作仍由已有执行器承担。 |
| [extended_geometry_tasks.py](../../../geomind/episodes/extended_geometry_tasks.py) | 投影、轴对称 | 明确 `TabletopPlane` 放置变体，独立几何恒等式检查；保留原 InfiniteLine3D 含义，不把空中计算点当落点。 |
| [extended_temporal_tasks.py](../../../geomind/episodes/extended_temporal_tasks.py) | 位移变化、空格更新、来源追踪 | 真实 PD 事件计划与实测求解分离；41 个路点/4 秒，来源过程可见；独立初末状态检查和无碰撞 context 线条。 |

静态结构只允许保持语义的整体平移，不做破坏图、区域或关系裕量的逐对象抖动。纯图结构、定义明确的区域与两跳参照分别保留其操作语义，不用最近距离替代拓扑或组合条件。

新增静态红/蓝参考标记是显式固定的运动学 context，待抓候选是真实动态刚体，直接受实测桌面支撑；不会将固定 context 的保持或辅助线条当作新动态操作验证。

投影与轴对称仅新建水平桌面变体，原三维数学定义与 [ProjectOntoLine 旧研究结论](../tasks/ProjectOntoLine.md)保持；本轮不声称一般三维操作覆盖。`DetectSceneChange` 仅采用已有位移语义，[新增/移除问题](../decisions/detect_scene_change_semantics.md)保持未决。

`UpdateEmptySlot` 的当前状态足以确定空格，不宣传历史记忆必要性。`TrackObjectOrigin` 使用同外观对象和初始来源框，但当前公开结构化输入仍有稳定 ID；即便动作与合同验证通过，也不能据此声称纯图像追踪能力。没有模型调用、训练或模型表现评估。

## 已执行 CPU 检查

当前已执行以下模块级测试：

```bash
python -m unittest \
  tests.test_extended_temporal_tasks \
  tests.test_extended_static_tasks \
  tests.test_extended_geometry_tasks -q
```

结果 **26 项通过**。覆盖拓扑/区域有效性、路径与欧氏距离不一致、双关系删条件、间接参照单跳捷径、投影/反射独立几何、位移阈值与多目标拒绝、占位事件/格间/错槽拒绝，以及相同终态换来源历史导致答案改变。独立检查器不调用生产求解器生成预期，但共享 schema 和绑定，此独立性边界保留。

随后完成 **104 项注册/绑定/导出等 CPU 集成检查**，持久日志为 [cpu_integration_104.log](../../../outputs/research/canonical27_delivery_v1/cpu_integration_104.log)。针对下述空格更新的局部路点预算修复，另有 **27 项回归通过**，见 [action_budget_regression.log](../../../outputs/research/canonical27_delivery_v1/action_budget_regression.log)。这些计数对应不同测试范围，不能相加后暗示彼此独立或全部物理能力已覆盖；CPU 通过仍不能代替实际机器人动作。

首轮八项 worker 全部退出后，已应用包含原生主入口路由、有效错对象负控、canonical 方块接触高度和显式路径错误候选的合并修复，以及下述几何反捷径修复。最新 [integrated_repair_regression.log](../../../outputs/research/canonical27_delivery_v1/integrated_repair_regression.log)记录 **139 项通过**。其后的高度、预算和相机修复分别运行受影响范围回归，全部物理重试均在 worker 退出后改源、再捕获独立快照，不能把旧运行追认为新源码执行。后续 80 项高度修复、49 项预算相关和 28 项相机相关回归日志分别保存，计数存在重叠，不相加为独立覆盖。

## 实际物理结果（最终选中版本）

以下十例均使用 scene65、seed0、Panda＋Robotiq；初始落稳、运行时 GT、正例实际动作、有效负控、独立审计及媒体来源全部通过。基座 Z 均为实测桌高 `0.9328504589976385 m`。默认 XY 基座 `[-.12,.32]`，特例单列；原 20 例保留其各自 scene65/91、120/240Hz 与来源版本，不并入本轮新执行。

| 新类型 | 最终目录（outputs/research 下） | 基座 XY / 最终说明 |
| --- | --- | --- |
| SelectByBranchPosition | canonical27_branch_contact_v2 | 默认；接触高度模型修正后正负均实际持稳。 |
| SelectByRegion | canonical27_smoke_v1 | 默认；区域唯一性及真实错误候选抓起通过。 |
| SelectByPathDistance | canonical27_path_rebase_v2 | `[-.04,.32]`；仍实际执行欧氏最远错误叶节点 n1，不替换容易对象。 |
| SelectByTwoRelations | canonical27_new_v1 | 默认；每个单条件留干扰，交集唯一，正负实际抓起通过。 |
| SelectByIndirectReference | canonical27_indirect_camera_v6 | 默认；三个候选统一 40×40×50 mm，原质量/XY 关系不变；固定前侧腕相机解决前臂遮挡。 |
| ProjectOntoLine | canonical27_geometry_replay_v2 | 默认；实测非中点布局，正例落点误差 2.056 mm。 |
| ReflectAcrossLine | canonical27_reflect_rebase_v4 | `[-.04,.32]`；正例 2.011 mm；有效错误命令误差 2.507 mm。 |
| DetectSceneChange | canonical27_detect_rebase_v2 | `[-.06,.44]`；实测位移检测及正负实际抓起通过。 |
| UpdateEmptySlot | canonical27_temporal_smoke_v2 | 默认；实际转移占位后正确空槽抓放，错误落点完整执行后拒绝。 |
| TrackObjectOrigin | canonical27_new_v1 | 默认；初始来源身份与当前蓝框不同，连续可见事件及正负持握通过。 |

逐例实际请求、布局缩放、时间步、原始最大步数、基座和源码/配置哈希另见 [30 实例复现矩阵](../../../outputs/research/portable_delivery_v1/validated_reproduction_cases.json)。矩阵明确区分保存版本验收与当前代码新生成，不能用统一场景/基座或统一 6000 步替代。
### 区域选择 smoke_v1 的真实正负动作

[独立质量复核](../../../outputs/research/canonical27_smoke_v1/SelectByRegion_v0/independent_quality_review.json)从原始状态分别重算正例正确对象抬升 **143.22 mm**、负例实际错误对象抬升 **143.44 mm**。负例并非到达失败：错误对象确实被夹爪抓起，而被冻结的正确对象 GT 拒绝。正负例均无动作期间物体位姿写入、无关节状态跳写、无吸附或操作期外力搬运。

实测查询时所有方块底部相对真实桌面间隙为 **−1.283 至 +1.000 mm**，没有悬空辅助台面；固定 context 标记的 +1 mm 初始间隙与动态候选的小量接触穿透分别保留。四面米色墙、实际桌面、蓝黄对角标记与区域线框已由实际渲染帧检查。最终腕相机审计通过，保存源码及配置完整；公开输入有中性别名和结构化位置，没有把 target/GT/角色字段传入 observation。本次复核是离线重算，不声称物理重执行或浏览器播放交互已验证。

### 空格更新 v1 的失败与最小修复

`canonical27_temporal_smoke_v1/UpdateEmptySlot_v0` 的初始落稳、实际占位事件及运行时 GT 已通过，但正例任务整体仍正确判为失败。末态物体中心误差虽约 **1.094 mm**、稳定计数达 **240 ticks**，阶段 7 的高位运输路点却在旧 **1600 个 240 Hz ticks** 的局部上限处耗尽，仍有 **4.391 mm** TCP 误差，未达到原 **3 mm / 0.08 rad** 路点条件及相应驻留；阶段 8 未完整执行，不能仅凭落点误差和末态稳定把该次执行算通过。对应错误落点负控也未完整完成所命令动作，不能当作有效语义反证。

[stage7_failure_diagnosis.json](../../../outputs/research/canonical27_temporal_smoke_v1/UpdateEmptySlot_v0/stage7_failure_diagnosis.json)保存实际原始报告/轨迹/请求哈希，重算误差尾部仍在收敛；采样检查未支持关节限位卡死、奇异位形或静态弹簧平衡的解释。约需 40–50 个额外 ticks 的尾部外推只构成修复依据，不是物理成功预测。

本次最小修复只针对 `canonical_extension` 重分配局部路点额度：从参考 120 Hz 的 800 ticks 调为 1200 ticks，即 Robotiq 240 Hz 下 **1600 → 2400 ticks**。全局 **12000 ticks**、每 worker **420 秒**、机器人基座/home、seed、对象和目标、控制器、关节限位及所有路点/放置/稳定判据不变。27 项动作预算回归已通过；随后按同 seed 在 v2 新目录实际重试，结果见下段。原失败报告、轨迹及录像保留，不以新的规划预测覆盖旧结果。

后续实际 v2 已完成完整正负放置：正例实测目标误差 **2.406 mm**，释放退离后连续有效 **240 ticks（1 秒）**；错误落点对照距被冻结正确 GT **78.407 mm**，但距其错误命令仅 **2.200 mm**，并在错误命令位置稳定 240 ticks，故是有效错误执行而非动作失败。v1 的失败仍按失败保留，v2 通过不更改原失败判定。

### 双关系和来源追踪的实际通过

双关系实例在当前实际位置下保留“每条单独关系都留下干扰项、交集唯一”的条件必要性。正例正确对象实际抬升约 **142.389 mm**，有效错误对象负控也完成实际抬升约 **142.550 mm**；语义拒绝与被选择错误对象自身的动作有效性分别复核。

来源追踪的 [independent_quality_review.json](../../../outputs/research/canonical27_new_v1/TrackObjectOriginClutter_v0/independent_quality_review.json)从原始初始状态确认源对象为 `obj_003`，查询时蓝框内已变成另一个灰块 `obj_002`；冻结 GT 仍是初始来源对象，未读取计划终点作为 GT。事件 4 秒命令窗口含 **961 条**连续物体状态和 **121 个**可见录像帧，最大单物理 tick 位移 **0.548 mm**，事件期间候选最小中心间距 **154.16 mm**。正例源对象实际抬升 **143.200 mm**，负例实际选择另一灰块抬升 **143.236 mm**，两者均完成连续 **1 秒**持稳，严格错误抓起命令审计通过。

92 条公开观察未发现来源角色、GT 或私有 `item_*` 名称；中性稳定 ID 和精确坐标仍然可见，纯图像追踪能力未评估。实际首帧、运动中帧、查询帧和抓起末帧中的蓝框及灰块混合过程已检查可辨，所有外部/腕部媒体哈希以及保存源码/配置完整。该独立复核是只读 CPU 重算，未冒充 PhysX 重执行。

### 失败保留与最终修复

| 问题 | 原始失败与最终处理 |
| --- | --- |
| 分支持稳 | 首轮实际抓起但角速度在 0.1 rad/s 附近越界；按真实接触角而非未达到的预载角计算指垫高度，保留 4 mm 名义桌隙，下降约 0.346 mm。v2 正负均连续 240 ticks，正例原始角速度 0.0391–0.0430 rad/s。 |
| 路径无效负控 | 原负控的阶段3末误差约17.587mm，没有抓起；旧流程曾误把语义拒绝当有效对照。现有独立被选对象审计要求完整抓取、抬升与持稳。原错误叶 n1 保持，基座 +X8cm 后真实正负通过。 |
| 间接指代持稳 | 高度模型修正、保留初始闭合命令、32/32→64/64求解迭代三项尝试仍分别失败，原始角速度评分不替换成姿态有限差分；失败均保留。64/64及保留闭合命令未采用，恢复32/32和原质量/弹簧保持模型。 |
| 间接指代接触几何 | 38mm指垫有效高与4mm名义桌隙要求居中物体至少46mm高；原40mm候选产生约3–4mm接触带偏心。三个候选统一加高到50mm，宽度40mm、质量0.05kg、材料及XY语义均不变，几何及惯性变化如实记录。v5物理正负均持稳1秒，但腕视野仍失败，不能完整交付。 |
| 间接指代腕自遮挡 | v5的90度抓姿使后置相机被实际panda_link5视觉网格遮挡。真实FK误差小于0.86µm的10例/143关键帧筛查后，Robotiq仅固定localZ −.15→+.02m，X=.13/Y=0、刚性parent/光学roll不变，profile_version升2。v6重新真实正负，持物与参照可见，严格腕审计通过；不是改图或动态瞄准。 |
| 变化检测下探 | 原阶段4在39.274mm误差平台处未抓取，无硬限位证据且不能断言具体碰撞。保留对象/事件/seed/GT/阈值，实际桌面支持的基座 `[-.06,.44,z]` 正负重试通过。 |
| 反射错误运输 | `.65`布局重跑时负控stage7到2400ticks末仍4.201mm且收敛；局部cap升3000后稳定在3.3387mm，额外等待无效。保持正确GT与同一错误命令，基座+X8cm后正负完整放置、退离、支撑和稳定通过。总12000步/420秒及所有判据不变。 |

最终 [Indirect独立复核](../../../outputs/research/canonical27_indirect_camera_v6/SelectByIndirectReference_v0/independent_quality_review.json)的正/负被抓对象各连续240ticks有效；原始角速度最大0.01381/0.08325rad/s，实际外部与腕部首末帧已查看。最终 [Reflect独立复核](../../../outputs/research/canonical27_reflect_rebase_v4/ReflectAcrossLine_v0/independent_quality_review.json)确认与前次正确GT及错误命令位移均精确为0，仅机器人配置/相机改变，不能以改目标换取通过。

版本化 Standard 主任务ID已走真实 AssembledNative 路径；裸任务名、显式原生配置根和显式semantic调用保留原含义，不回退mock或ReachAboveTarget。
### 几何任务反中点捷径质量修复

首轮投影的实测线参数约 **0.500045323**，正确投影与红蓝端点中点仅相差约 **0.0131 mm**，远小于 10 mm 动作容差。两段首轮几何录像的数学 GT 和实际动作成立，但忽略绿色源点、仅做中点（或以中点作点对称）即可在该实例上成功，不能达到本轮要求的源条件有效性。这是需要修复的任务质量问题，不能仅披露后照常计入最终完成数。

修复保留红蓝参考点 `A=(.32,-.12)`、`B=(.48,.12)`、物体尺寸和源点垂距，将绿色源点沿参考轴平移 `(+.024,+.036)` 到 `P=(.3324744677,.0970170215)`。手算 `Q=A+.65(B−A)=(.424,.036)`，投影为 Q，反射为 `2Q−P=(.5155255323,-.0250170215)`。投影与端点中点相差 **43.267 mm**，反射与“以端点中点作点对称”相差 **86.533 mm**，均超过原动作容差。

配方显式加入 `minimum_projection_midpoint_separation_m=.03`。生产求解器从实测对象位置计算投影并要求距端点中点至少 30 mm；独立检查器利用标量恒等式 `|(P−M)·d| / |d|` 另算这一间距，不调用生产投影。不能将质量阈值设为 0 绕过。固定参考点、只改变绿色源点，使线参数在 `.65` 和 `.35` 两个均合法的反事实之间改变，两种答案都随源点改变，沿用第一答案会被独立检查拒绝。

这两项已应用修复，独立手算和 9 项几何测试通过；[applied_guard_metadata.json](../../../outputs/research/canonical27_geometry_quality_v2/applied_guard_metadata.json)、[cpu_live_guard_validation.log](../../../outputs/research/canonical27_geometry_quality_v2/cpu_live_guard_validation.log)及 [hand_calculation.json](../../../outputs/research/canonical27_geometry_quality_v2/hand_calculation.json)保存具体证据。名义投影离最近参考中心 **100.955 mm**，方块 AABB 边隙约 **46.819 mm**；这些只是几何筛查，不是机器人可达或整条轨迹无碰撞证明。两个旧物理记录保留但不再计最终质量；修正后的Project与Reflect已在上述最终目录完成正例及有效错误落点负控，反中点质量门槛通过。

公共裸名 `InfiniteLine3D` API 与旧研究卡保持原三维含义；本轮只是明确的水平桌面操作变体，并未用 XY 结果代替一般三维能力。

## 最终网页与来源

[展示页](http://127.0.0.1:8781/outputs/research/canonical27_delivery_v1/index.html)按27种canonical类型组织，共30实例；First/Last/Second与Inside/OnTop保留各自子实例。最终 `--require-complete` 构建退出0，`summary.json`记录27/27类型、30/30实例、旧20与新10、13个保留重试版本。自然指令、实际外部/腕部录像、初始落稳、正负评分、原始状态和保存源码/配置分别可查。

历史20仍是12例120Hz和8例240Hz各自保存版本下的验收；本轮新10为240Hz，每worker420秒、GPU串行，不代表这30例曾在最终便携工作树上统一重跑。完整原参数与当前代码建议12000步/50秒新生成参数在复现矩阵中分开，后者未执行时不记验证。

最终网页核验通过：27张类型卡、30个实例、664个本地资源均存在且哈希稳定，190个媒体文件时长非零，664项HTTP HEAD全部200并符合HTML/MP4 MIME，JavaScript语法通过。报告见 [latest.json](../../../outputs/research/canonical27_delivery_v1/final_gallery_verification/latest.json)，首次旧服务失活的HTTP失败报告保留。浏览器播放交互未验证。未调用受评模型，没有模型成绩或广义分布覆盖结论。当前仅证明这些明确Base实例的合同、生成、实际操纵和验收链。
