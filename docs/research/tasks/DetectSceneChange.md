# DetectSceneChange 研究卡

更新：2026-11-07。阶段：位移配对生成/评分验收；模型历史利用尚未验证。版本证据见[实验记录](../experiments/2026-11-07-dsc-counterfactual.md)。

## 能力主张与定义版本

- 文档意图：[v3 契约](../../episode_contract_v3_zh.md)比较 observe.End 与 Respond；[分类](../../canonical_task_families_zh.md)属于 SceneUpdating。
- 实现事实：[resolver.py](../../../geomind/episodes/resolver.py) `solve` 对固定 ID 的三维位移作阈值判定，非视频轨迹辨识、非新增/移除。公共配置 [DetectSceneChange_v0](../../../geomind/configs/episodes/DetectSceneChange_v0.json) 使用 schema3/release2。
- 本研究操作化：`DetectSceneChange/displacement-counterfactual-v1`；历史 V4 新增/移除线索、未找到迁移决策及待确认问题见[定义差异](../decisions/detect_scene_change_semantics.md)。不擅自替换公共定义。
- 推断：稳定 ID 下主要测参考位置保存/比较。长时记忆、同质对象身份保持等独立研究意图未确认。

## 实际输入

[runtime.py](../../../geomind/episodes/runtime.py) `SemanticEpisode.observation` 公开 instruction/phase/step/scene_visible/objects/frame/allow_response；objects 完整包含稳定 ID、label、精确 position_m，无图像。隐藏期为空。历史由 agent 已见观测缓存；env.history、result、事件计划为评估特权。完整已有样例见[接入资料 §7.4](../../../geomind_handoff.md)，新配对样例及严格白名单见[输入契约](../model_input_contract.md)和本次报告。没有实际模型请求。

## 生成约束与替代解法

实现要求恰好一项 Δ>τ，所有项 |Δ−τ|>ε，参考来自同实例 Observe.End，协议 Observe→Delay→Respond；位置和事件须合法。依据 `EpisodeResolver._validate_protocol/_validate_parameters/solve`。旧 [generate_spatial_suite.generate](../../../geomind/tools/generate_spatial_suite.py) 只改一个对象 y、要求位移≥.12，固定 x 轨道；“隐藏事件”不排除最终坐标统计捷径。接入单例选唯一负 y 即能答对，这不是总体捷径成功率。

本次 [research_pairs.generate/independent_case/validate_pair](../../../geomind/tools/research_pairs.py) 固定共同终态，分别倒推不同目标初态；同幅度/时长/顺序、不同唯一标签。独立预期不调用生产 solve，但复用生产 schema/runtime，独立性范围须保留。

## 已有证据

- [test_spatial_episode_pipeline.py](../../../tests/test_spatial_episode_pipeline.py) 验证时间边界、历史隔离和可见性等实现；不证明模型使用历史。
- [test_research_pairs.py](../../../tests/test_research_pairs.py) 验证同终态不同标签、负例拒绝、独立答案与生产 GT 交叉检查、payload 边界及快照完整性。
- 本次执行的配置/结果/失败样例、数量与源码快照见[CPU 实验记录](../experiments/2026-11-07-dsc-counterfactual.md)。固定 final-only 规则的不变量验证不是模型评估。历史 oracle 到达报告不迁入当前模型成绩。

## 待验证假设、下一实验与完成标准

假设 H1：某个版本固定的受评模型在 full-history 中能随历史切换标签，而 final-only 无法区分同终态两例。**未执行**。

下一最小实验：使用这6对的白名单数据，固定模型版本/解码参数、逐例清空上下文，随机展示；分别发送完整六条观测或最终一条。保留实际请求/响应并单独记录解析/超时故障。完成标准：输入隔离审计通过、12例每条件均有可追溯尝试和分母、报告配对切换与两条件差异。模型失败也是实验结果；不得以正确率达标才“完成”。若要科学上确认能力，需进一步样本量/不确定性分析，本批不足以估计泛化。

当前工程完成标准见[配对协议](../history_counterfactual_protocol.md)。未解决：研究者认可的定义迁移与目标记忆尺度、真实模型接入。增删语义/视觉/PhysX 均未验证。

## 后续物理验收（2026-11-07）

本页前述CPU反事实记录不变。本次另外运行位移Base的真实PhysX观察→隐藏外力事件→查询→ReachAboveTarget，分开保存公开流、特权历史、事件计划、逐步实测状态与控制动作；结果见 [流水线实验](../experiments/2026-11-07-pipeline-validation.md)。这补齐当前位移执行证据，不解决新增/移除定义，也不证明模型历史利用或连续身份追踪。

## 2026-11-09 变化识别后的真实抓取

[本次实验](../experiments/2026-11-09-scene-updating-grasp.md)增加PickLiftChangedObject：外力隐藏事件结束后实际夹爪抓起正确物体并稳定保持。semantic_table两seed正例2/2，错误目标2/2正确拒绝，轨迹复算4/4；前两轮速度不稳定失败保留。模型记忆与跨目标标签泛化未测。
