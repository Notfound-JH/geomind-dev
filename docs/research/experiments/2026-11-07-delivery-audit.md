# geomind-research 交付核验与静态任务补验

核验日期：2026-11-07。仅CPU结构化语义与已有mock；没有模型、PhysX、训练或新后端。既有未提交文件保留，独立输出未覆盖旧结果。

## 对照上一份完整要求

| 原要求/此次重点 | 核验结论 | 实际文件与证据 | 剩余问题 |
|---|---|---|---|
| 读取规则、接入资料、当前源码；核对DSC定义 | 已完成，前摘要未展开；本轮再次核对。仓库/父目录未发现AGENTS.md | [定义差异](../decisions/detect_scene_change_semantics.md)、[接入资料](../../../geomind_handoff.md) | 两份资料体现为同文件工程部分与第7节；独立第二文件及早期V4原文未找到。新增/移除迁移决策未确认 |
| 项目skill与稳定流程 | 已完成；本轮实际使用后修正 | [SKILL.md](../../../.agents/skills/geomind-research/SKILL.md)、[使用记录](2026-11-07-select-exclusion-skill-use.md) | 新服务集成不在已实现能力内 |
| 三张研究卡与研究进度 | 已完成但摘要未逐项展示；本轮更新状态 | [进度](../progress.md)及三张任务卡 | 卡存在不证明后端实现；每卡待验证模型假设仍保留 |
| DSC 6对生成/独立校验/失败样例/来源快照 | 已完成，复用旧产物，无重复生成正式批次 | [原CPU报告](2026-11-07-dsc-counterfactual.md)；本轮10项历史回归通过 | 仍不证明真实模型历史利用；公共定义没有扩为新增/移除 |
| 后续模型白名单/缓存/身份/实际请求契约 | 已完成文档与纯数据导出，真实适配器未完成且先前就明确未接 | [model_input_contract](../model_input_contract.md) | 需要真实模型身份/不可变版本、服务schema、reset语义、预算；不允许oracle回退 |
| 任务类型与执行机制矩阵 | 前轮没有集中矩阵，本次补齐 | [execution_mapping](../execution_mapping.md) | 更新后查询、连续运动链本轮仅静态确认；不冒称全类型运行 |
| ProjectOntoLine/DSC设计到代码映射 | 前轮证据分散，本次集中补齐 | 同上两张映射表及共享/差异说明 | 科学意图中的未决点仍待研究者确认 |
| 静态查询的CPU闭环 | 前轮已有当前配置GT回答和相关单测，本次补成逐答案可复查产物 | [静态报告](../../../outputs/research/projection_diagnostic_20261107/reports/validation.json) | 不包含模型自然语言解析或物理到达 |
| 非零z偏移诊断 | 前轮未执行，本次实际补齐3对 | [research_projection.py](../../../geomind/tools/research_projection.py)、本报告 | 正式生成分布未改；泛化/模型捷径依赖未测 |
| SelectAfterExclusion实用审计、最小修改/验收方案 | 本次实际使用skill完成：两布局独立删条件、mock reset、方案；没有实施新后端/公共定义修改 | [审计与方案](2026-11-07-select-exclusion-skill-use.md) | 新必要性布局、语义适配器、mock语义修复仍未实施，明确留在下一阶段 |

## 静态闭环：实际做了什么

复用 `EpisodeResolver`、`SemanticEpisode`、`load_json` 和 `evaluate_answer`。生产投影公式本来就支持任意非退化三维方向，无需为了这个诊断改公共任务定义或生产求解器。新增的是 [research_projection.py](../../../geomind/tools/research_projection.py) 的确定性生成/校验；复用 `research_pairs.snapshot/check_provenance`，仅为快照增加可选索引名与额外源码参数，默认历史行为保持。

1. 原样复制当前 `ProjectOntoLine_v0` 及其引用配置到独立目录，seed仍为原配置0，CPU reset读取当前catalog坐标；没有读取USD内容。独立expected以NumPy最小二乘求线上的最近点，不调用生产solve。
2. 另生成3对几何诊断，seed=20261107，yaw由该seed采样；pitch分别 .2、−.35、.6，沿线位置s=−.14、0、.14。端点为c±.1d，q=c+sd；A法向量n=[−sin(yaw),cos(yaw),0]，B=d×n，均单位且与d正交，p=q+.07n。成对q、线、标签/ID、容差、偏移模长一致，只改偏移方向。B的非零z不是只把整场景平移到非零高度。
3. 在每个完整公开observation上独立核验q在线上、(p−q)·d≈0、生产GT一致、无历史、step0允许答。全部对象展开保存在逐例result.json。
4. 每例启动独立环境，将保存的答案JSON经生产 `load_json` 读取再交给 `step`。正确答案与0.5τ接受，2τ及偏移.01m拒绝；错frame、缺frame、多字段、维度错、NaN、重复键、语法错均抛ContractError并记录原因。τ=1e−6 m；源码条件为distance≤τ。本轮没有声称验证浮点恰好等于τ的边界。

共有7例×11项=77项答案检查：每例2次成功、2次合法错误评分、7次契约/解析拒绝。**这里的答案解析是JSON与接口schema校验，不是受评模型输出适配器。** 错frame按协议错误拒绝，不能与合法错点的score=0混报成同类语义错误。

## 具体结果

| 诊断对 | A：n_z=0的z插值误差 | B：n_z≠0的z插值误差 | 生产GT与独立构造 |
|---|---|---|---|
| 0，pitch=.2 | <1e−12m | .3453208413m | 两例一致，误差<1e−12m |
| 1，pitch=−.35 | <1e−12m | .1917658511m | 两例一致，误差<1e−12m |
| 2，pitch=.6 | <1e−12m | .1023187163m | 两例一致，误差<1e−12m |

当前catalog配置的z插值误差≈8.9e−17m。诊断证明该固定捷径在B不能提供容差内答案，生产语义求解/评分可处理这些样例；没有证明某模型会用该捷径，也没有证明模型一般三维投影能力。

完整示例：[当前配置](../../../outputs/research/projection_diagnostic_20261107/reports/ProjectOntoLine_v0/result.json)、[A](../../../outputs/research/projection_diagnostic_20261107/reports/projection_0_a/result.json)、[B](../../../outputs/research/projection_diagnostic_20261107/reports/projection_0_b/result.json)。每个目录的answers/保留包括故意非法JSON在内的全部回答输入；非法文本只用于解析测试。

## 执行、测试与复现

实际命令，仓库根目录：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m geomind.tools.research_projection --output outputs/research/projection_diagnostic_20261107 --seed 20261107
```

重跑请选择不存在的新output。命令不启动模型或仿真；输出已存在会拒绝覆盖。运行证据见上面机器报告及 [provenance](../../../outputs/research/projection_diagnostic_20261107/provenance.json)：commit=`8ab25dd2b8ddbc3abdfe0017b45f9da971f4f5de`；36份相关源文件内容快照=`b6677099cbb57b81f074e34250722a4fba1c962ee4188485109a03b4e7c79240`；配置集合=`a0aa02802c13f2e1c422fb2662d250031ee2edd2c571c9b7714bd2881e2456f5`。包括相关未提交内容，不只dirty标志。

| 本轮检查 | 结果 | 所证明的范围 |
|---|---|---|
| [静态新增测试日志](../../../outputs/research/projection_diagnostic_20261107/acceptance/projection.log) | 4项通过 | 7例答案/诊断、错误独立标注拒绝、生成不调用solve、新索引快照校验 |
| [历史回归日志](../../../outputs/research/projection_diagnostic_20261107/acceptance/history_regression.log) | 10项通过 | 快照工具扩展未破坏既有配对校验；不是新模型实验 |
| [空间流水线日志](../../../outputs/research/projection_diagnostic_20261107/acceptance/native_semantics.log) | 22项通过 | 明确测试名对应的几何/阶段/契约/迁移性质，不代表所有任务类型 |
| [组合逻辑日志](../../../outputs/research/projection_diagnostic_20261107/acceptance/constraint.log) | 6项通过 | 既有唯一性/并列拒绝/净距等；必要性另由本轮独立删条件审计 |
| [Skill格式](../../../outputs/research/projection_diagnostic_20261107/acceptance/skill.log) | 通过 | 仅语法；实际可用性证据见SelectAfterExclusion使用记录 |
| [引用与源码完整性检查](../../../outputs/research/projection_diagnostic_20261107/acceptance/integrity.json) | 通过 | 新报告引用有效；静态实验当前源码与保存副本一致 |

旧DSC和本轮Select审计保存的skill版本早于本轮修订，旧DSC快照还早于snapshot可选参数扩展；旧报告保持原样。若用“当前源码必须逐字节相同”的旧validate，会正确报告漂移，不能为了通过而覆盖旧快照或将旧成绩自动升级为新版本证据。本轮历史回归是在当前代码运行，单独保留日志。

仍未完成：真实模型接入/表现、SelectAfterExclusion必要性新布局及语义适配器、连续追踪CPU公共接口/逐时刻指标、更新后/连续运动的本轮真实执行、早期增删定义确认。它们没有被卡片、测试总数或oracle结果替代。
