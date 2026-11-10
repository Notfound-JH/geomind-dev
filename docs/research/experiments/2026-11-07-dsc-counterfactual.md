# DetectSceneChange 历史反事实配对：CPU 验收记录

执行日期：2026-11-07。状态：**生成与评分闭环完成；没有受评模型**。语义版本 `DetectSceneChange/displacement-counterfactual-v1`，仅位移变化，不覆盖早期新增/移除定义。

## 结果与证据

| 检查 | 本次结果 |
|---|---|
| 配对合法、同终态公开 observation、不同历史/答案 | 6/6 通过，0生成失败 |
| 目标平衡 | object_0/1/2 在 A、B 各2次 |
| 生产 GT 对独立位移答案 | 12/12 一致 |
| 正确 ID 评分 | 12/12 接受 |
| 每例另外两个错误 ID 评分 | 24/24 拒绝 |
| 六类故意无效实例 | 6/6 按预期原因拒绝 |
| 固定终态首对象规则 | 6/12；每对至多1，不是模型成绩 |
| 自动测试 | 新增10项、现有空间流水线22项、v3契约11项，合计43项通过 |
| Skill | skill-creator 格式检查通过；本地引用路径检查通过 |
| 独立目录复核 | 当前源码/保存源码/配置哈希验证通过，6对重新校验通过 |

机器报告：[validation.json](../../../outputs/research/dsc_counterfactual_v1_20261107/reports/validation.json)；[复核报告](../../../outputs/research/dsc_counterfactual_v1_20261107_recheck/validation.json)；[失败样例总表](../../../outputs/research/dsc_counterfactual_v1_20261107/reports/failure_cases/summary.json)。所有正式配置与历史结果保持原样，本轮文件均在新目录；未重置既有未提交修改。

测试原始输出：[新增测试](../../../outputs/research/dsc_counterfactual_v1_20261107/acceptance/research_tests.log)、[空间测试](../../../outputs/research/dsc_counterfactual_v1_20261107/acceptance/spatial_tests.log)、[契约测试](../../../outputs/research/dsc_counterfactual_v1_20261107/acceptance/contract_tests.log)、[skill格式](../../../outputs/research/dsc_counterfactual_v1_20261107/acceptance/skill_format.log)、[链接检查](../../../outputs/research/dsc_counterfactual_v1_20261107/acceptance/reference_check.json)。新增测试实现见 [test_research_pairs.py](../../../tests/test_research_pairs.py)。

## 复现标识和命令

- commit：`8ab25dd2b8ddbc3abdfe0017b45f9da971f4f5de`，分支 `chore/portable-dev-20260924`，dirty=true。
- 相关源文件内容快照（34份，含未提交/未跟踪实现、测试及skill）：`89a1f802454971709dff38db5e933683225d0cca9a84978d46e95f807a432e31`。
- 配置/索引集合（39份）哈希：`2650981fd6029bd4a5ac42e44f510e837d9d6f722914016120f1ad95ff033dd4`。
- 主 seed=20261107；每对 episode seed=主seed+pair序号，A/B一致。全部具体配置、展示顺序和方法保存在 [pair_index](../../../outputs/research/dsc_counterfactual_v1_20261107/pair_index.json)；只供评估端。
- Python3.13.9、numpy1.26.4、gymnasium1.2.3；逐文件 SHA256、git 状态、命令、时间见 [provenance.json](../../../outputs/research/dsc_counterfactual_v1_20261107/provenance.json)。源文件副本位于同目录 source_snapshot/，不只记录 dirty 标志。快照不是完整环境镜像，未收集凭据/资产/权重/数据集。

在仓库根目录实际执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m geomind.tools.research_pairs generate --output outputs/research/dsc_counterfactual_v1_20261107 --pairs 6 --seed 20261107 --failure-examples
PYTHONDONTWRITEBYTECODE=1 python -m geomind.tools.research_pairs validate --run outputs/research/dsc_counterfactual_v1_20261107 --report-dir outputs/research/dsc_counterfactual_v1_20261107_recheck
```

目录已存在，重新生成须换新的 output，复核须换 report-dir。其余测试命令与单个失败重放见[协议](../history_counterfactual_protocol.md)及 acceptance 日志。工具拒绝覆盖现有目录；无静默重用旧结果。

## 一个具体配对与失败样例

[pair_000 完整证据](../../../outputs/research/dsc_counterfactual_v1_20261107/reports/pairs/pair_000.json)：A正确 object_0，B正确 object_2；两者位移 .12 m，observe2/delay3/query step5。其终态 observation 哈希均为 `ac557c38d064dc000f3ad2310aab1beeb82857d8c932b2a742c56591c3b4d16a`。证据文件包含初态、终态、完整公开六步观测、独立距离与生产 GT 哈希，不是模型输入。

纯数据输入示例：[A full-history](../../../outputs/research/dsc_counterfactual_v1_20261107/reports/payload_candidates/full-history/cf_000_a.json)、[B full-history](../../../outputs/research/dsc_counterfactual_v1_20261107/reports/payload_candidates/full-history/cf_000_b.json)、[A final-only](../../../outputs/research/dsc_counterfactual_v1_20261107/reports/payload_candidates/final-only/cf_000_a.json)、[B final-only](../../../outputs/research/dsc_counterfactual_v1_20261107/reports/payload_candidates/final-only/cf_000_b.json)。这些文件内容不含pair/seed/答案；文件名仅为评估端存储键，未发送给任何模型。

| 故意破坏的条件 | 实际拒绝代码/原因 |
|---|---|
| 两例答案相同 | same_answer / Counterfactual pair has identical labels |
| B未移动对象的终态z改变 | different_final_observation / 含排序、文本、step的终态不同 |
| B另一个对象也移动 | changed_count / Expected one changed object, got 2 |
| 位移=.0305 m 落入阈值歧义带 | threshold_boundary |
| 初态x=5越出工作空间 | contract / Initial object position outside workspace |
| 同步重复移动事件 | contract / Conflicting simultaneous events |

每项均有自己的 configs/、pair_index.json 与 rejection.json；从总表所在目录进入对应子目录可重放。[same_answer 独立CLI重放](../../../outputs/research/dsc_counterfactual_v1_20261107_failure_recheck/failure.json) 按预期退出2。无效例不是从成绩分母中偷偷删掉的失败模型样本，而是预先指定的校验器负测试。

## 本次证明与未证明

证明范围：对这批12例，独立从初终态计算的唯一位移标签与生产 GT 一致，公开终态在配对内完全相同，协议/工作空间/阈值约束成立，正负身份评分符合约定；指定非法配对可被拒绝。独立 oracle 不使用生产 solve 构造预期，但共享配置 schema 和 runtime，因此不是全链路独立实现审计。

信息论限制：只使用相同终态且无额外上下文的固定规则不能同时正确回答一对不同标签。这说明该配对对历史信息有必要性约束，**尚未证明任何受评模型具备历史利用能力**。本次 model_results=null；没有真实请求/响应、checkpoint 或模型成绩，未执行 PhysX、训练、远程服务。也未验证视觉理解、长期记忆容量、动作、新增/移除语义、全任务分布泛化。

下一步所需最少信息：实际模型服务及不可变版本、JSON输入/对象ID输出协议、stateless或reset保证、解码参数与超时/预算。按照[输入契约](../model_input_contract.md)实现小型 fail-closed 适配器后，才能运行 full-history/final-only 模型配对实验。SelectAfterExclusion 排除必要性和 ProjectOntoLine 三维偏移实验仍是未执行建议。
