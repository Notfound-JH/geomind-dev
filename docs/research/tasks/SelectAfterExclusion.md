# SelectAfterExclusion 研究卡

更新：2026-11-07。状态：本轮实际使用skill完成两布局条件删减与mock reset复核，见[使用记录与最小工程方案](../experiments/2026-11-07-select-exclusion-skill-use.md)；新的必要性布局、语义适配器及反事实模型实验**未执行**。

## 能力主张与定义版本

文档意图：[canonical 分类](../../canonical_task_families_zh.md) ConstraintComposition。实现版本以当前 [constraint_selection_logic.py](../../../geomind_lab/mimic/envs/constraint_selection_logic.py) 的 `resolve_exclusion_nearest_constraint_problem`、legacy 协议配置和源码快照为准；没有查到独立科学语义版本号。

实现事实：属性过滤（红、cuboid）→在过滤集合内排除唯一最左者→选择距蓝盘世界 XY 最近者；expected_target 不符会拒绝。推断：可研究有限算子组合，但不是任意双关系交集或多跳。后两者的配置 [SelectByTwoRelations](../../../geomind/configs/tasks/SelectByTwoRelations_v0.json)、[SelectByIndirectReference](../../../geomind/configs/tasks/SelectByIndirectReference_v0.json) 仍为不可用，不能据本任务推定已完成。

## 实际输入与版本依据

已有 [geomind_handoff §7.5](../../../geomind_handoff.md) 包含完整 mock observation。该 mock 落入 [MockBackendEnv._build_generic_binding](../../../geomind/core/envs/mock_backend.py)，公开 goal_xy/binding，目标甚至为 blue_plate，不能代表组合任务研究输入。旧真实环境和 HTTP 字段范围与 OpenPI 不同，详见[输入契约中的适配器缺口](../model_input_contract.md)。目前没有可绑定 checkpoint 的真实请求快照；本轮不制造结构化模型成绩。

初始布局：[scene65_base](../../../geomind/configs/layouts/select_exclusion_unique_scene65_base.json)；scene17_stress 是 [task_profiles.py](../../../geomind_lab/tasks/manager_based/manipulation/geomind/task_profiles.py) 中的布局键，并非独立 JSON 文件。相关 [legacy 标注](../../../geomind/configs/legacy/layouts/standard.json) 经 [task_specs.load_annotated_tasks](../../../geomind_lab/tasks/manager_based/manipulation/geomind/task_specs.py) 合并。Oracle 可读取完整属性/坐标，不能假定模型同样收到这些字段。

## 生成约束与替代解法

令 S 为属性符合的候选，要求 |S|≥2，唯一极小 x（并列容差 ε_rank），删除后集合非空，唯一最近者（ε_dist）。当前布局常用两种 ε=.003 m。生产求解保证唯一性，不强制每个条件必要；属性来自配置而不是图像识别。现有布局人工指定，没有强制组合必要性的随机生成器。

接入资料发现上述两个名义布局中red_block_target也是忽略颜色/排除后的最近者。本轮重新加载合并配置，以独立距离/属性枚举分别删除排除、属性或两者，均仍选该目标；完整结果、初次审计脚本失败与修复记录见使用记录。此结论限两布局；对象名含target/leftmost仍是潜在捷径，不是模型实验。

## 已有证据与待验证假设

[test_constraint_selection_protocol.py](../../../tests/test_constraint_selection_protocol.py) 对标注答案、并列极值/距离拒绝和净距的检查支持实现边界；直接调用生产求解器，不是独立组合 oracle。旧 direct 消融同时改变语言和目标显著性，见 [rebuttal_metrics](../../../paper/rebuttal_metrics.md)，不能隔离组合算子。

H1（未执行）：当排除确实改变正确目标时，模型仍能遵循组合指令。H2（未执行）：去掉属性条件后表现改变。现有 oracle、generic mock 和历史 VLA 成绩不能证明这两项。

## 下一最小实验与完成标准

建议，同一结构化场景两条查询：A最近红块，B最近且非最左红块。布置最左红块 e 同时最近，t 为排除后唯一最近，差值>ε；中性随机 ID，属性公开，隐藏 expected_target 等元数据。固定位置、候选、控制预算，只改指令及正确答案；可再放更近非红块，但首轮只检验排除效应。**未生成、未执行**。

完成标准：独立枚举标注与生产解一致、A/B标签不同、去掉排除条件明确改变答案、无并列/越界、输入白名单可审计；模型阶段须另有不可变身份和实际请求，报告语义选择而非用到达评分代替。研究者待确认：“最左”作用域是否就是红色子集，以及一般关系组合的科学目标。
