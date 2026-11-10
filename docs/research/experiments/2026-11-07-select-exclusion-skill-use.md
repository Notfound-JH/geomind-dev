# geomind-research 实际使用：SelectAfterExclusion 审计

状态：**本轮已执行设计→源码→合并配置→独立条件删减计算→mock reset→验收方案**；未改公共任务定义、未启动新后端或真实模型。不是只做格式/链接检查。

## 设计定位与当前事实

[分类](../../canonical_task_families_zh.md)的ConstraintComposition对应 [constraint_selection_logic.py](../../../geomind_lab/mimic/envs/constraint_selection_logic.py) `resolve_exclusion_nearest_constraint_problem`：属性筛选→筛选集合中唯一极值排除→剩余集合距离最小。通过 `build_constraint_selection_problem_from_object_positions` 进入；原生 `resolver.TASKS` 尚无此任务。旧布局来自 [task_specs.load_annotated_tasks/load_task_spec/resolve_task_protocol](../../../geomind_lab/tasks/manager_based/manipulation/geomind/task_specs.py)；base还有独立初始化覆盖，stress是布局键，不能假造同名JSON路径。

实际读取的legacy字段是 `protocol.anchor_object.object_id`、`protocol.candidate_object_ids`、`object_positions[id].position`；不能把这些数据当作原生objects数组。本轮审计脚本先后错误假设anchor在gt_spec、坐标直接为列表，两次属于审计脚本系统故障，已保存 [attempt_01](../../../outputs/research/select_exclusion_audit_20261107/attempt_01_failure.json)、[attempt_02](../../../outputs/research/select_exclusion_audit_20261107/attempt_02_failure.json)，对照字段提取函数修正后才得出结果。生产求解器没有在这两次故障中失败，未隐藏为“无效任务”。

## 条件必要性实测

以下是独立按scalar XY距离与属性枚举的结果，与生产答案交叉核对；不是调用生产求解器模拟所有消融。

| 现有布局 | 完整条件 | 去排除 | 去属性（保留全候选中排除极值） | 两者均去掉 |
|---|---|---|---|---|
| scene65_base | red_block_target | 相同 | 相同 | 相同 |
| scene17_stress | red_block_target | 相同 | 相同 | 相同 |

所以这两份名义布局既有唯一解，又存在条件冗余。只支持这两布局，不代表所有seed/实测姿态，也不是模型忽略条件的实验证据。[audit.json](../../../outputs/research/select_exclusion_audit_20261107/audit.json)包含距离、协议、独立/生产目标和完整mock observation；[已消费配置](../../../outputs/research/select_exclusion_audit_20261107/configs/select_exclusion_unique_scene65_base.json)可复核输入。

本轮实际 `create_env('SelectAfterExclusion', backend='mock').reset(seed=0)`：`goal_kind=generic`、`target_object_id=blue_plate`、goal_xy≈[.46333334,.01333333]；生产语义目标是red_block_target。来源是 [MockBackendEnv._build_task_binding/_build_generic_binding](../../../geomind/core/envs/mock_backend.py) 没有识别exclusion resolver，落到所有物体位置均值，目标ID取首物体。完整raw observation还含meta与goal信息，cameras为null。它不证明组合选择、视觉理解或抓取完成。

旧真实环境有组合求解入口与goal sequence，但未在本次运行；[LayoutAwareEvaluator](../../../geomind/tasks/evaluators/simple.py)的距离/dwell评分不等于对象ID答案，也不必然证明“拿起”。因此应分别验收语义答案、backend目标和动作成功。

## 最小工程修改方案（建议，尚未实施）

| 目标 | 最小改动位置 | 验收，不以模型成功率代替 |
|---|---|---|
| 结构化语义研究闭环 | 复用 `constraint_selection_logic` 的生产求解器；在现有研究工具目录加薄的选择实验入口，先归一legacy坐标/属性，导出中性ID+属性+XY+指令；评估侧独占protocol/expected_target | 独立枚举答案；白名单禁止expected_target/goal/meta；正确ID接受、错ID拒绝、并列/缺参照拒绝；不需要增加新环境或修改公共任务定义 |
| 让排除条件必要 | 在独立实验输出目录构造同场景两查询：A最近红块，B最近且非最左红块；e同时最左且最近，t为次近，距离差>.003 | A=e、B=t；删排除明确改变GT；中性ID、不靠target命名。同步独立expected_target，避免旧固定答案校验错误阻止合法换目标 |
| 让属性条件也必要 | 放比e/t更近的非红候选，分别保持/去掉属性条件，明确“最左”的作用域 | 每次只改一条件并重算目标，保证唯一且有足够差距；两条属性边界应有反事实实例；设计作用域需研究者确认 |
| 若后续确需mock与语义等价 | 在 `MockBackendEnv._build_task_binding` 显式识别 `exclusion_nearest_constraint_selection`，新增binding调用已有求解器，将目标ID/位置设为语义答案；该resolver不得走generic后备 | base/stress及新必要性布局的binding等于独立GT；错误配置显式失败；仍须隔离goal泄露。修复mock binding不自动得到抓取或物理验证 |

首个结构化研究闭环无需先修mock；不要用generic轨迹作为“完成”捷径。此处只交付最小改动与验收方案，未声称实现这些建议。

## Skill 实际反馈与修正

原流程主要给出历史配对工具，容易把其他机制也套成历史实验；修订为按静态/更新后/历史比较/连续追踪分流。原规则说“唯一性不等于必要性”，但未要求实际删条件计算；现加入单条件删减和目标比较。此次两个脚本故障暴露legacy字段包装不能凭研究卡猜测，加入按实际合并配置核对schema。再加语义目标/backend binding/评分目标三方核对及“已完成未展示/本次补齐/未完成”交付区分。参见[修订skill](../../../.agents/skills/geomind-research/SKILL.md)。

复现记录：[audit.py](../../../outputs/research/select_exclusion_audit_20261107/audit.py)、[provenance](../../../outputs/research/select_exclusion_audit_20261107/provenance.json)。命令为 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python outputs/research/select_exclusion_audit_20261107/audit.py`；现有结果拒绝覆盖，重放应复制脚本到新目录。快照记录实际使用的旧skill，修订发生在审计之后，不能将后续skill字节冒称此次审计输入。没有模型或物理结果。
