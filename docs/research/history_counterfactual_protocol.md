# 位移历史反事实配对 v1

研究问题：在最终公开信息完全相同而正确身份不同的配对上，历史信息是否必要？当前先验收生成与评分，**尚未执行模型比较**。定义边界见[语义差异登记](decisions/detect_scene_change_semantics.md)。

## 生成与独立校验

[research_pairs.py](../../geomind/tools/research_pairs.py) 复用 EpisodeResolver 和 SemanticEpisode。每对三个中性 ID、同样排序/标签/指令/step、观察2步、隐藏3步、查询step=5；τ=.03 m、ε=.001 m、jitter=0。先采样共同终态，再以同一 XY 位移向量（长度 .12 或 .15 m）分别倒推 u、v 的初态，隐藏期 MoveObject 回共同终态。对象集合不变，无新增/移除。scene 的资产路径只作为 CPU schema 字符串，没有加载 USD 或 PhysX。

枚举三个对象的全部六种有序不同目标组合，每六对中每个身份在 A/B 各出现两次；用 `random.Random(seed)` 打乱组合、对象顺序和12个 episode 的展示顺序。同一 PRNG 流顺序已由配置/索引固定，不声称独立随机数流。默认只运行6对；这是匹配设计的小批构造，不代表原生成分布的随机样本。

`independent_case` 从配置初态与事件重建终态，独立计算平方和开根号距离和标签，**不调用生产 solve**。共享生产 schema 校验，故不声称全系统完全独立。随后比较生产 GT、runtime 终态/参考快照，检查每例唯一变化、阈值安全边界、有限数值/工作空间/协议合法、配对同控制量同位移幅度、最终公开 observation 按序列化字节相等。用独立正确 ID 与另外两个错误 ID 分别新建 episode 评分。`condition_payload` 输出白名单数据，参见[模型输入契约](model_input_contract.md)。

固定 final-only 规则总选公开顺序的第一个对象。相同输入得到相同预测，而同对两个标签不同，因此至多答对一个；实测命中数只作不变量演示，**不是受评模型成绩**。50% 是这类等权配对、仅终态且无外部泄露规则的平均上界，不是原三候选分布的通用 chance 指标。

## 执行命令

工作目录为仓库根目录，使用已能导入 geomind/numpy/gymnasium 的 CPU Python。工具不联网、不读模型。每次选择不存在的新目录：

```bash
python -m geomind.tools.research_pairs generate --output outputs/research/dsc_counterfactual_v1_20261107 --pairs 6 --seed 20261107 --failure-examples
python -m geomind.tools.research_pairs validate --run outputs/research/dsc_counterfactual_v1_20261107 --report-dir outputs/research/dsc_counterfactual_v1_20261107_recheck
python -m unittest discover -s tests -p test_research_pairs.py -v
python -m unittest discover -s tests -p test_spatial_episode_pipeline.py -v
python -m unittest discover -s tests -p test_episode_contract_v3.py -v
```

前两个路径已用于本次交付；复跑生成请换新路径，复核请换 report-dir。报告保留 configs、pair_index.json、source_snapshot、provenance.json、逐对证据、payload_candidates 和六类 failure_cases。provenance 记录 commit、实际加载的相关项目源码与 episodes 实现哈希输入文件（包括未提交/未跟踪内容）的 SHA256及副本、配置哈希、seed/命令、Python/numpy/gymnasium 版本。不是全仓导出，也不含资产/数据。validate 会校验保存副本和当前源码；当前代码漂移时拒绝复核，需新实验或另行在隔离检出中复现。

重放一个故意无效实例（**预期退出码2**，failure.json 记录 same_answer）：

```bash
python -m geomind.tools.research_pairs check-pair --case outputs/research/dsc_counterfactual_v1_20261107/reports/failure_cases/same_answer --report-dir outputs/research/dsc_counterfactual_v1_20261107_failure_recheck
```

check-pair 是单例诊断，不做批次平衡或 provenance 检查；正式复核必须用 validate。其余保存的无效例为终态不同、多对象变化、阈值附近、越界和同一步重复事件。测试还包括生产 GT 损坏、观测顺序改变、payload 多余字段、配置/快照损坏和禁止覆盖。

## 验收与解释

6/6 配对有效、12次正确答案接受、24次错误答案拒绝；六类无效样例按预期原因拒绝；受影响测试、skill 格式与引用检查通过，才完成本次生成闭环。没有模型身份/实际请求记录时，model_results 必须为 null。后续比较 full-history 与 final-only 的答案切换和配对正确率；都高先审计泄露/串缓存，都低先审计输入适配。一次成功只能支持此条件中的历史利用，不能外推视觉记忆、长期容量、增删变化或动作能力。
