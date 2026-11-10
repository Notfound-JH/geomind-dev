# DetectSceneChange 定义差异登记

状态：**设计决策缺失，非新公共定义**。核对日期：2026-11-07。

用户指出早期 V4 清单将本任务描述为“新增/移除”。本轮搜索本地文档、配置、任务清单和可用 git 历史，没有定位到该 V4 原文或从新增/移除改为位移的决策记录。因此早期定义是用户提供的历史线索，不能假称已独立核实；需要研究者提供原始清单位置和批准的语义版本。

当前可核实的定义是：在固定候选对象集合中，比较 Observe.End 与 Respond 的三维参考点位置，恰好一个对象位移超过 τ，且所有位移避开 τ±ε 的歧义带。依据：[v3 契约](../../episode_contract_v3_zh.md)、[resolver.py](../../../geomind/episodes/resolver.py) 的 `TASKS`、`EpisodeResolver._validate_protocol`、`solve`；协议仅支持 `MoveObject`，快照要求对象集合一致。[runtime.py](../../../geomind/episodes/runtime.py) 的 `SemanticEpisode.step` 实施隐藏期移动和参考快照保存。

本次实验操作化版本为 `DetectSceneChange/displacement-counterfactual-v1`；配置 schema_version=3、release_version=2.0 是格式版本，不能充当历史科学定义已经统一的证据。本次不更改公共 task definition、不实现增删事件，也不宣称早期新增/移除目标已完成。

待确认：两种定义应成为独立子任务，还是已有批准的替换/迁移？未经研究者确认，不调整公共分类或把位移实验结果归入新增/移除能力。
