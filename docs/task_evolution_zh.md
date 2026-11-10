# 任务迁移状态

| 旧任务 | 当前处理 |
|---|---|
| AnchorDistance | 合并已有实现到 SelectByDistance，归属 RelationalGrounding；Standard/Vla 和旧 ID 均保留兼容 |
| SelectBetweenObjects | 移出 Canonical Registry；Between 仅作为 RelationalGrounding 的关系变体记录 |
| SelectVisualSpatial | 改为 SelectWithVisualCue，保留视觉属性与空间关系联合筛选 |
| ReindexExtreme | 合并到 SelectByOrder，通过 index/from 参数区分最左、最右 |
| ReindexInsert | 改为 UpdateObjectOrder，保留插入事件；历史信息要求暂缓 |
| ReindexNeighbor | 改为 SelectByAdjacency，GT 使用有向链的直接前驱/后继，排除链外干扰物 |
| PlaceDirectional | 移出活动注册、评估清单和默认启动配置；旧配置归档在 configs/deprecated |
| PlaceInside | 合并为 PlaceByTopology 的 Inside variant |
| PlaceOnTop | 合并为 PlaceByTopology 的 OnTop variant |
| TrackClutter | 旧配置保存在 configs/legacy；TrackObjectOrigin 的 Clutter variant 为待实现占位，不参与评估 |

## 配置与选择

公开 task_id 仍为裸任务名，condition、layout、version 独立保存。
新增的顶层 variant 表达 Inside / OnTop 等任务变体；parameters 表达顺序等参数。
它们与 runtime.variant（旧执行协议版本）分开。

```json
{
  "task_id": "SelectByOrder",
  "family": "TargetIdentification",
  "condition": "Base",
  "layout": "Line",
  "version": 0,
  "parameters": {"index": 1, "from": "Right", "reference_time": "Current"}
}
```

同一个任务名匹配多个参数实例或变体时，注册器要求显式选择：

```bash
TASK_ID=SelectByOrder HEADLESS=0 bash geomind/scripts/run_task_isaaclab.sh --order-index 1 --order-from Right
TASK_ID=PlaceByTopology HEADLESS=0 bash geomind/scripts/run_task_isaaclab.sh --task-variant Inside
TASK_ID=SelectWithVisualCue HEADLESS=0 bash geomind/scripts/run_task_isaaclab.sh
TASK_ID=UpdateObjectOrder HEADLESS=0 bash geomind/scripts/run_task_isaaclab.sh
```

VLA 通过 --interface Vla 选择。旧版本化 ID 保留为兼容别名；PlaceDirectional 的旧 ID 不再注册。
TrackClutter 的旧 ID 会指向待实现占位，不能据此声明新来源追踪已实现。

## 暂缓的历史任务

UpdateObjectOrder 当前仍按插入后的顺序选择目标，parameters.history_mode 为 Deferred。
TrackObjectOrigin 已记录“初始来源区域”方向，但 runtime.implementation_status 为 placeholder。
当前不启用初始排序箭头、来源色块或新的来源历史 GT。

## 代码入口与检查范围

- geomind/core/schema.py、registry.py：独立任务字段、参数和 variant 选择。
- geomind_lab/tasks/manager_based/manipulation/geomind/task_evolution.py：将公共任务合同应用到复用的底层协议。
- geomind_lab/mimic/envs/task_contract_logic.py：结构邻接 GT、按环境读取已验证的运行时目标。
- geomind_lab/mimic/envs/task_context_visuals.py：有向链连线与箭头。
- tests/test_task_evolution.py：合并、废弃、邻接、历史暂缓与生成器兼容检查。

已执行离线逻辑、配置及直接消融检查。未执行真实 IsaacLab 渲染、碰撞和抓取验证。
