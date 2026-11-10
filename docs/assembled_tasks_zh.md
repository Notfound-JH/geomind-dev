# 组装场景与任务GT生成

沿用geomind-research工作流，新入口把agent提交的“场景、资产、布局、任务与角色选择”接到既有组装器和原生任务求解器。GT读取实际PhysX落稳后的物体状态，不从指令猜答案，也不另造一套任务物体。

`assemble_and_solve_gt`生成自然抓放任务、初始场景和语义GT，并检查初始落稳及目标静态几何；此阶段不执行机器人动作。新增 `assemble_solve_execute` 在此基础上选择本体、绑定原场景物体，重新求解运行时GT后执行真实接触动作，按物体末态验收。

## 机器人和动作阶段

按[本体说明](robot_embodiments_zh.md#接入组装任务与geomind-research)选择本体和基座，查询菜单中的实际动作支持；当前组装抓放入口首先支持Panda。请求需显式 `stage: "assemble_solve_execute"`、robot和execution，GT-only阶段带这两个字段会拒绝，避免默默执行。

示例见 `configs/scene_assembly_tasks/panda_pick_scene65.json` 与 `panda_place_scene60.json`。抓起动作PickLiftSelectedObject按正确物体抬升和持稳评分；放置动作PickPlaceObject要求真实抓取、运输、松爪、退离、桌面支持和末态稳定。每个worker最多420秒，错误物体/偏移目标对照保留。ReachAboveTarget不能代替这两种自然操作。

结果增加robot_execution，task_action_physics_executed与verified独立标记；只有正例、所请求对照和独立原始状态复核都通过才标为task_action_physics_passed。原GT configs和初始落稳证据保持只读，实际机器人合同在robot_configs、动作证据在robot_physics；运行时GT与机器人加载前GT分别保存和展示。

墙体展示可在组装 manifest 中显式设置 `"visualization": {"walls": "visible"}`，初始场景预览与机器人录像共用该策略；默认保持隐藏墙体。该选项只控制可见性，材质须实际组合在所加载 USD 中。两份新墙体的实际复测见[CompleteSquare 记录](research/experiments/2026-11-10-complete-square-wall-retest.md)。

Python 执行接口 `execute_robot_stage(..., isolate_negative_controls=True)` 可将正例与负控放在独立新 PhysX 进程中，保留每个 worker 的原始报告，再对合并索引执行同一严格审计。底层 `run_episode_physics --negative-only` 仅运行负控，不先执行正例；它不替代独立动作验收。这是显式 Python 选项，不是新增请求 JSON 字段。

需要显式采用运输型错误落点时，可传 `negative_placement_policy='offsets_first'`，底层对应 `--negative-placement-policy offsets_first`：优先检查 GT 周边原有有界偏移，再考虑物体初始位置。默认仍为 `initial_first`，所有桌面支撑、工作区与物体间距检查保持一致，完整候选和顺序写入报告。静态候选合法仍需实际搬放和独立负控验收。[机械臂居中实例](research/experiments/2026-11-10-complete-square-centered-robot.md)保留了原位回放失败及随后偏移对照的独立结果。

## Agent入口

查询实际可选项：

```bash
python -m geomind.tools.research_pipeline --assembly-options
```

返回102个空桌面场景及名称、28种可信尺度描述资产、5种布局、当前接通的4类静态GT及参数要求。菜单同时说明排除的资产和各任务的布局约束。规划agent读取菜单后，提交JSON选择；CLI自身不调用LLM，也不把固定规则伪称模型选择。

运行两个不同场景和布局的完整示例：

```bash
python -m geomind.tools.research_pipeline \
  --assembled-request configs/scene_assembly_tasks/midpoint_scene60_grid.json

python -m geomind.tools.research_pipeline \
  --assembled-request configs/scene_assembly_tasks/order_scene65_line.json
```

同一已落稳场景可以换任务重新求GT：

```bash
python -m geomind.tools.research_pipeline \
  --assembled-request configs/scene_assembly_tasks/projection_reuse_scene60_grid.json
```

每次输出必须使用新目录。示例已经执行；再次运行请更改请求的output，或使用展示页保存的独立replay请求。原请求、旧结果和失败证据不会覆盖。

## 请求协议

```json
{
  "schema_version": 1,
  "stage": "assemble_and_solve_gt",
  "output": "outputs/research/my_assembled_midpoint",
  "timeout_s": 420,
  "assembly": {
    "scene_id": 60,
    "object": {"asset": "blue_can", "count": 4},
    "layout": {"type": "grid", "rows": 2, "cols": 2},
    "seed": 23,
    "randomization": {"enabled": true}
  },
  "task": {
    "task_id": "InferMidpoint",
    "parameters": {"endpoints": ["blue_can_001", "blue_can_004"]},
    "operation": {"type": "place", "moving_object": "blue_can_002"}
  },
  "selection": {
    "actor": "agent",
    "reason": "选择网格对角物体作为参考，在中间保留放置区域。",
    "considered_scene_ids": [60, 65],
    "considered_layouts": ["grid", "line", "random"]
  }
}
```

`assembly`复用[场景组装请求](scene_assembly_zh.md)，完整支持其场景、数量、间距和随机化参数。也可用 `assembly_manifest` 替代assembly，读取已有且证据完整的落稳场景；两者只能提供一个。timeout为每个物理worker的上限，最多420秒，GPU串行。

selection必须记录actor、理由和考虑过的场景/布局；实际选择必须在记录中。未知字段、资产、任务、退化几何、排序并列、不合法目标和不一致来源均拒绝，保留失败，不自动换seed或降级。

## 已接通的任务

| task_id | parameters | 自然操作意图 | 关键约束 |
|---|---|---|---|
| SelectByOrder | index、from=Left/Right、candidates；tie_tolerance_m默认0.01 | 抓起从左/右数的指定物体 | 仅世界X排序；建议line axis=x，不能把微小网格扰动当稳定顺序 |
| InferMidpoint | endpoints，两物体ID | 把另一物体放到两参考中心正中间 | 被搬物体与端点分开；目标需要支持与空隙 |
| ProjectOntoLine | point、line=[两端点] | 把另一物体放到参考点的正交投影处 | 原生无限三维直线；本入口被搬物体与全部参考分开 |
| ReflectAcrossLine | point、line=[两端点] | 把另一物体放到绕参考线旋转180度的位置 | 原生三维轴反射，不泛化为任意二维镜像 |

后三项operation须为place并指定moving_object；Order须为pick。frame固定world、单位米；point任务geometry为InfiniteLine3D，退化容差默认0.1毫米。原生Point答案语义验收容差为1毫米；独立GT交叉计算误差另要求1e-10米。它们都不是机器人运动的物理误差阈值。

上表描述原生静态请求入口。下面的注册任务批量入口复用现有注册信息和旧任务求解器，接入异构资产及时间过程；未知注册项仍明确拒绝。

## 重新生成20个注册任务

2026-11-10已完成本轮20/20实物抓放、20/20运行时GT独立复算及错误目标对照。查看[统一Robotiq交付页](http://127.0.0.1:8780/registered-robotiq/delivery.html)与[完整实验记录](research/experiments/2026-11-10-registered-robotiq-regeneration.md)。结果按每例保存版本验收，历史失败保留，不代表全部在最新控制器同批重执行。

```bash
python -m geomind.tools.regenerate_registered_tasks \
  --output outputs/research/registered_robotiq_newrun \
  --scene-id 65 --seed 0 --max-steps 12000 \
  --robot-base -0.12 0.32 0.9328504589976385
```

输出目录必须是新目录。可用 `--tasks TASK_ID ...` 选择已注册实例，避免覆盖先前结果。上述命令统一使用场景65；agent应按物体支撑、间隙和可达性选择合适的实际场景与布局，并保留选择原因。球体在该桌面微倾斜下可能滚动；水平桌面场景91的安装基座为 `--robot-base -0.03 0.40 0.7062299448967717`。

`--layout-scale 2` 在原seed随机布局之后统一放大XY间距，物体尺寸、材质、Z和seed保持原值，随机化不再调用。记录缩放中心、必要的最小整体平移和每体前后位置；越过安全区域则拒绝。容器拼装件不支持这种缩放。导出几何检查通过后，仍需重新物理落稳、求解实测GT并执行动作，不能沿用缩放前的答案。

这20项来自当前注册表的19个Base实例和没有Base父项的SelectByDistance（保留其Semantic条件）。First、Last、Second分别保留不同契约；不包含退役的PlaceDirectional或未实现占位项。采用真实场景65、原结构随机化、原始形状/颜色/材质和`franka_robotiq`；原场景编号同时保留在来源中。

流程为注册协议编译→异构USD组装→初始PhysX落稳→实测绑定→机器人运行时GT→Robotiq抓放→原始状态独立复核。7项重排/历史任务保留原motion function的路径，经有界PD场景事件执行；查询只使用实际Observe.End和事件后的实测状态，计划端点不充当GT。追踪任务连续展示事件。事件结束后撤除外力，再执行接触抓放。

OnTop/Inside保留原支撑/容纳判据，包括松爪、稳定、支撑间隙与投影重叠，或容器内边界、底面和开口净空。其他放置目标仍使用桌面支持高度。原“手中物体”改为从桌面抓取，原协议保留在私有来源；不预先附着物体。GT求解和动作通过分别计数，失败和未执行项保留；每个物理worker最长420秒，GPU串行。结果以实际批次报告为准，不因接入接口就视为20项物理通过。

Robotiq闭链接触采用240Hz机器人执行时间步；抓取、运输、事件和落稳的物理时长保持原设置，连续验收至少0.5秒。夹持控制仅写关节目标；抬起后的保持模型是基于质量和关节弹簧的估计，不是测力反馈。物体状态、参考物体保持、速度和关节限制仍由原始轨迹独立验收。球体、密集布局或动作失败均保留，不通过更换seed补齐数量。

多个批次可用 `python -m geomind.tools.deliver_registered_tasks --batch BATCH --retry RETRY1 --retry RETRY2` 汇总。每个任务选择最后一个明确列出的版本，失败的新版本也不会回退到旧成功；用保存的原执行源码隔离重放GT，另核对原始状态、控制和保存配置。交付网页为 `BATCH/delivery.html`，原报告和各次失败录像保持原样。

## 实测绑定和输入隔离

组装完成后，严格检查manifest、物理报告、离线落稳复核、USD、点云及原始状态的哈希。新组装保存刚体局部网格点云原始hash及引用USD层hash；缺少点云原始绑定的历史manifest需重新生成，不补写历史证据冒充原始记录。

查询读取 `object_states.npz` 的第599行、20步预热后的第600个测量步，即物理tick620。按prim_paths匹配物体，用实测姿态变换原始网格点，取世界轴对齐包围盒中心作为语义参考点。此处的GeometricCenter特指该AABB中心，不是质量中心或网格体积质心。配置使用world frame和零二次jitter，直接调用原EpisodeResolver、Snapshot、solve和evaluate_answer，不重建场景，不通过SemanticEpisode.reset生成替身状态。

请求的角色引用使用内部资产编号，如blue_can_001；绑定后为执行agent生成独立置换的obj_001等语义编号和独立乱序的对象列表。映射仅在评估侧保存，同一场景跨任务保持一致；Order候选文字也不按摆放顺序列出。这消除明显的编号/列表顺序编码，不构成全面反捷径或模型能力验证。

公开输入位于 `observations/initial.json`，包含自然操作指令、实测对象及位置、world轴约定和图像引用。它是结构化状态加图像的输入，图像上的物体没有额外编号贴纸；不能报告为纯视觉任务。GT、参数角色、内部prim路径、身份映射和来源hash保存在evaluator，不放进公开观察。

放置GT另外用实际桌面网格检查9点支持、站立高度、保留家具AABB及至少5毫米保守物体包络间隙。此检查不证明机器人可达、抓取成功或松爪后稳定。

## 产物、复核与展示

每个输出保存request/raw_request、selection、启动前源码快照、configs原生任务契约、evaluator查询快照/GT/独立交叉/来源/目标几何、公开observations、task_result及result。初始布局与录像位于assembly目录，复用已有场景时保留原路径和hash。

```bash
python -m geomind.tools.research_pipeline \
  --audit-assembled outputs/research/assembled_tasks_v2/midpoint_scene60_grid
```

该命令离线重读原始实测状态、复算GT和目标几何、拒绝错误答案并核对公开观察与来源；不重新执行物理。旧 `--audit` 仍用于旧任务物理流水线，新入口使用明确的 `--audit-assembled`。

[本地任务页](http://127.0.0.1:8780/tasks.html)显示agent选择依据、实际场景、初始落稳录像与GT俯视图。GT区明确标为研究者特权视图，不能当受评agent输入。页面分别显示GT已求解、初始布局已物理落稳、任务动作未执行及模型未评估。实验记录见[场景任务结合验证](research/experiments/2026-11-09-assembled-task-generation.md)。
