# Episode v3：自动生成、参数读取与语义 GT 契约

## 状态和边界

59 份现有注册配置已转换为精简 v3 episode；27 个任务定义放在 `configs/task_definitions/`。
场景、布局、任务参数分别读取。旧 IsaacLab 协议记录隔离到 `configs/legacy/registrations/`，
由 `core/episode_compat.py` 转换，旧运行器与别名继续可用。新配置不填写 canonical/protocol/legacy ID。

原生语义实现支持 SpatialReasoning 的 `SelectByOrder`、`InferMidpoint`、`ProjectOntoLine`、
`ReflectAcrossLine`，以及 SpatialMemory 的 `DetectSceneChange`、`RecallObjectLocation`。
最后一个是额外示例，尚未加入 27 任务分类。其余任务继续使用已隔离的旧实现或明确保持未实现状态。

新增 `SemanticEpisode` CPU 后端，执行状态初始化、观察与延迟阶段、移动事件、历史快照、观测过滤、
查询和答案评分。通过 `create_env(..., backend="semantic")` 或 `create_episode_env(...)` 访问。
三个新任务现已实现无头原生 PhysX/Franka 执行；`registry.capabilities()` 分别报告语义、旧 IsaacLab、
原生 episode IsaacLab 能力。原有兼容注册项继续保持旧入口的占位状态，不再代表原生后端没有实现。
实际物理验证见 [无头物理执行报告](headless_physics_verification_zh.md)。

早期 `episode_v3_examples/` 中的场景与资产名称用于展示数据契约。
本轮物理验证覆盖正式配置中的程序化桌面与三个新任务；其他 USD 资产、相机可见性和任务可达性仍需各自验证。
配置校验成功不等于任意场景可运行或任务可解。

## 配置来源与唯一读取路径

正式配置根目录为 `geomind/configs/`；早期三个示例保留在独立的 `episode_v3_examples/` 中。

| 文件 | 唯一职责 | 读取者 |
|---|---|---|
| `task_definitions/<task_id>.json` | 任务名称、domain、family | resolver 校验任务注册 |
| `scenes/<scene_id>.json` | 环境、坐标约定、工作空间、相机 | 后端创建环境；任务读取坐标语义 |
| `layouts/<layout_id>.json` | 物体身份、资产、初始坐标、随机化 | 后端实例化对象 |
| `protocols/<protocol_id>.json` | 观察、保持、回忆阶段及事件 | 记忆 episode 校验与阶段控制器执行 |
| `episodes/<episode_id>.json` | 引用以上文件，填写任务参数、响应类型和 seed | 自动生成器输出、resolver 读取 |
| `tasks/`、`vla_tasks/` | 旧脚本所引用的注册实例，内容已为 v3 | 兼容注册器，不是纯任务定义 |
| `legacy/registrations/` | 原 v2 执行记录 | 旧 IsaacLab 适配器 |
| `legacy/layouts/` | 历史协议与标注索引 | 旧协议加载器 |
| `suites/`、`reports/`、`evidence/` | 自动生成清单、独立验证报告、状态证据 | 验证工具 |

原生解析器只从 `episodes/` 读取实例。兼容注册器从 `tasks/`、`vla_tasks/` 读取实例；
`execution: LegacyIsaacLab / Unavailable` 是兼容实例的明确标记。两类入口不会猜测或互相回退。
旧初始化布局带 `format: LegacyInitialization`；原生布局包含显式对象、坐标和随机化配置。
旧布局的物理初始化覆盖由 `load_annotated_tasks` 读取，协议、角色和 GT 保留在兼容标注中。

引用按文件名和内容 ID 双重校验。旧 scene_N 文件的数字 ID 由兼容适配器显式转换；原生 ID 完全匹配。
所有任务参数均显式写入 episode，不从 scene、layout、历史标注或代码默认值补齐。
未知字段、缺失字段、重复 JSON key、未知物体、错误类型和未实现配置均报 `ContractError`。
同一字段没有覆盖优先级；没有从 instruction 反向解析参数的步骤。

```text
生成器输出 episode 及其引用文件
                  ↓
EpisodeResolver.resolve：加载 → 校验 → 配置/实现哈希
                  ↓
后端：加载 scene → 实例化 layout → 随机化 → 物理稳定
                  ↓
提供实际 Snapshot；记忆协议在指定阶段边界额外保存 Snapshot
                  ↓
solve：校验快照归属 → 选择证据状态 → 同时生成指令和语义 GT
                  ↓
运行器按查询阶段发布指令；独立评估端保存 GT 和 provenance
```

## 空间推理参数

`SelectByOrder` 必填：`index`、`from`、`frame`、`candidates`、`tie_tolerance_m`。

- `index` 是从 1 开始的整数，不能超出候选集合长度。
- 候选集合必须显式列举，不能自动包含布局新增的干扰物。
- 当前实现限定同一工作空间坐标系：Right=+x、Forward=+y、Up=+z。Left 从 x 小到大，Right 相反。
- 所有坐标单位为米；当前不支持相机投影、左右视角转换或不同 frame 混用。
- 按查询快照中的实际物体位置排序。任意相邻候选的 x 差小于或等于 `tie_tolerance_m`，整例拒绝。
  不按 ID 或文件顺序打破并列；生成器可以更换布局或 seed 后重新验证。
- 输出语义答案是稳定 `object_id`，而非文件中的初始坐标。

`InferMidpoint` 必填：`endpoints`、`frame`。两个端点必须是不同物体，实际位置不能重合。
求解定义为两个物体中心的三维算术中点。

`ProjectOntoLine`、`ReflectAcrossLine` 必填：`point`、`line`、`frame`、`geometry`、`degeneracy_tolerance_m`。
point 和 line 中的两个端点必须是三个不同物体；实际端点距离不能小于或等于退化容差。
`geometry` 目前必须是 `InfiniteLine3D`，投影不限制在线段内。反射明确定义为绕该轴旋转 180 度，
结果为 `2 * projection - point`，不使用未指定法向的平面反射。

SpatialReasoning episode 不接受时序协议。后端必须在查询时固定任务世界状态，执行阶段另外处理。

## 空间记忆参数与时间边界

`RecallObjectLocation` 必填：`objects`、`query_object`、`frame`、`reference`。
`objects` 为需要编码的物体集合，`query_object` 必须在集合中。

```json
{
  "reference": {
    "phase": "observe",
    "boundary": "End"
  }
}
```

当前接受 Observe → Delay → Respond 三阶段协议：观察可见，延迟隐藏，查询在 Respond 发布，
不允许历史回放。时长为正整数控制步。位置回忆在 Respond 隐藏场景；变化检测在 Respond 展示当前场景。

`DetectSceneChange` 必填：`objects`、`frame`、`reference`、`change_threshold_m`、`ambiguity_tolerance_m`。
计算各物体中心相对参考快照的三维位移，恰好一个物体超过变化阈值才有唯一答案。
零个、多个变化目标，或任何物体落入阈值正负 ambiguity_tolerance_m 的区间，均拒绝本实例。

Delay 阶段可配置 `events`：`MoveObject` 必须显式给出 `object_id`、`after_steps`、`position_m`。
对象必须存在，目标必须在工作空间内，同一步不能给同一物体两个事件。
CPU 后端在指定步更新符号状态；物理后端在该边界启动外力 PD 干预，通过连续 PhysX 步推进物体。
物理干预不写入目标物体位姿，最终 GT 使用真实物理测量结果。

后端须在物理稳定并开始协议时将 episode 控制步归零：

- Observe 持续 N 步，完成第 N 步后捕获 `observe.End`。
- Delay 持续 M 步，Respond 最早在 N+M 步开始。
- 当前快照和历史快照必须具有相同 `instance_id`、配置哈希及坐标系。
- 不存在指定快照、保存时刻错误、跨运行拼接或尚未完成延迟，均拒绝求解。
- 位置回忆 GT 使用历史位置。未来的身份追踪任务应输出历史确定的身份，执行适配器再读取该身份的当前位置。
  不能把位置回忆的历史坐标直接复用为身份追踪的当前动作目标。

CPU 阶段控制器隐藏场景时返回空 objects，直到 Respond 才发布查询，并在 reset 时清空历史、重新创建实例 ID。
agent 观测不包含 GT、历史快照、事件计划或 provenance。原生 IsaacLab 后端已接入相同阶段控制和过滤，
额外公开机器人关节与 TCP 的当前本体状态。
语义后端不托管 agent 的内部缓存；调用方必须在 episode reset 时同步重置 agent 记忆。
StreamingOnly 指环境不提供历史回放，agent 可以在同一 episode 内维护自己的记忆。

## 实际状态接口

```python
from geomind.episodes import EpisodeResolver, Snapshot, solve

resolved = EpisodeResolver("geomind/configs/episode_v3_examples").resolve("select_second")

# 以下由后端提供，不能直接复制 layout.position_m 冒充实际位置。
snapshot = Snapshot.from_dict({
    "instance_id": "run_0001",
    "contract_hash": resolved.contract_hash,
    "frame": "table",
    "step": 0,
    "positions_m": measured_object_positions,
})
result = solve(resolved, snapshot)
```

快照必须包含布局中的全部对象，位置为物体中心的有限三维坐标；机器人和背景资产不在这个对象集合中。
物理后端需要把根位姿转换为约定的物体中心，不能未经检查混用两种坐标。
稳定 object_id 是评估用身份句柄，必须由后端维持，不能根据颜色或当前位置重新生成。
输入快照的真实性由采集后端负责，哈希只能检查数据归属和一致性。

记忆求解示例：

```python
result = solve(resolved_memory, current_snapshot,
               history={"observe.End": observed_snapshot})
```

新生成布局含唯一 `label`，同一标签同时用于公开观测和指令；未提供标签的早期示例使用逻辑 ID。
渲染图像时仍需验证标签与实际视觉外观一致，不能以标签唯一替代视觉可辨识性验证。
不得向 agent 暴露隐藏身份、历史快照、GT 或 provenance。

## 自动生成的验收阶段

1. 生成配置。按公开 task_id 选择实现，填写所有参数；不生成旧 canonical/protocol/legacy ID。
2. 静态校验。解析引用、对象集合、frame、参数类型、协议阶段；计算规范化 JSON 和当前求解实现的 SHA-256。
3. 实例化验证。运行后端检查资产、工作空间、碰撞、可见性与任务能力，保存 seed 和实际状态。
4. GT 校验。调用 solve；不唯一或证据缺失则拒绝本实例。保留明确的拒绝原因；重采样须有次数上限。
5. 可解性校验。确认可见线索足以确定答案；记忆任务确认观察阶段确实提供了所需信息。
6. 冻结实例。保存原配置、源文件、实现版本、实际状态/历史、生成器版本、指令、GT 和诊断结果。

生成器覆盖配置生成、静态校验、CPU 世界实例化、事件/观测流程、独立构造答案比对、错误答案拒绝和证据回放。
语义报告标记 `backend: semantic`、`physical_execution_verified: false`。
物理报告另外保存，所选真实动作和启用的错误目标对照全部通过才标记 `physical_execution_verified: true`。
seed 只是复现输入，不能替代实际状态存档，也不能承诺跨仿真版本逐位一致。

GT 输出包含配置/实现哈希、证据快照哈希、查询快照哈希、实例 ID、证据步数、实际参数与求解过程。
哈希基于 JSON 内容而非排版。修改 seed、引用文件或求解代码后，旧快照不能继续用于新契约。

语义 GT 与物理成功分开：SelectObject 返回目标身份，ReportPoint 返回坐标。
触达、抓取、放置应另外声明动作及容差，使用与任务语义匹配的评估器。
语义选择按 object_id 精确评估；点回答必须声明 `evaluation.position_tolerance_m`，并使用相同坐标系。
物理动作由 `physical_execution` 显式指定，和语义评分分别计算，最终成功要求两者均通过。

## 无头物理配置

场景的 `physics` 完整声明 dt、协议步长、工作空间到世界的平移、机器人基座位置、桌面和物体尺寸、
质量、落地稳定时长、落地速度容差以及干预 PD 参数。当前支持 `scene_mode: ProceduralTable`。
这一模式明确创建程序化桌面与彩色刚体；不加载 scene_65 的完整 USD 背景。
机器人使用 IsaacLab 标准 Franka 高 PD 配置，关节受真实物理驱动，机器人本体采用关闭重力的控制模型；
任务刚体启用重力、质量与碰撞。

episode 的 `physical_execution` 声明：

| 字段 | 当前用途 |
|---|---|
| `action` | 当前为 ReachAboveTarget |
| `target_offset_m` | 从语义目标到 TCP 目标的显式偏移，生成配置为 `[0,0,0.10]` |
| `position_tolerance_m` | 实际 TCP 对正确目标的距离阈值，当前 0.012 m |
| `orientation_tolerance_rad` | 末端向下姿态误差阈值，当前 0.15 rad |
| `dwell_steps` | 连续满足物理误差的步数，当前 20 |
| `max_steps` | 单次机器人响应的总物理步数上限，当前 1800 |
| `joint_step_limit_rad` | IK 每步关节目标增量限制 |
| `approach_clearance_m` | 抬升、水平接近、下降路径的高度 |

真实 TCP 根据 panda_hand 与标准 0.107 m 工具偏移计算。动作通过阻尼 IK 和关节位置驱动执行；
只有 reset 初始化时写入关节状态，动作过程中不直接改关节状态或物体位姿。
位置回答按查询时的测量状态求解；身份回答的物理目标绑定到正确物体的当前实际位置。

memory 使用无初始位置标记的平坦桌面，初始和移动后位置从同一范围采样；
几何任务使用辅助支撑保留三维点高度。辅助碰撞体不进入任务对象或 agent 观测。

协议一步在物理后端为 0.5 s、60 个 PhysX 步，dt 为 1/120 s。
`physical_clock` 记录历史边界和查询实际对应的物理 tick，便于和轨迹对齐。

## 命令行

只校验配置，不计算 GT：

```bash
python -m geomind.tools.resolve_episode select_second \
  --root geomind/configs/episode_v3_examples
```

使用后端导出的实际状态求解，状态文件格式为 `{"current": <snapshot>, "history": {}}`，
记忆任务的 history 必须包含明确的阶段边界键：

```bash
python -m geomind.tools.resolve_episode select_second \
  --root geomind/configs/episode_v3_examples \
  --state /path/to/measured_state.json
```

无状态文件时输出 `ground_truth_computed: false`。错误退出码为 2。

自动生成三个已注册占位任务，每个 seed 各一个实例，并验证：

```bash
python -m geomind.tools.generate_spatial_suite --seeds 0 1 2 --verify
```

生成器使用随机方向、平移和独立几何构造产生预期答案，不用 solve 的输出作为正确性 oracle。
生成的正式配置位于 `configs/episodes/`，报告为 `configs/reports/generated_spatial_suite.json`。
报告保留每例指令、GT、源文件/实现哈希与求解证据。`evidence/` 中的文件可以直接传给 resolve_episode 的 --state。

```python
from geomind import create_env

env = create_env("DetectSceneChange", backend="semantic")
observation, info = env.reset()
while not observation["allow_response"]:
    observation, reward, terminated, truncated, info = env.step()
# agent 只接收 observation，回答为 {"object_id": ...}。
observation, reward, terminated, truncated, info = env.step(agent_answer)
```

单独创建一个实例，先写参数 JSON 文件，然后使用新脚手架；只有校验通过才写入正式目录：

```bash
python -m geomind.tools.scaffold_task \
  --episode-id my_projection --task-id ProjectOntoLine \
  --scene-id semantic_table --layout-id geometry_s0 \
  --parameters-file /path/to/parameters.json \
  --response ReportPoint --position-tolerance-m 0.001 --seed 0
```

旧脚手架参数必须显式使用 --legacy，默认只输出到兼容记录目录。
新脚手架可用 `--physical-execution-file` 显式加入物理动作配置。

无头物理执行（使用已验证的 new_isaaclab 环境）：

```bash
conda activate new_isaaclab
python -m geomind.tools.run_episode_physics --negative-controls
```

单例可传 `--episode ProjectOntoLine_v0`。完整命令、动作定义与实测结果见 [物理报告](headless_physics_verification_zh.md)。

## 后续接入顺序

1. 将程序化桌面的物理适配扩展到完整 USD 场景，验证相机与视觉指代。
2. 在已实现的机器人到达动作上扩展抓取和放置，单独定义成功条件。
3. 扩展动态记忆任务、事件记录和其他空间推理任务；每个 task_id 明确验证输入、答案唯一性及可解性。
4. 将余下旧协议逐项改为原生任务实现，通过语义与物理等价性验证后退役兼容记录。
