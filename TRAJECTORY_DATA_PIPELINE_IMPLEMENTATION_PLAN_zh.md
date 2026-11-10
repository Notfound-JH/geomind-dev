# 轨迹数据管线：实施顺序与验收标准

## 1. 最终目标

采集的数据单位是“**随机布局上的独立随机轨迹**”，而不是固定布局上的重复轨迹。

设：

- $L$：每个任务的随机布局数。
- $K$：每个随机布局下采集的随机轨迹数。
- 总 episode 数：$N = L \times K$。

推荐起始规模：每个任务 $L=10$、$K=5$，即 50 条 episode；流程验证通过后再扩大。

```text
任务 task
  └─ layout_index = 0 ... L-1
       ├─ layout_jitter_seed_i
       ├─ layout_shift_seed_i
       ├─ 重载任务并创建环境，固定该随机布局
       └─ rollout_index = 0 ... K-1
            ├─ trajectory_seed_i_j
            ├─ 生成、校验并执行一条随机化机器人轨迹
            └─ 导出 10Hz H5 + 所有相机 512x512 RGB/MP4
```

布局随机化与轨迹随机化必须独立：

| 随机层 | 种子 | 影响内容 | 同布局内是否变化 |
|---|---|---|---|
| 局部布局 | `layout_jitter_seed` | 物体相对位置、小偏移 | 否 |
| 整体布局 | `layout_shift_seed` | 全部相关对象/容器整体平移 | 否 |
| 轨迹 | `trajectory_seed` | 可交换路径段、waypoint、执行时长 | 是 |

布局随机化开关在正式采集时必须开启：

```bash
GEOMIND_LAYOUT_JITTER_ENABLE=1
GEOMIND_LAYOUT_JITTER_SEED=<layout_jitter_seed>
GEOMIND_LAYOUT_SHIFT_ENABLE=1
GEOMIND_LAYOUT_SHIFT_SEED=<layout_shift_seed>
```

同一个 `layout_index` 下的 $K$ 条轨迹必须使用同一对 layout seed；每条轨迹仅更换 `trajectory_seed`。要进入下一个 layout，必须重建环境或重新加载 task spec 并重建场景，不能仅调用 `env.reset()`。

---

## 2. 分阶段实施顺序

### 当前实现状态（2026-08-10）

| Phase | 当前状态 | 已实现范围 | 尚未完成的验收 |
|---|---|---|---|
| 0 | 基本完成 | 原子标签 v1、可复用 trajectory 数据层、H5 schema 代码与三层 seed 约定 | 在安装 `.[trajectory]` 的 Isaac 环境读回真实 H5 |
| 1 | 实现完成，待端到端验收 | `anchor_reference` 的真实时间、目标/实际 EEF、关节/夹爪、10Hz 重采样、JSON/H5 写出 | 真实 Isaac/H5 单 episode 验收 |
| 2 | 代码实现完成，待真实环境验收 | 每个 layout 独立进程、同 layout 内 $K$ 次 reset/rollout、episode 级 manifest、Phase2 自动检查报告 | 真实 $L=2,K=2$ 布局独立性与成功率检查 |
| 3 | 实现完成，待端到端验收 | 512x512 相机 sidecar 流、H5 RGB、掩码、MP4 导出 | 多相机帧形状、时间对齐和内存曲线检查 |
| 4 | v2 语义修复完成，待真实轨迹验收 | 默认 `stage_v1` 保持不变；`anchor_reference`、`spatial_placement`、`geometric_inference` 可选接入 `event_kinematics_v2`、抓取/释放事件、夹爪状态阈值、运动学优先分类、短碎片归并、H5 边界原因与 Phase4 验收脚本 | 真实轨迹边界人工一致率、抓取/释放事件完整性与多类任务验收 |
| 5 | 三维增量算法已实现，待真实环境验收 | `anchor_reference`、`geometric_inference`、`spatial_placement` 的无接触 `approach` 均可将起终点位移分解为有符号 3D 增量；`geometric_inference`、`spatial_placement` 的放置前 `transit_xy` 也复用同一方法生成 `placement_transit_xy`，接触下降、开爪释放和 settle 仍冻结；候选全部经 runtime workspace、IK-relative action、速度、段长、endpoint error 与非目标对象点间距校验后选择；H5 metadata 写增量、排列、拒因和执行路径 | 全机械臂/被持物 mesh 碰撞和预解 IK validator；三个 family 的真实固定 layout 多样性/成功率复验；迁移到其它 protocol family |
| 6 | 三个 H5 protocol adapter 已接入，待真实环境验收 | L×K runner 新增 `--protocol anchor_reference|geometric_inference|spatial_placement`，保留 `--resume`、dataset-level episode index、质量报告、唯一 key 和 generator metadata；默认仍使用 anchor-reference | 在真实 Isaac 环境验收三个 protocol 的断点续采、H5 数据质量和 layout/rollout 独立性；再接入尚无 H5 契约的其它 protocol family |

本轮所有采集功能均为显式开关；未指定 `--trajectory-h5-out` 时，`EpisodeCollector`、相机 sidecar 和 H5 写出不会创建，不改变既有 rollout 控制逻辑或布局随机化逻辑。

### Phase 0：冻结数据契约和可复现约定

**实现内容**

1. 固定原子动作标签 v1：`IDLE_HOLD`、`REACH_ABOVE`、`DESCEND_TO_GRASP`、`GRIPPER_CLOSE`、`LIFT_OBJECT`、`TRANSIT_XY`、`ALIGN_POSE`、`DESCEND_TO_PLACE`、`GRIPPER_OPEN`、`RETREAT`。
2. 固定 HDF5 schema v1：10Hz 主时间轴、绝对/相对 EEF 动作、实际 EEF 位姿、夹爪、关节状态、关节速度、标签、分段表、相机 RGB 和元数据。
3. 固定三层种子定义与派生方式。
4. 确定 episode 输出路径和 manifest 格式。

**验收标准**

- HDF5 字段、dtype、shape、原子动作 ID 在实现前写入代码常量和测试样例。
- 同一组 `(task_id, layout seeds, trajectory seed)` 能生成唯一且稳定的输出目录。
- 元数据必含：`layout_index`、`rollout_index`、两类 layout seed、`trajectory_seed`、`layout_randomization_record`、`augmentation_record`。

### 2.1 当前 H5 数据契约补充

所有采样时间都使用同一个 reset-relative 仿真时间源：

```text
timestamp_s = episode_length_buf[0] * env.step_dt
```

因此逐步 JSON 与 H5 同时保存：

```text
/steps/index                 int32 [T]
/steps/source_index          int32 [T]  # 原始控制步记录下标
/steps/sim_step_index        int32 [T]  # 环境 episode step 计数
/steps/timestamp_s           float64 [T]
/actions/eef_abs             float32 [T,7]  # 控制目标，xyzw
/observations/eef_pose_world float32 [T,7]  # 实际执行位姿，xyzw
/actions/eef_delta_rel       float32 [T,7]
/events/object_grasped       bool [T]
/events/grasp_transition     int8 [T]   # 1=grasp, -1=release, 0=none
/events/raw_grasp_transition_*              # 原始控制步事件审计流
```

H5 导出频率由三个协议共用的 `--trajectory-sample-hz` 参数控制，默认值为 `10.0` Hz；该值同时写入 `metadata/sample_hz` 和 `metadata/sample_frequency_hz`。每个 H5 还必须保存以下可序列化任务事实：

```text
metadata/action_scale                         # 运行时 arm_action.scale（当前 Franka IK-relative 为 0.5）
metadata/object_info                          # object_id -> {object_id, position, orientation_xyzw}
metadata/ground_truth                         # {gt_spec, goal_sequence}
```

`object_info` 来自 reset、布局恢复和 settle 后的运行时世界位姿，而不是仅使用标注中的二维初始坐标；这使其能够准确描述布局随机化后的 episode 起始状态。

`event_kinematics_v2` 会将 raw grasp/release 事件投影到 10Hz 主时间轴。若多个事件落入同一个采样区间，不能用后写覆盖前写；当前实现会按事件顺序向后寻找可用采样点，并在 metadata 中写入：

```text
metadata/segmenter_diagnostics.event_projection.raw_event_count
metadata/segmenter_diagnostics.event_projection.projected_event_count
metadata/segmenter_diagnostics.event_projection.collision_count
metadata/segmenter_diagnostics.event_projection.spill_count
metadata/segmenter_diagnostics.event_projection.dropped_event_count
```

Phase4 验收时 `dropped_event_count` 必须为 0；否则说明主 10Hz 序列无法完整表达抓取/释放硬边界顺序，只能依赖 raw 审计流补救。

这样可直接计算控制误差：

$$e_t = T^{\mathrm{actual}}_{\mathrm{eef},t} \ominus T^{\mathrm{command}}_{\mathrm{eef},t}$$

HDF5 写出依赖是可选 extra，而不是基础 benchmark 依赖：

```bash
python -m pip install -e '.[trajectory]'
```

---

### Phase 1：单任务、单布局的无增强采集闭环

**目标**

先在 `anchor_reference` 协议打通采集，不改变机器人控制行为，不做轨迹重排。

**实现内容**

1. 新增通用 `EpisodeCollector`，但不修改 `env.step()`。
2. 从 `scripts/debug_protocol_anchor_reference.py` 的 `_trace_step()` 在每次 `env.step(action)` 后记录：
   - 仿真时间戳与 step。
   - instruction、stage、目标 EEF 位姿、实际 EEF 位姿。
   - 绝对 EEF action、夹爪 command/状态。
   - `joint_pos`、`joint_vel`。
   - 抓取、终止、截断事件。
3. 用已有 stage 直接映射原子动作标签。
4. 将高频记录统一重采样到 10Hz。
5. 导出单 episode H5；相机在同一 10Hz 时钟上流式写入 sidecar H5，完成后生成每相机 512x512 MP4 和最终 H5 数据集。

**验收标准**

- 固定 `anchor_reference` 任务在不开布局随机化时，能生成一份可读的 H5 和每相机一个 MP4。
- H5 的 `timestamp_s` 严格递增，采样频率为 $10\pm0.1$ Hz。
- 包含 `step`、`source_index`、`sim_step_index`、instruction、`eef_abs`、实际 `eef_pose_world`、`gripper_cmd`、`joint_pos`、`joint_vel`、`gripper_state`、atomic label、segment id。
- `eef_delta_rel[0]` 全为 0，之后每项由相邻 `eef_abs` 正确计算。
- 标签全部在 v1 标签表中，连续同类标签正确合并为 segment。
- 新采集参数未启用时，现有 debug 协议的动作数量、任务成功判据和输出行为保持不变。

---

### Phase 2：单任务、随机布局的采集闭环

**目标**

验证布局随机化与采集器可以共存，并正确描述“一个布局、多条轨迹”的层级关系。

**实现内容**

1. 新增一个 layout/rollout 双层 runner：每个 layout 重建一次环境；同一环境采 $K$ 次 reset/rollout。
2. 每个 layout 将 `applied_layout_randomization` 和最终对象初始 XY 写入每条 H5 的元数据。
3. 把 `layout_index` 和 `rollout_index` 编入输出目录、H5 metadata 和总 manifest。
4. 初期设置 `trajectory_seed` 固定，仅验证布局层随机性。

**验收标准**

以至少 $L=2$、$K=2$ 跑一个任务：

- 两个 layout 的 `layout_jitter_seed` / `layout_shift_seed` 不同。
- 两个 layout 的 `layout_randomization_record` 或初始对象 XY 至少有一项不同。
- 同一 layout 的两条 episode 初始对象 XY 必须一致，且 rigid group 的相对坐标不变。
- 不同 layout 的 H5 都可读，所有必填字段齐全。
- 第一版先执行人工质检门：逐条检查 episode 的初始布局一致性、H5 字段完整性与轨迹可读性。
- 第二版再执行统计门：在完成首轮人工质检后，再引入“成功率不低于预先约定门槛”的硬阈值。

---

### Phase 3：H5 内嵌多相机 512x512 RGB

**目标**

将视频与 proprio/action 数据按 10Hz 同一时间轴对齐，保存为训练可直接读取的 H5。

**实现内容**

1. 复用现有 Camera sensor 枚举和 RGB 转 uint8 逻辑，采集所有有 `rgb` 输出的 Camera。
2. 每个采样时刻 resize 为 512x512，并保存到：
   `/cameras/<name>/rgb [T,512,512,3]`。
3. 同时保留 MP4，便于人类审查；H5 保存 `video_path`、相机帧时间戳和采样时间戳。
4. 加入流式/分块 HDF5 写出：仅在共享 10Hz 时钟采集相机帧，先写临时 sidecar H5，不在 Python/Torch list 累积控制频率 RGB。

**验收标准**

- 每个启用的 RGB Camera 都有 H5 dataset，shape 为 `[T,512,512,3]`、dtype 为 `uint8`。
- 每个相机 `timestamp_s` 与 `/steps/timestamp_s` 数量相同并已对齐。
- 任取 $t=0$、中点、末帧，H5 RGB 与 MP4 同时间附近画面一致。
- H5 读取单帧无需读入整段 RGB；长 episode 没有 GPU 内存持续增长。
- 每个相机保存 `valid_mask [T]` 与 `filled_mask [T]`；缺帧使用最近有效帧填充时必须保留这两个标记，不能静默错位。
- 相机从未捕获或首个采样点前无有效帧时，默认严格失败并报告 `never_captured` 或 `missing_initial_frame`；不允许静默省略。

---

### Phase 4：原子动作分段器 v2

**目标**

从“脚本 stage 映射”升级为“事件 + 运动学”联合分段，覆盖抓取和放置协议。

**实现内容**

1. 将夹爪开闭命令、夹爪关节阈值、抓取/释放事件作为硬边界。
2. 从 EEF 实际速度区分 XY 主运动、上升、下降、静止。
3. 将短于最小长度的碎片段合并到相邻同语义段。
4. 将分段器接入 `geometric_inference` 和 `spatial_placement`。
5. 输出 segment table 及每个边界的判定原因。

**当前实现补充**

- `event_kinematics_v2` 的优先级为：抓取/释放 raw event、`object_grasped` 状态变化、`gripper_state` 阈值、`gripper_command` 阈值、EEF 运动学分类、stage 回退。
- 运动学分类优先使用实际 `eef_pose_world`，用于区分 XY 主运动、上升、下降与静止；stage 不再直接压过常规运动学分类，只在运动学不足或首步缺少速度信息时兜底。
- 同一 10Hz 区间内出现多次 grasp/release 时，主序列按顺序前向安置事件，并记录 `collision_count`、`spill_count`、`dropped_event_count`。
- 已新增脚本 `scripts/collect_phase4_acceptance.sh`：采集 `L=2,K=2` 的 `anchor_reference` H5，并自动检查 Phase2、Phase3、`event_kinematics_v2`、事件顺序、无 dropped event、segment 范围和原子标签范围。

**验收标准**

- 在每类抓取/放置任务上抽查至少 20 条 episode：原子动作边界人工一致率 >= 95%。
- `GRIPPER_CLOSE` 与抓取事件顺序正确；`GRIPPER_OPEN` 与释放事件顺序正确。
- 没有零长度 segment；没有未定义标签；每个 step 恰好属于一个 segment。
- 新旧 stage 可用时，规则分段与 stage 映射总体一致率 >= 95%，差异要可解释。

---

### Phase 5：轨迹随机化 v1（仅自由空间段）

**目标**

在每个随机布局中采集多条不同、可执行、语义一致的机器人轨迹。

**实现内容**

1. 用 `trajectory_seed` 生成受限随机 waypoint 候选。
2. v1 仅允许 anchor-reference 的无接触 `REACH_ABOVE`（脚本 stage 为 `approach`）参与增强；`TRANSIT_XY`、`ALIGN_POSE` 留待后续 protocol 扩展。
3. 冻结 `DESCEND_TO_GRASP`、`GRIPPER_CLOSE`、`LIFT_OBJECT`、`DESCEND_TO_PLACE`、`GRIPPER_OPEN`。
4. 候选由起点到 canonical approach endpoint 的 `dx,dy,dz` 分解为多个有符号 3D 增量；每轴增量和严格等于总位移，完整增量向量共同随机排列后累计为 waypoint。直线路径只作为回退，不属于随机候选。
5. 每个候选在执行前做 runtime workspace、现有 IK-relative action 可转换性、按 `env.step_dt` 稠密化后的速度上限、非目标运行时对象 3D 点间距检查；失败记录所有拒因后尝试下一候选。
6. 所有候选均完成校验和审计；在通过者中按最大偏离优先、较短路径次之、稳定 candidate index 再次之选择。选中候选按受速度约束的 smoothstep 稠密目标实际 `env.step()` 执行；全部拒绝时调用原来的 30-step 直达 approach 控制，禁止事后篡改 H5。

**当前随机性分析（2026-08-06）**

目前 `trajectory_seed` 由 runner 派生、透传并写入 manifest/H5 metadata；启用 Phase5 增强时，即使未显式传入 `--vary-trajectory-seed`，同一 layout 的每个 rollout 也会使用独立 seed。`anchor_reference` 的 Phase5 执行层已经接入真实 `env.step()`，但随机化仍只覆盖无接触 `approach`，其后的 `descend`、`close`、`lift`、`hold` 仍锁定同一目标 XY。因此即使不同 seed 的 H5 waypoint 和 EEF 序列已经不同，整段视频仍可能表现为后半段相同；这是增强范围仍受限，而非 Phase4 分段器或 seed 透传失败。

本轮 anchor-reference 接入位于 `scripts/debug_protocol_anchor_reference.py`：启用 `TRAJECTORY_AUGMENTATION_ENABLE=1` 后，使用 `TRAJECTORY_AUGMENTATION_CANDIDATES`（默认 16）、`..._SEGMENTS`（默认 6）、`..._REVERSE_PROBABILITY`（默认 0.75）、`..._REVERSE_RATIO`（默认 0.30）、`..._MAX_REVERSE`（默认 0.08 m）、`..._MAX_SEGMENT_LENGTH`（默认 0.14 m）、`..._MIN_CLEARANCE` 与 `..._MAX_SPEED` 生成和校验候选。该默认值刻意提高自由空间随机性；若 fallback 或成功率恶化，应先减小反向预算，再降低段数。每条 episode 的 `augmentation_record` 含完整增量、排列、路径风格、endpoint error、段长、候选拒因、执行 waypoint 与 fallback；被实际增强的 `REACH_ABOVE` segment 在 H5 `/segments/is_augmented` 中标记为 1。

当前环境可调用的 `target_eef_pose_to_action()` 是 IK-relative controller command 的转换，不是离线全关节 IK 求解器；因此本轮 `ik_command_rejected` 只表示目标无法形成有限控制命令。全机械臂、末端工具和被持物的 swept-volume mesh 碰撞，以及独立 IK 解、关节速度/加速度约束，仍是后续 validator 的必做项，不能将本轮点间距检查解释为完整碰撞验证。

固定 layout 的最小验收通过 runner 的 `--verify-phase5` 完成：它从 H5 读取同 layout 的 `$K\ge5$` episode，检查初始对象配置一致、至少两条 `REACH_ABOVE` 的 `/actions/eef_abs` 序列不同、原子标签包含 `DESCEND_TO_GRASP -> GRIPPER_CLOSE -> LIFT_OBJECT`，以及 augmentation record 已落盘。2026-08-08 已完成真实 `L=1,K=5` 运行并通过；相机开启时同时验证了 5 个 episode、7 路相机的 MP4 目录结构。该门只证明 seed 已影响执行路径，不代表整段视频具有明显视觉差异。

**验收标准**

先对固定的一个 layout、$K=5$，再扩展到至少 $L=2$、每个 layout $K=5$：

- 同一 layout 内至少两条轨迹的 waypoint 序列或 `eef_abs` 序列不同。
- 每个 layout 至少半数 rollout 实际 `augmentation_enabled=true`，且至少有 3 个离散 approach path signature。
- 接受路径 endpoint error 必须不超过 `1mm`，且同 layout 至少有 3 个增量轴向签名；必须分开报告 fallback、workspace、IK-relative command、段长、endpoint 与 clearance 拒绝率。
- 同一 layout 内初始物体 XY 完全一致。
- 硬约束段顺序始终保持：抓取下降 -> 闭爪 -> 抬升；放置下降 -> 开爪 -> 撤离。
- 每条接受的增强轨迹都通过当前版本的 workspace、IK-relative command、速度与点间距校验；完整 IK/mesh 碰撞校验未实现前，不可作为大规模采集的最终安全门。
- 被拒绝的候选必须有 `fallback_reason`，并实际执行原轨迹。
- 增强后的有效执行率相对于未增强基线下降不超过 5 个百分点；若不足，先缩小增强幅度或候选集合。

### 2.5 最小改动 Patch Checklist（Phase 2 -> Phase 5）

约束：仅改任务书、runner 和 metadata；不修改 IsaacSim 启动流程、不改 AppLauncher 参数路径。

#### Checklist A：Phase 2（布局层闭环）

1. 在 layout runner 中为每个 `layout_index` 生成并固定：`layout_jitter_seed`、`layout_shift_seed`。
2. 在同一 `layout_index` 内生成 `K` 个 `trajectory_seed`，并按 episode 顺序透传给协议脚本（即使 Phase 2 暂不执行增强）。
3. 将以下字段写入每条 H5 metadata：`layout_index`、`rollout_index`、`trajectory_seed`、`layout_jitter_seed`、`layout_shift_seed`、`layout_randomization_record`、`initial_runtime_object_poses`。
4. 在 dataset manifest 中按 episode 粒度落盘，至少包含：`layout_index`、`rollout_index`、`trajectory_seed`、`h5_path`、`returncode`。
5. 跑最小验收：`L=2,K=2`，并人工核对同 layout 初始位姿一致、不同 layout 初始位姿变化。

#### Checklist B：Phase 3（相机一致性门）

1. 保持现有 10Hz 共享时钟，确认每个 H5 都写出 `/cameras/<name>/rgb`、`valid_mask`、`filled_mask`。
2. 对每个 episode 执行三帧抽检（首帧/中帧/末帧）：H5 RGB 与 MP4 时间邻域一致。
3. 将相机失败原因（`never_captured`、`missing_initial_frame`）写入 metadata 并在 manifest 标注失败 episode。

#### Checklist C：Phase 4（分段器 v2 接入前门）

1. 保留现有 stage->atomic v1 映射作为回退路径。
2. 在 metadata 新增 `segmenter_version` 与 `segment_source`（`stage_v1` / `event_kinematics_v2`）。
3. 先对 `anchor_reference` 输出边界原因字段占位（可为空），保证 schema 可前向兼容。
4. 验收口径：先做离线对比，不替换线上默认分段；一致率达标后再切默认。

#### Checklist D：Phase 5（增强执行接入前门）

1. 在每个 episode metadata 写入 `trajectory_augmentation_enabled` 与 `augmentation_record`，Phase 2/3 阶段默认 `false` + `phase5_not_enabled`。
2. 把 `trajectory_seed` 作为唯一增强随机源输入（不得复用 layout seeds）。
3. 仅允许自由空间段候选进入增强；失败必须记录 `fallback_reason` 并回退原轨迹。
4. 先在固定 layout、$K=5$ 下运行 `--trajectory-augmentation-enable --verify-phase5`；增强模式会自动使用不同 rollout seed，`--vary-trajectory-seed` 仍可显式传入。验收口径：同 layout 内至少两条 `trajectory_seed` 对应 `eef_abs` 序列不同，且硬约束段顺序不变。

#### 出口标准（可进入实际增强执行）

1. `L=2,K=2` 的 Phase 2 人工质检门通过。
2. 相机与 H5 一致性门通过，无静默缺帧。
3. manifest 已具备 episode 粒度复现能力。
4. metadata 已完整携带 Phase 5 所需字段，即便增强尚未启用。

### 2.6 Phase 5.1：三维增量轨迹生成实现计划

本节的纯 Python 数据层、anchor protocol、runner 参数、审计 metadata 与离线单测已实现。目标是生成真正改变运动顺序的自由空间轨迹，同时保持接触段语义不变；尚待真实 Isaac 环境验收。

#### A. 数据层：重写候选生成器

修改 `geomind_lab/data/trajectory/augmentation.py`：

1. 新增不可变配置 `IncrementPlanConfig`，字段至少包括 `segment_count`、`candidate_count`、`reverse_probability`、`reverse_ratio`、`max_reverse_absolute_m`、`max_segment_length_m`、`max_segment_ratio`、`zero_axis_bound_m`。
2. 新增 `sample_increment_candidates(start_xyz, end_xyz, seed, config)`。
   - 计算 `dx, dy, dz`。
   - 对非零轴用 seeded positive weights 分配总位移。
   - 对零位移轴生成受限 zero-sum lateral increments。
   - 候选按 seed 轮换 `vertical_first`、`lateral_first`、`mixed`：`vertical_first` 将大 z/小 XY 增量配到前段、小 z/大 XY 增量配到后段，明确先下降到较低高度再完成 XY 对位；后两者提供对照路径。
   - 按概率加入受控反向预算，并重新归一化，确保每轴增量和等于目标差。
   - 将完整 `(x_i,y_i,z_i)` 向量共同打乱，再累计为 waypoint。
3. 新增 `increment_plan_metrics()`，输出 endpoint error、总路径长度、最大段长、最大偏离、轴向顺序签名和 path style。
4. 已删除旧 lateral 生成器；场景布局随机化保持独立且不受此替换影响。

#### B. 平滑与时间参数化

仍在 `augmentation.py` 实现纯 Python 函数：

1. 对内部 waypoint 做可选局部平滑，固定起点、终点和 canonical above-target 点。
2. 平滑后重新计算增量并检查 endpoint error；超过 `1mm` 时拒绝而不是静默修正。
3. 用五次 smoothstep 生成稠密控制目标，按 `max_speed` 计算每段持续时间。
4. 输出每个控制目标的 segment id、来源 waypoint 和目标速度，供协议执行及审计使用。

#### C. 协议层：替换 approach 规划，不改接触动作

修改 `scripts/debug_protocol_anchor_reference.py`：

1. `_plan_approach_augmentation()` 调用 `sample_increment_candidates()`；按 trajectory seed 偏好 `vertical_first`、`lateral_first` 或 `mixed`，不再使用单 lateral waypoint。
2. `_validate_approach_candidate()` 扩展为候选累计点和稠密线段检查：workspace、z 安全高度、segment length、点间距、IK-relative command、速度/加速度和 endpoint error。
3. `_step_approach_waypoint_plan()` 执行平滑后的稠密目标；保持 `GRIPPER_OPEN_COMMAND`，不得调用 descend/close/lift 的随机规划器。
4. `augmentation_record` 新增 `increment_vectors`、`increment_permutation`、`segment_count`、`path_style`、`endpoint_error_m`、`max_segment_length_m`、各类拒因计数。
5. 真实执行中一旦发生 terminated/truncated、实际 EEF 偏差超限或提前接触，停止增强路径并记录 `runtime_execution_failure`；不得继续执行剩余增量。

#### D. 参数与 runner

修改 `scripts/collect_anchor_reference_trajectory_dataset.py` 和协议 parser：

- `--trajectory-augmentation-segments`，当前默认 `6`。
- `--trajectory-augmentation-reverse-probability`，当前默认 `0.75`。
- `--trajectory-augmentation-reverse-ratio`，当前默认 `0.30`。
- `--trajectory-augmentation-max-reverse`，当前默认 `0.08`。
- `--trajectory-augmentation-max-segment-length`，当前默认 `0.14`。

所有参数进入 manifest 配置指纹；参数变化时 `--resume` 必须重新采集，不能复用旧 H5。

#### E. Validator 与验收

修改 `geomind_lab/data/trajectory/validator.py` 和 Phase 5 测试：

1. 检查每条接受轨迹的 endpoint error `<=1mm`。
2. 检查每轴增量和与 `end-start` 一致。
3. 检查至少 3 个不同增量顺序/轴向模式，而不是只比较浮点后的 EEF 数组。
4. 报告 `augmentation_enabled`、fallback、workspace、IK、速度、clearance、runtime failure 分项比例。
5. 明确当前 validator 仍不是全机器人/工具/被持物 mesh 碰撞安全门；该能力完成前禁止将 Phase 5 结果用于大规模正式采集。

#### F. 实施顺序

1. 已完成纯 Python 增量采样、积分、smoothstep、指标和单测。
2. 已完成协议层接入、回退和审计，不启动 Isaac Sim 的离线验证通过。
3. 下一步做真实固定 layout、`K=5`，检查路径签名、endpoint error、fallback 率和抓取成功率。
4. 通过后才扩大到 `L>=2`；最后再扩展 `pre_grasp_align`、lift 后 retreat 和其它 protocol family。

### 2.7 debug_scene_cam 任务迁移计划

本节专门覆盖目录 `outputs/debug_scene_cam_20260804_223814/` 下已有子目录的任务。该目录当前包含 19 个任务：

```text
grasp_anchor_direction, grasp_anchor_distance,
grid_rowcol_grounding, isometric_extension, perfect_midpoint,
place_inside, place_inside_boundary_control, place_on_top,
point_symmetry, reindex_extreme, reindex_subset,
select_exclusion_unique, select_visual_spatial,
square_completion, track_clutter, track_multi, track_single,
trend_continuation, weak_destruction_control
```

这些历史输出主要是 `scene_cam.mp4`、`camera_videos/`、日志和少量 `summary.json` / `trajectory.json`；当前目录没有可作为统一数据集输入的 H5。因此迁移的第一步不是直接给旧视频追加标签，而是对每个 task 重新走统一 collector，生成新的 10Hz H5、原子 segment、运行时初始位姿和 seed metadata。旧视频只能作为人工参考，不能作为 Phase 2--5 的结构证据。

#### 2.7.1 Protocol 与任务映射

| 任务族 | 任务 | 现有 protocol | 迁移入口 |
|---|---|---|---|
| anchor reference | `grasp_anchor_direction`, `grasp_anchor_distance` | `debug_protocol_anchor_reference.py` | 直接复用当前 collector、三维增量 approach 和抓取冻结段 |
| spatial placement | `place_inside`, `place_inside_boundary_control`, `place_on_top` | `debug_protocol_spatial_placement.py` | 新增 placement adapter、容器/支撑面几何目标和释放语义 |
| constraint selection | `select_exclusion_unique`, `select_visual_spatial` | `debug_protocol_constraint_selection.py` | 先完成目标解析，再复用抓取 approach；选择/intervention 前的视觉语义冻结 |
| state reindex | `grid_rowcol_grounding`, `reindex_extreme`, `reindex_subset` | `debug_protocol_state_reindex.py` | 将状态刷新/intervention 与最终 approach 分离记录 |
| historical tracking | `track_single`, `track_multi`, `track_clutter` | `debug_protocol_state_reindex.py` | identity history 和 relocation 作为不可随机化语义段 |
| geometric/shape inference | `isometric_extension`, `perfect_midpoint`, `point_symmetry`, `square_completion`, `trend_continuation`, `weak_destruction_control` | `debug_protocol_geometric_inference.py` | 先按随机 layout 重新求 GT endpoint，再生成任务特定 approach/placement plan |

#### 2.7.2 迁移级别

**A 类：可直接迁移自由空间随机化。**

适用任务：

- `grasp_anchor_direction`
- `grasp_anchor_distance`
- `select_exclusion_unique`
- `select_visual_spatial`
- `grid_rowcol_grounding`

迁移范围仅是目标已经确定后的 `REACH_ABOVE` / pre-grasp approach。三维增量生成器可以直接复用，但必须保留：

1. 目标解析、视觉筛选、行列索引等任务语义不被 trajectory seed 改变。
2. 最终 grasp endpoint、`DESCEND_TO_GRASP`、`GRIPPER_CLOSE`、`LIFT_OBJECT` 不改变。
3. `grid_rowcol_grounding` 的网格结构、行列顺序、间距和 cell identity 不改变。

**B 类：需要任务特定动作 planner。**

适用任务：

- `place_inside`
- `place_inside_boundary_control`
- `place_on_top`
- `perfect_midpoint`
- `isometric_extension`
- `point_symmetry`
- `square_completion`
- `trend_continuation`
- `weak_destruction_control`（仅其自由空间接近段候选属于 B；破坏过程本身属于 C 类冻结语义）

这些任务不能把 A 类 approach 随机化直接套到整条轨迹。迁移时必须先由 resolver 在随机布局上重新计算 GT：中点、延长点、对称点、正方形缺角、趋势点或容器/支撑面目标。然后把轨迹拆成：

```text
目标/结构推理（冻结）
   -> 空间接近（可按任务约束增强）
   -> 姿态/容器口/支撑面 ALIGN（任务特定）
   -> 接触下降（冻结）
   -> 夹爪事件（冻结）
   -> 放置/释放（冻结）
   -> RETREAT（通过任务特定安全检查后才可增强）
```

`place_inside_boundary_control` 和 `weak_destruction_control` 是高风险任务：边界擦碰/半进半出以及结构破坏程度本身是实验变量，默认只采集不随机化，待有专用 planner 与验收后再开放。对 `weak_destruction_control`，破坏过程始终按 C 类冻结，即使未来允许其接近段使用 B 类 planner。

**C 类：完整语义段当前禁止随机化。**

适用任务和段落：

- `track_single`、`track_multi`、`track_clutter`：冻结 relocation 顺序、遮挡/clutter 时序、identity history 和 settle 状态。
- `reindex_extreme`、`reindex_subset`：冻结 intervention、subset 构造、排序/重排过程；仅允许在 intervention 完成且目标重新解析后，对最后的自由空间 approach 采用 A 类策略。
- `weak_destruction_control`：冻结被破坏对象、破坏数量、破坏时序和残余趋势；不得让 trajectory seed 改变 control condition。

#### 2.7.3 跨任务原子动作映射

| 原子动作 | A 类抓取/筛选 | B 类放置/几何 | C 类 tracking/reindex |
|---|---|---|---|
| `IDLE_HOLD` | 可小范围改变时长 | 需保持稳定 | 冻结 |
| `REACH_ABOVE` | 三维增量增强 | 需目标特定 clearance 后增强 | intervention 期间冻结 |
| `DESCEND_TO_GRASP` | 冻结 | 重新抓取时冻结 | 冻结 |
| `GRIPPER_CLOSE` | 冻结 | 冻结 | 冻结 |
| `LIFT_OBJECT` | 冻结 | 冻结 | 冻结 |
| `TRANSIT_XY` | 目标确定后可选增强 | 由 placement/geometric planner 决定 | intervention-specific，默认冻结 |
| `ALIGN_POSE` | 通常冻结 | 必须任务特定 | 冻结 |
| `DESCEND_TO_PLACE` | 不适用 | 必须任务特定并冻结接触几何 | 不适用 |
| `GRIPPER_OPEN` | 不适用 | 必须冻结事件顺序 | 不适用 |
| `RETREAT` | 可作为后续自由空间增强 | 必须检查容器/支撑物 clearance | 默认冻结 |

每个 H5 segment 还必须记录：`semantic_lock`、`is_augmented`、`augmentation_policy`、`segment_source`、`boundary_reason`。`semantic_lock=true` 表示 trajectory seed 禁止改变该段的动作语义，而不是禁止采集该段。

#### 2.7.4 三层 seed 与任务特定 seed

跨任务迁移必须遵守以下契约：

```text
layout_jitter_seed  -> 结构保持型物体相对变化
layout_shift_seed   -> 相关对象/容器整体平移
trajectory_seed     -> 允许的自由空间 waypoint、路径风格、时长
intervention_seed   -> task-specific relocation / identity state plan
```

1. 同一 `layout_index` 的所有 rollout，运行时初始对象 pose 必须一致。
2. `trajectory_seed` 不得进入 `layout_randomizer` 或 intervention planner 的 RNG。
3. 几何任务的 endpoint 必须由随机化后的 GT 重新求解，不能从原 layout 复制坐标。
4. tracking/reindex 先生成并冻结 identity/intervention state，再生成最终 approach。
5. 每个 episode metadata 必须增加 `protocol_family`、`task_semantic_variant`、`intervention_seed`（无 intervention 时为 null）、`gt_snapshot` 和 `semantic_lock_summary`。

#### 2.7.5 统一跨任务 Runner 设计

Phase 6 的跨任务实现不应复制 19 份采集脚本，应新增 adapter 层：

```text
run_trajectory_dataset.py
   -> TaskCatalog / debug_task_registry.json
   -> FamilyRunnerRegistry
   -> LayoutSeedPlanner
   -> ProtocolRunnerAdapter
   -> TaskSemanticPlanner
   -> TrajectoryAugmenter
   -> EpisodeCollector / H5Exporter
   -> QualityGate / ManifestWriter
```

每个 adapter 必须实现：

```python
resolve_task(request) -> ResolvedTask
build_semantic_plan(task, runtime_state, intervention_seed) -> SemanticPlan
augmentation_policy(task, semantic_plan) -> AugmentationPolicy
run_episode(task, layout_seed, trajectory_seed, intervention_seed) -> EpisodeArtifact
```

其中 `build_semantic_plan` 负责 GT、目标物体、容器/支撑面、identity/intervention 状态；`augmentation_policy` 只返回可增强 segment 和硬锁定 segment，不能让通用 augmenter 猜任务语义。

#### 2.7.6 迁移实施顺序与验收门

1. **A1：** 将 `grasp_anchor_direction`、`grasp_anchor_distance`、`select_*`、`grid_rowcol_grounding` 接入统一 EpisodeCollector，先生成 H5，不启用增强，验收 10Hz、初始 pose、任务目标和 atomic label。
2. **A2：** 对 A 类目标解析完成后的 approach 启用三维增量，固定一个 layout 做 `K=5`，验收至少 3 种 `path_style`、endpoint error `<=1mm`、fallback/成功率。
3. **B1：** 先接 `place_on_top`，增加 support surface geometry、ALIGN、DESCEND_TO_PLACE、OPEN、RETREAT 的任务特定 planner；通过后接 `place_inside`。
4. **B2：** 接 geometric families，每个 resolver 单独验证 GT endpoint 随 layout 变化，禁止复制旧坐标。
5. **C1：** tracking/reindex 只接 collector、identity/intervention metadata 和冻结段标注，不开放 trajectory augmentation。
6. **C2：** 只有在 identity state、干预时序和状态判定有专用回放测试后，才评估 intervention 后最终 approach 的 A 类增强。

每个任务的最小验收必须包括：

- `L=2,K=2` 布局独立性和同 layout 初始 pose 一致。
- 至少一条无增强 baseline H5 和至少五条增强/冻结对照 H5。
- 每个 semantic lock 段的原子顺序、事件边界和 GT snapshot 不变。
- A 类至少 3 个 path style；B 类至少 2 个任务特定合法路径；C 类 augmentation 必须为 false。
- H5/manifest 记录 task、family、layout/trajectory/intervention seed、GT、初始 pose 和质量报告。
- 现有 `debug_scene_cam` 目录只作为任务清单和视频人工参考，不作为迁移验收数据集。

---

### Phase 6：批量数据集 Runner、质量门禁与恢复

**目标**

支持跨任务大规模、可中断恢复的数据收集。

**实现内容**

1. 新增统一 CLI：任务清单、$L$、$K$、三类 seed 策略、输出根目录、断点续采。
2. 生成 dataset-level manifest（JSONL/CSV）：每条 episode 的路径、任务、seeds、成功状态、布局和增强记录。
3. 生成质量报告：字段完整性、NaN/Inf、时间单调、相机覆盖、标签覆盖、成功率。
4. 失败 episode 不覆盖已有合格输出；同一唯一 key 可安全重跑。
5. 为每个 H5 保存 schema version 与生成器版本/commit hash。

**当前 anchor-reference 实现补充**

- `--resume` 只复用 seed、layout index、rollout 数和全部 H5 路径均匹配的成功 layout；缺 H5、失败或配置不匹配的 layout 必定重跑。
- 每轮结束写出 `episodes.jsonl`、`episodes.csv` 和 `quality_report.json`；索引按 `episode_uid`（缺失时按 layout/rollout/trajectory seed）构造唯一 key，并报告重复 key。
- `quality_report.json` 汇总 episode/H5 数、成功率、layout 完整性和可选 Phase3 H5 校验摘要；Phase3 未运行时不会伪造 H5 内容检查结果。
- H5 metadata 写入 `schema_version`、`generator_version` 和 `commit_hash`，用于后续复现。
- `--verify-phase6` 会自动执行 H5 schema 校验，并将重复 episode key、缺 H5、未完成 layout、时间/shape 错误或数值 NaN/Inf 作为非零退出的质量门。

**验收标准**

- 中断后重跑不会重复写已有且校验通过的 episode。
- manifest 总条目数等于成功输出 H5 数量，且无重复唯一 key。
- 数据质量检查能识别缺字段、错误 shape、时间倒退、无相机帧、NaN/Inf。
- 能够通过 manifest 精确定位并复现任一 episode。
- $L \times K$ 的统计与实际生成数量一致。

---

## 3. 正式采集前的端到端验收矩阵

在开始大规模采集前，至少完成以下矩阵：

| Case | Layout randomization | Trajectory augmentation | 最低样本数 | 核心检查 |
|---|---|---|---:|---|
| A | 关闭 | 关闭 | 1 | 采集器/H5/相机基础正确 |
| B | 开启，固定 layout seeds | 关闭 | 2 | 同布局 reset 后初始状态一致 |
| C | 开启，不同 layout seeds | 关闭 | 2 | 初始物体坐标与对应 H5 metadata 改变 |
| D | 开启，固定 layout seeds | 开启，不同 trajectory seeds | 5 | 初始布局相同、轨迹不同且任务有效 |
| E | 开启，不同 layout seeds | 开启，不同 trajectory seeds | 10 | 全链路的 seed 独立性、数据完整性与成功率 |

Case A 仅用于隔离采集器故障；正式数据集从 Case B/C/D/E 开始，绝不要求关闭布局随机化。

---

## 4. 统一数据元数据要求

每个 episode 的 H5 和 manifest 至少包含：

```text
task_id, variant, layout_key, scene_id, task_index
layout_index, rollout_index
episode_seed
layout_jitter_enabled, layout_jitter_seed
layout_shift_enabled, layout_shift_seed
trajectory_augmentation_enabled, trajectory_seed
layout_randomization_record
initial_runtime_object_poses
augmentation_record
success, termination_reason
schema_version, generator_version, commit_hash
```

`layout_randomization_record` 必须来自运行时的 `applied_layout_randomization`，而不是由外部脚本重新推算。

---

## 5. 实施原则

1. 先采集、再分段、后增强；每一阶段都要保留独立可验证输出。
2. 轨迹增强必须重新执行并采样，不能仅修改离线数据。
3. 三类随机种子严格独立；布局 seed 绝不由 trajectory seed 消耗或重置。
4. 默认关闭新采集/增强功能，确保既有 benchmark 结果可复现。
5. 新功能的失败必须可审计：记录原因、seed、阶段、路径与 traceback。
6. 将单任务闭环作为合并门槛，再扩展到更多 protocol family。

## 6. 本轮测试与待验收项

已新增并通过环境无关测试：

```text
tests/test_trajectory_segmenter.py
tests/test_trajectory_resample.py
tests/test_trajectory_augmentation.py
tests/test_trajectory_hdf5_exporter.py
tests/test_trajectory_validator.py
```

覆盖内容包括 stage 标签映射、连续 segment、`event_kinematics_v2` 运动/夹爪边界、真实 EEF 位姿保留、时间单调、相对 EEF delta、增强回退、H5 schema、相机严格缺失失败、相机缺帧的填充/掩码和 manifest/H5 校验器。

Phase4 新增覆盖重点：

- 同一 10Hz 区间内 grasp/release 不互相覆盖，主序列保持 raw event 顺序。
- `gripper_state` 阈值能触发 `GRIPPER_CLOSE` / `GRIPPER_OPEN` 硬边界。
- 在常规 stage 存在时，`event_kinematics_v2` 仍优先使用实际 EEF 运动学区分上升、下降、XY 与静止。
- `segmenter_diagnostics.event_projection.dropped_event_count` 暴露不可投影事件，真实验收中必须为 0。

Phase4 真实环境最小验收脚本：

```bash
cd /home/elm/projects/geomind_lab
bash scripts/collect_phase4_acceptance.sh
```

该脚本会启动 Isaac Sim，默认采集 `L=2,K=2` 的 `anchor_reference` H5，使用 `event_kinematics_v2`，并在采集后检查 Phase2、Phase3、事件顺序、无 dropped event、segment 范围和 atomic label 范围。通过时输出：

```text
PHASE4_ACCEPTANCE: PASS
```

只检查命令计划而不启动 Isaac Sim：

```bash
PHASE4_DRY_RUN=1 bash scripts/collect_phase4_acceptance.sh
```

当前系统解释器未安装 `h5py`，HDF5 相关测试会按设计 skip；同时也不具备 IsaacLab/torch。以下验收必须在实际 `isaac_curobo` 运行环境安装 `.[trajectory]` 后执行，且会启动 Isaac Sim：

1. Case A 单 episode：检查 H5 字段、真实 EEF 误差、10Hz 时间轴和所有相机 MP4。
2. Case B/C：$L=2,K=2$，检查同 layout 的初始对象位姿一致、不同 layout 位姿不同。
3. 多相机长 episode：观察 RSS/GPU 内存不随控制步数线性累积 RGB 帧。
4. 人工抽取首帧、中间帧、末帧，对比 H5 RGB 与 MP4，并确认 `valid_mask` / `filled_mask` 含义正确。
5. Phase4 边界抽检：确认 `GRIPPER_CLOSE` / `GRIPPER_OPEN` 与 raw grasp/release event 顺序一致，且 `event_projection.dropped_event_count == 0`。
6. Phase5 固定 layout、$K=5$ 的旧最小随机性验收已通过；本轮新增多 waypoint、全候选横向偏离选择和 H5 多样性质量门，仍需在真实环境复验，并比较增强组与基线成功率以及补齐完整 IK/mesh 碰撞验证。

## 7. 运行复盘与随机性改进记录

已完成事项、真实运行故障、相机/HDF5 注意事项、当前轨迹看起来相同的原因，以及扩大随机范围和解除后半段锁定的候选方案，统一记录在：

[`docs/trajectory_collection_status_and_randomization_zh.md`](docs/trajectory_collection_status_and_randomization_zh.md)

该文档的核心结论是：当前 Phase5 只随机化无接触 `approach`，而 `descend`、`close`、`lift`、`hold` 继续使用 canonical target XY；下一步应先将候选选择改为“验证全部候选后选择最大安全横向偏移”，再引入安全高度上的 `pre_grasp_align`，最后才考虑 lift 后自由空间段。扩大偏移必须与成功率、fallback、IK 拒绝和 clearance 拒绝统计同时验收。
