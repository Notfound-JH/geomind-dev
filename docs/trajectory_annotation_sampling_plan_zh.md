# 轨迹分段标注、增强与采样计划

## 1. 目标和边界

目标是为 GeoMind 的脚本化任务执行轨迹建立一条固定、可调用、可复现的数据管线：

1. 将机器人执行轨迹划分并标注为原子动作。
2. 在不改变任务语义和接触顺序的前提下，对可交换的机器人移动子段做受限随机化。
3. 以 10 Hz 同步采样指令、绝对末端动作、夹爪数据、关节状态与关节速度。
4. 保存所有相机的 512x512 RGB 数据；同时输出 MP4 以便人工检查。
5. 导出单 episode HDF5 文件、视频和可审计的 manifest。

本计划不修改任务布局随机化。布局随机化继续在 `load_task_spec()` 中发生，并应在每个 episode 的轨迹采集之前完成。

## 2. 已有基础与可行性

### 2.1 最适合的第一条接入协议

第一阶段接入 `scripts/debug_protocol_anchor_reference.py`，而不是直接改所有协议：

- 已经有 `--trajectory-json-out`。
- `_trace_step()` 已记录每一步的 action、目标末端位置、实际 EEF 位置、夹爪命令、抓取状态、终止状态和已有的 `stage`。
- `run_scripted_grasp_for_target()` 已将抓取过程分为 `approach`、`descend`、`close`、`lift` 与可选 `hold`。
- `run_episode()` 已在 episode 0 中使用 `RolloutCameraRecorder` 记录所有 Camera。
- 该协议是单对象抓取任务，能以最小风险验证分段、10Hz 对齐和 H5 写出。

当前 `debug_protocol_geometric_inference.py` 和 `debug_protocol_spatial_placement.py` 已有抓取/放置控制路径，后续可复用同一个采集器扩展；`debug_protocol_state_reindex.py` 属于对象移动干预任务，需要单独定义其原子动作语义。

### 2.2 不应在第一阶段做的事

- 不改 `GeoMindFrankaIKRelMimicEnv.step()`，以免影响现有 benchmark 行为。
- 不在第一阶段对接触段或抓取/放置段重排。
- 不把高频 RGB 全量缓存到 GPU/内存后再写盘；长 episode 会占用过多内存。
- 不修改现有 `layout_randomizer.py` 和两个独立布局种子。

## 3. 原子动作标签 v1

标签 ID 必须固定，H5 中保存同一张标签表。

| ID | 名称 | 含义 | 可参与重排 |
|---:|---|---|---|
| 0 | `IDLE_HOLD` | 保持、稳定、等待 | 否 |
| 1 | `REACH_ABOVE` | 运动到目标上方或预抓位 | 有条件 |
| 2 | `DESCEND_TO_GRASP` | 向抓取位下降 | 否 |
| 3 | `GRIPPER_CLOSE` | 闭合夹爪并等待抓取成立 | 否 |
| 4 | `LIFT_OBJECT` | 抓取后抬升 | 否 |
| 5 | `TRANSIT_XY` | 持物/空载的主要平面位移 | 是 |
| 6 | `ALIGN_POSE` | 位姿对齐、小幅 XY/yaw 微调 | 有条件 |
| 7 | `DESCEND_TO_PLACE` | 向放置位下降 | 否 |
| 8 | `GRIPPER_OPEN` | 打开夹爪并等待释放成立 | 否 |
| 9 | `RETREAT` | 放置后撤离到安全位 | 有条件 |

### 3.1 第一阶段的标签映射

`anchor_reference` 当前 stage 到标签的固定映射：

- `approach` -> `REACH_ABOVE`
- `descend` -> `DESCEND_TO_GRASP`
- `close` -> `GRIPPER_CLOSE`
- `lift` -> `LIFT_OBJECT`
- `hold` -> `IDLE_HOLD`

第一阶段只依赖脚本已知的 stage，不依赖速度阈值，从而确保标签可重复。第二阶段再将真实速度/夹爪状态/抓取事件加入规则校验。

## 4. 轨迹增强设计

### 4.1 原则

增强的对象是“生成给控制器的阶段内末端路径”，不是事后篡改记录的状态序列。每次增强都必须实际执行、采样、验证；不能只对 H5 中的动作数组调换顺序。

### 4.2 可交换性与约束图

每个原子段是有向依赖图的节点。以下边为硬约束：

```text
REACH_ABOVE -> DESCEND_TO_GRASP -> GRIPPER_CLOSE -> LIFT_OBJECT
LIFT_OBJECT -> TRANSIT_XY/ALIGN_POSE -> DESCEND_TO_PLACE -> GRIPPER_OPEN -> RETREAT
```

只有满足以下条件的相邻同阶段子段才允许交换：

1. 两段均为 `TRANSIT_XY`、`ALIGN_POSE` 或无接触的 `REACH_ABOVE` 子段。
2. 夹爪状态相同，且不存在抓取、释放、接触或事件边界。
3. 重排后保持相同起点和终点。
4. 每个插值 waypoint 都满足工作空间边界。
5. IK 求解成功，关节速度/加速度不越限。
6. 通过场景碰撞或最小距离检查。

### 4.3 下一版算法：三维增量分解与随机排列

旧实现是“先确定一条路径，再对中间 waypoint 做 lateral 偏移”。下一版改为先生成有符号的三维位移增量，再将完整增量向量随机排列，最后积分得到 waypoint。该方法改变的是运动段的顺序和三维形状，而不是在 canonical path 周围加小扰动。

对自由空间段起点和 canonical 终点定义：

```text
p0 = (x0, y0, z0)
pT = (xT, yT, zT)
D  = pT - p0 = (dx, dy, dz)
```

采样 $k$ 个增量向量：

$$
\delta_i=(x_i,y_i,z_i),\quad i=1,\ldots,k
$$

并强制：

$$
\sum_i x_i=d_x,\quad \sum_i y_i=d_y,\quad \sum_i z_i=d_z
$$

累计 waypoint 为 $p_i=p_{i-1}+\delta_i$，因此最后一点必须满足 $p_k=p_T$。随机排列作用于完整的 `delta_i` 向量，不能分别打乱三个轴。

对 $d_a\ne0$ 的轴 $a\in\{x,y,z\}$，先生成正权重并归一化：

$$
w^a_i\sim\operatorname{LogNormal}(0,\sigma^2),\qquad
\alpha^a_i=\frac{w^a_i}{\sum_jw^a_j},\qquad a_i=d_a\alpha^a_i
$$

无反向模式下，$d_a>0$ 时所有 $a_i>0$，$d_a<0$ 时所有 $a_i<0$。对 $d_a=0$ 的轴允许零和侧向扰动：

$$
a_i=\epsilon_i-\frac{1}{k}\sum_j\epsilon_j,\qquad |a_i|\le b_a
$$

实现中的反向预算仍沿用既有参数，但符号边界固定为 $0.05\,\mathrm{m}$：$d_a>0$ 时要求 $a_i>-0.05$，$d_a<0$ 时要求 $a_i<0.05$，$d_a=0$ 时要求 $-0.05<a_i<0.05$。生成后只对完整的三维 `delta_i` 向量做排列，再累计为 waypoint。

这样仍有 $\sum_i a_i=0$。

为产生受控来回运动，可按 seed 以 `reverse_probability` 分配反向预算：

$$
r_a=\min(\rho|d_a|,r_{max})
$$

初始建议 `reverse_probability=0.4`、`rho=0.15`、`r_max=0.04m`。例如 $d_x>0$ 时，反向增量必须满足 $x_i\ge-r_x$，并与正向预算共同归一化到 $\sum_i x_i=d_x$。workspace 紧张时关闭反向预算，不能为了保留反向而放宽硬约束。

首轮 anchor-reference 参数：`segment_count=4`、`candidate_count=12`；后续允许 `segment_count=3..7`、`candidate_count=8..16`。最大单段长度建议 `0.08..0.12m`，且不超过总路径长度的 `40%`，超限时重新采样整组增量。最后一个点严格回到 canonical approach endpoint，canonical above-target 和接触段不自由扰动。

### 4.4 平滑、时间参数化与安全校验

候选 waypoint 生成后不直接以折线瞬时移动执行。内部点可做受限局部平滑：

$$
	ilde p_i=(1-\lambda)p_i+\frac{\lambda}{2}(p_{i-1}+p_{i+1})
$$

起点、终点和 canonical above-target 点必须固定；平滑后重新修正增量总和，并重新执行全部校验。时间参数化使用五次 smoothstep：

$$
s(u)=10u^3-15u^4+6u^5
$$

每段时间设为 $T_i=\max(T_{min},\|p_i-p_{i-1}\|/v_{max})$，离散化后检查速度、加速度和控制频率下的相邻目标变化。

候选按以下顺序验证：

1. workspace、桌面/容器高度、累计点和线段长度。
2. 每 `2--5mm` 稠密采样点的非目标物体 clearance。
3. 每个控制目标的 finite IK-relative action、姿态连续性和 gripper-open 状态。
4. 速度、加速度、最大偏离及是否过早进入接触区域。
5. 当前版本的范围必须记录为 `runtime_workspace_ik_command_speed_point_clearance_only`；这不是全机器人、工具或被持物 swept-volume collision validator。

任一硬约束失败即拒绝候选；所有候选失败时执行 canonical fallback，不得事后改写 H5。

### 4.5 可追溯性

每个增强 episode 必须写入：

- `augmentation_enabled`
- `augmentation_seed`
- `original_segment_order`
- `augmented_segment_order`
- `waypoint_order`
- `validation_passed`
- `fallback_reason`（若回退）
- `segment_count`
- `increment_vectors`
- `increment_permutation`
- `path_style`
- `max_segment_length_m`
- `endpoint_error_m`
- `trajectory_validation_scope`

## 5. HDF5 Schema v1

一个 episode 对应一个 `.h5`；全部 dataset 的第 0 维共享 10Hz 时间轴 `T`。

```text
/metadata/task_id                         utf-8 scalar
/metadata/variant                         utf-8 scalar
/metadata/layout_key                      utf-8 scalar
/metadata/instruction                     utf-8 scalar
/metadata/scene_id                        int32 scalar
/metadata/episode_seed                    int64 scalar
/metadata/sample_hz                       float32 scalar (= 10.0)
/metadata/schema_version                  utf-8 scalar (= "geomind_trajectory_v1")
/metadata/camera_names                    utf-8 [C]

/steps/index                              int32 [T]
/steps/timestamp_s                        float64 [T]
/actions/eef_abs                          float32 [T, 7]  # x,y,z,qx,qy,qz,qw
/actions/eef_delta_rel                    float32 [T, 7]  # body/EEF frame delta, 首帧为 0
/actions/gripper_cmd                      float32 [T, 1]
/robot/joint_pos                          float32 [T, J]
/robot/joint_vel                          float32 [T, J]
/robot/gripper_state                      float32 [T, G]
/labels/atomic_id                         int16 [T]
/labels/segment_id                        int32 [T]
/labels/atomic_name_table                 utf-8 [10]

/segments/start_step                      int32 [S]
/segments/end_step                        int32 [S]
/segments/atomic_id                       int16 [S]
/segments/is_augmented                    uint8 [S]

/cameras/<camera_name>/rgb                uint8 [T,512,512,3]
/cameras/<camera_name>/timestamp_s        float64 [T]
/cameras/<camera_name>/video_path         utf-8 scalar

/augmentation/*                           标量或 UTF-8 JSON，见第 4.4 节
```

- `eef_abs` 表示控制器的绝对目标末端位姿；若有必要，可附加 `/observations/eef_pose_world` 保存实际位姿。
- `eef_delta_rel[t]` 从相邻 `eef_abs` 计算：平移先转入上一帧 EEF 坐标系，旋转用四元数相对变换表示。
- RGB 统一缩放到 512x512。为训练便利保存 H5 帧；为人工回放并行保存 MP4。
- H5 推荐 `gzip`，压缩级别 4，RGB chunk 为 `[1,512,512,3]` 或 `[8,512,512,3]`，由磁盘吞吐测试决定。

## 6. 固定处理流程

```text
TaskRequest
  -> task/layout randomization
  -> scripted or policy rollout
  -> 高频 StepCollector（状态、动作、事件、相机）
  -> AtomicSegmenter
  -> 可选 TrajectoryAugmenter（重新执行，不篡改日志）
  -> 10Hz Resampler
  -> H5 + MP4 Exporter
  -> schema/时序/成功判据 Validator
  -> dataset manifest
```

Python 公共接口建议：

```python
collect_episode(task_request, run_cfg) -> RawEpisode
segment_episode(raw_episode, segment_cfg) -> SegmentedEpisode
augment_segments(segmented_episode, augment_cfg) -> AugmentedEpisode
resample_to_hz(episode, sample_hz=10.0) -> SampledEpisode
export_episode_h5(sampled_episode, export_cfg) -> ExportManifest
run_trajectory_pipeline(tasks, pipeline_cfg) -> DatasetManifest
```

## 7. Phase 1：最小实现切入点清单

### 7.1 目标

只打通一个 `anchor_reference` episode：不做重排增强，采集并导出 10Hz H5 + 全相机 512x512 MP4/帧；标签完全由现有 stage 映射生成。

### 7.2 新增模块

1. `geomind_lab/data/trajectory/collector.py`
   - `StepRecord`：高频单步状态。
   - `EpisodeCollector`：保存高频序列，接收 `record_step()` 调用。
   - 读取：`env.get_robot_eef_pose()`、`env.scene["robot"].data.joint_pos`、`joint_vel`、现有夹爪 joint ids、控制目标姿态与 gripper command。
   - 不调用 `env.step()`，不改变控制逻辑。

2. `geomind_lab/data/trajectory/segmenter.py`
   - `stage_to_atomic_id(stage_name)`：实现第 3.1 节映射。
   - `build_segments(atomic_id)`：生成连续段的 start/end/id。

3. `geomind_lab/data/trajectory/resample.py`
   - `resample_episode(records, target_hz=10.0)`。
   - 优先按仿真时间戳抽样；控制目标、joint_pos、joint_vel、gripper 使用最近左侧样本，图像使用同时间或最近样本。
   - `compute_relative_eef_deltas()`。

4. `geomind_lab/data/trajectory/hdf5_exporter.py`
   - `export_episode_h5(sampled_episode, path, compression="gzip")`。
   - 使用 `h5py`；当前 `pyproject.toml` 未声明此依赖，实施时需将其加入项目依赖或明确写入运行环境要求。
   - 实现 schema v1、shape 检查和临时文件后原子 rename。

5. `geomind_lab/data/trajectory/camera_recorder.py`
   - 基于现有 `geomind_lab/utils/headless_camera_video.py` 的 `_iter_camera_sensors()`、`_rgb_env0_to_uint8_hwc()`、`_write_mp4_clip()` 重用/公开最小 API。
   - `record_step(timestamp_s)` 抓取所有相机 RGB 并 resize 到 512x512。
   - 第一版允许录制频率高于 10Hz，最后由 `resample.py` 按同一时间轴选帧。

### 7.3 对已有文件的精确修改

1. `scripts/debug_protocol_anchor_reference.py`
   - `parse_args()`：新增 `--trajectory-h5-out`、`--trajectory-sample-hz`（默认 10）、`--trajectory-camera-size`（默认 512）、`--trajectory-record-cameras`。
   - `_trace_step()`：新增可选参数 `collector: EpisodeCollector | None`；在 `env.step(action)` 返回后调用 `collector.record_step(...)`，把 `stage_name`、`stage_step`、action、target_pose、gripper command、terminated/truncated 交给 collector。
   - `_step_with_tracked_target_xy()`：将 collector 透传给 `_trace_step()`。
   - `run_scripted_grasp_for_target()`：接收 collector，并向所有 `_step_with_tracked_target_xy()` 调用透传。
   - `run_episode()`：`env.reset()` 和 reset settle 后创建 collector；首个动作前执行 `collector.record_initial_observation()`；执行后调用 `collector.finalize()`；若指定 `--trajectory-h5-out`，调用 `resample_episode(..., 10.0)` 与 `export_episode_h5(...)`。
   - `main()`：确保异常时关闭 collector，输出 H5 路径和 manifest。

2. `geomind_lab/utils/headless_camera_video.py`
   - 不改变 `RolloutCameraRecorder` 的现有行为。
   - 仅将当前私有 helper 提升为稳定的、无副作用的读取接口，或由新模块复制极少量 tensor 转 uint8/resize 逻辑。
   - 必须始终录制所有 `scene.sensors` 中有 RGB 输出的 Camera，不硬编码相机名称。

3. `pyproject.toml`
   - 增加 `h5py` 依赖（建议 `h5py>=3.10`）。
   - 若运行环境已有 conda h5py，则仍应在项目依赖中声明，避免不可复现实验环境。

### 7.4 单任务打通顺序

1. 选固定任务：`anchor_reference` 的默认 `standard` variant，关闭布局随机化。
2. 先只收集 proprio/action/stage 到内存，打印每项 shape 和前后时间戳；不写 H5、不录相机。
3. 加 10Hz resample，验证 `T`、时间单调、首帧 relative delta 为 0。
4. 加 H5 exporter，离线读回检查字段、dtype、shape、无 NaN。
5. 加相机采集与 512x512 resize，先输出一个 MP4，再写 RGB 至 H5。
6. 对同一 episode 读 H5，检查 H5 RGB 第 `t` 帧与 MP4 对应时间附近的帧一致。
7. 对启用/关闭布局随机化各采一条，确认 `/metadata/layout_key` 相同而初始物体位置、相应动作轨迹可区别；同时不改变现有任务成功判断。

### 7.5 Phase 1 验收标准

- 在不改 `env.step()` 的前提下，单 episode 能生成一个可读 H5 和每相机一个 512x512 MP4。
- 10Hz 样本数与 episode 时间一致，`timestamp_s` 严格递增。
- 数据集包含全部 schema v1 必填字段，形状一致，无 NaN/Inf。
- 每个时间步都有合法 atomic 标签与 segment id，标签只来自标签表。
- 未启用新 CLI 参数时，现有 debug 协议输出和控制路径不改变。
- 采集器关闭时不改变 rollout 成功率或动作数量。

## 8. 后续扩展顺序

1. 已完成 anchor-reference 的 collector、H5、三维增量与 Phase 6 基础 runner；真实 Isaac/H5 验收仍待执行。
2. 下一步 A1：将 `grasp_anchor_*`、`select_*`、`grid_rowcol_grounding` 接入统一 EpisodeCollector，先生成无增强 baseline H5。
3. 下一步 A2：只对目标解析后的自由空间 approach 开启三维增量，固定 layout 做 `K=5`，验收 path style、endpoint error、fallback 和成功率。
4. 再接 B 类 placement/geometric planner；tracking/reindex 先只采集并标注冻结的 identity/intervention 语义，不开放随机化。

### 8.1 debug_scene_cam 跨任务迁移顺序

`outputs/debug_scene_cam_20260804_223814/` 下的历史目录只有 MP4/日志和少量 JSON，没有统一 H5，因此不能直接在旧视频上追加随机轨迹标签。迁移必须重新运行 protocol collector，并按任务语义分级：

- A 类直接复用 `REACH_ABOVE` 增量增强：`grasp_anchor_*`、`select_*`、`grid_rowcol_grounding`。
- B 类先实现任务特定 planner：`place_*`、`perfect_midpoint`、`isometric_extension`、`point_symmetry`、`square_completion`、`trend_continuation`、`weak_destruction_control`。
- C 类先只采集和标注冻结语义：`track_single`、`track_multi`、`track_clutter`；`reindex_*` 的 intervention 段也冻结，仅允许最终目标解析后的 approach 进入 A 类流程。

每个 protocol adapter 必须先生成 `semantic_plan` 和 `semantic_lock`，通用 `TrajectoryAugmenter` 只能处理 adapter 明确授权的自由空间 segment；它不得自行推断目标、GT、identity 或 intervention 语义。
