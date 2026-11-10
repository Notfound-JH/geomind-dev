# 轨迹采集实现状态、踩坑记录与随机性改进方案

更新时间：2026-08-31

## 2026-08-31 几何推断任务轨迹采集

- 五个正式任务为 `perfect_midpoint`、`point_symmetry`、`isometric_extension`、
	`square_completion`、`trend_continuation`。任务实例真源是
	`assets/annotations/annotated_tasks.json`，并叠加
	`geomind/configs/tasks/legacy_layouts.json` 与
	`geomind/configs/vla_tasks/legacy_layouts.json`；`task_profiles.py` 解析
	task/variant 到 layout/scene，`task_specs.py` 注入 runtime objects 并调用
	`layout_randomizer.py`。
- 新增 `scripts/export_geometric_inference_trajectory_dataset.py`，默认对五个任务各采
	10 个随机 layout、每个 layout 10 条轨迹。它调用支持
	`--protocol geometric_inference` 的公共 layout runner；每个 layout 使用独立 Isaac
	子进程，同一 layout 内固定物体初态并改变 trajectory seed。
- 几何协议现导出统一 H5 metadata contract，并在 episode reset 后显式恢复 runtime
	object。机器人 reset seed 固定，轨迹多样性由持块后的高空自由空间 waypoint seed
	产生；接触、释放和几何目标求解保持不变。
- 正式采集启动前的 `perfect_midpoint` 1 layout x 2 rollout 冒烟结果：2/2 成功，
	Phase3/Phase6 均通过；两条轨迹 initial poses 相同，trajectory seed、waypoint 和实际
	EEF hash 均不同。输出在
	`outputs/geometric_inference_trajectory_smoke_fixed_reset_20260831/`。
- Isaac Sim 启动前必须检查现有进程，且只能在 tmux 中运行；启动环境显式设置：

```bash
export MDL_SYSTEM_PATH="${MDL_SYSTEM_PATH:-/home/elm/projects/InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes/Materials}"
```

## 2026-08-31 变更（可复现运动函数注册）

- 新增 `geomind_lab/mimic/envs/post_action_relocation_motion.py`，把 track/reindex 的桌面 XY
	轨迹从环境更新循环中抽成具名函数。运动函数只决定每帧位置，不改变 task plan、终态 rank、
	identity GT 或 readout。
- 默认模式：
	- `track_single`：`track_single_paired_arc`，两块沿相反半圆平滑交换，是本轮迁移基准。
	- `track_multi`：`track_multi_staggered_waypoint`，只移动 rank 0/2/4 三块，rank 1/3 固定。
	- `reindex_insert`：`reindex_insert_shallow_arc`，只移动原最右块，锁定桌面 Z，经浅侧向 waypoint 插入。
	- `reindex_neighbor`：`reindex_neighbor_bezier_lanes`，最多移动目标和一个挡路块，并保留静态保护块间距约束。
	- `reindex_subset`：`reindex_subset_two_block_swap`，只移动两块红色方块，其中一块走后侧 Bezier 路径。
- 选择参数有两种等价入口：标注协议的 `annotation.protocol.intervention.motion_mode`，或 debug
	运行时环境变量 `GEOMIND_RELOCATION_MOTION_MODE`。后者优先写入 runtime protocol，适合不改标注的复现实验，例如：

```bash
GEOMIND_RELOCATION_MOTION_MODE=track_single_paired_arc \
	HEADLESS=1 bash scripts/debug_reindex_insert.sh
```

- 解析后的 `motion_mode` 会写入 protocol snapshot 和 rollout summary，协议 shell 也会在启动头打印覆盖值。
	未注册名称直接报错，不静默回退。`track_single` semantic/visual 的专用遮挡绕行默认继续使用
	`track_single_middle_direct_legacy`；只有显式覆盖参数时才迁移到注册函数。
- 颜色：`GEOMIND_INSERT_COLOR_TAGS`、`GEOMIND_NEIGHBOR_COLOR_TAGS`、
	`GEOMIND_SUBSET_COLOR_TAGS` 仍默认关闭，五个标准任务均不在运动结束时改变方块颜色。
- 动作语义：本轮保留“抓取”。缩小到 60% 后的真实复验此前仍能稳定抓取；改成“指出”会同时改变
	prompt、readout 和成功判据，不能作为纯运动重构混入。若后续抓取回归，再作为独立协议版本实现。
- 离线验证：`tests/test_post_action_relocation_logic.py` 共 9 项通过，覆盖五个模式注册、轨迹端点、
	`track_multi` 三 mover、insert/subset/identity GT 与 accidental displacement。
- 真实复验统一使用 `scene_cam`、`GEOMIND_LIGHTING_POOL_INDEX=0`、单任务单 tmux，并在 tmux 内先设置
	`MDL_SYSTEM_PATH`。五项均 `exit_code=0`、`intervention_success_rate=1.0`、
	`selection_accuracy=1.0`、`valid_episode_rate=1.0`、`accidental_displacement_rate=0.0`。
- 本轮视频：
	[track_single](../logs/state_reindex_rollouts/track_single/standard/geometric/20260831_motion_registry_track_single/camera_videos/scene65_historical_identity_blocks_line_3_task0/scene_cam.mp4)、
	[track_multi](../logs/state_reindex_rollouts/track_multi/standard/geometric/20260831_motion_registry_track_multi/camera_videos/scene65_historical_identity_multistep_blocks_line_5_task0/scene_cam.mp4)、
	[reindex_insert](../logs/state_reindex_rollouts/reindex_insert/standard/geometric/20260831_motion_registry_reindex_insert/camera_videos/scene65_state_reindex_insert_blocks_line_4_task0/scene_cam.mp4)、
	[reindex_neighbor](../logs/state_reindex_rollouts/reindex_neighbor/standard/geometric/20260831_motion_registry_reindex_neighbor/camera_videos/scene65_state_reindex_neighbor_red_left_blocks_5_task0/scene_cam.mp4)、
	[reindex_subset](../logs/state_reindex_rollouts/reindex_subset/standard/geometric/20260831_motion_registry_reindex_subset/camera_videos/scene65_state_reindex_red_subset_blocks_line_5_task0/scene_cam.mp4)。
	时长依次为 14.017、14.017、14.267、14.600、14.600 秒。
- 端到端抓取结果：`track_single` / `track_multi` 的 `grasp_detection_rate=success_rate=1.0`；三项
	reindex 的重排和目标选择均正确，但 `grasp_detection_rate=success_rate=0.0`。`reindex_insert`
	同配置复跑仍为 `no_grasp`；诊断中方块高度为 0.027 m、夹爪已闭合但 lift 后目标距离约 0.172 m，
	说明缩小方块后现有抓取器不稳定。运动视频有效，但不能将其表述为端到端抓取成功。建议后续新增独立
	`point` readout 协议版本（含 prompt、执行器和成功判据），不要静默改变现有 benchmark 的抓取语义。

### 本次阅读确认的配置与采集链

- 本批 debug 任务走真实 `geomind_lab/` IsaacLab 主线；任务实例真源为
	`assets/annotations/annotated_tasks.json`，并由 `geomind/configs/tasks/legacy_layouts.json` 与
	`geomind/configs/vla_tasks/legacy_layouts.json` overlay。`task_profiles.py` 负责
	`task_id + variant -> layout_key/scene_id/task_index/object_mode`，`task_specs.py` 负责解析和注入 runtime objects。
- debug 链为 `debug_<task>.sh -> debug_protocol_state_reindex.sh -> debug_protocol_state_reindex.py`。
	state refresh 当前导出 summary 与首回合多相机 MP4，尚未接入统一 H5 `EpisodeCollector`。
- 物体 layout 随机化在 `layout_randomizer.py`，由 `GEOMIND_LAYOUT_JITTER_ENABLE` /
	`GEOMIND_LAYOUT_SHIFT_ENABLE` 分别控制，两套 seed 独立；两者默认关闭。
- 灯光随机化在 `geomind_scene_cfg.py` 的 15 项 `LIGHTING_POOL`，默认开启；
	`GEOMIND_LIGHTING_POOL_INDEX` 可固定变体，`GEOMIND_LIGHTING_POOL_SEED` 可复现抽样。

## 2026-08-25 变更（track / reindex 运动重构）

- 作用范围：仅 `post_action_relocation` 协议且 `track_name` 属于
	`state_reindexing` / `reorder_with_distractors` / `historical_identity_tracking` / `object_identity_tracking`。
- 方块尺寸：上述任务里的 `primitive_cuboid` 在 task spec 注入阶段统一缩放为原来的 `60%`（三轴同时缩放），不影响其他任务族。
- 碰撞规避：
	- `track_clutter` 只保留 identity set 的动态运动，不再把 clutter 物体一起加入动态 slot。
	- `reindex_insert` 的可视化滑动仅注册真实位移对象（通常 2-3 个），不再把静止对象加入同动。
	- `reindex_subset` 仅移动交换的两块红色方块，不再让全部桌面方块参与同步路径控制。
- 目标：在保持任务语义（GT 解析逻辑不变）的前提下，降低同一时刻多物体交叉运动导致的碰撞与旋转抖动。

### 审阅后修订

- `track_multi` 从 4 块链式重排改为 3 块循环换位，并同步终态 rank/GT 验证关系；初始最左目标仍在终态最右位置。
- `reindex_insert` 改为只移动原最右方块，固定桌面 Z，以浅侧向 waypoint 平滑滑入前两块之间。
- 首回合相机 recorder 改为在 `env.reset()` 前安装，确保录到 reset 后的完整物体运动；此前 `reindex_neighbor` 的运动可能在 recorder 安装前已完成。
- `reindex_insert` / `reindex_neighbor` / `reindex_subset` 分别默认关闭
	`GEOMIND_INSERT_COLOR_TAGS` / `GEOMIND_NEIGHBOR_COLOR_TAGS` / `GEOMIND_SUBSET_COLOR_TAGS`，
	标准执行不再在选择或抓取前对任何方块做运行时变色；这些开关仅保留给显式视觉调试。
- 最终复验发现 recorder 前移后，`run_agent_rollout()` 与 `run_agent_episode()` 各执行了一次 reset。对 `reindex_neighbor`，第二次 reset 会把第一次重排末态缓存成新初态，导致 moved object 身份与动态计划不一致。现由 `run_agent_rollout()` 统一负责每回合唯一一次 reset，既保留 reset-time 录像，也避免跨 reset 初态污染。
- 最终 `scene_cam` 复跑中，四项任务均为 `exit_code=0`、`intervention_success_rate=1.0`、`selection_accuracy=1.0`、`accidental_displacement_rate=0.0`；视频时长分别为 `track_multi=14.017s`、`reindex_insert=14.267s`、`reindex_neighbor=14.600s`、`reindex_subset=14.600s`。
- 关闭颜色标签后重新导出的 `reindex_insert` / `reindex_neighbor` 仍为 `exit_code=0`、
	`intervention_success_rate=1.0`、`selection_accuracy=1.0`、`valid_episode_rate=1.0`、
	`accidental_displacement_rate=0.0`；逐帧抽样确认 authored color 保持不变。
- 最终 `scene_cam` 视频：
	[track_single](../logs/state_reindex_rollouts/track_single/standard/geometric/20260825_refactor_debug_track_single/camera_videos/scene65_historical_identity_blocks_line_3_task0/scene_cam.mp4)、
	[track_multi](../logs/state_reindex_rollouts/track_multi/standard/geometric/20260825_review2_final_debug_track_multi/camera_videos/scene65_historical_identity_multistep_blocks_line_5_task0/scene_cam.mp4)、
	[reindex_insert](../logs/state_reindex_rollouts/reindex_insert/standard/geometric/20260825_colorless_final_debug_reindex_insert/camera_videos/scene65_state_reindex_insert_blocks_line_4_task0/scene_cam.mp4)、
	[reindex_neighbor](../logs/state_reindex_rollouts/reindex_neighbor/standard/geometric/20260825_colorless_final_debug_reindex_neighbor/camera_videos/scene65_state_reindex_neighbor_red_left_blocks_5_task0/scene_cam.mp4)、
	[reindex_subset](../logs/state_reindex_rollouts/reindex_subset/standard/geometric/20260825_review2_final_debug_reindex_subset/camera_videos/scene65_state_reindex_red_subset_blocks_line_5_task0/scene_cam.mp4)。

本文记录 GeoMind 轨迹采集管线目前已经实现的功能、真实运行中遇到的问题、容易犯的错误，以及当前轨迹随机性不足的原因和后续方案。它是实施计划的运行维护补充，不替代数据 schema 或验收定义。

## 1. 已完成的工作

### 1.1 数据管线

- `EpisodeCollector` 已接入 `anchor_reference` 的逐控制步采集。
- 统一保存 reset-relative 仿真时间、控制目标 EEF 位姿、实际 EEF 位姿、相对动作、关节状态、夹爪状态、抓取事件和终止信息。
- 高频记录可重采样到 10Hz，并通过 `stage_v1` 或 `event_kinematics_v2` 生成原子动作和 segment。
- H5 schema 已包含 steps、actions、observations、robot、events、labels、segments 和 metadata。
- `layout_randomization_record`、`initial_runtime_object_poses`、`trajectory_seed` 和 `augmentation_record` 会进入 episode 元数据。

### 1.2 布局和 rollout runner

- `scripts/collect_anchor_reference_trajectory_dataset.py` 已实现 layout x rollout 两层采集。
- 同一 layout 内固定 `layout_jitter_seed` / `layout_shift_seed`，每条 rollout 使用独立 `trajectory_seed`。
- 每个 layout 使用独立子进程，避免不同 layout 之间复用污染的 Isaac 场景。
- `manifest.json` 和 `layout_status.json` 提供运行中、成功、失败、子进程退出码和 H5 完整性状态。
- 缺失预期 H5 时，即使子协议返回 0，也会将 layout 标记为失败，避免“看似成功的空数据集”。
- Phase6 的 anchor runner 支持 `--resume`：仅复用 returncode=0、`collection_complete=true` 且预期 H5 全部存在的 layout；其它 layout 会重跑。
- 每轮收集完成后写出 episode 级 `episodes.jsonl`、`episodes.csv`、`quality_report.json`，并以 `episode_uid` 或 layout/rollout/seed 回退 key 检查重复。
- H5 metadata 记录 `generator_version` 与 `commit_hash`；跨 protocol family 的统一收集 CLI 仍未实现。
- `--verify-phase6` 会同时检查 episode key 唯一性、layout/H5 完整性与 H5 schema；H5 数值含 NaN/Inf 时会被拒绝。

### 1.3 相机导出

- `--record-cameras` 会开启 512x512 RGB 相机流、H5 相机数据和每个 episode 的 MP4 sidecar。
- 相机视频目录按 episode 分开：

```text
camera_videos/<task_label>/episode_0000/<camera>.mp4
camera_videos/<task_label>/episode_0001/<camera>.mp4
```

- 当前 anchor 场景的成功运行已验证 5 个 episode、7 路相机，共 35 个 MP4 可以生成。
- H5 相机数据包含 `valid_mask` 和 `filled_mask`，严格模式下未捕获相机或缺少首个有效帧会失败。

### 1.4 Phase 5 执行层

当前只在 `anchor_reference` 的无接触 `approach` / `REACH_ABOVE` 段启用：

1. 用 `trajectory_seed` 生成多个含两个安全高度 lateral waypoint 的候选。
2. 对候选执行 workspace、有限 IK-relative action、速度和非目标物体点间距检查。
3. 对通过的候选实际调用 `env.step()`，而不是只修改离线 H5。
4. 所有候选失败时回退原始直达 approach，并在 `fallback_reason` 中记录原因。
5. `candidate_evaluations`、拒绝原因、waypoint 和执行路径写入 JSON/H5/manifest。
6. 已通过固定 layout、K=5 的最小质量门：初始对象位姿一致、至少两条 EEF 序列不同、抓取接触顺序保持、增强记录存在。

## 2. 真实运行中修复过的问题

### 2.1 `env_ids` 不能传 Python list

在恢复 runtime object 初始状态时，`asset.write_root_state_to_sim(..., env_ids=[0])` 会在当前 PhysX torch 后端内部调用 `.to()`，导致：

```text
AttributeError: 'list' object has no attribute 'to'
```

修复方式是传入设备一致的 `torch.long` 索引张量，并用同一索引切片 root state：

```python
env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
```

### 2.2 相机存在时不能关闭 AppLauncher 相机开关

即使不要求写视频，场景中已经声明 camera sensor 时，IsaacLab 仍要求启动参数带 `--enable_cameras`。否则环境创建阶段会失败：

```text
RuntimeError: A camera was spawned without the --enable_cameras flag.
```

因此“是否录制相机”和“是否允许场景创建相机”是两个不同开关：

- `enable_cameras=True`：满足场景初始化要求。
- `--record-cameras` / `--trajectory-record-cameras`：决定是否采集并写盘。

不要用关闭 AppLauncher camera 的方式优化一个含 camera sensor 的 scene。

### 2.3 短 episode 的 HDF5 chunk 不能固定大于数据长度

相机 H5 导出曾将 mask 的 chunk 固定为 `(128,)`。当采样长度只有 32 时，h5py 报：

```text
ValueError: Chunk shape must not be greater than data shape
```

现在 `valid_mask` / `filled_mask` 按实际 `sample_count` 自适应 chunk；短 episode 也可以正常写出。

### 2.4 部分输出不能当作成功数据集

曾出现子进程返回 0，但只产生日志、没有完整 H5 的情况。现在必须同时检查：

- 子进程退出码。
- 预期 episode H5 是否全部存在。
- manifest 的 `collection_complete`。
- Phase5/Phase3 报告是否通过。

只看到 `console.log` 或一条视频不能说明一个 layout 完整成功。

### 2.5 reset 后 runtime object 可能残留上一条轨迹的状态

通用 `env.reset()` 不一定把脚本移动过的 runtime object 恢复到初始位姿。对 anchor 任务而言，这会让下一条 rollout 的 GT 关系产生歧义。当前协议在新 episode 开始时显式恢复 runtime object root state，并清零速度。

## 3. 当前采集时容易犯的错误

1. 忘记 `--record-cameras`：轨迹 H5 可以正常导出，但不会有相机流或 MP4。
2. 只查看 `camera_videos/.../episode_0000`：其它视频在 `episode_0001` 到 `episode_0004` 子目录中。
3. 把 MDL 材质错误、DLSS 分辨率警告和 mesh primvar 警告直接当成采集失败。应优先寻找 `[anchor_debug_stage]`、Python traceback、H5 路径和 layout status。
4. 把 `child_returncode=0` 当成完整成功。必须检查 H5 数量和 `collection_complete`。
5. 修改 layout seed 和 trajectory seed 的职责。一个 layout 内必须固定 layout seed，只改变 trajectory seed。
6. 把当前点间距 validator 当成完整碰撞规划器。它没有覆盖机械臂、末端工具和被持物的 swept-volume mesh 碰撞，也不是独立全关节 IK 求解。
7. 直接增加随机偏移而不先检查失败率。随机幅度扩大后必须比较增强组和未增强基线的成功率、fallback 率和 IK/clearance 拒绝原因。
8. 用视频整体观感判断随机性。应同时读取 H5 的 `eef_abs`、`observations/eef_pose_world`、waypoint、segment 和 seed 做定量比较。

## 4. 为什么当前轨迹看起来几乎一样

当前 Phase5 的随机化不是整条轨迹随机化，而是“固定语义动作 + approach 局部 waypoint 随机化”。相同视频观感由以下因素共同造成：

### 4.1 随机作用范围很短

随机 waypoint 只插入抓取前的无接触 approach。之后的 descend、close、lift 和 hold 使用固定阶段、固定步数和固定目标。抓取前的局部差异占整段视频比例很小，远景相机尤其不明显。

### 4.2 后半段显式锁定目标 XY

approach 完成后读取当前目标 XY，随后 descend、close、lift、hold 都传入 `fixed_xy=locked_target_xy`。这个设计用于保护抓取成功率和接触顺序，但也会把后半段收敛成同一条竖直路径。

### 4.3 旧候选策略是第一个可行即停止（已修复，待真实复验）

旧版本会在 candidate 0 通过后立即返回，候选集合的多样性没有充分转化为执行多样性。当前版本已先验证全部候选，再按最大横向偏离、较短路径、稳定 candidate index 的顺序选择；真实 Isaac 复验仍需确认 validator 不会把大偏移候选系统性拒绝。

### 4.4 当前偏移参数是保守值

默认最大 lateral offset 为 0.06 m，且候选偏移只在起点到终点的垂直方向上采样。它足以证明 seed 影响了路径，但在 224/512 画面、远距离视角和机械臂遮挡下不一定明显。

### 4.5 Oracle 固定目标物体

`agent=oracle` 会始终选择协议 GT 目标，故不会出现目标物体选择差异。这是语义正确性要求，不应通过随机换目标来伪造轨迹多样性。

## 5. 随机性改进方案

### 5.1 推荐的分阶段路线

#### 方案 A：扩大并结构化 approach 随机化，优先实施

保留“只增强自由空间”的安全边界，但将单个中间点扩展为可配置的多段安全路径：

- 在 approach 早期加入横向预对位点。
- 在目标上方保持安全高度，再加入一段横向偏移。
- 最后从目标正上方进入，保证 descend 的接触几何不变。
- 偏移范围从 0.06 m 做小规模网格试验，例如 0.06、0.09、0.12 m，而不是一次性放大。
- 保持目标上方安全高度、workspace、clearance、速度和有限 action validator。

优点是改动小，能明显增加视频中的横向差异；缺点是后半段仍然一致。

#### 方案 B：从“第一个可行”改成“多样性优先的可行候选选择”，推荐与 A 同时实施

候选全部验证后，不再立即返回 candidate 0。为每个候选计算：

- 中间 waypoint 到基准直线路径的最大横向距离。
- waypoint 与已采集同 layout 轨迹的最小距离。
- 路径长度和速度余量。
- clearance 余量。

在满足硬约束的候选中，使用 seed 决定的 tie-break 选择最大安全偏移，或选择与已有轨迹最远的候选。不能只按路径长度最大化，否则可能产生不必要绕路。

第一步可以只实现“所有候选验证后选择横向偏移最大的候选”，不需要引入复杂 planner。

#### 方案 C：对抓取前预对位段做受控随机化

不要随机化真正接触阶段，而是在 descend 之前增加短暂的 `pre_grasp_align`：

1. 在较高安全高度随机横向到达目标上方区域。
2. 进入目标正上方的 canonical pose。
3. 继续现有 descend/close/lift。

这样能让视频看到更长的差异，同时保留接触段的稳定性。需要新增 atomic/stage 语义，或者将其明确标记为 `REACH_ABOVE` 的子段，避免破坏当前抓取顺序验收。

#### 方案 D：对 lift 后自由空间段做随机化

抓取成功后，在安全抬升高度以上随机选择短暂的 lateral hold/retreat，再结束 episode。这个方案最容易让后半段可见不同，但会改变任务结束前的状态，必须明确成功判据是在 lift 成功时冻结，还是要求随机 retreat 后仍保持成功。

#### 方案 E：任务级随机选择

不建议在当前 Phase5 直接随机 oracle 目标物体。那会把“轨迹随机性”与“任务选择正确性”混在一起，导致同一 layout 的 episode 语义不同。若以后需要评估随机策略，应单独使用 `agent=random` 并单独报告选择成功率。

## 6. 推荐实施顺序和验收门

### 第一步：先提高可见性，不改变接触语义

- 候选数默认提高到 8，每个候选默认含 2 个中间 waypoint。
- 最大 lateral offset 做 0.06/0.09/0.12 m 三档小规模真实测试。
- 已实现“验证全部候选后选择安全横向偏移最大者”。
- 记录 `path_length_m`、`max_lateral_deviation_m`、`clearance_margin_m`、`speed_margin_m`。

### 第二步：加入 pre-grasp align

- 将 approach 拆成 `approach_free_space -> pre_grasp_align -> descend`。
- 只允许前两个无接触段随机化。
- `descend -> close -> lift` 继续使用 canonical target pose。

### 第三步：在成功率稳定后扩展 lift 后段

- 仅在抓取成功后随机化 retreat/hold。
- 将 success 判据与 retreat 行为解耦，并在 manifest 中记录阶段级结果。

### 建议的 Phase5 质量门

固定 layout、K=5 之外，增加以下定量门：

- 至少 4/5 条 episode 的 `augmentation_enabled=true`。
- 至少 3 个不同的离散 path signature，而不是仅要求 EEF 数组存在微小差异。
- `max_lateral_deviation_m` 的中位数达到配置目标的 50% 以上。
- 增强组成功率相对未增强基线下降不超过 5 个百分点。
- fallback 率、IK 拒绝率、clearance 拒绝率分开统计。
- 接触顺序保持：`DESCEND_TO_GRASP -> GRIPPER_CLOSE -> LIFT_OBJECT`。
- 同 layout 的初始对象位姿完全一致，layout seed 与 trajectory seed 不串扰。
- 视频只作为人工辅助证据，最终以 H5 action、实际 EEF、metadata 和 validator 结果为准。

## 7. 运行和排查顺序

不启动 Isaac Sim 的离线排查：

```bash
cd /home/elm/projects/geomind_lab
python3 -m py_compile scripts/debug_protocol_anchor_reference.py scripts/collect_anchor_reference_trajectory_dataset.py
python3 -m unittest tests/test_trajectory_augmentation.py tests/test_trajectory_resample.py tests/test_trajectory_segmenter.py tests/test_trajectory_validator.py
```

真实运行后按以下顺序看文件：

1. `manifest.json`：配置、run_status、layout returncode、collection_complete、episode H5 是否存在。
2. `layout_status.json`：子进程返回码和预期 H5 路径。
3. `console.log`：先看 `[anchor_debug_stage]`，再看 traceback。
4. `phase5_check_report.json`：检查是数据缺失、布局不一致、路径不多样还是接触顺序失败。
5. `augmentation_record`：检查 seed、候选、拒因、实际 waypoint 和 fallback。
6. H5：比较 `/actions/eef_abs` 与 `/observations/eef_pose_world`，不要只看 MP4。

Isaac Sim 必须在独立 tmux 会话中启动；启动前先执行：

```bash
export MDL_SYSTEM_PATH="${MDL_SYSTEM_PATH:-/home/elm/projects/InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes/Materials}"
```

批量任务使用单任务单 tmux 串行执行，避免多个 Isaac 进程并发占用 GPU。
