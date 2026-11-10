# 三个空间任务的无头物理执行验证

日期：2026-11-05。设备：NVIDIA RTX 3090，CUDA 设备 cuda:0。
运行环境：new_isaaclab，Isaac Sim 5.1.0，真实 PhysX 与 Franka Panda 关节位置驱动，无 GUI。

后续已重新运行 9 个正例并录制视频，最新数据位于 `outputs/headless_physics_video/`，
完整包为 `outputs/headless_physics_video.zip`。前轮无视频报告保留在 `outputs/headless_physics/`。

## 实际结果

| 任务 | 正确动作 | TCP 位置误差范围 | 错误目标对照 |
|---|---:|---:|---:|
| DetectSceneChange | 3/3 | 5.54–5.72 mm | 1/1 被拒绝 |
| ProjectOntoLine | 3/3 | 4.85–6.23 mm | 1/1 被拒绝 |
| ReflectAcrossLine | 3/3 | 4.73–7.52 mm | 1/1 被拒绝 |

9 个正确实例全部通过，平均 TCP 误差 5.80 mm。每次均满足 12 mm 位置容差、
0.15 rad 姿态容差和连续保持 20 个物理步。3 个错误目标在语义和物理评分中均被拒绝。
错误目标最终对正确目标的 TCP 距离分别约 73.99、72.29、270.73 mm，均超过容差。

策略为显式 oracle_gt，用来验证 GT 到动作再到物理成功判定的链路；这些结果不代表模型推理成绩。

## 物理执行的定义

当前动作是 ReachAboveTarget：Franka TCP 到达语义目标上方 0.10 m，并保持向下姿态。
点任务使用查询时求出的实际投影/反射点；物体选择任务绑定正确物体的当前实测位置。
动作经过抬升、水平接近、下降三个阶段，使用 PhysX Jacobian、阻尼 IK 和关节位置驱动。
当前没有执行抓取、释放或放置。

程序化场景的桌面顶面为世界 z=0.75 m，任务坐标原点为世界 `[0.5,0,0.75]`，机器人基座为 `[0,0,0.75]`。
场景模式在 `scenes/semantic_table.json` 的 physics 中显式声明为 ProceduralTable，不加载完整场景 65 USD。
几何任务的三维点由固定辅助平台支撑，任务方块本身是带质量、碰撞和重力的动态刚体。
memory 方块直接位于平坦桌面上；无初始位置标记，初始与移动后的坐标来自同一范围。

Franka 使用标准高 PD 配置，其本体关闭重力。任务方块保留重力，reset 后先真实落下 2 cm 并稳定，
然后才开始任务计时。历史与当前位姿都读自 PhysX，布局中的理想坐标不作为物理 GT。

Memory 的 MoveObject 由外力 PD 驱动，连续推进 PhysX，未直接写入移动目标的位姿。
三个 memory 实例的目标实测位移约为 188.72、205.48、171.87 mm；其他候选位移不超过 0.00003 mm。
观察结束于物理 tick 240，查询位于 tick 420；dt=1/120 s，协议一步=60 个物理步。
公开观测在延迟阶段不包含对象状态，也不包含历史、GT 或干预计划。

## 可审核的证据

- `outputs/headless_physics/report.json`：9 个正例、3 个负例、语义/物理评分、真实误差、来源哈希。
- `outputs/headless_physics/cases/`：逐实例详细报告。
- `outputs/headless_physics/evidence/`：查询实测快照、历史快照和物理时钟。
- `outputs/headless_physics/traces/`：逐物理步 NPZ，包含关节角、TCP 世界坐标、任务刚体世界坐标和阶段。

动作过程中没有写入关节状态或对象位姿；reset 的初始化写入单独处理。
实测动作期最大相邻步关节变化约 0.00185 rad，可从轨迹独立检查，不以瞬移代替执行。
真实物理证据已使用普通 Python 离线回放，恢复相同 memory 目标及证据哈希。
12 条轨迹还经过独立离线审计：快照与对应 PhysX tick 一致、末端误差可由轨迹复算、错误目标均超过物理容差。
物理执行阶段系统回归测试共 **190 项通过**，`git diff --check` 通过，本次创建的仿真进程均已退出。

## 复现

```bash
conda activate new_isaaclab
python -m geomind.tools.generate_spatial_suite --seeds 0 1 2 --verify
python -m geomind.tools.run_episode_physics \
  --suite generated_spatial_suite --negative-controls \
  --output outputs/headless_physics
```

只运行一个实例：

```bash
python -m geomind.tools.run_episode_physics \
  --episode ProjectOntoLine_v0 --output outputs/headless_physics_single
```

API 入口：

```python
from geomind import create_env

env = create_env("DetectSceneChange", backend="isaaclab", headless=True)
observation, info = env.reset()
while not observation["allow_response"]:
    observation, *_ = env.step()
# 将 agent 的身份选择交给机器人执行；评估器单独保留 GT。
observation, reward, terminated, truncated, info = env.step({"object_id": selected_object_id})
```

推荐批量使用上述 CLI，它在独立 worker 中运行仿真。当前主机的 Kit 会在关闭阶段阻塞，
父进程在完整报告落盘后给予 10 秒关闭时间，再清理只属于本次运行的进程组。
报告中 `shutdown: bounded_cleanup_after_report`、worker_exit_code=-9 记录这一情况；
父进程根据已经完成的物理验证返回退出码。执行未完成或新 worker 失败时会覆盖为失败报告，旧成功文件不会被复用。

物理证据回放：

```bash
python -m geomind.tools.resolve_episode DetectSceneChange_v0 \
  --root geomind/configs \
  --state outputs/headless_physics_video/evidence/DetectSceneChange_v0.json
```

## 视频数据

重新执行的 9 个实例全部成功，录制为 1280×720、30 FPS、H.264 MP4，无音轨，按仿真时间正常播放。
共 18 个 MP4：每个实例分别有完整 observer 视频和按协议遮罩的 agent 视频。
所有视频完成逐帧解码检查，共 5580 帧；三个 memory agent 视频各有 45 帧（1.5 秒）隐藏区间，合计 135 帧。

`dataset_manifest.json` 为可移植索引，路径相对于数据集目录。它包含查询指令、encoding 指令、
查询开始帧、视频路径、校验和、相机标定关联与轨迹关联。每个 `*_frames.jsonl` 将视频帧映射到 PhysX tick。
`configs/`、`source/` 保留本轮配置与实现快照。observer、GT 与 evidence 是评估端数据；模型视觉输入使用 agent 视频。

```bash
python -m geomind.tools.run_episode_physics --video --timeout-s 1800 \
  --output outputs/headless_physics_video
python -m geomind.tools.package_episode_videos --root outputs/headless_physics_video
```

ZIP 约 102 MB，旁边的 `.sha256` 文件可用于校验完整性。

能力查询区分 isaaclab_episode 与 isaaclab_legacy。旧兼容注册的 placeholder 状态没有被错误改成旧协议已实现；
三个任务的原生物理能力现在为 true。旧 ManagerBased 运行脚本不自动转换成此独立 episode 执行器，
请使用本页的原生 CLI 或 create_env API。

## 验证范围

本轮验证了程序化桌面上的真实物理执行、连续机器人运动、记忆干预、实际 GT 和错误目标拒绝。
未验证完整场景 65 的背景资产、图像策略输入、视觉指代、抓取/放置或真实机器人硬件。
