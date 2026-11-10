# GeoMind 机械臂本体与使用说明

更新：2026-11-10。本文介绍 GeoMind 原生 IsaacLab/PhysX episode 后端已接入的机械臂；已有资产不等于已支持全部任务。默认从仓库根目录运行。

## 已接入配置

目前为 **3 类机械臂、5 种配置**。自由度仅计手臂，不含夹爪。

| 配置 | `scene.robot` | 手臂自由度 | 已验证范围 | 配置与证据根目录（相对仓库） |
|---|---|---:|---|---|
| Franka Panda＋原装夹爪 | `franka` | 7 | target 到达；固定实例抓放及组装蓝罐的真实搬放验证 | `outputs/research/fr3_panda_target_v1/panda`，新增组装动作见下节 |
| Franka FR3＋原装夹爪 | `fr3` | 7 | target 到达；初始关节角对齐 Panda | `outputs/research/fr3_panda_target_matched_home_v2/fr3` |
| Panda＋Robotiq 2F-85 | `franka_robotiq` | 7 | target到达；20注册实例真实抓放已逐例验收 | [重生成与动作证据](research/experiments/2026-11-10-registered-robotiq-regeneration.md) |
| Piper 单臂＋AGX 原装夹爪 | `piper` | 6 | target 到达；历史测试带 40 cm 安装支架 | `outputs/research/piper_target_v1` |
| Piper 双臂＋AGX 原装夹爪 | `piper_dual` | 6＋6 | 左右臂独立到达、待机保持和双臂限位复核 | `outputs/research/piper_dual_edge_v3` |

上述目录中 `configs/` 保存实际输入，`physics/report.json` 保存物理结果。Panda 抓放的独立证据见[中点放置验证](research/experiments/2026-11-09-midpoint-placement-physics.md)与[场景更新抓取验证](research/experiments/2026-11-09-scene-updating-grasp.md)。这些是有限实例验证，不能泛化为通用抓放能力。

FR3与Piper尚未完成抓放验证。Robotiq新增组装场景接触动作适配：只控制`finger_joint`，按USD的mimic/闭链约束带动其他关节，以两个outer knuckle的实测角度及几何间隙验收夹爪状态；单位为弧度。140毫米TCP和夹持高度来自实际夹爪资产。支持接口不等于逐任务通过，当前批次结果见[20项重新生成入口](assembled_tasks_zh.md#重新生成20个注册任务)。双臂目前每次执行一条手臂，另一条受控保持初态；尚未支持同步协同抓放或物体交接。UR、Kinova、Sawyer、OpenArm、ALOHA 和人形等不计入本次已接入清单。

## 所有本体的手腕相机

原生 episode 后端为全部 5 种机器人配置创建随手腕运动的 Camera prim；双臂始终创建左右两路，相机不随 active_arm 选择而丢弃。挂载配置集中在 `geomind/episodes/wrist_cameras.py`，实际路径须对应已加载模型的刚体，路径不匹配会明确拒绝。

| 本体配置 | 相机的父刚体路径（相对机器人根） | 录像视角名称 |
| --- | --- | --- |
| Panda | `panda_hand` | `wrist` |
| FR3 | `fr3_hand` | `wrist` |
| Panda＋Robotiq | `Robotiq_2F_85_edit/Robotiq_2F_85/base_link` | `wrist` |
| Piper 单臂 | `gripper_base` | `wrist` |
| Piper 双臂 | 两个机器人根下各自的 `gripper_base` | `wrist_left`、`wrist_right` |

机器人根分别为 `/World/Robot`、双臂右侧的 `/World/RobotRight`，相机作为对应刚体下的 `wrist_cam` 子 prim。单臂配置的局部位置为 `[0.13, 0, -0.15] m`；双臂 Piper 使用 v2 配置 `[-0.13, 0, -0.15] m`，将镜头移到倾斜手腕上方，并将光学滚转旋转 180°，使画面朝上。两者光轴均朝向各本体指尖前方 15 cm；ROS 光学坐标使用 +Z 向前、−Y 向上，四元数为 wxyz。镜头为 16 mm、水平孔径 24 mm，裁剪距离 0.015–20 m。这是仿真观察配置，未声称为真机标定值，也不额外引入相机外壳的质量和碰撞。

使用原来的 `run_episode_physics --video` 即自动录制外部相机和全部手腕相机，不需要新增开关。未开启录像时仍创建挂载 prim，但不启动 RGB 渲染产品。旧的外部 `*_observer.mp4` 与 `*_agent.mp4` 字段保留；新增 `*_wrist_observer.mp4` / `*_wrist_agent.mp4`，双臂带 `wrist_left`、`wrist_right`。所有视角遵循相同的物理帧时钟，agent 版本共同应用原来的可见性遮挡。

报告的 `video.wrist_cameras` 保存各路录像、内参、挂载规格和文件哈希。各路 `*_frames.jsonl` 逐帧保存实际相机世界位姿、父刚体实测位姿和固定局部外参；Isaac Sim 5.1 的 Fabric 模式直接读取实时相机世界变换，避免把初始化时的静态 USD 位姿当成动态外参。双臂到达诊断额外比较实测画面上方向与重力上方向在像平面的投影，拒绝倒置或滚转超过 60° 的可观测帧；接近正上/正下的帧单独标记为不可观测。独立审计入口为 `python -m geomind.tools.verify_wrist_cameras audit --physics PHYSICS_DIRECTORY`。

具体逐本体实测状态与失败记录见[手腕相机场景修正与验证记录](research/experiments/2026-11-10-wrist-scene-corrections.md)，[最新录像页](http://127.0.0.1:8781/outputs/research/wrist_cameras_delivery_v2/index.html)采用四个单臂 v7 与两个双臂 v8 实例，最终诊断 6/6 通过。双臂桌面相机诊断将基座前移 10 cm，原安装配置下的两次到达失败保留；这是本次诊断变体，不替代下节历史桌边配置。相机随动诊断不扩大 FR3/Piper 的抓放能力声明。

## 当前采用的双臂 Piper

- 两侧基座直接安装在场景 65 桌面，无高支架；世界坐标为 `x=-0.11 m`、`y=±0.22 m`、`z=桌高`，底座后缘距桌后沿约 7 cm。
- 夹爪接近方向向下倾 20°，左右各向中间偏转 25°，采用镜像初始关节姿态。
- TCP 为 `gripper_base` 局部 z 方向偏移 `0.138 m`。保持原 target 上方 10 cm 的最终目标；中间接近点再高 5 mm，以避免桌边安装后的过度伸展。
- 左右正例误差约 0.58 / 0.37 mm，均连续驻留 20 步；错误目标对照及双臂状态复核通过。结果仅对应所保存的两个布局。

来源为松灵官方 URDF：`assets/robots/agilex_arm_source/`；准备后的 URDF、USD 与来源清单在 `assets/robots/piper_prepared/`。使用关节位置高 PD 控制，机器人链节关闭重力；这不是经过标定的真机动力学模型。

## 运行双臂测试并录像

本机使用 `/home/elm/miniconda3/envs/new_isaaclab/bin/python`，运行环境为 Isaac Sim 5.1 和仓库自带 IsaacLab。需可用 NVIDIA GPU、本地机器人 USD 和场景资产。迁移机器时需重新配置资产路径。

以下命令复制当前已验证配置到新目录，再执行左右两例及错误目标对照，不覆盖历史结果：

```bash
mkdir -p outputs/research
RUN_DIR=$(mktemp -d outputs/research/piper_dual_replay_XXXXXX)
cp -a outputs/research/piper_dual_edge_v3/configs "$RUN_DIR/configs"

/home/elm/miniconda3/envs/new_isaaclab/bin/python \
  -m geomind.tools.run_episode_physics \
  --root "$RUN_DIR/configs" --suite piper_target \
  --output "$RUN_DIR/physics" --device cuda:0 --timeout-s 420 \
  --video --negative-controls \
  --camera-eye 1.7 0 1.85 --camera-look-at .4 0 1.12

python -m geomind.tools.research_pipeline --audit "$RUN_DIR"
python -m geomind.tools.audit_dual_piper --run "$RUN_DIR"
```

GPU 任务串行运行。命令使用当前源码重新执行，不保证与历史源码快照逐位一致。runner 自动保存物理证据，但不会自动生成完整源码快照；正式研究记录还需保存命令、配置和相关源码 provenance。

只测单侧时，将 `--suite piper_target` 换为：

- `--episode ProjectOntoLine_v0`：左臂。
- `--episode ProjectOntoLine_s1_v0`：右臂。

单侧运行可做末端审计；`audit_dual_piper` 要求左右两侧都实际执行，因此用于完整双侧测试。

## 修改本体、执行臂和位置

在复制后的配置中修改，保留已验证原配置：

| 位置 | 字段或用途 |
|---|---|
| `configs/scenes/registered_scene_65.json` | `robot` 选择本体；`physics.robot_base_w_m` 设置基座世界坐标，单位 m |
| `configs/episodes/<episode_id>.json` | 双臂用 `physical_execution.active_arm` 显式选择 `left` 或 `right` |
| `configs/episodes/<episode_id>.json` | `physical_execution` 定义目标偏移、容差、驻留与运动预算 |
| `configs/suites/piper_target.json` | 要执行的 episode 列表 |

双臂的 `robot_base_w_m` 指左臂；右臂保持相同 x、z，并将 y 取相反数。`active_arm` 与 seed 无关。自然初态和左右工具四元数目前在源码中定义，并非 JSON 可调项。

切换本体时优先复制表格中对应证据目录的完整 `configs/`，不要只替换 `robot`：不同本体需要匹配的基座、TCP、姿态和路径；单臂 Piper 的历史支架方案也不能直接当作桌面双臂方案。

## 查看结果

| 产物 | 内容 |
|---|---|
| `physics/report.json` | 正例、负对照与 `physical_execution_verified` 汇总 |
| `physics/cases/*.json` | 单例目标、误差、姿态、驻留和执行步数 |
| `physics/videos/*_observer.mp4` | 完整执行录像；`*_agent.mp4` 为按可见性规则遮挡的录像 |
| `physics/traces/*_controls.npz` | 活动臂控制目标、实测关节角与速度 |
| `physics/traces/*_dual_controls.npz` | 双臂各自的控制目标、实测状态、限位和物理时钟 |
| `physics/offline_audit.json` | 目标、限位、姿态、误差及驻留的离线复核 |
| `dual_audit.json` | 双臂限位、时钟与待机保持复核 |

负对照的 `success=false` 是预期结果，还需查看 `verification_passed=true`。有录像或末端到位不能代替全部物理验收；本次到达测试未认证无碰撞，也不是 VLA 模型能力评估。

已整理的视频页可用下面的命令重新启动，再打开 `http://127.0.0.1:8780/`。它展示保存的历史结果，不会重跑仿真：

```bash
python -m http.server 8780 --bind 127.0.0.1 \
  --directory outputs/research/piper_dual_edge_v3
```

本体与 TCP 绑定见 [robot_profiles.py](../geomind/episodes/robot_profiles.py)，运行后端见 [isaaclab_physics.py](../geomind/episodes/isaaclab_physics.py)。位置调整及失败记录见[双臂桌边实验](research/experiments/2026-11-09-piper-table-edge.md)。

## 接入组装任务与geomind-research

2026-11-10新增 `assemble_solve_execute` 阶段，机器人选择不再只保存在独立历史配置里。agent可查询本体与动作兼容菜单，提交场景、布局、任务、机器人、世界基座位置和动作，由同一研究入口生成GT并执行：

```bash
python -m geomind.tools.research_pipeline --assembly-options
python -m geomind.tools.research_pipeline \
  --assembled-request configs/scene_assembly_tasks/panda_pick_scene65.json
```

请求显式增加robot和execution，例如：

```json
"robot": {"embodiment": "franka", "base_position_w_m": [0, -0.45, 0.9328504589976385]},
"execution": {"action": "PickLiftSelectedObject", "negative_controls": true, "timeout_s": 420}
```

该坐标是65号桌面示例，不是所有场景通用位置。完整请求另需stage、任务、场景来源、output及selection，见[组装任务协议](assembled_tasks_zh.md)。示例执行后原目录拒绝覆盖，再运行需更换output或使用网页生成的replay请求。

本轮组装接触动作入口以Panda为边界：SelectByOrder使用PickLiftSelectedObject；几何目标放置使用PickPlaceObject。其他四种配置保留已有资产与到达诊断能力，严格拒绝在这个自然操作阶段替代为到达上方。双臂仍需显式活动臂，未实现协同抓放。

AssembledScene直接组合已保存场景并绑定原物体prim，不重新生成方块、辅助台面或二次随机布局。机器人reset后读取真实网格包络中心重新求解GT并固定；动作使用关节位置与夹爪接触，不在执行期写物体姿态、吸附或外力搬运。Panda夹爪闭合量依据实际包络宽度，接触阻尼及求解迭代写入运行时来源记录；不是经过真机标定的动力学模型。

每例在 `robot_configs/` 保存独立机器人契约；`robot_physics/` 保存实际动作视频、运行时查询GT、原始物体和控制轨迹、正例和错误对照。`--audit-assembled RUN`同时复核原组装GT及真实动作数据。实际成功/失败与限制见[本轮机器人组装实验](research/experiments/2026-11-10-assembled-robot-actions.md)；查看[机器人任务证据页](http://127.0.0.1:8780/tasks.html)，初始落稳与机器人动作录像分开展示。
