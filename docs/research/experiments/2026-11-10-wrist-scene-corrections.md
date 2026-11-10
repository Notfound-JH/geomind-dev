# 手腕相机测试的场景、支撑与双臂朝向修正

2026-11-10。用户查看五种本体的腕相机录像后指出三项问题：墙体缺失、物体看似浮空、双臂 Piper 图像倒置。本轮已修正并实际重执行：最终五种本体、六个活动臂诊断 **6/6 通过**，四面源墙及米色纹理、真实桌面支撑、固定挂载与相机随动均通过核验，双臂图像方向通过独立审计。共实际运行 8 例，包含保留的 2 例到达失败及最终 6 例通过；最终 69 项受影响 CPU 检查通过。最终采用 v7 的四个单臂配置与 v8 的双臂两例，版本选择明确记录。

[新录像网页](http://127.0.0.1:8781/outputs/research/wrist_cameras_delivery_v2/index.html)集中展示真实视频、逐例指标、旧失败与配置改动；[机器可读汇总](../../../outputs/research/wrist_cameras_delivery_v2/summary.json)保存实际来源与审计详情。浏览器 UI 当前不可用，未声称完成页面播放按钮、交互或窄屏浏览器验证。

## 旧结果的实际边界

`outputs/research/wrist_cameras_v5` 的六个活动臂诊断曾通过固定挂载位姿重组、相机随动、渲染媒体与录像时钟核验。这些检查没有覆盖场景墙体完整性、真实桌面支撑或图像朝上方向，不能据此认定画面场景已完整还原。旧录像、报告及失败尝试保留，不覆盖为新结果。

六个实例对应 Panda、FR3、Panda＋Robotiq、Piper 单臂，以及 Piper 双臂分别活动左右臂。动作仍是显式 `ReachAboveTarget` 相机诊断；本轮不将其升级为抓放或视觉模型评估。

## 已确认的原因及修正

### 墙体缺失

原生 `RegisteredScene` 运行时通过 `GeoMindSceneCfg.apply_scene` 重建选定资产。旧墙体提取只识别 Cube，而当前场景 65 的四面墙是带纹理材质的 Mesh，因此运行资产列表只有桌子、桌布、地板和灯光。黑色区域是缺少墙面的背景，并非用户选择了黑墙。

`geomind/episodes/registered_scene.py` 现在以过滤后的源 USD 引用加载墙体，保留源墙的 Mesh/Cube 几何、层级变换、碰撞、材质绑定和纹理路径；排除该引用中的额外桌子、原始道具与灯光，避免重复加载。运行时 provenance 记录每面墙的源 prim、实际 prim、几何类型、碰撞开关、材质和可解析纹理。场景 65 的墙面来源仍是实际源文件所绑定的米色 `beige_wall_001`，没有另造黑色墙面。

### 方块浮空

旧相机测试直接复制三维投影诊断布局。`geometry_s0` 中三个 3 cm 方块的中心在桌面上方 17.95098、22.04902、20.07393 cm；执行器在方块下放置厚 2 cm 的运动学薄板，薄板本身没有连接桌面的支撑柱。因而低速度的落稳可以通过，但方块并未受到真实桌面的支撑。

从旧原始物体状态轨迹独立计算的最终方块底部离桌间隙如下，单位为 cm：

| 旧诊断实例 | 红方块 | 蓝方块 | 绿方块 |
| --- | ---: | ---: | ---: |
| Panda、FR3、Panda＋Robotiq、Piper 单臂、双臂左活动 | 16.451 | 20.549 | 18.574 |
| 双臂右活动 | 15.756 | 21.244 | 18.460 |

旧实例最终速度约 0.12–0.21 mm/s。这说明“速度小”不能代替“桌面支撑正确”。单独的 CompleteSquare 腕相机抓放示例采用另一条组装场景链：其最终实测物体底部与桌面约 −0.605 mm，释放后稳定 1 秒，没有上述十余厘米的辅助薄板悬空问题。

v7 显式生成命名为 `*_camera_tabletop` 的新布局变体，保留 XY、物体、seed、本体及到达语义，仅将三个方块中心的 Z 改为尺寸的一半，即桌上 0.015 m。原布局保留；Z 变化意味着该变体的几何目标变化，因此从当前 PhysX 落稳状态重新求解并固定 GT，不沿用旧三维实例的 GT 或历史语义证据。

运行时对桌面方块不再创建或停放辅助支撑 collider，必须使用实际桌面网格。provenance 记录实测桌面高度、边界、三角面、`helper_supports.enabled=false` 及 `real_table_only` 支撑策略。

新增 `geomind/tools/audit_wrist_scene.py` 从原始 13 维刚体状态重算四元数对应的方块底高与保守足迹。验收要求全部落稳后状态的底部离桌误差不超过 ±2 mm，足迹九点落在实测桌面三角网格内，最后至少 0.5 秒速度不超过 1 mm/s，并确认不存在辅助支撑 prim。该窄范围判据只用于物体应始终留在桌面的到达诊断，不能直接用于抓放动作。

### 双臂 Piper 图像倒置

双臂 Piper 的夹爪采用向下 20° 的倾斜接近姿态，其局部 +X 朝下，与单臂朝下抓取姿态下的相机安装约定不同。所有本体复用同一侧安装及光学 roll，导致双臂图像朝向反转，固定挂载随动审计仍可能通过。

双臂使用 `profile_version=2`：相机位置改到手部局部 −X 一侧，并使用相反的固定光学 roll；相机仍是对应 `gripper_base` 刚体的子 prim，光轴朝向指尖前方，ROS +Z 朝前、−Y 朝上。修正发生在真实相机局部外参，未翻转已录制视频，也未引入世界位姿跟踪。

新增朝向审计从实测 ROS 相机姿态得到图像向上轴，与重力向上在图像平面的投影比较。略过朝向近乎竖直而 roll 不可观测的帧，要求存在可观测帧且图像向上方向一致；精确阈值与逐例数值保存在 `wrist_camera_audit.json`。

## 验证状态与复现

已执行：首次 68 项受影响 CPU 检查通过，覆盖原生场景/墙体、桌面布局变体、支撑审计及双臂相机方向等相关接口；运行后补充默认双臂安装配置回归检查，最终 69 项通过，日志位于交付目录 `cpu_validation.log`。CPU 通过不计为物理执行通过。

物理重执行：`outputs/research/wrist_cameras_v7` 的六例及 `outputs/research/wrist_cameras_v8` 的两个双臂重试，GPU 串行、每个 worker 最长 420 秒。保存逐例独立配置、启动命令、源码快照、来源哈希、原始关节和物体状态、外部及腕部录像。与 v5 对比的是显式改变 Z 的桌面诊断变体，不声称严格复现旧三维布局。

运行后的独立审计入口：

```bash
python -m geomind.tools.verify_wrist_cameras audit --physics CASE_DIRECTORY/physics
python -m geomind.tools.audit_wrist_scene --physics CASE_DIRECTORY/physics
```

## 实际物理结果

下表依据最终显式选择版本的 `result.json`、`physics/report.json`、`wrist_camera_audit.json` 和 `wrist_scene_audit.json`，不复用旧 v5 的通过数。每例实际到达连续驻留 20 步；保留原 12 mm 位置、0.15 rad 朝向容差及动作预算。

| 最终实例 | 来源 | TCP 位置误差 mm | 最大绝对底高误差 mm | 末 0.5 秒最大物体速度 mm/s | 墙体及材质 | 挂载/支撑/朝向 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Panda | v7/franka | 5.716 | 1.111 | 0.7489 | 四面源墙通过 | 通过 |
| FR3 | v7/fr3 | 5.650 | 1.111 | 0.7489 | 四面源墙通过 | 通过 |
| Panda＋Robotiq | v7/franka_robotiq | 5.394 | 1.111 | 0.3402 | 四面源墙通过 | 通过 |
| Piper 单臂 | v7/piper | 1.065 | 1.111 | 0.6118 | 四面源墙通过 | 通过 |
| Piper 双臂，左活动 | v8/piper_dual_left | 3.536 | 1.111 | 0.6117 | 四面源墙通过 | 两路均通过 |
| Piper 双臂，右活动 | v8/piper_dual_right | 1.981 | 1.200 | 0.6274 | 四面源墙通过 | 两路均通过 |

所有实例均无辅助支撑 prim，保守足迹通过实测桌面三角面核验。最大绝对底高误差 **1.200 mm**；实际数值为接触容差范围内的轻微穿透，而非原先十余厘米的离桌悬空。最大末窗速度 **0.7489 mm/s**，低于 1 mm/s 门槛。双臂两个实例的左右腕相机均通过朝上审计，最大可观测滚转 **11.89°**；左右活动臂的最大滚转分别为 11.62192°、11.88123°。

### 双臂原基座失败及显式重试

v7 的两个双臂实例已经通过墙体、支撑和相机核验，但实际到达没有通过：左活动臂 `motion_budget_exhausted`，位置误差 **85.66 mm**、朝向误差 **0.879 rad**；右活动臂为 **33.56 mm**、**0.758 rad**，两者均为 0/20 步驻留。失败报告、轨迹和视频全部保留，没有将其记作完成。

将方块落到桌面后，目标高度由约 1.233 m 降为 1.047 m。离线几何求解找到满足原目标位置及固定工具朝向的关节解，因此不能将原失败解释为目标不可达。实测路径分别进入 joint4 正、负限位附近的受限分支。`v8/planning` 保存的 FK/雅可比与 DLS 重放检查只构成配置重试依据，不是物理证据或控制器普适修复。

v8 仅把两臂基座的世界 X 从 **−0.11 m 调为 −0.01 m**，即前移 10 cm；物体 XYZ、home 关节角、工具朝向、关节限位、时间步、路点规则、动作预算和验收阈值均保留。安装底座九点通过实测桌面网格覆盖检查，随后实际 PhysX 重试两例均通过全部判据。该变化是本次桌面相机诊断的显式安装配置调整，未修改机器人控制器，也不证明已修复所有 DLS 分支问题；底座覆盖不构成整条手臂轨迹无碰撞证明。

具体数值和来源哈希见 [离线分析说明](../../../outputs/research/wrist_cameras_v8/planning/README.md)、[逐偏移量预测](../../../outputs/research/wrist_cameras_v8/planning/offline_branch_predictions.json)及最终汇总中的 `robot_mount_change` 和 `retained_previous_mount_validations`。本轮 8 次实际执行为 v7 的 4 通过、2 失败，加 v8 的 2 通过；最终交付 6/6 不等于首次尝试全通过。

### 网页重建与完成边界

以下命令明确将 v8 双臂来源覆盖进 v7 基础集合，原 v7 双臂失败自动保留在交付历史中：

```bash
python -m geomind.tools.deliver_wrist_cameras \
  --diagnostics-root outputs/research/wrist_cameras_v7 \
  --case-root piper_dual_left=outputs/research/wrist_cameras_v8/piper_dual_left \
  --case-root piper_dual_right=outputs/research/wrist_cameras_v8/piper_dual_right \
  --previous-root outputs/research/wrist_cameras_v5 \
  --output outputs/research/wrist_cameras_delivery_v2
```

服务入口为 `http://127.0.0.1:8781/outputs/research/wrist_cameras_delivery_v2/index.html`。页面来自本轮实际录像，逐例来源、媒体和物理时钟检查保存在交付证据中。最终 HTTP/媒体检查通过：88 个资源、6 份保存源码/配置快照、14 路实际相机流，详见 [verification.json](../../../outputs/research/wrist_cameras_delivery_v2/verification.json)。浏览器宿主不可用，`browser_interaction_verified=false`，未声称已验证浏览器播放交互。无视觉模型性能结论，也不扩大 FR3/Piper 抓放能力声明。

### 运行后准备器修正与来源边界

成功运行后，仅调整准备阶段默认配置生成，使今后新建的双臂桌面变体自动采用已实测的基座前移 10 cm 配置。两臂解析后的 scene/episode JSON 均与实际 v8 配置做了精确相等检查；没有在成功执行后修改物理运行时代码，也未为此再次运行物理仿真。

后续源码快照额外纳入旧运行时的 `geomind_scene_cfg.py` 与 `scene_utils.py`，使墙体提取的来源链更完整。这些文件和准备器、回归测试的运行后副本单独保存在交付目录 `post_run_source_capture/`，其哈希与时间性质记录于 [post_run_source_capture.json](../../../outputs/research/wrist_cameras_delivery_v2/post_run_source_capture.json)。这份补充捕获明确是运行后记录，不能追溯称为原执行启动时的 provenance；原逐例启动快照与结果保持独立。
