# 中点放置真实物理验证

2026-11-09。对已生成的场景65两例中点放置任务补齐本地IsaacLab/PhysX执行；先seed0单例，成功后seed1及偏移8cm错误目标对照，每个worker限时420秒，GPU串行，未失败重试。

## 实现与判据

复用IsaacLabEpisode、既有Franka IK/关节位置控制、场景加载器、runner监督限时和录像。新增PickPlaceMidpoint控制与物体根状态轨迹导出；实际夹爪接触抓取、抬升、搬运、松爪、退离和稳定。执行期间没有物体姿态写入、刚性吸附或外力搬运。

目标冻结于查询时实测红蓝中心的中点，独立GT检查。目标物体需抬起>6cm；松爪后夹爪开度每指>3cm，中心误差<=1cm、标称桌面支撑高度误差<=4mm、线速度<=1cm/s、角速度<=0.1rad/s、参考物体位移<=5mm，连续>=60步(0.5秒)。本次末尾采样120步，两例均全部通过。支撑验收为高度与静止条件，不是直接接触传感证明；不认证全程无碰撞。

|任务|实际结果|物体末态误差|稳定步数|
|---|---|---:|---:|
|InferMidpointPlacement_s0_draft|成功|1.020mm|120|

证据目录：`outputs/research/midpoint_placement_physics_single_v1`；原始报告 physics/report.json、每步 traces/*.npz、物体姿态速度 *_objects.npz、动作与关节 *_controls.npz、physics/placement_audit.json、physics/videos/。
源码快照ID `29c5d2509c7d674bafeb99a0d76512c0e9ac85e78d7d9a63f9bfa8c1cfae8529`；配置ID `236bc76b5a2d4faedbf12a8590d86daf1c728a791e85e7adb024dea0fc31131f`；commit `8ab25dd2b8ddbc3abdfe0017b45f9da971f4f5de`，未提交实现内容已保存source_snapshot。来源检查均通过。

|InferMidpointPlacement_s1_draft|成功|1.043mm|120|
|InferMidpointPlacement_s1_draft 错误对照|按预期拒绝|79.693mm|0|

证据目录：`outputs/research/midpoint_placement_physics_second_v1`；原始报告 physics/report.json、每步 traces/*.npz、物体姿态速度 *_objects.npz、动作与关节 *_controls.npz、physics/placement_audit.json、physics/videos/。
源码快照ID `412f589a9c2d3ed29a722edcf8bbc3a2e164bb81cfa940f91a5ebf2dab258388`；配置ID `64e6af811b93bd1159918dc22a23d1f6858d6521ff67456ffbb247eb67d4c163`；commit `8ab25dd2b8ddbc3abdfe0017b45f9da971f4f5de`，未提交实现内容已保存source_snapshot。来源检查均通过。

## 校验与边界

正例2/2成功，错误对照1/1正确拒绝；物理轨迹独立复算3/3通过（查询证据/固定GT、时钟、关节目标与实测关节限制、抬升、松爪后速度与位置/稳定）。录像6/6通过帧数、首尾帧解码和哈希检查，并查看两例末帧。51项受影响测试通过，包括新物理契约2项、既有物理监督/语义/流水线测试；不代表所有操作任务覆盖。无system_failure、invalid_instance或implementation_failure，错误对照semantic_error/action_failure为预期注入。

本次为真实物理执行，后续审计是离线复算；oracle_gt不代表模型成绩。应用完成报告后采用有界清理。只支持本次3cm立方体、场景65两布局的证据，不代表所有几何任务或任意物体抓放。通用research_pipeline的manipulation默认入口仍拒绝，尚未把此窄范围执行扩展为完整自动生成入口。

## 可执行命令

```bash
/home/elm/miniconda3/envs/new_isaaclab/bin/python -m geomind.tools.run_episode_physics --root outputs/research/midpoint_placement_physics_single_v1/configs --episode InferMidpointPlacement_s0_draft --output NEW_OUTPUT_SINGLE --timeout-s 420 --video
/home/elm/miniconda3/envs/new_isaaclab/bin/python -m geomind.tools.run_episode_physics --root outputs/research/midpoint_placement_physics_second_v1/configs --episode InferMidpointPlacement_s1_draft --output NEW_OUTPUT_SECOND --timeout-s 420 --video --negative-controls
python -m geomind.tools.audit_midpoint_placement --run outputs/research/midpoint_placement_physics_single_v1
python -m geomind.tools.audit_midpoint_placement --run outputs/research/midpoint_placement_physics_second_v1
```

重跑使用新输出目录，原证据保持。下一最小实验（未执行）：单例增大绿色物体初始横向偏移，验证抓取裕量与参考物体保持，再决定是否接通自动生成入口。
