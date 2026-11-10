# CompleteSquare 机械臂沿桌边居中

2026-11-10。用户要求将偏向一侧的机械臂放到中间。本次将 Panda + Robotiq 基座沿原有桌边平移，对齐桌边中线，保留相同方块布局、seed、安装高度、基座前后位置、镜头和两份新墙面材质。

## 位置与证据

实际桌面世界 Y 边界为 `[-0.987077716242, 0.988017080112] m`，中线为 `0.000469681935 m`。基座由 `[-0.12, 0.32, 0.932850458998] m` 调整为 `[-0.12, 0.000469681935, 0.932850458998] m`，沿 Y 平移 `-0.319530318065 m`，即约 31.95 厘米。

使用实际桌面三角网格检查底座周围 16 × 16 cm 区域的中心、边中点和四角，共九个支撑采样点；该区域位于桌面内。配置与实际首段机器人 TCP 轨迹分别核对：候选 001 测得初始 TCP 平移 `[0, -0.3195302940, 0] m`，与请求平移一致。该检查验证位置与采样支撑，不认证整条机器人轨迹无碰撞。

产物保存在 `outputs/research/complete_square_centered_v1`，原墙体复测与旧视频保留。新[视频入口](http://10.81.44.63:8780/complete-square-centered/)包含真实抓放、错误落点对照、同镜头调整前后画面及基于真实桌面网格的俯视位置图。

## 实际物理结果

最终实例使用 `_centered_v2` 后缀。每个 worker 最长 420 秒、GPU 串行；240 Hz 控制、12000 动作步上限，正负例分别启动新仿真进程。机器人控制器、物体布局和动作评分阈值保持一致。

| 候选 | 初始落稳 | 正例误差 | 松爪后放稳 | 有效错误落点对照 | 状态 |
| --- | --- | --- | --- | --- | --- |
| beige_wall_001_centered_v2 | 600 / 600 步 | 0.963326 mm | 240 步 / 1 秒 | 距错误命令 1.111139 mm；距正确 GT 79.363432 mm，正确拒绝 | 独立验收通过 |
| beige_wall_002_centered_v2 | 600 / 600 步 | 0.963326 mm | 240 步 / 1 秒 | 距错误命令 1.111139 mm；距正确 GT 79.363432 mm，正确拒绝 | 独立验收通过 |

候选 001 的来源快照 ID：`27cb35733e669c8cdd228dc525aae745dfc06ae71373006125c368cb5f517c21`；配置 ID：`afdad0fff39e8f518f31af2cdfb17d8aa89314055e7bc8deea3c1545b5cd0525`。各实例完整配置、来源、原始关节/物体状态和录像哈希位于对应目录，结果以该实例的 `result.json` 和 `robot_physics/assembled_robot_audit.json` 为准。

候选 002 使用相同来源快照，配置 ID 为 `cff72627eac5f602f1b918c024e05091b6fca933b382692545e191af1e9875d7`；两例运行结束时 provenance 均通过。CPU 检查命令与四个测试文件的保存副本见 `validation_manifest.json`，网页与媒体交付检查见 `verification.json`。

交付检查已完成：4 段新正负 observer 视频完整解码通过；32 个入口及页面资源 HTTP 检查通过，两份保存源码与配置哈希完整。实际 headless Chrome 验证双视频元数据、播放/暂停按钮及 390 px 视口无横向溢出，并人工式检查桌面和窄屏截图。之前的墙体复测页与原任务对比页均加入居中版本入口，修改前 HTML 在新产物目录另存。

## 保留失败及显式对照调整

首次 `beige_wall_001_centered` 的正常任务已成功，误差 0.963326 mm；但原“抓起后放回初始位置”负控距所命令落点偏差 24.552200 mm，未通过实际错误命令执行门槛，整组被独立审计拒绝。原报告、失败审计和视频均保留，没有将动作执行失败记作有效语义拒绝。

随后显式改用 `offsets_first` 候选顺序：优先检查冻结 GT 周边原有的有界偏移，最后检查初始位置。仍使用原有 9 点桌面支撑、完整物体工作区、与保留物体/家具 6 cm 间距及与正确 GT 至少 5 cm 的距离要求。当前实例选择 GT 的世界 X `+0.08 m`；它确实完成运输、松爪、退离和放稳，再被原 GT 判据拒绝。该调整改变负控的落点设计，没有修改正常任务 GT、控制器或放宽验收阈值；也不声称解决了原位回放的执行问题。

实现增加显式选项 `run_episode_physics --negative-placement-policy offsets_first`，及 `execute_robot_stage(..., negative_placement_policy='offsets_first')`。默认仍为 `initial_first`。所有候选检查、选择顺序和来源都写入原始报告。43 项相关 CPU 检查通过，覆盖原有规划行为、偏移优先时碰撞候选拒绝、GT/初态不变、worker 完整性和独立审计；日志为产物根目录 `cpu_validation.log`。

## 复现与边界

在新目录复制并执行当前脚本，避免覆盖历史证据：

```bash
mkdir outputs/research/complete_square_centered_replay
cp outputs/research/complete_square_centered_v1/run_centered_v2.py outputs/research/complete_square_centered_replay/
python outputs/research/complete_square_centered_replay/run_centered_v2.py --candidate beige_wall_001
python outputs/research/complete_square_centered_replay/run_centered_v2.py --candidate beige_wall_002
```

脚本基于当前源码执行；历史严格复现应参考各实例的源码快照与原始命令。本次不评估视觉模型，已有同进程连续复位问题仍通过独立初始化避开。下一项最小实验是针对居中后原位回放的释放/退离阶段接触和物体偏移进行固定条件诊断；本次未扩大到该控制器修复。
