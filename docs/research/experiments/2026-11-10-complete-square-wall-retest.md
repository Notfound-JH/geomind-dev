# CompleteSquare 墙体纹理复测

2026-11-10，已实际执行。用户修改墙体纹理后要求复测；发现 `outputs/wall_texture_comparison` 保存了 `beige_wall_001`、`beige_wall_002` 两个候选，因此两者均验证。[新视频网页](http://10.81.44.63:8780/complete-square-walls/)并排展示实际抓放及错误落点对照；旧 CompleteSquare 对比页已增加入口。

## 固定条件与实际改动

复用 `outputs/research/registered_robotiq_v2/CompleteSquare_v0/assembly` 的保存布局，scene 65、seed 0、相同四个方块、桌子与 Panda + Robotiq 2F-85 基座，没有重新采样。每个实例复制候选 USD 和三张 2K JPEG 底色、粗糙度、OpenGL 法线贴图，保存源文件与副本 SHA-256，并组合到实际组装场景。

候选层原本将四面 Cube 墙改为带面 UV 的等尺寸 Mesh 墙；复测没有改写候选原件。四个任务物体、桌子、四面墙的世界包围盒逐项对齐，误差不超过 `1e-7 m`；四面墙的实际材质绑定和三张纹理的解析路径均核对。该几何核对不替代随后实际物理验收。

原录像流程会隐藏墙体。新增 manifest 的 `visualization: {"walls": "visible"}` 显式选项，初始场景与机器人运行时共用同一处理函数，仅设置渲染可见性；默认隐藏墙体的既有行为保留。运行时报告确认四面墙可见，抽查新实际渲染首末帧确认新材质显示。

本次使用当前 240 Hz 控制器、12000 动作步上限；每个 PhysX worker 最长 420 秒，GPU 串行。旧对比录像为 120 Hz，因此本次与旧视频不是只改纹理的严格单变量消融。两份本次候选之间使用相同控制与布局。没有执行视觉模型评测。

## 结果与保留失败

| 实例 | 初始落稳 | 正例误差 | 松爪后连续放稳 | 有效错误放置对照 | 独立验收 |
| --- | --- | --- | --- | --- | --- |
| beige_wall_001_isolated_v2 | 600 / 600 步 | 2.446968 mm | 240 步 / 1 秒 | 正确拒绝 | 通过 |
| beige_wall_002_isolated_v2 | 600 / 600 步 | 2.446968 mm | 240 步 / 1 秒 | 正确拒绝 | 通过 |

两个正例均真实抓取、抬升约 144.819 mm、搬运、释放、退离并放稳。错误落点对照从独立初始化后的同一布局执行真实抓放：距错误命令 2.450207 mm、距正确 GT 419.039692 mm，连续桌面支持和松爪稳定 1 秒，独立判据正确拒绝。关节命令、实测关节、动作时钟、冻结 GT、物体状态、参考物保持及禁止执行期瞬移/吸附/外力搬运等由现有严格审计器复算。

首次 `beige_wall_001/CompleteSquare_v0` 保留为失败：正例完成，但同一仿真进程的第二次 reset 后移动方块偏移 168.016947 mm，负控在抓取前出现关节越限，独立审计拒绝。不能把这类动作失败算作错误目标的有效拒绝。

随后只修正验证流程，使用 `--negative-only` 让负控在新 IsaacLab/PhysX 进程中独立初始化。正负原始报告分别保存在 `robot_physics/runs/positive` 与 `runs/negative`，聚合报告保留原始轨迹路径，严格审计规则未放宽。独立初始化下两次查询的四物体位置完全一致；这证明本次复测避开了该异常，**不证明同进程连续复位的根因已修复**。

候选 001 的原始公开 observation 索引链接通过单独步骤建立，记录于 `observation_index_links.json`；候选 002 将相同链接步骤并入执行包装器。链接不改写原始 observation 或仿真轨迹，因此两个来源快照不同，但控制器、任务与评分逻辑一致。两个实例运行结束时 provenance 均通过，交付时又核验全部保存源码与配置的哈希。

## 验证与复现

产物根目录：`outputs/research/complete_square_wall_retest_v1`。最终采用带 `_isolated_v2` 后缀的两个实例，首次失败目录完整保留。网页包含原始审计、材质几何核对、配置/源码清单和实际执行命令。

42 项 CPU 检查通过，覆盖可见性不影响材质/几何/碰撞、组装契约与审计、worker 超时、缺失负控拒绝及独立进程证据保存；不以该数量代表全部任务覆盖。4 段新正负 observer MP4 完整解码通过；28 个页面资源与入口 HTTP 检查通过。实际 headless Chrome 检查双视频元数据、播放/暂停按钮及 390 px iframe 无横向溢出，检查桌面与窄屏渲染。检查报告为同目录 `verification.json`，录像哈希与结果为 `summary.json`。

重新执行须使用新目录，例如：

```bash
mkdir outputs/research/complete_square_wall_replay
cp outputs/research/complete_square_wall_retest_v1/run_retest.py outputs/research/complete_square_wall_replay/
python outputs/research/complete_square_wall_replay/run_retest.py --candidate beige_wall_001
python outputs/research/complete_square_wall_replay/run_retest.py --candidate beige_wall_002
```

脚本拒绝覆盖已有实例。该命令使用当前源码；严格复现历史版本应参照每个实例的 `source_snapshot` 和 `provenance.json`。候选原件、默认任务配置及先前 20 个已交付实例均保留。

下一项最小实验是固定候选 001，对同进程第二次 reset 的关节/接触状态与独立初始化进行配对定位，完成具体修复后再验收连续运行；本次未扩大执行此项。
