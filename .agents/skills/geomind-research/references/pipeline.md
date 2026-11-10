# 原生研究流水线

## 先匹配操作意图，再调用接口

通用 research_pipeline 入口当前仍以 ReportPoint/选择类诊断为主，SceneUpdating 默认 manipulation 已接入 DetectSceneChange + PickLiftChangedObject；GeometricInference 的通用 manipulation 请求仍拒绝；另有原生 InferMidpoint + PickPlaceMidpoint 的有限抓放执行链，见下方“中点放置物理验证”。不能把已有窄范围执行链说成通用生成入口已接通。自然操作生成原则见主 skill；下列命令是已有诊断执行接口，不代表自然搬放任务已经实现。对“把绿色方块放到红蓝正中间”等请求，先选择中点语义并检查放置执行能力，不能因默认代表是 ProjectOntoLine 而调用它替代。

评估端保留几何公式、GT 和坐标；机器人指令应表达用户目标。中文展示改写不能取代模型实际输入契约和执行器修改。若新增自然指令进入模型 observation，应保存实际 payload 并验证其与 GT 的一致性；未接模型时只称展示文案。新设计字段先落实到 schema/执行器，不直接塞入 parse_request 的严格字段集合。

场景未指定且无其他上下文时，使用已接入的 registered_scene_65；显式指定 semantic_table 则保留并标注程序化诊断场景。示例不覆盖用户选择。

## 现有诊断流水线

从仓库根目录执行。CPU 用可导入项目的 Python；物理用已有 IsaacLab 环境（本机 `/home/elm/miniconda3/envs/new_isaaclab/bin/python`）。入口 `geomind.tools.research_pipeline`；未知组合报错，已有输出拒绝覆盖，不自动恢复。

自然语言由 skill 解析成 JSON；脚本接受确定性的 JSON，不声称实现任意自然语言解析器。请求保存为 raw_request.json 和解析后的 request.json。必须给 task_family 和 output；可给 task_id。字段：interaction_mode（GeometricInference 默认 manipulation；旧问答/到达诊断必须显式 diagnostic）、condition、augmentation、scene_id、layout、seed、count、time_protocol、artifacts、backend、timeout_s、coverage、split。默认值和完整分布由 parse_request 保存。

诊断模式支持：GeometricInference→ProjectOntoLine；SceneUpdating→DetectSceneChange（位移版）；显式 ReflectAcrossLine 可运行但三维解释需单独说明。场景仅 semantic_table 与 registered_scene_65；布局 spatial_generator_v1；每次1–4例，seed递增，单次物理超时1–1800秒。完整类别覆盖会报告所有缺失项并拒绝执行，不悄悄缩小范围。

Base 是已有生成器分布，允许两个场景；不是训练集。Aug 目前仅 ProjectOntoLine 的 translate_x_0.03m：全体对象沿任务 x 平移3cm，GT重新求解，保存相同 seed/scene 的 paired_base。保留相对几何、角色、三维直线语义、输入和时间协议；属于文档允许的几何变换工程实现，不宣称更难或研究定义已批准。原生成器 nz=0 的偏移限制保留，非零z诊断由 research_projection 回归覆盖。

以下示例仅供用户明确要求旧诊断时使用，不用于默认操作任务生成。manipulation 请求当前会在生成前拒绝，不能报告成功。

例：
```json
{"task_family":"GeometricInference","interaction_mode":"diagnostic","condition":"Base","scene_id":"registered_scene_65","seed":0,"count":1,"backend":"isaaclab","timeout_s":420,"output":"outputs/research/my_base"}
```
Aug 增加 `"condition":"Aug","augmentation":"translate_x_0.03m"`；换场景使用 `"scene_id":"registered_scene_65"`，仍可为 Base。动态改 `"task_family":"SceneUpdating","task_id":"DetectSceneChange"`，time_protocol 自动展开为 observe_delay_respond_displacement。CPU 请求必须指定 `"backend":"semantic","artifacts":["gt","observations","events"]`；不能请求机器人轨迹。

```bash
python -m geomind.tools.research_pipeline --request REQUEST.json
python -m geomind.tools.research_pipeline --audit outputs/research/my_base
python -m geomind.tools.research_capabilities --output docs/research/task_capabilities.json
```

产物：configs 保存完整引用；semantic 保存 CPU 流/GT/证据，reports/sampling 保存采样拒绝原因；physics/observations 仅模型可见流（含机器人本体状态，区别于纯CPU白名单），physics/evidence 是特权查询及历史；events 分开计划与实测轨迹链接；traces 为每物理步 NPZ 状态和 controls NPZ 动作/全关节位置速度/TCP四元数；cases 为固定目标/waypoints/语义与物理评分。没有真实模型调用。相机条件未验证，不能把结构化任务报告为纯视觉任务。

物理时钟120Hz，协议步0.5s；绝对关节位置控制，机械臂rad、夹爪m，TCP世界坐标m。阶段0稳定、1观察/事件、2抬升、3接近、4下降。reset初始化与执行期区分。事件由外力PD驱动，CPU版本是离散更新。控制器使用关节限制；离线复核实测关节含5mrad/5mm数值容差，未接触传感器，不认证无碰撞规划。

provenance 保存 commit、相关未提交源码内容及哈希、配置集合与命令；物理case保存资产来源。旧结果不可只凭存在跳过：先 check_provenance、配置/定义/条件/执行器比对。当前入口故意不提供自动恢复。源码漂移时保留旧快照和原结论，不能覆写历史来制造通过。

连续身份追踪仍是旧运动/末态判定链，不能由 DetectSceneChange 代替；具体缺口见能力矩阵。旧 condition/split/variant 与旧结果保持兼容，不迁移标签。

## 中点放置物理验证

已实现接口：`geomind/episodes/placement.py:execute_midpoint_placement`，由 `IsaacLabEpisode.execute_answer` 按 `physical_execution.action=PickPlaceMidpoint` 调用；复用同一场景、Franka与PhysX，不另建环境。实例使用 `task_id=InferMidpoint`、`parameters.endpoints` 指定两参考物体，`physical_execution.moving_object` 指定被搬物体，动作配置沿用 execution_profile 的完整字段并覆盖 action、增加 moving_object。物体和场景须满足该执行器条件；不能只把任意配置的 action 改名后声称兼容。

生成新实例后先独立计算GT并解析验证配置，再使用现有runner执行；每次指定独立新输出目录。下面命令中的占位符须替换为本次实际配置根、episode ID和运行目录，不把示例旧配置当新实例：

```bash
/home/elm/miniconda3/envs/new_isaaclab/bin/python -m geomind.tools.run_episode_physics --root CONFIG_ROOT --episode EPISODE_ID --output RUN_DIR/physics --timeout-s 420 --video --negative-controls
python -m geomind.tools.audit_midpoint_placement --run RUN_DIR
```

`RUN_DIR/configs` 必须为 runner 实际使用的完整配置副本（使审计解析同一配置），保留请求、命令和源码内容快照。GPU先单例再批量串行执行。逐例同时检查 `verification_passed`、正例 physical.success、负对照按预期拒绝、placement_audit.json 与录像可读性，不把 audit.passed（复算一致）单独当放置成功。

当前判据：3cm立方体、绿色物体实测抬升>6cm，松爪开指并退离后物体误差<=1cm、名义桌面支撑高度误差<=4mm、线速度<=1cm/s、角速度<=0.1rad/s、参考物体位移<=5mm，连续>=60个120Hz物理步。阈值属于该执行器，换资产/尺寸/支撑面须重新核验，不能外推到全部任务。支撑高度与稳定并非接触传感证明，不认证无碰撞。

`*_objects.npz` 保存每步全部物体根位置、四元数、线/角速度；`*_controls.npz` 保存关节目标/状态；`audit_midpoint_placement` 独立复算冻结中点、时钟、关节限制、抬升及松爪后末态。不得用旧投影 `research_pipeline --audit` 的TCP驻留审计替代放置审计。实际两例与错误目标证据见 [中点放置实验](../../../../docs/research/experiments/2026-11-09-midpoint-placement-physics.md)，结论受当时配置/源码快照限制。

## SceneUpdating 变化识别后抓取

SceneUpdating Base 默认选择 DetectSceneChange 位移版，`interaction_mode=manipulation` 生成 `physical_execution.action=PickLiftChangedObject` 并自动录制视频。无需将操作请求改为 diagnostic。协议隐藏阶段仍是PD外力事件；机器人抓取开始时清除事件外力，按查询时冻结的变化物体评分。自然物理指令为“记住桌上方块的位置，等画面恢复后，把刚才被移动的方块拿起来”。

执行器复用placement.py的夹爪接触抓取；当前3cm方块夹爪目标每指0.014m、物体接触位置/速度求解迭代32。验收正确物体抬升>6cm、速度<1cm/s、其他物体位移<=5mm、夹爪闭合，连续>=0.5秒。不是放置，不以松爪后末态评分。`research_pipeline --audit RUN_DIR` 按action路由到 `audit_changed_lift.audit` 复算物体/历史/GT/关节/持握，保留失败证据和原始源码快照。

场景与输出沿用显式请求，目录存在须另建。前两轮稳定失败、修复后的两例与错误目标证据见 [SceneUpdating抓取实验](../../../../docs/research/experiments/2026-11-09-scene-updating-grasp.md)。两个seed均选中block_1，不代表跨标签或模型记忆已验证。
