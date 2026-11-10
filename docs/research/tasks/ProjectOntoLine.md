# ProjectOntoLine 研究卡

更新：2026-11-07。状态：当前配置CPU闭环与3对非零z偏移**确定性诊断已执行**，见[交付补验](../experiments/2026-11-07-delivery-audit.md)；受评模型的配对实验仍未执行。

## 能力主张与定义版本

文档意图：[v3 契约](../../episode_contract_v3_zh.md) 定义 InfiniteLine3D 正交投影，允许线段外答案；[分类](../../canonical_task_families_zh.md) GeometricInference。[GEOMIND_PROTOCOL §8](../../../GEOMIND_PROTOCOL.md) 的早期 XY/placement 目标不能直接替代这个定义。

实现事实：[resolver.solve](../../../geomind/episodes/resolver.py) 计算 q=a+((p−a)·d)/(d·d)d；非线段夹紧。[当前配置](../../../geomind/configs/episodes/ProjectOntoLine_v0.json) 为 schema3/release2；科学定义没有单独审批版本记录。推断：与中点/等距延长不同，可测连续几何构造；独立设立的认知理论和预期难度需研究者确认。

## 实际输入与约束

[SemanticEpisode.observation](../../../geomind/episodes/runtime.py) 公开稳定 ID、标签、精确3D坐标、frame 与指令，无图像/历史；完整实例见[接入资料 §7.3](../../../geomind_handoff.md)。角色 ID line_a/line_b/point 会省去语言绑定。catalog 配置的参考点为 rigid-body origin，未必视觉中心。评估端持有 GT，oracle 直接读取它不等于模型推理。

端点非退化（||b−a||>1e−4）、对象ID合法、有限坐标和工作空间验证支持唯一投影；原生 `evaluate_answer` 按 frame 与欧氏误差容差评分。当前 [generate_spatial_suite.generate](../../../geomind/tools/generate_spatial_suite.py) 端点距离 .2、pitch∈[−.4,.4]，s∈[−.16,.16]，偏移长度 .07。没有强制视觉可解、遮挡控制或额外角色干扰项。

## 潜在替代解法与已有证据

正式生成器垂直方向固定为 [−sin(yaw),cos(yaw),0]，因此 q_z=p_z。非水平线可用 z 插值替代一般三维投影；本轮独立诊断中A误差<1e−12m、B误差约.10–.35m，不是模型基线成绩。物理漂移可能改变等高关系，不外推到 PhysX。

[test_spatial_episode_pipeline.py](../../../tests/test_spatial_episode_pipeline.py) 覆盖线段外、退化、交换端点、非法回答等实现；生成器构造 q 不调用生产投影公式，但仍共享 schema/评分。历史物理 oracle 残差支持当时控制闭环，不是视觉/学习模型证据。新增 [research_projection.py](../../../geomind/tools/research_projection.py) 已执行当前配置加3对诊断，逐例保存observation、GT、JSON解析/frame/容差/错答，共77项接口检查。具体源码快照与报告见交付补验；没有模型成绩。

## 待验证假设、下一实验与完成标准

H1（模型实验未执行）：改变偏移的 z 分量后模型仍能投影。其构造与接口诊断已完成：固定线、真投影 q、偏移长度与对象身份，A用现有 n_z=0，B将单位 n 在 d 的正交平面内旋转使 n_z≠0；p=q+.07n，始终 n·d=0。下一步是固定版本模型的结构化输入比较，不重复生成器建设，不混合视觉与动作。

完成标准：独立检验 q 在线上与残差正交、A/B同答案同偏移模长/合法工作空间，z插值基线在B失效；若接模型，保存不可变模型版本和真实白名单输入，报告配对误差/正确率与pitch分层。A好B差仅支持特定捷径/分布偏好疑问，不能断言模型内部算法；都好只排除这一条捷径。需确认目标究竟是任意3D投影还是该受限生成分布，以及原点的可观察性需求。

## 后续物理验收（2026-11-07）

本页前述CPU诊断不变。本次新增 Base、平移x=0.03m Aug、注册场景65，每组两个seed的真实PhysX ReachAboveTarget闭环及错误目标对照；新请求入口和逐例证据见 [流水线实验](../experiments/2026-11-07-pipeline-validation.md)。GT取稳定后的实测对象原点；动作目标冻结于查询时刻。nz=0生成限制、视觉原点可见性及模型表现仍未解决。

## 2026-11-09 本次 Base 执行

[本次实验](../experiments/2026-11-09-next-base.md)已执行：semantic_table，seeds 0/1，GT检查2/2、真实PhysX正例2/2、错误目标对照1/1及三条轨迹离线复核均通过。仅 oracle ReachAboveTarget；未执行受评模型或视觉评估。

2026-11-09：[原两例录像重执行](../experiments/2026-11-09-next-base-video.md)已完成；真实PhysX正例2/2、错误对照1/1、轨迹复核3/3、视频6/6检查通过。

2026-11-09：[场景与指令修正](../experiments/2026-11-09-scene65-instruction-fix.md)：使用注册场景65真实重跑并录像，正例2/2成功；中文展示补充最近点解释，GT及英文契约保持原定义。

2026-11-09：[本次重复请求重新生成与执行](../experiments/2026-11-09-next-base-repeat.md)已完成：保留next_base历史，在其下独立目录新生成seeds0/1；真实PhysX正例2/2通过、错误对照1/1正确拒绝、轨迹3/3复核通过。仅oracle ReachAboveTarget；下一模型配对实验未执行。
