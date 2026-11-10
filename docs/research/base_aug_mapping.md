# Base/Aug 与当前工程字段

2026-11-07。任务定义入口实际为 [docs/task_descriptions.md](../task_descriptions.md)，`doc/` 不存在。分类已确认；操作性定义与增强方向仍是草案。本文记录工程映射，不重新解释历史结果。

|层|当前源码事实|本次处理|
|---|---|---|
|旧 task ID|`core/ids.py` 把 BindingAugSem 等拆成 split=aug、variant=sem；紧凑 ID 默认 split=base|保留读取；这些不是 train/test|
|兼容 episode condition|`core/episode_compat.py` 只接受 Base/Semantic/Visual/Execution|不塞入新 Aug，不改变 release1记录|
|旧 Aug suite|`presets/aug_suite.json` 当前仅 SelectByDistance；其 condition=Semantic，场景10|不据名称推断增强分布或动态性|
|旧 variant|用于 OnTop/Inside/Clutter 等任务变体；interface另有 Standard/Vla|不当作通用增强字段|
|原生 release2 episode|schema3，场景/布局/时间协议/响应/seed分别引用；没有 condition/split字段|新请求侧记录 Base/Aug、增强及 split，实际 episode 保持现有 schema|
|新 Base|已有 generate_spatial_suite v1 的明确采样分布；允许程序化桌面与注册65|换seed为重采样，换场景仍可Base；不改输入模态|
|新 Aug|ProjectOntoLine全体角色沿任务x平移0.03m|配同scene/seed的Base，保存paired_base；重新校验/求解GT；物理GT从稳定后实测状态计算|

Base分布完整保存在每个 request.json。几何：yaw∈[-π,π]，pitch∈[-.4,.4]，中心xy∈[-.05,.05]、z=.2，线端点±.1，沿线参数∈[-.16,.16]，垂直偏移.07且nz=0。该限制不能证明一般三维推理；独立非零z回归继续保留。

位移Base：3块固定x轨道（-.15/0/.15），初始与目标y∈[-.2,.2]，一个目标位移≥.12；最多100次拒绝采样并逐项保存失败候选和原因。语义阈值.03、边界带.001；Observe2步→隐藏Delay3步（第1步后施力）→Respond，协议步.5秒。CPU是离散状态更新，PhysX是外力PD。当前更新只测试位移，新增/移除语义仍未决。

Aug保留任务ID、角色、无限三维直线投影、答案类型、相对几何、观察字段和时间协议；不会因为Aug新增记忆需求。名义GT按变换独立构造，生产求解器重新求解；物理稳定后的GT另算，不要求与名义值逐位相同。同seed跨场景是名义任务坐标对应，世界高度、背景与物理状态不同。

场景切换复用 `registered_scene.scene_description/spawn_registered_scene`：解析真实USD桌面高度，映射table原点/机器人基座，运行时核验来源与变换，实际加载场景网格碰撞；目标块使用同尺寸、同角色、刚体原点。有限实际到达可证明该实例可达，不能外推全工作空间。未启用相机/视觉模型，本轮不认证图像可解性。程序化桌面不会仅改名冒充注册场景。

`SelectBetweenObjects` 已有配置保留在 `configs/legacy/relation_variants/SelectBetweenObjects_v0.json`，关系记录在 `configs/relational_variants.json`；canonical registry拒绝该名字，历史结果不删除、不映射至双关系。`SelectWithVisualCue` 保持27项中的独立任务。本轮未改变这项既有兼容决策。
