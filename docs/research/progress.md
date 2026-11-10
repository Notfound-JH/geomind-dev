# GEOMIND 研究进度

范围：第一版仅结构化输入、任务语义和反捷径验证；不代替公共任务目录。

| 任务 | 当前证据 | 下一步 | 未解决问题 |
|---|---|---|---|
| [DetectSceneChange](tasks/DetectSceneChange.md) | CPU历史反事实配对、独立答案和正负评分校验，见[实验记录](experiments/2026-11-07-dsc-counterfactual.md) | 固定模型版本后比较 full-history/final-only | 早期新增/移除定义迁移未查到决策；无真实模型 |
| [SelectAfterExclusion](tasks/SelectAfterExclusion.md) | 本轮独立删条件计算两布局并reset mock，确认条件冗余和语义不一致 | 按最小方案补必要性布局与结构化决策适配器（未执行） | mock修复方案未实施；真实任务/动作未运行 |
| [ProjectOntoLine](tasks/ProjectOntoLine.md) | 当前配置+3对非零z诊断CPU闭环已执行，77项答案检查；z插值在B失效 | 固定模型版本后的配对比较（未执行） | 模型表现、视觉原点可见性未确认 |

本轮资料来源为 [geomind_handoff.md](../../geomind_handoff.md) 中工程接入与第7节研究证据两部分；未找到第二个独立接入文件。对本轮相关生产函数重新核对，历史结果保留原时间/版本限制。

稳定工作流：[geomind-research skill](../../.agents/skills/geomind-research/SKILL.md)。执行协议：[CPU配对](history_counterfactual_protocol.md)。后续模型接入：[输入与缓存契约](model_input_contract.md)。设计意图未决见各卡和[定义差异记录](decisions/detect_scene_change_semantics.md)。

状态迁移必须链接具体配置、commit/dirty快照和报告；禁止将建议转写为已执行或将 oracle 分数转写为模型表现。研究问题缺失不阻断独立的本地实现校验。

新增[任务执行矩阵与设计映射](execution_mapping.md)、[完整交付核验](experiments/2026-11-07-delivery-audit.md)、[skill实际使用记录](experiments/2026-11-07-select-exclusion-skill-use.md)。更新后查询/连续运动链本轮仅静态确认；测试数量不是全部类型覆盖证据。

## 2026-11-07 完整流水线扩展

上述CPU研究记录保持其原版本边界。新增已授权的真实物理验证见 [流水线实验](experiments/2026-11-07-pipeline-validation.md)：统一请求、Base/Aug配对、真实场景切换、语义/事件/机器人轨迹分层导出及离线复核。入口 `geomind.tools.research_pipeline`；[27项矩阵](task_capabilities.md) / [详细JSON](task_capabilities.json)；[Base/Aug映射](base_aug_mapping.md)。本次没有训练或模型评估，也没有补齐全部任务。

下一入口：先阅读本次报告的失败/未验证表，再选定连续身份追踪的原生观察与指标契约，或接入版本固定的结构化模型。不要重复搭建投影/位移环境，不将成功ReachAboveTarget扩展成抓放能力。

## 2026-11-09 请求执行

[GeometricInference Base 本次实验](experiments/2026-11-09-next-base.md)已执行，输出 `outputs/research/next_base`：ProjectOntoLine代表，semantic_table，seed0/1，GT与真实PhysX正例2/2通过、错误目标对照1/1正确拒绝、实测轨迹复核3/3通过。来源快照与配置ID见报告；无模型评估。下一步为固定模型版本的非零z配对比较（未执行）。

2026-11-09：[ProjectOntoLine录像重执行](experiments/2026-11-09-next-base-video.md)已完成：原生成配置2例真实PhysX成功，错误对照拒绝，轨迹3/3、视频6/6检查通过；无模型评估。

2026-11-09：场景与指令修正已执行，见[实验记录](experiments/2026-11-09-scene65-instruction-fix.md)。真实注册场景65正例2/2、错误对照1/1、轨迹复核3/3；网页已更新为直观中文说明。

2026-11-09：[用户重复生成请求](experiments/2026-11-09-next-base-repeat.md)已重新执行，独立输出 `outputs/research/next_base/run_20261109_repeat`；semantic_table、seeds0/1、420秒上限；GT及真实PhysX正例2/2通过、错误目标对照1/1拒绝、轨迹复核3/3通过。历史结果保留，未接模型或执行抓放。网页与来源ID见实验记录，下一固定模型非零z配对实验未执行。

2026-11-09：[操作路由回归](experiments/2026-11-09-operation-routing-regression.md)已修正静默诊断降级，9项测试与CLI拒绝检查通过。抓放生成与执行仍未完成；旧物理结果仅为诊断证据。

2026-11-09：[中点放置设计](experiments/2026-11-09-midpoint-placement-design.md)已生成2例自然操作任务及GT，CPU交叉核验2/2；实际抓放未执行，网页明确标注。

2026-11-09：[中点放置物理验证](experiments/2026-11-09-midpoint-placement-physics.md)已执行：真实抓放正例2/2，错误目标1/1正确拒绝，物体状态/关节/释放后稳定离线复核3/3，录像6/6校验通过。仅固定两例，不代表通用生成入口已支持。

2026-11-09：[FR3/Panda target对照](experiments/2026-11-09-fr3-panda-target.md)已完成：本体选择/FR3 TCP适配，真实场景65同目标正例各1/1，错误对照各1/1拒绝，轨迹复核4/4，原始录像8条及原速并排视频已校验。仅ReachAboveTarget，无抓放/模型性能结论。网页 http://127.0.0.1:8768/ 。

2026-11-09：[SceneUpdating Base抓取验证](experiments/2026-11-09-scene-updating-grasp.md)已执行，semantic_table seeds0/1：GT2/2、真实抓起正确目标2/2、错误目标拒绝2/2、实测轨迹复核4/4通过，前两轮持握稳定失败保留。无模型评估。

2026-11-09：[FR3初始姿态对齐](experiments/2026-11-09-fr3-matched-home.md)已完成：7关节角映射Panda，FR3正例/错误对照与轨迹复核通过，更新对照视频（Panda复用v1）。

2026-11-09：[Panda+Robotiq target](experiments/2026-11-09-robotiq-target.md)已完成：换装2F-85，真实到达正例/错误对照及轨迹复核通过，视频 http://127.0.0.1:8770/ 。未验证抓取。

2026-11-09：[Robotiq七轴轨迹比较](experiments/2026-11-09-robotiq-joint-comparison.md)已完成离线分析：关节累计转动约3.35倍，差异集中首段抬升，IK指令存在持续绕行；因果消融未执行。

2026-11-09：[Robotiq绕行debug](experiments/2026-11-09-robotiq-collision-debug.md)已完成：雅可比差分、TCP/自碰撞/精确接触对消融定位底座与内连杆碰撞，3对窄过滤修复；正负例/轨迹复核通过，累计转动642→205度。网页 http://127.0.0.1:8772/ ，开合抓取未验证。

2026-11-09：[Piper target接入](experiments/2026-11-09-piper-target.md)已完成：官方URDF转USD、六轴绑定/动态轨迹审计，场景65正例2/2、错误对照1/1正确拒绝、轨迹3/3、视频6/6。采用40cm支架和专用初态；无抓放。网页 http://127.0.0.1:8773/ 。

2026-11-09：[双臂Piper桌面配置](experiments/2026-11-09-piper-dual-table.md)已完成：去支架，双Articulation与显式active_arm，左/右独立target2/2、负对照拒绝、主轨迹3/3和双臂待机/限位复核3/3通过；早期限位和待机干扰失败保留。网页 http://127.0.0.1:8774/ 。同步协同抓放未实现。

2026-11-09：[双臂Piper自然姿态](experiments/2026-11-09-piper-natural-pose.md)已完成：向下20度、内偏25度与镜像home，去水平平行感；双臂正例/负对照、主轨迹与双臂待机复核通过。网页 http://127.0.0.1:8775/ 。

2026-11-09：[双臂Piper后移10cm](experiments/2026-11-09-piper-rear-placement.md)已执行：仅改两侧基座x，左右正例与负对照、末端/双臂复核通过，视频 http://127.0.0.1:8776/ 。

2026-11-09：[双臂Piper桌边配置](experiments/2026-11-09-piper-table-edge.md)已完成：测真实边界后选x=-.11m（底座后缘余约7cm），保留两次可达性/轨迹失败，调整接近高度后左右正例/负对照、主/双臂复核通过。视频 http://127.0.0.1:8777/ 。

2026-11-09：[场景图鉴材质重渲染](experiments/2026-11-09-scene-atlas-materials.md)完成102/102静态RTX预览，768×768，正式材质日志错误0、未解析引用0，旧图保留并支持对比。网页 http://127.0.0.1:8780/ 。不涉及物理任务验收；配套旧JSON与部分新版USD不一致已标注，元数据核对未执行。

2026-11-09：[102场景空桌面另存](experiments/2026-11-09-scene-atlas-cleared.md)完成102/102，移除638件物体与容器/桌面小件；源文件保留、保存后重开与阶段元数据复核均通过。空桌面USDA目录assets/scenes/tabletop_cleared_v1；依赖原资产。102张新RTX预览已更新到 http://127.0.0.1:8780/ ，可切原始版。无物理任务执行。

2026-11-09：102个空桌面场景的[名称索引](../../assets/scenes/tabletop_cleared_v1/scene_index.json)已建立：按桌子资产、地板、保留家具种类与数量归为35组，同配置共名并保留原编号；13组基础名冲突时追加家具消歧。Excel/CSV/JSON位于outputs/scene_index_v1，网页支持名称检索。未改USD文件名或内容，不代表35组几何字节完全去重。

2026-11-09：[可交互资产与摆放支持清单](../interactive_assets_and_layouts_zh.md)已整理：25类登记候选、4类原场景便携摆件、6类未登记文件及4类程序化形状，35条文件候选均找到本地模型，5条原配置旧路径另列。Excel/CSV/JSON在outputs/interactive_assets_layouts_v1，区分旧摆放器、静态方块、混合脚本与当前结构保持随机化；CPU随机化测试3项通过，无新GPU或逐资产物理执行。

2026-11-09：[场景×物体×结构组装](experiments/2026-11-09-scene-assembly.md)已实现受限中英文/JSON→空桌面USD+实际物体+布局，grid/line/ring/random/t_shape。3类实例真实落稳通过、3个离线错误布局拒绝、23项CPU测试通过，保留两次实现失败及修复。入口python -m geomind.tools.assemble_scene；网页 http://127.0.0.1:8780/assembly.html 。范围为初始场景，未执行机器人动作。

2026-11-09：用户要求保留[布局随机化检查结论](../scene_assembly_zh.md#布局随机化核对与后续接入2026-11-09)。当前新组装入口未接旧摆放器或运行时结构保持随机化；结构化布局的seed尚不生效，物体朝向固定。接入随机化及其重新验收登记为待办，未实施；现有代码与全部生成证据保留。

2026-11-09：随后按用户要求完成[随机化接入与实际展示](experiments/2026-11-09-scene-assembly-randomization.md)。grid复用旧运行时结构保持规划器，加入独立整体偏移及受限物体yaw；60号4蓝罐2×2、seeds7/19最终实例均真实PhysX连续600步稳定、离线复算及错误偏移拒绝通过，36项组装CPU测试、3项旧随机化测试通过。原规则版与工程验证证据保留，最终两例近景对照和录像位于 http://127.0.0.1:8780/assembly.html 。只验证初始摆放，未执行机器人操作；其他布局本轮仅接整体平移与yaw，未扩大为全资产/全场景物理认证。

2026-11-09：[组装场景与任务GT结合](experiments/2026-11-09-assembled-task-generation.md)已接入原research_pipeline的显式assemble_and_solve_gt阶段。agent可查询102场景/5布局/28资产/4静态原生任务并提交可审计选择；最终60号网格中点、65号X直线顺序选择、同60号实测状态投影三份GT独立核验与离线复核通过，两个独立初始组装均600步稳定。修正编号/列表顺序编码，GT与公开输入分离；62项CPU检查通过。产物outputs/research/assembled_tasks_v2，网页 http://127.0.0.1:8780/tasks.html 。机器人任务动作未执行、物理任务通过数0，无模型评估；不能把初始落稳当抓放完成。

2026-11-10：[本体与实际动作接入](experiments/2026-11-10-assembled-robot-actions.md)完成skill/请求/原生AssembledScene绑定及严格动作执行入口。五种本体列能力边界，组装接触动作先限Panda；最终60号真实蓝罐中点搬放通过：抬升13.3厘米、松爪退离后误差3.119毫米、连续120步稳定，8厘米错误落点真实执行后正确拒绝。65号持续抓起分支仍未通过，v1-v4失败与对照全部保留，最终完成机器人任务1例、未通过1例，不把审计一致或初始落稳计作动作成功。97项相关CPU检查、skill验证与独立路由检查通过；最终四个正/负动作结果离线复算一致，源码/关节/状态/视频hash核对通过。产物outputs/research/assembled_robot_v4；网页 http://127.0.0.1:8780/tasks.html 显示实际机器人正负录像与运行时GT，无VLA模型评估。


## 2026-11-10 20项注册任务与Robotiq重生成

[本轮实验](experiments/2026-11-10-registered-robotiq-regeneration.md)已完成：原20注册实例、seed0、保留异构资产与原结构随机化，实际场景65/91，Panda＋Robotiq真实抓放20/20、运行时GT独立复算20/20。13项抓起、7项放置；20个实物错误对照正确拒绝，7个放置负控均通过独立错误命令/桌面支持/稳定核验，不能用动作失败充当拒绝。初始物理20、保存源码/配置20、原运行结束时provenance20。190项相关CPU检查、80视频与40帧时钟核对通过。最终依据每例最后显式修复版本，12例120Hz与8例240Hz；不是全部在最新控制器上的同一次重跑。失败与旧派生报告全部保留。无模型评测。

[统一网页](http://127.0.0.1:8780/registered-robotiq/delivery.html)包含任务指令、场景/seed/布局、实际正负录像、特权GT和独立审计。下一项为固定条件下夹爪朝向的配对验证，未执行；详细边界与来源见实验记录。


## 2026-11-10 CompleteSquare 双视频对比

已完成[原任务与 Robotiq 重生成对比页](http://10.81.44.63:8780/complete-square/)，保留历史桌面 / 全景 / 腕部镜头及新版全景 / 原片裁切，支持双播放、末帧、倍速、阶段跳转和原视频下载。旧录像缺少同次状态与验收，明确标注未按新标准复核；新版沿用原独立审计（2.408225 mm 误差、1 秒放稳），未重执行。详见[记录](experiments/2026-11-10-complete-square-video-comparison.md)。

2026-11-10：[CompleteSquare 新墙体复测](experiments/2026-11-10-complete-square-wall-retest.md)已实际执行，候选 001 / 002 均显示四面新墙并完成 Panda + Robotiq 抓放，正例误差 2.446968 mm、连续放稳 1 秒，两份有效错误放置对照均拒绝。首次同进程复位偏移及关节越限失败保留；显式改用正负独立初始化后验收通过，连续 reset 根因仍待修复。42 项 CPU 检查、4 段视频完整解码、28 项 HTTP 检查与窄屏/播放暂停浏览器检查通过。保存布局未重采样，无视觉模型评测。[新视频页](http://10.81.44.63:8780/complete-square-walls/)已连接旧对比页。

2026-11-10：[CompleteSquare 机械臂居中](experiments/2026-11-10-complete-square-centered-robot.md)已实际执行，两份墙面均将基座沿桌边平移 31.95 cm，对齐世界 Y 中线，正常抓放误差 0.963326 mm、松爪后放稳 1 秒。两份 GT +80 mm X 的有效运输型负控均完成并正确拒绝；首次原位回放对照偏离命令的失败保留。新增显式 offsets_first 候选顺序，默认行为、布局、GT、控制器及物理评分阈值保持一致；43 项相关 CPU 检查通过，实际首段 TCP 平移与基座请求一致。无视觉模型评测。[新视频及位置前后对比](http://10.81.44.63:8780/complete-square-centered/)。

2026-11-10：[102场景墙面随机替换](experiments/2026-11-10-randomized-wall-materials.md)完成：184源USD、本地两款PBR各51场景；修复20份旧格式材质编号偏移，非墙面与原始备份一致。全量绑定/路径/UV/哈希核验通过，2个Isaac RTX样例通过；本批次未执行机器人动作与物理重验证。[总览](../../outputs/wall_randomization_20261110/index.html)。

2026-11-10：[手腕相机场景修正](experiments/2026-11-10-wrist-scene-corrections.md)已实际重执行完成：源墙几何/材质、真实桌面支撑及双臂朝上画面通过，最终五种本体、六个活动臂诊断 6/6；最终 69 项 CPU 检查、88 个 HTTP/媒体资源、6 份保存源码/配置及 14 路实际相机流检查通过。v7 显式降低物体 Z 并从落稳状态重算 GT，四个单臂配置通过；原双臂两例到达失败保留，离线确认原目标存在可行逆解后，v8 仅将双臂基座前移 10 cm 重试通过。合计 8 次实际运行、6 通过/2 失败，最大绝对底高误差 1.200 mm、末窗速度 0.7489 mm/s、双臂滚转 11.89°。[新录像页](http://127.0.0.1:8781/outputs/research/wrist_cameras_delivery_v2/index.html)明确记录 v7＋v8 来源；默认准备器已 CPU 精确对齐 v8，补充源码为明确标记的运行后捕获，浏览器交互未验证，无模型评估或新增抓放能力结论。

2026-11-10：[27类任务补齐与展示](experiments/2026-11-10-canonical27-completion.md)已完成任务证据验收：严格网页27/27类、30/30实例，原20保存版本覆盖17类，本轮10新实例均通过实际正例、有效错误命令负控、运行时GT、独立原始状态与外部/腕部媒体。13个明确重试版本保留，不回退挑成功。修复包括接触高度、Path/Detect/Reflect实测支持的基座、几何反中点条件、统一50mm间接指代候选和固定前侧Robotiq腕相机v2；保持评分与12000步/420秒总界限，保留闭合命令/64次迭代失败未采用。最终Reflect2.011mm、错误命令2.507mm，Indirect正负各稳定1秒且新腕视图可辨。

[展示页](http://127.0.0.1:8781/outputs/research/canonical27_delivery_v1/index.html)、[实现卡](tasks/canonical_extensions.md)、[30实例复现矩阵](../../outputs/research/portable_delivery_v1/validated_reproduction_cases.json)均绑定逐例保存源码与参数。旧20中12例120Hz、8例240Hz，不是全部在最新便携源码上统一重跑；矩阵分开原验收与当前代码建议参数。最终664项HTTP HEAD/MIME、190媒体文件及JS语法检查通过，旧服务失活的第一次HTTP失败记录保留；浏览器交互未验证，无模型评测。
