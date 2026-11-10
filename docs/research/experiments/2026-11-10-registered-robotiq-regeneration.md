# 20个注册任务重新生成：Panda＋Robotiq

2026-11-10。按用户要求复用已有20个注册实例，完成“场景×原资产×原结构随机化→初始物理落稳→实测GT→真实抓放→独立复核”。本轮最终20/20正例通过，20/20运行时GT复算通过，20例保存源码/配置完整、原运行结束时provenance通过。13例抓起、7例放置；另有20次真实错误目标/错误落点执行。没有训练或学习模型评测。

[统一网页](http://127.0.0.1:8780/registered-robotiq/delivery.html) · [完整验收JSON](../../../outputs/research/registered_robotiq_v2/delivery.json) · [实际末帧总览](../../../outputs/research/registered_robotiq_v2/review/selected_contact_sheet_20.png) · [最终独立审计目录](../../../outputs/research/registered_robotiq_v2/delivery_audits/20261109_204513_7515b98e/)

## 注册范围与实际选中版本

来自原注册表19个Base及无Base父项的SelectByDistance Semantic。保留原task ID、资产描述、GT规则和原注册来源；不包括退役PlaceDirectional、占位TrackObjectOriginClutter或另算Aug。13项静态、7项时序；SelectByOrderSecond原契约是静态/noop。

所有实例seed0。默认原布局缩放1；Adj显式扩大XY间距2.5倍，VisualCue为2倍，均在同一次原随机化之后执行，不重新采样、不缩放物体、Z或材质。记录缩放中心、最小整体平移及逐物体前后位置，超出安全区拒绝。Grid保留原量化规则，做有记录的行槽约束修正，不改变GT公式。

| 任务及原始结果 | 场景 | XY布局倍数 | 最终选择目录后缀 | 物理结果 |
|---|---:|---:|---|---|
| [CompleteSquare_v0](../../../outputs/research/registered_robotiq_v2/CompleteSquare_v0/result.json) | 65 | 1 | v2 | 通过 |
| [ContinueSpatialPattern_v0](../../../outputs/research/registered_robotiq_v2/ContinueSpatialPattern_v0/result.json) | 65 | 1 | v2 | 通过 |
| [ExtendEqualSpacing_v0](../../../outputs/research/registered_robotiq_spacing_control240_v1/ExtendEqualSpacing_v0/result.json) | 65 | 1 | spacing_control240_v1 | 通过 |
| [InferMidpoint_v0](../../../outputs/research/registered_robotiq_repairs_v3/InferMidpoint_v0/result.json) | 65 | 1 | repairs_v3 | 通过 |
| [InferPointSymmetry_v0](../../../outputs/research/registered_robotiq_v2/InferPointSymmetry_v0/result.json) | 65 | 1 | v2 | 通过 |
| [PlaceByTopologyInside_v0](../../../outputs/research/registered_robotiq_repairs_v3/PlaceByTopologyInside_v0/result.json) | 65 | 1 | repairs_v3 | 通过 |
| [PlaceByTopologyOnTop_v0](../../../outputs/research/registered_robotiq_contact240_v2/PlaceByTopologyOnTop_v0/result.json) | 65 | 1 | contact240_v2 | 通过 |
| [SelectAfterExclusion_v0](../../../outputs/research/registered_robotiq_v2/SelectAfterExclusion_v0/result.json) | 65 | 1 | v2 | 通过 |
| [SelectByAdjacency_v0](../../../outputs/research/registered_robotiq_layout240_v1/SelectByAdjacency_v0/result.json) | 65 | 2.5 | layout240_v1 | 通过 |
| [SelectByDirection_v0](../../../outputs/research/registered_robotiq_contact240_v1/SelectByDirection_v0/result.json) | 65 | 1 | contact240_v1 | 通过 |
| [SelectByDistance_v0](../../../outputs/research/registered_robotiq_horizontal_tgs240_v1/SelectByDistance_v0/result.json) | 91 | 1 | horizontal_tgs240_v1 | 通过 |
| [SelectByGridIndex_v0](../../../outputs/research/registered_robotiq_repairs_v3/SelectByGridIndex_v0/result.json) | 65 | 1 | repairs_v3 | 通过 |
| [SelectByOrderFirst_v0](../../../outputs/research/registered_robotiq_canonical240_v1/SelectByOrderFirst_v0/result.json) | 65 | 1 | canonical240_v1 | 通过 |
| [SelectByOrderLast_v0](../../../outputs/research/registered_robotiq_repairs_v4/SelectByOrderLast_v0/result.json) | 65 | 1 | repairs_v4 | 通过 |
| [SelectByOrderSecond_v0](../../../outputs/research/registered_robotiq_repairs_v3/SelectByOrderSecond_v0/result.json) | 65 | 1 | repairs_v3 | 通过 |
| [SelectWithVisualCue_v0](../../../outputs/research/registered_robotiq_visual_pgs240_v1/SelectWithVisualCue_v0/result.json) | 91 | 2 | visual_pgs240_v1 | 通过 |
| [SelectWithinSubset_v0](../../../outputs/research/registered_robotiq_repairs_v4/SelectWithinSubset_v0/result.json) | 65 | 1 | repairs_v4 | 通过 |
| [TrackMovingObject_v0](../../../outputs/research/registered_robotiq_repairs_v4/TrackMovingObject_v0/result.json) | 65 | 1 | repairs_v4 | 通过 |
| [TrackObjectSwap_v0](../../../outputs/research/registered_robotiq_contact240_v2/TrackObjectSwap_v0/result.json) | 65 | 1 | contact240_v2 | 通过 |
| [UpdateObjectOrder_v0](../../../outputs/research/registered_robotiq_repairs_v3/UpdateObjectOrder_v0/result.json) | 65 | 1 | repairs_v3 | 通过 |

场景65基座为[-0.12,0.32,0.9328504589976385]m；水平桌面场景91基座为[-0.03,0.40,0.7062299448967717]m。球体与圆柱任务选择91，以避开65桌面的微倾斜；不是只改scene_id，运行时实际加载相应USD并测量支撑。家具、背景、材质路径与原房间层保留。每体独立点云和哈希绑定原prim，未把异构资产统一替成蓝罐。

最终实例分版本执行：12例120Hz，8例240Hz。早期成功只证明其保存的配置和源码；不能声称全部20例在最后一个控制器版本上同一次重跑。每个worker最多420秒、GPU串行；失败、超时和中间版本全部保留，没有换seed补齐。最新显式retry版本优先，新的失败也不回退挑旧成功。

## GT、操作与负控

初始组装均运行600个物理步并独立复算。原随机化器只调用一次；机器人绑定既有刚体，不重建替身桌面或物体。初始CPU绑定GT与机器人运行时查询GT分别保存，后者从实际位姿/几何重新求解并固定；初始13静态GT和7时序pending不混计为20个初始答案。

时序任务通过原motion函数的路径执行真实PD场景事件，保留实际Observe.End历史。场景事件结束后清除事件力并真实落稳，再求查询GT；计划终点不当观测。事件原2秒时长保持，240Hz时每协议步40 ticks，120Hz为20；视频、历史和查询时钟独立检查。公开方向沿用原任务约定并显式说明，不统一偷换为世界+X。

OnTop/Inside为Relation GT，保留原支撑/容纳区域、target/reference与完整契约。代表点用于执行策略，不是唯一合法答案；语义检查不以代表点10mm限制替代原关系。松爪后必须通过实际区域、速度、参考保持和连续驻留。其他5项Point放置要求实际抓取、运输、释放、退离、支持与目标误差。

抓起要求正确物体实际抬升超过60mm、保持在夹爪附近、闭爪、线速度≤0.01m/s、角速度≤0.1rad/s、参考位移≤5mm、参考姿态误差≤0.12rad，至少连续0.5秒；原始关节限制与时间一起复核。只写关节目标，动作期间无物体位姿写入、吸附、焊接或搬运外力。路径没有接触自由证明。

20个错误对照均真实抓起物体并保持冻结GT，评分正确拒绝。7个放置负控还必须确实完成错误命令、退离并在合法桌面支持点稳定，不能用动作失败当语义反证。旧ExtendEqualSpacing负控曾在错误点执行失败，新审计据原始轨迹明确拒绝该旧负控，而旧正例仍通过；原目录201个文件哈希不变。最终新版本用实测初始空位作为错误命令，重新执行并通过完整负控要求。该回归证据另存review/spacing_negative_regression.json。

正负例分别reset、真实落稳、重新经历事件和查询，微状态与物理时钟可能不同。这是控制/任务合同验证，不是严格相同初态上的因果配对，不把差值解释为模型性能。

## Robotiq接触修复与边界

实际使用本地Franka的Robotiq_2F_85变体：命令finger_joint弧度，通过mimic/闭链带动其余关节，读取两outer knuckle及140mm TCP。保留标准驱动刚度和资产材料；没有通过放宽速度或保持阈值取得通过。

旋转方形primitive cuboid用原局部面宽计算闭合命令，避免方块转正后世界AABB变窄导致过开；评分宽度仍是实测世界AABB。0/90度朝向由实测参考间隙筛选，标准朝向有10mm余量时优先使用；筛选不是整条机器人无碰撞证书。抬起后的保持模型只估计质量、关节弹簧和接触力矩，假设的摩擦系数不是实测材料或力传感数据，最终仍以实际状态验收。

球体PGS版本确有自旋（raw角速度与四元数差分一致），TGS单变量对照解决了本次持稳；未增加扭转摩擦或改物体材料。盒/闭链接触采用PGS；求解器与实际USD属性均保存在证据中。

VisualCue的解析圆柱在GPU后端多次未落稳。Carb设置读回不足以证明表示变化，因此最终在原刚体内部添加明确的32边、64顶点凸网格碰撞体，禁用原Cylinder碰撞，保留原Cylinder视觉、尺寸、物理材质及源USD文件。网格取自hash-bound原256点圆柱云，最大径向近似误差≤0.1252mm，不添加新刚体或支持面。统计记录新增3个convex形状；原disabled圆柱仍可能出现在形状总计里，不能把总计当激活状态证明。[NVIDIA圆柱/网格碰撞说明](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.1/dev_guide/rigid_bodies_articulations/collision.html)是所用近似接口依据；GPU解析圆柱失败的底层原因未完全定位。

90mm高圆柱原中部抓位侵入上部机构，虽然暂时带起物体，两指垫仍有空隙。最终上移TCP约30.203mm，使顶部距近端指垫边缘4mm、侧壁夹持重叠约34mm，按实测状态完成抓取。对应正例实际抬升117.640mm、连续持稳240/120 ticks，末态角速度约0.02786rad/s。上部机构AABB检查是几何支持，不是接触传感证书。

## 复核与交付

190项CPU契约、几何、请求、夹爪、历史/时钟和审计测试通过；新增碰撞适配5项真实USD测试包含闭合外向网格、缩放、原材质/偏移、来源哈希与无效输入零修改。CPU测试不代替GPU任务验证。最新测试日志和源码hash见[validation_evidence.json](../../../outputs/research/registered_robotiq_v2/review/validation_evidence.json)。

独立复核将原episode包完整恢复到隔离目录，再用最新auditor重算实际历史GT、原始物体/控制和评分。缺失的非episode依赖来自显式哈希记录的当前fallback，不伪称它们是旧执行时快照。跨Python浮点GT比较仅容忍1e-12m舍入差；类型、frame、ID和完整Relation契约保持严格。原报告不改写。保存源码完整、原结束时provenance通过与当前工作区源码变化分别报告；当前变化不等于保存证据损坏。

媒体核对覆盖20个正例和20个负控的observer/agent共80视频、40组帧时钟，全部文件/哈希、帧数、实际dt/fps与边界无缺口。新增/修复的格只使用本例真实末帧；旧展示版本与各次失败保留。网页GT区为研究者特权视图，精确状态oracle不是视觉或VLA模型评测。

下一项最小实验尚未执行：在固定场景、资产、seed和任务契约下比较标准Robotiq朝向与90度朝向，配对记录参考物保持、实际持稳、关节/时间与失败类型；若换模型，先固定公开输入条件与模型版本，不能把本轮oracle闭环当模型成绩。
