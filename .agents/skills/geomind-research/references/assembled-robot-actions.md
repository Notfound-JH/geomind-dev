# 组装场景、机器人与动作

用于用户要求将场景组装、GT和真实机器人动作接起来时。先读仓库 `docs/robot_embodiments_zh.md` 与 `docs/assembled_tasks_zh.md`；支持范围以当前能力菜单、严格请求校验和逐例物理证据为准。

## 请求入口

使用现有 `python -m geomind.tools.research_pipeline --assembly-options` 查询场景/布局/资产/静态GT及机器人动作能力。agent提交选择理由，场景、布局和本体不隐式替换。

新执行阶段为 `stage=assemble_solve_execute`，沿用 `--assembled-request REQUEST.json`。原 `assemble_and_solve_gt` 仍仅生成GT，不声称机器人动作完成。请求增加：

```json
"robot": {
  "embodiment": "franka",
  "base_position_w_m": [0.0, -0.45, 0.9328504589976385]
},
"execution": {
  "action": "PickLiftSelectedObject",
  "negative_controls": true,
  "timeout_s": 420
}
```

这是65号现有桌面示例的基座值，不可无测量地套到其他桌子。基座单位世界米，Z须匹配实测桌面；双臂须显式active_arm。robot字段支持的本体与执行动作兼容性由请求校验器检查，不能因文档列出资产而忽略不支持的动作组合。

动作选择：SelectByOrder的自然抓起使用PickLiftSelectedObject；InferMidpoint/ProjectOntoLine/ReflectAcrossLine的自然搬放使用PickPlaceObject。ReachAboveTarget是到达诊断，不能在自然操作阶段冒充抓取/放置。首版组装接触动作以Panda为支持边界；新增Robotiq接口使用实际finger_joint和mimic/闭链、弧度控制及双侧实测间隙验收。FR3、Robotiq与Piper的历史到达证据均不自动升级为新场景上的抓放通过。

已有20个注册实例重新生成使用`python -m geomind.tools.regenerate_registered_tasks --output NEW_DIRECTORY`，默认scene65、franka_robotiq、每worker420秒、6000动作步、GPU串行及错误目标对照。先`--tasks InferMidpoint_v0`单例；支持`--robot-base X Y Z`显式基座，不能换seed掩盖失败。该批次边界是原注册表19个Base及无Base父项的SelectByDistance Semantic，不另建任务定义。保留原异构资产、材料和结构随机化；7个动态任务实际执行原路径的场景事件，严格Observe.End历史，撤除事件力后才开始夹爪动作。tracking事件须可见，DirectedChain查询箭头按实测位置刷新。计划位置不可求GT，动态GT在运行时求解。OnTop/Inside按原关系判据及独立状态复核验收。各任务方向约定显式公开，不能把旧Y方向静默解释成世界X方向。逐例以batch.json中的动作状态为准；有源码/GT/录像均不自动计通过。

Robotiq新执行配置使用240Hz细时间步；建议显式`--max-steps 12000`，每worker上限仍为420秒。阶段、事件、复位和释放后落稳按物理秒数保持，连续验收至少0.5秒，原始状态/控制/配置中的dt必须一致。保留标准夹爪刚度；抬升后保持模型只是位置伺服的质量/弹簧估计，不是测力反馈。只对方形primitive cuboid按局部面宽计算闭合命令，旋转后的实测世界AABB仍用于评分；保持已加载刚体，禁止吸附或动作期间改物体位姿。标准朝向有至少10mm保守参考物间隙时优先使用，否则选择更大间隙的0/90度朝向；这是规划筛选，不是整条轨迹无碰撞证明。

选择实际水平支撑和安装基座，不能只改scene_id：scene65已使用`--robot-base -0.12 0.32 0.9328504589976385`；球体在微倾斜桌面上可能滚动，scene91水平桌面使用`--robot-base -0.03 0.40 0.7062299448967717`。可显式`--layout-scale N`在同一次原seed随机化后统一放大XY间距，保存前后位置和最小整体平移，不缩放物体或重采样；拼装容器拒绝此选项。实际支撑、运行时GT、接触动作和参考保持必须全部重新验收，不能把CPU规划预测升级为动作通过。

最终跨版本证据用`python -m geomind.tools.deliver_registered_tasks --batch BATCH --retry OLDER_RETRY --retry NEWER_RETRY`汇总。每任务取最后一个显式列出的版本，包含新失败；不回退挑成功。原执行源码隔离恢复后，用最新独立审计器复算实测GT及动作，区分保存源码完整、原结束时provenance与当前源码变化。派生网页`BATCH/delivery.html`显示真实正负录像、场景/布局、特权GT及证据链接；原报告不覆盖。早期不同时间步/控制版本的成功只证明其保存的实例，不能声称全部在最新控制器上重执行。

## 实物体与GT绑定

AssembledScene须引用原scene.usda、核对manifest与点云hash、绑定已有AssembledObjects prim，保留尺度与原始姿态；禁止生成替代对象或用辅助台面改变任务。reset初始化可以恢复保存的实测初态，执行中不得瞬移、吸附或用外力搬物体。保存reset写入次数与执行期写入次数。

世界坐标、AABB几何中心、刚体root和TCP必须分清。从运行时真实位置与四元数求对象参考点，再重新计算并冻结查询GT；不能沿用机器人加载前另一条轨迹的GT当当前执行证据。语义ID绑定与公开列表沿用独立置换，评估端保存prim映射，公开观察不泄露GT或内部角色。

抓起按正确物体实际抬升、速度、夹爪保持、驻留时长和参考物体保持评分。搬放须真实抓取、抬升、运输、松爪与退离，再检查目标物体中心/底高、实际桌面支持、速度、参考保持和至少所配置的稳定时长；不以TCP到位代替物体末态。

## 运行与证据

每个物理worker最多420秒，GPU串行；先一例再小批量。保存原请求、独立机器人配置、启动前源码快照、运行时查询GT、动作与全部关节状态、原始物体状态、实际机器人录像和逐例报告。负对照用错误物体或偏移目标，评分仍读取冻结正确GT。失败保留，只有明确修复后才新目录重试，不静默换seed补齐。

`--audit-assembled RUN`分别复核GT阶段与组装动作证据；离线复算不称物理重执行。网页将初始落稳录像和机器人动作录像分开，列出正例、错误对照及模型是否评估。oracle_gt执行器用于实现验证，不是VLA模型成绩。

若本体、资产、场景或动作尚不匹配，报告具体缺口并继续已授权的最小修复；不把自然操作降级为诊断。实际成功/失败和当前未验证项放在实验记录与progress，技能保留稳定入口及验收边界。
