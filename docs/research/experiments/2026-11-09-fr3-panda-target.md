# FR3 / Panda 同目标到达验证

用户授权：配置FR3，以简单target任务验证并交付与Panda的对照视频。假设：复用原生ReachAboveTarget执行器、切换本体关节和TCP后，两种本体均能在相同目标到达并连续驻留20步。

固定registered_scene_65、ProjectOntoLine seed0、布局、基座位置、目标偏移10cm、120Hz物理时钟、相机和DLS控制算法。各自沿用已有高PD配置与初始姿态；不作为本体性能基准。FR3使用本机4.5资产和资产TCP（103.4mm），Panda使用107mm TCP。任务是到达诊断，无模型调用或抓放。

本体适配：`geomind/episodes/robot_profiles.py`；scene.robot支持franka/fr3；动态绑定关节、末端和轨迹列名。FR3的PickPlaceMidpoint明确拒绝，避免误用Panda夹爪绑定。18项相关契约/录像测试通过。

证据目录：`outputs/research/fr3_panda_target_v1/`。每个本体保留独立configs、request.json、source_snapshot、provenance.json、PhysX报告、实测轨迹与原始录像。最终状态以result.json与对应physics/report.json为准。

Panda首次远程资产运行在开始动作前停滞，主动结束并保留于`panda_remote_stall`，属于系统/资产加载失败，不计作动作失败。修复：Panda优先本机5.1同名资产，保留远程回退；在`panda`独立目录重试。FR3结果不受该Panda分支修改影响。

运行期间发现其他会话新增PickLiftChangedObject分支。保留其修改；两侧各自源码快照保留差异，ReachAboveTarget执行分支未变，不能将运行后current_source变化视为历史快照一致。资产路径/版本不同、初始姿态及高PD参数不同，均限制跨本体性能归因。

最终已执行：FR3/Panda各正例1/1通过，错误目标对照各1/1正确拒绝，轨迹复核共4/4通过。FR3误差3.890mm、动作1149步（9.575s）；Panda误差6.226mm、动作992步（8.267s）；均连续驻留20步。录像8条与合成视频完成逐帧解码及SHA256，查看两侧末帧与合成预览。Panda复核使用保存源码快照重新导入episodes，其implementation hash与实测证据一致（c6a958d21e3d52c1785e737c846c772c9cbae6e0306fdae52097ba9f1db4d6c6）；当前源码已被其他会话修改，直接当前版本审计的Mixed configuration被保留解释，不篡改原证据。

[观看网页](http://127.0.0.1:8768/)；[视频与校验结果](../../../outputs/research/fr3_panda_target_v1/result.json)。左FR3右Panda，原速，同起点，较短片段末帧冻结。下一项最小验证：FR3另选一个邻近目标及其错误对照（未执行）；本次仅覆盖固定单目标到达。
