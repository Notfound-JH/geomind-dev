# Piper 接入 GeoMind target 验证

用户授权在GeoMind测试已下载的Piper，沿用前述target到达语义。假设：官方模型转换并补六轴绑定后，可在注册场景65完成固定布局seed0/1的ReachAboveTarget。验收：冻结查询GT、<=12mm位置误差、<=0.15rad姿态误差、连续20步驻留，实测关节限位合法，错误目标对照拒绝。无模型评估或抓放。

## 本次补齐

松灵官方agx_arm_urdf commit 680fa925cd00db7e2628563d2edc5540963045de；依赖齐全的Piper+AGX夹爪URDF转为assets/robots/piper_prepared/usd/piper.usd，保留固定关节、mimic与自碰撞。转换命令及配置见outputs/research/piper_target_v1/convert.py与conversion.log，原始URDF/网格来源和哈希见source_manifest，转换产物哈希见result.json。

robot_profiles新增Piper六轴配置，gripper_base末端，局部TCP z=0.138m（URDF指尖安装平面）。通过URDF前向运动学与有界数值求解选取初始关节角：约[0,1.863925,-1.286246,0,1.080384,1.570796]rad。CPU尝试显示直接照搬Panda初态/手腕朝向存在可达性问题；最终使用朝下水平转向90度、基座x=.14m、桌面上方.40m实体支架（有碰撞）。场景与目标不变，不宣称同安装条件跨本体对照。手臂采用关闭重力的高PD关节位置控制（400/40），不是标定真机控制模型。

原生执行和审计移除七轴硬编码：轨迹TCP列从columns定位，六轴关节变化范围按实际数量计算，姿态审计使用保存的tool_quaternion_wxyz，对旧报告默认原Panda姿态。原始runner的robot_control文本仍写Franka high-PD，属继承描述；实际资产/robot_name/关节名称/姿态字段才是本体证据。

## 已执行结果

先单例attempt1正例物理通过，按函数接口离线审计通过。最初误用CLI --audit到非标准目录导致找不到physics/report.json（工具调用路径问题、非物理失败），改用audit_physics(configs,attempt1)通过；原日志和输出保留。

正式420秒有界串行GPU worker：两正例2/2通过，误差2.538/2.452mm；各20步驻留。错误目标对照1/1正确拒绝（目标误差77.46mm）。实测轨迹离线复核3/3通过（GT重算、状态/配置一致性、动作及实测限位、姿态/位置/驻留）；21项相关测试通过，新增Piper配置接受与不支持抓放拒绝测试。七轴兼容测试保持通过。六轴累计动作转动67.38/82.09度，峰值速度0.397/0.400rad/s。没有观察到前次Robotiq式大幅绕行，但未测接触力，不认证无碰撞。

录像6/6逐帧解码与帧数、SHA256检查通过，查看首例/第二例末帧，确认Piper、夹爪、支架和场景65。物理worker完成后由原runner有界清理。正式request、configs、provenance、源码快照、physics报告/原始轨迹/录像/复核及资产SHA256保存在outputs/research/piper_target_v1。

[观看网页](http://127.0.0.1:8773/)；[物理报告](../../../outputs/research/piper_target_v1/physics/report.json)；[轨迹审计](../../../outputs/research/piper_target_v1/physics/offline_audit.json)。已完成target到达，不包含开合抓取、搬放、双臂或模型能力评估。下一最小验证：原装夹爪开合后单方块接触抓取（未执行）。
