# Panda + Robotiq 2F-85 target验证

按用户授权将Panda夹爪换为Robotiq 2F-85，复用本机5.1 FrankaPanda/franka.usd的Gripper=Robotiq_2F_85变体。保持手臂初始角、基座、注册场景65、seed0、相机、ReachAboveTarget目标与算法；手臂执行器使用Panda high-PD参数，夹爪沿用现成配置。robot_profiles新增franka_robotiq绑定。诊断TCP取hand z=140mm，对应USD指尖几何z约91..148mm区域；这是明确选取的到达参考点，未经抓取标定，不宣称资产原生TCP或抓取验证。

已执行真实PhysX正例1/1通过：误差10.287mm，小于12mm阈值，连续驻留20步，1679动作步。错误目标对照1/1正确拒绝；实测关节/姿态/误差/驻留离线复核2/2通过。18项相关测试通过。夹爪保持默认张开，仅验证换装及到达，无开合抓取、模型评估或无碰撞认证。

证据：outputs/research/panda_robotiq_target_v1/robotiq内request、configs、provenance、源码快照、physics报告/轨迹/录像。原Panda视频复用fr3_panda_target_v1已验证结果；并排左Robotiq右Panda，原速，短片末帧冻结。result.json保存逐帧解码及SHA256。已查看换爪末帧与对照预览。网页 http://127.0.0.1:8770/ 。下一最小验证为夹爪开合与单方块真实抓取，尚未执行。
