# 双臂 Piper 直接桌面安装

用户明确要求双臂，并指出单臂40cm支架不符合预期。本次配置两条完整Piper，各带原装AGX平行夹爪，基座直接固定在registered_scene_65桌面（x=.10m，y=-.22/+.22m，z=桌高）；无高支架。沿用已转换USD，手臂关闭重力的高PD模型，保留自碰撞、臂间/场景碰撞。此配置不冒称经典ALOHA或RoboDojo的ARX X5。

## 实现

原生IsaacLabEpisode同一world创建两条独立Articulation，各自执行位置目标并逐步更新；reset均恢复两臂状态。scene.robot=piper_dual，physical_execution.active_arm显式选择left/right（缺失或未知值拒绝；当前仅ReachAboveTarget）。普通单臂接口保留。每步除活动臂原trace外，保存双臂独立动作/位置/速度、joint_names、限位、物理时钟与active_arm至*_dual_controls.npz。audit_dual_piper独立复核两臂限位、连续时钟和待机保持（阈值.02rad）。这是逐臂到达与待机，不是同步双臂规划/抓放。

保持场景65与原target的GT，采用Piper可达的前伸腕部姿态（wxyz≈[.737277,0,.675590,0]），不强迫Panda朝下姿态。根据官方URDF FK及有界CPU IK选取镜像初态。轨迹为真实关节执行；CPU求解只用于reset初态，不瞬移执行关节或任务物体。正式初态见v3/request.json及robot_profiles，TCP为gripper_base局部z=.138m。

## 保留的问题与修复实验

v1/single：从q5=0初态出发，左臂在腕关节限位处耗尽1800步，误差79mm。记录失败；观察到腕初态奇异风险，改用离开该姿态的镜像home，home_retry通过。

v1/physics：两侧到达均通过，但额外双臂复核发现右臂运动时待机左臂最大偏移.05563rad（约3.19度），未满足.02rad标准；不将其交付为完整通过。未采接触力，仅认为两臂靠近造成干扰的可能性需排除，不将待机偏移单独当作接触证明。

v2：初态末端收回各自一侧，待机六轴漂移0，左右到达均通过。v3将执行臂从早期演示用seed选择改为显式active_arm配置后重新完整执行，结果与v2一致。所有目录/失败录像保留，无换seed掩盖失败。中间版本未独立完整快照；正式v3有配置/源码/命令provenance。

## 最终已执行证据

outputs/research/piper_dual_table_v3，420秒有界、GPU串行。左臂seed0误差.182mm、533动作步；右臂seed1误差.252mm、590动作步；均20步驻留，正例2/2。错误目标对照1/1正确拒绝（80.007mm）。主任务轨迹复核3/3、双臂限位/时钟/待机保持复核3/3通过，待机六轴记录漂移0。24项相关测试通过，包括双臂配置和待机扰动即使主任务通过也拒绝的审计回归。没有无接触/无碰撞认证，不外推真机精度。

正面录像先左后右，原速拼接，不是两臂同步动作；每例另一条手臂都在同一真实物理world受控保持home。原始视频6条及合成视频共7条逐帧解码/SHA256通过，查看双臂首帧与右臂末帧，桌面直接安装、两臂和夹爪可见。网页 http://127.0.0.1:8774/ 。关键配置/命令request.json，物理报告physics/report.json，末端复核physics/offline_audit.json，双臂复核dual_audit.json，录像校验和USD哈希result.json，来源provenance.json。

本次完成双臂配置与独立target运动验证；同步协同、抓放、交接、双臂公开模型动作接口与接触验收未实现/未验证。下一最小验证为原装夹爪开合和单物体真实抓取，未执行。
