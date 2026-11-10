# FR3 与 Panda 初始关节姿态对齐

按用户选择，仅调整原生target后端FR3初始关节姿态，与Panda逐关节一致：`[0, -0.569, 0, -2.810, 0, 3.037, 0.741]` rad，夹爪0.04m。从Panda配置映射关节名，不改FR3资产默认文件。USD实测限位检查七关节均合法。底座、场景65、seed0、target、控制与相机沿用前次。

已执行FR3正例1/1通过（误差6.059mm，991物理步，驻留20步），错误目标1/1正确拒绝，实测轨迹复核2/2通过；18项契约/录像测试通过。Panda沿用v1已验证原视频与报告，无重复运行。两侧初始关节角相同，不宣称几何外观完全重合或本体性能等同。

证据：`outputs/research/fr3_panda_target_matched_home_v2/`内request、configs、provenance、源码快照、physics报告和轨迹、视频校验result.json。对照视频原速，左FR3右Panda。已查看FR3新首帧与合成预览。网页 http://127.0.0.1:8769/ 。旧v1结果保留。仅ReachAboveTarget，不包含抓放或模型评估。
