# 组装场景接入本体与真实动作

2026-11-10。用户要求在geomind-research中加入已有机器人本体和执行动作。应用该skill与skill-creator，扩展现有research_pipeline和原生IsaacLabEpisode，不建立替代模拟环境。

## 已接入口

`stage=assemble_solve_execute`要求显式robot、世界基座位置与execution。机器人菜单列出Panda、FR3、Panda+Robotiq、Piper单臂和双臂；组装接触动作首版严格限Panda，其他本体保留既有到达能力并拒绝自然抓放替代。SelectByOrder使用PickLiftSelectedObject；静态几何目标使用PickPlaceObject。旧GT-only请求行为保留。

AssembledScene组合原scene.usda，`spawn=None`绑定原任务prim；reset只恢复已记录实测root/quaternion，保留0.5尺度，无辅助支撑、二次随机化或额外drop。运行时使用实际旋转的网格包络中心重新求解并冻结GT。原manifest、资产、初始落稳证据保持只读，独立机器人合同保存在robot_configs。

机器人动作只写关节位置目标，以真实夹爪接触搬动物体。原始root状态、关节控制/位置/速度、TCP姿态、查询GT、实际录像及正负对照均保存在robot_physics。独立审计用另一种四元数矩阵计算世界包络，复核查询GT、抬升/支持、速度、参考保持、手指状态、关节限制与最终驻留；不以TCP到位替代物体动作。

## 工程验证与保留失败

源场景为既有 `assembled_tasks_v2/order_scene65_line/assembly`：65号椭圆桌，4个蓝罐，X直线，seed11，左起第2个。Panda基座世界坐标[0,-0.45,0.9328504589976385]。该位置是本例选择，不是其他桌面通用安装位置。

1. `outputs/research/assembled_robot_v1/panda_pick_scene65`真实夹起正确物体约14.10厘米，但hold线速约0.022米/秒、角速约1.78弧度/秒，未满足0.01/.1阈值，稳定步数0。参考物体仅微米位移，关节限位正常。原闭合目标0与较大持续夹持压力是抖动的候选原因；正例失败，错误物体对照拒绝，原始证据和录像保留。
2. v2同场景、seed和任务：以实测宽度减6毫米的闭合目标约0.01969米，并明确配置接触线阻尼1.2、角阻尼4、位置求解32次和速度求解4次，参数仅写运行时组合层、记录在scene_provenance。角速降至约0.259，但仍未达阈值；同样保留失败，不计完成。
3. v3继续使用同任务与来源，将闭合余量减至1.5毫米（目标约0.02419米），其他控制过程、接触参数和验收阈值不变。能抬升约13.5厘米，但持稳窗口出现约3毫米轻滑与转动，仍未通过。
4. v4最后一次Pick专用几何修复将抓点从中心上方8毫米改为下方8毫米；该例在下降段末端误差约4.65毫米，未达3毫米到位门槛，未继续闭爪和抬升。原阈值保留，最终持续抓起例仍为action_failure；没有为覆盖两种动作继续盲调。

接触阻尼和高PD控制是显式仿真配置，并非经过真机标定的动力学参数。未放宽速度/稳定性标准、没有静默换seed或资产，失败不因看起来已拿起而标为通过。

## 证据和验证入口

使用 `python -m geomind.tools.research_pipeline --assembled-request configs/scene_assembly_tasks/panda_pick_scene65.json`；搬放示例为panda_place_scene60.json。请求输出目录存在时拒绝覆盖，网页复现使用独立replay请求。

`--audit-assembled RUN`分别核对原组装GT和机器人运行时查询/动作数据；也可直接调用 `python -m geomind.tools.audit_assembled_robot --run RUN`。离线复算不算物理重执行。只有正例实际动作、所请求对照和审计通过才标为task_action_physics_passed。

新请求/契约/配置/动作CPU测试41项、既有物理契约14项、旧research_pipeline6项、原场景组装36项通过，共97项。skill quick_validate与独立前向路由检查通过：能提交Panda自然抓起请求，并明确拒绝Piper抓放或改成ReachAboveTarget冒充操作。

## 最终实际结果

最终源码与配置证据为 `outputs/research/assembled_robot_v4/`，两个动作分别报告，不因离线审计一致就把动作失败当成功：

| 最终实例 | 正例动作 | 最终指标 | 对照 | 独立审计 |
|---|---|---|---|---|
| panda_place_scene60 | **通过**：抓取、抬升、搬运、松爪、退离与放稳 | 抬升13.305厘米；中心误差3.119毫米；稳定120步；参考最大位移约5.72微米 | 实际执行偏移8厘米落点；末态误差79.798毫米，正确拒绝 | 正/负例2/2一致，关节限位、GT、原始状态及视频hash核对通过 |
| panda_pick_scene65 | **未通过**：下降到位失败，未继续抓起 | stage4误差约4.65毫米；持稳0步 | 错误物体对照正确拒绝；该对照没有成功抓起错物体 | 正/负例2/2一致，不能据此说正例动作成功 |

最终完成机器人任务1例，未通过1例，负对照2例正确拒绝；v1-v3是保留的工程验证，不额外计入最终完成数。v3搬放也曾通过，v4在同样的Place参数下验证最终完整源码、技能入口、来源与媒体校验；Pick抓点变化没有修改Place参数。

最终两例共用启动前源码快照 `0f92df6a4373ae4e6f28ce9dd824b47cc6e171f68f7aa05794f5eb182ea9962a`，逐例配置ID与运行时GT见provenance/report。源USD和资产未改写，只有reset初始化恢复物体，执行期间物体姿态写入0、关节状态写入0、无吸附或外力搬运。Place运行时查询tick120，正例2810步、负例2798步；实际视频673/670帧，30fps。初始组装GT与机器人加载后的GT分别保存。

可直接查看[最终搬放证据页](http://127.0.0.1:8780/tasks.html)中的Panda搬放正例及错误落点录像。独立离线复算用 `python -m geomind.tools.research_pipeline --audit-assembled outputs/research/assembled_robot_v4/panda_place_scene60`，不重新执行物理。重复执行可使用该目录的replay_request.json（新输出目录），原始输入和历史证据保留。

持稳失败的潜在接触改进（有限指腹扭转摩擦或进一步夹持策略）未在本轮继续实施，不能归因为已验证原因。其他本体的组装抓放、VLA模型性能与无碰撞路径均未验证。

查看 http://127.0.0.1:8780/tasks.html ，机器人动作录像与初始落稳录像分开。实际机器人加载后的GT单列，研究者特权视图不作为受评agent输入；本轮仍是oracle_gt控制器实现验证，没有VLA模型成绩或无碰撞认证。
