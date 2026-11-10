# SceneUpdating Base：变化识别后抓起物体

日期2026-11-09。用户请求SceneUpdating Base、semantic_table、seed0起2例、isaaclab、420秒、输出next_base。原目录存在，按skill保留旧产物；失败v1/v2及修复单例v3、第二例v3分别独立存放。预算按每worker420秒，GPU串行。

## 任务和执行

代表DetectSceneChange位移版，自然操作指令“记住桌上方块的位置，等画面恢复后，把刚才被移动的方块拿起来。”未替换公共SelectObject语义，增加PickLiftChangedObject执行动作；选择GT仍为同实例observe.End到query唯一位移>3cm的物体。不是新增/移除定义，也非全部SceneUpdating覆盖。

生成配置采用semantic_table的ProceduralTable，明确遵从用户场景，未宣称加载搭建场景65。2步观察、3步隐藏延迟（0.5秒/协议步），隐藏期以外力PD移动物体，恢复可见后冻结GT并停止事件外力；机器人通过夹爪接触抓起选中物体，不瞬移、吸附或用外力辅助搬运。生成CPU流保留SelectObject诊断契约，实际物理observation使用自然指令；无模型调用。

成功需正确物体抬升>6cm、线速度<1cm/s、其他物体位移<=5mm、夹爪保持闭合、连续>=60个120Hz物理步。物理持握，不是放置任务，故不要求松爪。独立审计历史/查询快照与轨迹、唯一位移GT、隐藏观察、时间、动作/实测关节限制和持握末态。无接触传感/无碰撞认证。

## 结果与失败保留

|seed|位移GT|抬升|稳定步数|对照|
|---|---:|---:|---:|---|
|0|0.188721m|0.150998m|108|错误物体正确拒绝|
|1|0.205481m|0.150977m|106|错误物体正确拒绝|

CPU/GT两例通过，最终物理正例2/2、错误目标对照2/2正确拒绝、轨迹复算4/4、视频8/8（帧数/首尾解码/SHA256）检查通过，查看seed1末帧确认持握。63项受影响测试通过，包括新SceneUpdating默认动作路由与既有语义/物理契约，不能外推全部任务能力。

首轮next_base_scene_updating_v1两正例因夹持接触速度约0.054m/s不稳定失败；v2降低夹爪挤压至0.0125m并增加接触求解迭代后仍约0.016m/s，未过1cm/s阈值。v3夹爪目标0.014m、物体位置/速度求解迭代均32，保持判据不变，先seed0单例再seed1串行通过。原始失败录像、配置、源码快照全部保留，审计确认失败判定一致；不能将仅重算一致算执行成功。

失败正例归action_failure；未发生semantic_error、invalid_instance、system_failure（错误对照的语义/动作失败为预期注入）。早期中点契约测试fixture缺失robot字段已修正，其失败为测试实现问题，实际物理运行未受影响。

## 来源与复现

- seed0：`outputs/research/next_base_scene_updating_single_v3`，源码快照 `5a361eb58cca2ff5448b92d912d14b825acb7250580a09cabd96435afe0236b4`，配置 `253fc977cce44c823e9c87e300a0f79baf0d1e85fd0f42a0e62e5030c7861ae7`。来源检查通过，实际命令见physical_command.json，所有相关未提交源码内容保存在source_snapshot。
- seed1：`outputs/research/next_base_scene_updating_second_v3`，源码快照 `5a361eb58cca2ff5448b92d912d14b825acb7250580a09cabd96435afe0236b4`，配置 `9ecc1e1fd547c7826c2ade3ad235c7ffcf1161a8cf85afd12b123927fc0216fd`。来源检查通过，实际命令见physical_command.json，所有相关未提交源码内容保存在source_snapshot。

```bash
/home/elm/miniconda3/envs/new_isaaclab/bin/python -m geomind.tools.research_pipeline --request configs/research_requests/scene_updating_base_20261109_v3.json
/home/elm/miniconda3/envs/new_isaaclab/bin/python -m geomind.tools.research_pipeline --request configs/research_requests/scene_updating_base_20261109_second_v3.json
/home/elm/miniconda3/envs/new_isaaclab/bin/python -m geomind.tools.research_pipeline --audit outputs/research/next_base_scene_updating_single_v3
```

生成入口拒绝覆盖，重新执行需复制请求指定新目录。上列audit仅离线复算，非再执行。真实PhysX+oracle_gt验证实现闭环，不是模型能力或纯视觉评估。应用报告后有界清理成功；重试总时长不等于单worker420秒限时。

网页：http://127.0.0.1:8766/ ，可切换完整和agent遮黑视角，来源摘要scene_updating/summary.json。下一最小实验（未执行）：在保持物理控制不变的同终态反事实配对中验证标签切换，再接固定版本模型比较full-history/final-only；本次两个seed的目标均为block_1，不能证明跨标签泛化。
