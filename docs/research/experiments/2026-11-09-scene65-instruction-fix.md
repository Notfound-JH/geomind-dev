# 场景与指令修正

用户检查发现上一轮为程序化桌面且中文指令抽象。本次已执行：使用现有 research_pipeline.parse_request/prepare 生成 registered_scene_65 Base ProjectOntoLine seeds0/1，既有PhysX runner限时420秒，带录像和错误目标对照。原失败报告未覆盖；原semantic_table请求和配置保留。

实际加载器 GeoMindSceneCfg.apply_scene(65)，桌面/地毯/木地板/墙体/灯光的来源记录于 outputs/research/next_base_scene65_video/physics/report.json。源USD SHA256 d1272332bdbf4dd87e5f13ee9fce3505e729b4eb32030ffc3d80b341bd1553f5。已查看真实末帧，确认使用大理石桌面与室内背景，非程序化白桌。

CPU GT检查2/2；实际物理正例2/2成功，错误目标1/1正确拒绝，离线轨迹复核3/3通过；无新增系统/实例/实现失败。源码快照和配置哈希见 outputs/research/next_base_scene65_video/provenance.json；独立来源校验见 result.json。未接入模型或认证无碰撞。

中文展示使用“在红蓝中心连成的直线上，找离绿色物体中心最近的位置”，保留无限三维、可在线段外、坐标单位及刚体原点边界。另列机器人执行到目标上方10cm，不改变GT或英文求解器指令。已有指令测试4/4通过。网页沿用 http://127.0.0.1:8766/，新视频使用独立名称避免旧缓存。旧物理证据保留。

下一步（未执行）：由用户确认场景65是否为期望场景，若需要改用真实目录物体再单独核验角色与刚体原点。当前仍为几何标记方块，不能声称复用所有原场景物体。
