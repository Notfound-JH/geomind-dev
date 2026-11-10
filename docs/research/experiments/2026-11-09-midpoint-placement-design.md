# 自然操作任务：中点放置设计

本次重新应用geomind-research，按用户已确认意图生成两例“把绿色方块放到红蓝两个方块的正中间”。采用registered_scene_65 Base seeds0/1。独立新目录 outputs/research/midpoint_placement_design_v1。未使用会拒绝manipulation的诊断流水线替代操作生成；产物为任务设计草案，非已注册可执行放置任务。

独立构造先采样中点c和水平朝向，再生成c±0.12d参考物体、c+0.13n绿色物体，三个3cm立方体中心标称高度1.5cm。检查等距、参考物体与落点间隙、名义边界和支撑高度；复用现有InferMidpoint resolver/SemanticEpisode做内部GT交叉核验2/2通过，共享schema/评分，不视作独立物理验证。ReportPoint episode仅为内部求解检查，实际自然操作指令及放置草案独立保存于tasks。

GT及设计已生成；物理抓放、接触、实际桌面支撑/可达性、释放后物体评分、模型评估均未执行。阈值1cm目标误差、参考物体5mm位移、稳定0.5s为草案，待物理标定。源码内容快照/配置ID见provenance.json及provenance_check.json。未复用历史视频冒充当前任务执行。

网页沿用 http://127.0.0.1:8766/，展示自然指令和初始布局示意，明确不是录像。下一步补齐原生操作执行器与实测物体末态评分，再在420秒单例预算中验证。
