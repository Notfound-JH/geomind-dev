# 102个场景的空桌面另存版本

用户要求去掉102个场景桌面上的可操作物体并保存。已完成102/102，目录 `assets/scenes/tabletop_cleared_v1/`，网页 http://127.0.0.1:8780/ 默认空桌面版，可切换原始版。未修改原始资产，未上传或发布外部网站。

## 处理范围

直接检查102个实际USD的组合层级，不使用旧JSON推断物体。移除 `object` 任务物体、`container` 操作容器；旧格式场景识别World下平铺的瓶罐、盒子、魔方、篮子等。对decoration分组及平铺装饰中的杯子、小盆、书和摆件，结合桌面世界包围盒、XY范围、底面高度与尺寸识别桌面便携小件。17–32号误归入装饰的杯子和小摆件也移除。保留桌椅、柜子、灯具、周边落地装饰、墙体、地面和结构性桌面支撑。

总计移除638件。102例保存后均独立重新打开，比较剩余有效prim路径集合与预期，核验移除对象不再激活、桌子及其世界变换保持、原场景SHA256未变。物体元数据清零，原元数据保存在各目录source_scene.json。

## 保存格式与依赖

每例 `{编号}/{编号}.usda` 为标准USD分层文件，以原场景为sublayer，使用active=false移除对象。该方式使几何、刚体和碰撞一并退出组合场景，不只是visibility隐藏。不是脱离原场景的flatten副本；原场景、原外部资产及已修复材质缓存均需保留。每例目录有清理后JSON与removal_report.json，总索引manifest.json。

原场景阶段元数据已显式保留，包括60 timeCodesPerSecond、起止时间、upAxis和metersPerUnit等；102例通过metadata_audit.json复读校验。最终元数据补全发生在静态渲染之后，不改变default-time几何、材质或激活状态。渲染时的精确USD字节快照保存在outputs/scene_atlas_cleared_v1/usda_before_metadata/，各渲染记录保留source_sha256、source_snapshot和最终保存source SHA256，避免混淆版本。

## 预览与验证

17/65/100先保存并渲染，核验对象、容器和杯子移除后，再执行剩余99例。正式渲染GPU串行，每worker最多420秒，102/102生成768×768实际RTX预览，材质错误日志0、未解析资源引用0、图片解码与哈希102/102通过。人工查看代表全尺寸图与102图总览。只为预览添加相机、环境补光和墙体剖视隐藏，这些预览改动不写入空桌面资产。未运行物理任务，未验证抓取、碰撞或真实机器人。

网页保留原始版、空桌面版及详情内图片对比，空桌面详情复制新USD路径、物体数0。脚本语法及本机HTTP资源检查通过。浏览器控制服务不可用，未完成浏览器视觉/交互验收。

证据位于outputs/scene_atlas_cleared_v1/：delivery.json（pending_save/pending_render为空）、qa.json、inventory.json、metadata_audit.json、all_cleared_scenes_contact_sheet.jpg、render_status.json、各渲染报告与logs/、source_snapshot/sha256.json。
