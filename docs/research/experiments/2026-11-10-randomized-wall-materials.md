# 102 场景墙面随机替换

已完成：场景1–102，共184份源USD（102份原场景及82份_new版本）。使用本地Poly Haven Beige Wall 001/002的2K颜色、粗糙度、OpenGL法线贴图；seed=20261110，平衡随机分配，各51场景。同一房间四面墙一致，纹理每2米重复。

[总览网页](../../../outputs/wall_randomization_20261110/index.html)、[完整分配/备份清单](../../../outputs/wall_randomization_20261110/manifest.json)、[核验](../../../outputs/wall_randomization_20261110/validation.json)。源文件在相邻geomind_assets/stage目录；修改前备份逐文件保存于parts/*/backup，清单包含前后SHA256。

墙体Cube改为等尺寸Mesh，显式faceVarying UV；保留原变换、extent及碰撞属性。184份源文件与备份哈希、实际墙面材质绑定、三类贴图本地路径、UV、非墙面内容一致性均核验通过。六张贴图MD5与下载清单一致。场景1（002）/2（001）已实际Isaac RTX渲染，图像和来源哈希核对通过。未进行本批次机器人动作或物理重执行，不将材质加载检查称为物理通过。

首次固定材质名替换漏掉83–102旧格式的visual_material_4，并误替换其PhysicsGroundPlane材质。独立复核发现后，按每面墙真实绑定从最初备份定点重做20文件；地面材质逐字恢复，四墙贴图均连接。保留[修复记录](../../../outputs/wall_randomization_20261110/binding_correction/manifest.json)与首次替换快照，增强后的核验覆盖这类编号差异。

历史录像与静态场景图未批量刷新；样例剖视隐藏仅存在于预览层。后续可按本清单重渲染需要展示的历史预览，未执行。
