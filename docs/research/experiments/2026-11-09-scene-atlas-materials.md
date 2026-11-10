# 场景图鉴材质重渲染

2026-11-09。用户要求修复场景图鉴预览的纹理，沿用本地网页 http://127.0.0.1:8780/ 。本轮未上传、未发布在线站点。

## 结果

102/102 场景已逐个从实际 USDA 静态 RTX 重渲染，768×768；优先使用 `_new.usda`，不存在时使用原文件。全部图片可解码，102个图片哈希与逐场景记录一致；未解析资产引用0，最终正式批次与最终材质试例日志中的 `[Error]` 行0。184份 USDA 本地文件路径仍保留。

总览已人工查看，没有此前大面积红色缺失材质效果。网页保留旧缩略图对比，21个此前缺预览的场景也有新渲染。网页脚本语法检查通过，本地HTTP资源读取通过；浏览器控制服务不可用，未声称完成浏览器视觉/交互验收。

## 修复

- 启动 Kit 前设置 `MDL_SYSTEM_PATH`，包含 `InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes/Materials` 与本轮本地 NVIDIA 模块根目录。
- 四种原木地板模块 Raw02/03/04/05、官方模板和13份原始贴图已本地缓存，保留原相对目录结构。仅覆盖材质参数仍会触发原模块远程默认贴图加载，最终改为完整原模块镜像。没有用近似材质代替原材质。
- 旧 `/media/elm/files/nvidia_assets/.../Assets/ArchVis/` 家具贴图映射到本机已有官方 `isaacsim_assets/Assets/Isaac/4.5/NVIDIA/Assets/ArchVis/` 对应文件。
- 每个场景生成独立 USDA 预览叠加层：相机、补光、墙体剖视隐藏、路径修复均在新层中，不写回原始场景。正式批次GPU串行，worker上限420秒。仅静态渲染，不运行物理任务或模型评估。

## 证据

输出 `outputs/scene_atlas_materials_v1/`：`delivery.json`（102完成、pending为空）、`qa.json`、`all_scenes_contact_sheet.jpg`、`batch_status.json`、各 `batch_*/scene-*.json` 与 `scene-*-preview.usda`、`material_cache/`、`source_snapshot/sha256.json`。1和100的最终试例在 `retry_local_mdl_probe/`，其余100例在9个正式batch目录。

旧图与旧索引保留在 `original_atlas/`。首轮Replicator等待、清理API错误、在线木地板红色试图与日志均保留在probe/retry目录及 `debug_logs/`，没有以失败图作为最终网页交付。

配套旧JSON与部分新版USDA的物体类型/数量不同；本次保留配置值并在网页明确标注，不将JSON坐标当作渲染资产实测值。下一独立工作是核对并更新这些旧元数据，本次未执行，不影响材质预览完成状态。
