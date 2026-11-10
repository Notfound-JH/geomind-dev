# 原有场景用于 native spatial episodes

三个生成任务现在引用 `registered_scene_65`，通过原任务的
`GeoMindSceneCfg.apply_scene(65)` 加载桌椅、地面、墙体、装饰和灯光。
桌面使用原 USD 网格碰撞，不再创建方盒代理桌面；与已有几何任务一样隐藏篮子，避免工作区遮挡。
场景内原任务的目标物不自动复制：新任务目标物仍由自己的 layout 定义。
当前三个任务仍使用红、蓝、绿几何块，任务动作仍是 ReachAboveTarget。

场景配置保存 `scene_mode=RegisteredScene`、`registered_scene_id`、`lighting_mode` 和
`hide_basket`。资产根目录复用 `GEOMIND_ASSETS_ROOT` 解析规则；USD 路径相对资产根目录。
桌面高度复用资产目录元数据计算，显式写入工作坐标原点和机器人底座位置。
运行时核对原场景与桌面高度，变化时要求重新生成配置，不静默改写任务坐标。
当前 table 坐标轴与世界坐标轴平行，原点在世界 x=0.5、y=0 的桌面标称高度。

生成和语义验证：

```bash
python -m geomind.tools.generate_spatial_suite --scene-id 65 --verify
```

不指定 `--scene-id` 则显式生成旧的 ProceduralTable 测试场景。
复用其他场景可传其他注册场景数字；每个新场景仍需单独验证工作空间与可达性。

无头执行并录制（本机环境）：

```bash
/home/elm/miniconda3/envs/new_isaaclab/bin/python -m geomind.tools.run_episode_physics \
  --suite generated_spatial_suite --video \
  --camera-eye 1.6 1.3 1.93 --camera-look-at 0.4 0 1.23 \
  --output outputs/registered_scene_video --timeout-s 1800
```

复用已有 debug 脚本的 MDL 材质目录：若未设置 `MDL_SYSTEM_PATH`，自动查找同级
`InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes/Materials`。
其他机器可显式设置此环境变量。case JSON 的 `scene_provenance` 记录加载器、实际场景
USD、场景文件 SHA256 以及生成的背景资产路径。

旧视频保留在 `outputs/headless_physics_video`，不覆盖；其冻结配置和代码对应旧实现。

## 本次验证

- CPU 测试 198 项通过；9 个生成配置通过独立语义 GT 校验。
- scene 65 中 3 个任务 × 3 个 seed，9/9 物理执行成功。
- TCP 位置误差约 4.7–7.5 mm，判定阈值 12 mm。
- `outputs/registered_scene_video/report.json` 保存逐 case 结果。
- `outputs/registered_scene_video.zip` 包含 18 段 1280×720、30 FPS 视频，
  同步帧索引、物理轨迹、配置快照和 GT。observer 可看到干预，agent 版本的记忆隐藏阶段遮黑。
- 抽查了记忆起始帧与推理完成帧，原桌面、地板和地毯材质正常显示。

scene 65 实际加载的背景为桌面、地毯、地板、四面墙和灯光；本场景没有加载椅子或台灯。
目标物仍为任务 layout 中的彩色块。执行策略仍为 GT oracle 控制机械臂到目标上方，
不代表已经接入视觉模型或抓取策略。数据包不包含外部 USD/MDL 资产本体，重放物理场景需原资产目录。

## 复用物体资产

```bash
python -m geomind.tools.generate_spatial_suite --scene-id 65 --object-set catalog --verify
```

`catalog` 使用已有 catch_object 资产中的 `blue_can`（咖啡罐）、`cube_can`（肉罐）、
`yellow_bottle`（黄色芥末瓶），使用原始模型贴图和原始轴向，不强制改成立放。
缩放分别为 0.45、0.6、0.4；布局对象的 `physical_asset` 显式保存 USD 路径、缩放、
尺寸、底部偏移、文件 SHA256、凸包碰撞方式与 `RigidBodyOrigin` 参考点。
运行时检查模型文件哈希，模型变动需重新生成配置。

GT、MoveObject 目标和 ReachAboveTarget 的位置均使用刚体原点，指令明确此约定；
它不必等于模型质心或视觉包围盒中心。记忆对象按照各自底部偏移放置在桌面上；
推理任务保留不同高度的参考点，用支撑台托住物体。模型碰撞来自网格凸包，
这对罐子和瓶子是近似碰撞，不保留凹腔。

近景视频录制：

```bash
python -m geomind.tools.run_episode_physics --suite generated_spatial_suite --video \
  --camera-eye 1.3 0.95 1.85 --camera-look-at 0.45 0 1.13 --camera-focal-length 24 \
  --output outputs/catalog_objects_verified --timeout-s 1800
python -m geomind.tools.package_episode_videos --root outputs/catalog_objects_verified
```

近景优先展示桌面物体，机械臂上部可能位于画外。`blocks` 选项仍保留最初的方块配置，
不会静默回退：资产文件缺失或哈希不匹配时直接报错。外部模型与贴图不复制到视频 ZIP 中。

物体替换最终验证：`catalog_objects_verified` 批次 9/9 正例通过，TCP 误差 4.8–7.6 mm；
CPU 测试 209 项通过。原先 `catalog_objects_video` 为试跑，最终证据与打包数据请使用 verified 批次。
