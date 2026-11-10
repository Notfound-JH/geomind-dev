# 资产包与放置位置

本次代码与大资源分开交付。`geomind-assets-v1.zip` 包含当前项目使用的场景、物体、机器人、材料和纹理引用闭包；Isaac Sim/IsaacLab SDK、Conda 环境及运行录像不属于此 ZIP。每个文件的相对路径、字节数和 SHA-256 记录在包内 `asset_bundle_manifest.json`。


## 本轮下载文件

- 文件名：`geomind-assets-v1.zip`
- ZIP 大小：8,696,938,222 字节（约 8.70 GB）
- 解压后的资源：2,489 个文件，33,385,692,594 字节
- SHA-256：`f979987acc249e6ea5eb62a6e0aa14e171e91274042d05272e5dd3e24598175a`
- 包内 manifest SHA-256：`b13fa35efe7f79b8669f530e62d8693707d72bf94a3f6549e07e4194ad1586a9`

推荐放在仓库的 `external/geomind-assets-v1/`，此目录不入 Git：

```bash
mkdir -p external
unzip /path/to/geomind-assets-v1.zip -d external
python scripts/configure_portable_assets.py \
  --assets-root "$PWD/external/geomind-assets-v1" \
  --isaac-python "$(python -c 'import sys; print(sys.executable)')"
source .env.geomind
```

也可以放在其他磁盘，使用下面的 `--assets-root` 指向实际目录。

## 解压和配置

先核对随交付提供的 ZIP SHA-256，再解压到有足够空间的磁盘。可以使用任意用户名和路径，无需重建原主机 `/home/elm`。

```bash
sha256sum geomind-assets-v1.zip
unzip geomind-assets-v1.zip -d /your/data/directory
cd /your/code/geomind-dev
python scripts/configure_portable_assets.py --assets-root /your/data/directory/geomind-assets-v1
source .env.geomind
python scripts/check_portable_install.py --assets-root "$GEOMIND_ASSET_BUNDLE_ROOT" --deep-assets
```

`--deep-assets` 会读取整个资产包，可能耗时较长。普通 preflight 检查文件存在与尺寸；深度检查进一步核对所有 SHA-256。配置器写入本地 `.env.geomind` 与 `.geomind_local/assets_catalog.json`，这两个位置不提交到 Git。

```text
/your/data/directory/geomind-assets-v1/
├── asset_bundle_manifest.json
├── geomind_assets/          # stage、layout、objects、table、ground、robots 等
├── grscenes_home/           # 被场景实际引用的 GRScenes 材料与纹理
├── isaacsim_assets/         # 保持 Assets/Isaac/4.5 与 5.1 的必要机器人依赖结构
├── material_cache/         # 场景本地材质缓存
├── repo_assets/
│   ├── assets/             # 本项目的大资产，如处理后的 Piper 与材料
│   └── scene_atlas/        # 场景缩略图和本地展示资源
└── curobo_assets/           # 若 manifest 列入，可选 FR3 描述与 mesh
```

配置器会检查 checkout 中 `assets/`、`scene_atlas/` 与 bundle 的关系。scene atlas 会先复制到本机 `.geomind_local/scene_atlas`，新渲染和 UI 派生修改写入这个工作副本，保持包内 atlas 不变。已有非空目录不会被无提示覆盖。遇到冲突按命令报告处理；不要直接把 ZIP 覆盖到含有自定义数据的工作区。

## 环境变量

| 变量 | 指向 |
| --- | --- |
| `GEOMIND_ASSET_BUNDLE_ROOT` | 上述 `geomind-assets-v1/` 根目录 |
| `GEOMIND_ASSETS_ROOT` | `$GEOMIND_ASSET_BUNDLE_ROOT/geomind_assets` |
| `GEOMIND_GRSCENES_HOME` | `$GEOMIND_ASSET_BUNDLE_ROOT/grscenes_home` |
| `GEOMIND_ISAAC_ASSETS_ROOT` | `$GEOMIND_ASSET_BUNDLE_ROOT/isaacsim_assets`，其下是 `Assets/Isaac/...` |
| `GEOMIND_MATERIAL_CACHE` | `$GEOMIND_ASSET_BUNDLE_ROOT/material_cache` |
| `GEOMIND_REPO_ASSETS_ROOT` | `$GEOMIND_ASSET_BUNDLE_ROOT/repo_assets/assets` |
| `GEOMIND_SCENE_ATLAS_ROOT` | checkout 的 `.geomind_local/scene_atlas`，由配置器从包内 atlas 建立本机副本 |
| `GEOMIND_LAB_ASSET_CATALOG_PATH` | 配置器生成的 `.geomind_local/assets_catalog.json` |

USD/MDL 文件内的引用已按包内相对路径归一。只设置环境变量、却仍使用原工作站绝对引用的旧 USD，不等于完成迁移；应使用本次 ZIP 及其 manifest。

## 资产与证据的区别

本包提供重新运行所需资源。历史视频、物体/关节轨迹、配置与 source snapshots 属于另行交付的研究证据；重新安装代码与资产不会自动生成这些结果。

SDK 从官方固定版本单独安装。原供应方的许可、署名及 notices 随对应资源保留；可用包内来源 manifest 追溯文件，勿将资产 ZIP 误作为无条件再授权的源码包。

## Atlas 可迁移视图与原始证据


资产包中的 `repo_assets/scene_atlas` 保留展示所需资源。为迁移而重绑路径的 result、manifest、index JSON 会带有 `portable_view_only=true`、`original_source_sha256` 和 `portable_metadata_notice`；这些文件是可迁移 UI 视图，不是原始 PhysX 证据。原主机的源报告没有被修改。27 类任务的原始验收报告、轨迹、录像及其哈希在单独证据交付中保持不可变；不能用资产包内的 UI 元数据替代它们。

配置器从包内 atlas 建立本机 `.geomind_local/scene_atlas` 工作副本，后续新渲染和 UI 修改使用该副本。MDL 显式引用的 thumbnail PNG 属于资源依赖，会进入资产包；其余临时缓存、Git 元数据和编译产品不进入包。

为追溯官方 FR3 mesh，包内保留了固定版本的 Xacro/YAML 来源。重新展开这些可选模板需要对应 ROS/xacro/Gazebo 生成工具；manifest 的 `optional_source_template_requirements` 将它们与现成 URDF/DAE 的资源闭包分开。此次补全核对了 mesh 及其关节/几何参考框架，没有执行可选 cuRobo/CUDA planner，也没有重写两份历史 URDF 的参数。

