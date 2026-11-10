# 在另一台 Linux 主机配置 GeoMind

这份指南复现本次交付的代码和资产。安装检查与真正的 GPU 动作验收分开执行；目标主机实际通过有界 smoke 后，才可认定该主机能够运行对应工作流。

## 固定版本与主机条件

| 组件 | 本次观察/固定值 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS，x86_64 |
| 仿真 Python | 3.11.14 |
| Isaac Sim | 5.1.0，安装元数据为 5.1.0.0 |
| IsaacLab 上游 | `54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0` + `minimal-runtime.patch` |
| IsaacLab distributions | `isaaclab 0.51.1`、`isaaclab-assets 0.2.4`、`isaaclab-tasks 0.11.10`、`isaaclab-mimic 1.0.16` |
| PyTorch / torchvision | 2.7.0+cu128 / 0.22.0 |
| NumPy / Gymnasium | 1.26.0 / 1.2.1 |
| h5py / pytest | 3.15.1 / 9.0.2 |
| 观察到的 GPU / 驱动 | RTX 3090，24 GiB，580.173.02 |

本机 GPU 和驱动是实际观察，不是任意主机的兼容性保证。NVIDIA 的 5.1 固定版本要求页面列出 Ubuntu 22.04/24.04、x86_64、32 GB RAM，GPU 表列 RTX 4080/16 GB，Linux 测试驱动为 580.65.06；应以[官方 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)核对目标机。这里没有验收 Windows 或 macOS。NVIDIA 当前将 5.1 文档标为 unsupported；为复现本次结果仍固定该版本，升级应另作验证。

资产解压体积约数十 GB，最终字节数以资产 manifest 为准。磁盘需同时容纳 SDK/扩展缓存、下载 ZIP、解压资产与新输出。不要仅按源码仓库大小安排空间。

## 1. 获取代码和系统工具

```bash
git clone https://github.com/Notfound-JH/geomind-dev.git
cd geomind-dev
# 固定到本次交付的 commit；不要用未记录的 latest 替代。
```

安装 `git`、`unzip`、`ffmpeg`（包含 `ffprobe`）、C/C++ 编译工具及 CMake/Ninja。NVIDIA 驱动应在目标机正常运行，先检查 `nvidia-smi`。驱动和 SDK 的系统依赖依照上面的官方要求安装。

源码中保留了 `cripts/` 和原作者的 `scene_construction ` 目录；后者末尾有空格，shell 使用时必须引用完整路径。它们是历史 authoring 工具，不是当前 27 类原生抓放入口。原代码、固定框架补丁和文档已保留；实验结果、个人凭据、缓存与 SDK/vendor 副本不提交到源码 Git。

## 2. 建立仿真环境

```bash
conda create -n geomind_sim51 python=3.11.14 pip -y
conda activate geomind_sim51
python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

安装和许可步骤依据[Isaac Sim 5.1 pip 文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)。Python 3.11 是该版本要求。阅读并按官方步骤接受所需许可。不要把 Isaac Sim 5.1 装进旧的 Python 3.10 环境。

PyTorch/CUDA wheel 组合依据 [IsaacLab 2.3.1 的固定安装说明](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html)；项目源码仍以本交付实际 IsaacLab SHA 为准。PyTorch wheel 自带运行依赖不等于为可选 CUDA 扩展提供完整编译工具链。

## 3. 恢复固定 IsaacLab 与本地 runtime 修改

在 GeoMind 仓库根执行：

```bash
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
git -C IsaacLab checkout --detach 54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0
git -C IsaacLab apply --check "$PWD/migration/isaaclab/minimal-runtime.patch"
git -C IsaacLab apply "$PWD/migration/isaaclab/minimal-runtime.patch"
./IsaacLab/isaaclab.sh -i none
python -m pip install -c constraints.sim51.txt -e '.[dev,analysis]'
```

`-i none` 安装框架扩展并跳过当前任务生成不需要的 RL 训练框架；该参数由固定提交的安装脚本支持。项目依赖使用随仓库提供的 constraints，避免新机自动选到不同的分析/数值库版本。

目标 `IsaacLab/` 应为新 clone，不能重复应用补丁。`migration/isaaclab/runtime_sources.json` 记录固定 SHA、补丁 SHA-256 和必要源文件哈希。这个 runtime 补丁包含当前使用的 SkillGen/mimic 合同扩展、对象 remapping 和 FR3 配置。

`full_source.patch` 是另一种重建选项，保留原工作站全部 tracked 差异与记录的新增源码，包括自定义探针、教程和实验脚本；它包含原 `record_demos.py` 的缩进问题。它作为完整源码溯源保存，默认使用单独检查过的 `minimal-runtime.patch`。**不要把 full 和 minimal 叠加应用。** 详见[框架迁移说明](../migration/isaaclab/README-current.md)。

安装脚本可能调整依赖，之后核对版本和 editable 路径：

```bash
python -m pip show isaacsim isaaclab isaaclab-assets isaaclab-tasks isaaclab-mimic torch numpy
python -c 'import geomind, geomind_lab; print(geomind.__file__); print(geomind_lab.__file__)'
```

原主机曾有外部 IsaacLab editable 路径。目标机应从上述新 checkout 重新安装扩展，不复制旧 `.pth` 或环境目录。项目 bootstrap 默认使用仓库内 `IsaacLab/`；如采用不同布局，应配置对应 root 并通过 preflight 核验所有导入源。

## 4. 解压资产并生成本机配置

取得随交付下载的 `geomind-assets-v1.zip` 与 SHA-256 文件。ZIP 不上传到源码 Git；该包保存当前项目资源和递归 USD/MDL/纹理依赖。

```bash
sha256sum geomind-assets-v1.zip
unzip geomind-assets-v1.zip -d /your/data/directory
python scripts/configure_portable_assets.py \
  --assets-root /your/data/directory/geomind-assets-v1 \
  --isaac-python "$(python -c 'import sys; print(sys.executable)')"
source .env.geomind
```

包内已经将资产引用归一为可迁移路径。配置器生成 `.env.geomind` 与 `.geomind_local/assets_catalog.json`，并配置 repo assets 的本机入口。scene atlas 从包内复制到 `.geomind_local/scene_atlas`，后续 UI 渲染使用本机工作副本。保持资产 ZIP 顶层目录结构，详见 [ASSETS.md](../ASSETS.md)。已有非空目录不会被悄悄覆盖。

重新登录后，先激活 `geomind_sim51`，再在仓库根 `source .env.geomind`。不需要把资源挂回 `/home/elm`，也不要通过建立原用户名的系统目录掩盖漏迁移的引用。

## 5. 运行无仿真的安装检查

```bash
python scripts/check_portable_install.py \
  --simulation --assets-root "$GEOMIND_ASSET_BUNDLE_ROOT" \
  --output outputs/new_host_preflight.json

# 首次传输后建议核对完整包；会顺序读取数十 GB。
python scripts/check_portable_install.py \
  --simulation --assets-root "$GEOMIND_ASSET_BUNDLE_ROOT" --deep-assets

PYTHONPATH=tests python -m unittest \
  tests.test_extended_static_tasks \
  tests.test_extended_temporal_tasks \
  tests.test_extended_geometry_tasks \
  tests.test_native_registration_routing -q
```

preflight 检查文件、版本、固定框架源、模块路径和资产 manifest。`--simulation` 还会在所选 SDK Python 的 USD-only 环境中运行一个最长 30 秒的 CPU 探针，只导入 `pxr.Usd` 并返回版本，确认独立审计能找到 USD 库；不导入 Kit/App，不启动仿真或执行机器人动作。普通 CPU 模式不运行该探针。完整历史测试集还包含旧接口测试；运行并保存其结果，不把旧通过数复制到新主机，也不为包装修改任务答案。

## 6. 真正的 GPU / 资产 / 动作 smoke

先运行一个真实选择任务：

```bash
python -m geomind.tools.regenerate_registered_tasks \
  --output outputs/new_host_region_smoke \
  --tasks SelectByRegion_v0 --scene-id 65 --seed 0 \
  --max-steps 12000 --robot-base -0.12 0.32 0.9328504589976385 \
  --isaac-python "$GEOMIND_ISAAC_PYTHON"
```

读取 `batch.json` 和任务 `result.json`，分别确认：初始落稳、运行时 GT、完整正例抓起持稳、错误对象实际抓起持稳后被 GT 拒绝、独立原始状态审计和来源检查。失败例的日志、原始状态与视频应保留。每个 worker 上限 420 秒、GPU 串行、最多 12000 动作 ticks；不要先跑 27 类大批次。

选择任务至少实际抬升 60 mm，连续稳定至少 0.5 秒；放置任务还要完成运输、松爪、退离、合法支撑和落稳。原项目更严格的实例要求优先，不能把 CPU/mock、到达目标上方或无效负控记作抓放成功。

再根据研究卡使用同一 pipeline 运行所需类型。本轮新增是明确命名的有界 Base 配方；固定变体、稳定结构化 ID 或精确状态的可用性不构成纯图像模型能力或一般化成绩。当前测试默认不调用学习模型。

## 7. 录像、网页和断点下载

```bash
python -m geomind.tools.dashboard --port 8765
```

在浏览器打开 `http://127.0.0.1:8765`。远端服务器可用 SSH 转发：

```bash
ssh -L 8765:127.0.0.1:8765 -L 8782:127.0.0.1:8782 user@server
```

发布文件可用只读 release 服务：

```bash
python scripts/serve_delivery.py --directory /absolute/path/to/completed_release --port 8782
# 在本机下载；中断后重复可续传。
curl --fail --continue-at - --output geomind-assets-v1.zip \
  http://127.0.0.1:8782/geomind-assets-v1.zip
```

该服务只允许指定 release 目录中的文件，默认绑定 loopback，支持单 Range、HEAD、Content-Length 与 ZIP 下载头。不提供目录列表，也不允许目录外 symlink。代码 ZIP 和资产 ZIP 应先完成并校验，再启动服务；不要边写 ZIP 边发布。大 ZIP 使用 ZIP64，并用 SHA-256 校验最终内容。

历史原始研究录像/轨迹属于独立证据交付。仅 clone 代码和解压资源，不会自动恢复旧 outputs；也不要把下载资源完整记为新主机动作已通过。

## 可选工作流

`scene_construction ` 中的 LLM 辅助 authoring 运行前应显式配置 `OPENAI_BASE_URL` 和所选服务对应的 `OPENAI_API_KEY`，使服务地址与凭据来源一致。凭据只放本地环境变量，不写入源码、配置、ZIP 或 Git。当前 27 类原生组装/验收不调用这些服务。

可选 cuRobo 源码固定为 `ebb71702f3f70e767f40fd8e050674af0288abe8`，本交付保存两套原工作站 dirty 变体的源码补丁与 provenance，见 `migration/curobo/`。当前 Sim 5.1 验收环境未安装 nvidia-curobo，27 类 pipeline 不依赖它。需要旧规划工作流时，选择相应变体，恢复其资产配置，并按目标机 CUDA/PyTorch 重建扩展及单独 smoke；不要复制原机器的 `.so` 或声称它已被本轮验收。

## 环境记录

```bash
python scripts/export_environment.py --output outputs/new_host_environment
```

输出包含已安装的包名/版本和 GPU 驱动摘要，不记录环境变量值、凭据或私有 index URL。它是观察清单，不是可盲目交给 pip 的完整 solver lock。保留源码 commit、资产 manifest、环境清单以及新机实际 smoke 结果，构成该主机的复现记录。
