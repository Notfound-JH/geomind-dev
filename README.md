# GeoMind

GeoMind 是基于 IsaacLab 的空间推理、空间记忆和机器人操作研究项目。代码包含任务注册、场景与物体组装、实测 GT、真实抓放执行、独立审计，以及录像展示工具。

主仓库：<https://github.com/Notfound-JH/geomind-dev>。

本交付按 **27 种 canonical 任务**组织；顺序选择的多个序位和拓扑放置的多个变体可属于同一种任务。任务定义、CPU 语义检查、真实机器人动作和模型表现分别记录。已有 GT、源码或录像本身不等于真实抓放通过。

本轮任务展示严格验收为 **27/27 类、30/30 实例**（含原20项的保存版本）。便携源码还通过一次独立新目录的真实正负抓放测试和行动后资产深度校验；SDK使用已有5.1安装。完整范围与结果见 [迁移验证记录](docs/validation/portable_release_20261111.md)。

## 新主机安装

完整步骤见 [安装与迁移](docs/installation_zh.md)，资产解压位置与环境变量见 [ASSETS.md](ASSETS.md)。AI agent 的工作约定见 [AGENTS.md](AGENTS.md)；用户要求的 [agent.md](agent.md) 为同一约定的入口。

本轮仿真复现基线为 Ubuntu 22.04、Python 3.11.14、Isaac Sim 5.1.0、PyTorch 2.7.0/cu128，以及固定 IsaacLab 提交 `54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0` 与随仓库交付的 runtime 补丁。主流程面向 Linux x86_64；Windows、macOS 和其他版本组合没有被本交付验收。

```bash
git clone https://github.com/Notfound-JH/geomind-dev.git
cd geomind-dev
conda create -n geomind_sim51 python=3.11.14 pip -y
conda activate geomind_sim51

# 安装官方固定版本；先阅读官方许可和主机要求。
python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
git -C IsaacLab checkout --detach 54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0
git -C IsaacLab apply --check "$PWD/migration/isaaclab/minimal-runtime.patch"
git -C IsaacLab apply "$PWD/migration/isaaclab/minimal-runtime.patch"
./IsaacLab/isaaclab.sh -i none
python -m pip install -c constraints.sim51.txt -e '.[dev,analysis]'
```

然后取得本次单独交付的 `geomind-assets-v1.zip`，核对其 SHA-256，解压至任意有足够空间的磁盘。ZIP 自带顶层目录 `geomind-assets-v1/`。

```bash
unzip geomind-assets-v1.zip -d /your/data/directory
python scripts/configure_portable_assets.py \
  --assets-root /your/data/directory/geomind-assets-v1 \
  --isaac-python "$(python -c 'import sys; print(sys.executable)')"
source .env.geomind
python scripts/check_portable_install.py --simulation --assets-root "$GEOMIND_ASSET_BUNDLE_ROOT"
```

代码、资产包和 SDK 分别交付。不要复制原主机的 Conda 环境、`.pth`、Kit cache 或个人凭据。SDK 按上面的固定版本在目标机安装；cuRobo 属于可选历史规划工作流，安装说明单列在迁移指南中。

## 验证

CPU 语义检查可以验证合同、GT 与拒绝条件，真实抓放还需要目标机的 GPU、资产和 PhysX 执行。

```bash
PYTHONPATH=tests python -m unittest \
  tests.test_extended_static_tasks \
  tests.test_extended_temporal_tasks \
  tests.test_extended_geometry_tasks \
  tests.test_native_registration_routing -q
```

先运行一个有界真实操作实例。以下命令会启动真实本地仿真、组装 scene 65、实测落稳、抓起正确对象，并实际执行错误对象负控：

```bash
python -m geomind.tools.regenerate_registered_tasks \
  --output outputs/new_host_region_smoke \
  --tasks SelectByRegion_v0 --scene-id 65 --seed 0 \
  --max-steps 12000 --robot-base -0.12 0.32 0.9328504589976385 \
  --isaac-python "$GEOMIND_ISAAC_PYTHON"
```

每个物理 worker 最长 420 秒，GPU 串行。先单例，再明确任务的小批次。失败保留到原目录；修复后使用新目录，不换 seed 或降低阈值补齐成功。腕相机与媒体检查、原始状态独立审计也属于交付条件。

逐例参数见 [30 实例复现矩阵](configs/reproduction/canonical27_verified.json)；可用 `python scripts/reproduce_canonical27.py --list` 或 `--dry-run --all` 无仿真查看，明确指定 `--tasks TASK_ID...`（或 `--all`）及 `--output 新目录` 才按各例场景、基座与尺度串行执行真实动作及腕相机审计；这是当前源码的新生成，默认首个失败即停止，不宣称历史字节复现。

## 展示与研究记录

```bash
python -m geomind.tools.dashboard --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。HTML 文件直接在编辑器中打开会显示源码。跨主机访问时使用 SSH 转发，例如 `ssh -L 8765:127.0.0.1:8765 user@server`，再在本机浏览器打开相同地址。

27 任务交付的研究边界与逐例证据说明见 [本轮实验记录](docs/research/experiments/2026-11-10-canonical27-completion.md)。历史 20 个实例覆盖 17 类，本轮新增 10 类；最终网页可按 27 类、30 个实例组织。原始录像、轨迹和历史 source snapshots 很大，放在独立证据交付中，不把它们当作源码提交到 Git。

## 目录

| 路径 | 内容 |
| --- | --- |
| `geomind/` | 任务 registry、配置、episode、组装、执行、审计与网页工具 |
| `geomind_lab/` | IsaacLab manager/mimic 任务及适配层 |
| `configs/`、`geomind/configs/` | 场景、任务、合同和请求配置 |
| `scripts/`、`tests/` | 入口、安装检查与回归检查 |
| `docs/`、`migration/` | 使用说明、研究记录、固定框架版本和补丁 |
| `assets/`、`scene_atlas/` | 源码中的轻量元数据；大资源由资产包提供 |
| `outputs/` | 本地主机运行结果，不随源码 Git 提交 |

原 `environment.isaac_curobo.yml`、`docs/migration.md` 与旧框架补丁记录的是 Python 3.10 / Isaac Sim 4.5 历史环境。新主机复现当前交付时使用 [新的安装指南](docs/installation_zh.md)。
