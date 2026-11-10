# IsaacLab 本地改动迁移包

本目录保存必要的本地源码差异，不包含 IsaacLab/cuRobo 上游副本、Conda 环境、模型、USD/mesh 资产或运行数据。补丁通过 `git diff --binary` 导出，新增文件通过 `git diff --no-index /dev/null ...` 导出；没有修改源 IsaacLab 工作区。

## 固定上游及内容

- IsaacLab：<https://github.com/isaac-sim/IsaacLab.git>
  - 基线提交：`54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0`
- cuRobo：<https://github.com/NVlabs/curobo.git>
  - 本仓库内 `IsaacLab/curobo`、`IsaacLab/src/nvidia-curobo` 的 HEAD 均为 `ebb71702f3f70e767f40fd8e050674af0288abe8`。
  - `git describe`：`v0.7.7-5-gebb7170`；安装元数据版本：`0.7.7.post1.dev5`。
  - **只记录上游版本，不复制 vendor，也不声称干净上游等价于本机已修改的 cuRobo。**

| 文件 | 内容 |
| --- | --- |
| `0001-skillgen-fr3.patch` | `SubTaskConfig` 对象别名/起始信号/偏移范围验证；数据生成器当前对象重映射钩子；FR3 注册及完整新增 `robots/fr3.py`；Isaac Sim 4.5 rendering_modes 的 `extension.toml` 标记 |
| `0002-optional-visual-debug.patch` | 可选：地面默认蓝色，以及 cuRobo 规划状态打印。非核心功能依赖，不默认应用 |

新增 `.fr3` 导入所需的 `fr3.py` 已完整配套；该文件只新增标准库 `math` 和已有 IsaacLab 模块依赖。保留了原始注释、参数及本地路径，不借迁移修改机器人行为。`extension.toml` 只有 `# This is not an extension`，不是新增渲染扩展。

没有收录：文档安装镜像地址调整 `isaaclab.sh`、教程/探针、`egl_probe/`、`my_sc/`、cuRobo 的整个目录、CUDA 安装包及实验 FR3 配置。另行检查发现 `IsaacLab/scripts/tools/record_demos.py` 的本地 gamepad 分支有缩进错误，`IsaacLab/scripts/tutorials/05_controllers/run_diff_ik.py` 注释掉了关节目标计算。这两项实验性修改不作为迁移依赖导出；新机器保留固定提交的上游脚本。**本包不是整个 dirty IsaacLab 的完整快照。**

## 干净环境应用

以下命令在新服务器的 Geomind 仓库根目录执行。目标 `IsaacLab` 必须是尚不存在的新目录；不要覆盖已有工作目录。此名称是项目 bootstrap 的默认约定。

```bash
REPO_ROOT="$PWD"
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
git -C IsaacLab checkout --detach 54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0
git -C IsaacLab apply --check "$REPO_ROOT/migration/isaaclab/0001-skillgen-fr3.patch"
git -C IsaacLab apply "$REPO_ROOT/migration/isaaclab/0001-skillgen-fr3.patch"

# 可选：需要原工作区的视觉/调试行为时才执行
git -C IsaacLab apply --check "$REPO_ROOT/migration/isaaclab/0002-optional-visual-debug.patch"
git -C IsaacLab apply "$REPO_ROOT/migration/isaaclab/0002-optional-visual-debug.patch"
git -C IsaacLab status --short
```

应用前应确认目标无已有更改；补丁不能重复应用。不要使用 `--reject` 掩盖版本不符。cuRobo 应从上面记录的官方仓库单独获取并 checkout 固定提交，再按该版本安装说明编译；CUDA 扩展必须在目标机器及目标 PyTorch 环境重建。

## 必须迁移时处理的路径与环境风险

1. 实际环境记录为 **Python 3.10.19、Isaac Sim 4.5.0.0、PyTorch 2.7.0+cu128**。这是现有环境记录，不是已验证的全新安装兼容性保证；不要仅依据当前上游默认版本重建。
2. 本机 `isaaclab-0.51.1.dist-info/direct_url.json` 明确指向
   `/home/elm/projects/IsaacLab/source/isaaclab`，而本包提取自
   `/home/elm/projects/geomind_lab/IsaacLab`。**这是两个不同路径**；不能假设本包源代码就是本机运行时加载的代码。
3. `nvidia_curobo-0.7.7.post1.dev5.dist-info/direct_url.json` 指向
   `/home/elm/projects/IsaacLab/src/nvidia-curobo`，也不是本仓库内的两个 cuRobo checkout。本包确认的提交属于仓库内 checkout，不证明外部 editable 路径具有同一提交或修改。
4. 迁移后必须在选定环境重新安装目标源码的 editable 包，检查 `pip show isaaclab isaaclab-assets isaaclab-mimic nvidia-curobo` 和相应 `direct_url.json`；在正常启动 Isaac Sim/IsaacLab 的进程内核对模块 `__file__`。不要复用指向旧机器路径的 `.pth` / editable finder。
5. FR3 配置中的 USD 路径仍是
   `/home/elm/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/Franka/FR3/fr3.usd`。
   必须单独取得合法的对应资产及其递归 USD/mesh/纹理依赖，挂载到同一路径，或在任务配置中覆盖 `spawn.usd_path`（也可迁移后修改新 checkout）。本包不包含这些外部资产，不能仅靠补丁运行 FR3。
6. 仓库内两个 cuRobo checkout 都修改了 `franka.yml`；其中一个还新增多套 `fr3*.yml`、碰撞球配置、`fr3_description`，另一个有 `non_shipping` 资产。本包没有新增对这些实验文件的引用，也没有在非 scripts 的项目 Python/YAML 中找到相应 FR3/cuRobo 配置引用。**不能据此证明外部 editable checkout 或脚本没有依赖它们**。若运行工作流选择这些配置，需另行确认实际配置、URDF/mesh/碰撞球文件及许可并打包；不要把这些目录全部当作源码提交。

## 验证范围

已用 `git archive HEAD` 导出纯基线副本，在独立验证目录对两个补丁依次执行 `git apply --check`、`git apply`，并逐文件与原工作区比较；验证目录验证后删除。新增的 FR3 和 rendering_modes 标记均在比较范围内。

补丁采用白名单文件导出，并对输出做常见凭据模式检查；未发现匹配，不输出疑似凭据值。它不是全面安全审计。

本次没有安装/升级环境，没有运行 GPU 仿真、FR3 资产加载、cuRobo CUDA 扩展构建或完整录制/生成流程，也没有核查外部 `/home/elm/projects/IsaacLab` 源码。通过补丁检查只证明文本可重建，不证明硬件、资产及运行时依赖已齐备。
