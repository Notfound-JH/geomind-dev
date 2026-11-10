# 当前 Isaac Sim 5.1 runtime 源码恢复

固定上游：<https://github.com/isaac-sim/IsaacLab.git>，commit `54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0`。

本次实际仿真环境使用 Python 3.11.14、Isaac Sim 5.1.0.0。原机器运行环境曾指向仓库外 editable checkout；已对该实际源码导出补丁，而不是假定工作区中任意副本就是运行版本。

| 文件 | 用途 |
| --- | --- |
| `minimal-runtime.patch` | 默认安装。SkillGen/mimic 合同字段、当前对象 remapping、FR3 注册与定义，共 4 个源文件 |
| `full_source.patch` | 完整溯源。全部 8 个 tracked 差异，以及 manifest 列出的 45 个新增作者/probe 源文件，共 53 个文件 |
| `runtime_sources.json` | 上游 SHA、两个补丁哈希、必要源文件哈希、完整新增源列表 |
| `patch_verification.json` | 在独立临时目录从 pinned baseline 重建并逐文件比较的结果 |

两个补丁是替代选项，**不能叠加应用**。完整补丁保留了原实验 `scripts/tools/record_demos.py` 的 `IndentationError`、教程探针和调试修改，不能当默认生产安装。原始自定义源码没有丢弃；用户可从 full patch 单独审阅并修复后使用。

默认安装从新 clone 执行：

```bash
git -C IsaacLab checkout --detach 54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0
git -C IsaacLab apply --check "$PWD/migration/isaaclab/minimal-runtime.patch"
git -C IsaacLab apply "$PWD/migration/isaaclab/minimal-runtime.patch"
./IsaacLab/isaaclab.sh -i
```

只有 `git apply --check`、逐字源重建和 Python 语法检查在补丁验证范围内；它们不等于目标主机 GPU 或资产成功。必要 runtime 四个源文件的重建与语法检查全通过。完整 53 文件逐字重建通过，已将原有实验脚本语法问题记录在 verification 中。

FR3 的历史定义保留了原资源字符串，GeoMind 的可迁移本体选择/manager 配置必须通过 `GEOMIND_ISAAC_ASSETS_ROOT` 的实际资产覆盖路径。不要在新机器建立旧用户名目录。相应资产及其引用闭包位于单独 ZIP，不能只应用框架补丁便认为模型资源已齐备。

可选 cuRobo 不嵌入此 patch，见 `migration/curobo/` 的两个独立变体。SDK、Conda 环境、CUDA 安装包、Kit/cache 和探针编译产品均不放进源码 Git。
