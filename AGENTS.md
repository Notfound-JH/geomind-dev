# GeoMind agent 工作约定

## 项目入口

- 先读 `README.md`、`docs/installation_zh.md`、`ASSETS.md`。
- 机器人研究工作使用 `.agents/skills/geomind-research/SKILL.md` 及其对应 reference。
- 任务 taxonomy 以 `geomind/core/taxonomy.py` 为准；原始任务定义位于 `geomind/configs/task_definitions/`。不要为补齐展示另建一套互相冲突的 registry。
- 当前主机安装配置来自 `.env.geomind` 和 `.geomind_local/`，它们不入 Git。源码不得假设原作者的用户名、home 目录或 Conda 环境名称。

## 安装与可迁移性

当前交付固定 Python 3.11、Isaac Sim 5.1.0、IsaacLab commit `54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0` 与 `migration/isaaclab/minimal-runtime.patch`。先确认实际解释器和模块导入路径，再运行任务。不要以旧 Python 3.10 / Sim 4.5 环境文档替代当前安装流程。

资产包目录由 `GEOMIND_ASSET_BUNDLE_ROOT` 指定，使用 `scripts/configure_portable_assets.py` 配置。不得把个人绝对路径、SDK 副本、虚拟环境或缓存加入代码。新增可配置路径必须有明确默认规则、缺失资源报错和异路径检查。

## 研究和物理验证

- 自然操作意图必须保持对应的语义机制与输入必要性。选择任务实际抓起并持稳；放置任务实际抓取、搬运、松爪、退离和放稳。`ReachAboveTarget` 只是一类诊断动作，不能替代抓放。
- GT 从运行时实测状态求解。时序任务分开保存事件计划和实测历史，不用计划终点冒充观测。
- 正例与负控分开。错误物体/错误落点必须被真实执行并稳定，再由冻结 GT 拒绝；到达失败、抓取失败或运输失败不是有效负控。
- GPU 并发为 1，每个物理 worker 上限 420 秒；先单例再小批次。采用用户明确的预算。不要自动扩大到训练、远程服务或大规模实验。
- 机器人动作期间只发送控制目标。禁止用物体瞬移、吸附、焊接、隐藏托板或外力搬运替代抓放。事件期允许的力必须在机器人操作前撤除，并保留实测证据。
- 判据、timestep、预算、joint limits、稳定时长和角色保持不可为凑通过而放宽。每次修复有具体诊断，在新输出目录同 seed 重试，原失败、配置、原始状态和录像保留。
- 运行批次期间冻结所有被 snapshot 的源码与配置。代码改动应在 worker 全部退出后实施，并在新目录重试。
- 资源存在、CPU 通过、注册项的 implemented 标志、初始落稳、GT 正确、机器人动作通过和模型评估是不同结论。没有模型调用就标 `model_evaluated=false`。
- 每个结论绑定实际本体、场景、布局、配置、保存版本和证据。历史实例可以引用其原验收，不能静默标成最新版本重跑。

## 代码和验证

优先复用现有组装、绑定、执行和审计链。用 `rg` 查找源码，修改前保留用户现有未提交内容。只运行与改动相关的有效检查；源码/配置 CPU 检查不能替代目标机 GPU smoke。

提交前至少核对：入口可导入、相关回归通过、新配置可加载、资产 manifest 与引用闭包完整、敏感文件未进入 export。安装 preflight 不启动仿真；真实 GPU 单例必须单独执行并读原始报告。

## 发布与数据

源码仓库包括全部当前 first-party Python、配置、测试、脚本、文档、网页和必要轻量元数据，包括尚未跟踪但属于项目的模块。大型 USD/纹理/mesh 放独立资产包；SDK/CUDA 从固定官方版本安装；结果、录像、轨迹和 source snapshots 放独立证据包。

GitHub 发布前扫描即将发布的 clean export，报告只给文件位置和规则名，不打印密钥值。不要提交 `.env*`、AWS/SSH/Git 凭据、虚拟环境、checkpoint、Kit cache 或个人编辑器状态。不要在用户的 dirty 工作区执行 reset/clean 来制作发布包。

`migration/isaaclab/full_source.patch` 保留原工作站完整自定义源，可能包含实验脚本缺陷；默认安装使用已单独验证的 `minimal-runtime.patch`。两个补丁是替代选项，不能叠加应用。可选 cuRobo 的源码差异和资产另列，不能声称其 CUDA 扩展在新主机已验证。
