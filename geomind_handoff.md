# GEOMIND skill 接入资料

采集日期：2026-11-06（Asia/Shanghai）。依据：本机工作区源码、配置、项目文档及少量已有结果摘要。本文件是接入说明，不是新 skill 实现。

本次只新增本文件；未修改项目源码或现有配置，未启动 Isaac Sim、训练、模型服务或大规模评估。轻量运行产物位于 `/tmp/geomind_handoff_*_20261106`。未读取或导出密钥、令牌、`.env`、私有数据集、模型权重；未展开仓库中的 ZIP，也未读取轨迹数据或视频。

文中状态含义：**本次验证**＝实际运行的轻量命令；**静态确认**＝已检查函数/配置，但未执行相应仿真；**历史记录**＝磁盘已有结果，不代表当前源码重新通过；**未确认**＝缺少足够证据。

## 1. 项目概览

### 1.1 版本与规则

| 项目 | 本次读取结果 |
|---|---|
| 工作目录 | `/home/elm/projects/geomind_lab`；后续命令均从仓库根目录执行 |
| 当前分支 | `chore/portable-dev-20260924` |
| HEAD | `8ab25dd2b8ddbc3abdfe0017b45f9da971f4f5de` |
| 工作区 | 不干净；交接文件生成前：61 个已跟踪文件修改、56 个已跟踪文件删除、112 个未跟踪条目（Git 会折叠目录，不等于文件总数） |
| 重要影响 | 新增的 episode 源码、配置和测试大量尚未提交；另一台机器只 checkout HEAD 无法复现本工作区 |
| AGENTS.md | 仓库根及上级路径未发现；所检查的项目子目录亦未发现。未将依赖树 `IsaacLab/` 的规则用于项目代码 |

证据：`git branch --show-current`、`git rev-parse HEAD`、`git status --porcelain -z`、`git diff --stat`。本文件创建后未跟踪条目会增加 1。

项目规则和说明入口：`README.md`、`GEOMIND_PROTOCOL.md`、`DEVELOPMENT_CONTEXT.md`、`docs/development_checklist_zh.md`。后两者和工程协议包含历史架构或规划，不能覆盖当前实现事实。例如工程协议描述桌面 XY MVP，当前 `geomind/episodes/resolver.py:solve` 已支持无限三维直线任务。v3 接口优先参照 `docs/episode_contract_v3_zh.md`、`docs/registered_scene_episodes_zh.md` 和源码。

### 1.2 目录与依赖

| 路径 | 职责及关键入口 |
|---|---|
| `geomind/core/` | 注册、ID/schema 归一化、配置解析、环境工厂；`GeoMindRegistry`、`resolve_task_request`、`create_env` |
| `geomind/episodes/` | 原生 v3 契约、状态证据、语义 GT、CPU/PhysX 后端及录像；`EpisodeResolver`、`SemanticEpisode`、`IsaacLabEpisode` |
| `geomind/configs/` | `task_definitions/` 定义任务；`tasks/`、`vla_tasks/` 是注册；`episodes/` 是原生实例；`scenes/`、`layouts/`、`protocols/`、`suites/` 分离复用；`legacy/` 保存兼容执行记录；`reports/`、`evidence/` 保存生成验证结果 |
| `geomind/tasks/` | family context builder、evaluator、变体；builder 主要提供元数据，不是完整任务实现 |
| `geomind/runners/`、`geomind/tools/` | 单任务/套件/策略运行，以及生成、解析、验证、打包等 CLI |
| `geomind_lab/` | IsaacLab 业务包：bootstrap、场景/任务配置、Franka/Mimic 环境、协议逻辑、资产工具 |
| `assets/` | 项目拥有的标注 JSON、场景/资产目录元数据；外部 USD/MDL 不保证包含在 Git 中 |
| `configs/` | benchmark manifest，例如 `geomind_benchmark_main.json`、`geomind_benchmark_phase1.json` |
| `scripts/`、`geomind/scripts/` | Shell/Python 启动与分析入口；前者包含 benchmark 和 VLA 批量实验，不能与后者的轻量 suite 混为一谈 |
| `tests/` | CPU 单测与部分仿真验证脚本；脚本名称中出现 smoke 不代表不启动 GPU |
| `IsaacLab/`、`migration/` | 外部框架 checkout；迁移说明与兼容补丁 |
| `outputs/`、`logs/`、`run_eval_*`、`analysis_outputs/` | 本地结果；不作为另一台机器必然具有的输入 |
| `scene_construction ` | 历史离线场景工具；目录名末尾有空格 |

依赖文件：`pyproject.toml`（Python >=3.10，基础依赖 numpy >=1.26,<2、gymnasium >=1,<2；dev/trajectory/analysis/viewer extras）、`environment.isaac_curobo.yml`（环境配方，不是完整 lock）、`MANIFEST.in`（打包范围）。本次未安装依赖。

README 入口：`README.md`、`scripts/README.md`、`geomind/scripts/README.md`、`assets/README.md`、`migration/isaaclab/README.md`、两个 family runner 目录下的 `README.md`。迁移指南：`docs/migration.md`。

## 2. 任务开发链路

### 2.1 注册与任务清单

1. `geomind/core/registry.py:GeoMindRegistry.load` 扫描 `geomind/configs/tasks/*.json` 和 `vla_tasks/*.json`，调用 `normalize_task_config`，检查重复 identity 并建立别名。它会跳过 Deprecated/RelationVariant 和特定旧方向任务记录，并非读取每个 JSON 都注册。
2. `geomind/core/schema.py:normalize_task_config` 与 `geomind/core/episode_compat.py:legacy_definition` 将紧凑 v3 注册接到 `configs/legacy/registrations/`；注册配置与原生 `episodes/` 是两种用途。
3. `GeoMindRegistry.resolve_task_id` 支持别名及 condition/layout/version/interface/task_variant/parameters 选择；公共裸名优先 Standard、Base 和最新版本，存在歧义会报错。自动化应保存最终 resolved ID。
4. `GeoMindRegistry.register_gym_envs` 注册统一入口 `geomind.core.envs.task_env:GeoMindTaskEnv`。原生 episode 另外由 `geomind/core/loader.py:create_env` 分流到 `create_episode_env`。
5. 当前注册清单：`registry.list_task_ids(include_placeholders=True)` 本次返回 59 项，排除 placeholder 为 48 项；这是注册实例数，不是独立科学任务数或物理已验证数。

其他清单：`configs/current_task_mapping.json`、`planned_task_mapping.json`（均位于 `geomind/` 下）、`geomind/configs/vla_tasks/task_cards.yaml`、`geomind/configs/presets/*.json`、根目录 benchmark manifests。旧 canonical 语义和版本能力在 `geomind_lab/mimic/utils/task_metadata_registry.py:build_task_metadata`、`supported_task_versions`；原生语义任务白名单在 `geomind/episodes/resolver.py:TASKS`，物理白名单在 `geomind/episodes/physical_contract.py:PHYSICS_TASKS`。

**能力判断用 `registry.capabilities(name)`，不能只用 placeholder 字段。** 本次查询：`ProjectOntoLine`、`DetectSceneChange` 的 semantic 和 isaaclab_episode 均为 true，但旧执行注册仍为 placeholder；`CompleteSquare` 则是 semantic=false、isaaclab_legacy=true。这里 true 是实现声明，不代表本次跑过物理仿真。

### 2.2 代表任务：ProjectOntoLine（原生 v3，推荐新 skill 首先接入）

选择理由：已有公共注册、独立配置、样例生成、GT、答案判定、物理执行和历史结果，能展示当前完整接口边界。

| 环节 | 文件与关键符号 | 实际接口/行为 |
|---|---|---|
| 任务定义 | `geomind/configs/task_definitions/ProjectOntoLine.json`；`resolver.py:TASKS` | SpatialReasoning / GeometricInference / ReportPoint |
| 注册 | `geomind/configs/tasks/ProjectOntoLine_v0.json`；`GeoMindRegistry.load/capabilities` | 与旧兼容注册的 placeholder 状态并存，原生能力另行判断 |
| 实例配置 | `geomind/configs/episodes/ProjectOntoLine_v0.json` | seed=0，scene=`registered_scene_65`，layout=`geometry_s0`，点和两端点作为 parameters 引用；release_version=2.0 |
| 校验/解析 | `geomind/episodes/resolver.py:EpisodeResolver.resolve`、`_validate_scene_layout`、`_validate_parameters` | 检查坐标、引用、几何退化及字段；输出 `ResolvedEpisode` 和契约哈希 |
| 实现与基类 | `geomind/episodes/runtime.py:SemanticEpisode`；`geomind/episodes/isaaclab_physics.py:IsaacLabEpisode(SemanticEpisode)` | 语义环境本身不是 Gym Env 子类；物理后端复用协议与答案语义，不为每个任务定义新 Env 类 |
| 场景/布局生成 | `geomind/tools/generate_spatial_suite.py:generate`；`geomind/episodes/registered_scene.py:scene_description/spawn_registered_scene` | 按 seed 构造三维直线、点和正交偏移；默认程序化桌面，可用 `--scene-id 65` 复用注册场景；catalog 模式另取资产描述 |
| 状态 | `SemanticEpisode.reset/snapshot`、`IsaacLabEpisode.snapshot/_sync_positions` | CPU 从布局及 seed jitter 得状态；PhysX 从实测物体位置得状态；GT 不直接当作 layout 字段读取 |
| 指令与 GT | `geomind/episodes/resolver.py:solve` 的 ProjectOntoLine 分支 | 同一份经校验的状态生成英文指令及 Point GT。设 d=b-a，t=((p-a)·d)/(d·d)，q=a+t*d；不把 t 截断到线段。frame=table，单位米 |
| 动作/答案接口 | `SemanticEpisode.step(answer)`、`evaluate_answer` | `{"position_m":[x,y,z],"frame":"table"}`；返回 `(obs,reward,terminated,truncated,info)`；该任务 reset 后即可回答，一次回答即结束 |
| CPU 成功判定 | `runtime.py:evaluate_answer` | 距离 <= episode.evaluation.position_tolerance_m，当前 1e-6 m；错 frame、非有限数、错误字段是契约异常，不是普通失败答案 |
| 物理动作 | `IsaacLabEpisode.step/execute_answer`；`physical_contract.py:REACH_ABOVE_TARGET/validate_execution` | 接受语义答案后用 Franka IK 和关节位置执行器到答案上方 0.1 m；不是抓取放置训练，也不是任意 VLA action chunk 接口 |
| 物理成功判定 | `IsaacLabEpisode.execute_answer` | 当前位置误差 <=0.012 m、朝向误差 <=0.15 rad，末尾连续 dwell>=20；运动上限 1800 步。总 success=semantic.success AND physical.success |
| 验证入口 | `generate_spatial_suite.py:verify`；`run_episode_physics.py:semantic_check/run_case` | CPU 独立构造 oracle、错误答案拒绝、证据重放、端到端；物理再检验 GT 几何条件和机器人运动，可选负对照 |

Agent 只接收 `SemanticEpisode.observation` 的 instruction、phase、step、scene_visible、objects、frame、allow_response。`result`、GT、`export_evidence`、历史 snapshot 是评估端特权数据，skill 不应将其喂给模型。记忆任务的不可见阶段返回空 objects；参见同文件 `observation/step` 和 `DetectSceneChange` 协议。

### 2.3 兼容任务补充：CompleteSquare

大量现有任务仍走旧机器人链路，不能用原生答案接口直接替换：

- 配置：`geomind/configs/tasks/CompleteSquare_v0.json` → `task_definitions/CompleteSquare.json`、`scenes/scene_65.json`、`layouts/geometric_inference_square_completion_corner_3.json` 和 `legacy/registrations/CompleteSquare_v0.json`；`episode_compat.py:legacy_definition` 明确隔离兼容执行。
- 任务语义：`geomind_lab/mimic/utils/task_metadata_registry.py` 的 `square_completion` 条目；标注入口 `assets/annotations/annotated_tasks.json`，由 `task_specs.py:resolve_task_payload/load_task_spec/resolve_task_instruction` 解析。场景由 `geomind_scene_cfg.py:get_scene_profile/GeoMindSceneCfg.apply_scene` 装载。
- GT：`geomind_lab/mimic/envs/geometric_inference_logic.py:solve_square_completion_gt` 使用相邻角 B+C-A 得缺角位置，并计算中心与向心 yaw；`build_geometric_inference_problem_from_object_positions` 可根据运行时位置构造问题。
- 环境：`geomind/core/envs/task_env.py:GeoMindTaskEnv(gym.Env)` 包装 mock 或 native delegate；`isaaclab_native_delegate.py:GeoMindIsaacLabNativeDelegate(gym.Env)` 继承原生 action_space，`step` 转交原生动作；Mimic 环境为 `geomind_lab/mimic/envs/geomind_franka_ik_rel_mimic_env.py:GeoMindFrankaIKRelMimicEnv(ManagerBasedRLMimicEnv)`。动作维数/缩放依具体 cfg，不能跨任务硬编码。
- `geomind/tasks/builders/geometric.py:GeometricBuilder(BaseFamilyBuilder)` 仅建 family context，不承担场景、GT 或动作实现。实际 `CompleteSquare` 的内部 family 可归一到 `shape_completion`，勿只按文件名推断分派。
- `geomind/tasks/evaluators/__init__.py` 注册 `square_completion_evaluator`；`simple.py:LayoutAwareEvaluator.evaluate` 用 goal_kind/resolver/有效距离及连续停留判定，默认 dwell=5，可被 `GEOMIND_SUCCESS_DWELL_STEPS` 覆盖。协议 debug/readout 的 `evaluate_square_completion_readout` 是另一个口径，不能混报。
- `policy_runtime.py:run_episode` 对 IsaacLab 的 oracle/oracle_protocol 可转交 family 协议 Shell runner；不保证经过普通逐步 policy loop。

### 2.4 新增任务通常改哪些文件

**已有原生语义的新实例**：新增/复用 `configs/task_definitions/`、`scenes/`、`layouts/`、记忆任务的 `protocols/`，新增 `episodes/<episode_id>.json` 并加入 `suites/`。`geomind/tools/scaffold_task.py:main` 默认创建原生 v3 episode，先临时校验再发布且拒绝覆盖；参数通过 `--parameters-file`、seed 通过 `--seed`，ReportPoint 需显式容差。仅新增 episode 不等于新增公共 registry 项。

**全新语义**：还要修改 `episodes/resolver.py:TASKS`、参数/协议校验、`solve`，必要时修改 `runtime.py` 的阶段/答案处理、`tools/generate_spatial_suite.py` 的生成与独立 oracle，并增加对应测试。若要物理能力，还需扩展 `physical_contract.py:PHYSICS_TASKS` 和执行验证，以及 `isaaclab_physics.py` 的动作实现；不能只添加白名单便宣称支持。物理工具当前 `semantic_check` 专门处理三个任务，新增第四种也需适配。

**旧执行链的新任务**：通常涉及 `task_metadata_registry.py`、标注 JSON、`task_specs.py`、相应 `mimic/envs/*_logic.py`、环境/场景 cfg、`tasks/evaluators/`，再补 compact 注册、legacy 绑定、VLA 版本和 suite/benchmark manifest。新 family 还要检查 `geomind_lab/mimic/utils/benchmark_manifest_utils.py:registered_runner_families/runner_family_config` 和 `policy_runtime.py:_resolve_protocol_runner_script_for_family`。`scaffold_task --legacy` 只是兼容配置工具，不会自动完成这些实现。

建议先读 `tests/test_spatial_episode_pipeline.py`、`test_episode_contract_v3.py`、`test_episode_physics_contract.py`、`test_public_task_ids.py`；根据选定链路添加最少且有意义的接口/边界验证。上述修改清单是静态接口判断，本次未新增任务。

## 3. 运行链路

### 3.1 环境准备

所有相对路径命令均在仓库根运行。CPU 开发文档命令：

```bash
conda activate isaac_curobo
python -m pip install -e '.[dev]'
python scripts/check_development_setup.py
```

这组安装/预检命令**本次未执行**。`docs/migration.md` 的历史基线是 Python 3.10.19 / Isaac Sim 4.5 / IsaacLab 0.51.1 / PyTorch 2.7.0+cu128；README 的原生物理示例又使用 `new_isaaclab`。环境名不是完整环境定义，另一台机器应核对实际 import 路径、GPU 驱动与兼容补丁，不能只复制 `.venv`。

GPU 运行还需 `IsaacLab/source`、外部 USD/MDL 和可能的 cuRobo；参见 `docs/migration.md`、`migration/isaaclab/README.md`。配置变量名称包括 `GEOMIND_ASSETS_ROOT`、`GEOMIND_LAB_ASSET_CATALOG_PATH`、`MDL_SYSTEM_PATH`，这里只记录接口名，不导出本机取值。远程策略还需模型服务及 `GEOMIND_MODEL_POLICY_HOOK`；本次未连接服务。

### 3.2 本次实际验证过的命令

本次解释器 `/home/elm/miniconda3/bin/python`，Python 3.13.9，numpy 1.26.4、gymnasium 1.2.3；它不是上述 GPU 基线。该解释器没有 pytest，因此用 unittest 运行选定的 CPU 测试，未安装包或跑全量测试。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_spatial_episode_pipeline.py -v

PYTHONDONTWRITEBYTECODE=1 python -m geomind.tools.generate_spatial_suite \
  --root /tmp/geomind_handoff_semantic_20261106 --seeds 0 --verify

PYTHONDONTWRITEBYTECODE=1 python -m geomind.tools.resolve_episode ProjectOntoLine_v0 \
  --root /tmp/geomind_handoff_semantic_20261106 \
  --state /tmp/geomind_handoff_semantic_20261106/evidence/ProjectOntoLine_v0.json

PYTHONDONTWRITEBYTECODE=1 python -m geomind.runners.run_task \
  --task-id CompleteSquare --backend mock --max-steps 8 \
  --output-dir /tmp/geomind_handoff_mock_20261106
```

| 命令 | 输入、输出与结果 |
|---|---|
| 选定 unittest | 22 tests / OK；测试内部用临时配置运行，无 GPU |
| generate+verify | 输入当前源码和公共注册，生成 1 个记忆任务、2 个推理任务；输出 root 下 task_definitions/scenes/layouts/protocols/episodes/suites/evidence/reports；passed=true，verified_cases=3，physical_execution_verified=false |
| resolve+state | 读取上述临时配置和 evidence，重放 GT 成功，evidence_backend=semantic；不带 `--state` 仅校验配置，不计算 GT |
| mock 单任务 | 输入当前 CompleteSquare 注册/标注；输出 `result.json`、`trace.json`；退出 0，8 步，done_reason=terminated，metric_valid=true，success=false，dwell_satisfied=false。退出码只证明执行完成，不代表任务成功 |

生成器默认 `--root` 是仓库内 `geomind/configs`，会写配置和报告。自动化实验必须显式传独立 root，避免一次样例验证重写工作区。本次没有执行 README 中省略 `--root` 的生成命令。

### 3.3 仅静态确认、未执行的命令

```bash
# CPU/mock 批量：普通策略运行链，不是原生答案环境
python -m geomind.runners.run_suite --task-ids CompleteSquare \
  --backend mock --policy oracle --episodes-per-task 1 --max-steps 8 \
  --seed 0 --no-save-video --output-dir outputs/skill_mock_suite

# 原生物理单例；GPU/资产准备后另行运行
python -m geomind.tools.run_episode_physics \
  --root geomind/configs --episode ProjectOntoLine_v0 \
  --timeout-s 600 --output outputs/skill_physics_one

# 原生物理 suite，含三个任务类型的负对照；比单例昂贵
python -m geomind.tools.run_episode_physics \
  --root geomind/configs --suite generated_spatial_suite \
  --negative-controls --timeout-s 600 --output outputs/skill_physics_suite

# 真实 benchmark smoke：两类任务、每类一个 agent、各一个 episode，会启动仿真
bash scripts/run_geomind_smoke.sh --output logs/skill_real_smoke

# manifest 驱动的 family benchmark；会启动多个作业
bash scripts/run_geomind_benchmark.sh configs/geomind_benchmark_phase1.json logs/skill_phase1

# VLA 批量入口，仅展示调度，不运行模型/仿真
python scripts/run_vla_eval.py --tasks CompleteSquare --runs-per-level 1 \
  --base-seed 0 --output-root outputs/skill_vla_plan --dry-run
```

证据分别是 `run_suite.py:main`、`run_episode_physics.py:main/supervise`、`scripts/run_geomind_smoke.sh`、`run_geomind_benchmark.sh:prepare_manifest_selection` 和 `run_vla_eval.py:parse_args/run_trial`。最后一条仍可能写 eval_config/trial_metadata/汇总，且依赖模型配置解析；dry-run 不等于只读。执行级 VLA 命令需在修复下述接口故障后，再补模型选择与已验证服务配置。

`run_suite` 结果是 `<output>/<task>/episode_NNN/{result,trace}.json`、task_summary.json 和根 summary.json；默认输出在 `geomind/outputs/suite/<timestamp>`。`run_task` 默认在 `geomind/outputs/single/`，CLI 没有 seed 参数。原生物理输出是 `report.json`、`cases/`、`evidence/`、`traces/*.npz`，可选 `videos/`，默认 `outputs/headless_physics`；manifest benchmark 默认 `logs/geomind_benchmark/<timestamp>`。这些路径不能互换。

## 4. 实验与结果

### 4.1 找到的已有配置与对应结果

主例：`geomind/configs/suites/generated_spatial_suite.json`（seeds=[0,1,2]，9 cases）与 `geomind/configs/reports/generated_spatial_suite.json`（verified_cases=9，passed=true）。本次检查二者 episode ID 集合相同，generator_hash 相同：

```text
e44623faf5967599d25579d76a47215ec99aa3cac07ade51cd6c4833ccff9c25
```

单例配置为 `geomind/configs/episodes/ProjectOntoLine_v0.json`，关联 `layouts/geometry_s0.json`、`scenes/registered_scene_65.json`；证据文件在 `geomind/configs/evidence/`。这是可识别的历史配置/报告配对，**不是当前源码重跑证明**：本次生成器哈希为 `61d1ebd40b26c095452bb1a19222cccf298fad987fb5da6dc6a9287f5c26b550`，且已有报告该单例 provenance.contract_hash 与当前解析值不一致。

物理历史结果：`outputs/registered_scene_video/report.json`，9 个 positive、9 个 success、无负对照，physical_execution_verified=true；`outputs/headless_physics/report.json` 则为 9 positive + 3 negative controls，验证通过。本次只读取报告摘要和少量案例字段，未读取轨迹/视频。两者不是同一次实验；不能把其负对照和视频数量拼成一次结果。

`outputs/registered_scene_video/report.json` 对应单例契约哈希也与当前配置/实现不一致；只证明磁盘留有历史成功证据，完整原始配置快照、当时 Git commit 和精确启动命令未能确认，不可宣称历史物理结果绑定当前工作区。

### 4.2 字段与指标口径

| 链路 | 主要字段/计算来源 |
|---|---|
| 原生 CPU 报告 | `generate_spatial_suite.py:verify`：configuration_valid、独立 oracle_error_m、incorrect_answer_rejected、evidence_replay_equal、end_to_end_success、instruction、ground_truth、provenance；passed 需要所有断言通过 |
| 原生语义答案 | `runtime.py:evaluate_answer`：Object 精确 ID 相等；Point 为欧氏距离 <=显式容差，输出 distance_m/tolerance_m/score |
| 原生物理 | `isaaclab_physics.py:execute_answer`：position_error_m、orientation_error_rad、dwell_steps、motion_steps 等；`run_episode_physics.py:run_case` 保存 semantic/physical/independent_gt_check、预期成功与 verification_passed |
| 物理整体验证 | 无 failure、全部正例完成且所有正负例 verification_passed 才通过；负例 success=false 且语义和物理都失败才是预期通过。不是简单统计所有 success=true 的比例 |
| 普通 runner | `policy_runtime.py:run_episode/save_episode_payload`：result 包括配置绑定、runtime_mode、max_steps、episode_steps、reward_sum、done_reason、evaluation 和 success；trace 为逐步 action/agent_obs/reward/terminated/truncated/meta |
| LayoutAwareEvaluator | success 检查最终连续窗口；reach_success 检查历史任一连续窗口；score=success_radius-goal_distance；metric_valid=false 不可当作成功。各向异性 goal_distance 可能不是普通米制距离，物理距离诊断应读 goal_distance_l2_m |
| 普通 suite | `run_suite.py:main`：success_count/episode_count，reach_count 同理。其计数从 episode summary 的顶层字段取值；普通 runtime 的 reach_success 在 evaluation 内，故顶层 reach_rate 不宜直接作为可靠指标 |
| VLA 分级汇总 | `scripts/run_vla_eval.py:classify_success/threshold_for_task_level/summarize`：优先使用距离与 task/level 阈值，否则 raw_success；loose/medium/strict 的 success_rate 分母仅 completed_trials，failed 与 not_run 另列；没有完成项时为 null |

VLA 的距离分级与 evaluator 的 dwell success 不是同一口径；即使普通 evaluator success=false，分级距离也可能判为成功。必须保存 method、threshold、raw_success 和 level_success 后再比较。

### 4.3 失败、超时与恢复语义

- 原生语义：`ContractError` 表示配置非法、歧义或证据不足；`resolve_episode.py:main` 退出码 2。生成验证中的断言失败会中止，不保证输出一份完整失败汇总；skill 需自行记录进程退出和错误类型。
- 普通 runtime：`run_episode` 记录 success/terminated/truncated/max_steps 等 done_reason，finally 关闭 env；未见通用墙钟超时，异常可能导致没有 result.json。不可将“文件缺失”默认为模型任务失败。
- 原生物理：`run_episode_physics.py:main` 捕获执行异常并先写 report 再关 Kit；`supervise` 用独立进程组和 `--timeout-s` 限时。无本次 run_token 对应报告则写失败并返回 1；报告完成后 shutdown 卡住，可在 10 秒清理期后 SIGKILL。
- 历史物理报告的 `shutdown=bounded_cleanup_after_report`、`worker_exit_code=-9` 与 `physical_execution_verified=true` 并存，是监督器允许的报告后清理，不应仅凭 -9 丢弃成功证据；报告前超时则不同。
- VLA `run_trial`：子进程非零或 result 缺失标记 failed，`--stop-on-error` 可中断；当前 subprocess.run 未设 timeout。默认继续时不要只看总脚本退出码，应检查 failed_trials。resume 只认 result 文件存在，不是模拟器状态 checkpoint。

### 4.4 可复现性记录

| 项目 | 当前实现与缺口 |
|---|---|
| seed | 原生 episode 必填非负 seed；生成器用 random.Random(seed)，suite 记录 seeds。普通 suite 使用 seed+episode_index，但 `run_episode` 的普通 summary 未直接保存 seed；VLA 的每 trial 显式保存 seed 和索引 |
| VLA seed 调度 | `run_vla_eval.py:derived_seed`：base_seed + task_index*100003 + level_index*1009 + trial_index；task_index 来自完整未筛选列表，另存 task_list_fingerprint。改变完整目录/顺序可能改变 seed |
| checkpoint | 本次未读取任何权重。已检查运行结果没有统一 checkpoint 路径/哈希保证；VLA 记录 model 名称与配置不等于证明服务加载的权重版本，需服务侧提供不可变 revision |
| 数据划分 | registry 中的 split=base/aug 和 condition/variant 是任务或增强轴，未确认存在统一 train/val/test 数据 manifest；原生生成 suite 只列 seeds/episodes，不能把它称作训练/测试划分 |
| 代码/配置版本 | 原生 provenance 包含 contract_hash、source_hashes、实现哈希、证据哈希和 instance_id；resolver 的 IMPLEMENTATION_HASH 对 episodes/*.py 计算。不是整个仓库 Git commit；generator_hash 只覆盖生成器源文件 |
| Git/环境 | 所检查的原生/VLA输出写入逻辑未自动保存完整 commit+dirty diff。VLA eval_config.json 记录 args、模型配置、阈值策略、seed schedule、tasks；还需 skill 额外保存 Git commit、dirty 状态/源码快照标识、Python/框架/资产版本及权重 revision |

### 4.5 少量脱敏样例

以下为实际结果字段的裁剪；未保留机器地址、服务地址、绝对资产路径或 run_token。

历史语义案例（已有报告）：

```json
{"episode_id":"ProjectOntoLine_v0","configuration_valid":true,"oracle_error_m":2.2551405187698492e-17,"end_to_end_success":true}
```

历史物理案例（`outputs/registered_scene_video/report.json`）：

```json
{"episode_id":"ProjectOntoLine_v0","success":true,"semantic":{"distance_m":0.0,"tolerance_m":1e-6},"physical":{"position_error_m":0.006225705612450838,"orientation_error_rad":0.023920940850729943,"dwell_steps":20,"motion_steps":992}}
```

本次 stdout/测试日志摘录（非历史 GPU 日志）：

```text
Ran 22 tests in 0.343s
OK
{"backend":"semantic","physical_execution_verified":false,"verified_cases":3,"passed":true}
mock: {"episode_steps":8,"done_reason":"terminated","success":false,"metric_valid":true,"dwell_satisfied":false}
```

本次新结果位于 `/tmp/geomind_handoff_semantic_20261106/reports/generated_spatial_suite.json`、`/tmp/geomind_handoff_mock_20261106/result.json`；临时目录不保证跨机器或重启保留。未将原始日志整份导出。

## 5. 自动化接口与已知限制

### 5.1 能力表

| 能力 | 状态与依据 |
|---|---|
| 批量任务 | 支持：原生生成器 TASK_IDS+seeds、物理 `--suite`，普通 `run_suite --task-ids/--group`，manifest benchmark，以及 VLA task×level×trial 循环；这些循环大多串行，不等于 GPU 并行 |
| 指定 seed | 支持：原生 episode/scaffold、生成器 --seeds、普通 run_suite --seed、VLA --base-seed；单任务 run_task CLI 未提供 --seed，直接 runtime API 有 seed 参数 |
| 独立输出 | 支持：生成器 --root、物理 --output、普通 --output-dir、VLA --output-root、benchmark 输出参数；生成配置 root 与运行输出目录是不同概念 |
| 断点恢复 | VLA 支持 --resume，按 result.json 跳过并重新分类；不恢复进行中的 episode/机器人状态，不校验旧结果与当前配置或权重一致 |
| 跳过已完成 | VLA 同上，并传 `VLA_EVAL_SKIP_IF_RESULT_JSON` 给 Shell 防重复；其余所检查入口未见统一 skip/resume，实现上会重新循环/写同名文件 |
| 仅汇总 | VLA --summarize-only 会重读结果、重新分类并重写 JSONL/汇总，不是只读；基于当前阈值，需保存旧版本以便审计 |
| 作业方式 | `python -m`、直接 Python、Bash、subprocess；物理工具有独立进程组监督。对 scripts/ 和 geomind/scripts/ 的 .sh/.py 搜索未发现 sbatch/srun/slurm 调度入口；未确认仓库外是否另有调度系统 |
| 训练自动化 | 本次未确认统一训练入口、训练 checkpoint 管理或训练续跑能力，不能从实验评估工具推导出来 |

### 5.2 已知故障和人工步骤

1. **当前源码确有参数接口不匹配**：`geomind/runners/run_env_policy.py:main` 向 `policy_runtime.py:run_episode` 传 ablation_id、pair_id，而后者签名没有这两个参数；静态确认会形成 unexpected keyword 风险。`docs/migration.md` 也注明。VLA Shell 链经此入口时需先修复/验证，本次遵照只读要求未修复。
2. **策略 hook 可回退 oracle**：`policy_runtime.py:build_policy` 在 model/agent hook 不可用时返回 oracle 动作。自动化必须先验证 hook 实际加载及模型身份，不能只读 policy_mode=model 就声称模型评估完成。
3. **mock 成功与停留判定边界**：本次距离 0.028018690645694733、半径 0.05、dwell 要求 5，但 env 在窗口满足前终止，success=false；只应报告链路能执行。参见 `MockBackendEnv.step`、`LayoutAwareEvaluator.evaluate`。
4. **历史结果漂移**：已有语义/物理报告契约哈希与当前源码不一致；现有工作区又有大量未提交文件。迁移时应明确打包/提交哪些授权源码，不能交付整个工作目录压缩包，更不能顺带复制数据、凭据和权重。本次没有进行打包或提交。
5. **环境与外部资产须人工就绪**：另一台机器需准备兼容驱动、Isaac Sim/IsaacLab、必要补丁、资产/材质和远程服务。GPU、USD 加载、模型服务与新机器路径本次均未验证。
6. **历史全量测试失败记录不能当作当前结论**：`docs/migration.md` 记录 2026-09-24 的 120 passed/7 failed，涉及旧 VLA layout 预期和英文指令；本次只跑所列 22 项，未复验全量失败是否仍存在。
7. **接口分流和覆盖行为**：`create_env` 对原生 semantic/physics 只允许有限 overrides，参数必须落 episode；不能把普通 runner 的 max_steps/backend_kwargs 或 VLA action API 直接套入。物理固定输出名和普通 result 写入会覆盖旧记录，应每次使用唯一目录。
8. **恢复/并发的边界**：VLA resume 以文件存在为准，可能复用不相容结果；main 每轮重写累计 trial_results.jsonl，筛选任务重跑也可能缩小原汇总。未见多写者锁，多个作业不得共享输出根。

## 6. 交接结论与最小闭环

### 已确认可用的能力

- 本机 CPU 可加载当前 registry，解析公共任务/别名及查询后端能力。
- seed=0 的三个原生任务能在独立临时配置根生成、校验、拒绝错误答案并端到端评分；22 项相关测试通过。
- 导出证据能由 CLI 重放出相同 GT；接口明确区分 agent observation 与评估特权数据。
- CompleteSquare mock 可执行并输出 result/trace；本次任务成功判定为 false。
- 源码提供物理监督、批量实验、输出隔离和 VLA 文件级恢复接口；这些执行能力本次仅静态确认。

### 未验证的信息

- 新机器依赖安装、GPU/PhysX、资产/材质路径、cuRobo、模型服务及 checkpoint 身份。
- 当前工作区的全量测试、全部注册任务的物理完成率、真实 VLA 链路、长期批量稳定性。
- 历史实验的完整源码/配置快照、精确启动命令、统一 train/val/test 划分、完整模型权重 revision。
- Slurm 或仓库外调度，以及训练/训练断点恢复。

### 建议优先打通的最小实验闭环

1. 固定待迁移源码版本和工作区快照标识；记录 Git commit/dirty、Python/依赖版本，排除环境、凭据、私有数据和权重。skill 的输入先限定为 backend、episode_id、config_root、seed、唯一 output_root 和超时。
2. 从本节已验证的 CPU 命令开始：独立 root、seed=0、3 cases；以 report.passed、错误答案拒绝及证据重放共同验收。保存配置哈希和简短 stdout，结果明确标注 semantic，不算物理成功。
3. 为单个 ProjectOntoLine 原生物理 episode 准备 GPU/资产，用监督器和独立输出限时运行；核对 semantic、physical、verification_passed、shutdown 和报告新鲜度，再增加一个负对照。ReachAboveTarget 成功只代表到达任务，不代表抓取放置或模型能力。
4. 需要模型实验时，先处理 run_env_policy 参数不匹配和 hook 身份验证，再跑一个任务、一个 seed；保留 raw_success/level_success/metric_valid/失败类型，最后扩展批量与文件级 resume。恢复前比较配置、代码、阈值与模型 revision，不能仅看 result.json 是否存在。

建议 skill 的首个交付目标是“生成 → 校验 → 一次有限运行 → 证据落盘 → 指标与失败原因汇总 → 可复现重放”。先完成 CPU 闭环，再独立验收物理与远程模型链路。

## 7. 研究设计与验证证据（2026-11-07 补充）

本节不重复工程接入信息。证据分成三层：**实现正确性**（程序按约定生成/评分）、**能力有效性**（成功必须依赖所声称的能力）、**模型表现**（实际受评模型在规定信息条件下的成绩）。第一层通过，不能自动推出后两层。

本轮仅静态阅读、查询注册能力、用当前配置重建两个 CPU episode 的 observation，以及对已有关系配置做少量坐标计算；额外 reset 一次 mock 以核对观测。没有运行受评模型、PhysX 或新批量实验。下文所有补充实验均为**建议、未执行**；2026-11-06 的测试和磁盘历史报告沿用上一节的时间与版本限制。本节 observation 中的坐标/物体名称来自项目任务配置，不是私有数据；不包含机器路径、令牌或资产文件内容。

### 7.1 文档设计的时间层次与冲突

| 文档/位置 | 应如何理解 | 与源码对照后的结论 |
|---|---|---|
| [Canonical 分类](docs/canonical_task_families_zh.md) 的 27 任务/6 family | 当前公共分类设计，覆盖已实现和占位项 | 不能把 27 当作已实现或已验证数量；`SelectByTwoRelations`、`SelectByIndirectReference` 均是 Unavailable，原生 `TASKS` 仅覆盖其中一部分 |
| [v3 契约](docs/episode_contract_v3_zh.md) 的参数、阶段、GT 边界 | 当前原生研究操作化定义 | 与 `EpisodeResolver`、`SemanticEpisode`、`solve/evaluate_answer` 基本一致；其后半“当前只支持 ProceduralTable”及后续接入顺序已落后于 RegisteredScene/catalog 实现 |
| [任务迁移状态](docs/task_evolution_zh.md) 的历史任务暂缓 | 当前对旧任务主张的收缩 | UpdateObjectOrder 按插入后的当前顺序求解、history_mode=Deferred，不能仅因归到 SpatialMemory 就报告历史记忆成绩；TrackObjectOrigin placeholder 不能算历史来源追踪已实现 |
| [工程协议](GEOMIND_PROTOCOL.md) §6、§7、§8 | v0.1 研究目标和 MVP 方案，含明确科学意图 | §6 本就把状态重索引定义为最终状态可解，明确“不需要 history ablation”；§7 同质外观、历史 middle、final_frame_only 是历史身份追踪方案，不是当前 DetectSceneChange 的实现描述；§8 XY 目标位推断/continuous_placement 与原生 InfiniteLine3D+ReachAboveTarget 不同 |
| [开发 Checklist](docs/development_checklist_zh.md) Phase 0–6 | 规划，未勾选的条目不是完成证据 | 统一指标、阶段成功、随机化、泛化、微调等验收尚不能据此声称已完成；不能反过来因未打勾就否认源码中已有局部实现 |
| [系统概述](geomind/docs/系统概述.md)“临时层”、[读者文档](geomind/docs/读者文档.md) | 混有历史状态的说明 | “native delegate 非真实物理 / ManagerBased 未接入”已与 `GeoMindIsaacLabNativeDelegate._build_native_env` 和 `IsaacLabEpisode` 不符；前者实际创建原生环境，后者有真实 PhysX 实现。不是全部内容都能视为现状 |
| [生成验证报告](docs/generated_tasks_verification_zh.md)、[物理报告](docs/headless_physics_verification_zh.md)、[原场景复用报告](docs/registered_scene_episodes_zh.md) | 各轮历史工程证据，场景/资产逐步变化 | 测试数 179/190/198/209、程序化桌面/注册场景/目录资产是不同轮次；不能拼成一个固定版本成绩。当前配置已是 catalog 物体，不能再把早轮彩色方块表述套到本节实例 |
| [返修解耦草稿](paper/rebuttal_disentangle_draft.md) 与 [指标汇总](paper/rebuttal_metrics.md) | 历史模型实验解释与内部警告，不是当前三任务的新结果 | 草稿正文“全部四种消融已跑完”“controller 不是瓶颈”强于内部说明：轨迹最接近点只是意图代理，direct 同时改语言与显著性，并非干净的 oracle-target 输入；IK planner 对照仍缺。dwell 离线重算、起始已命中分母、TTE 修正、语言通道失效等限制必须保留 |
| [论文附录说明](paper/README.md) 的 oracle 默认 100% | 作图用名义上限 | 不是实测 oracle 成绩，不可用于证明可解性或控制上界 |

这里“历史规划”不等于研究意图无效，而是不能把建议的控制实验、同质物体或理想 readout 当作已经落实。特别是稳定 ID、精确坐标、可区分 catalog 外观与 §7 的同质视觉身份追踪，测量条件不同。

### 7.2 三项选择与“关系组合”的范围

选择 ProjectOntoLine、DetectSceneChange，以及**已经实现的有限约束组合 SelectAfterExclusion**，后者组合属性谓词、相对排序排除和参照距离选择。源码事实是存在这套组合求解器；把它解释为一般关系组合是过度外推。

若“关系组合”特指 `R1(x,a) AND R2(x,b)` 的双参照关系交集，或通过中间实体的多跳指代，则当前查到的是**实现缺口**：[SelectByTwoRelations](geomind/configs/tasks/SelectByTwoRelations_v0.json)、[SelectByIndirectReference](geomind/configs/tasks/SelectByIndirectReference_v0.json) 都为 `execution=Unavailable`；其 legacy 绑定为 placeholder/unimplemented；`resolver.py:TASKS` 和 `physical_contract.py:PHYSICS_TASKS` 也未实现它们。本轮 `registry.capabilities` 返回 semantic/isaaclab 全 false。因此没有这两种任务的可执行 observation、实例 GT、负对照或模型成绩可提供；不可用示意场景填补“已有证据”。

SelectWithVisualCue 的实现是属性与单个方向区域相交（`constraint_selection_logic.py:resolve_visual_spatial_constraint_problem`），同样不能替代双空间关系任务。下文第三项完整剖析 SelectAfterExclusion，同时保留上述强主张缺口。

### 7.3 ProjectOntoLine

#### A. 研究主张：无限三维直线上的连续目标推断

- **文档原意**：[v3 契约](docs/episode_contract_v3_zh.md)“空间推理参数”定义三维正交投影，允许落在线段外；[分类](docs/canonical_task_families_zh.md) 将其列为 GeometricInference。[工程协议](GEOMIND_PROTOCOL.md) §8 的更早目标是固定身份、隔离抓取获取后的目标位推断，但没有单独论证为什么必须增加 ProjectOntoLine。
- **源码事实**：[resolver.py](geomind/episodes/resolver.py) 的 `solve` 用三个参考点计算投影，不预测对象身份，也不需要历史；当前 catalog 指令末尾明确参考点是 rigid-body origin，并非视觉中心。
- **研究解释/推断**：它可区分“沿已知线寻找最近点”与 InferMidpoint 的固定平均、ExtendEqualSpacing 的延长、ReflectAcrossLine 的轴对称点输出；答案通常不在任何对象上，因此值得独立考察连续几何构造。独立设立的原作者动机、目标认知理论和预期难度排序，文档未明确，需要研究者确认。

#### B. 模型实际可获得的信息

以下为本轮在当前 `ProjectOntoLine_v0` 配置上 `SemanticEpisode.reset()` 返回的**完整 observation**，没有省略 objects，也没有调用模型：

```json
{
  "instruction": "Report the orthogonal projection of the reference point of the yellow mustard bottle using the infinite 3D line through the reference points of the coffee can and the meat tin, in frame table, in metres. Each object's reference point is its configured rigid-body origin.",
  "phase": "query",
  "step": 0,
  "scene_visible": true,
  "objects": [
    {
      "object_id": "line_a",
      "label": "the coffee can",
      "position_m": [
        0.0467784617338528,
        -0.10526084107006703,
        0.17950980582075943
      ]
    },
    {
      "object_id": "line_b",
      "label": "the meat tin",
      "position_m": [
        -0.0626641455676838,
        0.05704419112865969,
        0.2204901941792406
      ]
    },
    {
      "object_id": "point",
      "label": "the yellow mustard bottle",
      "position_m": [
        -0.0679553238916608,
        -0.06031567934469169,
        0.2007392679365108
      ]
    }
  ],
  "frame": "table",
  "allow_response": true
}
```

由 [runtime.py:SemanticEpisode.observation](geomind/episodes/runtime.py) 可确认：无图像、无相机内外参、无历史帧；三个稳定 object_id、语义 label 和精确三维坐标全部公开；ID 甚至直接表明 line_a/line_b/point 的角色。frame 和查询时刻公开，不直接公开 seed、GT、契约哈希或生成器参数。

**CPU 信息条件测的是结构化坐标输入下的几何求解，不能报告为视觉几何理解。** [isaaclab_physics.py:IsaacLabEpisode.observation](geomind/episodes/isaaclab_physics.py) 仍公开上述对象坐标，另外增加 `robot.joint_positions_rad`、`robot.tcp_position_m`；并未在这个 API observation 中添加 RGB。录像由 [video.py](geomind/episodes/video.py) 与录制/打包工具另行提供，视频存在不证明受评模型曾收到图像。没有本轮实际物理/模型 observation 样本，不能补造机器人读数。

评估端的 `resolved`、`env.result`、snapshot、provenance 能访问全部状态和 GT。`generate_spatial_suite.verify` 使用构造答案作测试输入；`run_episode_physics.run_case` 直接读取 `env.result['ground_truth']` 后执行，所以 oracle 不需要从 observation 推理。只有明确限制为上述 observation 的受评模型才处于较弱的信息条件；Python hook 若持有完整 env 对象，隔离还依赖调用方约束。

#### C. 完整实例与求解

实例来源：[ProjectOntoLine_v0](geomind/configs/episodes/ProjectOntoLine_v0.json)、[geometry_s0](geomind/configs/layouts/geometry_s0.json)。上面 JSON 已含完整英文指令、初始场景及回答时观察；没有事件，reset 即 query/step=0。

令 a=咖啡罐参考点、b=肉罐参考点、p=芥末瓶参考点。预期求解：d=b-a，计算 `t=((p-a)·d)/(d·d)`，输出 q=a+td，不截断 t；本例 t≈0.5180395542。正确答案及本轮 CPU 评分：

```json
{
  "answer": {
    "position_m": [
      -0.009917137762001677,
      -0.021180414547081805,
      0.2007392679365108
    ],
    "frame": "table"
  },
  "evaluation": {
    "success": true,
    "distance_m": 0.0,
    "tolerance_m": 1e-06,
    "score": 1.0
  }
}
```

按 `evaluate_answer`，同 frame 下欧氏误差不超过 1e-6 m 成功。本例用 GT 回答得到 distance_m=0、score=1，这是评分闭环验证，**不是模型成绩**。物理层另以答案上方 0.10 m 的 TCP 到达、姿态和 dwell 判定，不把语义 1e-6 m 误差当作物理控制精度。

**更简单的替代解法（源码推出，未跑模型）**：生成器固定垂线向量 `[-sin(yaw),cos(yaw),0]`，所以所有这类 CPU 构造都有 `q_z=p_z`。当 `b_z != a_z`，只需令 `t_z=(p_z-a_z)/(b_z-a_z)` 并沿线插值，就能绕开一般三维正交条件。本例同样成立。这仍有几何计算，但证明不了对任意三维偏移的正交投影泛化；公开坐标与角色 ID 又完全绕开了视觉定位和语言角色绑定。物理漂移可能破坏等高关系，不能直接把这个捷径成功率外推到 PhysX。

#### D. 生成约束与未保证事项

依据：[generate_spatial_suite.py:generate](geomind/tools/generate_spatial_suite.py)、[EpisodeResolver._validate_parameters/_validate_scene_layout/solve](geomind/episodes/resolver.py)。

- 对象必须为三个不同 ID，端点不能共用 ID；实际 `||b-a|| > degeneracy_tolerance_m=1e-4`。因此到无限直线的欧氏投影唯一；点在线上仍有合法答案，不要求一定偏离直线。
- 当前生成器 d 由 yaw∈[-π,π]、pitch∈[-0.4,0.4] 构造，端点距离恒为 0.2 m；中心 x/y∈[-0.05,0.05]、z=0.2；线方向上的有符号距离 s∈[-0.16,0.16]，可超出端点±0.1；垂直偏移恒为 0.07 m，且仅在 XY 平面。
- 坐标有限、在工作空间内，label 唯一，jitter=0；resolver 对一般非法配置会拒绝，而非自动修补。
- 当前仅三个任务对象，没有角色竞争干扰项；未保证任意视角下可见、无互相遮挡、参考点能由图像估计、任意点都在机器人可达域。点与参数角色对应是通过标签/ID 给出的，不是从图像验证得出的。
- catalog 使用刚体原点。即使标签与资产一致，也不代表视觉中心与该原点一致；此差异由 [注册场景说明](docs/registered_scene_episodes_zh.md)“复用物体资产”明确承认。

#### E. 验证证据到底支持什么

| 证据 | 支持 | 不支持 |
|---|---|---|
| [test_spatial_episode_pipeline.py](tests/test_spatial_episode_pipeline.py) 中线段外投影、退化线、端点交换、错误 frame/NaN、重放测试；上一轮所选 22 项通过 | 投影定义、接口边界及状态一致性实现 | 图像可解、未见几何分布泛化、模型会投影 |
| `generate/verify` 独立构造 q，再与 `solve` 输出比较 | 构造 oracle 没有调用生产投影公式来生成 expected_answer，能检测部分实现错误 | 生成器只覆盖受限分布；resolver/校验/评分共用，不能算完全独立系统审计 |
| `run_episode_physics.semantic_check` 的在线残差与正交残差 | 独立几何条件核验，不直接调用生产投影公式 | 仍使用同一物理快照和 env.result 中的目标；不是视觉感知正确性验证 |
| 现有 oracle 物理报告 | 对所记录版本的 GT→IK→到达链路可执行 | 当前版本重跑结果、学习模型推理或抓取放置能力 |

CPU 错误答案对照将 expected 的三个分量各加 0.01 m；其余配置与 oracle 逻辑不变，检验评分器能否拒绝明显错点。物理负对照将回答 x 加 0.08 m，仍沿同一执行器运动；场景重新 reset，不能保证正负例实测状态逐位相同。监督工具每个任务类型选择一个实例做负例，不是每个 seed 都配对。两种负对照都没有改变视觉难度或生成捷径，不能证明能力有效性。[verify](geomind/tools/generate_spatial_suite.py)、[run_case/main](geomind/tools/run_episode_physics.py)。

#### F. 最小补充实验：打破等高投影捷径

**未执行。** 对同一条非水平直线、同一 q 和相同偏移长度 0.07，构造一对 p=q+0.07n：A 使用现有 n_z=0；B 将 n 在 d 的正交平面内旋转，使 n_z≠0，仍满足 n·d=0、工作空间和唯一性约束。保持对象数、ID/标签、答案 q、容差与输入格式不变；先用结构化坐标输入隔离视觉/控制，再按需扩展图像条件。

观察 paired 点误差、容差内正确率及按 pitch 分层的差值。A 好 B 差支持“依赖生成约束或分布偏好”的疑问，但不单独证明内部算法；A/B 都好且在 n_z≠0 的新实例上稳定，只能排除这条特定捷径；都差需先排查计算/输出格式，不能归因为视觉。应额外计算 t_z 捷径基线，预期仅 A 成立。

### 7.4 DetectSceneChange

#### A. 研究主张：记住参考状态后检测唯一变化对象

- **文档原意**：[v3 契约](docs/episode_contract_v3_zh.md)“空间记忆参数与时间边界”要求比较 observe.End 与 Respond 状态；[分类](docs/canonical_task_families_zh.md) 将其归为 SceneUpdating。
- **源码事实**：`solve` 计算每个指定对象的三维位移，恰好一个超过阈值才产生答案。`SemanticEpisode.step` 保存历史边界，在 Delay 执行事件，再于 Respond 发布查询。它不是依据视频运动轨迹来检测变化。
- **研究解释/推断**：与 SelectByOrder 等当前状态任务不同，它的语义答案原则上依赖两时刻；与 RecallObjectLocation 不同，返回变化身份而非历史坐标；与历史身份追踪不同，不要求在同质物体交换中维持身份。稳定 ID 和可区分外观让身份匹配很容易，剩余负担主要是存储参考位置并比较。作者希望测长时记忆、工作记忆容量还是短间隔变化检测，文档未规定实验尺度。

#### B. 完整 observation 与可见性

以下分别为当前配置重建的完整编码、隐藏、回答观测；没有省略 objects。step=1 与 step=0 的对象状态相同；step=3、4 与下面 step=2 的字段相同，仅 step 改变。

编码时：

```json
{
  "instruction": "Remember the locations of the coffee can, the meat tin, the yellow mustard bottle.",
  "phase": "observe",
  "step": 0,
  "scene_visible": true,
  "objects": [
    {
      "object_id": "block_0",
      "label": "the coffee can",
      "position_m": [
        -0.15,
        0.16729373271405273,
        0.023019588432768235
      ]
    },
    {
      "object_id": "block_1",
      "label": "the meat tin",
      "position_m": [
        0.0,
        0.13194116146359658,
        0.0172803293180392
      ]
    },
    {
      "object_id": "block_2",
      "label": "the yellow mustard bottle",
      "position_m": [
        0.15,
        0.1871199979680686,
        0.011649691321335177
      ]
    }
  ],
  "frame": "table",
  "allow_response": false
}
```

延迟时：

```json
{
  "instruction": "Wait for the query.",
  "phase": "delay",
  "step": 2,
  "scene_visible": false,
  "objects": [],
  "frame": "table",
  "allow_response": false
}
```

回答时：

```json
{
  "instruction": "Select the one object among the coffee can, the meat tin, the yellow mustard bottle whose reference point moved more than 0.03 metres since the end of phase observe, in frame table. Each object's reference point is its configured rigid-body origin.",
  "phase": "recall",
  "step": 5,
  "scene_visible": true,
  "objects": [
    {
      "object_id": "block_0",
      "label": "the coffee can",
      "position_m": [
        -0.15,
        0.16729373271405273,
        0.023019588432768235
      ]
    },
    {
      "object_id": "block_1",
      "label": "the meat tin",
      "position_m": [
        0.0,
        -0.05678025012200469,
        0.0172803293180392
      ]
    },
    {
      "object_id": "block_2",
      "label": "the yellow mustard bottle",
      "position_m": [
        0.15,
        0.1871199979680686,
        0.011649691321335177
      ]
    }
  ],
  "frame": "table",
  "allow_response": true
}
```

可确认：无图像、无历史帧数组、无原始事件计划、无 seed；稳定 ID、label 和可见阶段精确坐标公开。agent 可在自身缓存保留编码观测，`allow_history_replay=false` 仅表示环境不提供回放，**不表示 agent 被禁止记忆**。step/phase/scene_visible 公开事件所在的隐藏时间窗，但不直接透露哪个对象移动。CPU 中 observe.End 在 step=2 刚进入隐藏阶段保存；由于此前没有事件，它与最后一个可见 step=1 的位置一致。[runtime.py:reset/observation/step](geomind/episodes/runtime.py)。

原生物理 API 同样提供精确对象状态并加当前 robot 字段，隐藏时仍提供机器人本体状态；录像 agent 版隐藏阶段遮黑、observer 版可看见干预。这是两类输入渠道，不能把坐标 API 成绩称为纯视频记忆成绩。`export_evidence` 的完整历史、observer 视频、查询 GT 均属于评估端；物理 oracle 直接读取 GT。接口不阻止持有 env 的任意 Python 调用访问这些数据，需由实际模型适配器限制。

#### C. 完整事件实例

来源：[DetectSceneChange_v0](geomind/configs/episodes/DetectSceneChange_v0.json)、[change_s0](geomind/configs/layouts/change_s0.json)、[change_protocol_s0](geomind/configs/protocols/change_protocol_s0.json)。初始场景、编码和完整回答指令见上面 observation。

| 控制步 | 内部事件/阶段 | agent 看到什么 |
|---|---|---|
| 0、1 | Observe，位置不变 | 三对象精确坐标和记住位置的指令 |
| 2 | 保存 observe.End；进入 Delay | Wait、objects=[] |
| 3 | Delay.after_steps=1，MoveObject(block_1) | 仍隐藏；内部肉罐从 y=0.13194116146359658 移到 -0.05678025012200469，x=0、z=0.0172803293180392 保持 |
| 4 | 继续 Delay | 仍隐藏 |
| 5 | Respond/recall，生成查询 | 当前三对象坐标，允许一次身份回答 |

预期解法：按 object_id 存 observe.End 对应位置，与当前位置计算 `Δ_i=||p_i(now)-p_i(ref)||`；得到咖啡罐=0、肉罐=0.18872141158560127 m、芥末瓶=0，阈值 0.03 m。正确答案 `{"object_id":"block_1"}`，CPU ID 精确匹配，reward=1、terminated=true。本轮用 GT 回答得到 success=true，未测试任何受评模型。PhysX 事件是连续外力 PD、GT 取实测位置，不能把上述符号更新数值当作物理轨迹。

**更简单的替代解法**：该单例中肉罐是最终唯一 y<0 的物体，所以“选负 y”不读历史也答对；这只证明单例不能证明记忆，不意味着整个生成分布都可如此求解。生成器按固定 x 轨道、固定 ID/外观关联采样；初始 y 与最终 y 都来自[-0.2,0.2]，但最终采样附带位移≥0.12的拒绝条件，因此“同一范围”不等于“与变化身份统计独立”。需要 final-only 基线检验潜在位置边缘偏好/ID偏好，不能仅凭代码隐藏历史就宣称不存在捷径。[generate](geomind/tools/generate_spatial_suite.py)。

#### D. 生成约束

依据：[EpisodeResolver._validate_parameters/_validate_protocol/solve](geomind/episodes/resolver.py) 与 [generate](geomind/tools/generate_spatial_suite.py)。

- 恰好三阶段 Observe→Delay→Respond；观察/延迟为正整数步，参考必须是首阶段 End；DetectSceneChange 的 Respond 可见。对象列表非空、无重复且 ID 必须存在。
- 阈值 τ>ε>0；对所有候选要求 `|Δ_i-τ|>ε`，并要求 `|{i:Δ_i>τ}|=1`。当前 τ=0.03、ε=0.001 m：阈值附近、零变化、多变化都拒绝，而非随意选一个对象。
- snapshot 的 instance_id、frame、contract_hash 必须一致；历史步等于 Observe 长度、查询步至少 Observe+Delay，禁止跨 episode 拼历史和提前求解。
- 事件对象存在、位置在工作空间内、发生在 Delay 的合法步数；同一步同对象的重复事件拒绝。生成器只选一个 moved，保持 x/z、仅重采样 y，最多尝试 100 次以保证位移≥0.12，再进行显式断言。
- 两个未移动物体就是“未变化”干扰项；对象数量、固定 x 间距、唯一标签和易区分资产没有被系统随机化。CPU 不检验真实遮挡、图像分辨率、物体跨帧身份歧义或碰撞。物理 reset/稳定和变化唯一性会再检验，但这仍不是模型可辨识性检验。

#### E. 验证证据与 oracle 独立性

`tests/test_spatial_episode_pipeline.py` 的 `test_memory_visibility_query_timing_and_event_application` 验证公开字段、隐藏期 objects=[]、历史不被后续更新覆盖；提前回答/reset/跨运行历史、无变化/多变化/阈值边界等测试验证契约执行。它们**不证明模型无法从最终帧猜中**，也不检验实际远程服务是否被多给了历史或 GT。

`generate` 用抽样 moved ID 作为 expected_answer，不调用生产 `solve` 的位移比较来生成 expected；但两者共享同一协议和布局，事件执行也是同一个 runtime，属于计划身份对状态实现的交叉检查，不是完全独立的数据采集。物理 `semantic_check` 自己计算实测位移集合并与 planned event ID 比较，仍使用同一快照且与生产 GT 共享“位移阈值”这一数学定义。离线重放再次调用相同 `solve`，证明可追溯和确定性，**不构成独立 oracle**。

CPU 负对照把 answer.object_id 换成另一个现有对象；物理负对照亦选一个错误 ID，使机械臂到错物体上方。它们只改回答/动作目标，未改参考历史、隐藏长度、候选数量、外观或事件分布。它们证明评分能区分错对象，不能证明正确回答必须使用记忆。[verify](geomind/tools/generate_spatial_suite.py)、[run_case/semantic_check](geomind/tools/run_episode_physics.py)。历史 oracle 物理成功是动作闭环证据，当前未找到这批原生任务上 full-history 对 final-only 的受评模型配对成绩。

#### F. 最小补充实验：相同最终状态、不同参考历史

**未执行。** 构造配对 episode A/B：最终三个对象的坐标、ID、外观、查询文本和 step 完全相同；A 的参考状态只让对象 u 与终态相差>τ+ε，B 只让对象 v 与终态相差>τ+ε，其余与终态一致。保持观察/隐藏时长、位移幅度和物体数一致，保证两例唯一答案不同。

分别评估完整流与只给最终 observation 的条件，模型每例 reset 缓存，随机化配对顺序且不把 pair ID 暴露给模型。观察 paired 身份正确率及答案是否随历史正确切换。无历史输入分布相同，不可能可靠区分这对的两个不同标签；若 full-history 能切换、final-only 不能，才支持历史信息的必要性。两者都好应先审计串缓存、seed/文件名/metadata 或 GT 泄露；两者都差只能说明该模型未利用历史，还须排除输入/延迟格式不适配。受限配对中的 final-only 最优平均上限取决于标签平衡，不应机械引用原三候选 1/3。

### 7.5 SelectAfterExclusion：已实现的有限组合与尚缺的一般组合

#### A. 研究主张与边界

- **文档原意**：[分类](docs/canonical_task_families_zh.md) 将其列为 ConstraintComposition；没有明确逐任务写出独立设立的因果论证。
- **源码事实**：[constraint_selection_logic.py:resolve_exclusion_nearest_constraint_problem](geomind_lab/mimic/envs/constraint_selection_logic.py) 先按 color/shape 过滤，在过滤后集合内找唯一最左对象并排除，再在剩余对象中选距蓝盘最近者。不是按最终答案写死返回；不过 `expected_target` 与计算结果不一致会报错。
- **研究解释/推断**：相对单一 SelectByDistance，它原则上要求同时尊重属性、集合内极值和排除范围；可研究单谓词正确但组合失败。它不涉及中间实体多跳，不是任意两关系 conjunction；当排除不改变答案时，这种设计主张就没有被实例强制。

#### B. 完整观测样例与信息泄露边界

本轮只能实测**mock 的 agent-ready observation**，完整如下（不是原生语义 objects 格式，也不是受评 VLA 请求）：

```json
{
  "instruction": "Pick up the red block closest to the blue plate that is not the leftmost one.",
  "state": {
    "backend_obs": {
      "qpos": [
        0.38333332538604736,
        0.013333330862224102,
        0.4633333384990692,
        0.013333330862224102,
        0.08000001311302185,
        0.05000000074505806,
        0.0
      ],
      "qvel": [
        0.0,
        0.0,
        0.5999999046325684,
        6.0,
        0.0,
        0.0,
        65.0
      ],
      "task_vector": [
        5.0,
        7.0,
        6.0,
        6.0,
        0.0,
        0.27835050225257874,
        0.5051546096801758
      ],
      "cursor_xy": [
        0.38333332538604736,
        0.013333330862224102
      ],
      "goal_xy": [
        0.4633333384990692,
        0.013333330862224102
      ],
      "binding": {
        "goal_kind": "generic",
        "target_object_id": "blue_plate",
        "layout_signature": "select_exclusion_unique_scene65_base",
        "scene_signature": "scene_65"
      }
    }
  },
  "cameras": {
    "ego_view": null,
    "table_cam": null
  },
  "episode_step": 0
}
```

这个 observation 没有 objects 列表、图像像素、历史帧；cameras=null，state 公开 goal_xy、goal_distance 所在 qpos 以及 binding.target_object_id。更关键的是这里 target_object_id=blue_plate、goal_kind=generic，并不是语义 GT 的 red_block_target。原因是 [MockBackendEnv._build_task_binding/_build_generic_binding](geomind/core/envs/mock_backend.py) 没有处理 exclusion resolver，落到所有物体均值位置的 generic 目标。因此这份完整样例只证明**mock 不能作为该任务研究有效性的证据**，不能冒充模型真正做了排除选择。

真实旧 IsaacLab 接口的源码可见性与模型传输还要分开：

- [ObservationsCfg.PolicyCfg](geomind_lab/tasks/manager_based/manipulation/geomind/geomind_env_cfg.py) 声明 joint_pos/vel、object、object_positions/orientations、eef_pos/quat、gripper_pos、goal_error 等；`_adapt_observation` 把 native_obs 放入 state，另外将 info 合入 raw.meta。本地 hook 能拿到 env、raw_obs 和 trace，权限大于纯图像模型。
- [geomind_client.py:build_payload](geomind_client.py) 的通用 HTTP payload 发送 observation、raw.meta、可选 trace_tail（默认长度 0）和提取出的图像；meta/goal_error 等是否泄露目标，取决于服务端消费范围，不能把 `to_agent_ready_observation` 去掉顶层 meta 等同于 HTTP 没发送 meta。
- 同文件 `_build_openpi_infer_element` 单独构造四个输入：observation/image、observation/wrist_image、observation/state（末端位置、轴角、夹爪的本体信息）、prompt。它没有直接把 object_positions、goal_error、raw.meta 发给 OpenPI infer。这不替所有 HTTP 服务保证隔离。

**缺失证据**：本次没有该任务某个已绑定 checkpoint/服务端的实际完整 RGB/proprioception 请求快照，也没有现成可确认的输入 allowlist 审计。因此无法声称任何历史模型实际只收到哪些字段，更不能编造 native_obs 数值或用空图像代替真实模型 observation。上面的 mock JSON 与源码字段清单明确区分了实测和未确认。

#### C. 一个完整的语义实例（不冒充实际物理 rollout）

实例是兼容标注 `select_exclusion_unique_scene65_base`，由 [task_specs.py:load_annotated_tasks](geomind_lab/tasks/manager_based/manipulation/geomind/task_specs.py) 合并 [legacy/layouts/standard.json](geomind/configs/legacy/layouts/standard.json) 与独立 [初始化布局](geomind/configs/layouts/select_exclusion_unique_scene65_base.json) 读取。本轮只解析任务配置、计算名义 XY 距离，没有读取私有轨迹。

公开任务指令：`Pick up the red block closest to the blue plate that is not the leftmost one.` 协议 query_text：“拿起离蓝盘最近且不是最左边的红方块。”初始且查询时名义场景如下；没有历史干预事件，查询为静态选择。

| 对象 ID | 属性 | 世界 XY（m） | 到蓝盘距离（m） |
|---|---|---|---|
| blue_plate | 蓝、陶瓷、盘 | [0.39,0.02] | 参照物 |
| red_block_leftmost | 红、方块 | [0.20,-0.15] | 0.254950976 |
| red_block_target | 红、方块 | [0.56,0.02] | 0.170000000 |
| red_block_second | 红、方块 | [0.48,0.19] | 0.192353841 |
| red_block_far | 红、方块 | [0.60,-0.12] | 0.252388589 |
| orange_block_near_plate | 橙、方块 | [0.55,0.12] | 0.188679623 |

这些是**oracle 的完整场景事实**，不是模型 objects observation；真实回答时图像/实测物体姿态本次缺失。预期：保留四个红方块→排除 x 最小的 red_block_leftmost→在三个剩余对象中找最近，答案 red_block_target。本轮求解器返回该 ID，符合标注 expected_target。

真实机器人执行仍按旧 goal sequence/实测距离等逻辑评分，普通 [LayoutAwareEvaluator](geomind/tasks/evaluators/simple.py) 在 `constraint_selection_evaluator` 名下检查有效距离与末尾连续 dwell（默认5）。它不是把模型回答 JSON 直接做 ID 精确匹配，也不必然检验“拿起”的抓取事实；本轮没有该任务动作成功证据。

**已确认的实例捷径**：本例 red_block_target 本来就是全部五候选中最近者；忽略颜色和排除均得到相同答案。对另一份 `select_exclusion_unique_scene17_stress` 的本轮配置计算也得到同样结论。故这两个已检查布局不能区分组合求解器与“离盘最近就选它”，对象名中 target/leftmost 又是结构化接口的额外捷径。该判断限定为两份名义布局，不外推所有 seed 或实测漂移。

#### D. 约束：唯一性与必要性不是同一件事

设 C 为 candidate_object_ids，S={i∈C:color=red且shape=cuboid}，a 为参照盘位置。`resolve_exclusion_nearest_constraint_problem` 要求：

1. |S|≥2；极小 x 的 ε_rank 邻域内恰好一个对象 e（当前 ε_rank=0.003 m），否则拒绝。
2. T=S\{e} 非空；`argmin(i∈T)||p_i-a||_XY` 的 ε_dist 邻域内恰好一个对象（当前 ε_dist=0.003 m），否则拒绝。
3. 仅支持世界 XY 欧氏距离；若计算答案不等于 expected_target，拒绝；属性来自 protocol.object_semantics，生产求解不从图像识别颜色/形状。

这保证唯一解，但没有要求“去掉任一条件都会改变答案”，也没有强制设置最近的非红干扰物或让被排除物成为最近红方块。已有静态布局是人工指定，而非一个保证组合必要性的随机实例生成器。测试中的蓝盘净距检查（至少0.16 m）只是这些名义坐标的一个约束，不证明任意资产/随机姿态下的碰撞和抓取可行性。[tests/test_constraint_selection_protocol.py](tests/test_constraint_selection_protocol.py)。

#### E. 验证证据与缺口

同测试文件的 annotation unique GT、并列最近拒绝、极值并列拒绝、红块净距测试证明生产求解器在所列场景上的输出和边界行为。测试直接调用生产 `build_constraint_selection_problem_from_object_positions` / `resolve_exclusion_nearest_constraint_problem`，并与硬编码 expected_target 对比，**不是另一套独立组合 oracle**。本轮另以名义坐标独立算了距离来检查捷径，仅支持上面的实例分析。

这套单测没有“删除排除条件时模型变差”的对照；并列用例是 invalid-input 测试，不是能力消融。未找到该选定任务按相同输入预算、绑定模型版本的“最近单规则 vs 完整组合”成绩。已有 direct 消融会同时改指令与目标颜色显著性，[返修指标说明](paper/rebuttal_metrics.md) 也承认这一点；它不能隔离组合算子。mock 的 generic binding 更不能作为关系组合实现已完成端到端验证的依据。

因此可确认“组合语义求解逻辑已实现”，不能确认“现有实例迫使模型组合”，更不能确认“模型具备/缺乏一般关系组合能力”。强形式双关系交集与多跳任务仍无实例、评分和验证链。

#### F. 最小补充实验：让排除条件真正改变答案

**未执行。** 构造同场景两条查询配对：A“选离蓝盘最近的红方块”，B“选离蓝盘最近且不是最左侧的红方块”。布置至少三个红块，使唯一最左红块 e 同时是最近红块、另一个红块 t 是排除后的唯一最近者，二者距离差>ε_dist；固定相机、图像、物体姿态、ID、动作预算和控制器，只改查询及相应正确目标（A=e，B=t）。另放一个更近的非红块可确保颜色条件也非冗余，但首轮只分析排除效应。

观察语义目标选择和配对答案切换，先用隔离过目标元数据的结构化/图像决策接口，不先用末端到达混合控制。A、B 都选 e 支持忽略排除；A=e、B=t 支持该实例上的组合使用，但仍不是多跳泛化；两者都错须检查单关系识别和指令理解；语义正确而动作失败定位到控制链。为避免 role-ID 捷径需随机重命名对象且不暴露 target/leftmost，并让属性/目标在配对集合中平衡。实现新布局时还须同步 expected_target 和独立标注，避免旧的固定答案校验阻止合法换目标。

### 7.6 当前覆盖面与尚缺证据的主张

| 能力层 | 已覆盖的具体部分 | 仍不能据此声称 |
|---|---|---|
| 结构化推理 | 三维点/对象 ID 输入下的投影、位移比较；旧协议的属性过滤、排序排除、最近选择有生产求解器 | 学习模型真正做了上述运算；一般关系组合、多跳、任意几何/干扰项分布泛化 |
| 视觉理解 | 有资产/标签、相机录制和旧 VLA 图像适配器；历史报告描述过渲染检查 | 当前两个原生 API 是纯图像任务；刚体原点可被视觉准确定位；排除了坐标/ID/GT metadata 捷径；历史模型分数已绑定当前三任务 |
| 记忆 | Observe/Delay/Respond、隐蔽事件、历史快照、跨运行隔离、当前/历史位移比较 | 身份追踪、长时容量、同质对象保持、模型必须记忆；本轮未有相同终态的历史反事实对照 |
| 动作 | 三个原生任务有 oracle 到达、姿态/dwell 控制及错误目标历史证据；旧环境有动作策略接口 | 抓取、释放、稳定放置、真实机器人、模型端到端控制普遍可靠；普通 mock 语义与真实任务等价 |

位置与依据分别见上面各任务 A–E，以及 `runtime.py`、`isaaclab_physics.py:execute_answer`、`constraint_selection_logic.py`、`geomind_client.py`。**当前最强证据集中在实现正确性；能力有效性需要反捷径对照；模型表现必须重新绑定明确输入条件和模型版本。** 现有历史返修统计可作为另一个研究批次参考，不能移植到新原生任务上。

尚缺证据的关键主张：投影对一般三维偏移有效、变化检测成绩必须依赖历史、已有排除实例测到组合必要性、双关系/多跳已实现、视觉指标不受结构化元数据污染，以及语义失败能与控制/接口失败干净解耦。上面每个判断均有具体源码或历史说明支撑；未见模型配对结果的地方明确留空，不用 oracle 得分替代。

### 7.7 最先应解决的三个问题

1. **冻结受评输入边界并审计真正发到模型的 payload。** 原生 API 公开精确坐标，mock 甚至公开 goal，通用 HTTP 又发送 meta；不同接口结论无法混比。先以 allowlist、输入快照和 checkpoint revision 区分坐标推理/纯图像/视频记忆，本地 hook 与评估端隔离，再谈能力归因。依据：`SemanticEpisode.observation`、`MockBackendEnv._obs`、`geomind_client.build_payload/_build_openpi_infer_element`。
2. **让任务主张在实例中不可被省略。** 优先做 DetectSceneChange 相同终态/不同历史配对、排除条件改变答案配对，并补投影 n_z≠0 对照。当前可给出显式捷径或条件冗余，增加随机 seed 本身不能解决；不先修这个问题，更多模型评估也只会更精确地量到混合能力。依据：本节7.3C/F、7.4C/F、7.5C/F。
3. **把语义决策、动作执行和结果版本拆开验收。** 双关系/多跳明确保留缺口，SelectAfterExclusion 不用 generic mock 替代；先报告独立标注的目标/坐标误差，再固定同一控制器看动作。报告绑定配置/代码/模型/指标版本，保留 invalid/system failure；否则旧指标修正、历史哈希漂移与到达失败仍会污染研究结论。依据：`_build_generic_binding`、`LayoutAwareEvaluator.evaluate`、历史 `paper/rebuttal_disentangle_draft.md` 内部限制和前述报告契约不匹配。

### 7.8 需研究者确认的设计问题

- ProjectOntoLine 独立设立的理论依据是什么？要测一般三维正交关系，还是允许固定偏移平面的程序化几何分布？评价对象是坐标推理模型、视觉模型还是动作策略？
- catalog 的 rigid-body origin 是否是预期视觉目标？若是，如何提供足够标定/关键点监督，使答案不依赖不可见的 USD 原点约定？
- DetectSceneChange 旨在测位置缓存、事件察觉、长时记忆还是跨外观歧义的身份保持？稳定 object_id、可区分资产和机器人本体状态是否属于允许的输入？
- “关系组合”是否接受排除+距离这种有限算子组合？若要双参照交集/多跳，需要什么科学假设、单关系对照和反事实标注标准？
- 指令中的“不是最左侧”作用域是红色子集还是所有对象？当前代码固定在属性过滤后集合内排除；同类中文/英文表述是否经过研究者确认？
- 研究主指标应是语义答案、reach@k、末步分级距离还是实际抓取/放置？旧返修的不同口径和目标显著性修改，应如何限制其因果解释？

这些是从现有材料无法确定的意图，不作为本次已确认设计写入事实层。
