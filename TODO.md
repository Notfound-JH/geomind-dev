# TODO

## P0 跑通最小闭环

- [] 把 `IsaacLab/` 复制到仓库根目录，使 `geomind_lab/bootstrap/isaaclab_path.py` 能找到 `IsaacLab/source`
- [ ] 在仓库根目录执行 `pip install -e .`
- [ ] 确认资产路径可用：
  - [ ] `GEOMIND_ASSETS_ROOT`
  - [ ] `assets/assets_catalog_with_metadata.json`
  - [ ] scene USD 路径 `GEOMIND_ASSETS_ROOT/stage/<scene_id>/...`
- [ ] 准备最小任务 JSON：
  - [ ] `assets/tasks/annotated_tasks.json` 或 `assets/tasks/annotated_tasks_1.json`
- [ ] 写一个固定任务注册入口
  - [ ] 新增 `gym.register(...)`
  - [ ] 先只注册 1 个固定 env
  - [ ] 推荐任务：`scene_id=1`, `task_layout_key="line_3"`, `task_index=0`
- [ ] 验证最小 env 可运行：
  - [ ] `gym.make(...)`
  - [ ] `env.reset()`
  - [ ] `env.step(action)`


## P0 稳定当前配置链

- [ ] 检查 [geomind_scene_cfg.py](/home/elm/projects/geomind_lab/geomind_lab/tasks/manager_based/manipulation/geomind/geomind_scene_cfg.py) 的 scene 解析是否覆盖你当前常用 scene
- [ ] 检查 [task_specs.py](/home/elm/projects/geomind_lab/geomind_lab/tasks/manager_based/manipulation/geomind/task_specs.py) 的 object remap 是否足够稳定
- [ ] 确认 [geomind_env_cfg.py](/home/elm/projects/geomind_lab/geomind_lab/tasks/manager_based/manipulation/geomind/geomind_env_cfg.py) 中：
  - [ ] `goal_error`
  - [ ] `success`
  - [ ] `grasp_*`
  - [ ] `*_dropping`
  的动态注入逻辑符合预期
- [ ] 确认 [franka_env_cfg.py](/home/elm/projects/geomind_lab/geomind_lab/tasks/manager_based/manipulation/geomind/config/franka/franka_env_cfg.py) 的桌面高度解析和机器人初始高度一致


## P1 场景加载与环境加载

- [ ] 明确 `SceneProfile` 的最终字段边界
  - [ ] `scene_id`
  - [ ] `usd_path`
  - [ ] `object_names`
  - [ ] `object_assets`
  - [ ] `xforms`
  - [ ] `container/region` 信息
- [ ] 如果后续任务依赖 region 语义，在 `geomind_scene_cfg.py` 中补 scene region 抽取层
- [ ] 让运行时只消费 `SceneProfile`，不要重新调用离线 `scene_construction`
- [ ] 保持 `scene_construction/` 为离线工具目录，不并入 runtime


## P1 任务标注与环境加载

- [ ] 固定 `TaskSpec` schema
  - [ ] `task_name`
  - [ ] `instruction`
  - [ ] `goal_sequence`
  - [ ] `object_positions`
  - [ ] `scene_id`
  - [ ] `layout_key`
  - [ ] `task_index`
  - [ ] `metadata`
- [ ] 把 annotation JSON 到 `task_spec` 的转换逻辑继续集中在 [task_specs.py](/home/elm/projects/geomind_lab/geomind_lab/tasks/manager_based/manipulation/geomind/task_specs.py)
- [ ] 明确哪些字段由 annotation 提供，哪些字段由 env 推导
- [ ] 对 `goal_sequence` 加最基本的 schema 校验
- [ ] 对“标注 object 名称”和“scene object 名称”不一致的情况补单元测试


## P1 轨迹收集与环境加载

- [ ] 不再引入旧仓库自定义 recorder 主链
- [ ] 接 IsaacLab 原生 recorder
  - [ ] 在 env cfg 或运行入口配置 `recorders`
  - [ ] 优先对接 `ActionStateRecorderManagerCfg`
- [ ] 设计 episode 元数据结构
  - [ ] `scene_id`
  - [ ] `layout_key`
  - [ ] `task_index`
  - [ ] `task_name`
  - [ ] `instruction`
  - [ ] `policy_name`
  - [ ] `success`
- [ ] 决定 metadata 保存方案
  - [ ] sidecar JSON
  - [ ] 或自定义 dataset handler
- [ ] 在 `geomind_lab/data/io/` 下建立统一的 episode metadata 读写接口


## P1 CLI 与运行入口

- [ ] 完善 [collect.py](/home/elm/projects/geomind_lab/geomind_lab/cli/collect.py)
  - [ ] 接 `gym.make`
  - [ ] 接 teleop device
  - [ ] 接 recorder
- [ ] 完善 [replay.py](/home/elm/projects/geomind_lab/geomind_lab/cli/replay.py)
  - [ ] 接 dataset 路径
  - [ ] 接 replay 逻辑
- [ ] 完善 [annotate.py](/home/elm/projects/geomind_lab/geomind_lab/cli/annotate.py)
  - [ ] 支持 episode metadata 或 subtask annotation
- [ ] 完善 [generate.py](/home/elm/projects/geomind_lab/geomind_lab/cli/generate.py)
  - [ ] 接 Mimic/SkillGen 数据生成
- [ ] 保持 `scripts/*.sh` 只做薄封装，不承载业务逻辑


## P2 模型交互与轨迹收集

- [ ] 定义统一 runner 接口
  - [ ] teleop runner
  - [ ] policy runner
  - [ ] replay runner
- [ ] 保证 teleop 和 policy 走同一个 `env.step(action)` 主链
- [ ] 保证 teleop 和 policy 写同一种 episode schema
- [ ] 明确 policy rollout 的 metadata
  - [ ] 模型名
  - [ ] checkpoint
  - [ ] inference config
- [ ] 在 `geomind_lab/data/io/` 中加入 rollout metadata 输出


## P2 Mimic / SkillGen

- [ ] 继续完善 [geomind_franka_ik_rel_mimic_env.py](/home/elm/projects/geomind_lab/geomind_lab/mimic/envs/geomind_franka_ik_rel_mimic_env.py)
- [ ] 继续完善 [geomind_franka_ik_rel_mimic_env_cfg.py](/home/elm/projects/geomind_lab/geomind_lab/mimic/configs/geomind_franka_ik_rel_mimic_env_cfg.py)
- [ ] 把 “`task_spec -> subtask_configs`” 提炼成单独转换层
- [ ] 不要把复杂 subtask 拆分规则硬编码进 env cfg
- [ ] 对多步任务补更细的 `subtask_term_signal`
- [ ] 明确 `get_expected_attached_object()` 的规则
- [ ] 对 place 之外的 task category 设计新的 subtask 编译逻辑


## P2 测试与验证

- [ ] 给 `task_specs.py` 补基本单测
- [ ] 给 scene object remap 补单测
- [ ] 给 `goal_sequence_reached()` 补单测
- [ ] 给 Mimic `subtask_configs` 构造补单测
- [ ] 写一个最小 smoke test
  - [ ] 创建 env
  - [ ] reset
  - [ ] 随机 step 5-10 次


## P3 清理与冻结旧链路

- [ ] 明确旧仓库这些目录只作为参考，不再继续开发
  - [ ] `/home/elm/projects/geomind/envs`
  - [ ] `/home/elm/projects/geomind/collection`
  - [ ] `/home/elm/projects/geomind/scripts`
  - [ ] `/home/elm/projects/geomind/geomind_tasks`
  - [ ] `/home/elm/projects/geomind/geomind_skillgen`
- [ ] 保留 `/home/elm/projects/geomind/scene_construction` 作为离线工具参考
- [ ] 在新仓库 README 里明确“新主线”和“旧参考目录”的边界


## 旧文件参考

优先参考这些旧文件：

- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/geomind_env_cfg.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/geomind_scene_cfg.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/mdp/observations.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/mdp/terminations.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/mdp/common.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/config/franka/franka_env_cfg.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/config/franka/geomind_ik_rel_visuomotor_env_cfg.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/utils/asset_utils.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/utils/scene_utils.py`

Mimic 参考：

- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/config/franka/geomind_skillgen_scripts/grid_2_2_4_projective_relation_skillgen_env.py`
- `/home/elm/projects/geomind/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env.py`
- `/home/elm/projects/geomind/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env_cfg.py`
- `/home/elm/projects/geomind/IsaacLab/source/isaaclab/isaaclab/envs/mimic_env_cfg.py`


## 当前最短路径

如果只做最重要的事情，顺序就是：

1. 复制 `IsaacLab/`
2. 注册 1 个固定任务 env
3. 跑通 `gym.make -> reset -> step`
4. 接上 IsaacLab recorder
5. 跑通 collect / replay
