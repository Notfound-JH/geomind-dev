# GeoMind Lab Development Context

## 1. 项目定位

`geomind_lab` 现在的目标是：

- `geomind_lab` 作为独立业务包
- `IsaacLab` 作为同级依赖目录，由你手动复制到仓库根目录
- 所有 GeoMind 任务逻辑、环境配置、Mimic 包装、CLI 入口都收敛到 `geomind_lab/`
- 不再把 GeoMind 主逻辑塞进 `IsaacLab/source/...`
- 不再以旧仓库根目录的 `envs/ + collection/ + scripts/` 为主线

推荐的最终关系如下：

```text
geomind_lab/
├── IsaacLab/          # 手动复制的上游依赖
├── geomind_lab/       # 业务代码主包
├── assets/            # 任务定义、标注、静态配置
├── scene_construction/# 离线场景工具
├── scripts/           # 薄封装命令
└── outputs/           # 本地产物
```


## 2. 当前目录下的系统逻辑

### 2.1 启动与路径管理

入口文件：

- `geomind_lab/bootstrap/isaaclab_path.py`
- `geomind_lab/bootstrap/env_setup.py`

职责：

- 假定 `IsaacLab/source` 位于仓库根目录下的 `IsaacLab/source`
- 在运行 CLI 前，把 `IsaacLab/source` 注入 `sys.path`
- 设置一些默认环境变量：
  - `GEOMIND_LAB_ROOT`
  - `GEOMIND_LAB_ASSETS_ROOT`
  - `GEOMIND_LAB_OUTPUTS_ROOT`

这部分的作用是让 `geomind_lab` 保持独立，不依赖修改 `PYTHONPATH` 的外部手工操作。


### 2.2 资产与场景解析

核心文件：

- `geomind_lab/utils/asset_utils.py`
- `geomind_lab/utils/scene_utils.py`
- `geomind_lab/tasks/manager_based/manipulation/geomind/geomind_scene_cfg.py`

逻辑分层：

1. `asset_utils.py`
   - 解析 `GEOMIND_ASSETS_ROOT`
   - 解析 `assets_catalog_with_metadata.json`
   - 提供资产查询方法：
     - `get_asset_info_by_name()`
     - `get_ground_plane_info()`

2. `scene_utils.py`
   - 负责解析 `.usda` 场景文件
   - 提取 Xform 层级结构
   - 把原始 USD 文本结构转成 Python 可消费的数据结构

3. `geomind_scene_cfg.py`
   - 基于 `scene_utils` 解析 scene USD
   - 构造 `SceneProfile`
   - 从场景结构中识别：
     - table
     - ground
     - basket
     - decoration
     - objects
   - 将 scene 中的 object name 映射到真实资产 USD
   - 在 `GeoMindSceneCfg.apply_scene()` 中把这些资产应用到 IsaacLab 的 scene cfg 上

这条链路的核心思想是：

- `scene_construction` 只负责离线生成 USD
- 运行时不调用 `scene_construction`
- 运行时只消费 scene USD 和资产 catalog


### 2.3 任务定义与 task spec

核心文件：

- `geomind_lab/tasks/manager_based/manipulation/geomind/task_specs.py`

职责：

- 解析标注任务 JSON
- 根据 `layout_key + task_index + scene_profile` 生成运行时 `task_spec`
- 如果标注里的 object name 和场景里的 object name 不一致，自动做 remap
- 如果没有标注，自动生成默认任务：
  - 把最左侧物体放进容器

当前 `task_specs.py` 的主要接口：

- `resolve_annotated_tasks_path()`
- `load_annotated_tasks()`
- `build_default_task_spec()`
- `load_task_spec()`

这部分现在是新系统里“任务语义”的入口。


### 2.4 基础环境配置

核心文件：

- `geomind_lab/tasks/manager_based/manipulation/geomind/geomind_env_cfg.py`

职责：

- 定义基础 `GeoMindEnvCfg(ManagerBasedRLEnvCfg)`
- 在 `__post_init__()` 中串起：
  - scene profile 加载
  - task spec 加载
  - 动态 object 层生成
  - observation term 注入
  - termination term 注入

当前系统逻辑如下：

1. 根据 `scene_id` 调 `get_scene_profile()`
2. 根据 `task_layout_key + task_index` 调 `load_task_spec()`
3. 根据 scene 中的 object list 动态配置：
   - `object`
   - `object_positions`
   - `object_orientations`
   - `grasp_1`, `grasp_2`, ...
   - `*_dropping`
4. 把 `task_spec["goal_sequence"]` 注入：
   - `observations.policy.goal_error`
   - `observations.subtask_terms.success`
   - `terminations.success`
5. 在 scene 里动态挂载 object rigid bodies

这意味着：

- 场景、任务、obs、termination 已经开始收敛到一个 cfg 内部
- 后面继续做 IsaacLab 接入时，应当继续以 `EnvCfg` 为中心，而不是回到脚本手动拼装


### 2.5 MDP 逻辑

核心文件：

- `geomind_lab/tasks/manager_based/manipulation/geomind/mdp/common.py`
- `geomind_lab/tasks/manager_based/manipulation/geomind/mdp/observations.py`
- `geomind_lab/tasks/manager_based/manipulation/geomind/mdp/terminations.py`

职责：

- `common.py`
  - 统一从 `env.scene[...]` 中取 world pose
  - 同时兼容 rigid object 和 extras/xform asset

- `observations.py`
  - `object_obs`
  - `cube_positions_in_world_frame`
  - `cube_orientations_in_world_frame`
  - `object_grasped`
  - `task_goal_xy_error`

- `terminations.py`
  - `asset_height_below_minimum`
  - `goal_sequence_reached`

这部分基本就是 GeoMind 的 MDP 自定义层。


### 2.6 Franka 机器人配置

核心文件：

- `geomind_lab/tasks/manager_based/manipulation/geomind/config/franka/franka_env_cfg.py`
- `geomind_lab/tasks/manager_based/manipulation/geomind/config/franka/geomind_ik_rel_env_cfg.py`
- `geomind_lab/tasks/manager_based/manipulation/geomind/config/franka/geomind_ik_rel_visuomotor_env_cfg.py`

逻辑分层：

1. `franka_env_cfg.py`
   - 机器人本体配置
   - IK relative action
   - gripper action
   - reset event
   - end-effector frame
   - teleop keyboard 设备

2. `geomind_ik_rel_env_cfg.py`
   - 当前只是 `FrankaGeoMindEnvCfg` 的兼容别名
   - 方便后面做注册时使用更稳定的命名

3. `geomind_ik_rel_visuomotor_env_cfg.py`
   - 添加 wrist/table/default cameras
   - 定义 visuomotor observation group
   - 定义 skillgen 用的 subtask observation group

这部分的整体思路是：

- 基础 robot cfg 和 task cfg 分离
- 视觉版本只在上层加 camera 和 obs，不污染 base env cfg


### 2.7 Mimic / SkillGen 包装

核心文件：

- `geomind_lab/mimic/envs/geomind_franka_ik_rel_mimic_env.py`
- `geomind_lab/mimic/configs/geomind_franka_ik_rel_mimic_env_cfg.py`

当前实现逻辑：

1. `GeoMindFrankaIKRelMimicEnv`
   - 基于 `ManagerBasedRLMimicEnv`
   - 实现标准 Mimic 接口：
     - `get_robot_eef_pose`
     - `target_eef_pose_to_action`
     - `action_to_target_eef_pose`
     - `actions_to_gripper_actions`
     - `get_subtask_term_signals`
     - `get_expected_attached_object`

2. `GeoMindFrankaIKRelMimicEnvCfg`
   - 继承 `FrankaGeoMindIKRelVisuomotorSkillgenEnvCfg + MimicEnvCfg`
   - 默认配置 `datagen_config`
   - 从 `task_spec` 自动构造一个最简单的 place-style `subtask_configs`

注意：

- 这部分不是旧文件的原样搬运
- 当前是“通用版 place-task Mimic 框架”
- 对更复杂的多步任务，后续要继续细化 `subtask_configs` 的构造逻辑


### 2.8 CLI 与 scripts

核心文件：

- `geomind_lab/cli/*.py`
- `scripts/*.sh`

当前状态：

- CLI 还是占位入口
- 只做了 bootstrap 和参数框架
- 还没有真正接 `gym.make()`、`parse_env_cfg()`、`record_demos.py`

也就是说：

- 当前 `geomind_lab` 的“配置与代码骨架”已迁好
- “任务注册与端到端运行入口”还没完成


## 3. 已迁移的文件来源

### 3.1 优先迁移源

这批文件是本次迁移的主要来源：

- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/geomind_env_cfg.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/geomind_scene_cfg.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/mdp/observations.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/mdp/terminations.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/mdp/common.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/config/franka/franka_env_cfg.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/config/franka/geomind_ik_rel_visuomotor_env_cfg.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/utils/asset_utils.py`
- `/home/elm/projects/geomind/geomind_env/geomind_tasks/utils/scene_utils.py`


### 3.2 Mimic 参考源

旧 GeoMind SkillGen 参考：

- `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/config/franka/geomind_skillgen_scripts/grid_2_2_4_projective_relation_skillgen_env.py`

IsaacLab 官方 Mimic 参考：

- `/home/elm/projects/geomind/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env.py`
- `/home/elm/projects/geomind/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env_cfg.py`
- `/home/elm/projects/geomind/IsaacLab/source/isaaclab/isaaclab/envs/mimic_env_cfg.py`

当前新包里的 Mimic 实现更偏向 IsaacLab 官方写法，而不是旧 GeoMind 的特化逻辑。


## 4. 可参考的旧文件夹与作用

### 4.1 应继续参考的旧目录

1. `/home/elm/projects/geomind/geomind_env/geomind_tasks/manipulation/geomind_env/`
   - 这是最接近 IsaacLab-native 风格的旧实现
   - 后续继续迁代码时，优先参考这里

2. `/home/elm/projects/geomind/geomind_env/geomind_tasks/utils/`
   - 资产与 scene 解析工具层

3. `/home/elm/projects/geomind/scene_construction/`
   - 离线造场景工具
   - 不应并入运行时 env 主链


### 4.2 应降级为历史参考的旧目录

1. `/home/elm/projects/geomind/envs/`
2. `/home/elm/projects/geomind/collection/`
3. `/home/elm/projects/geomind/scripts/`
4. `/home/elm/projects/geomind/geomind_tasks/`
5. `/home/elm/projects/geomind/geomind_skillgen/`

这些目录的问题是：

- 与 IsaacLab 原生接口不够对齐
- 存在多套并行运行链
- 存在 `__path__.append(...)` 这种混包方式
- 后续继续在上面开发会放大维护成本


## 5. 当前迁移思路

### 5.1 已确定的原则

1. `geomind_lab` 是主代码包
2. `IsaacLab` 是依赖，不是业务代码容器
3. 运行时逻辑以 `EnvCfg`、`SceneCfg`、`MimicEnv` 为中心
4. `scene_construction` 保持离线工具身份
5. 不再引入包路径 hack
6. 后续优先对接 IsaacLab 官方工具链：
   - `record_demos.py`
   - `replay_demos.py`
   - `generate_dataset.py`


### 5.2 当前阶段的目标

当前不是“大清理所有旧代码”，而是先把一个可持续演进的新骨架立起来。

因此现在的状态是：

- 先迁基础 env cfg / scene cfg / mdp / franka cfg
- 先让新包内部引用自洽
- 先建立 Mimic 接口骨架
- 运行入口、动态注册、端到端测试留到下一阶段


## 6. 还没完成的事情

下面这些仍然需要继续开发：

1. `IsaacLab/` 还没复制到当前新仓库根目录
2. 还没有 `gym.register(...)` 的任务注册入口
3. `geomind_lab/tasks/.../config/franka/task_registry.py` 仍然只是占位 dataclass
4. CLI 还没有真正连接：
   - `parse_env_cfg`
   - `gym.make`
   - `record_demos`
   - `replay_demos`
5. 还没有真实跑通：
   - 单任务 env 创建
   - teleop collect
   - replay
   - mimic generate
6. episode 级字符串元数据还没设计正式导出方案
   - 例如 instruction、scene_id、layout_key、task_category
   - 可能需要 sidecar JSON 或自定义 dataset handler


## 7. 下一步推荐开发顺序

建议在新窗口里按下面顺序继续：

1. 把 `IsaacLab/` 拷到当前目录下
2. `pip install -e .`
3. 新建 GeoMind 的任务注册文件
   - 建议先只注册一个固定 env
4. 用最小任务测试：
   - `scene_id=1`
   - `task_layout_key=line_3`
   - `task_index=0`
5. 跑通：
   - `gym.make(...)`
   - env reset
   - env step
6. 再接 `record_demos.py` 风格的数据采集
7. 再接 replay
8. 最后再扩展 task registry 和多任务动态注册


## 8. 最小可运行优先级

如果只做最短路径，请先完成这四件事：

1. 注册一个固定 `FrankaGeoMindIKRelVisuomotorEnvCfg`
2. 确认 `GEOMIND_ASSETS_ROOT` 和 scene USD 路径可用
3. `gym.make()` 能正常 reset / step
4. 让 IsaacLab 官方 `record_demos.py` 能录一条 demo

只要这四步通了，后面的 replay、registry、mimic 都是继续往上堆，不会再陷入“目录很多但跑不起来”的状态。


## 9. 开发注意事项

1. `task_specs.py` 当前默认会找：
   - `assets/tasks/annotated_tasks.json`
   - `assets/tasks/annotated_tasks_1.json`
   - 根目录 `annotated_tasks_1.json`

2. `franka_env_cfg.py` 当前默认会找：
   - `assets/assets_catalog_with_metadata.json`

3. `geomind_scene_cfg.py` 当前默认假定：
   - `GEOMIND_ASSETS_ROOT/stage/<scene_id>/...`

4. `mimic/configs/geomind_franka_ik_rel_mimic_env_cfg.py`
   - 当前只自动构造最简单的 grasp/place 两段子任务
   - 对复杂任务要继续重写

5. `scene_construction/`
   - 继续保留为离线工具
   - 不要重新耦合回运行时 env


## 10. 一句话总结

当前 `geomind_lab` 已经从旧仓库里迁出了一条更干净的 IsaacLab-compatible 主线：

- scene USD -> scene profile -> task spec -> env cfg -> franka cfg -> visuomotor cfg -> mimic wrapper

但还处在“代码骨架已经建立、运行入口尚未打通”的阶段。接下来的正确方向不是回旧根目录修脚本，而是在这个新包里继续完成注册、运行、采集和回放。
