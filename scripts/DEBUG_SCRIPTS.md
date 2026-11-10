# Debug Scripts Inventory

当前仓库顶层 `scripts/` 目录下共有 41 个 `debug_*` 相关文件：

- 28 个任务级 wrapper：`debug_*.sh`
- 5 个协议层 shell 入口：`debug_protocol_*.sh`
- 5 个协议层 Python 入口：`debug_protocol_*.py`
- 2 个统一入口 / 辅助工具：`debug_task.sh`、`debug_new_registry_tasks.py`
- 1 个 registry：`debug_task_registry.json`

其中 `debug_task_registry.json` 当前登记了 33 个可分发入口：

- 28 个任务 wrapper
- 5 个协议层 shell 后端

## 快速查看命令

```bash
# 按 registry 列出所有 canonical task_id
bash scripts/debug_task.sh --list

# 按任务族查看
bash scripts/debug_task.sh --list-categories

# 直接查看所有 debug 相关文件
find scripts -maxdepth 1 -type f -name 'debug_*' | sort
```

## 分层说明

- `debug_task.sh`
  统一入口。负责 alias -> canonical `task_id` -> 实际 wrapper 脚本的分发。
- `debug_task_registry.json`
  `debug_task.sh` 的 registry，只记录 shell 入口，不记录 Python 协议实现。
- `debug_*.sh`
  任务级 wrapper。通常负责选择默认 `SCENE_ID` / `LAYOUT_KEY` / `TASK_VERSION` 或 `PRESET`，然后委托给对应 `debug_protocol_*.sh`。
- `debug_protocol_*.sh`
  协议层 shell 入口。通常负责 conda、缓存目录、输出路径、AppLauncher 参数拼装。
- `debug_protocol_*.py`
  真正的 IsaacLab / GeoMind 调试执行入口，直接跑 annotation-backed rollout、agent baseline、summary 生成等。
- `debug_new_registry_tasks.py`
  面向“新 canonical task_id”的 smoke probe / scene-10 planner，不走 `debug_task.sh` registry 分发。

## 统一入口 / Registry / 辅助工具

| 文件 | 类型 | 作用 |
| --- | --- | --- |
| `scripts/debug_task.sh` | 统一入口 | 列表展示、alias 解析、按 `task_id` 分发到实际 wrapper |
| `scripts/debug_task_registry.json` | registry | 维护 `id`、`category`、`entry`、`backend` 映射 |
| `scripts/debug_new_registry_tasks.py` | 辅助调试器 | 针对新 registry 任务做 scene-10 smoke plan、manual task spec 构造与真实执行 |

## 任务级 Wrapper 清单

### Anchor Reference

常见开关：`TASK_VERSION=base|stress`

| 脚本 | task_id | 用途 | 后端 |
| --- | --- | --- | --- |
| `scripts/debug_grasp_anchor_direction.sh` | `grasp_anchor_direction` | 按 anchor 方位关系选择并抓取目标 | `debug_protocol_anchor_reference.sh` |
| `scripts/debug_grasp_anchor_distance.sh` | `grasp_anchor_distance` | 按 anchor 距离关系选择并抓取目标 | `debug_protocol_anchor_reference.sh` |

### Spatial Placement

常见开关：`TASK_VERSION=base|skill|stress` 或 `TASK_VERSION=control|stress`

| 脚本 | task_id | 用途 | 后端 |
| --- | --- | --- | --- |
| `scripts/debug_place_directional.sh` | `place_directional` | 相对 anchor 的方向区域放置 | `debug_protocol_spatial_placement.sh` |
| `scripts/debug_place_on_top.sh` | `place_on_top` | 物体顶面堆叠 / 放置 | `debug_protocol_spatial_placement.sh` |
| `scripts/debug_place_inside.sh` | `place_inside` | 容器内部放置 | `debug_protocol_spatial_placement.sh` |
| `scripts/debug_place_inside_boundary_control.sh` | `place_inside_boundary_control` | 容器边界控制版放置 | `debug_protocol_spatial_placement.sh` |

### Constraint Selection

常见开关：`TASK_VERSION=base|stress` 或 `TASK_VERSION=control|stress`

| 脚本 | task_id | 用途 | 后端 |
| --- | --- | --- | --- |
| `scripts/debug_select_visual_spatial.sh` | `select_visual_spatial` | 视觉 + 空间约束联合筛选 | `debug_protocol_constraint_selection.sh` |
| `scripts/debug_select_exclusion_unique.sh` | `select_exclusion_unique` | 排他约束下的唯一目标筛选 | `debug_protocol_constraint_selection.sh` |
| `scripts/debug_select_visual_spatial_control.sh` | `select_visual_spatial_control` | `select_visual_spatial` 的 control 版本 | `debug_protocol_constraint_selection.sh` |

### State Reindex

常见开关：`PRESET=standard|stress|skill`

| 脚本 | task_id | 用途 | 后端 |
| --- | --- | --- | --- |
| `scripts/debug_reindex_extreme.sh` | `reindex_extreme` | 极值位置刷新 / leftmost-rightmost 类更新 | `debug_protocol_state_reindex.sh` |
| `scripts/debug_reindex_insert.sh` | `reindex_insert` | 插入后重排的状态刷新 | `debug_protocol_state_reindex.sh` |
| `scripts/debug_reindex_subset.sh` | `reindex_subset` | 子集重排后的状态刷新 | `debug_protocol_state_reindex.sh` |
| `scripts/debug_reindex_neighbor.sh` | `reindex_neighbor` | 邻近关系驱动的状态刷新 | `debug_protocol_state_reindex.sh` |
| `scripts/debug_ordinal_insert_static.sh` | `ordinal_insert_static` | 静态 1D 序数插入 grounding | `debug_protocol_state_reindex.sh` |
| `scripts/debug_grid_rowcol_grounding.sh` | `grid_rowcol_grounding` | 2D 网格行列 grounding | `debug_protocol_state_reindex.sh` |
| `scripts/debug_reindex_neighbor_control.sh` | `reindex_neighbor_control` | `reindex_neighbor` 的 control 版本 | `debug_protocol_state_reindex.sh` |
| `scripts/debug_reindex_subset_control.sh` | `reindex_subset_control` | `reindex_subset` 的 control 版本 | `debug_protocol_state_reindex.sh` |

### Identity Tracking

常见开关：`PRESET=standard|stress`

| 脚本 | task_id | 用途 | 后端 |
| --- | --- | --- | --- |
| `scripts/debug_track_single.sh` | `track_single` | 单步身份追踪 | `debug_protocol_state_reindex.sh` |
| `scripts/debug_track_multi.sh` | `track_multi` | 多步身份追踪 | `debug_protocol_state_reindex.sh` |
| `scripts/debug_track_clutter.sh` | `track_clutter` | 带静态干扰物的鲁棒身份追踪 | `debug_protocol_state_reindex.sh` |
| `scripts/debug_track_single_control.sh` | `track_single_control` | `track_single` 的 control 版本 | `debug_protocol_state_reindex.sh` |

注：`debug_track_clutter.sh` 在 wrapper 内部会把实际任务切到 `track_clutter_base` / `track_clutter_stress`。

### Geometric Inference

常见开关：`PRESET=...` 或 `TASK_VERSION=base|skill|control|stress`

| 脚本 | task_id | 用途 | 后端 |
| --- | --- | --- | --- |
| `scripts/debug_perfect_midpoint.sh` | `perfect_midpoint` | 完美中点推断 | `debug_protocol_geometric_inference.sh` |
| `scripts/debug_isometric_extension.sh` | `isometric_extension` | 等距延展 / extension | `debug_protocol_geometric_inference.sh` |
| `scripts/debug_point_symmetry.sh` | `point_symmetry` | 点对称推断 | `debug_protocol_geometric_inference.sh` |
| `scripts/debug_trend_continuation.sh` | `trend_continuation` | 趋势延续 | `debug_protocol_geometric_inference.sh` |
| `scripts/debug_weak_destruction_control.sh` | `weak_destruction_control` | 趋势任务的弱破坏控制版 | `debug_protocol_geometric_inference.sh` |
| `scripts/debug_square_completion.sh` | `square_completion` | 方形补全 | `debug_protocol_geometric_inference.sh` |
| `scripts/debug_square_completion_decoy_control.sh` | `square_completion_decoy_control` | 方形补全的 decoy control 版本 | `debug_protocol_geometric_inference.sh` |

## 协议层后端清单

### Shell 入口

| 脚本 | 服务任务族 | 作用 |
| --- | --- | --- |
| `scripts/debug_protocol_anchor_reference.sh` | anchor reference | 准备 conda / 输出目录 / 相机参数，并调用 Python runner |
| `scripts/debug_protocol_spatial_placement.sh` | spatial placement | 准备放置类 rollout 参数，并调用 Python runner |
| `scripts/debug_protocol_constraint_selection.sh` | constraint selection | 准备筛选类 rollout 参数，并调用 Python runner |
| `scripts/debug_protocol_state_reindex.sh` | state reindex + identity tracking | 准备 live debug / agent rollout 参数，并调用 Python runner |
| `scripts/debug_protocol_geometric_inference.sh` | geometric inference | 可单任务直跑，也可转调 `geometric_inference_primary_oracle/random/all` 基线 case |

### Python 入口

| 脚本 | 服务任务族 | 作用 |
| --- | --- | --- |
| `scripts/debug_protocol_anchor_reference.py` | anchor reference | annotation-backed 抓取任务调试、随机/Oracle baseline、summary/trajectory 输出 |
| `scripts/debug_protocol_spatial_placement.py` | spatial placement | directional / on-top / inside 放置评测与 summary 输出 |
| `scripts/debug_protocol_constraint_selection.py` | constraint selection | 候选物筛选、抓取执行、trajectory / summary 输出 |
| `scripts/debug_protocol_state_reindex.py` | state reindex + identity tracking | live debugger、suite 运行、agent episode、summary 与可选相机视频输出 |
| `scripts/debug_protocol_geometric_inference.py` | geometric inference | 几何规则解析、agent baseline、paired-control 元数据检查、suite summary 输出 |

## `debug_task.sh` 当前 alias

如果你更习惯短命令，可以直接用这些 alias：

| alias | canonical task_id |
| --- | --- |
| `anchor1` | `grasp_anchor_direction` |
| `anchor2` | `grasp_anchor_distance` |
| `place1` | `place_directional` |
| `place2` | `place_on_top` |
| `place3` | `place_inside` |
| `place3c` | `place_inside_boundary_control` |
| `filter1` | `select_visual_spatial` |
| `filter1c` | `select_visual_spatial_control` |
| `filter2` | `select_exclusion_unique` |
| `refresh1` | `reindex_extreme` |
| `refresh2` | `reindex_insert` |
| `refresh3` | `reindex_subset` |
| `refresh3c` | `reindex_subset_control` |
| `refresh4` | `reindex_neighbor` |
| `refresh4c` | `reindex_neighbor_control` |
| `refresh5` | `ordinal_insert_static` |
| `refresh6` | `grid_rowcol_grounding` |
| `track1` | `track_single` |
| `track1c` | `track_single_control` |
| `track2` | `track_multi` |
| `track3` | `track_clutter` |
| `structure1` | `perfect_midpoint` |
| `structure2` | `isometric_extension` |
| `structure3` | `point_symmetry` |
| `structure4` | `trend_continuation` |
| `structure5` | `weak_destruction_control` |
| `structure6` | `square_completion` |
| `structure6c` | `square_completion_decoy_control` |

## 常见定位方式

- 想找“某个任务应该从哪里进”：
  先看上面的任务级 wrapper 表，再决定是否直接跑 `debug_task.sh` 或具体 `debug_*.sh`。
- 想找“某个任务最终执行的是哪段 Python”：
  先看 wrapper 对应的 `debug_protocol_*.sh`，再顺着表跳到对应 `debug_protocol_*.py`。
- 想确认 registry 是否覆盖了某个新脚本：
  直接看 `scripts/debug_task_registry.json`，或者执行 `bash scripts/debug_task.sh --list`。
