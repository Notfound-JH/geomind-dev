# v3 配置迁移与三个自动生成任务的验证

验证日期：2026-11-05。CPU 语义后端，Python 3.13.9。
本文件记录配置迁移与语义验证阶段。后续已完成三个任务的真实无头物理执行，见 [物理报告](headless_physics_verification_zh.md)。

## 配置与系统改动

- 27 个纯任务定义：`geomind/configs/task_definitions/`。21 个 SpatialReasoning、6 个 SpatialMemory。
- 59 个旧注册实例：34 个 Standard、25 个 Vla，均转换为精简 v3。
- 活跃注册配置移除 runtime、canonical_task_id、legacy_task_id、protocol_family 和 evaluator_key。
- 旧执行记录隔离到 `legacy/registrations/`，由 `core/episode_compat.py` 适配。
- 独立布局共 46 个，其中 40 个保留旧物理初始化、6 个为新生成原生布局。
- 旧物理初始化不包含任务列表、GT、preview_proxy_goal、任务约束或角色；后端从新布局文件读取物理配置。
- 旧协议标注移入 `legacy/layouts/standard.json`、`vla.json`，依赖工具同步改用新位置。
- 任务映射增加 domain，并把 SceneUpdating、IdentityTracking 的 dimension 归入 spatial_memory。
- 新增原生阶段控制、观测过滤、历史快照、移动事件、语义 GT 与回答评估。
- 新脚手架先在临时目录验证，再发布正式 episode；未知参数或同名覆盖被拒绝。

旧执行协议没有被宣称为原生实现。其参数与坐标约定仍由隔离的兼容记录维护，
原生求解器不会把旧坐标、旧默认值或协议 ID 当作新参数使用。

## 自动选择的三个任务

原注册记录均标记 implementation_status=placeholder。保留物理占位状态，新增语义能力：

| 任务 | Domain | 唯一语义 | 自动生成实例数 |
|---|---|---|---:|
| DetectSceneChange | SpatialMemory | 比较 observe.End 与当前状态，恰好一个物体中心位移超过阈值 | 3 |
| ProjectOntoLine | SpatialReasoning | 物体中心在无限三维直线上的正交投影 | 3 |
| ReflectAcrossLine | SpatialReasoning | 绕无限三维直线的轴对称点，定义为 2×投影−原点 | 3 |

seed 为 0、1、2。生成器同时输出场景、布局、任务引用、参数、协议、响应类型、评分容差和 suite。
memory 与 reasoning 分别有独立 suite 清单。

## 正确性证据

9/9 个生成实例通过：

1. 严格字段、类型、引用、frame、物体身份与时序校验。
2. 独立构造 oracle 与运行时 GT 比对；几何点的最大距离误差为 `6.55e-17 m`。
3. 正确答案被接受，刻意构造的错误答案被拒绝。
4. 保存状态与历史，再离线求解，得到相同 GT 和 provenance。
5. 运行 reset、观察、延迟、事件、查询、回答至终止的完整语义流程。

memory 只在观察和查询阶段提供物体状态，延迟阶段返回空对象集合；
查询前不提供具体变化对象，公开观测不包含 GT、历史快照或事件计划。

额外测试覆盖退化直线、线段外投影、端点交换、反射两次还原、没有变化、多个变化、
变化阈值边界、错误坐标系、缺失历史、跨运行历史、提前回答、reset 清理、错误参数不发布等情况。

全套 `unittest`：**179 项通过**。59 个注册 ID 可读取，48 个旧可执行项全部通过环境配置解析。
VLA 再生成也在临时仓库中验证，输出仍是 59 个有效 v3 注册项，不把 runtime 字段写回活跃配置。

## 查看与复现

```bash
python -m geomind.tools.generate_spatial_suite --seeds 0 1 2 --verify
python -m unittest discover -s tests -p 'test_*.py'
```

三个主要实例：

- `geomind/configs/episodes/DetectSceneChange_v0.json`
- `geomind/configs/episodes/ProjectOntoLine_v0.json`
- `geomind/configs/episodes/ReflectAcrossLine_v0.json`

清单：`geomind/configs/suites/generated_spatial_suite.json`。
独立 domain 清单：`generated_spatial_memory.json`、`generated_spatial_reasoning.json`。
逐实例报告：`geomind/configs/reports/generated_spatial_suite.json`。
状态证据：`geomind/configs/evidence/`。

```bash
python -m geomind.tools.resolve_episode DetectSceneChange_v0 \
  --root geomind/configs \
  --state geomind/configs/evidence/DetectSceneChange_v0.json
```

suite 中的 expected_answer 和 evidence 是评估端数据，不能作为 agent 输入。
配置哈希覆盖配置内容与 episodes 包实现；生成器自身版本以哈希记录。修改实现后须重新采集证据。

## 验证边界

以上结果验证参数读取、CPU 世界状态、时序记忆协议、语义 GT 和答案评分。
这份 CPU 报告不覆盖物理执行，明确保存 `backend: semantic`、`physical_execution_verified: false`。
后续原生 PhysX/Franka 的报告独立保存。旧兼容占位状态不用于判断原生后端的实现能力。
