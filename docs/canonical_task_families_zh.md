# 最终 Canonical Task Family 分类

配置中的 family 使用 CamelCase；下表使用对应的展示名称。

新增 domain：前四类属于 `SpatialReasoning`（21 个任务），后两类属于 `SpatialMemory`（6 个任务）。
纯任务定义位于 `geomind/configs/task_definitions/`；场景、布局和实例参数不存放在任务定义中。

| Family | Canonical Task IDs | 数量 |
|---|---|---:|
| Target Identification | `SelectByOrder`, `SelectByGridIndex`, `SelectByBranchPosition`, `SelectByRegion` | 4 |
| Relational Grounding | `SelectByDirection`, `SelectByDistance`, `SelectByAdjacency`, `SelectByPathDistance`, `PlaceByTopology` | 5 |
| Constraint Composition | `SelectWithinSubset`, `SelectWithVisualCue`, `SelectAfterExclusion`, `SelectByTwoRelations`, `SelectByIndirectReference` | 5 |
| Geometric Inference | `InferMidpoint`, `InferPointSymmetry`, `ReflectAcrossLine`, `ExtendEqualSpacing`, `ProjectOntoLine`, `CompleteSquare`, `ContinueSpatialPattern` | 7 |
| Scene Updating | `DetectSceneChange`, `UpdateObjectOrder`, `UpdateEmptySlot` | 3 |
| Identity Tracking | `TrackMovingObject`, `TrackObjectSwap`, `TrackObjectOrigin` | 3 |
| 合计 | | **27** |

`SelectWithVisualCue` 为正式拼写。此分类同时覆盖已实现与占位任务，具体 condition、variant 和接口不重复计数。

公开 Family 由 `geomind/core/taxonomy.py` 维护；`runtime.protocol_family` 继续负责底层 oracle 分发。
