# 场景 × 物体 × 摆放组装

入口 `python -m geomind.tools.assemble_scene` 把请求解析成三个独立维度：编号场景、同类物体及数量、布局结构。它生成任务的初始桌面环境并保存 USDA；可选的 PhysX 验收检查初始物体落稳。组装过程中没有机器人抓放动作。

组装后接任务生成与GT求解，见[组装场景与任务GT](assembled_tasks_zh.md)：agent选择场景和布局，读取实际落稳状态，调用已有原生求解器；[本地任务展示](http://127.0.0.1:8780/tasks.html)分别记录GT、初始布局物理状态和任务动作状态。

## 一句指令组装

```bash
python -m geomind.tools.assemble_scene \
  --instruction "put 4 blue cans in grid of 2*2 in scene 60" \
  --output outputs/assemblies/my_scene60_grid \
  --verify-physics
```

自然语言入口是明确的中英文语法，不调用外部语言模型。支持 `put/place/arrange <数量> <资产> in <结构> in scene <编号>`，例如：

```text
put 4 blue cans in grid of 2*2 in scene 60
place 6 red cans in grid of 2x3 in scene 65
put 3 yellow bottles in line in scene 60
put 5 blue cans in ring in scene 65
put 4 blue cans in random in scene 60 with seed 7
在场景60中，将4个蓝色罐摆成2×2网格。
```

用户原示例中的 `2* *2`、`2* \* 2` 也可识别，兼容 Markdown 转义；未知语句、多余动作、未知物体与布局会报错。结构化请求中的物体ID必须为规范资产名。

只检查解析，不创建文件或启动仿真：

```bash
python -m geomind.tools.assemble_scene \
  --instruction "put 4 blue cans in grid of 2*2 in scene 60" --parse-only
```

## JSON入口

```json
{
  "schema_version": 1,
  "scene_id": 60,
  "object": {"asset": "blue_can", "count": 4},
  "layout": {"type": "grid", "rows": 2, "cols": 2, "spacing_m": 0.12},
  "seed": 0
}
```

```bash
python -m geomind.tools.assemble_scene \
  --request configs/scene_assembly/scene65_redcans_grid2x3.json \
  --output outputs/assemblies/my_redcan_grid --verify-physics
```

未知字段、布尔值冒充数量、非有限间距、数量与网格尺寸冲突均拒绝。`schema_version` 默认1，`seed` 默认0。一次请求1种物体、1–64件；每次使用新输出目录，已有目录拒绝覆盖。

## 首版支持

| 维度 | 范围 |
|---|---|
| 场景 | `assets/scenes/tabletop_cleared_v1/` 中1–102号空桌面场景 |
| 物体 | 19种catch_object、5种有效尺寸容器，以及cup_1/2、pot_1/2，共28种可解析尺度描述；各资产能否实际稳定，以逐例物理检查为准 |
| grid | 显式rows×cols，数量必须完全匹配；未给形状时仅对平方数量推导正方网格 |
| line | 沿世界Y轴，JSON可设 `axis: "x"` |
| ring | 等角环形，JSON可设 `radius_m` |
| random | 有界拒绝采样，支持seed，采不满时报错，不减少数量 |
| t_shape | 固定5件物体 |

6种未登记文件候选（basket_4、box_1/2/3、blue_tin、tin）能被名称识别，但缺少可信尺度/姿态描述，组装前明确拒绝。`bin` 的旧目录尺度会产生毫米级周转箱，本轮也拒绝，避免静默使用错误尺度。未修改原资产目录。多种物体混排、TwoType、L形、堆叠、篮内结构、程序化形状、任意物体姿态和机器人执行不在本入口首版范围。

JSON中的 `spacing_m` 为中心间距，原规则模式默认按实际变换后的包络加2.5厘米间隙生成。ring使用保守圆包络。grid的rows沿世界+X、cols沿世界+Y，按行优先记录对象。原规则模式保持目录中的首个允许姿态；显式开启随机化后增加受限世界Z轴转角。场景名称可在清单中查询；自然语言入口当前用原编号选择场景。

## 布局随机化核对与后续接入（2026-11-09）

以下保留接入前的检查结论；目前已接入的功能见下一节。

用户反馈当前布局看起来较硬。源码核对确认：**本组装入口尚未沿用之前的随机化链路**。现有实现和生成结果保留，下面记录当前事实与后续工作，不能把待接入项当成已实现功能。

- 当前 `usd.py` 调用新建的 `geomind/scene_assembly/layouts.py::generate_positions`。grid、line、ring、t_shape采用确定的居中规则；只有random散点分支使用seed。改变seed不会改变这些结构化布局。
- grid默认中心间距为实际包络尺寸加2.5厘米间隙。已交付的60号场景4蓝罐2×2实例约为7.62厘米等间距，各物体采用目录首个允许姿态，没有朝向随机化；这解释了视觉上的整齐和重复。
- 旧摆放器 `scene_construction /arrange_object.py::arrange_objects` 有位置噪声及姿态随机化入口，默认位置噪声2厘米、角度噪声π/24；它没有接入新组装入口。
- 当前任务运行时的 `geomind_lab/tasks/manager_based/manipulation/geomind/layout_randomizer.py::randomize_runtime_object_layout` 是另一条结构保持随机化链路，jitter与整体shift默认关闭。其 `_plan_grid` 可分别改变行列间距（原间距乘0.85–1.18，并有最小间距约束）、调整原点和加入受限单元扰动，保留行列语义；它也没有接入新组装入口。该模块主要随机化XY位置，不负责资产自身朝向随机化。

后续接入应保留用户指定的2×2等结构和数量，复用已有结构保持随机化，补充可复现的seed、间距变化、整体偏移、受限位置扰动和允许范围内的物体朝向变化。随机化后的实际包络、桌面支持、物体间距及落稳需重新验证；既有规则布局的PhysX通过记录不能代替新随机布局的验收。本次仅保存检查结论，未修改布局代码或重新执行仿真。

原实例、失败记录、源码快照、录像和网页均保留，证据见[组装实验记录](research/experiments/2026-11-09-scene-assembly.md)。

## 已接入的随机化入口

上节保留接入前的检查记录。当前可以显式开启随机化，原有不带随机化的请求仍生成原规则布局：

```bash
python -m geomind.tools.assemble_scene \
  --instruction "put 4 blue cans in grid of 2*2 in scene 60 with randomization and seed 7" \
  --output outputs/assemblies/my_randomized_grid --verify-physics
```

也可给原指令追加CLI参数 `--randomize --seed 7`，或在JSON中增加：

```json
"randomization": {
  "enabled": true,
  "jitter_xy_m": 0.02,
  "shift_max_m": 0.15,
  "yaw_deg": 7.5
}
```

该片段是请求的一个字段；完整示例见 `configs/scene_assembly/scene60_bluecans_grid2x2_randomized_seed7.json`。存在randomization对象时默认开启并补全上述参数；省略对象时关闭。jitter范围0–5厘米、shift范围0–30厘米、yaw范围0–180度，所有数值必须有限；这些上限不代表所有资产都能在全部参数下通过验收。

grid实际复用原运行时 `randomize_runtime_object_layout` 的 `_plan_grid`：行列间距各自乘0.85–1.18，最小间距受0.02米量化尺度约束；网格原点在jitter范围内调整，每格XY噪声上限为min(jitter,5毫米)。整体平移用独立seed，最大幅度由shift和实际桌面边界共同约束。不任意旋转整个网格，保留行列语义。未指定spacing时，随机化模式先采用旧摆放器13厘米舒适间距（实际包络有更高要求则增大），放不下时使用包络加间隙的原最小间距；显式spacing作为随机化前的基准值。

line、ring、t_shape和random目前复用整体平移阶段，不添加局部XY噪声，避免把线、圆和T形打散；random本身仍按seed采样散点。各布局另外对物体施加±yaw_deg的世界Z轴转角，与目录首个姿态组合，不随机翻倒物体。位置、整体偏移和朝向使用相互独立的可复现seed。

导出时逐物体实测旋转后的USD包络和底高，用保守共同包络复查间距、家具和9点桌面支持；旧随机化没有有效应用或发生不支持的降级时明确失败。保存实际随机化记录、每件转角和行列标签；新结果需单独通过PhysX与离线复核，原规则版证据不复用作通过记录。

实际执行与展示记录见[随机化实验](research/experiments/2026-11-09-scene-assembly-randomization.md)。本地[组装展示页](http://127.0.0.1:8780/assembly.html)提供不同seed实例、原规则版与随机化版的实际近景对比及各次落稳录像。

## 几何和物理验收

- 引用已清理场景，任务物体写到 `/World/AssembledObjects`；原场景、空桌面USD及原物体资产均不改写。输出USD依赖原资产目录，不是完整独立资产包。
- 从实际桌子网格的近水平三角面选择桌面，优先命名的table_top；不把整桌AABB或显示器顶部直接当桌面。保存所选网格、边界、平面高度和面积。
- 使用目录scale与wxyz姿态，实测变换后的网格包络和底部偏移。保存标准translate/orient/scale，避免物理初始化丢失矩阵内尺度。
- 检查物体包络边距、相互间隙、中心/边中点/角点共9点的桌面三角形支持采样，及保留家具的保守AABB。采样不是任意凹桌面或孔洞的完整覆盖证明；无法通过检查的请求拒绝。
- 新物体是真实动态刚体，每个网格用凸包碰撞；背景已有刚体设为运动学，保持房间家具不因仿真落下。凸包可能封住容器凹腔，本入口不认证容器内部交互。
- `--verify-physics` 在本地Isaac Sim/PhysX运行20步预热、600步测量，dt=1/120秒；连续至少60步满足XY误差≤12毫米、底高误差≤6毫米、姿态误差≤0.12弧度、线速度≤1厘米/秒、角速度≤0.1弧度/秒及支持采样。
- 每步读取真实刚体姿态和速度，用实测四元数变换原始网格点重新计算底高和包络；reset后不瞬移物体或用外力维持。保存最终与峰值平面偏移，完整逐步状态、实际录像和全景/近景图。
- 离线复核使用独立的四元数向量表达式重算末态和稳定时长；将一个物体平移5厘米的离线错误布局对照必须拒绝。这个对照没有物理重执行，不计作物理负例。
- 每个worker最长420秒、GPU串行；失败保留。`--timeout-s` 可缩短上限，`--isaac-python` 可指定本机Isaac Python。没有 `--verify-physics` 时明确记录物理未执行。

## 产物和网页

输出目录包含 `scene.usda`、`request.json`、`manifest.json`、源代码快照与日志。物理检查另有 `object_states.npz`、`asset_body_vertices.npz`、`physics_report.json`、`offline_audit.json`、`settling.mp4`、`preview.png`、`detail_preview.png`。

状态区分 `assembled_physics_unverified`、`assembled_physics_failed`、`assembled_physics_passed`、`assembled_audit_failed`。物理或离线复核失败时命令返回非零；文件已生成不冒充物理通过。

命令完成后自动更新已有本地图鉴的[场景组装页](http://127.0.0.1:8780/assembly.html)。网页展示已生成实例与实际证据，并可复制复现命令；它不伪造浏览器内生成操作，也不上传到外部站点。网页服务器未启动时可运行：

```bash
python -m http.server 8780 --bind 127.0.0.1 --directory scene_atlas/dist
```

单测入口：

```bash
python -m unittest discover -s tests -p 'test_scene_assembly*.py'
```
