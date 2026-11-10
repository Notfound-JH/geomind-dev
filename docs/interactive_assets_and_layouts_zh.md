# 可交互物体与摆放支持清单

2026-11-09。范围为当前资产目录、桌面资产文件、原场景中的便携杯盆与运行时几何。清单共39条：25类目录登记资产、4类原场景便携摆件、6类未登记文件、4类程序化形状。不把桌椅柜、机器人和同一模型的旋转版本算作独立可交互物体。

文件存在、被列为候选或可生成布局，不代表已逐资产验证刚体、碰撞、抓取或稳定放置。本次只做源码/文件核查和CPU随机化测试，没有启动物理任务或外部模型。完整路径、来源及范围见[Excel清单](../outputs/interactive_assets_layouts_v1/interactive_assets_and_layouts.xlsx)、[JSON](../outputs/interactive_assets_layouts_v1/inventory.json)、[资产CSV](../outputs/interactive_assets_layouts_v1/interactive_assets.csv)、[摆放CSV](../outputs/interactive_assets_layouts_v1/layout_support.csv)。

## 1. 已登记的小物体与桌面容器

| 类别 | 资产ID | 物品 |
|---|---|---|
| 罐类 | `blue_can` | 蓝色咖啡罐 |
| 罐类 | `red_can` | 红色番茄汤罐 |
| 罐类 | `low_can` | 矮金枪鱼罐 |
| 罐类 | `cube_can` | 方形肉罐 |
| 罐类 | `glass_can` | 玻璃罐 |
| 罐类 | `green_tin` | 绿色罐 |
| 瓶类 | `yellow_bottle` | 黄色芥末瓶 |
| 瓶类 | `white_bottle` | 白色清洁剂瓶 |
| 盒类 | `red_box` | 红色饼干盒 |
| 盒类 | `brown_box` | 棕色布丁盒 |
| 其他 | `rubiks_cube` | 魔方 |
| 餐具 | `plate_1`, `plate_2` | 两种盘子 |
| 餐具 | `glass_cup` | 玻璃杯 |
| 水果 | `banana`, `apple`, `lime`, `pomegranate`, `avocado` | 香蕉、苹果、青柠、石榴、牛油果 |
| 容器 | `bowl_1`, `bowl_2` | 两种碗 |
| 容器 | `basket_1`, `basket_2`, `basket_3` | 三种篮子 |
| 容器 | `bin` | KLT周转箱 |

当前目录中19类属于 `catch_object`，6类属于桌面 `container`。`container` 分类中的5个柜子未计入本清单。

35条文件资产候选均找到本机文件。其中 `glass_cup`、`apple`、`lime`、`pomegranate`、`avocado` 的原目录配置仍是旧磁盘路径，本机已有对应的 NVIDIA 模型；表中给出原路径及本机路径，没有改写原目录。`basket_1` 另有 with_collision/no_collision 等版本。`bin` 源文件名含 visual，其碰撞能力应另行核验。

## 2. 补充候选

| 范围 | 资产ID | 当前状态 |
|---|---|---|
| 原场景便携摆件 | `cup_1`, `cup_2`, `pot_1`, `pot_2` | 原场景和资产常量中存在；需要单独确认交互、刚体和碰撞配置 |
| 本地补充文件 | `basket_4`, `box_1`, `box_2`, `box_3`, `blue_tin`, `tin` | objects目录有文件，未在当前目录登记；名称按文件名，几何重复性和尺寸尚未核验 |
| 程序化几何 | `primitive_cuboid`, `primitive_sphere`, `primitive_cylinder`, `primitive_cone` | 代码可生成，无独立USD文件；具体执行链支持仍依任务入口 |

`*_RZ90` 是同一模型的姿态配置，不额外计数。旧摆放器默认随机池仅7种：

- `OBJECT_ALL`：`BLUE_CAN`, `RED_CAN`, `LOW_CAN`, `RUBIKS_CUBE`。
- `OBJECT_X/OBJECT_Y` 增加：`YELLOW_BOTTLE`, `WHITE_BOTTLE`, `CUBE_CAN`；Y池为对应RZ90姿态。

`red_box`、`brown_box` 虽在原场景中用过且有配置常量，目前在默认池中被注释，不会自动抽到。其他资产也不能因文件存在就假设默认会采样。

## 3. 结构化与随机摆放

### 旧资产摆放器

入口：[arrange_object.py](<../scene_construction /arrange_object.py:219>)，处理 USDAsset 模型。

| 参数 | 结构 | 细节 |
|---|---|---|
| `Line` | 直线 | 自动沿桌子长轴排列 |
| `Grid` | 二维网格 | 自动分行列，最后一行可不满 |
| `Tshaped` | T形 | 完整结构固定5点，入口根据桌子长边选择朝向 |
| `Ring` | 等角环形 | 半径不超过安全半宽、半长和0.2米 |
| `TwoType` | 两条平行线/两组 | 每组数量单独传入；两次资产抽样可能抽到同类 |
| `Random` | Poisson随机散点 | 默认半径r=0.08米，实际最小中心距离2r=0.16米 |

随机项包括资产类型、资产在坐标上的顺序、XY位置噪声、资产预设朝向、角度噪声和随机翻转。默认XY噪声±0.02米，角度噪声π/24；Random在自动interval分支关闭额外XY噪声。桌面尺寸、中心、间距和offset可配置，自动估算15厘米边缘。

限制：T形超过5个物体会索引越界；TwoType需要两组数量；Poisson拥挤时可能采不满且上层没有补齐处理；中心间距/边缘估算不是网格碰撞验证。T形实现还有offset重复叠加。目录名末尾有空格，而部分旧导入按不带空格路径编写，直接运行前需适配旧依赖与导入路径。

`arrange_object_new.py` 是另一旧分支，仅 Line/Grid/Tshaped/Random/TwoType，没有Ring；Random区域固定±0.2米，Grid硬编码白瓶并覆盖传入资产。不能因为文件名含new就当作更完整入口。

### 静态方块展示入口

入口：[render_static_arrangement_scene.py](../tests/render_static_arrangement_scene.py:77)，主要处理运行时注入、有碰撞配置的彩色长方体。

| 参数 | 结构 | 限制 |
|---|---|---|
| `line` | 直线 | X/Y轴可选 |
| `matrix` | 矩阵 | 仅2×2、2×3、3×2、3×3 |
| `t_shape` | T形 | CLI强制5块 |
| `circle` | 圆环 | 半径可指定或推导 |
| `basket_center` | 篮子中心摆放 | 加固定basket_1，在篮内近中心排列块 |
| `stack_center` | 垂直堆叠 | 共用XY，可控制上层缩小比例；跳过XY噪声 |

支持seed、数量范围、间距比例噪声和每块XY噪声。matrix数量范围模式在上述4种行列结构中抽样，不按任意min/max；T形仍固定5块。颜色为差异HSV色或统一色，不是随机颜色抽样。该入口不是通用的19种物体资产替换器。

### 混合随机场景脚本

入口：[render_random_scene.py](../scripts/render_random_scene.py:61)。CLI有 `auto / line / grid / t_shape / l_shape / circle / scatter / stack / container`。

- 随机选择USD模型、彩色外观块或混合模式，CLI没有直接固定模式选项。
- stack的USD模式仅LOW_CAN，其余模式堆方块；container为专用篮子加黄色瓶分支。
- scatter的自有分支独立均匀抽点，无Poisson最小距离保证。
- 彩色块为VisualCuboid，不能与静态入口中的交互刚体块混同；混合模式分支使用不同坐标生成逻辑，整组结构不一定统一。
- 当前预期同级目录 `/home/elm/projects/SiFERbench` 不存在，代码入口保留，但不能列为当前开箱可跑。

## 4. 当前主线结构保持随机化

入口：[layout_randomizer.py](../geomind_lab/tasks/manager_based/manipulation/geomind/layout_randomizer.py:1096)。它修改已注入的 runtime_objects 的XY布局，不负责随机资产种类、数量、质量、摩擦或物体自身朝向。

| 策略 | 保持/改变的关系 |
|---|---|
| `rank_line` | 改间距，保留一维顺序 |
| `midpoint` | 改锚点位置、距离和角度，保持中点定义 |
| `point_symmetry` | 改中心与参考向量，保持点对称 |
| `extension` | 改基点、步长和角度，保持共线延伸 |
| `square` | 改位置、边长和朝向，保持正方形 |
| `trend` | 平移、旋转和等比缩放，保留趋势残差 |
| `grid` | 改行列间距和起点，保持轴对齐及行列身份 |
| `free` | 刚体组自由小幅XY扰动 |
| `spatial_free` | 特定方向/容纳/支撑放置任务在工作区内重新采样 |
| `frozen` | 相对几何不变，允许整体平移 |

两阶段默认均关闭，seed独立：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `GEOMIND_LAYOUT_JITTER_ENABLE` | 0 | 开结构感知扰动 |
| `GEOMIND_LAYOUT_JITTER_XY` | 0.02米 | 扰动幅度参数，具体解释依策略 |
| `GEOMIND_LAYOUT_SHIFT_ENABLE` | 0 | 开整体平移 |
| `GEOMIND_LAYOUT_SHIFT_MAX` | 0.15米 | 整体平移幅度 |
| `GEOMIND_LAYOUT_JITTER_SEED`、`GEOMIND_LAYOUT_SHIFT_SEED` | 未设时随机 | 两阶段独立种子 |
| `GEOMIND_LAYOUT_MIN_CLEARANCE` | 0.025米 | 近似平面足迹间距 |
| `GEOMIND_LAYOUT_BOUNDS` | 0.15,0.68,-0.32,0.34 | x_lo,x_hi,y_lo,y_hi，需按具体场景重设 |
| `GEOMIND_LAYOUT_MAX_ATTEMPTS` | 256 | 有界拒绝采样次数 |

容器底板与墙体作为刚体组移动。检查边界、近似间距与派生目标点，但不等于完整PhysX接触验收。结构不匹配或采样失败时可能保留原布局或仅做整体平移，以实际返回的 `applied_layout_randomization` 为准。默认工作区不能直接当作全部102张桌面的可用区域。

CPU预览示例：

```bash
python scripts/preview_layout_randomization.py \
  --layout-key place_inside_scene65_skill \
  --jitter --shift --jitter-seed 7 --shift-seed 9
```

本轮已执行 `python -m unittest discover -s tests -p test_layout_randomizer.py`，3项通过。它验证有限的CPU布局性质，不代表以上全部结构、全部资产或所有物理场景已经验证。
