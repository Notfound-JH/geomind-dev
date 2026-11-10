# Rebuttal 指标汇总

> 数据来源：`run_eval_pi05_v4`（base, 168）、`run_eval_pi05_v4_direct`（direct, 81）、`run_eval_openpi_05_v3`（π0 base, 324）
> 计算入口：`scripts/rebuttal_metrics.py`（成功类）、`scripts/tte_recompute.py`（TTE 类）
> 逐 episode 明细：`analysis_outputs/tte_recompute_episodes.csv`
> 生成日期：2026-07-28

---

## 0. 一句话结论

镜像修复改善的是**执行/趋近**能力（reach、以及全部 xy 版 TTE 距离指标全线提升），
但**没有改善 grounding**（`target_closest_rate` p = 0.3840，无变化）；
而 direct 消融只改指令与目标显著性、不改场景，却单独把 `target_closest_rate`
从随机水平（0.2472，p = 0.4822）拉到显著优于随机（0.3765，p = 0.0066）。
因此 benchmark 上的失败可被定位到 **grounding** 这一步，而非执行能力不足。

⚠️ **关于论文已报 TTE 的重要更正，见 §6。** 简言之：论文报的 TTE（0.33–1.32）与本文档
早期误称的「TTE」（0.24–0.39 二元选中率）**是两个完全不同的量**；后者已重命名为
`target_closest_rate`，前者的绝对值因三重口径污染而不可用。

---

## 1. 指标体系

### 1.1 保留的指标

| 指标 | 定义 | 范围 | 角色 |
|---|---|---|---|
| `reach` | `evaluation.reach_success`，末端进入目标半径且满足 dwell | 全部任务 | **主指标（执行）** |
| `target_closest_rate` | 逐步：离末端最近的物体 == GT 目标物体，全回合平均 | 仅 `goal_kind == move_to` | **主指标（grounding）** |
| `tte_min_xy` | 全回合最小 $\lVert \text{goal\_error}_{xy}\rVert$ | 全部任务 | **主连续指标**（等价于 `sge_xy_best`） |
| `tte_mean_raw_xy` | 逐步 $\lVert \text{goal\_error}_{xy}\rVert$ 直接平均 | 全部任务 | 轨迹级误差（与论文 TTE 口径对齐） |
| `tte_mean_resampled_xy` | 按归一化时间重采样到 50 点后平均 | 全部任务 | 跨模型可比版本 |
| `tte_final20_xy` | 轨迹后 20% step 平均 | 全部任务 | 发散性诊断 |
| `min_goal_dist` | `evaluation.min_goal_distance`（3D 全回合最小） | 全部任务 | 辅助连续指标 |
| `success` | `evaluation.success`（完整任务成功） | 全部任务 | 仅作说明，见 §5.1 |

### 1.2 弃用的指标及理由

| 指标 | 弃用理由 |
|---|---|
| `sge_xy_final` / `tte_final20` 作为主指标 | 对消融**完全失效**：Δ = −0.0005, p = 0.9407；tte_final20_xy Δ = −0.0026, p = 0.6682 |
| `sge_3d_final`（原始 SGE，末步 3D 距离） | 同上失效（p = 0.8881）；且被悬停高度支配、**反转模型排名**，见 §5.2 |
| **所有 `*_3d` 版本的 TTE** | **符号会反转**：base→direct 在 xy 上是 −0.0056，加上 z 后变成 +0.0264 且“显著变差”，见 §4.3 |
| `target_closest_final`（末步版选中率） | 比逐步版不敏感：同一数据 p = 0.1793 vs 逐步版 p = 0.0022 |

**共同根因**：策略够到目标后会继续下压执行抓取。一方面**末步状态被抓取动作污染**，
另一方面 **z 方向的下压会抵消 xy 上的发散**。凡是“末步”型或含 z 的指标都会失效甚至反号，
凡是“全回合最优 + xy”型都能测到。这与 `reach_success`（能测到）vs `success`（测不到）是同一个机制。

> **写作要点**：端点类指标必须取 *best over episode*，且必须用 xy 而非 3D。

### 1.3 目标点 $g$ 的选取（差异根源，务必先读）

所有距离类指标都依赖目标点 $g$。本文档统一取 **环境自己的真值 goal**：

$$g_t = \text{eef}_t + \text{goal\_error}_t \qquad (\text{goal\_error 为 xy 二维})$$

这与 `reach` / `success` 判定用的是同一个点，因此整套数字内部自洽。

**不采用** `scripts/metrics_trajectory.py` 的做法（即论文已报 TTE 的来源），原因有三：

1. 它从 `legacy_layouts.json` 按名称取**标称位置**，用常数 `DEFAULT_TABLETOP_Z_M = 0.784` 补 z；
   实测物体 z 中位为 **0.9543**，凭空注入 **+0.17 m** 的常数误差。
2. 324 条里 **156 条**走 `static_layout`，**忽略物体在回合中被移动**（reindex / track 族物体是会动的）。
3. 同一列里混着二维和三维距离（`xy_only`：204 条 True / 120 条 False），再直接平均。

后果见 §6：论文已报 TTE 的绝对值被系统性放大 **3–4 倍**。

### 1.4 TTE 的四个版本

对第 $t$ 步，$d_t = \lVert g_t - \text{eef}_t \rVert$。四个版本全部计算，`_xy` 与 `_3d` 各一份：

| 字段 | 算法 | 用途 |
|---|---|---|
| `tte_mean_raw` | 全部 step 的 $d_t$ 直接平均 | 与论文口径对齐，但受 step 数影响 |
| `tte_mean_resampled` | 按归一化时间重采样到 50 点后平均 | 跨模型可比版本 |
| `tte_final20` | 轨迹后 20% step 平均 | 回答"是否起步阶段拉高了均值" |
| `tte_min` | 全轨迹 $d_t$ 最小值 | 最接近时刻，排除到位后下压污染 |

$d_{xy}$ 直接用 `goal_error`（环境真值，全 `goal_kind` 通用）。
$d_{3d} = \sqrt{d_{xy}^2 + (\text{eef}_z - g_z)^2}$，其中 $g_z$ 取离 goal 点最近物体的中心 z。

### 1.5 `target_closest_rate`（grounding 指标）

按 `scripts/metrics_trajectory.py` 原口径，逐步判定后全回合平均：

$$\text{rate} = \frac{1}{T}\sum_{t=1}^{T} \mathbb{1}\Big[\arg\min_i \lVert P_i^{(t)} - \text{eef}_t \rVert_{xy} = \arg\min_i \lVert P_i^{(t)} - g_t \rVert_{xy}\Big]$$

- 物体位姿来自 `trace.json[i].agent_obs.state.native_obs.policy.object`
  （长度 80 = 11 槽 × 7；前 `execution_binding.runtime_object_count` 个为真实物体，其余为填充，z < 0.1）。
- **不依赖物体被移动**，因此夹爪冻结不影响其有效性。
- 仅对 `goal_kind == move_to` 有定义。适用范围验证（目标点到最近物体的距离）：

| `goal_kind` | n | 中位距离 | <3 cm | 是否采用 |
|---|---|---|---|---|
| `move_to` | 159 | **0.0000** | **100%** | ✅ 目标点与物体完全重合 |
| `placement` | 45 | 0.0001 | 60% | ❌ 目标为区域 |
| `generic` | 36 | 0.0726 | 33% | ❌ 目标为区域 |
| `square_completion` | 9 | 0.0111 | 56% | ❌ 目标为区域 |

**随机基线**：逐场景 $1/n_{\text{objects}}$（物体数 3–7 不等，分布 `{3:9, 4:12, 5:18, 6:9, 7:3}`）。
二元版用**泊松二项精确分布**；逐步平均版用**对 $1/n_i$ 的单样本符号翻转置换检验**。

### 1.6 统计方法

| 数据类型 | 检验 |
|---|---|
| 配对二元 | McNemar 精确二项（双侧） |
| 配对连续 | 符号翻转置换检验，20000 次（双侧） |
| 对随机基线（二元） | 泊松二项精确分布（单侧） |
| 对随机基线（比率） | 单样本符号翻转置换检验，20000 次（双侧） |

**配对可证性**：种子由 `base_seed + task_index*100003 + level_index*1009 + trial_index` 派生。
direct 的 81 个种子（runs-per-level 1）是 base 种子（runs-per-level 2）的真子集。
`geomind/tasks/direct_ablation.py` 的 `apply_direct_override` 断言除指令文本与
目标重着色 / 视觉标记盘之外，场景、物体位姿、干预脚本、随机种子逐字节一致。

---

## 2. 主表 A：消融 base → direct

π0.5 + 镜像修正，同种子配对 **n = 81**

| 指标 | base | direct | Δ | 翻转 (D赢/B赢) | p |
|---|---|---|---|---|---|
| `reach` | 23.5% | **33.3%** | **+9.9** | 9 / 1 | **0.0215 ★** |
| `target_closest_rate` (n=51) | 0.2596 | **0.3765** | **+0.1169** | — | **0.0022 ★** |
| `success` | 3.7% | 7.4% | +3.7 | 4 / 1 | 0.3750 |
| `tte_min_xy` | 0.0765 | **0.0631** | **−0.0134** | — | **<0.0001 ★** |
| `min_goal_dist` | 0.0675 ± 0.0566 | **0.0579 ± 0.0545** | **−0.0096** | — | **0.0001 ★** |
| `tte_mean_raw_xy` | 0.1523 | 0.1467 | −0.0056 | — | 0.0552 |
| `tte_mean_resampled_xy` | 0.1523 | 0.1467 | −0.0056 | — | 0.0552 |
| ~~`tte_final20_xy`~~ | 0.1794 | 0.1768 | −0.0026 | — | 0.6682 |
| ~~`target_closest_final`~~ | 0.2745 | 0.3922 | +0.1176 | 10 / 4 | 0.1793 |
| ~~`sge_xy_final`~~ | 0.1847 ± 0.1095 | 0.1842 ± 0.1110 | −0.0005 | — | 0.9407 |
| ~~`sge_3d_final`~~ | 0.1749 ± 0.1188 | 0.1758 ± 0.1183 | +0.0009 | — | 0.8881 |
| ⚠️ `tte_mean_raw_3d` | 0.2402 | 0.2666 | **+0.0264** | — | **<0.0001 ★（反向）** |
| ⚠️ `tte_min_3d` | 0.1841 | 0.2100 | **+0.0259** | — | **<0.0001 ★（反向）** |

最后两行是 3D 版本反号的直接证据：**同一组数据，xy 上 direct 更好，加上 z 后变成“显著更差”**。
原因是 direct 的手臂悬得更高（末端 z 中位 1.112 vs base 1.095）。详见 §4.3。

### 2.1 reach 的稳健性检查

| 子集 | n | base | direct | 翻转 | p |
|---|---|---|---|---|---|
| 全部 27 任务 | 81 | 23.5% | 33.3% | 9 / 1 | **0.0215 ★** |
| 剔除 `unchanged` 任务 | 75 | — | — | — | **0.0391 ★** |
| 仅含视觉线索的 22 任务 | 66 | — | — | — | **0.0391 ★** |
| margin > 0（排除白送） | 77 | — | — | — | **0.0215 ★** |
| **margin > 0.03（真需移动）** | 54 | 13.0% | **27.8%** | 8 / 0 | **0.0078 ★** |
| 剔除 step ≤ 2 的即时成功 | 81 | 11.1% | 21.0% | — | **0.0215 ★** |

难度越高效应越强，方向一致。

### 2.2 按难度等级 / 起始 margin 分层（reach）

| 等级 | base | direct |
|---|---|---|
| loose | 29.6% | 44.4% |
| medium | 25.9% | 29.6% |
| strict | 14.8% | 25.9% |

| 起始 margin | n | base | direct | Δ |
|---|---|---|---|---|
| ≤ 0（白送） | 4 | 100.0% | 100.0% | +0.0 |
| 0 < m ≤ 0.03（擦边） | 23 | 34.8% | 34.8% | +0.0 |
| **> 0.03（真走）** | 54 | 13.0% | **27.8%** | **+14.8** |

### 2.3 direct 的改动机制分布（81 配对）

| 机制 | n | 梯度层级 | n |
|---|---|---|---|
| `recolor` | 51 | `instruction_plus_recolor` | 42 |
| `marker` | 24 | `instruction_plus_marker` | 24 |
| `none` | 6 | `instruction_only` | 9 |
| | | `unchanged` | 6 |

- `unchanged`：`RelationBaseBase_PlaceInside`、`RelationBaseBase_PlaceOnTop`
- `instruction_only`：三个 `*_ReindexExtreme` 变体，base 4/9 → direct 4/9，**无变化**（见 §5.4）

### 2.4 逐族 `target_closest_rate`（配对）

| family | n | base | direct | Δ |
|---|---|---|---|---|
| Target Identification | 6 | 0.0833 | **0.4933** | **+0.4100** |
| Constraint Composition | 6 | 0.2667 | 0.4500 | +0.1833 |
| Identity Tracking | 15 | 0.1827 | 0.3400 | +0.1573 |
| Scene Updating | 24 | 0.3500 | 0.3517 | +0.0017 |

Geometric Inference / Relational Grounding 的 `goal_kind` 为区域型，该指标无定义。

---

## 3. 主表 B：模型 π0 → π0.5 + 镜像

均为 base 条件，同种子配对 **n = 168**

| 指标 | π0 | π0.5+镜像 | Δ | 翻转 | p |
|---|---|---|---|---|---|
| `reach` | 16.1% | **23.2%** | **+7.1** | 20 / 8 | **0.0357 ★** |
| `target_closest_rate` (n=108) | 0.2852 | 0.2472 | −0.0380 | — | **0.3840（无变化）** |
| `success` | 1.2% | 4.8% | +3.6 | 8 / 2 | 0.1094 |
| `tte_min_xy` | 0.0868 | **0.0751** | −0.0118 | — | **0.0002 ★** |
| `tte_mean_raw_xy` | 0.1728 | **0.1530** | −0.0198 | — | **0.0003 ★** |
| `tte_mean_resampled_xy` | 0.1728 | **0.1530** | −0.0198 | — | **0.0002 ★** |
| `tte_final20_xy` | 0.2160 | **0.1811** | −0.0348 | — | **0.0001 ★** |
| `min_goal_dist` | 0.0760 ± 0.0519 | **0.0667 ± 0.0551** | −0.0093 | — | **0.0010 ★** |
| ⚠️ `tte_mean_raw_3d` | 0.2369 | 0.2405 | +0.0036 | — | 0.5042 |
| ⚠️ `tte_min_3d` | 0.1754 | 0.1843 | +0.0089 | — | 0.1313 |

**xy 版本四个全部显著改善，3D 版本全部不显著且符号相反。** 又一次说明 z 分量会抹掉真实效应。

### 3.1 按起始 margin 分层（reach）

| 子集 | n | π0.5+镜像 | π0 |
|---|---|---|---|
| 全部 | 168 | 23.2% | 16.1% |
| margin ≤ 0 | 10 | 100.0% | 90.0% |
| 0 < m ≤ 0.03 | 48 | 27.1% | 18.8% |
| m > 0.03 | 110 | 14.5% | 8.2% |

---

## 4. 目标选中率对随机基线：核心证据

随机基线 = 逐场景 $1/n_{\text{objects}}$（物体数 3–7）。

### 4.1 逐步版 `target_closest_rate`（主报口径）

| 运行 | 条件 | n | rate | 基线 | Δ | p（单样本符号翻转） | 判定 |
|---|---|---|---|---|---|---|---|
| π0 | base | 204 | 0.2684 | 0.2261 | +0.0424 | 0.0457 | 临界（见下方注意） |
| π0.5+镜像 | base | 108 | 0.2472 | 0.2246 | +0.0226 | **0.4822** | **与随机无显著差异** |
| π0.5+镜像 | **direct** | 51 | **0.3765** | 0.2261 | **+0.1504** | **0.0066** | **★ 显著优于随机** |

⚠️ **如实记录**：π0 的 p = 0.0457 刚好跨过 0.05，不能声称它“完全处于随机水平”；
且其 n = 204 大于 π0.5 的 n = 108，两者检验力不对等。稳妥的表述是：
**base 条件下两个模型都只略高于随机（Δ ≤ 0.04），direct 条件高出随机 0.15**。

### 4.2 末步二元版 `target_closest_final`（对照，不作主报）

随机基线（泊松二项）≈ **22.6%**

| 运行 | 条件 | 命中 | 率 | $P(\text{随机} \ge \text{观测})$ | 判定 |
|---|---|---|---|---|---|
| π0 | base | 28/108 | 25.9% | 0.2225 | 与随机无显著差异 |
| π0.5+镜像 | base | 26/108 | 24.1% | 0.3790 | 与随机无显著差异 |
| π0.5+镜像 | base | 14/51 | 27.5% | 0.2473 | 与随机无显著差异 |
| π0.5+镜像 | **direct** | 20/51 | **39.2%** | **0.0052** | **★ 显著优于随机** |

末步版在 base→direct 配对上 p = 0.1793（不显著），逐步版 p = 0.0022（显著）。
**逐步版更敏感，应作为主报口径**；末步版仅作稳健性对照。

### 4.3 3D 版本会反号（为何必须用 xy）

base → direct 配对 n = 81，同一组数据：

| 版本 | base | direct | Δ | p | 方向 |
|---|---|---|---|---|---|
| `tte_mean_raw_xy` | 0.1523 | 0.1467 | −0.0056 | 0.0552 | direct 更好 |
| `tte_mean_raw_3d` | 0.2402 | 0.2666 | **+0.0264** | **<0.0001** | **direct “显著更差”** |
| `tte_min_xy` | 0.0765 | 0.0631 | −0.0134 | **<0.0001** | direct 更好 |
| `tte_min_3d` | 0.1841 | 0.2100 | **+0.0259** | **<0.0001** | **direct “显著更差”** |

原因：direct 的手臂悬得更高（末端 z 中位 1.112 vs base 1.095），z 分量盖过了 xy 上的真实改善。
**任何含 z 的距离指标在本设置下都不可用。**

### 4.4 解读

1. **base 条件下，两个模型的目标选择都只略高于随机猜测（Δ ≤ 0.04）。**
2. **只有 direct 条件显著超过随机（Δ = +0.15, p = 0.0066）。**
   direct 不改场景、不改物体位姿、不改种子，只改指令文本与目标显著性
   —— 因此该差异只能归因于 grounding。
3. **镜像修复提升 reach（+7.1，p=0.0357）与全部 xy 距离指标，却不改变目标选中率（p=0.3840）。**
   这证明“执行/趋近”与“grounding”是**两个可分离的能力维度**，
   而 benchmark 的失败定位在后者。

### 4.5 目标选中与 reach 的交叉表（配对 n=51，末步二元版）

| 条件 | 选中且 reach 成功 | 选中但 reach 失败 | 未选中但 reach 成功 |
|---|---|---|---|
| base | 7 | 7 | 7 |
| direct | 10 | 10 | 5 |

"找对物体"与"走到位"是两个可分离的失败模式，可在论文中做二级归因。

---

## 5. 诊断与已知污染

### 5.1 自由度锁定（必须写进 Methods）

从 v4 trace 实测（2000 步 / 40 回合）验证：

- **旋转锁定**：`action[3:6]` 恒为 0.0000，std = 0。`eef_quat` 漂移中位 0.00804（≈1°）为 IK 跟踪误差。
- **夹爪锁定**：每回合内指令恒定（0/60 回合有变化）；`gripper_pos` 全程变化 < 9e-6，**物理上冻结**。

**影响**：`Relation*` 族（PlaceInside / PlaceOnTop / PlaceDirectional）需要抓取+放置，
在当前设置下**永远不可能真正成功**。因此：

> `reach_success` 必须作为主指标，且必须在 Methods 中声明。
> 本工作评估的是**空间目标定位**能力，**不能**声称评估了"放置能力"。

### 5.2 原始 3D SGE 的污染（说明为何弃用）

GT 的 z 由桌面常数 0.7552 填充，导致 3D 距离被**悬停高度**支配：

| 运行 | 末端 z 首→末 | dz（相对桌面） | 末步 xy 误差 | $\sqrt{xy^2+dz^2}$ | 实际 SGE |
|---|---|---|---|---|---|
| π0.5+镜像 base | 1.170 → 1.095 | 0.340 | 0.186 | 0.397 | 0.4146 |
| π0.5+镜像 direct | 1.176 → 1.112 | 0.357 | 0.184 | 0.411 | 0.4417 |
| π0 base | 1.171 → 1.033 | 0.278 | 0.226 | 0.367 | 0.3506 |

dz 是 xy 误差的 1.5–2 倍，**并且反转排名** —— π0 的 SGE 更小纯粹是因为它把手臂压得更低。

### 5.3 先验坍缩检验

各任务净移动方向与全局平均方向的余弦：

| 运行 | 任务数 | cos 中位 | 最小 | 偏离 (<0.9) | 全局平均方向 |
|---|---|---|---|---|---|
| π0 base | 27 | +0.977 | −0.809 | **5/27** | (−0.334, +0.943) |
| π0.5+镜像 base | 27 | +0.813 | +0.032 | **21/27** | (+0.998, +0.069) |
| π0.5+镜像 direct | 27 | +0.877 | +0.000 | 15/27 | (+0.998, +0.060) |

- π0 只有 5/27 任务方向可区分 —— 其排行榜实际在衡量"目标方向离 +Y 有多近"。
- 修复后坍缩被打破（21/27 可区分），但**残余偏置仍在**，只是从 +Y 转到 +X。
- **反例（如实记录）**：direct 的偏离任务数（15）少于 base（21），方向一致性反而更高，
  不支持"direct 让轨迹更多样"的说法。
- ⚠️ 存在先验坍缩时，"朝目标 cos"是**误导性**指标：π0 的 goal-directed cos 中位 +0.162
  高于 π0.5 的 −0.182，仅因为 π0 的恒定 +Y 恰好与许多目标方向对齐。

### 5.4 逐任务族结果

| family | n | reach base→direct | `target_closest_final` base→direct |
|---|---|---|---|
| Binding | 6 | 0.0% → 16.7% | **0.0% → 33.3%** (n=6) |
| Constraint | 6 | 33.3% → 50.0% | **0.0% → 50.0%** (n=6) |
| Geometric | 15 | **20.0% → 60.0%** | n/a（区域目标） |
| Reindex | 24 | 25.0% → 20.8% | 50.0% → 41.7% (n=24) |
| Relation | 15 | 13.3% → 20.0% | n/a（区域目标） |
| Track | 15 | 40.0% → 40.0% | 13.3% → 33.3% (n=15) |

- **Binding / Constraint 的 base 选中率恰好为 0**（完全没找对过物体），direct 升到 33% / 50%。
  消融效应集中在**需要语言指代解析**的族。
- **Reindex 是唯一负向的族**，且它包含全部 `instruction_only`（无视觉改动）任务，方向自洽。
- Geometric / Relation 的 `goal_kind` 为区域型，选中率无定义（但 TTE 距离指标仍可算）。

### 5.5 逐任务选中情况（move_to，末步二元版，base → direct）

```
ConstraintBaseBase_SelectExclusionUnique   0/3 → 3/3  ↑
TrackBaseVisual_TrackSingle                1/3 → 3/3  ↑
BindingAugBase_AnchorDistance              0/3 → 1/3  ↑
BindingBaseBase_AnchorDirection            0/3 → 1/3  ↑
TrackBaseBase_TrackSingle                  1/3 → 2/3  ↑
ConstraintBaseBase_SelectVisualSpatial     0/3 → 0/3
ReindexBaseBase_GridRowcolGrounding        0/3 → 0/3
ReindexBaseBase_OrdinalInsertStatic        3/3 → 3/3
ReindexBaseBase_ReindexInsert              3/3 → 3/3
ReindexBaseBase_ReindexNeighbor            0/3 → 0/3
ReindexBaseBase_ReindexSubset              0/3 → 0/3
ReindexBaseVisual_ReindexExtreme           2/3 → 2/3
TrackBaseBase_TrackClutter                 0/3 → 0/3
TrackBaseBase_TrackMulti                   0/3 → 0/3
TrackBaseSem_TrackSingle                   0/3 → 0/3
ReindexBaseBase_ReindexExtreme             2/3 → 1/3  ↓
ReindexBaseSem_ReindexExtreme              2/3 → 1/3  ↓
```

### 5.6 后段发散（`tte_final20 > tte_mean_raw` 的 episode 占比）

| 运行 | xy 版本 | 均值差 | 3D 版本 | 均值差 |
|---|---|---|---|---|
| openvla | 78.0% | +0.0314 | 45.4% | −0.0070 |
| internvla | 64.5% | +0.0215 | 42.3% | −0.0039 |
| openpi_0（旧） | 93.2% | +0.0564 | 82.0% | +0.0410 |
| spatialvla | 64.2% | +0.0166 | 61.1% | +0.0060 |
| spatialforcing | 79.0% | +0.0114 | 24.7% | −0.0121 |
| π0 (v3) | 83.6% | +0.0444 | 56.2% | +0.0056 |
| π0.5+镜像 base | 76.8% | +0.0281 | 52.4% | −0.0024 |
| π0.5+镜像 direct | 72.8% | +0.0301 | 56.8% | −0.0063 |

按族拆开（π0.5+镜像 base，xy）：

| family | n | 后段发散占比 |
|---|---|---|
| Target Identification | 18 | **100.0%** |
| Identity Tracking | 30 | **100.0%** |
| Scene Updating | 48 | 93.8% |
| Constraint Composition | 12 | 91.7% |
| Relational Grounding | 30 | 46.7%（不成立） |
| Geometric Inference | 30 | 36.7%（不成立） |

> **不能笼统地说“模型轨迹是发散的”。** 发散只在点目标（move_to）族成立；
> 区域目标族不成立；3D 版本下大部分不成立。

### 5.7 成功 / 失败分报（π0.5+镜像 base，剔除 Track 族）

| 子集 | n | `tte_mean_raw_xy` | `tte_final20_xy` | `tte_min_xy` | `tte_mean_raw_3d` | `tte_min_3d` |
|---|---|---|---|---|---|---|
| reach 成功 | 31 | 0.0777 | 0.0895 | 0.0234 | 0.1865 | 0.1435 |
| reach 失败 | 107 | 0.1630 | 0.1901 | 0.0937 | 0.2382 | 0.1804 |

xy 版本在成功/失败上相差 2 倍（3D 版本只相差 1.3 倍），再次确认 xy 版本才有区分度。

---

## 6. ⚠️ 论文已报 TTE 的更正

### 6.1 三个不同的量都叫“TTE”

| 叫法 | 实现位置 | 类型 | 量级 | 现状 |
|---|---|---|---|---|
| (a) `mean_target_distance` | `scripts/metrics_trajectory.py` | 连续距离（米） | 0.30–1.32 | **论文数的真正来源** |
| (b) `tte` ← `mean_mta` | `scripts/export_paper_appendix_data.py` L428 | — | — | `diagnostics_results.csv` 该列**全空** |
| (c) 二元目标选中率 | 本文档早期版本 | 准确率 | 0.24–0.39 | **已重命名为 `target_closest_rate`** |

`paper/appendix_results_dashboard.tex` L262/L275 写的是
“TTE measures trajectory-level execution error”、“higher values indicate worse performance”
—— 与 (a) 一致。本文档早期把 (c) 误称为 TTE，现已全部改正。

### 6.2 论文值 0.33–1.32 vs 重算值 0.13–0.24：差异来源

**不是 step 数，不是单位，不是归一化。是目标点 $g$ 取了不同的点。**

| | 论文（`metrics_trajectory.py`） | 重算（`tte_recompute.py`） |
|---|---|---|
| $g_{xy}$ | `legacy_layouts.json` 按名称匹配的**标称位置** | `eef + goal_error`（环境真值） |
| $g_z$ | 常数 `DEFAULT_TABLETOP_Z_M = 0.784` | 最近物体实测 z（中位 0.9543） |
| 物体运动 | 324 条中 **156 条**走 `static_layout`，**忽略物体移动** | 逐步从 trace 读实时位姿 |
| 维度 | 同一列混着 `xy_only=True`（204 条）与 `False`（120 条） | xy / 3D 严格分开两列 |

**三个污染叠加，使论文已报 TTE 绝对值被系统性放大约 3–4 倍。**
其中光是 z 常数错误就凭空注入 **+0.17 m**（且 `run_eval_openpi_0` 的物体 z 是 0.8143，
与其它 run 相差 0.14 m，**跨模型对比也不成立**）。

### 6.3 “step 数不一致”不是原因（可直接回应审稿人）

各 run 的 step 数分布：

```
run_eval_openvla          n= 328  全部 10 步
run_eval_spatialvla       n= 324  全部 10 步
run_eval_spatialforcing   n= 324  全部 10 步
run_eval_internvla        n= 730  混合 {10:280, 15:11, 40:26, 50:413}
run_eval_openpi_0         n= 616  混合 {20:292, 50:324}
run_eval_openpi_05        n= 324  全部 50 步
run_eval_pi05_v4          n= 168  全部 50 步
run_eval_pi05_v4_direct   n=  81  全部 50 步
```

**没有任何提前终止的 episode**（每个 run 内 step 数要么定值、要么只有极少几种）。
对归一化时间重采样到 50 点后，**均值到小数点后 4 位几乎不变**：

| 运行 | step 分布 | `mean_raw_xy` | `mean_resampled_xy` |
|---|---|---|---|
| internvla | 混合 4 种 | 0.1586 | 0.1586 |
| openpi_0（旧） | 混合 2 种 | 0.1842 | 0.1842 |
| openvla | 全 10 步 | 0.1523 | 0.1525 |

距离序列近似平稳，重采样不移动均值。

### 6.4 全模型重算 TTE（剔除 Track 族，口径统一）

| 运行 | n | `mean_raw_xy` | `resampled_xy` | `final20_xy` | `min_xy` | `mean_raw_3d` | `min_3d` |
|---|---|---|---|---|---|---|---|
| openvla | 268 | 0.1523 | 0.1525 | 0.1817 | 0.0906 | 0.2198 | 0.1669 |
| internvla | 610 | 0.1586 | 0.1586 | 0.1810 | 0.0962 | 0.1881 | 0.1330 |
| openpi_0（旧） | 528 | 0.1842 | 0.1842 | 0.2378 | 0.1011 | 0.2282 | 0.1623 |
| spatialvla | 264 | 0.1318 | 0.1318 | 0.1462 | 0.0974 | 0.2387 | 0.2089 |
| spatialforcing | 264 | **0.1294** | **0.1296** | **0.1376** | 0.1059 | 0.2217 | 0.1950 |
| π0 (v3) | 264 | 0.1756 | 0.1756 | 0.2205 | 0.0943 | 0.2344 | 0.1751 |
| π0.5+镜像 base | 138 | 0.1439 | 0.1439 | 0.1675 | 0.0779 | 0.2266 | 0.1721 |
| π0.5+镜像 direct | 66 | 0.1382 | 0.1382 | 0.1632 | **0.0642** | 0.2594 | 0.2037 |

Track（Identity Tracking）族单列，因其目标在回合中会移动：

| 运行 | `mean_raw_xy` | `final20_xy` | `min_xy` |
|---|---|---|---|
| openvla | 0.1439 | 0.1837 | 0.0600 |
| internvla | 0.1155 | 0.1323 | 0.0672 |
| openpi_0（旧） | 0.2047 | 0.2775 | 0.0738 |
| spatialvla | 0.1221 | 0.1481 | 0.0672 |
| spatialforcing | 0.1167 | 0.1427 | 0.0552 |
| π0 (v3) | 0.1828 | 0.2250 | 0.0633 |
| π0.5+镜像 base | 0.1950 | 0.2438 | 0.0619 |
| π0.5+镜像 direct | 0.1844 | 0.2366 | 0.0582 |

⚠️ **跨模型对比仍不可直接作为主结论**：旧批 run（openvla / internvla / openpi_0 / spatialvla / spatialforcing）
均在**镜像修复之前**跑得，且它们连 `code_version.json` / `code_snapshot` 都没有（见 §9.5）。

### 6.5 归档数值（仅供对照，勿引用）

`analysis_outputs/tte_by_task_family_models.json` 中的 `mean_target_distance`：

| family | internvla | openvla | openpi_0 | spatialvla | spatialforcing |
|---|---|---|---|---|---|
| anchor_reference | 0.8015 | 0.8098 | 0.8604 | 0.7841 | 0.7994 |
| constraint_selection | 0.5636 | 0.6483 | 0.6450 | 0.6148 | 0.6609 |
| explicit_rule_extrapolation | 1.1604 | 1.2576 | 1.3173 | 1.3089 | 1.3000 |
| historical_identity_tracking | 0.5083 | 0.5087 | 0.3927 | 0.5107 | 0.5078 |
| implicit_trend_fitting | 0.3654 | 0.4692 | 0.4687 | 0.4874 | 0.4454 |
| shape_completion | 0.3818 | 0.4962 | 0.4724 | 0.5040 | 0.4641 |
| spatial_placement | 0.6412 | 0.6509 | 0.7000 | 0.6941 | 0.6750 |
| state_reindexing | 0.3404 | 0.3281 | 0.3590 | 0.3178 | 0.3037 |

这些数字包含 §6.2 的全部三个污染，**不应再作为绝对值引用**。

---

## 7. 根因说明：镜像修正

`openpi/examples/libero/main.py` 向策略喂入 `obs["agentview_image"][::-1, ::-1]`。
robosuite 自底向上渲染，因此**第一次翻转仅仅是把图像摆正**，**第二次是真正的左右镜像** ——
即 LIBERO 策略是在**镜像图像**上训练的。IsaacLab 已经是正立渲染，
所以只需施加水平镜像（`OPENPI_IMAGE_HFLIP=1`，见 `geomind_client.py`）。

**受控验证**（noflip vs hflip）：

| 量 | 无镜像 | 有镜像 |
|---|---|---|
| reach | 8.3% | **33.3%**（4×） |
| 置换检验方向散度 | 0.025 | **0.186**（≈8×） |
| 方向余弦 | −0.108 | **+0.775** |

---

## 8. 实验设置

| 项 | 值 |
|---|---|
| 仿真 | IsaacLab（Isaac Sim 4.5），`env_step_hz=24`，`physics_step_hz=120`，`action_repeat=5`（1 步 = 41.7 ms） |
| 策略服务 | `pi05_libero`（`ws://127.0.0.1:8092`，`action_horizon=10`，`extra_delta_transform=False`）<br>`pi0_libero`（`ws://127.0.0.1:8091`，`action_horizon=50`，`extra_delta_transform=True`，语言通道失效） |
| 观测 | `observation/image`、`observation/wrist_image` uint8 (224,224,3)；`observation/state` float32(8,)；`prompt` |
| 动作缩放链 | raw ∈ [−1,1] × 0.1（client）× 0.5（IsaacLab DifferentialIK）= 最大 0.05 m/步 |
| 动作分块 | `REMOTE_ACTION_CHUNK_LEN=5` |
| 自由度 | 旋转锁定（`action[3:6]=0`）；夹爪保持初始值（`hold_initial`） |
| 回合上限 | `--max-steps 50`，`GEOMIND_SUCCESS_DWELL_STEPS=1` |
| 随机化 | `--native-object-pose-perturb-range 0.015`，`--base-seed 20260502` |
| 成功半径 | `--move-to-success-radius-max 0.045`；逐任务 0.01 ~ 0.05 |
| 吞吐 | ≈ 37.5 s/trial；base 168 + direct 81 共 1 h 52 min |

---

## 9. 局限（须在 rebuttal 中如实声明）

1. **夹爪与旋转被锁定**，`Relation*` 族无法真正成功；本工作只评估空间目标定位，不评估放置。
2. **残余方向偏置**：坍缩被打破但未消除，仅从 +Y 旋转到 +X。
3. **`target_closest_rate` 仅对 `move_to` 有定义**（51/81 配对；π0 对比 108/168）；
   TTE 距离指标则对全部任务可算。
4. **随机基线对比存在检验力不对等**：π0 base 的 p = 0.0457（n=204）刚好跨过 0.05，
   π0.5 base 的 p = 0.4822（n=108）。不应声称“两个模型 base 都与随机完全无异”，
   只能声称“base 只略高于随机（Δ ≤ 0.04），direct 高出 0.15”。
5. **direct 的选中率部分是显著性/感知控制**：目标被重着色为绿色且指令中直接点名，
   因此 direct 的高选中率衡量的是“目标被明确指出时能否定位”，
   这正是它作为**对照**的价值（预期 direct 高、base 低），而非记忆能力度量。
6. **`instruction_only` 任务（三个 `ReindexExtreme`）无效应**（4/9 → 4/9），
   说明当前效应依赖视觉线索改动，纯指令改写不足以产生可测差异。
7. **后段发散只在点目标族成立**（见 §5.6）：Relational Grounding 46.7%、
   Geometric Inference 36.7%，不能笼统声称“模型轨迹发散”。
8. **π0 的语言通道失效**，故 π0 与 π0.5 的对比不是纯粹的镜像消融。
9. **旋转开启 A/B 无效应**（n=18，33.3% vs 33.3%，p=1.0000）；
   机制证据更强：`ROTATION_SCALE=2.0` 下每步旋转 ≈ 0.3°，策略本身不发出旋转指令。
   因该幅度接近 0，已知的 LIBERO 与本环境 180° yaw 坐标系差异**未被测试**；
   若将来解锁夹爪，必须先施加 $R_{\text{lib}} = R_z(\pi) \, R_{\text{ours}}$。
10. **π0.5 是随机策略**（flow matching），同一输入重复调用逐元素最大差 0.62–0.86，
    因此单次余弦测量无效，所有方向类结论均基于重复采样 + 置换检验。
11. **论文已报 TTE 绝对值不可用**（见 §6）：z 常数错误 + static layout + xy/3D 混用三重污染。

---

## 10. 复现

```bash
# 统一指标（成功 / reach / 选中率 / SGE / 先验坍缩 / 各类检验）
~/miniconda3/envs/isaac_curobo/bin/python scripts/rebuttal_metrics.py

# TTE 四个版本 + target_closest_rate 重算（输出逐 episode CSV）
~/miniconda3/envs/isaac_curobo/bin/python scripts/tte_recompute.py
# → analysis_outputs/tte_recompute_episodes.csv（2895 行）
```

依赖 `numpy`。注意系统 `python3` 指向 conda base（无 scipy），必须使用 `isaac_curobo` 环境。
所有统计检验均为手写实现（`math.comb` + 置换），不依赖 scipy。

### 10.1 旧批 run 的可追溯性缺失（§6.4 引用处）

- `scripts/metrics_trajectory.py` / `scripts/metrics_lite.py` / `scripts/export_paper_appendix_data.py`
  **从未提交**（`git status` 为 `??`，`git log --follow` 为空），无第二版可对照。
- 镜像修复也**不在任何 commit 里**，至今是未提交的工作区改动
  → 论文那批数全部是**修复前**跑的。
- `run_eval_openvla` / `internvla` / `openpi_0` / `spatialvla` / `spatialforcing`
  连 `code_version.json` 和 `code_snapshot` 都没有。
- 归档数据文件 mtime（2026-05-05 / 05-07）比脚本 mtime（2026-05-07 08:57）**还早**，
  说明那批 CSV 不是用当前脚本生成的。
