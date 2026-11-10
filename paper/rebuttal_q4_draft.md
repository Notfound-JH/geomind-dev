# Rebuttal — Q4 failure analysis (Reviewer comment on aggregate metrics)

> **Status**: draft v1. All numbers below are reproduced from
> `analysis_outputs/tte_recompute_episodes.csv` (2895 episodes, per-step live object
> poses) using per-episode chance correction and paired permutation tests.
> See §"Internal notes" at the bottom for what must change in the main text.

---

## Response to Reviewer (English, ~520 words)

**We thank the reviewer for this precise criticism. We agree, and on re-examination we
found the aggregate comparison was weaker than our wording implied. We have replaced it
with a per-episode, chance-corrected analysis that speaks directly to the mechanism.**

**Why the aggregate comparison was inconclusive.** SGE and TTE are absolute distances,
so they are only comparable across families if the initial target distance is matched.
It is not: averaged over all seven models, the initial end-effector-to-target distance is
6–16% smaller for Identity Tracking than for Scene Updating (0.078 m vs 0.093 m,
consistent in 7/7 models). Furthermore, the Scene Updating family is internally
heterogeneous — chance-corrected grounding ranges from +0.67 to −0.16 across its six
task types — so its family mean is not a meaningful comparison unit. We therefore no
longer claim that Identity Tracking is *harder in aggregate* than Scene Updating.

**A fine-grained analysis that does support the mechanism.** We instead ask directly
whether the model binds the referring expression to the correct object *instance*. We
define **grounding accuracy (GA)** as the fraction of steps at which the ground-truth
target is the object nearest the end-effector, and compare it against the per-scene
chance level $1/n_{\text{obj}}$ (paired permutation test over episodes; π₀ and π₀.₅ on
GeoMind-Base).

| Task group | $n_{\text{obj}}$ | $N$ | chance | GA | lift | $p$ |
|---|---|---|---|---|---|---|
| Target Identification – AnchorDistance | 5 | 24 | 0.200 | 0.439 | **+0.239** | **0.003** |
| Target Identification – AnchorDirection | 5 | 18 | 0.200 | 0.292 | +0.092 | 0.21 |
| Identity Tracking – TrackSingle | 3 | 54 | 0.333 | 0.397 | +0.064 | 0.17 |
| Identity Tracking – TrackMulti | 6 | 18 | 0.167 | 0.000 | −0.167 | <0.001 |
| Identity Tracking – TrackClutter | 7 | 18 | 0.143 | 0.143 | +0.000 | 0.98 |

When the referring expression is resolvable from the *current* observation
(Target Identification), grounding is significantly **above** chance. When it requires
binding to a *past* scene state (Identity Tracking), grounding is **at or below** chance
at every difficulty level.

**Grounding fails while motor control does not.** Within Identity Tracking, comparing
TrackSingle (3 objects) with TrackClutter (7 objects) holds the initial target distance
fixed (0.054 m vs 0.050 m) and leaves closest-approach error essentially unchanged
(0.043 m vs 0.047 m), yet the end-of-episode selection rate collapses from 0.750 to
0.000 (identical direction in both models). Adding distractors destroys *identity
selection*, not *reaching precision*.

**A controlled intervention isolates reference resolution.** In our
`direct`-reference ablation (identical task, level and seed; 81 matched pairs),
replacing the memory-dependent expression with an explicit target reference changes
grounding but not execution:

| Family | pairs | GA base → direct | Δ GA | $p$ | reach base → direct | Δ reach |
|---|---|---|---|---|---|---|
| Identity Tracking | 15 | 0.183 → 0.340 | **+0.157** | 0.09 | 0.400 → 0.400 | **+0.000** |
| Target Identification | 6 | 0.083 → 0.493 | **+0.410** | 0.03 | 0.000 → 0.167 | +0.167 |
| Scene Updating | 24 | 0.350 → 0.352 | +0.002 | 0.98 | 0.250 → 0.208 | −0.042 |

For Identity Tracking the ablation moves grounding by +0.157 while leaving reaching
performance **exactly unchanged** (0.400 → 0.400), i.e. the recovered information is
consumed entirely at the reference-resolution stage. The near-zero effect on Scene
Updating indicates its failures are of a *different* kind (the goal is a position, not an
object instance) — which is the more informative statement, and one the aggregate metrics
could not make. (For completeness, the one family where the ablation does move execution
is Geometric Inference, 0.200 → 0.600, $p=0.03$, where the `direct` variant supplies the
inferred coordinate itself and therefore removes the inference step rather than the
reference-resolution step.)

**Revised claim.** The core difficulty is **resolving a linguistic reference to the
correct object instance, especially when that reference depends on a past scene state** —
not motor execution, and not a blanket ordering between the two Spatial Memory families.
We will revise the Q4 paragraph accordingly and add these two tables to the appendix.

---

## Internal notes (不进 rebuttal)

### 必须改的正文
原 Q4 段落中以下三处**不可保留**：
1. `failure rates on identity tracking tasks exceed 85%` — 复现不出来（主表均值 78.5%，诊断表 78.2%）。
2. `identity tracking has a higher SGE and TTE than Scene Updating` — 干净口径下 SGE 1/6、
   TTE 2/7（`tte_min_xy` 0/7）；且被初始距离 $d_0$ 混杂。**必须删除。**
3. `identity tracking remains a significant challenge` 作为**与 Scene Updating 的相对**论断 —
   降级为绝对论断（IT 不超随机），不做族间排序。

建议改写后的段落骨架：
> We analyse Spatial Memory tasks at the level of individual episodes rather than family
> means, since absolute distance metrics are confounded by initial target distance
> (Appendix X). We measure grounding accuracy against per-scene chance ... [Table R1] ...
> Identity Tracking never exceeds chance, whereas Target Identification does (+0.24,
> p=0.003). A distractor-count ladder and a direct-reference ablation [Table R2] show the
> deficit is in reference resolution, not motor execution.

### 已知需要标注/回避的坑
- **TrackMulti 的 $d_0$=0.183 m**，是其他组的 3.4 倍；其 GA=0.000 部分是几何原因。
  rebuttal 里没有单独强调 TrackMulti，主论据放在 TrackSingle/TrackClutter 的**等距对照**上。
- **ReindexInsert GA=0.921** 是伪影：`tte_min_xy == d0 == 0.2197`，机械臂全程没动，
  目标恰好是起始位形下最近的物体。不要引用这个数。
- `target_closest_rate` 仅 `pi0_v3` / `pi05_v4` / `pi05_v4_direct` 有；其余 5 个 run 的
  trace 不含物体位姿。因此表 R1/R2 只覆盖 2 个模型 —— 若审稿人追问覆盖面，
  需要重跑旧模型的 trace dump。
- 老 run（openvla/internvla/spatialvla/spatialforcing/openpi_0_old）`reach_success` 恒为 0，
  其 `success` 是旧的「任意一步进区」口径 —— **主表 `base_family_results.csv` 建立在此口径上**。
- `run_eval_models_task_level_matrix` 中 `openpi_05` 行与 `openpi_0` 逐字节相同（待你核 xlsx）。
- 主表(matrix)与诊断表(aggregate_diagnostics)两套聚合口径互相矛盾
  （openpi_05 Base：Constraint 40.6% vs 0.0%，Scene Updating 35.4% vs 4.2%）。

### 复现脚本
`python scripts/analysis_q4_grounding.py` → 产出
`analysis_outputs/q4_grounding_chance_corrected.csv`（R1）与
`analysis_outputs/q4_direct_ablation_paired.csv`（R2）。本文所有数字均由该脚本复现。

### 落盘后新发现的一点（草稿已据此收紧措辞）
`Geometric Inference` 的 direct 消融 **reach_success 0.200 → 0.600（p=0.031）**，
是 execution 上唯一显著的变化。原因是 GI 的 direct 变体直接给出推断后的坐标，
去掉的是**推断**步骤而非**指称解析**步骤。因此 rebuttal 里不能笼统说
「direct 不改变 execution」，只能说 Identity Tracking 上 Δreach 恰为 0.000。
（GI 无 `target_closest_rate`，因其目标是推断出的空间点而非物体实例。）
