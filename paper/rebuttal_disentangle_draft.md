# Rebuttal — Disentangling spatial understanding from control / interface / deployment

> **Status**: draft v1 (2026-07-29).
> Reproduce with `python scripts/reach_dwell_audit.py` and `python scripts/rebuttal_metrics.py`.
> Per-episode data: `analysis_outputs/rebuttal_disentangle/reach_dwell_audit.csv` (1185 episodes).
> See §"Internal notes" for the caveats that must be stated in the paper.

---

## Reviewer comment

> How do the authors disentangle failures caused by deficient spatial understanding from
> failures caused by low-level control, action-interface mismatch, or policy deployment
> artifacts? For example, could the authors provide ablations such as oracle
> target + learned controller, ground-truth object pose + planner, VLM-predicted
> target + oracle controller, or pure reaching/placing baselines?

---

## Response to Reviewer (English, ~700 words)

**We thank the reviewer for this question. We agree it is the load-bearing question for
our claim, and we have run all four requested ablations. They are reported below; all of
them are new to this revision except the oracle-target ablation.**

**Setup.** Every ablation below runs through the *identical* environment, action
interface, step budget (50), evaluator and seed schedule as the evaluated policies; only
the stated component is replaced. Our primary metric is $\text{reach}@5$: the
end-effector enters the task's success radius and *stays* inside for 5 consecutive steps.
Episodes whose initial pose is already inside the radius are removed from the denominator,
since neither the policy nor the oracle has to do anything in those cases. All comparisons
are paired on (task, level, seed) and tested with an exact McNemar test.

**(a) Ground-truth target + non-learned controller (control upper bound).** We replace the
policy with a controller that reads the environment's ground-truth goal vector and moves
in a straight line at constant speed toward it, injected through the same policy hook as
the remote VLAs. On the full benchmark (27 tasks × 3 levels × 2 seeds = 162 episodes):

| | paired $n$ | reach@5 | $b$ | $c$ | $p$ |
|---|---|---|---|---|---|
| GT-target controller | 102 | **70.6%** | 50 | 5 | **$<10^{-4}$** |
| π₀.₅ (best config) | 102 | 26.5% | | | |

The same environment that a π₀.₅ policy solves 26.5% of the time is solved 70.6% of the
time once — and only once — the *target* is supplied. Under the unconditioned denominator
the gap is 44.4% vs 16.7% ($n=162$, $b=50$, $c=6$, $p<10^{-4}$). Task success (not just
reaching) moves from 4.9% to 60.5%. The control channel is therefore demonstrably not the
binding constraint.

**(b) Pure reaching baseline.** The controller in (a) *is* a pure reaching baseline: the
target object is pre-loaded in the gripper, rotation is zeroed and the gripper command is
held constant, so the residual action space is a 2-D translation. There is no grasp,
release or re-orientation skill to learn. We make this explicit in the revised Methods and
no longer describe the benchmark as evaluating placement.

**(c) VLM-predicted target + oracle controller.** We upper-bound this by taking the
*closest point the policy's own trajectory ever reaches* as its implied target, which is
strictly optimistic. That distance averages $0.075$ m for π₀.₅, against a success radius
of $0.05$ m; for the GT-target controller it is $0.014$ m. Even granting a perfect
controller and the most favourable reading of the policy's intent, the implied target is
outside tolerance on average. A discrete version tells the same story: the fraction of
episodes ending closest to the ground-truth target object is 25.5% for π₀.₅ versus a
per-scene chance level of 22.6% (Poisson-binomial, $P=0.275$, *not* distinguishable from
chance), and 91.2% for the GT-target controller.

**(d) Oracle target + learned controller.** In our `direct` ablation the scene, object
poses, intervention script and seed are byte-identical; only the instruction text and the
target's visual saliency change. Reaching improves ($23.5\% \to 33.3\%$, $n=81$,
$p=0.022$) and, critically, target selection moves from indistinguishable from chance
($0.260$, $p=0.482$) to significantly above it ($0.377$, $p=0.007$). The learned
controller is therefore able to exploit target information it does not extract on its own.

**Action-interface mismatch and deployment artifacts.** We treat these as first-class
confounds rather than assuming them away. Holding the checkpoint fixed and varying one
interface knob at a time (paired on task/level/seed):

| interface variant | paired $n$ | variant | reference | Δ | $p$ |
|---|---|---|---|---|---|
| no image h-flip | 24 | 0.0% | 20.8% | −20.8 | 0.063 |
| image h-flip (= reference) | 24 | 25.0% | 20.8% | +4.2 | 1.00 |
| **rotation d.o.f. enabled** | 18 | 50.0% | 44.4% | +5.6 | **1.00** |
| LIBERO-style proprioception | 168 | 2.4% | 16.7% | −14.3 | $<10^{-4}$ |

The interface is *not* innocuous — image orientation and proprioceptive encoding each move
reaching by 14–21 points — so all reported numbers are best-of-configuration, and we say
so. But two things follow. First, unlocking the rotation degrees of freedom, the specific
interface restriction one would suspect, changes nothing ($p=1.00$). Second, the entire
measured interface bandwidth (0%–25%) sits below the 70.6% attained by the same
environment with the target supplied: no interface configuration we found closes the gap.
On deployment artifacts, across all 1185 episodes there are no truncated rollouts, no
invalid-metric episodes, and an offline recomputation of the reach criterion reproduces the
logged verdict in 1185/1185 cases.

**Conclusion.** Supplying the target closes most of the gap; supplying a better controller,
a better interface, or a cleaner deployment does not. We have added these ablations as
Appendix Y and softened the Methods to state that the benchmark measures spatial target
localisation, not manipulation skill.

---

## 数据出处（每个数字对应哪张表）

| rebuttal 中的数字 | 来源 |
|---|---|
| 70.6% / 26.5% / b=50 c=5 | `reach_dwell_audit.py` 判定门 [条件口径] |
| 44.4% / 16.7% / b=50 c=6 | 同上 [全量口径] |
| 4.9% → 60.5% (success) | `rebuttal_metrics.py`「控制上界」段 |
| 0.075 / 0.014 m (sge_xy_best) | 同上 连续指标 |
| 25.5% / 91.2% / 随机 22.6% | 同上 tte + 泊松二项基线 |
| 23.5% → 33.3%, p=0.022 | `rebuttal_metrics.py`「base → direct」段 |
| 0.260 → 0.377, p=0.482 → 0.007 | `paper/rebuttal_metrics.md` §4.1 |
| 接口四行 | `reach_dwell_audit.py` [表 5] |
| 1185/1185 | `reach_dwell_audit.py` 验证门 |

---

## Internal notes（不进 rebuttal）

### 必须如实声明的三点局限

1. **oracle 不是 IK planner。** `scripted_goal_policy.py` 只在 xy 平面上以恒定幅值
   （magnitude 1.0 × action_scale 0.1）直线走向 GT goal，$a_z \equiv 0$、旋转维 $\equiv 0$。
   写成 "ground-truth object pose + planner" 会失实。**只能写 "GT goal vector + constant-speed
   straight-line controller"**，并说明它给出的是 *xy 平面内* 的上界。
2. **21/162 条 oracle episode 贴着步数预算**（reach_step / 50 ≥ 0.7）。
   162 条里真实未达成 31 条，其中 17 条 50 步内从未进入半径、min_d 中位 **0.083 m**
   —— 这是恒速直线走不到，属 **budget-limited**，不能算作「控制上界也解决不了」。
   若审稿人追问 oracle 为何不是 100%，标准答复就是这条。
3. **接口是 best-of-config。** 表 5 证明接口带宽有 14–21 个百分点，绝不能写成「接口无影响」。
   立论必须是「带宽的上端 < oracle 下端」，而不是「接口不重要」。

### 口径钉死（写 Methods 时逐条抄）

- 实跑配置 `GEOMIND_SUCCESS_DWELL_STEPS=1`，即 reach 退化为「任意单步进区」（flyby 也算）。
  reach@3 / reach@5 是**离线重算**，不是重跑。必须说明。
- 逐步判定用 `trace[t].meta.goal_distance` vs `success_radius`，即 evaluator 看到的判决量；
  **不用** $\lVert\text{goal\_error}\rVert$（那是长度不是判决），也**不用**
  `metrics_trajectory.py` 的 `*_target_distance`（名义布局坐标 + 桌面 z 常数，放大 3–4 倍）。
- `summary.json` 的 `success_rate` 是第三套口径（`classify_success()` 的末步 level 阈值，无 dwell），
  **任何情况下都不要引用**。
- **评测器版本漂移**：新版 `meta.goal_distance` 是各向异性重标定后的判决量，原始欧氏距离移到
  `goal_distance_l2_m`，且新版 `_l2_m` 恰等于旧版 `goal_distance`。只有 `goal_kind=generic`
  是新增重标定。审计脚本用双口径：native 过验证门，aligned（generic 回退 `_l2_m`）出结论。
  对齐后新旧 scripted 在重合 24 条上 reach@5 不一致 **0 条**（对齐前 7 条）。

### 起始态口径的坑（差点写错）

`reach_k5_excl_start` = `reach@5 and not start_inside`，把起手即命中的 episode **当失败留在分母**。
各 run 起手命中率悬殊（oracle 36.4% vs π₀.₅ 6.0%），这会系统性**低估 oracle**。
正文用的是**条件口径**：把这些 episode 从分母剔除，并只在双方都起手在半径外的配对上做检验
（$n=102$，剔 60 对）。全量口径作为保守下界一并给出，两个口径结论一致。

### 覆盖面缺口（若审稿人追问）

- 接口扫描里 `chunk5` / `scale005` / `scale01` 各只有 2 task × 12 episode，$p=0.5$，
  **不要单独引用**，只作为带宽区间的组成部分。
- `noflip` 的 $p=0.063$ 未过 0.05（$b/c = 0/5$，n 太小）。表述用「0/24 vs 5/24」的原始计数，
  不要写成「显著变差」。
- oracle 未覆盖的对照：GT 物体位姿 + 真正的 IK/motion planner。若审稿人坚持要，
  需要新跑一批（curobo 已在环境里，但要接 `--policy` 通道）。
- π₀ 侧没有对应的 direct 消融（只有 π₀.₅ 有），跨模型的 (d) 结论只能说 π₀.₅。

### 仍需补的一件事

`scripts/rebuttal_metrics.py` 的 reach 用的是 `result.json` 的 `reach_success`（dwell=1、
新版评测器口径），因此表里 oracle 的 88.9% 对 generic 那 24 条用了更严的尺子，是**保守值**。
若要与正文 70.6% 完全同源，需把该脚本也改成 aligned 口径。目前两套数字方向一致、
量级一致，暂不阻塞投稿。
