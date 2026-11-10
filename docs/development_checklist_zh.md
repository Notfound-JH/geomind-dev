# GeoMind 后续开发 Checklist

> 目标：将当前实验型代码收敛为一套指标可信、生成稳定、数据可训练、能够开源复现的 GeoMind 平台。
>
> 建议周期：12–18 人周。执行顺序原则：先固定指标与数据契约，再做随机化和轨迹生成，最后开展微调与开源。

## 使用约定

- [ ] 未开始
- [x] 已完成并通过验收
- 在条目后添加负责人、截止日期或关联 issue，例如：`（Owner: @name，Due: YYYY-MM-DD，Issue: #123）`
- 只有满足阶段验收门槛后，才勾选对应的“阶段完成”。
- 原始测量值、阈值判定值、人工复核值必须分别保存，不能相互覆盖。

## 总体里程碑

- [ ] M0：冻结当前基线，明确任务、指标和数据契约
- [ ] M1：统一任务指标体系，可以可靠重算历史结果
- [ ] M2：实现可复现、受约束的物体位置随机化
- [ ] M3：统一物体移动与机器人操作原语
- [ ] M4：跑通可恢复、可审计的批量轨迹生成
- [ ] M5：完成小规模模型微调闭环
- [ ] M6：完成代码封装和开源准备

---

## Phase 0：基线冻结与契约定义（P0，1–2 周）

### 0.1 保护当前工作区

- [ ] 备份当前未提交源码、配置和关键实验结果
- [ ] 区分源码、静态资产、第三方依赖、实验输出和论文产物
- [ ] 清点当前修改文件和未跟踪文件，确认哪些需要纳入版本控制
- [ ] 标记并处理可疑临时项：`0`、`1`、`24`、锁文件、误拼目录和尾部带空格目录
- [ ] 记录当前可运行 Conda 环境、Python、IsaacLab、Isaac Sim、CUDA 和驱动版本
- [ ] 保存一组可复现的 baseline 命令与结果

### 0.2 选择开发验证集

- [ ] 选定方向放置代表任务
- [ ] 选定约束选择代表任务
- [ ] 选定几何推理代表任务
- [ ] 选定 reindex 代表任务
- [ ] 选定 track clutter 代表任务
- [ ] 为每个代表任务保存 nominal seed、预期 GT 和 oracle baseline

### 0.3 固定核心 schema

- [ ] 定义并版本化 `TaskSpec v1`
- [ ] 定义并版本化 `EpisodeSchema v1`
- [ ] 定义并版本化 `MetricResult v1`
- [ ] 定义并版本化 `RandomizationManifest v1`
- [ ] 定义统一的阶段状态：select、approach、grasp、lift、transport、place、release、settle
- [ ] 定义统一失败分类：system、data、planning、control、semantic、timeout、unknown
- [ ] 明确 `geomind/` 与 `geomind_lab/` 的依赖方向和职责边界

### Phase 0 验收门槛

- [ ] 任一代表 episode 能追溯到 task config、scene、seed、policy、代码版本和最终指标
- [ ] baseline 在同一环境、同一 seed 下可重复运行并得到一致结果
- [ ] schema 文档不存在未定义的必填字段
- [ ] M0：阶段完成

---

## Phase 1：任务指标完善（P0，2–3 周）

### 1.1 统一指标基础设施

- [ ] 建立统一 metric registry，替代散落在 runner 和后处理脚本中的隐式口径
- [ ] 定义 `MetricResult(name, value, unit, valid, source, version, details)`
- [ ] 定义统一 `EpisodeEvaluation`
- [ ] 保存连续原始测量值，而不只保存 success 布尔值
- [ ] 所有阈值从 task protocol 或版本化评测配置读取
- [ ] 明确区分动态测量、静态 layout proxy 和不可计算状态
- [ ] 缺失指标使用 `valid=false + reason`，不得按零分处理
- [ ] 自动评测和人工复核分别存储
- [ ] 聚合结果保留到 episode 的可追溯索引

### 1.2 系统健康指标

- [ ] 环境启动成功率
- [ ] reset 成功率
- [ ] policy 请求成功率
- [ ] policy 平均延迟、P50、P95、P99
- [ ] action 合法率和异常值计数
- [ ] 相机帧完整率与有效帧率
- [ ] trace、trajectory、object pose 完整率
- [ ] timeout、仿真崩溃和落盘失败率
- [ ] 系统失败不计入模型任务失败

### 1.3 通用阶段指标

- [ ] Target Selection Accuracy
- [ ] Approach Success
- [ ] Grasp Success
- [ ] Lift Success
- [ ] Transport Success
- [ ] Placement Accuracy
- [ ] Release Success
- [ ] Stable Success
- [ ] Collision Rate
- [ ] Timeout Rate
- [ ] 记录首个失败阶段和根因

### 1.4 任务族指标

- [ ] Relation：方向区域满足率
- [ ] Relation：containment 满足率
- [ ] Relation：stack 稳定性与最终相对位姿误差
- [ ] Binding：目标识别准确率与参照关系满足度
- [ ] Constraint：唯一解准确率与约束满足率
- [ ] Geometric：几何目标误差与尺度归一化误差
- [ ] Reindex：状态更新后索引准确率
- [ ] Track：identity accuracy、轨迹保持率和干扰物 margin
- [ ] 将现有 SGE、SCS、MTA 纳入统一 registry
- [ ] 将 target distance、near fraction、closest rate、target margin 纳入统一 registry

### 1.5 难度与统计口径

- [ ] 固定 `loose / medium / strict` 的语义和适用范围
- [ ] 保存每档阈值来源和 evaluator 版本
- [ ] 按 task、family、difficulty、model、seed 聚合
- [ ] 报告均值、标准差、置信区间和有效样本数
- [ ] 区分 micro average 与 macro average
- [ ] 加入 random、oracle 和 scripted oracle baseline
- [ ] 对不同 `max_steps`、action scale 等配置禁止直接无标注横比

### 1.6 测试与历史数据验证

- [ ] 每个指标补正常、边界、缺失字段和异常输入测试
- [ ] 每个代表任务准备一条人工核对的 golden episode
- [ ] 使用统一指标重算现有评测结果
- [ ] 输出新旧指标差异报告
- [ ] 抽样检查视频、原始 trace 与自动评测是否一致
- [ ] 记录无法重算的历史 episode 及原因

### Phase 1 验收门槛

- [ ] 五类代表任务均返回完整、版本化的 `EpisodeEvaluation`
- [ ] golden episode 的指标与人工计算一致
- [ ] 同一份原始数据重复计算结果完全一致
- [ ] 所有聚合值都能追溯到 episode
- [ ] M1：阶段完成

---

## Phase 2：受约束物体位置随机化（P0，2–4 周）

### 2.1 随机化 API

- [ ] 新增独立 `SceneRandomizer`，将采样逻辑从 native delegate 中抽离
- [ ] 定义 `RandomizationSpec`
- [ ] 支持位置、旋转、物体实例、视觉和物理参数的独立开关
- [ ] 按 target、reference、distractor、container、preloaded object 分角色配置
- [ ] 相同 task、seed 和版本产生完全一致的样本
- [ ] 所有随机化结果写入 `RandomizationManifest`

### 2.2 随机化等级

- [ ] L0：场景整体平移或旋转，保持相对结构
- [ ] L1：小范围独立 XY/yaw jitter
- [ ] L2：结构保持的参数化重采样
- [ ] L3：结构变化后自动重算 GT
- [ ] L4：视觉、物理和语义联合随机化
- [ ] 第一版正式发布范围限定为 L0–L2

### 2.3 合法性检查

- [ ] 检查桌面边界和安全 margin
- [ ] 检查物体之间的最小间距和穿模
- [ ] 检查容器、目标区域和障碍占用
- [ ] 检查机械臂基础可达性
- [ ] 写入仿真后等待物理稳定
- [ ] 检查 settled pose 与采样 pose 的偏差
- [ ] 检查物体是否掉出台面或发生非预期翻倒
- [ ] 采样失败时有限次数重试并记录拒绝原因

### 2.4 语义保持验证

- [ ] Relation validator
- [ ] Binding validator
- [ ] Constraint 唯一解 validator
- [ ] Geometric 结构 validator
- [ ] Reindex 序列与索引 validator
- [ ] Track 路径合法性 validator
- [ ] 对 L3 随机化实现 GT 自动重算

### 2.5 稳定性实验

- [ ] 每个代表任务运行至少 1,000 个 seed
- [ ] 统计采样合法率、平均重试次数和失败原因
- [ ] 比较 nominal 与 randomized oracle 成功率
- [ ] 绘制初始位置和角度覆盖分布
- [ ] 检查随机化是否引入明显采样偏差

### Phase 2 验收门槛

- [ ] 相同 seed 复现率 100%
- [ ] L0–L2 合法采样率不低于 95%
- [ ] 1,000-seed 测试无穿模、掉出台面或未标记非法状态
- [ ] GT 在随机化后仍唯一、可计算、可审计
- [ ] M2：阶段完成

---

## Phase 3：物体移动函数与运动原语（P1，3–5 周）

### 3.1 明确三类移动

- [ ] `teleport_object`：只用于初始场景构造
- [ ] `relocate_object_kinematic`：用于 tracking/reindex 的外部状态变化
- [ ] `manipulate_object_robotically`：用于正式执行和训练轨迹
- [ ] 三类移动分别记录来源、时机和结果，禁止使用含糊的通用 `move_object()`

### 3.2 统一接口

- [ ] 定义 `MotionRequest`
- [ ] 定义 `MotionResult`
- [ ] 定义 waypoint、约束、速度、重试和 seed 字段
- [ ] 统一记录 phase results、contact、trajectory 和 failure reason
- [ ] 支持环境无关的规划请求和 IsaacLab executor

### 3.3 基础原语

- [ ] `move_eef_to_pose`
- [ ] `approach_object`
- [ ] `close_gripper`
- [ ] `lift_object`
- [ ] `transport_object`
- [ ] `place_in_region`
- [ ] `place_on_object`
- [ ] `release_object`
- [ ] `retreat`
- [ ] 动态目标运动与路径插值
- [ ] 碰撞检测、超时和有限重试

### 3.4 轨迹变化策略

- [ ] approach direction 采样
- [ ] grasp pose 采样
- [ ] lift height 采样
- [ ] waypoint 和 transport path 采样
- [ ] placement point 采样
- [ ] speed profile 采样
- [ ] recovery 行为
- [ ] 所有 variation 均受 seed 控制并写入 metadata

### Phase 3 验收门槛

- [ ] 五个代表任务均通过统一 motion API 执行
- [ ] 每个 phase 可单独测试和定位失败
- [ ] planner 和 executor 失败不会产生伪成功轨迹
- [ ] scripted oracle 的成功率不低于迁移前基线的可接受范围
- [ ] M3：阶段完成

---

## Phase 4：轨迹生成、质量与多样性（P1，2–3 周）

### 4.1 生成漏斗指标

- [ ] requested count
- [ ] config resolution rate
- [ ] scene sampling rate
- [ ] semantic validity rate
- [ ] env reset rate
- [ ] planning success rate
- [ ] rollout success rate
- [ ] data completeness rate
- [ ] quality acceptance rate
- [ ] 平均重试次数
- [ ] 每条有效轨迹的时间、GPU 小时和存储成本

### 4.2 Episode 数据内容

- [ ] task、scene、layout、seed 和 randomization metadata
- [ ] policy、checkpoint 和 inference config
- [ ] simulator、资产版本和代码 commit
- [ ] 多相机图像及精确时间戳
- [ ] robot proprioception、EEF pose 和 gripper state
- [ ] 相关物体逐帧 pose
- [ ] action、reward、contact 和 phase event
- [ ] 自动指标、失败分类和人工复核
- [ ] 文件 checksum 与数据版本

### 4.3 数据格式

- [ ] JSON/JSONL 只承载 metadata、索引和调试信息
- [ ] 确定视觉数据存储格式和压缩参数
- [ ] 确定数值时序格式
- [ ] 实现统一 writer、reader 和 validator
- [ ] 支持 episode 原子写入，避免中断产生半条数据
- [ ] 支持 resume、retry 和去重
- [ ] 提供数据格式迁移工具

### 4.4 质量分级

- [ ] `gold`：成功、平滑、无异常、指标完整
- [ ] `silver`：成功但存在轻微质量问题
- [ ] `hard_negative`：接近成功且失败原因明确
- [ ] `invalid`：系统、环境或数据异常
- [ ] 训练默认排除 `invalid`
- [ ] 保存 quality gate 的规则版本和拒绝原因

### 4.5 多样性指标

- [ ] 初始位置、角度和物体实例覆盖
- [ ] GT 与目标角色分布熵
- [ ] 路径长度、曲率和 waypoint 分布
- [ ] action variance、smoothness 和 jerk
- [ ] 轨迹 embedding 聚类数和簇分布
- [ ] 成功、近失败和失败类型覆盖
- [ ] 检查 action noise 是否带来有效多样性而非单纯降低质量
- [ ] 建立重复轨迹和近重复轨迹检测

### 4.6 扩展性验证

- [ ] 任务数：1 / 5 / 全量
- [ ] 并行环境数：1 / 4 / 16（按硬件调整）
- [ ] 随机化等级：L0 / L1 / L2
- [ ] 生成规模：100 / 1k / 10k
- [ ] 记录吞吐、显存、失败率、恢复率和输出一致性
- [ ] 输出有效生成率和每千条有效轨迹成本

### Phase 4 验收门槛

- [ ] 连续生成 1,000 条轨迹时流程可恢复、可审计且无静默数据损坏
- [ ] 每条 accepted 轨迹通过 schema 和质量校验
- [ ] 动态任务不再依赖未标记的静态 layout proxy 计算正式指标
- [ ] 生成漏斗中的每次丢失都有明确原因
- [ ] M4：阶段完成

---

## Phase 5：轨迹统计与模型微调（P1，2–4 周）

### 5.1 数据统计与切分

- [ ] 按 task、family、scene、seed 和 quality tier 统计数量
- [ ] 绘制轨迹长度、初始位姿、action 和阶段失败分布
- [ ] 统计相机丢帧和 object pose 缺失
- [ ] 按 scene、layout group 或物体实例切分 train/val/test
- [ ] 检查近重复轨迹跨 split 泄漏
- [ ] 固定 dataset manifest 和版本号

### 5.2 训练前 profiling

- [ ] 用 100–500 step 测量单 step 时间和峰值显存
- [ ] 测量数据解码、CPU、磁盘和 GPU 利用率
- [ ] 记录每秒样本数和 checkpoint 大小
- [ ] 估算完整训练 GPU 小时与存储开销
- [ ] 单独估算数据生成和仿真评测成本

### 5.3 Pilot A：可学习性

- [ ] 选择 1–3 个任务
- [ ] 每任务准备 100–300 条 gold/silver 轨迹
- [ ] 固定零样本、随机和 oracle baseline
- [ ] 完成一组参数高效微调
- [ ] 至少重复三个 seed
- [ ] 在独立仿真测试集评估，而不只观察 training loss

### 5.4 Pilot B：数据规模曲线

- [ ] 使用 10%、30%、100% 数据训练
- [ ] 比较重复数据与高多样性数据
- [ ] 绘制数据量—任务成功率曲线
- [ ] 分析不同 quality tier 的贡献

### 5.5 Pilot C：多任务泛化

- [ ] 覆盖主要任务族
- [ ] 留出未见位置、布局和物体实例
- [ ] 比较单任务、多任务和 curriculum
- [ ] 检查灾难性遗忘和任务间负迁移

### Phase 5 验收门槛

- [ ] 至少一个任务族取得可复现且统计显著的正向收益
- [ ] 微调收益在未见随机化测试集上仍然成立
- [ ] 完整报告训练、数据生成、存储和评测成本
- [ ] M5：阶段完成

---

## Phase 6：代码封装与开源（P2，2–4 周）

### 6.1 包结构与 API

- [ ] 将 task、metrics、randomization、motion、generation、data、policies、CLI 分层
- [ ] 消除 `geomind/` 与 `geomind_lab/` 的不必要双向依赖
- [ ] 将 `geomind` 加入 `pyproject.toml` 打包范围
- [ ] 固定公共 API，内部实验 API 明确标注 unstable
- [ ] 配置 schema 提供版本和迁移策略

### 6.2 依赖与资产

- [ ] 固定 Python、IsaacLab、Isaac Sim、CUDA 等版本范围
- [ ] 明确 IsaacLab 使用外部依赖、子模块或安装脚本中的哪一种方式
- [ ] 检查第三方代码、USD 资产、模型和数据许可证
- [ ] 大型视频、轨迹、checkpoint 不进入 Git
- [ ] 提供可选的数据与资产下载方式
- [ ] 清理服务器地址、账号、token、绝对路径和私有配置

### 6.3 文档

- [ ] 更新根 README，说明 `geomind/` 与 `geomind_lab/`
- [ ] 安装说明
- [ ] 五分钟 mock quickstart
- [ ] IsaacLab 单任务 quickstart
- [ ] 批量生成与评测教程
- [ ] 自定义任务、指标、随机化和 policy adapter 教程
- [ ] 数据 schema 和指标定义文档
- [ ] benchmark reproduction 文档
- [ ] 常见错误与排障说明

### 6.4 测试与发布

- [ ] 纯 Python 单元测试不依赖 Isaac Sim
- [ ] 配置和 schema 测试
- [ ] mock backend 集成测试
- [ ] IsaacLab smoke test：make、reset、step、close
- [ ] 单 episode 端到端测试
- [ ] 小型 generation pipeline 测试
- [ ] 添加格式化、lint、类型检查和 CI
- [ ] 添加 LICENSE、CONTRIBUTING、CITATION 和 Code of Conduct
- [ ] 发布前从全新环境执行安装和复现测试

### Phase 6 验收门槛

- [ ] 新用户可按文档在全新环境运行 mock 示例
- [ ] 满足依赖条件的用户可运行 IsaacLab 示例
- [ ] 公共仓库不包含私密信息和无授权资产
- [ ] benchmark 的核心指标可以从公开步骤复现
- [ ] M6：阶段完成

---

## 每周例行检查

- [ ] 本周新增配置是否通过 resolver 和 schema 校验
- [ ] 本周新增指标是否有版本、单位、来源和测试
- [ ] 本周生成数据是否通过 completeness 与 quality gate
- [ ] 随机化、policy 和 planner 是否记录 seed
- [ ] 是否出现新的静默 fallback 或 static proxy
- [ ] baseline 是否发生未解释回退
- [ ] 实验命令、代码 commit 和配置是否完整归档
- [ ] 大文件、临时输出和私密配置是否误入 Git
- [ ] 更新阶段风险、负责人和下周目标

## 当前最高优先级

- [ ] 1. 定义 `EpisodeSchema v1` 和 `MetricResult v1`
- [ ] 2. 选定五个代表任务及 golden episodes
- [ ] 3. 将现有 SGE/SCS/MTA 和轨迹指标统一到 metric registry
- [ ] 4. 为现有 XY 扰动补完整 manifest 和合法性检查
- [ ] 5. 设计三类物体移动 API，先不大规模迁移实现
- [ ] 6. 在正确 Conda 环境恢复测试基线
