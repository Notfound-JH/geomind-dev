# geomind-research 完整流水线验收

日期：2026-11-07（Asia/Shanghai）。完成本地实现和真实 PhysX 验证；没有训练、远程模型评估、提交或发布。保留已有工作区修改及全部历史实验。本次目标是代表任务链路，未实现全部27项。

## 实际结果

正式最终源码批次 **8/8 正例通过、5/5 错误目标对照被拒绝**；13条真实机器人轨迹全部离线复核通过。另有初始静态单例1正1负，物理完成，但新导出审计首次发生类型错误；已保留，不计入正式批次。

|任务|条件|场景|seed|后端|生成/GT|实际执行正例|错误目标拒绝|正例TCP误差|
|---|---|---|---|---|---|---|---|---|
|ProjectOntoLine|Base|semantic_table|0,1|semantic + isaaclab_physx|2/2|2/2|1/1|5.95–6.23 mm|
|ProjectOntoLine|Aug：translate_x_0.03m|semantic_table|0,1|semantic + isaaclab_physx|2/2|2/2|1/1|5.96–6.04 mm|
|ProjectOntoLine|Base（场景切换）|registered_scene_65|0,1|semantic + isaaclab_physx|2/2|2/2|1/1|5.95–6.23 mm|
|DetectSceneChange 位移版|Base|semantic_table|0,1，分两次运行|semantic + isaaclab_physx|2/2|2/2|2/2|5.54–5.72 mm|

本表各项均已定义工程请求、已实现、已生成、已产生控制waypoints、已执行、已验证。规划是既有三段waypoint与逐步阻尼IK控制，不是离线全局避障规划。动作契约 **ReachAboveTarget**：TCP到语义目标上方0.10m，位置≤12mm、姿态≤0.15rad，连续保持20个120Hz物理步。不称抓取、放置或学习模型成功。

[机器汇总](../../../outputs/research/pipeline_20261107/summary.json) 保存逐run指标/证据/配置与源码快照ID；[环境版本](../../../outputs/research/pipeline_20261107/environment.json)；[产物SHA256](../../../outputs/research/pipeline_20261107/artifact_sha256.json)。正式批次相关源码快照ID均为 `1fce79268832210e651658a6d383c3b42adbc868f90aa37835bf0fb35e2e486a`，commit及可恢复内容见各run/provenance.json与source_snapshot/，不是仅dirty标记。

## 27项能力、定义与兼容

[27项可读矩阵](../task_capabilities.md)、[逐项详细JSON](../task_capabilities.json) 包含ID、Domain、Family、定义hash、配置来源、Base/Aug、时间机制、GT状态、输入/答案、场景/布局/后端、生成/事件/控制/执行/导出声明及缺口。计数为4/5/5/7/3/3。矩阵中的实现声明不替代上表执行证据。

实际定义入口是 [docs/task_descriptions.md](../../task_descriptions.md)，不存在doc目录。保留文档的分类已确认、操作定义/增强草案、位移与增删未决等地位；未将文档的“当前电脑没有远端源码”背景描述当作当前实现证据。

[Base/Aug映射](../base_aug_mapping.md) 记录旧split/condition/variant的不同含义。新condition在请求侧保存，不修改旧release1兼容schema。SelectWithVisualCue属于canonical；SelectBetweenObjects旧JSON和Between关系记录保留，canonical入口仍按既有实现拒绝，不自动改为双关系。

## 三个可直接复制的skill调用

```text
使用 $geomind-research：生成 GeometricInference 类别的 Base 样例，scene_id=semantic_table，seed=0，数量2，backend=isaaclab，每次限时420秒；生成实例、GT和机器人参考/执行轨迹并验证，保存到 outputs/research/next_base。未指定task时记录实际选择理由。
```

```text
使用 $geomind-research：为 ProjectOntoLine 生成 Aug，增强明确为 translate_x_0.03m，继承同scene/seed的Base并保存配对；scene_id=semantic_table，seed=0，数量2，backend=isaaclab，限时420秒，输出 outputs/research/next_aug。重新求解GT并实际执行、复核轨迹。
```

```text
使用 $geomind-research：生成 SceneUpdating 的 DetectSceneChange 位移版 Base，scene_id=semantic_table，seed=0，数量1，backend=isaaclab，限时420秒，输出 outputs/research/next_dynamic。执行观察—隐藏事件—查询—机器人动作，分开保存模型可见流、GT、事件计划/实测轨迹与机器人执行记录。
```

场景切换可将第一个请求的scene_id改为registered_scene_65；它仍是Base。自然语言由skill解释，确定性脚本接受解析后的JSON；没有声称脚本本身是任意自然语言解析器。本次实际按这三类请求运行，另增加第二场景检查。

## 入口与复现

入口 [research_pipeline.py](../../../geomind/tools/research_pipeline.py)，原始请求模板在 [configs/research_requests](../../../configs/research_requests)。执行环境为本机new_isaaclab。现有输出不可覆盖，重跑先使用新目录：

```bash
python - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('configs/research_requests/base.json').read_text())
r['output']='outputs/research/reproduce_base_01'
Path('/tmp/geomind_reproduce.json').write_text(json.dumps(r))
PY
/home/elm/miniconda3/envs/new_isaaclab/bin/python -m geomind.tools.research_pipeline --request /tmp/geomind_reproduce.json
```

换模板aug.json、scene65.json、dynamic_single.json或dynamic_second.json并设置各自新output可复现对应组。GPU串行；每个执行器使用独立进程组、420秒deadline，完整报告后最多10秒关闭等待。Kit关闭阻塞时仅清理本次组，记录bounded_cleanup_after_report；不终止其他工作进程。

只复核现有数据（不启动物理、不重执行）：

```bash
python -m geomind.tools.research_pipeline --audit outputs/research/pipeline_20261107/base
python -m geomind.tools.research_capabilities --output /tmp/geomind_capabilities.json
```

CPU生成请求指定backend=semantic以及artifacts=["gt","observations","events"]。请求机器人轨迹却选择CPU会明确拒绝。类别完整覆盖请求报告逐项支持状态并拒绝缺失组合，不缩小到单任务。

## 产物路径与数据边界

共同根为 `outputs/research/pipeline_20261107/`：

|组目录|内容|
|---|---|
|[base](../../../outputs/research/pipeline_20261107/base)|投影Base两个seed|
|[aug](../../../outputs/research/pipeline_20261107/aug)|平移增强两个seed；paired_base/保存原布局/episode/独立名义答案|
|[scene65](../../../outputs/research/pipeline_20261107/scene65)|注册场景65的投影Base两个seed|
|[dynamic_single](../../../outputs/research/pipeline_20261107/dynamic_single)|位移检测seed0|
|[dynamic_second](../../../outputs/research/pipeline_20261107/dynamic_second)|位移检测seed1|

每组的 raw_request.json/request.json 保存原始和展开请求；configs/保存任务/场景/布局/时间协议/执行参数；configs/reports/sampling.json保存有界拒绝采样候选、失败原因、最终目的地。复用生成器会暂存三类任务配置，但research_request suite和验收只选择请求任务，不把多出的配置算作覆盖。

semantic/*.json保存CPU观察、语义GT与事件计划；CPU事件是离散状态更新。physics/cases/*.json保存实际查询GT、语义/物理判定、计划waypoints、固定目标、终止原因和资产来源。physics/evidence保存评估端历史/当前实测状态；physics/observations是协议步模型可见流；physics/events分别保存计划与实测trace链接。

physics/traces/*.npz为实测步号、协议步、阶段、7臂关节、TCP世界位置和3对象世界位置；同名_controls.npz保存每步全关节目标、实测位置/速度、TCP wxyz四元数、关节名、限制、dt、instance与contract hash。动作是绝对关节位置；机械臂rad、夹爪m；位置m，物理120Hz。夹爪状态已记录，但未执行抓放。阶段0稳定、1观察/事件、2抬升、3接近、4下降。查询GT采用稳定后的实测状态，目标冻结后不追随执行误差。

位移seed0实际变化block_1约0.18872m，其他对象最大约2.98e-8m；独立检查从参考/查询状态重算位移并核对计划对象与唯一阈值。动态观察参考为Observe.End；隐藏Delay仅返回空objects；Respond后才接受答案和执行机器人动作。公开流额外的robot字段只有本体状态，与纯CPU白名单条件分开。没有GT/事件目标/特权历史泄露到公开流，也未调用真实模型。

注册65通过GeoMindSceneCfg.apply_scene加载真实桌面/背景/灯光和网格碰撞，保存源USD SHA256 `d1272332bdbf4dd87e5f13ee9fce3505e729b4eb32030ffc3d80b341bd1553f5`。坐标原点/基座高度从资产解析并复核；不是给程序化桌面换名。机器人资产记录Isaac/5.1版本URI；外部USD的全部传递依赖未内容冻结，完整跨机器物理复现仍需原资产环境。

## 修改文件及原因

|文件|修改|
|---|---|
|.agents/skills/geomind-research/SKILL.md；references/pipeline.md|加入实际生成/增强/场景/执行请求路由、授权边界、阶段判断与用法；保留原研究审计流程|
|geomind/tools/research_pipeline.py|统一解析/支持检查、配对Aug、CPU生成评分、串联有界PhysX、导出审计与provenance|
|geomind/tools/research_capabilities.py|从现有taxonomy/配置建立27项可复查矩阵；不另建任务注册|
|geomind/tools/generate_spatial_suite.py|保留采样分布与已有生成机制，补拒绝采样过程记录|
|geomind/episodes/isaaclab_physics.py|增加控制/全关节状态/TCP姿态轨迹；冻结查询目标；保存waypoints与终止状态|
|geomind/tools/run_episode_physics.py|保存公开观察流、计划/实测事件索引、带实例绑定的控制轨迹；旧NPZ列保持兼容|
|tests/test_research_pipeline.py|支持选择/拒绝组合、平移配对GT、动态隐藏与历史检查|
|configs/research_requests/*.json|五份实际使用的可复现请求|
|docs/research/task_capabilities.{md,json}；base_aug_mapping.md|27项能力/缺口和新旧字段映射|
|docs/research/execution_mapping.md；model_input_contract.md|补设计到代码映射、物理观察与特权数据边界|
|docs/research/progress.md；tasks/ProjectOntoLine.md；tasks/DetectSceneChange.md；本报告|更新证据、版本范围和下一入口|

## 检查与失败记录

- [65项回归](../../../outputs/research/pipeline_20261107/acceptance/unittest.log)：4项新增请求/增强/动态检查，以及原物理契约、场景、资产、任务分类、历史反事实、投影非零z与排除逻辑回归。
- [32项兼容检查](../../../outputs/research/pipeline_20261107/acceptance/compatibility.log)：episode schema、公开ID、定义schema、占位注册、release版本。合计97项，不能据数量声称27任务全覆盖。
- [导出篡改检查](../../../outputs/research/pipeline_20261107/acceptance/artifact_checks.json)：时间倒序、越界控制动作、错instance、目标不一致全部拒绝；两个Base/Aug对的名义GT平移量独立复核通过。脚本保存在同目录artifact_checks.py，执行需PYTHONPATH=.。
- 每组offline_audit.json重算数值有限/维度/时间单调、关节目标与实测限制、同实例查询状态、GT状态回放、最终误差与姿态dwell。状态回放调用生产solve，证明一致性；独立GT检查另用几何残差/实测位移。没有将状态回放或离线重算称为物理重执行。
- skill-creator quick_validate通过，skill本地链接有效，git diff --check通过。
- 初始base_single物理1正1负完成，首次离线审计将JSON dict直接传给solve而失败；修复为Snapshot.from_dict后对保存数据审计通过。failure.json、原result、原源码快照保留，正式base在修复后重新单独运行。未把初始失败覆盖成成功。
- pytest在现有默认Python和.venv未安装，改用项目原生unittest运行，未升级环境；临时验收脚本首次缺PYTHONPATH，补明确项目路径后通过。GPU沙箱不可访问时通过授权的沙箱外执行完成，没有更换后端。

## 尚未验证、研究决策与下一入口

已检查关节限制、工作空间、稳定速度与有限样例可达性；PhysX碰撞器启用，但未增加接触传感器或独立无碰撞认证，不能称所有轨迹无接触。未验证图像输入、相机可见性、学习模型、抓取/释放/稳定放置、真实机器人。Aug+第二场景组合和动态第二场景本轮未运行，不外推。

IdentityTracking：已有旧paired arc/staggered waypoint运动与初始身份末态判定，但缺统一原生连续观察、交换对应性/遮挡可辨识性验收、逐时身份切换指标和来源历史求解；TrackObjectOrigin仍明确Unavailable。没有用前后变化检测替代身份追踪。

研究者仍需决定：DSC新增/移除与位移的版本关系；三维投影正式分布和可见参考点；轴反射二维/三维语义；更新任务是否要求历史必要性；连续追踪允许的身份线索；各Aug方向是否正式采纳。上述未决项不阻断已声明工程变体的本次验收。

下一次从本报告、能力矩阵与research_pipeline入口继续。优先选一项：为TrackObjectSwap定义不泄露身份的连续观察及独立对应性验收，或给现有结构化配对接入不可变版本模型；不要重复搭建环境。新增物理组合先1例、新输出、有限时，再扩至2个seed；复用旧结果前核对provenance和定义/配置/条件/执行器版本。
