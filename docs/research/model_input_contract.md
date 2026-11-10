# 结构化语义模型输入契约（待接模型）

状态：纯数据导出和字段校验已实现；**没有远程服务或真实模型适配器，没有模型成绩**。[research_pairs.py](../../geomind/tools/research_pairs.py) 的 `public_observation`、`condition_payload` 只返回脱离环境的 JSON。

## 白名单与信息隔离

模型请求的数据体只有 `observations` 数组。每项必须恰好包含 instruction、phase、step、scene_visible、objects、frame、allow_response；objects 的每项恰好包含 object_id、label、position_m。隐藏阶段 objects=[]。均为结构化精确坐标；无图像、机器人状态、事件计划、pair ID、seed、文件名、GT、env.info、provenance 或 raw.meta。未知字段拒绝，不静默透传。本工具的固定标签为 object 0/1/2，无 target 等角色提示；未来自由文本配置仍须审计语义泄露，字段白名单不能自动消毒文本。

评估端保存 pair_index、答案、配置和 payload 文件名；模型只接收文件内的数据体，不能拥有文件系统、环境对象或评估目录访问权。输出目录中的 `payload_candidates` 是**待发送数据样例**，不是实际服务请求证据。

## 两种条件与缓存

- full-history：一次请求按 step=0…5 顺序传入全部六条公开观测。相当于评估端保留 agent 当时可见的历史；没有读取隐藏事件/环境历史快照。环境 allow_history_replay=false 不禁止 agent 在 episode 内自存已见信息。此条件测显式上下文利用，不能直接称自主长期记忆。
- final-only：一次请求只传 step=5 的一条公开观测。每对 A/B 请求数据体字节相同；不会告诉模型这是配对。
- 两条件每个 episode 都建立全新模型上下文，清空对话、动作缓存、检索记忆和服务 session；不得跨 episode 留存状态。后续若采用流式历史缓存，应另定协议和记录版本，不与一次请求混报。
- 若服务无法证明 stateless 或提供可靠 reset，记录接口阻碍并停止该条件评估；不能用额外 Wait 代替隔离，也不能把 Wait 数量当成长时记忆证据。

## 后续接入最少信息

需要真实模型/服务名称、不可变 checkpoint digest 或 revision、服务/适配器版本，及支持的请求/回答 schema（本任务期望 `{"object_id":"object_N"}`）、上下文/reset 机制、解码参数/随机种子、超时/重试规则和预算。浮动 alias 单独记录为版本未固定，不能假称可复现。凭据由现有安全配置加载，不进入记录。

实际评估保存脱敏**实际发出的**请求数据体、系统指令/模板版本、请求哈希、模型不可变身份、真实响应和解析结果；不保存 auth headers、token、含凭据 URL。若传输封装另增字段，必须审计后列入新白名单。配对映射只在评估端，随机 transport request ID 不进入模型上下文。

语义错误（合法回答错误）、无效实例（生成/协议不合法）、系统故障（加载、超时、格式/解析失败）分别统计并报告分母；有动作才记动作失败。本轮无动作项写未执行。实现校验失败单独阻断实验。模型加载失败不可切换 oracle；重试保留全部尝试，不能只保留最好成绩。

## 现有适配器的具体缺口

[geomind_client.py](../../geomind_client.py) 的通用 `build_payload` 会发送 observation/meta，不能直接符合本白名单；`_build_openpi_infer_element` 是图像/本体状态/动作接口，不是这里的对象 ID 报告接口。[policy_runtime.py](../../geomind/runners/policy_runtime.py) 的 `build_policy` 有缺少 hook 时返回 oracle 的路径，不能用作模型实验的透明后备。本轮复用原生 `SemanticEpisode`，只新增传输无关数据导出，没有为未知服务伪造可用集成。接模型时需一个小的 fail-closed 适配器：纯 JSON 输入→真实服务→严格答案解析；评估器独占 env 和 GT。

## 2026-11-07 原生物理流水线观察导出

`research_pipeline` 新增的 `physics/observations/*.json` 保存真实环境在各协议步公开的完整流：上述七字段之外另含 `robot`，内容仅为实测关节位置和table坐标系TCP位置。它不是此页既有纯语义模型白名单的悄悄扩展；送入仅接受七字段的适配器前必须明确选择输入条件。`audit_physics` 对移除robot后的语义字段执行严格校验。机器人动作发生于查询回答之后；查询前流不包含GT、事件对象、特权历史或未来动作。

本轮没有将这些数据发给模型，没有模型身份或模型成绩。精确坐标和稳定ID是允许输入，不可报告为纯视觉或无身份线索的追踪任务。事件计划、完整物理trace、reference history、case/provenance都属于评估端，不能把整个输出目录交给模型。
