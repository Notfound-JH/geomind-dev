# Remote Policy 接入说明

这份文档给写远程模型 policy 的人看。GeoMind 当前不直接启动网络服务，而是在每个 step 调用一个 Python hook；hook 里可以用 HTTP / WebSocket / gRPC 去请求远程模型。

## 运行方式

远程模型建议走 `POLICY=model`。仓库已经提供了 GeoMind 侧 client hook：`geomind_client:policy`。

```bash
export GEOMIND_MODEL_POLICY_HOOK=geomind_client:policy
export REMOTE_POLICY_URL=http://127.0.0.1:8000/policy

TASK_ID_ENV=PlaceDirectional_v0 \
POLICY=model \
MAX_STEPS=100 \
BACKEND=isaaclab \
bash geomind/scripts/run_env_policy.sh
```

如果使用 `POLICY=agent`，对应环境变量是 `GEOMIND_AGENT_POLICY_HOOK`。没有设置 hook 时，`model` / `agent` 会回退到本地 oracle。

可选环境变量：

- `REMOTE_POLICY_URL`：远程服务地址，默认 `http://127.0.0.1:8000/policy`。
- `REMOTE_POLICY_TIMEOUT_S`：每步请求超时，默认 `10` 秒。
- `REMOTE_POLICY_TRACE_TAIL`：请求中携带最近多少步 trace，默认 `0`。IsaacLab 的 trace 可能很大，真实跑任务时建议保持 0。
- `REMOTE_POLICY_DEBUG=1`：打印远程请求失败原因。
- `REMOTE_POLICY_CAMERA_NAME`：从 native observation 中选择相机图像，默认 `table_cam,wrist_cam`，每次请求会同时携带这两张图。
- `REMOTE_POLICY_IMAGE_B64_TABLE_CAM` / `REMOTE_POLICY_IMAGE_B64_WRIST_CAM`：按相机名临时指定图像，优先级高于 observation 中的真实图像。
- `REMOTE_POLICY_IMAGE_B64`：旧版单图接口兼容字段；新接口优先读取 `image_b64_by_camera` 或 `images`。
- `REMOTE_POLICY_USE_DUMMY_IMAGE=1`：没有真实相机图像时，用 1x1 PNG 占位图测试通信链路。
- `REMOTE_POLICY_USE_ACTION_CHUNK=1`：启用 action chunk 缓存执行；remote IsaacLab 脚本默认开启。chunk 缓存期间不会请求 server，缓存走完后的下一步立刻用新 observation / image 请求 server。
- `REMOTE_ACTION_CHUNK_LEN`：启用 chunk 时每次最多执行多少个 action，默认 `3`，即 server 返回 chunk 时只取前三个动作闭环执行。
- `REMOTE_POLICY_TRANSLATION_SCALE`：发送给 IsaacLab 前对 action[0:3] 乘的系数，默认 `1.0`。
- xyz 总范数 clipping 已禁用。
- `REMOTE_POLICY_MAX_XY_NORM_M`：xy 单步合位移上限，默认 `0`，表示不限制。
- `REMOTE_POLICY_MAX_XY_COMPONENT_M`：x/y 单轴单步位移上限，默认 `0`，表示不限制。
- `REMOTE_POLICY_GOAL_ALIGNED_XY_GUARD=1`：可选开启。有 goal_error 时，禁止 xy 动作背离目标方向，并限制单步 xy 不超过剩余目标距离；默认关闭。
- `REMOTE_POLICY_WORKSPACE_BOUNDS`：末端 workspace clipping，格式 `xmin,xmax,ymin,ymax,zmin,zmax`，默认 `0.15,0.75,-0.45,0.45,0.70,1.20`。
- `REMOTE_POLICY_ROTATION_SCALE`：发送给 IsaacLab 前对 action[3:6] 乘的系数，默认 `2.0`，用于抵消 IK action term 的 `scale=0.5`。
- `REMOTE_POLICY_DISABLE_ROTATION=1`：调试阶段默认禁用 rotation。
- `REMOTE_POLICY_GRIPPER_MODE`：夹爪动作适配模式，默认 `isaac`，即 raw gripper 正值=open、负值=close；如果模型输出约定相反，可设为 `sign_invert`。OpenPI 评测可设为 `hold_initial`，忽略模型输出的夹爪维度，按 episode 初始 `gripper_pos` 持续发送 open/close 命令。
- `REMOTE_POLICY_BINARIZE_GRIPPER=1`：夹爪单独二值化，不混入 xyz/rpy 连续控制。
- `REMOTE_POLICY_OVERSHOOT_GUARD=1`：下一步预测会远离目标，或观测到距离变大时，停止 xy 运动。

## Hook 签名

hook 必须是可 import 的 `module:function`：

```python
def policy(env, raw_obs: dict, step: int, trace: list[dict]) -> list[float]:
    ...
```

参数含义：

- `env`：本地 gym env，可读 `env.action_space.shape` 得到动作维度。
- `raw_obs`：当前 step 的完整 observation。
- `step`：当前 episode step，从 0 开始。
- `trace`：已完成 step 的轨迹列表，通常只读。

返回值：

- 返回一维 action list / numpy array。
- 长度最好等于 `action_dim = env.action_space.shape[0]`。
- GeoMind 会把 action 转成 `float32`，过长会截断，过短会补 0。
- action 通常限制在 `[-1, 1]`。

Franka IK 相对控制常用 7 维动作：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

其中前 3 维是末端相对位移，3-5 维是相对旋转轴角，最后 1 维是夹爪命令。mock 后端只使用前 2 维。

## Observation 结构

`raw_obs` 主要字段：

```python
{
    "instruction": "...",          # 自然语言任务说明
    "timestep": {
        "episode_step": 0,
        "max_episode_steps": 100,
    },
    "state": {
        "native_obs": {...},       # IsaacLab 后端
        # 或 "backend_obs": {...}  # mock 后端
    },
    "cameras": {
        # 当前通常是占位；后续可放 RGB / depth
    },
    "action": {
        "joint_action": [...],     # 上一步 action
        "ee_action": [],
        "joint_velocity": [],
        "extra": {},
    },
    "meta": {
        "task_id": "...",
        "family": "...",
        "scene_id": 65,
        "layout_key": "...",
        "runtime_mode": "...",
        ...
    },
}
```

远程模型一般只需要 `instruction`、`state`、`cameras`、`meta` 和最近几步 `trace`。

## GeoMind Client

`geomind_client.py` 中已经实现了可直接使用的 hook：

```python
def policy(env, raw_obs, step, trace):
    ...
```

它每步会向 `REMOTE_POLICY_URL` POST JSON。默认推荐远程服务实现 `/policy`，返回：

请求中的图像字段：

```json
{
  "image_b64_by_camera": {
    "table_cam": "...",
    "wrist_cam": "..."
  },
  "images": [
    {"camera": "table_cam", "image_b64": "..."},
    {"camera": "wrist_cam", "image_b64": "..."}
  ],
  "image_b64": "..."
}
```

`image_b64` 保留给旧版单图服务，默认等于 `REMOTE_POLICY_CAMERA_NAME` 中第一张可用图。

```json
{
  "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
}
```

也兼容部分推理服务的嵌套返回：

```json
{
  "ok": true,
  "result": {
    "action": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
  }
}
```

如果远程服务返回 action chunk，例如 `[[a0], [a1], ...]`：

- 默认 remote IsaacLab 脚本设置 `REMOTE_POLICY_USE_ACTION_CHUNK=1 REMOTE_ACTION_CHUNK_LEN=3`，client 只取前三个动作 `a0,a1,a2`。
- 执行 `a0` 后缓存 `a1,a2`；step1/step2 使用缓存动作，并用当前 observation 重新做本地 adapter / clipping；缓存全部走完后，下一步立刻用新 observation / image 请求 server。
- 设置 `REMOTE_POLICY_USE_ACTION_CHUNK=0` 时，GeoMind client 只执行第一个动作 `a0`，下一次 env step 用新 observation / image 再请求 server。
- 如果 server 只返回一个 7 维动作但启用了 chunk，client 会把这一个动作重复为 `N` 步执行。

当前 runtime 只消费 `action`。如果远程服务还返回解释、置信度或 debug 信息，需要 hook 自己记录日志。

## 注意事项

- 每个 step 都会同步等待 hook 返回，远程请求要设置 timeout。
- hook 抛异常会中断 rollout；建议在 hook 内部兜底返回零动作或安全动作。
- 不要把 `env` 直接发给远程服务，它不能 JSON 序列化。
- 需要真实仿真时用 `BACKEND=isaaclab`；快速联调用 `BACKEND=mock` 更轻。
