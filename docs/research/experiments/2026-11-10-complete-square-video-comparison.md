# CompleteSquare 原任务与 Robotiq 重生成录像对比

2026-11-10。用户要求原任务与本次生成任务的对比网页，包含视频。新增[对比页](http://10.81.44.63:8780/complete-square/)；同机可用[localhost 入口](http://localhost:8780/complete-square/)。产物位于 `outputs/research/complete_square_comparison_v1`，通过 `scene_atlas/dist/complete-square` 链接发布在已有场景展示服务。

左侧选择 `logs/geometric_inference_rollouts/square_completion/base/geometric/20260425_005116/camera_videos/square_completion__scene65__base_blocks` 的桌面、全景和腕部录像。该目录是 Base 几何录像中日期标签最新、三个视角齐全的一组；未按成功结果挑选。三个 MP4 原样复制并校验 SHA-256。原指令写“Place the held block…”；逐帧抽查 0～6.8 秒的 15 帧可见棕色方块先在桌面，随后接近、抓起、搬运、释放和退离，因此页面区分指令假设与录像实际起点。

该历史目录仅保留录像，没有同次运行的结果、seed、状态轨迹或不可变配置。Scene 65 来自目录标注；历史机器人精确 USD 版本未绑定。原片约 7.23 秒，217 帧；不能把其文件时长当作已核对的仿真时间。页面将旧版标为“历史录像，未按新版判据复核”，不填造历史成功率或落点误差。

右侧复用 `registered_robotiq_v2/CompleteSquare_v0` 已交付的实际正例、错误落点对照及 `delivery_audits/20261109_204513_7515b98e` 的独立审计。Panda + Robotiq 2F-85，注册场景 65、seed 0；原结构随机化仍生效。正例实际抓取、搬运、释放、退离、连续放稳 1 秒，目标误差 2.408225 mm，小于 10 mm 门槛；查询 GT 为 `(0.4225683510, 0.1275009755, 0.9554591412)` m。负控真正完成错误命令并放稳 1 秒：距错误命令 2.573849 mm，距原 GT 78.157495 mm，评分拒绝。

新版提供原始全景及其桌面裁切视图：原片 768×768，裁切矩形 `(x=224, y=176, width=384, height=384)` 后用 Lanczos 放大到 768×768。仅做空间裁切，737 帧、30 fps 和约 24.57 秒时长均保留，没有补帧或改变速度。页面支持两边同时从头播放、暂停、末帧、独立进度、倍速、视角切换、按各自百分比定位，以及基于原帧时钟的新动作阶段定位。

折叠的 GT 图左侧为当前原注册配置的**标称布局**，用 `B+C−A` 独立算得 `(0.48, 0.12)` m；不声称该坐标来自历史录像。右侧为新版实际查询布局与固定 GT。两段录像的 seed、相机和验证条件不配对，页面用于查看任务沿用与执行变化，不作单变量机械臂消融或模型表现比较。

本次只整理展示、裁切已有录像并验证媒体，没有重执行 PhysX 或模型评测。各视频与原始证据来源见[对比清单](../../../outputs/research/complete_square_comparison_v1/comparison.json)。验证结果另见同目录 `verification.json`。浏览器连接在本次会话不可用，未声称完成浏览器交互验收；通过原片抽帧、完整视频解码、页面引用检查和 HTTP 访问核验交付。

展示服务改由可停止的用户临时 systemd 单元 `geomind-scene-atlas.service` 托管，监听原 8780 端口，避免跟随短期命令会话退出。可用 `systemctl --user stop geomind-scene-atlas.service` 停止；未设置开机自启。

后续若要公平比较两套机械臂，最小新增实验应固定同一实测布局、相机、初态和验收阈值，分别执行并保留失败。该配对实验尚未执行，不属于这次录像对比交付。
