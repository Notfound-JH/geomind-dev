# CompleteSquare_v0

自然操作：把棕色方块放到三个彩色方块缺失的顶点，让四个方块构成正方形。配置入口 `geomind/configs/tasks/CompleteSquare_v0.json`；运行时使用实际测得的参考方块位置重新求解并冻结 GT，再执行真实抓放。

2026-11-10：[墙体纹理复测](../experiments/2026-11-10-complete-square-wall-retest.md)已执行。保存布局 scene 65 / seed 0、Panda + Robotiq，两份米色墙体候选均通过实际抓放与有效错误落点对照，正例误差 2.446968 mm、放稳 1 秒。来源、配置、状态与录像保存在 `outputs/research/complete_square_wall_retest_v1`，通过[视频页](http://10.81.44.63:8780/complete-square-walls/)查看。

保留边界：首次共享仿真进程的负控复位出现物体偏移与关节越限；复测采用独立新进程，原问题尚未修复。没有视觉模型评测。历史原任务录像缺乏同次轨迹，不按本次标准认定通过。

2026-11-10：按用户要求将机械臂沿桌边平移 31.95 cm，对齐实际桌边中线，两份墙面的[居中版本](../experiments/2026-11-10-complete-square-centered-robot.md)均重新执行并通过：正常抓放误差 0.963326 mm、松爪放稳 1 秒，GT 世界 X +80 mm 的运输型错误对照实际完成并被正确拒绝。初始原位回放对照偏离命令 24.552200 mm 的失败保留；新增显式 offsets_first 对照顺序，默认行为、正常任务 GT、控制器与评分阈值保持一致。[居中视频与前后对比](http://10.81.44.63:8780/complete-square-centered/)。
