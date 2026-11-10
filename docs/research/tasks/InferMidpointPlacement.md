# InferMidpointPlacement 研究卡

自然操作变体草案，非已实现注册任务。指令：把绿色方块放到红蓝两个方块正中间。

[本次设计证据](../experiments/2026-11-09-midpoint-placement-design.md)：2例布局/GT已生成，独立构造与既有InferMidpoint语义求解交叉核验2/2；抓取释放、实测支撑与可达性、物体末态验收均未执行。下一步为操作执行链与物理标定。

## 真实物理执行（2026-11-09）

[本次物理实验](../experiments/2026-11-09-midpoint-placement-physics.md)：已补齐有限PickPlaceMidpoint执行链，原生场景65 seeds0/1真实抓放2/2成功、错误目标拒绝1/1、独立实测轨迹审计3/3。物体末态误差约1.02/1.04mm，各稳定1秒。模型评估未执行；通用manipulation生成入口尚未接通。
