# 异路径迁移验证（2026-11-11）

已在独立代码目录、独立 IsaacLab clone、独立 venv 及解压资产目录完成一次当前源码真实操作 smoke。运行时 commit 为 `0289dd2f18be9d4f589e105b9e48515636c712ff`；固定框架为 `54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0`＋`minimal-runtime.patch`，四个文件哈希及 Python 3.11 编译均通过。实际项目/框架导入路径和 editable 元数据均指向新前缀。

- 验证实例：`SelectByIndirectReference_v0`，scene65、seed0、Panda＋Robotiq；每物理 worker 最多420秒、动作预算12000ticks。
- 正例实际抬升 **145.232mm**；错误对象负控实际抬升 **145.561mm**。两者均连续持稳240ticks（1秒）；错误身份由冻结GT正确拒绝。
- 完整流水线、独立原始状态/命令审计、腕相机审计和源码/配置来源均通过。正负外部/腕部首末共8帧已查看，墙体、桌面材质正常，腕视图可见持物和参照，没有旧灰屏自遮挡。
- 行动后重新深度核对 **2489个资产文件**，缺失0、SHA-256不符0；资产manifest前后均为 `b13fa35efe7f79b8669f530e62d8693707d72bf94a3f6549e07e4194ad1586a9`。记录中的场景与机器人依赖均来自解压bundle或该次新输出。
- 首次迁移v1曾因离线USD环境问题整体失败；其原结果及SHA保持不变，未用后来CPU回放回填成功。完整迁移通过来自新v2。

这次 venv 共享本机已安装的 Isaac Sim 5.1/PyTorch等SDK及第三方依赖，**不是从零安装SDK或另一硬件的验证**。目标主机仍须按安装指南执行preflight及有界smoke；没有把原27类展示追认为全部在新便携源码上重跑，也没有模型成绩。

完整验证记录：[relocation_final_verification.json](../../migration/verification/relocation_final_verification.json)；source snapshot `e19229b34146e5abd598f7eeae56c62f9da62caa816a53e6f695f2a36845488e`，configuration `52a109f4fa35eac04c8737a450dfba96dafeb7018a32ebdf6692dff029024f5a`。
