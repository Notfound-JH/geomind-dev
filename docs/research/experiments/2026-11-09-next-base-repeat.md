# GeometricInference 本次重新生成与物理执行

2026-11-09。用户指定 semantic_table、seed=0、数量2、backend=isaaclab、限时420秒。原 `outputs/research/next_base` 已存在，保留历史，使用其下独立目录 `run_20261109_repeat`，未复用历史成绩。类别按现有实现选择 ProjectOntoLine，Base、静态查询；不代表覆盖7项任务。

假设与完成标准：两例独立几何GT检查通过；真实机器人到固定查询GT上方0.1m，误差≤12mm、姿态误差≤0.15rad、连续驻留≥20步；错误目标偏移8cm被拒绝；实测轨迹、控制与录像一致。未要求或实现抓放。

已执行：新生成seeds 0/1，CPU语义校验通过；IsaacLab/PhysX cuda:0、oracle_gt真实正例2/2成功，错误目标对照1/1正确拒绝，实测轨迹离线复核3/3通过。正例TCP误差6.23/5.95mm，各驻留20步。语义GT独立检验使用线上残差和正交残差，不调用生产solve构造独立答案。semantic_table为程序化诊断桌面，资产来源保存在逐例scene_provenance；不是注册场景65。

[网页](../../../outputs/research/next_base/run_20261109_repeat/index.html)集中展示任务、实际录像和通过/未验证项。[运行结果](../../../outputs/research/next_base/run_20261109_repeat/result.json)、[物理报告](../../../outputs/research/next_base/run_20261109_repeat/physics/report.json)、[轨迹复核](../../../outputs/research/next_base/run_20261109_repeat/physics/offline_audit.json)、[视频检查](../../../outputs/research/next_base/run_20261109_repeat/video_audit.json)、[来源快照](../../../outputs/research/next_base/run_20261109_repeat/provenance.json)。来源快照ID `423c5f0f767e4a199b3427cbba39478e3bb604eaef570eedc8f3414530418b69`；配置ID `4d1b42b0dffcd5f6f6afdb826bd9f3dad3ddc941821c4ce1c5522f580f0c3956`。包装入口仅为现有物理命令追加 `--video`，原请求及包装源码副本和哈希另存source_snapshot、launcher_hashes.json，未修改执行器。

```bash
PYTHONPATH=. /home/elm/miniconda3/envs/new_isaaclab/bin/python configs/research_requests/run_next_base_video_20261109.py
/home/elm/miniconda3/envs/new_isaaclab/bin/python -m geomind.tools.research_pipeline --audit outputs/research/next_base/run_20261109_repeat
```

原命令拒绝覆盖，重跑须给独立新目录。未接模型，不报告模型表现；未执行抓取、搬运、释放或接触传感认证。无实例、系统或实现失败；负对照语义/动作失败为预期。下一项最小实验（未执行）：固定模型版本，在同直线、同GT、同偏移长度下比较零/非零z偏移配对的结构化输入投影误差，检查z插值捷径；模型接入仍缺固定版本服务与适配器。
