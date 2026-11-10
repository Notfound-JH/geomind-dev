# ProjectOntoLine 实际重执行与录像

2026-11-09，使用上一轮生成的 semantic_table Base seeds 0/1 配置，复制到独立目录，未重新采样。原来源一致性检查通过。假设与完成标准：两例GT几何检查及ReachAboveTarget成功，错误目标对照正确拒绝，实测轨迹和录像一致性通过。

已执行：IsaacLab/PhysX cuda:0，正例2/2成功，错误目标对照1/1正确拒绝，轨迹复核3/3通过。末端误差6.23/5.95mm，各连续驻留20步。视频6/6完成解码帧数、首尾帧读取和SHA256检查，并人工查看seed0末帧，桌面、机器人与三对象可见。源配置来源及当前源码快照一致性通过。

[请求与命令](../../../outputs/research/next_base_video/request.json)；[物理报告](../../../outputs/research/next_base_video/physics/report.json)；[轨迹审计](../../../outputs/research/next_base_video/physics/offline_audit.json)；[录像检查和快照ID](../../../outputs/research/next_base_video/result.json)；[源码内容快照](../../../outputs/research/next_base_video/provenance.json)。

```bash
/home/elm/miniconda3/envs/new_isaaclab/bin/python -m geomind.tools.run_episode_physics --root outputs/research/next_base_video/configs --suite research_request --output outputs/research/next_base_video/physics --timeout-s 420 --negative-controls --video
/home/elm/miniconda3/envs/new_isaaclab/bin/python -m geomind.tools.research_pipeline --audit outputs/research/next_base_video
```

本次是真实物理重执行，后续审计为离线复算。仅oracle_gt、到投影点上方0.1m，无受评模型、抓放或无接触认证。无实例/系统/实现失败；错误对照的语义及动作失败为预期结果。报告写出后应用退出采用有界清理，详见shutdown字段。下一最小实验仍为固定版本模型的非零z偏移配对比较（未执行）。
