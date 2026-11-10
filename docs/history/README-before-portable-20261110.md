# GeoMind Lab

GeoMind Lab is a GeoMind task package built on top of IsaacLab.

This repository keeps `geomind_lab` independent from `IsaacLab` so project code
stays isolated from upstream framework code.

## Layout

- `geomind_lab/`: Python package with tasks, mimic wrappers, CLI entry points, and utilities.
- `geomind/docs/remote_policy.md`: short guide for connecting `POLICY=model` / `POLICY=agent` to a remote model service.
- `assets/`: task JSON, annotations, and other project-owned static data.
- `geomind/`: registry, task configuration, mock/native delegates and evaluation runners.
- `scripts/`: development, collection, evaluation and analysis entry points.
- `outputs/`: local run artifacts.

## Task IDs

Task definitions in `geomind/configs/task_definitions/` contain `task_id`,
`domain` (`SpatialReasoning` or `SpatialMemory`) and `family`. Episodes explicitly
reference independent scenes, layouts, parameters and memory protocols.
Existing registrations use compact v3 configs; old execution records are isolated
under `geomind/configs/legacy/`. Versioned IDs and historical aliases still resolve.
See the [v3 configuration and GT contract](docs/episode_contract_v3_zh.md) and
[registered scene reuse](docs/registered_scene_episodes_zh.md).

Generate and verify one memory task and two reasoning tasks on the CPU semantic backend:

```bash
python -m geomind.tools.generate_spatial_suite --scene-id 65 --object-set catalog --seeds 0 1 2 --verify
```

The semantic backend verifies configuration, temporal observations, GT and answer
grading. The native IsaacLab backend additionally executes Franka reach actions
using real PhysX states and joint-position actuators, with a bounded headless worker:

```bash
conda activate new_isaaclab
python -m geomind.tools.run_episode_physics --negative-controls
```

Record synchronized 720p RGB videos for every episode and package the data:

```bash
python -m geomind.tools.run_episode_physics --video --timeout-s 1800 \
  --camera-eye 1.6 1.3 1.93 --camera-look-at 0.4 0 1.23 \
  --output outputs/registered_scene_video
python -m geomind.tools.package_episode_videos --root outputs/registered_scene_video
```

Each episode has a complete observer video and a separate agent video with hidden
memory phases blacked out, plus frame timestamps, calibration and physical traces.

This reuses registered scene 65 through the original GeoMind scene loader and reaches 10 cm above
the semantic target. See the [physical verification report](docs/headless_physics_verification_zh.md).

## IsaacLab

已接入机械臂、本体选择与双臂测试命令见[机械臂使用说明](docs/robot_embodiments_zh.md)。

Copy your pinned `IsaacLab/` checkout next to this README:

```text
geomind_lab/
├── IsaacLab/
├── geomind_lab/
└── scripts/
```

The bootstrap helpers in `geomind_lab.bootstrap` expect `IsaacLab/source` to live
under that sibling directory.

## Development and migration

Use a source checkout with an editable installation:

```bash
conda activate isaac_curobo
python -m pip install -e '.[dev]'
python -m pytest tests -q
```

See [the server migration guide](docs/migration.md) for the inspected Python 3.10 /
Isaac Sim 4.5 baseline, external assets, local IsaacLab patches, validation gates
and known limitations. A standalone wheel is not a complete simulation bundle.
Experiment outputs and local environments are intentionally excluded from Git.

## 本地展示页

```bash
python -m geomind.tools.dashboard --port 8765
```

打开 http://127.0.0.1:8765 查看任务、场景、测试视频和更新记录。
新结果自动发现；在“数据清理”页预览后输入目录名确认，才会删除测试数据和视频。
详见 [展示页使用说明](docs/dashboard_zh.md)。
