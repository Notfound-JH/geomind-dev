#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import traceback


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.bootstrap.isaaclab_path import ensure_isaaclab_source_on_path


ISAACLAB_SOURCE_DIR = ensure_isaaclab_source_on_path(REPO_ROOT)
for package_dir in (
    ISAACLAB_SOURCE_DIR / "isaaclab",
    ISAACLAB_SOURCE_DIR / "isaaclab_tasks",
    ISAACLAB_SOURCE_DIR / "isaaclab_mimic",
    ISAACLAB_SOURCE_DIR / "isaaclab_assets",
):
    package_dir_str = str(package_dir)
    if package_dir_str not in sys.path:
        sys.path.insert(0, package_dir_str)

# Keep renderer load low by creating only scene_cam for this script.
os.environ["GEOMIND_ACTIVE_CAMERAS"] = "scene_cam"

if sys.platform != "win32":
    try:
        import pinocchio  # noqa: F401
    except Exception:
        pass

from isaaclab.app import AppLauncher


@dataclass(frozen=True)
class TaskInstance:
    task_id: str
    scene_id: int
    layout_key: str
    task_index: int


DEFAULT_TASK_IDS = (
    "perfect_midpoint",
    "isometric_extension",
    "point_symmetry",
    "square_completion",
    "trend_continuation",
)


def _parse_task_ids(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return tuple(DEFAULT_TASK_IDS)
    values = [chunk.strip() for chunk in str(raw).replace(";", ",").split(",")]
    return tuple(value for value in values if value)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture randomized scene_cam images for geometric tasks. "
            "Each requested task exports at least N captures using different layout seeds."
        )
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=",".join(DEFAULT_TASK_IDS),
        help="Comma-separated task_id list. Default: perfect_midpoint,isometric_extension,point_symmetry,square_completion,trend_continuation",
    )
    parser.add_argument(
        "--include-non-vla-layouts",
        action="store_true",
        default=False,
        help="Include legacy non-_vla layouts. Default keeps only *_vla layouts to match geometric colored-block tasks.",
    )
    parser.add_argument(
        "--captures-per-scene",
        type=int,
        default=2,
        help="Minimum captures per task. The option name is retained for compatibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: logs/geometric_randomized_scene_cam/<timestamp>",
    )
    parser.add_argument("--resolution", type=int, default=1024, help="scene_cam square resolution.")
    parser.add_argument("--settle-steps", type=int, default=24, help="Zero-action settle steps after reset.")
    parser.add_argument("--meta-seed", type=int, default=20260824, help="Base seed for jitter/shift seed sampling.")
    parser.add_argument("--jitter-xy", type=float, default=0.02, help="GEOMIND_LAYOUT_JITTER_XY")
    parser.add_argument("--shift-max", type=float, default=0.15, help="GEOMIND_LAYOUT_SHIFT_MAX")
    parser.add_argument("--min-clearance", type=float, default=0.025, help="GEOMIND_LAYOUT_MIN_CLEARANCE")
    parser.add_argument(
        "--disable-ground-mdl",
        action="store_true",
        default=True,
        help="Set GEOMIND_DISABLE_GROUND_MDL=1 to avoid remote ground MDL texture loading.",
    )
    parser.add_argument(
        "--layout-bounds",
        type=str,
        default="0.15,0.68,-0.32,0.34",
        help="GEOMIND_LAYOUT_BOUNDS",
    )
    parser.add_argument("--disable-fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = build_argparser().parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

simulation_app = None
try:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
except Exception:
    traceback.print_exc()
    raise SystemExit(1)


import omni.usd
import torch
from torchvision.io import write_png

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass

from geomind_lab.tasks.manager_based.manipulation.geomind.config.franka import FrankaGeoMindIKRelVisuomotorEnvCfg
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import load_annotated_tasks


def _safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _set_scene_cam_resolution(scene_cfg, resolution: int) -> None:
    scene_cam = getattr(scene_cfg, "scene_cam", None)
    if isinstance(scene_cam, CameraCfg):
        scene_cam.height = int(resolution)
        scene_cam.width = int(resolution)


def _build_runtime_env_cfg(*, scene_id: int, layout_key: str, task_index: int, resolution: int):
    def __post_init__(self):
        self.scene_id = int(scene_id)
        self.task_layout_key = str(layout_key)
        self.task_index = int(task_index)
        self.task_mode = "vla"
        self.object_mode = "geometric"
        self.scene.num_envs = 1
        self.sim.device = args_cli.device
        self.sim.use_fabric = not args_cli.disable_fabric

        FrankaGeoMindIKRelVisuomotorEnvCfg.__post_init__(self)
        _set_scene_cam_resolution(self.scene, resolution)

    runtime_cfg_cls = configclass(
        type(
            "RuntimeGeoMindRandomizedSceneCamCfg",
            (FrankaGeoMindIKRelVisuomotorEnvCfg,),
            {
                "__doc__": "Runtime cfg for randomized scene_cam capture.",
                "scene_id": int(scene_id),
                "task_layout_key": str(layout_key),
                "task_index": int(task_index),
                "task_mode": "vla",
                "object_mode": "geometric",
                "__post_init__": __post_init__,
            },
        )
    )
    return runtime_cfg_cls()


def _collect_instances(task_ids: tuple[str, ...]) -> dict[int, list[TaskInstance]]:
    task_id_set = {str(task_id).strip() for task_id in task_ids if str(task_id).strip()}
    payload = load_annotated_tasks()
    by_scene: dict[int, list[TaskInstance]] = defaultdict(list)
    for layout_key, layout in payload.items():
        if not isinstance(layout, dict):
            continue
        if not args_cli.include_non_vla_layouts and not str(layout_key).endswith("_vla"):
            continue
        scene_id = int(layout.get("scene_id", 1))
        tasks = layout.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        for task_index, task_payload in enumerate(tasks):
            if not isinstance(task_payload, dict):
                continue
            annotation = task_payload.get("annotation", {})
            if not isinstance(annotation, dict):
                continue
            task_id = str(annotation.get("task_id", "")).strip()
            if task_id not in task_id_set:
                continue
            by_scene[scene_id].append(
                TaskInstance(
                    task_id=task_id,
                    scene_id=scene_id,
                    layout_key=str(layout_key),
                    task_index=int(task_index),
                )
            )
    return {scene_id: instances for scene_id, instances in sorted(by_scene.items()) if instances}


def _make_zero_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.action_space.shape, device=env.device)


def _should_reset(terminated, truncated) -> bool:
    if terminated is not None and torch.any(terminated):
        return True
    if truncated is not None and torch.any(truncated):
        return True
    return False


def _settle_env(env: ManagerBasedRLEnv, settle_steps: int) -> None:
    if settle_steps <= 0:
        return
    zero_actions = _make_zero_actions(env)
    for _ in range(int(settle_steps)):
        _, _, terminated, truncated, _ = env.step(zero_actions)
        if _should_reset(terminated, truncated):
            env.reset()


def _save_scene_cam(env: ManagerBasedRLEnv, output_path: Path) -> None:
    sensor = env.scene.sensors.get("scene_cam")
    if not isinstance(sensor, Camera):
        raise RuntimeError("scene_cam sensor was not created.")

    sensor.update(dt=0.0, force_recompute=True)
    rgb = sensor.data.output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        raise RuntimeError("scene_cam produced no RGB output.")

    image = rgb[0].detach().cpu()
    if image.dtype != torch.uint8:
        image = image.to(torch.uint8)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise RuntimeError(f"Unexpected scene_cam RGB tensor shape: {tuple(image.shape)}")

    image = image[..., :3].permute(2, 0, 1).contiguous()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(image, str(output_path))


def _new_stage() -> None:
    omni.usd.get_context().new_stage()


def _seed_pair(rng: random.Random) -> tuple[int, int]:
    # Avoid zero to keep seed behavior explicit in env vars.
    return (rng.randrange(1, 2**31 - 1), rng.randrange(1, 2**31 - 1))


def _capture_one(
    *,
    instance: TaskInstance,
    output_dir: Path,
    capture_index: int,
    jitter_seed: int,
    shift_seed: int,
) -> dict[str, object]:
    env = None
    os.environ["GEOMIND_LAYOUT_JITTER_ENABLE"] = "1"
    os.environ["GEOMIND_LAYOUT_SHIFT_ENABLE"] = "1"
    os.environ["GEOMIND_LAYOUT_JITTER_SEED"] = str(int(jitter_seed))
    os.environ["GEOMIND_LAYOUT_SHIFT_SEED"] = str(int(shift_seed))
    os.environ["GEOMIND_LAYOUT_JITTER_XY"] = str(float(args_cli.jitter_xy))
    os.environ["GEOMIND_LAYOUT_SHIFT_MAX"] = str(float(args_cli.shift_max))
    os.environ["GEOMIND_LAYOUT_MIN_CLEARANCE"] = str(float(args_cli.min_clearance))
    os.environ["GEOMIND_LAYOUT_BOUNDS"] = str(args_cli.layout_bounds)
    os.environ["GEOMIND_LAYOUT_VERBOSE"] = "0"
    if bool(args_cli.disable_ground_mdl):
        os.environ["GEOMIND_DISABLE_GROUND_MDL"] = "1"

    scene_dir = output_dir / f"task_{_safe_filename(instance.task_id)}" / f"scene_{int(instance.scene_id)}"
    image_name = (
        f"capture_{int(capture_index):02d}"
        f"__{_safe_filename(instance.task_id)}"
        f"__{_safe_filename(instance.layout_key)}"
        f"__j{int(jitter_seed)}_s{int(shift_seed)}.png"
    )
    image_path = scene_dir / image_name

    try:
        env_cfg = _build_runtime_env_cfg(
            scene_id=int(instance.scene_id),
            layout_key=str(instance.layout_key),
            task_index=int(instance.task_index),
            resolution=int(args_cli.resolution),
        )
        env = ManagerBasedRLEnv(cfg=env_cfg)
        env.reset()
        with torch.inference_mode():
            _settle_env(env, int(args_cli.settle_steps))
        _save_scene_cam(env, image_path)

        task_spec = dict(getattr(env.cfg, "task_spec", {}) or {})
        return {
            "ok": True,
            "scene_id": int(instance.scene_id),
            "task_id": str(instance.task_id),
            "layout_key": str(instance.layout_key),
            "task_index": int(instance.task_index),
            "capture_index": int(capture_index),
            "jitter_seed": int(jitter_seed),
            "shift_seed": int(shift_seed),
            "image_path": str(image_path),
            "layout_randomization_record": task_spec.get("applied_layout_randomization", {}),
            "object_positions": task_spec.get("object_positions", {}),
            "error": None,
        }
    except Exception as exc:
        traceback.print_exc()
        return {
            "ok": False,
            "scene_id": int(instance.scene_id),
            "task_id": str(instance.task_id),
            "layout_key": str(instance.layout_key),
            "task_index": int(instance.task_index),
            "capture_index": int(capture_index),
            "jitter_seed": int(jitter_seed),
            "shift_seed": int(shift_seed),
            "image_path": str(image_path),
            "layout_randomization_record": {},
            "object_positions": {},
            "error": str(exc),
        }
    finally:
        if env is not None:
            env.close()


def main() -> int:
    captures_per_task = max(2, int(args_cli.captures_per_scene))
    task_ids = _parse_task_ids(args_cli.task_ids)
    by_scene = _collect_instances(task_ids)
    if not by_scene:
        raise SystemExit(f"No annotated task instances found for task_ids={task_ids}")

    instances_by_task: dict[str, TaskInstance] = {}
    for instances in by_scene.values():
        for instance in instances:
            instances_by_task.setdefault(instance.task_id, instance)
    missing_task_ids = [task_id for task_id in task_ids if task_id not in instances_by_task]
    if missing_task_ids:
        raise SystemExit(f"No annotated VLA task instances found for task_ids={missing_task_ids}")
    selected_instances = [instances_by_task[task_id] for task_id in task_ids]

    output_dir = (
        args_cli.output_dir.expanduser().resolve()
        if args_cli.output_dir is not None
        else (REPO_ROOT / "logs" / "geometric_randomized_scene_cam" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("GeoMind Geometric Randomized scene_cam Capture", flush=True)
    print("=" * 72, flush=True)
    print(f"task_ids:            {', '.join(task_ids)}", flush=True)
    print(f"captures_per_task:   {captures_per_task}", flush=True)
    print(f"resolution:          {int(args_cli.resolution)}", flush=True)
    print(f"settle_steps:        {int(args_cli.settle_steps)}", flush=True)
    print(f"output_dir:          {output_dir}", flush=True)
    print(f"scene_ids:           {sorted({instance.scene_id for instance in selected_instances})}", flush=True)
    print("=" * 72, flush=True)

    rng = random.Random(int(args_cli.meta_seed))
    results: list[dict[str, object]] = []

    total_jobs = len(selected_instances) * captures_per_task
    job_index = 0
    for instance in selected_instances:
        for capture_index in range(captures_per_task):
            if job_index > 0:
                _new_stage()
            jitter_seed, shift_seed = _seed_pair(rng)
            job_index += 1
            print(
                f"[{job_index}/{total_jobs}] scene={instance.scene_id} task={instance.task_id} "
                f"layout={instance.layout_key} j={jitter_seed} s={shift_seed}",
                flush=True,
            )
            result = _capture_one(
                instance=instance,
                output_dir=output_dir,
                capture_index=capture_index,
                jitter_seed=jitter_seed,
                shift_seed=shift_seed,
            )
            results.append(result)
            if result["ok"]:
                print(f"  saved: {result['image_path']}", flush=True)
            else:
                print(f"  failed: {result['error']}", flush=True)

    summary = {
        "task_ids": list(task_ids),
        "captures_per_task": captures_per_task,
        "resolution": int(args_cli.resolution),
        "settle_steps": int(args_cli.settle_steps),
        "meta_seed": int(args_cli.meta_seed),
        "jitter_xy": float(args_cli.jitter_xy),
        "shift_max": float(args_cli.shift_max),
        "min_clearance": float(args_cli.min_clearance),
        "layout_bounds": str(args_cli.layout_bounds),
        "results": results,
    }
    summary_path = output_dir / "capture_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    per_task_success: dict[str, int] = defaultdict(int)
    for item in results:
        if item.get("ok"):
            per_task_success[str(item.get("task_id", ""))] += 1

    failures: list[str] = []
    for task_id in task_ids:
        count = int(per_task_success.get(task_id, 0))
        if count < captures_per_task:
            failures.append(f"task {task_id}: expected {captures_per_task} captures, got {count}")

    print("=" * 72, flush=True)
    print(f"summary_path: {summary_path}", flush=True)
    print(f"per_task_success: {dict(sorted(per_task_success.items()))}", flush=True)
    if failures:
        print("failures:", flush=True)
        for item in failures:
            print(f"  - {item}", flush=True)
        print("=" * 72, flush=True)
        return 1
    print("status: success", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if simulation_app is not None:
            simulation_app.close()
