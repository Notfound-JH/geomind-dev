#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


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

# Keep renderer load low by creating only scene_cam for this visualization script.
os.environ["GEOMIND_ACTIVE_CAMERAS"] = "scene_cam"

if sys.platform != "win32":
    try:
        import pinocchio  # noqa: F401
    except Exception:
        pass

from isaaclab.app import AppLauncher


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture scene_cam images for all fixed GeoMind lighting-pool variants "
            "(default 15 variants)."
        )
    )
    parser.add_argument("--scene-id", type=int, default=1, help="Scene id to render.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "lighting_pool_scene_cam",
        help="Directory for output PNG images and capture summary.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Square scene_cam RGB output resolution.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=30,
        help="Number of zero-action settle steps after reset.",
    )
    parser.add_argument(
        "--base-lighting-mode",
        type=str,
        default="natural",
        choices=("natural", "bright"),
        help="Base preset used before applying pool deltas.",
    )
    parser.add_argument(
        "--disable-fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O instead.",
    )
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

from geomind_lab.tasks.manager_based.manipulation.geomind.config.franka import (
    FrankaGeoMindIKRelVisuomotorEnvCfg,
)
from geomind_lab.tasks.manager_based.manipulation.geomind.geomind_scene_cfg import LIGHTING_POOL


def build_runtime_env_cfg(scene_id: int, lighting_mode: str, resolution: int):
    def __post_init__(self):
        self.scene_id = int(scene_id)
        self.task_layout_key = "__lighting_pool_capture__"
        self.task_index = 0
        self.task_mode = "vla"
        self.lighting_mode = str(lighting_mode)
        self.scene.num_envs = 1
        self.sim.device = args_cli.device
        self.sim.use_fabric = not args_cli.disable_fabric

        FrankaGeoMindIKRelVisuomotorEnvCfg.__post_init__(self)
        _set_scene_cam_resolution(self.scene, resolution)

    runtime_cfg_cls = configclass(
        type(
            "LightingPoolSceneCamEnvCfg",
            (FrankaGeoMindIKRelVisuomotorEnvCfg,),
            {
                "__doc__": "Runtime GeoMind cfg for scene_cam lighting-pool capture.",
                "scene_id": int(scene_id),
                "task_layout_key": "__lighting_pool_capture__",
                "task_index": 0,
                "task_mode": "vla",
                "lighting_mode": str(lighting_mode),
                "__post_init__": __post_init__,
            },
        )
    )
    return runtime_cfg_cls()


def _set_scene_cam_resolution(scene_cfg, resolution: int) -> None:
    scene_cam = getattr(scene_cfg, "scene_cam", None)
    if isinstance(scene_cam, CameraCfg):
        scene_cam.height = int(resolution)
        scene_cam.width = int(resolution)


def make_zero_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.action_space.shape, device=env.device)


def should_reset(terminated, truncated) -> bool:
    if terminated is not None and torch.any(terminated):
        return True
    if truncated is not None and torch.any(truncated):
        return True
    return False


def settle_env(env: ManagerBasedRLEnv, settle_steps: int) -> None:
    if settle_steps <= 0:
        return
    zero_actions = make_zero_actions(env)
    for _ in range(int(settle_steps)):
        _, _, terminated, truncated, _ = env.step(zero_actions)
        if should_reset(terminated, truncated):
            env.reset()


def save_scene_cam(env: ManagerBasedRLEnv, output_path: Path) -> None:
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


def new_stage() -> None:
    omni.usd.get_context().new_stage()


def run_single_variant(scene_id: int, variant_index: int, output_dir: Path) -> dict[str, object]:
    env = None
    variant = LIGHTING_POOL[variant_index]
    variant_name = variant.name
    os.environ["GEOMIND_LIGHTING_POOL_ENABLE"] = "1"
    os.environ["GEOMIND_LIGHTING_POOL_INDEX"] = str(variant_index)
    os.environ.pop("GEOMIND_LIGHTING_POOL_SEED", None)

    try:
        env_cfg = build_runtime_env_cfg(scene_id, args_cli.base_lighting_mode, args_cli.resolution)
        env = ManagerBasedRLEnv(cfg=env_cfg)
        env.reset()

        with torch.inference_mode():
            settle_env(env, args_cli.settle_steps)

        output_path = output_dir / f"lighting_{variant_index:02d}_{variant_name}.png"
        save_scene_cam(env, output_path)
        print(f"[variant {variant_index:02d}] saved {output_path}", flush=True)
        return {
            "variant_index": int(variant_index),
            "variant_name": str(variant_name),
            "lighting_variant": {
                "ambient_scale": float(variant.ambient_scale),
                "key_scale": float(variant.key_scale),
                "fill_scale": float(variant.fill_scale),
                "lamp_scale": float(variant.lamp_scale),
                "ambient_color": list(variant.ambient_color),
                "key_color": list(variant.key_color),
                "fill_color": list(variant.fill_color),
                "lamp_color": list(variant.lamp_color),
                "key_delta_xyz": list(variant.key_delta_xyz),
                "fill_delta_xyz": list(variant.fill_delta_xyz),
                "lamp_delta_z": float(variant.lamp_delta_z),
            },
            "image_path": str(output_path),
            "error": None,
        }
    except Exception as exc:
        print(f"[variant {variant_index:02d}] failed: {exc}", flush=True)
        traceback.print_exc()
        return {
            "variant_index": int(variant_index),
            "variant_name": str(variant_name),
            "lighting_variant": {
                "ambient_scale": float(variant.ambient_scale),
                "key_scale": float(variant.key_scale),
                "fill_scale": float(variant.fill_scale),
                "lamp_scale": float(variant.lamp_scale),
                "ambient_color": list(variant.ambient_color),
                "key_color": list(variant.key_color),
                "fill_color": list(variant.fill_color),
                "lamp_color": list(variant.lamp_color),
                "key_delta_xyz": list(variant.key_delta_xyz),
                "fill_delta_xyz": list(variant.fill_delta_xyz),
                "lamp_delta_z": float(variant.lamp_delta_z),
            },
            "image_path": None,
            "error": str(exc),
        }
    finally:
        if env is not None:
            env.close()


def main() -> int:
    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GeoMind Lighting Pool scene_cam Capture")
    print("=" * 72)
    print(f"scene_id:            {args_cli.scene_id}")
    print(f"resolution:          {args_cli.resolution}x{args_cli.resolution}")
    print(f"settle_steps:        {args_cli.settle_steps}")
    print(f"base_lighting_mode:  {args_cli.base_lighting_mode}")
    print(f"pool_size:           {len(LIGHTING_POOL)}")
    print(f"output_dir:          {output_dir}")
    print("=" * 72)

    results: list[dict[str, object]] = []
    for variant_index in range(len(LIGHTING_POOL)):
        if variant_index > 0:
            new_stage()
        results.append(run_single_variant(args_cli.scene_id, variant_index, output_dir))

    summary = {
        "scene_id": int(args_cli.scene_id),
        "resolution": int(args_cli.resolution),
        "settle_steps": int(args_cli.settle_steps),
        "base_lighting_mode": str(args_cli.base_lighting_mode),
        "pool_size": len(LIGHTING_POOL),
        "results": results,
    }
    summary_path = output_dir / "capture_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    failed = sum(1 for item in results if item["error"])
    print("=" * 72)
    print(f"summary_path: {summary_path}")
    print(f"failed:       {failed}")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if simulation_app is not None:
            simulation_app.close()
