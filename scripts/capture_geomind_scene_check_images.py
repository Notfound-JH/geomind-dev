#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.bootstrap.isaaclab_path import ensure_isaaclab_source_on_path
from geomind_lab.utils import asset_utils


ensure_isaaclab_source_on_path(REPO_ROOT)

# Import pinocchio before AppLauncher so IsaacLab uses the conda-installed version.
if sys.platform != "win32":
    try:
        import pinocchio  # noqa: F401
    except Exception:
        pass

from isaaclab.app import AppLauncher


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load GeoMind scenes one by one, settle for a fixed number of physics steps, and save "
            "all camera RGB images under logs/scene_check_images."
        )
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=None,
        help=(
            "Optional scene id selector. Leave unset to capture all scenes. "
            'Examples: "19", "[1, 19, 27]", or "1,19,27".'
        ),
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=60,
        help="Number of zero-action settle steps after reset before saving images.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Square RGB output resolution used for every camera.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "scene_check_images",
        help="Output directory used to save camera images.",
    )
    parser.add_argument(
        "--lighting-mode",
        type=str,
        default="natural",
        choices=("natural", "bright"),
        help="Scene lighting preset.",
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


SCENE_ROOT = Path(asset_utils.GEOMIND_ASSETS_ROOT) / "stage"


def parse_scene_ids_arg(raw_value: str | None) -> list[int]:
    if raw_value is None or not str(raw_value).strip():
        return collect_all_scene_ids()

    value = str(raw_value).strip()
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError(f"--scene-id JSON form must be a list. Received: {value}")
        return sorted({int(item) for item in parsed})

    if "," in value:
        return sorted({int(item.strip()) for item in value.split(",") if item.strip()})

    return [int(value)]


def collect_all_scene_ids() -> list[int]:
    scene_ids: list[int] = []
    for path in sorted(SCENE_ROOT.iterdir()):
        if path.is_dir() and path.name.isdigit():
            scene_ids.append(int(path.name))
    return scene_ids


def build_runtime_env_cfg(scene_id: int):
    resolution = int(args_cli.resolution)

    def __post_init__(self):
        self.scene_id = int(scene_id)
        self.task_layout_key = "__scene_check__"
        self.task_index = 0
        self.task_mode = "vla"
        self.lighting_mode = args_cli.lighting_mode
        self.scene.num_envs = 1
        self.sim.device = args_cli.device
        self.sim.use_fabric = not args_cli.disable_fabric

        FrankaGeoMindIKRelVisuomotorEnvCfg.__post_init__(self)
        _set_all_camera_cfg_resolutions(self.scene, resolution)

    runtime_cfg_cls = configclass(
        type(
            "SceneCheckGeoMindEnvCfg",
            (FrankaGeoMindIKRelVisuomotorEnvCfg,),
            {
                "__doc__": "Runtime GeoMind cfg used for scene camera image capture.",
                "scene_id": int(scene_id),
                "task_layout_key": "__scene_check__",
                "task_index": 0,
                "task_mode": "vla",
                "lighting_mode": args_cli.lighting_mode,
                "__post_init__": __post_init__,
            },
        )
    )
    return runtime_cfg_cls()


def _set_all_camera_cfg_resolutions(scene_cfg, resolution: int) -> None:
    for attr_name, attr_value in scene_cfg.__dict__.items():
        if isinstance(attr_value, CameraCfg):
            attr_value.height = int(resolution)
            attr_value.width = int(resolution)


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


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


def save_camera_images(env: ManagerBasedRLEnv, scene_id: int, output_dir: Path) -> list[Path]:
    scene_dir = output_dir / f"scene_{scene_id:03d}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for sensor_name, sensor in sorted(env.scene.sensors.items()):
        if not isinstance(sensor, Camera):
            continue

        sensor.update(dt=0.0, force_recompute=True)
        rgb = sensor.data.output.get("rgb")
        if rgb is None or rgb.numel() == 0:
            continue

        image = rgb[0].detach().cpu()
        if image.dtype != torch.uint8:
            image = image.to(torch.uint8)
        if image.ndim != 3 or image.shape[-1] < 3:
            continue

        image = image[..., :3].permute(2, 0, 1).contiguous()
        image_path = scene_dir / f"{_sanitize_name(sensor_name)}.png"
        write_png(image, str(image_path))
        saved_paths.append(image_path)

    return saved_paths


def new_stage() -> None:
    omni.usd.get_context().new_stage()


def run_scene_capture(scene_id: int, output_dir: Path) -> dict[str, object]:
    env = None
    try:
        print(f"[scene {scene_id}] building env", flush=True)
        env_cfg = build_runtime_env_cfg(scene_id)
        env = ManagerBasedRLEnv(cfg=env_cfg)

        print(f"[scene {scene_id}] reset", flush=True)
        env.reset()

        with torch.inference_mode():
            print(f"[scene {scene_id}] settle {args_cli.settle_steps} steps", flush=True)
            settle_env(env, args_cli.settle_steps)

        print(f"[scene {scene_id}] save camera images", flush=True)
        saved_paths = save_camera_images(env, scene_id, output_dir)
        print(f"[scene {scene_id}] saved {len(saved_paths)} images", flush=True)
        for path in saved_paths:
            print(f"  {path}", flush=True)

        return {
            "scene_id": int(scene_id),
            "saved_images": [str(path) for path in saved_paths],
            "error": None,
        }
    except Exception as exc:
        print(f"[scene {scene_id}] failed: {exc}", flush=True)
        traceback.print_exc()
        return {
            "scene_id": int(scene_id),
            "saved_images": [],
            "error": str(exc),
        }
    finally:
        if env is not None:
            env.close()


def main() -> int:
    scene_ids = parse_scene_ids_arg(args_cli.scene_id)
    if not scene_ids:
        raise RuntimeError(f"No scene ids found under: {SCENE_ROOT}")

    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GeoMind Scene Check Image Capture")
    print("=" * 72)
    print(f"scene_ids:      {scene_ids}")
    print(f"settle_steps:   {args_cli.settle_steps}")
    print(f"resolution:     {args_cli.resolution}x{args_cli.resolution}")
    print(f"lighting_mode:  {args_cli.lighting_mode}")
    print(f"output_dir:     {output_dir}")
    print("=" * 72)

    results: list[dict[str, object]] = []
    for index, scene_id in enumerate(scene_ids):
        if index > 0:
            new_stage()
        results.append(run_scene_capture(scene_id, output_dir))

    summary_path = output_dir / "capture_summary.json"
    summary_payload = {
        "scene_ids": scene_ids,
        "settle_steps": int(args_cli.settle_steps),
        "resolution": int(args_cli.resolution),
        "lighting_mode": str(args_cli.lighting_mode),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    num_failed = sum(1 for item in results if item["error"])
    num_saved = sum(len(item["saved_images"]) for item in results)
    print("=" * 72)
    print(f"summary_path:   {summary_path}")
    print(f"saved_images:   {num_saved}")
    print(f"failed_scenes:  {num_failed}")
    print("=" * 72)
    return 0 if num_failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if simulation_app is not None:
            simulation_app.close()
