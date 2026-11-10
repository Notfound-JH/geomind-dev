#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.bootstrap.isaaclab_path import ensure_isaaclab_source_on_path
from geomind.core.schema import normalize_task_config


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

# Keep the renderer from constructing unused cameras. This must be visible before
# the runtime env cfg executes its __post_init__ camera-selection logic.
os.environ["GEOMIND_ACTIVE_CAMERAS"] = "table_cam"

if sys.platform != "win32":
    try:
        import pinocchio  # noqa: F401
    except Exception:
        pass

from isaaclab.app import AppLauncher


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Static-load every GeoMind VLA task in IsaacLab and save the initial table_cam RGB image "
            "for each task."
        )
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=REPO_ROOT / "geomind" / "configs" / "vla_tasks",
        help="Directory containing VLA task config JSON files.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help=(
            "Optional task selector. Accepts one id, comma-separated ids, or a JSON list. "
            "Leave unset to capture every task config in --task-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "vla_task_initial_table_cam",
        help="Output directory for task images and capture_summary.json.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Square table_cam RGB output resolution.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=0,
        help=(
            "Optional zero-action physics steps after reset. Default 0 preserves the initial static state; "
            "nonzero values may change dynamic-task layouts."
        ),
    )
    parser.add_argument(
        "--lighting-mode",
        type=str,
        default="natural",
        choices=("natural", "bright"),
        help="Scene lighting preset.",
    )
    parser.add_argument(
        "--object-mode",
        type=str,
        default=None,
        choices=("geometric", "concrete"),
        help="Optional object-mode override. Leave unset to use each task/config default.",
    )
    parser.add_argument(
        "--disable-fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O instead.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        default=False,
        help="Stop after the first failed task.",
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
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import load_task


SKIP_CONFIG_FILES = {"legacy_layouts.json", "task_cards.yaml"}


def parse_task_ids(raw_value: str | None) -> set[str] | None:
    if raw_value is None or not str(raw_value).strip():
        return None

    value = str(raw_value).strip()
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError(f"--task-id JSON form must be a list. Received: {value}")
        return {str(item).strip() for item in parsed if str(item).strip()}

    if "," in value:
        return {item.strip() for item in value.split(",") if item.strip()}

    return {value}


def discover_task_configs(task_dir: Path, selected_task_ids: set[str] | None) -> list[dict[str, Any]]:
    task_dir = task_dir.expanduser().resolve()
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task config directory does not exist: {task_dir}")

    tasks: list[dict[str, Any]] = []
    for path in sorted(task_dir.glob("*.json")):
        if path.name in SKIP_CONFIG_FILES:
            continue
        payload = normalize_task_config(json.loads(path.read_text(encoding="utf-8")))
        if payload.get("implementation_status") == "placeholder":
            continue
        task_id = str(payload.get("task_id", "")).strip()
        if not task_id:
            continue
        if selected_task_ids is not None and task_id not in selected_task_ids and payload.get("public_task_id") not in selected_task_ids:
            continue
        tasks.append(
            {
                "task_id": task_id,
                "config_path": str(path),
                "canonical_task_id": str(payload.get("canonical_task_id", "")),
                "task_contract": dict(payload.get("task_contract", {})),
                "scene_id": int(payload.get("scene_id", 65)),
                "layout_key": str(payload.get("annotation_layout_key", "")),
                "task_index": int(payload.get("annotation_task_index", 0) or 0),
                "object_mode": str(payload.get("object_mode", "") or ""),
            }
        )

    if selected_task_ids is not None:
        found = {item["task_id"] for item in tasks}
        missing = sorted(selected_task_ids - found)
        if missing:
            raise ValueError(f"Task id(s) not found under {task_dir}: {missing}")

    return tasks


def static_task_spec(task_info: dict[str, Any]) -> dict[str, Any]:
    object_mode = args_cli.object_mode or task_info.get("object_mode") or None
    loaded = load_task(
        task_id=task_info["canonical_task_id"] or task_info["task_id"],
        scene_id=task_info["scene_id"],
        layout_key=task_info["layout_key"],
        task_index=task_info["task_index"],
        object_mode=object_mode,
        task_contract=task_info.get("task_contract"),
    )
    spec = deepcopy(loaded.task_spec)
    protocol = spec.get("protocol")
    if isinstance(protocol, dict):
        protocol.pop("intervention", None)
        spec["protocol"] = protocol
    metadata = spec.setdefault("task_metadata", {})
    if isinstance(metadata, dict):
        metadata["static_initial_capture"] = True
        metadata["source_config_path"] = task_info["config_path"]
    return spec


def build_runtime_env_cfg(task_info: dict[str, Any]):
    task_spec = static_task_spec(task_info)
    resolution = int(args_cli.resolution)

    def __post_init__(self):
        self.task_spec = deepcopy(task_spec)
        self.task_id = str(task_info["task_id"])
        self.scene_id = int(task_info["scene_id"])
        self.task_layout_key = str(task_info["layout_key"])
        self.task_index = int(task_info["task_index"])
        self.task_mode = "vla"
        self.lighting_mode = args_cli.lighting_mode
        self.scene.num_envs = 1
        self.sim.device = args_cli.device
        self.sim.use_fabric = not args_cli.disable_fabric

        FrankaGeoMindIKRelVisuomotorEnvCfg.__post_init__(self)
        _set_table_cam_resolution(self.scene, resolution)

    runtime_cfg_cls = configclass(
        type(
            "StaticInitialTaskTableCamEnvCfg",
            (FrankaGeoMindIKRelVisuomotorEnvCfg,),
            {
                "__doc__": "Runtime GeoMind cfg for static initial table_cam capture.",
                "scene_id": int(task_info["scene_id"]),
                "task_layout_key": str(task_info["layout_key"]),
                "task_index": int(task_info["task_index"]),
                "task_mode": "vla",
                "lighting_mode": args_cli.lighting_mode,
                "__post_init__": __post_init__,
            },
        )
    )
    return runtime_cfg_cls()


def _set_table_cam_resolution(scene_cfg, resolution: int) -> None:
    table_cam = getattr(scene_cfg, "table_cam", None)
    if isinstance(table_cam, CameraCfg):
        table_cam.height = int(resolution)
        table_cam.width = int(resolution)


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


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


def save_table_cam_image(env: ManagerBasedRLEnv, task_info: dict[str, Any], output_dir: Path) -> Path:
    sensor = env.scene.sensors.get("table_cam")
    if not isinstance(sensor, Camera):
        raise RuntimeError("table_cam sensor was not created.")

    sensor.update(dt=0.0, force_recompute=True)
    rgb = sensor.data.output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        raise RuntimeError("table_cam produced no RGB output.")

    image = rgb[0].detach().cpu()
    if image.dtype != torch.uint8:
        image = image.to(torch.uint8)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise RuntimeError(f"Unexpected table_cam RGB tensor shape: {tuple(image.shape)}")

    image = image[..., :3].permute(2, 0, 1).contiguous()
    task_dir = output_dir / sanitize_name(task_info["task_id"])
    task_dir.mkdir(parents=True, exist_ok=True)
    image_path = task_dir / "table_cam.png"
    write_png(image, str(image_path))
    return image_path


def new_stage() -> None:
    omni.usd.get_context().new_stage()


def run_task_capture(task_info: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    env = None
    try:
        print(f"[{task_info['task_id']}] building env", flush=True)
        env_cfg = build_runtime_env_cfg(task_info)
        env = ManagerBasedRLEnv(cfg=env_cfg)

        print(f"[{task_info['task_id']}] reset", flush=True)
        env.reset()

        with torch.inference_mode():
            if int(args_cli.settle_steps) > 0:
                print(f"[{task_info['task_id']}] settle {args_cli.settle_steps} steps", flush=True)
            settle_env(env, int(args_cli.settle_steps))

        print(f"[{task_info['task_id']}] save table_cam", flush=True)
        image_path = save_table_cam_image(env, task_info, output_dir)
        print(f"[{task_info['task_id']}] saved {image_path}", flush=True)

        return {
            **task_info,
            "saved_image": str(image_path),
            "resolution": int(args_cli.resolution),
            "settle_steps": int(args_cli.settle_steps),
            "error": None,
        }
    except Exception as exc:
        print(f"[{task_info['task_id']}] failed: {exc}", flush=True)
        traceback.print_exc()
        return {
            **task_info,
            "saved_image": None,
            "resolution": int(args_cli.resolution),
            "settle_steps": int(args_cli.settle_steps),
            "error": str(exc),
        }
    finally:
        if env is not None:
            env.close()


def main() -> int:
    selected_task_ids = parse_task_ids(args_cli.task_id)
    tasks = discover_task_configs(args_cli.task_dir, selected_task_ids)
    if not tasks:
        raise RuntimeError(f"No task configs found under: {args_cli.task_dir}")

    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GeoMind VLA Static Initial TableCam Capture")
    print("=" * 72)
    print(f"task_dir:       {args_cli.task_dir.expanduser().resolve()}")
    print(f"tasks:          {len(tasks)}")
    print(f"resolution:     {args_cli.resolution}x{args_cli.resolution}")
    print(f"settle_steps:   {args_cli.settle_steps}")
    print(f"lighting_mode:  {args_cli.lighting_mode}")
    print(f"output_dir:     {output_dir}")
    print("=" * 72)

    results: list[dict[str, Any]] = []
    for index, task_info in enumerate(tasks):
        if index > 0:
            new_stage()
        result = run_task_capture(task_info, output_dir)
        results.append(result)
        if result.get("error") and args_cli.stop_on_error:
            break

    summary_path = output_dir / "capture_summary.json"
    summary_payload = {
        "task_dir": str(args_cli.task_dir.expanduser().resolve()),
        "task_count": len(tasks),
        "captured_count": sum(1 for item in results if not item.get("error")),
        "failed_count": sum(1 for item in results if item.get("error")),
        "camera": "table_cam",
        "resolution": int(args_cli.resolution),
        "settle_steps": int(args_cli.settle_steps),
        "static_initial_capture": True,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print("=" * 72)
    print(f"summary_path:   {summary_path}")
    print(f"captured:       {summary_payload['captured_count']}")
    print(f"failed:         {summary_payload['failed_count']}")
    print("=" * 72)
    return 0 if summary_payload["failed_count"] == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if simulation_app is not None:
            simulation_app.close()
