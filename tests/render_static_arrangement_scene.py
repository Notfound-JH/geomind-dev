#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import datetime
import faulthandler
import json
import math
import random
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any


try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.bootstrap.isaaclab_path import ensure_isaaclab_source_on_path


ensure_isaaclab_source_on_path(REPO_ROOT)

# Import pinocchio before AppLauncher so IsaacLab uses the conda-installed version.
if sys.platform != "win32":
    try:
        import pinocchio  # noqa: F401
    except Exception:
        pass

from isaaclab.app import AppLauncher


def print_static_failure_context(
    stage: str,
    *,
    step_count: int | None = None,
    env_cfg=None,
) -> None:
    print("\n" + "=" * 72, file=sys.stderr, flush=True)
    print("GeoMind Static Arrangement Scene — failure", file=sys.stderr, flush=True)
    print("=" * 72, file=sys.stderr, flush=True)
    print(f"stage:          {stage}", file=sys.stderr, flush=True)

    args_obj = globals().get("args_cli")
    if args_obj is not None:
        print(f"annotation_key: {getattr(args_obj, 'annotation_layout_key', None)}", file=sys.stderr, flush=True)
        print(f"annotation_path:{getattr(args_obj, 'annotation_path', None)}", file=sys.stderr, flush=True)
        print(f"arrangement:    {getattr(args_obj, 'arrangement', None)}", file=sys.stderr, flush=True)
        print(f"scene_id_arg:   {getattr(args_obj, 'scene_id', None)}", file=sys.stderr, flush=True)
        print(f"device:         {getattr(args_obj, 'device', None)}", file=sys.stderr, flush=True)
        print(f"headless:       {getattr(args_obj, 'headless', None)}", file=sys.stderr, flush=True)

    if env_cfg is not None:
        print(f"resolved_scene: {getattr(env_cfg, 'scene_id', None)}", file=sys.stderr, flush=True)
        scene_profile = getattr(env_cfg, "_scene_profile", None)
        if scene_profile is not None:
            print(f"scene_usd:      {getattr(scene_profile, 'usd_path', None)}", file=sys.stderr, flush=True)

    if step_count is not None:
        print(f"step_count:     {step_count}", file=sys.stderr, flush=True)

    print("-" * 72, file=sys.stderr, flush=True)
    traceback.print_exc()
    print("=" * 72, file=sys.stderr, flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a static GeoMind scene with runtime-injected colored blocks. "
            "By default, this script mimics the reindex_extreme line layout."
        )
    )
    parser.add_argument(
        "--annotation-layout-key",
        type=str,
        default="state_reindex_blocks_line_4",
        help="Template layout key from assets/annotations/annotated_tasks.json.",
    )
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=REPO_ROOT / "assets" / "annotations" / "annotated_tasks.json",
        help="Path to annotated_tasks.json.",
    )
    parser.add_argument(
        "--scene-id",
        type=int,
        default=None,
        help="Override scene id. Defaults to the scene id defined by --annotation-layout-key.",
    )
    parser.add_argument(
        "--arrangement",
        type=str,
        default="line",
        choices=("line", "matrix", "t_shape", "circle", "basket_center", "stack_center"),
        help="Object arrangement type.",
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=4,
        help="Number of blocks when not using --num-objects-min/max.",
    )
    parser.add_argument(
        "--num-objects-min",
        type=int,
        default=None,
        help="If set with --num-objects-max, sample integer count uniformly in [min, max] (inclusive).",
    )
    parser.add_argument(
        "--num-objects-max",
        type=int,
        default=None,
        help="See --num-objects-min.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="If set, seeds random for count/spacing/position jitter (reproducible runs).",
    )
    parser.add_argument(
        "--spacing-jitter",
        type=float,
        default=0.0,
        help="Relative spacing noise: multiply layout spacing by U(1-j,1+j). 0 disables.",
    )
    parser.add_argument(
        "--xy-jitter",
        type=float,
        default=0.0,
        help="After layout, add uniform XY noise in [-j,j] meters per block (stack_center skips).",
    )
    parser.add_argument(
        "--block-color-mode",
        type=str,
        default="distinct",
        choices=("distinct", "uniform"),
        help="distinct: HSV vary per block; uniform: single color for all blocks.",
    )
    parser.add_argument(
        "--uniform-block-color",
        type=float,
        nargs=3,
        default=(0.72, 0.58, 0.42),
        metavar=("R", "G", "B"),
        help="RGB in [0,1] when --block-color-mode uniform.",
    )
    parser.add_argument(
        "--export-media",
        type=str,
        default="none",
        choices=("none", "images", "video", "both"),
        help="After settle, save PNG and/or MP4 for selected cameras (see --export-cameras).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "static_arrangement_capture",
        help="Output root; each run uses a subfolder (see --export-run-tag).",
    )
    parser.add_argument(
        "--export-run-tag",
        type=str,
        default=None,
        help="Subfolder name under export-dir. Default: timestamp.",
    )
    parser.add_argument(
        "--export-cameras",
        type=str,
        default="scene_cam,ego,table_cam",
        help="Comma-separated sensor names. Short alias: ego -> ego_view_camera.",
    )
    parser.add_argument(
        "--export-video-frames",
        type=int,
        default=90,
        help="Frames per camera for MP4 (when export-media is video or both).",
    )
    parser.add_argument(
        "--export-video-fps",
        type=float,
        default=30.0,
        help="FPS for exported MP4.",
    )
    parser.add_argument(
        "--export-video-warmup",
        type=int,
        default=24,
        help="Zero-action warmup steps before recording video frames.",
    )
    parser.add_argument(
        "--line-axis",
        type=str,
        default="y",
        choices=("x", "y"),
        help="Axis used by line arrangement.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="Center-to-center spacing in meters. Defaults to template-derived spacing.",
    )
    parser.add_argument("--matrix-cols", type=int, default=3, help="Number of columns for matrix arrangement.")
    parser.add_argument(
        "--circle-radius",
        type=float,
        default=None,
        help="Circle radius in meters for circle arrangement.",
    )
    parser.add_argument("--cube-size", type=float, default=0.045, help="Cube edge length in meters.")
    parser.add_argument(
        "--stack-top-ratio",
        type=float,
        default=0.78,
        help=(
            "stack_center only: edge length of the top cube divided by the bottom cube (0.25–1). "
            "Intermediate layers are linearly interpolated; bottom uses --cube-size."
        ),
    )
    parser.add_argument(
        "--center-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Optional XY center override for arrangement.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=60,
        help="Zero-action settle steps after reset.",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=0,
        help="Number of static simulation steps to keep the scene visible. Use 0 to run until window close.",
    )
    parser.add_argument(
        "--min-live-steps",
        type=int,
        default=512,
        help=(
            "GUI mode only: zero-action warmup steps after settle, before the main "
            "``while simulation_app.is_running()`` loop (mirrors Kit startup jitter handling)."
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
        "--disable-fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O instead.",
    )
    parser.add_argument(
        "--viewer-eye",
        type=float,
        nargs=3,
        default=(2.2, 2.2, 1.8),
        metavar=("X", "Y", "Z"),
        help="Viewer camera eye position.",
    )
    parser.add_argument(
        "--viewer-lookat",
        type=float,
        nargs=3,
        default=(0.35, 0.0, 0.75),
        metavar=("X", "Y", "Z"),
        help="Viewer camera look-at position.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = build_argparser().parse_args()
setattr(args_cli, "enable_cameras", True)

simulation_app = None
try:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
except Exception:
    print_static_failure_context("app_launcher")
    raise SystemExit(1)

_msg = (
    "[static_arrangement] Kit SimulationApp is up; importing torch / IsaacLab / GeoMind "
    "(this line confirms Python reached the post-AppLauncher stage; Kit logs above are normal)."
)
print(_msg, flush=True)
print(_msg, file=sys.stderr, flush=True)

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import Camera
from isaaclab.utils import configclass
from torchvision.io import write_png

from geomind_lab.tasks.manager_based.manipulation.geomind.config.franka import FrankaGeoMindIKRelVisuomotorEnvCfg
from geomind_lab.tasks.manager_based.manipulation.geomind.geomind_scene_cfg import get_scene_profile
from geomind_lab.utils import asset_utils
from geomind_lab.utils.headless_camera_video import record_rgb_frame_lists, save_rgb_frame_lists_to_mp4s
from geomind_lab.utils.ui_viewport_utils import configure_env_camera_viewports


def _should_reset(terminated, truncated) -> bool:
    if terminated is not None and torch.any(terminated):
        return True
    if truncated is not None and torch.any(truncated):
        return True
    return False


def _zero_step_env(env: ManagerBasedRLEnv, zero_actions: torch.Tensor) -> None:
    _, _, terminated, truncated, _ = env.step(zero_actions)
    if _should_reset(terminated, truncated):
        env.reset()


def settle_env_after_reset(env: ManagerBasedRLEnv, settle_steps: int) -> None:
    if settle_steps <= 0:
        return
    zero_actions = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(settle_steps):
        _zero_step_env(env, zero_actions)


_CAMERA_EXPORT_ALIASES: dict[str, str] = {
    "ego": "ego_view_camera",
    "ego_view": "ego_view_camera",
    "egocentric": "ego_view_camera",
}


def _sanitize_filename_component(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


def _resolve_export_camera_names(raw: str) -> list[str]:
    resolved: list[str] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        resolved.append(_CAMERA_EXPORT_ALIASES.get(t.lower(), t))
    return resolved


def _apply_xy_jitter(
    points_xy: list[tuple[float, float]],
    jitter: float,
    arrangement: str,
) -> list[tuple[float, float]]:
    if jitter <= 0.0 or arrangement == "stack_center":
        return points_xy
    j = float(jitter)
    return [(px + random.uniform(-j, j), py + random.uniform(-j, j)) for px, py in points_xy]


def _seed_python_random(seed: int | None) -> None:
    if seed is None:
        return
    s = int(seed)
    random.seed(s)
    try:
        torch.manual_seed(s)
    except Exception:
        pass


def _export_static_media(
    env: ManagerBasedRLEnv,
    *,
    run_dir: Path,
    manifest: dict[str, Any],
) -> tuple[list[Path], list[Path]]:
    """Save PNG and/or MP4 for requested IsaacLab Camera sensors (env index 0)."""
    export_mode = str(getattr(args_cli, "export_media", "none"))
    if export_mode == "none":
        return [], []

    want_img = export_mode in ("images", "both")
    want_vid = export_mode in ("video", "both")
    names = _resolve_export_camera_names(str(getattr(args_cli, "export_cameras", "")))
    if not names:
        return [], []

    run_dir.mkdir(parents=True, exist_ok=True)
    png_paths: list[Path] = []
    mp4_paths: list[Path] = []

    if want_img:
        for cam_name in names:
            sensor = env.scene.sensors.get(cam_name)
            if not isinstance(sensor, Camera):
                print(f"[static_arrangement] export skip (not a Camera or missing): {cam_name}", flush=True)
                continue
            sensor.update(dt=0.0, force_recompute=True)
            rgb = sensor.data.output.get("rgb") if sensor.data is not None else None
            if rgb is None or rgb.numel() == 0:
                print(f"[static_arrangement] export skip (no rgb): {cam_name}", flush=True)
                continue
            image = rgb[0].detach().cpu()
            if image.dtype != torch.uint8:
                image = image.to(torch.uint8)
            if image.ndim != 3 or image.shape[-1] < 3:
                continue
            image = image[..., :3].permute(2, 0, 1).contiguous()
            out = run_dir / f"{_sanitize_filename_component(cam_name)}.png"
            write_png(image, str(out))
            png_paths.append(out)

    if want_vid:
        n_frames = max(1, int(getattr(args_cli, "export_video_frames", 90)))
        warmup = max(0, int(getattr(args_cli, "export_video_warmup", 24)))
        fps = float(getattr(args_cli, "export_video_fps", 30.0))
        zero_actions = torch.zeros(env.action_space.shape, device=env.device)
        frame_lists = record_rgb_frame_lists(
            env,
            n_frames,
            warmup_steps=warmup,
            make_actions=lambda _e: zero_actions,
        )
        filtered = {k: v for k, v in frame_lists.items() if k in names}
        missing = [k for k in names if k not in filtered]
        for k in missing:
            print(f"[static_arrangement] video skip (no frames from sensor): {k}", flush=True)
        if filtered:
            mp4_paths = save_rgb_frame_lists_to_mp4s(filtered, run_dir, fps=fps)

    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[static_arrangement] media saved under {run_dir}", flush=True)
    return png_paths, mp4_paths


def _load_layout_template(path: Path, layout_key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid annotation payload at {path}. Expected a JSON object.")
    layout = payload.get(layout_key)
    if not isinstance(layout, dict):
        raise KeyError(f"Layout key not found: {layout_key}")
    return layout


def _template_center_and_spacing(layout: dict[str, Any]) -> tuple[tuple[float, float], float]:
    scene_structure = layout.get("scene_structure", {})
    object_positions = scene_structure.get("object_positions", {}) if isinstance(scene_structure, dict) else {}
    if not isinstance(object_positions, dict) or not object_positions:
        return (0.42, 0.0), 0.09

    xy_points: list[tuple[float, float]] = []
    for payload in object_positions.values():
        if not isinstance(payload, dict):
            continue
        position = payload.get("position", [])
        if not isinstance(position, list) or len(position) < 2:
            continue
        xy_points.append((float(position[0]), float(position[1])))

    if not xy_points:
        return (0.42, 0.0), 0.09

    center_x = sum(p[0] for p in xy_points) / len(xy_points)
    center_y = sum(p[1] for p in xy_points) / len(xy_points)

    sorted_x = sorted(p[0] for p in xy_points)
    sorted_y = sorted(p[1] for p in xy_points)
    delta_x = [abs(sorted_x[i + 1] - sorted_x[i]) for i in range(len(sorted_x) - 1) if abs(sorted_x[i + 1] - sorted_x[i]) > 1e-6]
    delta_y = [abs(sorted_y[i + 1] - sorted_y[i]) for i in range(len(sorted_y) - 1) if abs(sorted_y[i + 1] - sorted_y[i]) > 1e-6]
    all_deltas = delta_x + delta_y
    inferred_spacing = float(statistics.median(all_deltas)) if all_deltas else 0.09

    return (center_x, center_y), inferred_spacing


def _min_safe_spacing(cube_size: float) -> float:
    return max(1.25 * cube_size, cube_size + 0.01)


def _resolve_spacing(inferred_spacing: float, cube_size: float, user_spacing: float | None) -> float:
    spacing = float(inferred_spacing if user_spacing is None else user_spacing)
    return max(spacing, _min_safe_spacing(cube_size))


def _build_line_positions(
    center_xy: tuple[float, float], count: int, spacing: float, axis: str
) -> list[tuple[float, float]]:
    cx, cy = center_xy
    offset0 = 0.5 * (count - 1)
    points: list[tuple[float, float]] = []
    for index in range(count):
        offset = (index - offset0) * spacing
        if axis == "x":
            points.append((cx + offset, cy))
        else:
            points.append((cx, cy + offset))
    return points


def _build_matrix_positions(
    center_xy: tuple[float, float], count: int, spacing: float, cols: int
) -> list[tuple[float, float]]:
    cx, cy = center_xy
    cols = max(1, int(cols))
    rows = int(math.ceil(count / cols))
    x0 = -0.5 * (cols - 1) * spacing
    y0 = -0.5 * (rows - 1) * spacing
    points: list[tuple[float, float]] = []
    for index in range(count):
        row = index // cols
        col = index % cols
        points.append((cx + x0 + col * spacing, cy + y0 + row * spacing))
    return points


def _build_t_shape_positions(center_xy: tuple[float, float], count: int, spacing: float) -> list[tuple[float, float]]:
    if count <= 3:
        return _build_line_positions(center_xy, count, spacing, axis="x")

    cx, cy = center_xy
    top_count = max(3, int(math.ceil(count * 0.5)))
    if top_count % 2 == 0:
        top_count -= 1
    top_count = min(top_count, count)
    stem_count = count - top_count

    points: list[tuple[float, float]] = []
    top_offset0 = 0.5 * (top_count - 1)
    top_y = cy + spacing
    for i in range(top_count):
        x = cx + (i - top_offset0) * spacing
        points.append((x, top_y))

    for i in range(stem_count):
        points.append((cx, cy - i * spacing))
    return points


def _build_circle_positions(
    center_xy: tuple[float, float], count: int, spacing: float, radius: float | None
) -> list[tuple[float, float]]:
    cx, cy = center_xy
    resolved_radius = radius
    if resolved_radius is None:
        resolved_radius = max(spacing, (count * spacing) / (2.0 * math.pi))
    resolved_radius = max(float(resolved_radius), 0.5 * spacing)

    points: list[tuple[float, float]] = []
    for i in range(count):
        angle = (2.0 * math.pi * i) / count
        points.append((cx + resolved_radius * math.cos(angle), cy + resolved_radius * math.sin(angle)))
    return points


def _build_basket_positions(
    center_xy: tuple[float, float], count: int, spacing: float, cube_size: float
) -> list[tuple[float, float]]:
    """Place blocks on a ring inside the basket; radius scales with cube size and count to reduce overlap."""
    cx, cy = center_xy
    if count == 1:
        return [(cx, cy)]
    cs = max(float(cube_size), 0.015)
    n = max(count, 2)
    # Neighbor separation ~ chord length >= cube edge (with margin).
    need_r = (cs * 1.12) / (2.0 * math.sin(math.pi / n))
    cap = max(0.014, min(0.075, 0.32 * float(spacing)))
    radius = float(min(max(need_r, 0.007), cap))
    points: list[tuple[float, float]] = []
    for i in range(count):
        angle = (2.0 * math.pi * i) / count
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _build_stack_positions(center_xy: tuple[float, float], count: int) -> list[tuple[float, float]]:
    return [center_xy for _ in range(count)]


def _stack_cube_sizes(base_edge: float, count: int, top_ratio: float) -> list[float]:
    """Bottom (index 0) largest; top (index count-1) smallest. Linear interpolation in edge length."""
    base = max(float(base_edge), 0.015)
    tr = min(max(float(top_ratio), 0.22), 1.0)
    if count <= 1:
        return [base]
    sizes: list[float] = []
    for i in range(count):
        t = i / (count - 1)
        edge = base * (1.0 - (1.0 - tr) * t)
        sizes.append(max(float(edge), 0.012))
    return sizes


def _pairwise_clearance_ok(points: list[tuple[float, float]], cube_size: float) -> bool:
    min_dist = _min_safe_spacing(cube_size) * 0.9
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            if math.hypot(dx, dy) < min_dist:
                return False
    return True


def _build_positions(
    arrangement: str,
    center_xy: tuple[float, float],
    count: int,
    spacing: float,
    line_axis: str,
    matrix_cols: int,
    circle_radius: float | None,
    cube_size: float,
) -> list[tuple[float, float]]:
    if arrangement == "line":
        return _build_line_positions(center_xy, count, spacing, line_axis)
    if arrangement == "matrix":
        return _build_matrix_positions(center_xy, count, spacing, matrix_cols)
    if arrangement == "t_shape":
        return _build_t_shape_positions(center_xy, count, spacing)
    if arrangement == "circle":
        return _build_circle_positions(center_xy, count, spacing, circle_radius)
    if arrangement == "basket_center":
        return _build_basket_positions(center_xy, count, spacing, cube_size)
    if arrangement == "stack_center":
        return _build_stack_positions(center_xy, count)
    raise ValueError(f"Unsupported arrangement: {arrangement}")


def _distinct_rgb(index: int, total: int) -> tuple[float, float, float]:
    hue = float(index) / float(max(total, 1))
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
    return (float(red), float(green), float(blue))


def _build_task_spec(
    points_xy: list[tuple[float, float]],
    cube_sizes: list[float],
    arrangement: str,
    center_xy: tuple[float, float],
    *,
    block_color_mode: str,
    uniform_rgb: tuple[float, float, float],
) -> dict[str, Any]:
    if len(cube_sizes) != len(points_xy):
        raise ValueError("cube_sizes length must match points_xy length.")

    runtime_objects: list[dict[str, Any]] = []
    object_positions: dict[str, dict[str, list[float]]] = {}

    uniform = tuple(float(c) for c in uniform_rgb)
    for index, (px, py) in enumerate(points_xy):
        name = f"demo_block_{index}"
        if block_color_mode == "uniform":
            color = uniform
        else:
            color = _distinct_rgb(index, len(points_xy))
        edge = max(float(cube_sizes[index]), 0.012)
        runtime_objects.append(
            {
                "name": name,
                "kind": "primitive_cuboid",
                "size": [edge, edge, edge],
                "visual_color": [color[0], color[1], color[2]],
                "visual_material": {
                    "fallback_diffuse_color": [color[0], color[1], color[2]],
                    "fallback_roughness": 0.32,
                },
                "physics": {
                    "collision": {
                        "collision_enabled": True,
                        "contact_offset": 0.002,
                        "rest_offset": 0.0,
                    },
                    "material": {
                        "static_friction": 1.0,
                        "dynamic_friction": 0.9,
                        "restitution": 0.0,
                    },
                },
                "pose": {"xy": [float(px), float(py)], "yaw": 0.0},
            }
        )
        object_positions[name] = {"position": [float(px), float(py)]}

    resolved_runtime_objects = list(runtime_objects)
    if arrangement == "basket_center":
        resolved_runtime_objects = [
            {
                "name": "demo_center_basket",
                "kind": "usd",
                "asset_category": "container",
                "asset_key": "basket_1",
                "scale": [1.0, 1.0, 1.0],
                "pose": {"xy": [float(center_xy[0]), float(center_xy[1])], "yaw": 0.0},
                # Keep basket rigid in scene and included in physics, but fixed in place.
                "physics": {
                    "rigid_body": {
                        "rigid_body_enabled": True,
                        "kinematic_enabled": True,
                        "disable_gravity": True,
                    },
                    "collision": {
                        "collision_enabled": True,
                        "contact_offset": 0.0015,
                        "rest_offset": 0.0,
                    },
                    "material": {"static_friction": 1.0, "dynamic_friction": 0.9, "restitution": 0.0},
                },
            },
            *runtime_objects,
        ]

    scene_adaptation: dict[str, Any] = {
        "runtime_objects": resolved_runtime_objects,
        # Keep the showcase visually clean by hiding the built-in side basket by default.
        "background_overrides": {"hide_basket": True},
    }

    return {
        "task_name": "static_arrangement_render",
        "instruction": f"Render static {arrangement} arrangement ({block_color_mode} block colors).",
        "goal_sequence": [],
        "object_positions": object_positions,
        "runtime_objects": resolved_runtime_objects,
        "scene_adaptation": scene_adaptation,
        "protocol": {"family": "scene_showcase"},
        "task_metadata": {"task_id": "static_arrangement_render", "arrangement": arrangement},
    }


def _build_runtime_env_cfg(task_spec: dict[str, Any], scene_id: int):
    def __post_init__(self):
        self.scene_id = int(scene_id)
        self.task_layout_key = "__runtime_static_arrangement__"
        self.task_index = 0
        self.task_mode = "vla"
        self.lighting_mode = args_cli.lighting_mode
        self.scene.num_envs = 1
        self.sim.device = args_cli.device
        self.sim.use_fabric = not args_cli.disable_fabric
        self.task_spec = task_spec

        FrankaGeoMindIKRelVisuomotorEnvCfg.__post_init__(self)
        self.viewer.eye = tuple(float(v) for v in args_cli.viewer_eye)
        self.viewer.lookat = tuple(float(v) for v in args_cli.viewer_lookat)

    runtime_cfg_cls = configclass(
        type(
            "StaticArrangementRenderEnvCfg",
            (FrankaGeoMindIKRelVisuomotorEnvCfg,),
            {
                "__doc__": "Runtime GeoMind cfg for static arrangement rendering.",
                "__post_init__": __post_init__,
            },
        )
    )
    return runtime_cfg_cls()


def _run_render_loop(
    env: ManagerBasedRLEnv,
    hold_steps: int,
    *,
    min_live_steps: int,
) -> None:
    """Match scripts/view_geomind_env.py: optional startup guard steps, then ``while simulation_app.is_running()``."""
    zero_actions = torch.zeros(env.action_space.shape, device=env.device)
    min_live = max(0, int(min_live_steps))
    headless = bool(getattr(args_cli, "headless", False))

    with torch.inference_mode():
        if headless:
            for _ in range(1200):
                _zero_step_env(env, zero_actions)
            return

        if int(hold_steps) > 0:
            for step_index in range(int(hold_steps)):
                _zero_step_env(env, zero_actions)
                if (
                    simulation_app is not None
                    and not simulation_app.is_running()
                    and step_index >= min_live
                ):
                    break
            return

        # GUI + hold_steps==0: pump a few hundred steps before trusting is_running (Kit startup jitter).
        for _ in range(min_live):
            _zero_step_env(env, zero_actions)

        print("[static_arrangement] entering main render loop (close the Kit window to exit).", flush=True)
        while simulation_app is not None and simulation_app.is_running():
            _zero_step_env(env, zero_actions)


def _resolve_table_center(scene_id: int, fallback_center_xy: tuple[float, float]) -> tuple[float, float]:
    try:
        profile = get_scene_profile(scene_id)
    except Exception:
        return fallback_center_xy
    if profile.table_name and profile.table_name in profile.xforms:
        table_pos = profile.xforms[profile.table_name]["pos"]
        return (float(table_pos[0]), float(table_pos[1]))
    return fallback_center_xy


def _resolve_table_top_height(scene_id: int, fallback: float = 0.75) -> float:
    try:
        profile = get_scene_profile(scene_id)
    except Exception:
        return float(fallback)
    if profile.table_name and profile.table_name in profile.xforms:
        table_xform = profile.xforms[profile.table_name]
        resolved = asset_utils.resolve_tabletop_height(
            profile.table_name,
            position_z=table_xform["pos"][2],
            scale_z=table_xform["scale"][2],
        )
        if resolved is not None:
            return float(resolved)
    return float(fallback)


def _apply_stack_center_pose(
    env: ManagerBasedRLEnv,
    points_xy: list[tuple[float, float]],
    cube_sizes: list[float],
    scene_id: int,
) -> None:
    """Rewrite all cubes to a vertical stack after reset.

    GeoMind spawns every ``primitive_cuboid`` at ``table_top + half_height`` (same z), so identical XY causes
    full overlap at reset. We batch-write final poses (per-layer height from ``cube_sizes``) then settle.
    """
    table_top = _resolve_table_top_height(scene_id)
    cx = float(points_xy[0][0])
    cy = float(points_xy[0][1])
    zero_actions = torch.zeros(env.action_space.shape, device=env.device)
    gap = 0.0012

    with torch.inference_mode():
        cumulative = 0.0
        for index in range(len(points_xy)):
            object_name = f"demo_block_{index}"
            rigid_object = env.scene.rigid_objects.get(object_name)
            if rigid_object is None:
                continue
            edge = max(float(cube_sizes[index]), 0.012)
            z_center = table_top + cumulative + 0.5 * edge
            cumulative += edge + gap

            root_state = rigid_object.data.root_state_w.clone()
            root_state[:, 0] = cx
            root_state[:, 1] = cy
            root_state[:, 2] = z_center
            root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            root_state[:, 7:13] = 0.0
            rigid_object.write_root_state_to_sim(root_state)

        for _ in range(400):
            _zero_step_env(env, zero_actions)


def main() -> int:
    print("[static_arrangement] main() entered — building task from annotations (before env spawn).", flush=True)

    _seed_python_random(getattr(args_cli, "random_seed", None))

    has_range = args_cli.num_objects_min is not None or args_cli.num_objects_max is not None
    selected_matrix_cols = int(args_cli.matrix_cols)
    if has_range:
        if args_cli.num_objects_min is None or args_cli.num_objects_max is None:
            raise ValueError("Use both --num-objects-min and --num-objects-max together, or neither.")
        lo, hi = int(args_cli.num_objects_min), int(args_cli.num_objects_max)
        if lo < 1 or hi < lo:
            raise ValueError("--num-objects-min/max must satisfy 1 <= min <= max.")
        if args_cli.arrangement == "matrix":
            allowed_pairs = ((2, 2), (2, 3), (3, 2), (3, 3))
            rows, cols = random.choice(allowed_pairs)
            num_obj = rows * cols
            selected_matrix_cols = cols
        elif args_cli.arrangement == "t_shape":
            num_obj = 5
        else:
            num_obj = random.randint(lo, hi)
    else:
        if args_cli.arrangement == "matrix":
            if int(args_cli.num_objects) not in (4, 6, 9):
                raise ValueError("For arrangement=matrix, --num-objects must be one of 4, 6, 9.")
            num_obj = int(args_cli.num_objects)
            if selected_matrix_cols not in (2, 3):
                # Keep matrix branch usable with defaults: auto-select a legal column count.
                selected_matrix_cols = 2 if num_obj == 4 else 3
            if num_obj % selected_matrix_cols != 0:
                # When mismatch (e.g. num=4 and cols=3), auto-fix to a valid shape instead of hard exit.
                if num_obj == 4:
                    selected_matrix_cols = 2
                elif num_obj == 9:
                    selected_matrix_cols = 3
                elif num_obj == 6:
                    selected_matrix_cols = random.choice((2, 3))
            rows = num_obj // selected_matrix_cols
            if rows not in (2, 3) or selected_matrix_cols not in (2, 3):
                raise ValueError("For arrangement=matrix, allowed shapes are exactly: 2x2, 2x3, 3x2, 3x3.")
        elif args_cli.arrangement == "t_shape":
            num_obj = 5
        else:
            if int(args_cli.num_objects) < 1:
                raise ValueError("--num-objects must be >= 1.")
            num_obj = int(args_cli.num_objects)

    if args_cli.cube_size <= 0.0:
        raise ValueError("--cube-size must be > 0.")

    ucol = tuple(float(x) for x in args_cli.uniform_block_color)
    if any(c < 0.0 or c > 1.0 for c in ucol):
        raise ValueError("--uniform-block-color components must be in [0, 1].")

    sj = float(getattr(args_cli, "spacing_jitter", 0.0))
    if sj < 0.0 or sj > 0.5:
        raise ValueError("--spacing-jitter must be in [0, 0.5].")

    layout = _load_layout_template(args_cli.annotation_path, args_cli.annotation_layout_key)
    scene_id = int(args_cli.scene_id if args_cli.scene_id is not None else layout.get("scene_id", 65))
    inferred_center_xy, inferred_spacing = _template_center_and_spacing(layout)
    default_center_xy = _resolve_table_center(scene_id, inferred_center_xy)
    center_xy = (
        (float(args_cli.center_xy[0]), float(args_cli.center_xy[1]))
        if args_cli.center_xy is not None
        else default_center_xy
    )
    spacing = _resolve_spacing(inferred_spacing, float(args_cli.cube_size), args_cli.spacing)
    if sj > 0.0:
        spacing *= random.uniform(max(0.0, 1.0 - sj), 1.0 + sj)

    if args_cli.arrangement == "stack_center":
        cube_sizes = _stack_cube_sizes(float(args_cli.cube_size), num_obj, float(args_cli.stack_top_ratio))
    else:
        cube_sizes = [float(args_cli.cube_size)] * num_obj

    points_xy = _build_positions(
        arrangement=args_cli.arrangement,
        center_xy=center_xy,
        count=num_obj,
        spacing=spacing,
        line_axis=args_cli.line_axis,
        matrix_cols=selected_matrix_cols,
        circle_radius=args_cli.circle_radius,
        cube_size=float(args_cli.cube_size),
    )
    points_xy = _apply_xy_jitter(points_xy, float(args_cli.xy_jitter), args_cli.arrangement)

    clearance_size = max(cube_sizes) if cube_sizes else float(args_cli.cube_size)
    if args_cli.arrangement not in ("stack_center",) and not _pairwise_clearance_ok(points_xy, clearance_size):
        raise ValueError(
            "Generated arrangement is too dense and may cause initial overlaps. "
            "Increase --spacing, reduce --num-objects, or adjust --circle-radius."
        )

    task_spec = _build_task_spec(
        points_xy,
        cube_sizes,
        args_cli.arrangement,
        center_xy,
        block_color_mode=str(args_cli.block_color_mode),
        uniform_rgb=ucol,
    )

    print("=" * 72, flush=True)
    print("GeoMind Static Arrangement Scene Render", flush=True)
    print("=" * 72, flush=True)
    print(f"scene_id:               {scene_id}", flush=True)
    print(f"annotation_layout_key:  {args_cli.annotation_layout_key}", flush=True)
    print(f"arrangement:            {args_cli.arrangement}", flush=True)
    if getattr(args_cli, "random_seed", None) is not None:
        print(f"random_seed:            {args_cli.random_seed}", flush=True)
    if has_range:
        if args_cli.arrangement == "matrix":
            print(
                f"num_objects:            {num_obj}  (sampled matrix shape with cols={selected_matrix_cols}; "
                "allowed: 2x2,2x3,3x2,3x3)",
                flush=True,
            )
        elif args_cli.arrangement == "t_shape":
            print("num_objects:            5  (forced for t_shape)", flush=True)
        else:
            print(
                f"num_objects:            {num_obj}  (sampled in [{args_cli.num_objects_min}, {args_cli.num_objects_max}])",
                flush=True,
            )
    else:
        if args_cli.arrangement == "t_shape":
            print("num_objects:            5  (forced for t_shape)", flush=True)
        else:
            print(f"num_objects:            {num_obj}", flush=True)
    if args_cli.arrangement == "matrix":
        print(f"matrix_cols:            {selected_matrix_cols}", flush=True)
    print(f"block_color_mode:       {args_cli.block_color_mode}", flush=True)
    print(f"cube_size:              {args_cli.cube_size:.4f} m", flush=True)
    if args_cli.arrangement == "stack_center":
        print(f"stack_top_ratio:        {args_cli.stack_top_ratio}", flush=True)
        print(
            "stack_cube_edges_m:     "
            + ", ".join(f"{float(s):.4f}" for s in cube_sizes),
            flush=True,
        )
    print(f"center_xy:              ({center_xy[0]:.4f}, {center_xy[1]:.4f})", flush=True)
    print(f"spacing:                {spacing:.4f} m  (after optional jitter)", flush=True)
    print(f"spacing_jitter:         {sj}", flush=True)
    print(f"xy_jitter:              {float(args_cli.xy_jitter)}", flush=True)
    print(f"settle_steps:           {args_cli.settle_steps}", flush=True)
    print(f"hold_steps:             {args_cli.hold_steps}", flush=True)
    print(f"min_live_steps:         {args_cli.min_live_steps}", flush=True)
    if str(args_cli.export_media) != "none":
        print(
            f"export_media:           {args_cli.export_media} dir={args_cli.export_dir} "
            f"cams={args_cli.export_cameras}",
            flush=True,
        )
    print("=" * 72, flush=True)

    env = None
    env_cfg = None
    stage = "build_runtime_env_cfg"

    try:
        stage = "build_runtime_env_cfg"
        env_cfg = _build_runtime_env_cfg(task_spec, scene_id)

        stage = "create_env"
        if args_cli.arrangement == "basket_center":
            print(
                "[static_arrangement] creating env (basket USD + extra rigid bodies may take tens of seconds; "
                "see omni.kvdb warning if another Kit process is running).",
                flush=True,
            )
        env = ManagerBasedRLEnv(cfg=env_cfg)
        if env.sim.has_gui():
            configure_env_camera_viewports(env, None)

        print(
            f"[static_arrangement] env ready | scene_usd="
            f"{env_cfg._scene_profile.usd_path if env_cfg._scene_profile is not None else '<unknown>'}",
            flush=True,
        )

        stage = "env_reset"
        env.reset()
        if args_cli.arrangement == "stack_center":
            stage = "stack_center_pose"
            _apply_stack_center_pose(env, points_xy, cube_sizes, scene_id)

        stage = "settle_after_reset"
        settle_env_after_reset(env, int(args_cli.settle_steps))

        stage = "export_media"
        if str(getattr(args_cli, "export_media", "none")) != "none":
            tag = args_cli.export_run_tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(args_cli.export_dir).expanduser().resolve() / str(tag)
            manifest: dict[str, Any] = {
                "annotation_layout_key": str(args_cli.annotation_layout_key),
                "arrangement": str(args_cli.arrangement),
                "block_color_mode": str(args_cli.block_color_mode),
                "export_cameras_resolved": _resolve_export_camera_names(str(args_cli.export_cameras)),
                "export_media": str(args_cli.export_media),
                "export_video_fps": float(args_cli.export_video_fps),
                "export_video_frames": int(args_cli.export_video_frames),
                "num_objects": int(num_obj),
                "random_seed": getattr(args_cli, "random_seed", None),
                "scene_id": int(scene_id),
                "spacing_effective_m": float(spacing),
            }
            _export_static_media(env, run_dir=run_dir, manifest=manifest)

        stage = "render_loop"
        _run_render_loop(env, int(args_cli.hold_steps), min_live_steps=int(args_cli.min_live_steps))
        return 0
    except Exception:
        print_static_failure_context(stage, env_cfg=env_cfg)
        return 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if simulation_app is not None:
            simulation_app.close()
