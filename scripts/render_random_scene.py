#!/usr/bin/env python3
"""
Randomly generate a hybrid SiFERbench / GeoMind scene and export camera renders.

Tabletop objects are randomly chosen to be either:
  - SiFERbench-style (YCB/custom USD assets, via arrange_objects)
  - Colored blocks (like GeoMind's primitive_cuboid, via VisualCuboid)
  - Mixed (some of each)

Cameras match geomind_ik_rel_visuomotor_env_cfg.py exactly (positions,
focal lengths, apertures). wrist_cam is omitted because there is no robot.

Usage:
  python render_random_scene.py [--headless] [--seed N] [--output-dir PATH]
                                [--settle-steps N] [--num-objects N]
"""
from __future__ import annotations

import argparse
import colorsys
import datetime
import math
import os
import random
import sys
from pathlib import Path

# --- repository roots ---
REPO_ROOT = Path(__file__).resolve().parents[1]       # geomind_lab/
SIFERBENCH_ROOT = REPO_ROOT.parent / "SiFERbench"    # SiFERbench/

for _p in [str(REPO_ROOT), str(SIFERBENCH_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- MDL material search path (mirrors debug_perfect_midpoint.sh) ---
# Must be set before SimulationApp so the MDL compiler picks it up at startup.
_MDL_MATERIALS = (
    "/home/elm/projects/InternUtopia/internutopia/assets/scenes/"
    "GRScenes-100/home_scenes/Materials"
)
_existing_mdl = os.environ.get("MDL_SYSTEM_PATH", "")
if _MDL_MATERIALS not in _existing_mdl:
    os.environ["MDL_SYSTEM_PATH"] = (
        f"{_existing_mdl}:{_MDL_MATERIALS}" if _existing_mdl else _MDL_MATERIALS
    )

# --- CLI args must be parsed before SimulationApp is imported ---
_parser = argparse.ArgumentParser(description="Random scene renderer with camera export.")
_parser.add_argument("--headless", action="store_true", default=False,
                     help="Run without a display (off-screen render).")
_parser.add_argument("--seed", type=int, default=None,
                     help="Random seed for reproducibility.")
_parser.add_argument("--output-dir", type=str, default=None,
                     help="Directory for output images. Defaults to logs/random_scene_render/<timestamp>.")
_parser.add_argument("--settle-steps", type=int, default=300,
                     help="Physics settle steps after world.reset().")
_parser.add_argument("--num-objects", type=int, default=None,
                     help="Number of tabletop objects (random 2-5 if not set).")
_parser.add_argument(
    "--arrangement",
    type=str,
    default="auto",
    choices=["auto", "line", "grid", "t_shape", "l_shape", "circle", "scatter", "stack", "container"],
    help=(
        "Spatial arrangement of tabletop objects. "
        "'auto' keeps the existing random behaviour. "
        "'line' places objects in a straight row. "
        "'grid' forms a rectangular grid. "
        "'t_shape' / 'l_shape' form the corresponding letter shapes. "
        "'circle' places objects on a ring. "
        "'scatter' places objects at random positions. "
        "'stack' piles objects vertically at the same XY (colored blocks or LOW_CAN only). "
        "'container' places a basket at table-centre with a yellow bottle inside; "
        "outputs only table_cam and scene_cam at 1024×1024."
    ),
)
args = _parser.parse_args()

# ── launch Kit ──────────────────────────────────────────────────────────────
from isaacsim import SimulationApp  # noqa: E402
simulation_app = SimulationApp({"headless": args.headless})

# ── isaacsim / omniverse imports (must follow SimulationApp) ─────────────────
import numpy as np  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.api.objects import VisualCuboid  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from isaacsim.core.utils.prims import is_prim_path_valid  # noqa: E402
from isaacsim.core.utils.string import find_unique_string_name  # noqa: E402
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402
import carb  # noqa: E402

# --- Register MDL search path with the running Kit renderer as well ----------
# The env-var approach covers the MDL compiler startup; carb settings covers
# any renderer that resolves paths at material bind time.
def _register_mdl_search_path(path: str) -> None:
    """Append *path* to /renderer/mdl/searchPaths/system via carb settings."""
    try:
        settings = carb.settings.get_settings()
        key = "/renderer/mdl/searchPaths/system"
        current = settings.get(key) or ""
        if path not in str(current):
            updated = f"{current}:{path}" if current else path
            settings.set(key, updated)
    except Exception as exc:
        print(f"[mdl] carb settings update skipped: {exc}")

_register_mdl_search_path(_MDL_MATERIALS)

# ── SiFERbench assets / utilities ────────────────────────────────────────────
from configs.asset_config import *  # noqa: F401, F403, E402
from configs.stage_config import *  # noqa: F401, F403, E402
from configs.object_config import *  # noqa: F401, F403, E402
from siferbench.utils.scene_utils.add_light import add_dome_light  # noqa: E402
from siferbench.utils.scene_utils.add_planes import add_default_plane  # noqa: E402
from siferbench.utils.scene_utils.add_collision import add_collision_to_stage  # noqa: E402
from siferbench.utils.scene_utils.add_walls import add_walls  # noqa: E402
from siferbench.utils.arrange_object import arrange_objects  # noqa: E402

# Build a lookup from OBJECT_ALL / OBJECT_X string names → USDAsset objects.
# The wildcard imports above place e.g. BLUE_CAN, LOW_CAN in globals().
_TABLETOP_ASSETS: list = []
for _nm in [*OBJECT_ALL, *OBJECT_X]:  # type: ignore[name-defined]
    _asset = globals().get(_nm)
    if _asset is not None:
        _TABLETOP_ASSETS.append(_asset)

# Basket half-heights (m) derived from USDA extents × internal scale × asset_config scale.
# basket_1/2 geometry spans ≈ ±6 cm in world space; origin is at the centre.
_BASKET_HALF_H: dict[str, float] = {
    "basket_1": 0.06,
    "basket_2": 0.06,
}
_CONTAINER_BASKETS = [BASKET_1, BASKET_2]  # type: ignore[name-defined]

# Map from --arrangement value to arrange_objects() task name (SiFERbench).
_ARRANGE_OBJECTS_TASK_MAP: dict[str, str] = {
    "line": "Line",
    "grid": "Grid",
    "scatter": "Random",
}

# ── Camera constants — mirrors geomind_ik_rel_visuomotor_env_cfg.py ──────────
LOOK_AT_HEIGHT_ABOVE_TABLE = 0.0448
TABLE_VIEW_CAMERA_HEIGHT_ABOVE_TABLE = 0.4
ROOM_VIEW_CAMERA_HEIGHT_ABOVE_TABLE = 1.7448


# ── helpers ──────────────────────────────────────────────────────────────────

def _rpy_look_at(pos, target) -> np.ndarray:
    """
    Compute [roll=0, pitch, yaw] in degrees so that an isaacsim Camera
    at *pos* points toward *target*.

    Convention matches euler_angles_to_quats used in SiFERbench scripts:
      yaw=0   → faces +X
      yaw=90  → faces +Y
      yaw=180 → faces -X
      pitch>0 → tilts downward
    """
    d = np.array(target, dtype=float) - np.array(pos, dtype=float)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    horiz = math.hypot(d[0], d[1])
    pitch = math.degrees(math.atan2(-d[2], horiz))
    return np.array([0.0, pitch, yaw])


def _set_camera_intrinsics(
    prim_path: str,
    focal_length: float = 24.0,
    horizontal_aperture: float = 20.955,
    focus_distance: float = 400.0,
    near_clip: float = 0.01,
    far_clip: float = 10000.0,
) -> None:
    """Apply PinholeCameraCfg-equivalent intrinsics via the USD camera API.

    near_clip defaults to 0.01 m (1 cm) — the Omniverse default is 1.0 m,
    which clips table-view cameras whose near-surface distance is ~1 m.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[intrinsics] prim not found: {prim_path}")
        return
    cam = UsdGeom.Camera(prim)
    cam.GetFocalLengthAttr().Set(float(focal_length))
    cam.GetHorizontalApertureAttr().Set(float(horizontal_aperture))
    cam.GetFocusDistanceAttr().Set(float(focus_distance))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(float(near_clip), float(far_clip)))


def _make_camera(
    scene,
    name: str,
    position,
    target,
    resolution: tuple[int, int] = (224, 224),
    focal_length: float = 24.0,
    horizontal_aperture: float = 20.955,
    focus_distance: float = 400.0,
) -> Camera:
    """Create, add, and initialise an isaacsim Camera; then set intrinsics."""
    prim_path = find_unique_string_name(
        initial_name=f"/World/{name}",
        is_unique_fn=lambda x: not is_prim_path_valid(x),
    )
    rpy = _rpy_look_at(position, target)
    orientation = euler_angles_to_quats(rpy, degrees=True)
    cam = Camera(
        name=name,
        prim_path=prim_path,
        position=np.array(position, dtype=float),
        frequency=20,
        resolution=resolution,
        orientation=orientation,
    )
    scene.add(cam)
    cam.initialize()
    _set_camera_intrinsics(prim_path, focal_length, horizontal_aperture, focus_distance)
    return cam


def add_all_cameras(
    scene,
    table_height: float,
    table_cam_res: tuple[int, int] = (224, 224),
) -> dict[str, Camera]:
    """
    Add all static cameras from geomind_ik_rel_visuomotor_env_cfg.py.
    *table_height* is the actual tabletop Z (e.g. stage.height) so camera
    heights and look-at points track the chosen stage.
    *table_cam_res* overrides the table_cam resolution (default 224×224).
    wrist_cam is omitted (requires a robot articulation in the scene).
    """
    look_at = (0.3, 0.0, table_height + LOOK_AT_HEIGHT_ABOVE_TABLE)
    h_t = TABLE_VIEW_CAMERA_HEIGHT_ABOVE_TABLE
    h_r = ROOM_VIEW_CAMERA_HEIGHT_ABOVE_TABLE

    cameras: dict[str, Camera] = {}

    # table_cam  (focal_length=24.0)
    cameras["table_cam"] = _make_camera(
        scene, "table_cam",
        position=(1.05, 0.0, table_height + 0.65),
        target=look_at,
        resolution=table_cam_res,
        focal_length=24.0,
    )

    # scene_cam  (focal_length=24.0, 1024×1024)
    # Z is hardcoded in the geomind env cfg (1.85, 0, 2.22); look-at uses table_height.
    cameras["scene_cam"] = _make_camera(
        scene, "scene_cam",
        position=(1.85, 0.0, table_height + h_r),
        target=look_at,
        resolution=(1024, 1024),
        focal_length=24.0,
    )

    # ego_view_camera  (focal_length=10.0 — wide angle)
    ego_look_at = (0.35, 0.0, table_height + LOOK_AT_HEIGHT_ABOVE_TABLE)
    cameras["ego_view_camera"] = _make_camera(
        scene, "ego_view_camera",
        position=(0.12, 0.0, table_height + h_t),
        target=ego_look_at,
        resolution=(224, 224),
        focal_length=10.0,
    )

    # front_camera  (focal_length=24.0)
    cameras["front_camera"] = _make_camera(
        scene, "front_camera",
        position=(1.0, 0.0, table_height + h_r),
        target=look_at,
        resolution=(224, 224),
        focal_length=24.0,
    )

    # left_camera  (focal_length=24.0)
    cameras["left_camera"] = _make_camera(
        scene, "left_camera",
        position=(0.35, 1.0, table_height + h_r),
        target=look_at,
        resolution=(224, 224),
        focal_length=24.0,
    )

    # right_camera  (focal_length=24.0)
    cameras["right_camera"] = _make_camera(
        scene, "right_camera",
        position=(0.35, -1.0, table_height + h_r),
        target=look_at,
        resolution=(224, 224),
        focal_length=24.0,
    )

    return cameras


# ── scene building ────────────────────────────────────────────────────────────

def add_panda_robot(scene, table_height: float) -> None:
    """
    Load the Franka Panda robot and place its base on the tabletop.
    Mirrors geomind franka_env_cfg.py: init_state.pos = [0, 0, table_height].
    Default joint pose from the USD is used (arm is visible but static).
    """
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.core.prims import SingleXFormPrim

    assets_root = get_assets_root_path()
    panda_usd = f"{assets_root}/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"

    robot_prim_path = find_unique_string_name(
        initial_name="/World/Robot",
        is_unique_fn=lambda x: not is_prim_path_valid(x),
    )
    add_reference_to_stage(panda_usd, prim_path=robot_prim_path)

    robot_xform = SingleXFormPrim(
        prim_path=robot_prim_path,
        name="robot",
        position=np.array([0.0, 0.0, float(table_height)]),
    )
    scene.add(robot_xform)
    print(f"[scene] panda robot at z={table_height:.4f}  ({robot_prim_path})")


def _add_asset(scene, asset, position):
    """Add a USDAsset as a SingleRigidPrim."""
    obj_name = find_unique_string_name(
        initial_name=asset.name, is_unique_fn=lambda x: not scene.object_exists(x)
    )
    prim_path = find_unique_string_name(
        initial_name=asset.prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
    )
    add_reference_to_stage(asset.usd_path, prim_path=prim_path)
    scene.add(
        SingleRigidPrim(
            prim_path=prim_path,
            name=obj_name,
            position=position,
            orientation=asset.orientation[0],
            scale=asset.scale,
        )
    )


def add_stage_furniture(scene, stage_asset) -> None:
    """Load all table/chair/sofa/etc. prims that make up the room stage."""
    for asset, position in zip(stage_asset.items, stage_asset.positions):
        _add_asset(scene, asset, position)


def add_container_objects(scene, tabletop_z: float) -> None:
    """
    Container mode: place one basket at the table centre, then a yellow bottle inside.

    Basket half-height ≈ 0.06 m (derived from USDA extent × scale).
    Bottle is dropped from above the rim; physics settling seats it inside.
    """
    basket = random.choice(_CONTAINER_BASKETS)
    half_h = _BASKET_HALF_H.get(basket.name, 0.06)

    basket_pos = np.array([0.4, 0.0, tabletop_z + half_h])
    _add_asset(scene, basket, basket_pos)
    print(f"[scene] container basket={basket.name} pos_z={basket_pos[2]:.4f}")

    # Drop the bottle from just above the basket rim so it settles inside.
    bottle_z = tabletop_z + half_h * 2.0 + 0.04 + OBJECT_HEIGHT["YELLOW_BOTTLE"]  # type: ignore[name-defined]
    bottle_pos = np.array([0.4, 0.0, bottle_z])
    _add_asset(scene, YELLOW_BOTTLE, bottle_pos)  # type: ignore[name-defined]
    print(f"[scene] container bottle=yellow_bottle pos_z={bottle_pos[2]:.4f}")


def add_colored_blocks(
    scene,
    tabletop: float,
    num_objects: int,
    mid: tuple[float, float] = (0.4, 0.1),
    cube_size: float = 0.045,
) -> None:
    """
    Place colored VisualCuboid blocks on the tabletop (GeoMind-style).
    Blocks are arranged in a line with random orientation and small XY jitter.
    """
    interval = random.uniform(0.12, 0.14)
    axis = random.choice(["x", "y"])
    z = tabletop + cube_size / 2.0

    for i in range(num_objects):
        offset = interval * (i - (num_objects - 1) / 2.0)
        nx = random.uniform(-0.02, 0.02)
        ny = random.uniform(-0.02, 0.02)
        if axis == "y":
            pos = np.array([mid[0] + nx, mid[1] + offset + ny, z])
        else:
            pos = np.array([mid[0] + offset + nx, mid[1] + ny, z])

        hue = float(i) / float(max(num_objects, 1))
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.92)

        prim_path = find_unique_string_name(
            initial_name=f"/World/colored_block_{i}",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        block_name = find_unique_string_name(
            initial_name=f"colored_block_{i}",
            is_unique_fn=lambda x: not scene.object_exists(x),
        )
        scene.add(
            VisualCuboid(
                prim_path=prim_path,
                name=block_name,
                position=pos,
                color=np.array([r, g, b]),
                size=cube_size,
            )
        )


def _arrangement_positions(
    arrangement: str,
    num_objects: int,
    center: tuple[float, float] = (0.4, 0.0),
    spacing: float = 0.12,
) -> list[tuple[float, float]]:
    """
    Return a list of *num_objects* absolute (x, y) table positions for the
    requested spatial pattern, centred around *center*.

    Coordinate convention (as seen from the front camera looking along -X):
      X  = table depth  (larger X = further from robot)
      Y  = table width  (positive Y = camera-left)

    Patterns
    --------
    line    – row of objects along the Y axis
    grid    – rectangular NxM grid, as square as possible
    t_shape – horizontal bar along Y at the top, stem extends toward robot (-X)
    l_shape – vertical arm along X, horizontal arm along +Y from its far end
    circle  – objects evenly spaced on a ring of adaptive radius
    scatter – random positions within ±1.5 × spacing of center
    """
    cx, cy = center
    n = max(1, num_objects)

    if arrangement == "line":
        return [(cx, cy + spacing * (i - (n - 1) / 2.0)) for i in range(n)]

    elif arrangement == "grid":
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        pos: list[tuple[float, float]] = []
        for r in range(rows):
            for c in range(cols):
                if len(pos) >= n:
                    break
                pos.append((
                    cx + spacing * (r - (rows - 1) / 2.0),
                    cy + spacing * (c - (cols - 1) / 2.0),
                ))
        return pos

    elif arrangement == "t_shape":
        # Top bar: up to 3 items along Y; stem: remaining items toward robot (-X).
        top = min(3, n)
        stem = n - top
        pos = []
        for i in range(top):
            pos.append((cx, cy + spacing * (i - (top - 1) / 2.0)))
        for j in range(1, stem + 1):
            pos.append((cx - j * spacing, cy))
        return pos

    elif arrangement == "l_shape":
        # Vertical arm along +X (away from robot), horizontal arm along +Y at the far end.
        vert = math.ceil(n / 2)
        horiz = n - vert
        pos = []
        for i in range(vert):
            pos.append((cx + spacing * (vert - 1 - i), cy))
        x_corner = cx + spacing * (vert - 1)
        for j in range(1, horiz + 1):
            pos.append((x_corner, cy + spacing * j))
        return pos

    elif arrangement == "circle":
        radius = spacing * max(1.0, n / (2 * math.pi))
        return [
            (cx + radius * math.cos(2 * math.pi * i / n),
             cy + radius * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]

    else:  # scatter / unknown
        return [
            (cx + random.uniform(-spacing * 1.5, spacing * 1.5),
             cy + random.uniform(-spacing * 1.5, spacing * 1.5))
            for _ in range(n)
        ]


def _add_colored_blocks_at_positions(
    scene,
    tabletop: float,
    positions: list[tuple[float, float]],
    cube_size: float = 0.045,
) -> None:
    """Place one colored VisualCuboid at each (x, y) position."""
    n = len(positions)
    z = tabletop + cube_size / 2.0
    for i, (px, py) in enumerate(positions):
        hue = float(i) / float(max(n, 1))
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
        prim_path = find_unique_string_name(
            initial_name=f"/World/block_arr_{i}",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        blk_name = find_unique_string_name(
            initial_name=f"block_arr_{i}",
            is_unique_fn=lambda x: not scene.object_exists(x),
        )
        scene.add(VisualCuboid(
            prim_path=prim_path,
            name=blk_name,
            position=np.array([px, py, z]),
            color=np.array([r, g, b]),
            size=cube_size,
        ))


def _add_siferbench_objects_at_positions(
    scene,
    tabletop: float,
    positions: list[tuple[float, float]],
) -> None:
    """Place one randomly chosen SiFERbench USD asset at each (x, y) position."""
    assets = _TABLETOP_ASSETS or [LOW_CAN]  # type: ignore[name-defined]
    for px, py in positions:
        asset = random.choice(assets)
        half_h = OBJECT_HEIGHT.get(asset.name.upper(), 0.05)  # type: ignore[name-defined]
        _add_asset(scene, asset, np.array([px, py, tabletop + half_h]))


def add_stacked_colored_blocks(
    scene,
    tabletop: float,
    num_objects: int,
    center: tuple[float, float] = (0.4, 0.0),
    cube_size: float = 0.045,
) -> None:
    """Place colored VisualCuboid blocks in a vertical stack at *center*."""
    for i in range(num_objects):
        z = tabletop + cube_size * (i + 0.5)
        hue = float(i) / float(max(num_objects, 1))
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
        prim_path = find_unique_string_name(
            initial_name=f"/World/stacked_block_{i}",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        block_name = find_unique_string_name(
            initial_name=f"stacked_block_{i}",
            is_unique_fn=lambda x: not scene.object_exists(x),
        )
        scene.add(
            VisualCuboid(
                prim_path=prim_path,
                name=block_name,
                position=np.array([center[0], center[1], float(z)]),
                color=np.array([r, g, b]),
                size=cube_size,
            )
        )


def add_stacked_low_cans(
    scene,
    tabletop: float,
    num_objects: int,
    center: tuple[float, float] = (0.4, 0.0),
) -> None:
    """Place LOW_CAN objects in a vertical stack at *center*."""
    half_h = OBJECT_HEIGHT["LOW_CAN"]   # 0.05 m
    full_h = 2.0 * half_h               # 0.10 m  (one can's height)
    for i in range(num_objects):
        z = tabletop + half_h + full_h * i
        obj_name = find_unique_string_name(
            initial_name="stacked_low_can",
            is_unique_fn=lambda x: not scene.object_exists(x),
        )
        prim_path = find_unique_string_name(
            initial_name="/World/stacked_low_can",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        add_reference_to_stage(LOW_CAN.usd_path, prim_path=prim_path)
        scene.add(
            SingleRigidPrim(
                prim_path=prim_path,
                name=obj_name,
                position=np.array([center[0], center[1], float(z)]),
                orientation=LOW_CAN.orientation[0],
                scale=LOW_CAN.scale,
            )
        )


def populate_tabletop(
    scene,
    stage_asset,
    num_objects: int,
    arrangement: str = "auto",
) -> None:
    """
    Place objects on the tabletop.

    Object type (mode) is always chosen at random:
      siferbench     – YCB/custom USD assets via arrange_objects()
      colored_blocks – coloured VisualCuboid blocks (GeoMind-style)
      mixed          – some of each

    *arrangement* controls the spatial pattern:
      auto    – picks a random arrangement (existing behaviour, including
                probabilistic stacking for colored_blocks / LOW_CAN)
      stack   – forces vertical stacking regardless of mode
      line / grid / t_shape / l_shape / circle / scatter
              – places objects in the named pattern; overrides arrange_objects
                for siferbench when the pattern isn't natively supported
    """
    mode = random.choice(["siferbench", "colored_blocks", "mixed"])
    tabletop_z = stage_asset.height

    # ── forced stack ─────────────────────────────────────────────────────────
    if arrangement == "stack":
        if mode == "siferbench":
            add_stacked_low_cans(scene, tabletop_z, num_objects)
            print(f"[scene] mode=siferbench/stack obj=LOW_CAN n={num_objects}")
        else:
            add_stacked_colored_blocks(scene, tabletop_z, num_objects)
            print(f"[scene] mode={mode}/stack n={num_objects}")
        return

    # ── auto (probabilistic legacy behaviour) ────────────────────────────────
    if arrangement == "auto":
        if mode == "siferbench":
            obj = random.choice(OBJECT_ALL)
            half_h = OBJECT_HEIGHT.get(obj, 0.05)
            if obj == "LOW_CAN" and random.random() < 0.5:
                add_stacked_low_cans(scene, tabletop_z, num_objects)
                print(f"[scene] mode=siferbench/stacked obj=LOW_CAN n={num_objects}")
                return
            arrange_objects(
                task=random.choice(["Line", "Grid", "Random"]),
                tabletop=tabletop_z + half_h,
                object_num=[num_objects],
                noise=0.02,
                znoise=np.pi / 24,
            )
        elif mode == "colored_blocks":
            if random.random() < 0.5:
                add_stacked_colored_blocks(scene, tabletop_z, num_objects)
                print(f"[scene] mode=colored_blocks/stacked n={num_objects}")
                return
            add_colored_blocks(scene, tabletop_z, num_objects)
        else:  # mixed
            n_sifer = max(1, num_objects // 2)
            n_color = num_objects - n_sifer
            if n_sifer > 0:
                obj = random.choice(OBJECT_ALL)
                half_h = OBJECT_HEIGHT.get(obj, 0.05)
                arrange_objects(
                    task=random.choice(["Line", "Grid", "Random"]),
                    tabletop=tabletop_z + half_h,
                    object_num=[n_sifer],
                    noise=0.02,
                    znoise=np.pi / 24,
                )
            if n_color > 0:
                add_colored_blocks(scene, tabletop_z, n_color, mid=(0.5, -0.15))
        print(f"[scene] mode={mode} arrangement=auto n={num_objects}")
        return

    # ── explicit arrangement ──────────────────────────────────────────────────
    positions = _arrangement_positions(arrangement, num_objects)

    if mode == "siferbench":
        task = _ARRANGE_OBJECTS_TASK_MAP.get(arrangement)
        if task:
            # line / grid / scatter: delegate to arrange_objects (its own layout)
            obj = random.choice(OBJECT_ALL)
            half_h = OBJECT_HEIGHT.get(obj, 0.05)
            arrange_objects(
                task=task,
                tabletop=tabletop_z + half_h,
                object_num=[num_objects],
                noise=0.02,
                znoise=np.pi / 24,
            )
        else:
            # t_shape / l_shape / circle: place USD assets at computed positions
            _add_siferbench_objects_at_positions(scene, tabletop_z, positions)

    elif mode == "colored_blocks":
        _add_colored_blocks_at_positions(scene, tabletop_z, positions)

    else:  # mixed
        n_sifer = max(1, num_objects // 2)
        n_color = num_objects - n_sifer
        if n_sifer > 0:
            task = _ARRANGE_OBJECTS_TASK_MAP.get(arrangement)
            if task:
                obj = random.choice(OBJECT_ALL)
                half_h = OBJECT_HEIGHT.get(obj, 0.05)
                arrange_objects(
                    task=task,
                    tabletop=tabletop_z + half_h,
                    object_num=[n_sifer],
                    noise=0.02,
                    znoise=np.pi / 24,
                )
            else:
                _add_siferbench_objects_at_positions(scene, tabletop_z, positions[:n_sifer])
        if n_color > 0:
            _add_colored_blocks_at_positions(scene, tabletop_z, positions[n_sifer:])

    print(f"[scene] mode={mode} arrangement={arrangement} n={num_objects}")


# ── image export ──────────────────────────────────────────────────────────────

def _save_rgb(rgb: np.ndarray, path: Path) -> None:
    """Save a (H, W, 3) uint8 array as PNG, falling back to .npy if needed."""
    try:
        import torch
        from torchvision.io import write_png
        t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        write_png(t, str(path))
        return
    except Exception:
        pass
    try:
        from PIL import Image
        Image.fromarray(rgb, "RGB").save(str(path))
        return
    except Exception:
        pass
    np.save(str(path.with_suffix(".npy")), rgb)


def export_images(cameras: dict[str, Camera], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, cam in cameras.items():
        try:
            rgba = cam.get_rgba()
            if rgba is None or rgba.size == 0:
                print(f"[export] {name}: no frame available, skipping")
                continue
            rgb = rgba[..., :3].astype(np.uint8)
            out_path = output_dir / f"{name}.png"
            _save_rgb(rgb, out_path)
            print(f"[export] {name} → {out_path}")
        except Exception as exc:
            print(f"[export] {name}: failed — {exc}")


# ── lighting ──────────────────────────────────────────────────────────────────

def add_scene_lights(
    table_height: float = 0.755,
    dome_intensity: float = 1500.0,
    key_intensity: float = 3500.0,
    fill_intensity: float = 2500.0,
) -> None:
    """
    Three-point lighting.  Default values sit midway between the bare
    add_dome_light() (450) and geomind_scene_cfg's bright preset (dome 3000,
    key 8000, fill 5500).
    """
    stage = omni.usd.get_context().get_stage()

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr().Set(dome_intensity)
    dome.CreateColorAttr().Set(Gf.Vec3f(0.82, 0.84, 0.88))
    dome.CreateEnableColorTemperatureAttr().Set(True)
    dome.CreateColorTemperatureAttr().Set(6500.0)

    key = UsdLux.SphereLight.Define(stage, "/World/key_light")
    key.AddTranslateOp().Set(Gf.Vec3d(1.0, -1.0, table_height + 2.4))
    key.CreateIntensityAttr().Set(key_intensity)
    key.CreateRadiusAttr().Set(0.3)
    key.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.9))
    key.CreateEnableColorTemperatureAttr().Set(True)
    key.CreateColorTemperatureAttr().Set(4600.0)

    fill = UsdLux.SphereLight.Define(stage, "/World/fill_light")
    fill.AddTranslateOp().Set(Gf.Vec3d(-1.0, 1.0, table_height + 1.9))
    fill.CreateIntensityAttr().Set(fill_intensity)
    fill.CreateRadiusAttr().Set(0.45)
    fill.CreateColorAttr().Set(Gf.Vec3f(0.88, 0.92, 1.0))
    fill.CreateEnableColorTemperatureAttr().Set(True)
    fill.CreateColorTemperatureAttr().Set(7200.0)

    print(f"[lights] dome={dome_intensity}  key={key_intensity}  fill={fill_intensity}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "logs" / "random_scene_render" / ts

    world = World()

    # Stage chosen first so lights are positioned relative to table height
    stage = random.choice([STAGE_1, STAGE_2, STAGE_3, STAGE_4, STAGE_5])
    print(f"[scene] stage height={stage.height}")

    # Moderate three-point lighting (dome 1500 / key 3500 / fill 2500)
    add_scene_lights(table_height=stage.height)
    add_default_plane(random.choice(["wood_ground", "wood_ground_2", "wood_ground_3", "wood_ground_4"]))
    add_walls(world.scene)
    add_stage_furniture(world.scene, stage)

    # Franka Panda robot — base at table surface (mirrors franka_env_cfg.py)
    add_panda_robot(world.scene, stage.height)

    is_container = args.arrangement == "container"

    if is_container:
        add_container_objects(world.scene, stage.height)
    else:
        num_objects = args.num_objects if args.num_objects else random.randint(2, 5)
        populate_tabletop(world.scene, stage, num_objects, arrangement=args.arrangement)

    # Enable collision on all scene geometry
    add_collision_to_stage()

    # Cameras — heights and look-at tied to actual stage table height.
    # Container mode uses 1024×1024 for both table_cam and scene_cam.
    table_cam_res = (1024, 1024) if is_container else (224, 224)
    cameras = add_all_cameras(world.scene, stage.height, table_cam_res=table_cam_res)

    # Simulate: reset → settle → warm-up renders
    world.reset()
    for step in range(args.settle_steps):
        world.step(render=True)

    # Extra frames so camera sensors have valid data
    for _ in range(30):
        world.step(render=True)

    # Container mode: only export the two overview cameras.
    cameras_to_export = (
        {k: cameras[k] for k in ("table_cam", "scene_cam") if k in cameras}
        if is_container else cameras
    )
    export_images(cameras_to_export, out_dir)
    print(f"\n[render_random_scene] Done. Images saved to: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
