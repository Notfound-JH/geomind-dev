from __future__ import annotations

import re
import os
import random
from dataclasses import MISSING, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from geomind_lab.utils import asset_utils, scene_utils


GEOMIND_ASSETS_ROOT = asset_utils.GEOMIND_ASSETS_ROOT
TEST_SCENE_ID = 1
SCENE_PATH = f"{GEOMIND_ASSETS_ROOT}/stage"
SCENE_CATEGORIES = ("function", "object", "container", "decoration")
WALL_COLOR = tuple(channel / 255.0 for channel in (139, 69, 19))
AMBIENT_LIGHT_COLOR = (0.82, 0.84, 0.88)
KEY_LIGHT_COLOR = (1.0, 0.95, 0.9)
FILL_LIGHT_COLOR = (0.88, 0.92, 1.0)
PRACTICAL_LAMP_COLOR = (1.0, 0.86, 0.7)
AMBIENT_LIGHT_INTENSITY =3000.0
KEY_LIGHT_INTENSITY = 8000.0
FILL_LIGHT_INTENSITY = 5500.0
PRACTICAL_LAMP_INTENSITY = 6000.0
DEFAULT_LIGHTING_MODE = "natural"
GROUND_PLANE_COLOR = (0.34, 0.28, 0.22)
GROUND_PLANE_SIZE = (12.0, 12.0)
GROUND_CUBOID_THICKNESS = 0.02
TABLE_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=3.0,
    dynamic_friction=2.5,
    restitution=0.0,
    friction_combine_mode="max",
)


@dataclass(frozen=True)
class LightingPreset:
    ambient_intensity: float
    key_intensity: float
    fill_intensity: float
    lamp_intensity: float


@dataclass(frozen=True)
class LightingPoolVariant:
    """A gentle lighting variation around the base preset."""

    name: str
    ambient_scale: float
    key_scale: float
    fill_scale: float
    lamp_scale: float
    ambient_color: tuple[float, float, float]
    key_color: tuple[float, float, float]
    fill_color: tuple[float, float, float]
    lamp_color: tuple[float, float, float]
    key_delta_xyz: tuple[float, float, float]
    fill_delta_xyz: tuple[float, float, float]
    lamp_delta_z: float
    key_temperature: float
    fill_temperature: float
    lamp_temperature: float
    ambient_temperature: float


LIGHTING_PRESETS: dict[str, LightingPreset] = {
    "natural": LightingPreset(
        ambient_intensity=AMBIENT_LIGHT_INTENSITY,
        key_intensity=KEY_LIGHT_INTENSITY,
        fill_intensity=FILL_LIGHT_INTENSITY,
        lamp_intensity=PRACTICAL_LAMP_INTENSITY,
    ),
    "bright": LightingPreset(
        ambient_intensity=3000.0,
        key_intensity=12000.0,
        fill_intensity=8500.0,
        lamp_intensity=9000.0,
    ),
}


_LIGHTING_COLOR_BASES: tuple[tuple[float, float, float], ...] = (
    (0.15, 0.85, 1.00),
    (1.00, 0.22, 0.32),
    (0.74, 0.25, 1.00),
    (1.00, 0.72, 0.16),
    (0.20, 0.95, 0.38),
    (0.20, 0.34, 1.00),
    (1.00, 1.00, 1.00),
)

_KEY_LIGHT_BASE_POS = (1.1, -1.0, 1.75)
_FILL_LIGHT_BASE_POS = (-1.0, 0.85, 1.35)


def _light_position_delta(
    desired_relative_pos: tuple[float, float, float],
    base_relative_pos: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(float(desired - base) for desired, base in zip(desired_relative_pos, base_relative_pos))


def _sample_light_positions(
    rng: random.Random,
    variant_index: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    mode = variant_index % 5
    if mode == 0:
        key_pos = (rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), rng.uniform(3.0, 4.6))
        fill_pos = (rng.uniform(-2.2, 2.2), rng.uniform(-2.0, 2.0), rng.uniform(1.5, 2.8))
        lamp_delta_z = rng.uniform(-0.08, 0.30)
    elif mode == 1:
        side = rng.choice((-1.0, 1.0))
        key_pos = (side * rng.uniform(3.0, 5.2), rng.uniform(-2.4, 2.8), rng.uniform(2.0, 4.5))
        fill_pos = (-side * rng.uniform(1.2, 2.8), rng.uniform(-1.6, 1.8), rng.uniform(1.0, 2.4))
        lamp_delta_z = rng.uniform(-0.16, 0.18)
    elif mode == 2:
        key_pos = (rng.uniform(-1.2, 1.2), rng.uniform(2.3, 3.8), rng.uniform(1.5, 3.3))
        fill_pos = (rng.uniform(-1.6, 1.6), rng.uniform(-2.4, -1.3), rng.uniform(0.9, 2.2))
        lamp_delta_z = rng.uniform(-0.10, 0.24)
    elif mode == 3:
        key_pos = (rng.uniform(-0.8, 0.9), rng.uniform(-0.7, 0.8), rng.uniform(0.35, 0.95))
        fill_pos = (rng.uniform(-3.0, 3.0), rng.uniform(-2.6, 2.6), rng.uniform(1.8, 3.8))
        lamp_delta_z = rng.uniform(0.00, 0.42)
    else:
        key_pos = (rng.uniform(-2.8, 2.8), rng.uniform(-2.8, 2.8), rng.uniform(1.0, 4.2))
        fill_pos = (rng.uniform(-2.8, 2.8), rng.uniform(-2.8, 2.8), rng.uniform(0.8, 3.2))
        lamp_delta_z = rng.uniform(-0.20, 0.34)
    return (
        _light_position_delta(key_pos, _KEY_LIGHT_BASE_POS),
        _light_position_delta(fill_pos, _FILL_LIGHT_BASE_POS),
        lamp_delta_z,
    )


def _pastel_light_color(base: tuple[float, float, float], tint: float) -> tuple[float, float, float]:
    return tuple(round(1.0 - tint + channel * tint, 4) for channel in base)


def _vivid_light_color(base: tuple[float, float, float], saturation: float, lift: float) -> tuple[float, float, float]:
    return tuple(round(min(1.0, max(0.0, lift + channel * saturation)), 4) for channel in base)


def _lighting_variant_signature(variant: LightingPoolVariant) -> tuple[float, ...]:
    return (
        variant.ambient_scale,
        variant.key_scale,
        variant.fill_scale,
        variant.lamp_scale,
        *variant.key_delta_xyz,
        *variant.fill_delta_xyz,
        variant.lamp_delta_z,
        *variant.key_color,
    )


def _lighting_signature_distance(first: LightingPoolVariant, second: LightingPoolVariant) -> float:
    first_signature = _lighting_variant_signature(first)
    second_signature = _lighting_variant_signature(second)
    weights = (1.6, 1.4, 1.4, 1.0, 0.7, 0.7, 0.8, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
    return sum(weight * abs(left - right) for left, right, weight in zip(first_signature, second_signature, weights))


def _build_lighting_pool(count: int = 15, seed: int = 20260814) -> tuple[LightingPoolVariant, ...]:
    """Generate visibly distinct variants with deterministic rejection sampling."""
    rng = random.Random(seed)
    variants: list[LightingPoolVariant] = []
    max_attempts = count * 1000
    attempts = 0
    while len(variants) < count and attempts < max_attempts:
        attempts += 1
        if not variants:
            # Reserve one deliberately weak but still visible sample for low-light coverage.
            weak_base = (0.38, 0.78, 0.82)
            variants.append(
                LightingPoolVariant(
                    name="pool_00",
                    ambient_scale=0.02,
                    key_scale=0.04,
                    fill_scale=0.02,
                    lamp_scale=1.35,
                    ambient_color=(0.12, 0.18, 0.26),
                    key_color=(0.18, 0.48, 1.0),
                    fill_color=(0.18, 0.72, 0.42),
                    lamp_color=(1.0, 0.46, 0.10),
                    key_delta_xyz=(2.8, -0.4, 0.7),
                    fill_delta_xyz=(-2.2, 1.3, 1.0),
                    lamp_delta_z=0.16,
                    key_temperature=6500.0,
                    fill_temperature=8200.0,
                    lamp_temperature=5200.0,
                    ambient_temperature=7200.0,
                )
            )
            continue
        variant_index = len(variants)
        if variant_index <= 4:
            weak_variants = (
                {
                    "ambient_scale": 0.05,
                    "key_scale": 0.82,
                    "fill_scale": 0.03,
                    "lamp_scale": 0.04,
                    "ambient_color": (0.06, 0.10, 0.18),
                    "key_color": (0.12, 0.42, 1.0),
                    "fill_color": (0.08, 0.12, 0.18),
                    "lamp_color": (0.18, 0.30, 0.55),
                    "key_delta_xyz": (-5.30, 3.60, 1.75),
                    "fill_delta_xyz": (1.80, -2.50, 0.20),
                    "lamp_delta_z": -0.10,
                },
                {
                    "ambient_scale": 0.03,
                    "key_scale": 1.10,
                    "fill_scale": 0.01,
                    "lamp_scale": 0.06,
                    "ambient_color": (0.10, 0.02, 0.12),
                    "key_color": (1.0, 0.04, 0.78),
                    "fill_color": (0.08, 0.02, 0.14),
                    "lamp_color": (1.0, 0.18, 0.45),
                    "key_delta_xyz": (-1.00, 0.85, -1.10),
                    "fill_delta_xyz": (2.60, -2.60, 0.30),
                    "lamp_delta_z": -0.12,
                },
                {
                    "ambient_scale": 0.01,
                    "key_scale": 1.05,
                    "fill_scale": 0.08,
                    "lamp_scale": 0.45,
                    "ambient_color": (0.03, 0.12, 0.08),
                    "key_color": (0.10, 1.0, 0.26),
                    "fill_color": (0.08, 0.22, 0.14),
                    "lamp_color": (0.30, 1.0, 0.42),
                    "key_delta_xyz": (-1.35, 1.60, -1.20),
                    "fill_delta_xyz": (3.10, -0.90, 1.80),
                    "lamp_delta_z": 0.30,
                },
                {
                    "ambient_scale": 0.48,
                    "key_scale": 0.18,
                    "fill_scale": 0.22,
                    "lamp_scale": 0.06,
                    "ambient_color": (0.18, 0.32, 0.55),
                    "key_color": (0.40, 0.55, 1.0),
                    "fill_color": (0.22, 0.78, 0.88),
                    "lamp_color": (0.20, 0.34, 0.80),
                    "key_delta_xyz": (0.00, 0.40, 2.75),
                    "fill_delta_xyz": (-1.20, 1.80, 1.00),
                    "lamp_delta_z": -0.18,
                },
            )
            selected = weak_variants[variant_index - 1]
            variants.append(
                LightingPoolVariant(
                    name=f"pool_{variant_index:02d}",
                    ambient_scale=float(selected["ambient_scale"]),
                    key_scale=float(selected["key_scale"]),
                    fill_scale=float(selected["fill_scale"]),
                    lamp_scale=float(selected["lamp_scale"]),
                    ambient_color=tuple(selected["ambient_color"]),
                    key_color=tuple(selected["key_color"]),
                    fill_color=tuple(selected["fill_color"]),
                    lamp_color=tuple(selected["lamp_color"]),
                    key_delta_xyz=tuple(selected["key_delta_xyz"]),
                    fill_delta_xyz=tuple(selected["fill_delta_xyz"]),
                    lamp_delta_z=float(selected["lamp_delta_z"]),
                    key_temperature=6500.0,
                    fill_temperature=7200.0,
                    lamp_temperature=3600.0,
                    ambient_temperature=7600.0,
                )
            )
            continue
        elif variant_index <= 9:
            ambient_scale = rng.uniform(0.18, 0.62)
            key_scale = rng.uniform(0.38, 1.35)
            fill_scale = rng.uniform(0.18, 1.05)
            lamp_scale = rng.uniform(0.18, 1.15)
        else:
            ambient_scale = rng.uniform(0.40, 1.00)
            key_scale = rng.uniform(0.85, 2.35)
            fill_scale = rng.uniform(0.35, 1.80)
            lamp_scale = rng.uniform(0.35, 1.90)
        key_delta, fill_delta, lamp_delta_z = _sample_light_positions(rng, variant_index)
        base = rng.choice(_LIGHTING_COLOR_BASES)
        key_color = _vivid_light_color(base, rng.uniform(0.55, 1.00), rng.uniform(0.00, 0.18))
        fill_base = rng.choice(_LIGHTING_COLOR_BASES)
        fill_color = _vivid_light_color(fill_base, rng.uniform(0.45, 0.95), rng.uniform(0.04, 0.24))
        ambient_color = _vivid_light_color(base, rng.uniform(0.25, 0.60), rng.uniform(0.18, 0.36))
        lamp_color = _vivid_light_color(base, rng.uniform(0.60, 1.00), rng.uniform(0.02, 0.20))
        variant = LightingPoolVariant(
            name=f"pool_{len(variants):02d}",
            ambient_scale=ambient_scale,
            key_scale=key_scale,
            fill_scale=fill_scale,
            lamp_scale=lamp_scale,
            ambient_color=ambient_color,
            key_color=key_color,
            fill_color=fill_color,
            lamp_color=lamp_color,
            key_delta_xyz=key_delta,
            fill_delta_xyz=fill_delta,
            lamp_delta_z=lamp_delta_z,
            key_temperature=rng.uniform(4200.0, 7800.0),
            fill_temperature=rng.uniform(5200.0, 9000.0),
            lamp_temperature=rng.uniform(3000.0, 6200.0),
            ambient_temperature=rng.uniform(6000.0, 7800.0),
        )
        if all(_lighting_signature_distance(variant, previous) >= 3.20 for previous in variants):
            variants.append(variant)

    if len(variants) != count:
        raise RuntimeError(f"Unable to generate {count} distinct lighting variants after {attempts} attempts")
    return tuple(variants)


LIGHTING_POOL = _build_lighting_pool()


def normalize_lighting_mode(lighting_mode: str | None) -> str:
    if lighting_mode is None:
        return DEFAULT_LIGHTING_MODE

    normalized = str(lighting_mode).strip().lower().replace("-", "_")
    alias_map = {
        "natural": "natural",
        "normal": "natural",
        "bright": "bright",
        "high": "bright",
        "high_brightness": "bright",
        "pool": "pool",
        "random_pool": "pool",
    }
    resolved = alias_map.get(normalized)
    if resolved is None:
        valid_modes = ", ".join(sorted(LIGHTING_PRESETS))
        raise ValueError(f"Unsupported lighting_mode '{lighting_mode}'. Expected one of: {valid_modes}")
    return resolved


def resolve_lighting_preset(lighting_mode: str | None) -> LightingPreset:
    mode = normalize_lighting_mode(lighting_mode)
    if mode == "pool":
        mode = DEFAULT_LIGHTING_MODE
    return LIGHTING_PRESETS[mode]


def resolve_lighting_pool_variant(scene_id: int, lighting_mode: str | None) -> tuple[int, LightingPoolVariant] | None:
    """Resolve one entry from the fixed 15-lighting pool.

    Selection policy:
    - `GEOMIND_LIGHTING_POOL_INDEX` wins when set.
    - otherwise choose pseudo-randomly when pool randomization is enabled.
    - random selection becomes reproducible when `GEOMIND_LIGHTING_POOL_SEED` is set.
    - `GEOMIND_LIGHTING_POOL_ENABLE` defaults to disabled; set it to `1/true/yes/on` to enable.
    """
    mode = normalize_lighting_mode(lighting_mode)
    enabled_raw = str(os.environ.get("GEOMIND_LIGHTING_POOL_ENABLE", "0")).strip().lower()
    enabled = mode == "pool" or enabled_raw not in {"0", "false", "no", "off"}
    if not enabled:
        return None

    index_raw = str(os.environ.get("GEOMIND_LIGHTING_POOL_INDEX", "")).strip()
    if index_raw:
        try:
            index = int(index_raw, 0)
        except ValueError:
            index = 0
        index = max(0, min(index, len(LIGHTING_POOL) - 1))
        return index, LIGHTING_POOL[index]

    seed_raw = str(os.environ.get("GEOMIND_LIGHTING_POOL_SEED", "")).strip()
    if seed_raw:
        try:
            seed = int(seed_raw, 0)
        except ValueError:
            seed = abs(hash(seed_raw))
        layout_index_raw = str(os.environ.get("LAYOUT_INDEX", "")).strip()
        try:
            layout_index = int(layout_index_raw, 0) if layout_index_raw else 0
        except ValueError:
            layout_index = 0
        rng = random.Random(seed ^ (int(scene_id) * 0x9E3779B1) ^ (layout_index * 0x85EBCA6B))
        index = rng.randrange(len(LIGHTING_POOL))
        return index, LIGHTING_POOL[index]

    index = random.SystemRandom().randrange(len(LIGHTING_POOL))
    return index, LIGHTING_POOL[index]


def env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SceneProfile:
    scene_id: int
    usd_path: str
    structure: dict[str, dict[str, dict[str, Any]]]
    xforms: dict[str, dict[str, tuple[float, ...]]]
    wall_specs: tuple[scene_utils.WallSpec, ...]
    object_names: tuple[str, ...]
    object_assets: dict[str, dict[str, Any]]
    table_name: str | None
    ground_name: str | None
    basket_name: str | None
    lamp_name: str | None
    chair_name: str | None
    blanket_name: str | None
    table_usd_path: str | None
    ground_usd_path: str | None
    basket_usd_path: str | None
    lamp_usd_path: str | None
    chair_usd_path: str | None
    blanket_usd_path: str | None


def get_scene_usd_path(scene_id: int) -> str:
    new_path = Path(f"{SCENE_PATH}/{scene_id}/{scene_id}_new.usda")
    if new_path.is_file():
        return str(new_path)
    return f"{SCENE_PATH}/{scene_id}/{scene_id}.usda"


def _coerce_tuple(value: Any, length: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)) and len(value) >= length:
        return tuple(float(item) for item in value[:length])
    return default


def _node_to_xform(node: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    attrs = node.get("attrs", {}) if isinstance(node, dict) else {}
    pos = _coerce_tuple(attrs.get("translate"), 3, (0.0, 0.0, 0.0))
    rot = _coerce_tuple(attrs.get("orient"), 4, (0.0, 0.0, 0.0, 1.0))
    scale = _coerce_tuple(attrs.get("scale"), 3, (1.0, 1.0, 1.0))
    return {"pos": pos, "rot": rot, "scale": scale}


def _load_scene_structure(nodes: list[scene_utils.PrimNode]) -> dict[str, dict[str, dict[str, Any]]]:
    xform_tree = scene_utils.extract_xform_tree(nodes)

    category_nodes: dict[str, dict[str, Any]] = {}

    def walk(node: dict[str, Any]) -> None:
        if node.get("name") in SCENE_CATEGORIES:
            category_nodes[node["name"]] = node
        for child in node.get("children", []):
            walk(child)

    for node in xform_tree:
        walk(node)

    missing = [name for name in SCENE_CATEGORIES if name not in category_nodes]
    if missing:
        raise ValueError(f"Scene USD missing category prims: {', '.join(missing)}")

    structured: dict[str, dict[str, dict[str, Any]]] = {}
    for category in SCENE_CATEGORIES:
        children = category_nodes[category].get("children", [])
        structured[category] = {child["name"]: child for child in children}
    return structured


def _flatten_xforms(structured: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, tuple[float, ...]]]:
    flattened: dict[str, dict[str, tuple[float, ...]]] = {}
    for assets in structured.values():
        for name, node in assets.items():
            flattened[name] = _node_to_xform(node)
    return flattened


def _object_prefix(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def _find_asset_usd_path(categories: tuple[str, ...], asset_name: str) -> str | None:
    for category in categories:
        info = asset_utils.get_asset_info_by_name(category, asset_name)
        if info and info.get("usd_path"):
            return str(info["usd_path"])
    return None


def _resolve_asset_usd_path(scene_category: str, asset_name: str) -> str | None:
    if scene_category == "function":
        if asset_name.startswith("wood_ground"):
            return asset_utils.get_ground_plane_info(asset_name)
        return _find_asset_usd_path(("table", "furniture", "decoration"), asset_name)
    if scene_category == "object":
        return _find_asset_usd_path(("catch_object",), _object_prefix(asset_name))
    if scene_category == "container":
        return _find_asset_usd_path(("container",), asset_name)
    if scene_category == "decoration":
        return _find_asset_usd_path(("decoration", "furniture"), asset_name)
    return None


def _select_asset_name(structure: dict[str, dict[str, dict[str, Any]]], category: str, prefix: str) -> str | None:
    matches = [name for name in structure.get(category, {}) if name.startswith(prefix)]
    if not matches:
        return None
    return matches[0]


def _build_object_assets(
    object_nodes: dict[str, dict[str, Any]],
    xforms: dict[str, dict[str, tuple[float, ...]]],
) -> dict[str, dict[str, Any]]:
    prefix_usd: dict[str, str] = {}
    result: dict[str, dict[str, Any]] = {}
    for name in object_nodes:
        prefix = _object_prefix(name)
        if prefix not in prefix_usd:
            usd_path = _resolve_asset_usd_path("object", prefix)
            if usd_path is None:
                raise ValueError(f"Unable to resolve object USD path for prefix '{prefix}'")
            prefix_usd[prefix] = usd_path
        xform = xforms[name]
        result[name] = {
            "usd_path": prefix_usd[prefix],
            "orientation": list(xform["rot"]),
            "position": list(xform["pos"]),
        }
    return result


@lru_cache(maxsize=32)
def get_scene_profile(scene_id: int) -> SceneProfile:
    scene_path = get_scene_usd_path(scene_id)
    nodes = scene_utils.parse_usda_scene(scene_path)
    structure = _load_scene_structure(nodes)
    xforms = _flatten_xforms(structure)
    wall_specs = scene_utils.extract_wall_specs(nodes)

    table_name = _select_asset_name(structure, "function", "table")
    ground_name = _select_asset_name(structure, "function", "wood_ground")
    basket_name = _select_asset_name(structure, "container", "basket")
    lamp_name = _select_asset_name(structure, "decoration", "lamp")
    chair_name = _select_asset_name(structure, "decoration", "chair")
    blanket_name = _select_asset_name(structure, "decoration", "blanket")

    object_assets = _build_object_assets(structure.get("object", {}), xforms)

    return SceneProfile(
        scene_id=scene_id,
        usd_path=scene_path,
        structure=structure,
        xforms=xforms,
        wall_specs=wall_specs,
        object_names=tuple(structure.get("object", {}).keys()),
        object_assets=object_assets,
        table_name=table_name,
        ground_name=ground_name,
        basket_name=basket_name,
        lamp_name=lamp_name,
        chair_name=chair_name,
        blanket_name=blanket_name,
        table_usd_path=_resolve_asset_usd_path("function", table_name) if table_name else None,
        ground_usd_path=_resolve_asset_usd_path("function", ground_name) if ground_name else None,
        basket_usd_path=_resolve_asset_usd_path("container", basket_name) if basket_name else None,
        lamp_usd_path=_resolve_asset_usd_path("decoration", lamp_name) if lamp_name else None,
        chair_usd_path=_resolve_asset_usd_path("decoration", chair_name) if chair_name else None,
        blanket_usd_path=_resolve_asset_usd_path("decoration", blanket_name) if blanket_name else None,
    )


def get_scene_xform(prim_name: str, scene_id: int = TEST_SCENE_ID) -> dict[str, tuple[float, ...]]:
    profile = get_scene_profile(scene_id)
    if prim_name not in profile.xforms:
        raise KeyError(f"Prim '{prim_name}' not found in scene USD: {profile.usd_path}")
    return profile.xforms[prim_name]


def spawn_usd_static_mesh_collision(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.from_files import from_files
    from isaaclab.sim.utils import bind_physics_material
    from pxr import Usd, UsdGeom

    prim = from_files.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    root = prim
    resolved_prim_path = root.GetPath().pathString

    from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, SDFMeshPropertiesCfg

    col_props = CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0)
    mesh_props = SDFMeshPropertiesCfg()
    physics_material = getattr(cfg, "_geomind_physics_material", None)
    physics_material_path = None
    if physics_material is not None:
        physics_material_path = f"{resolved_prim_path}/physicsMaterial"
        physics_material.func(physics_material_path, physics_material)

    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Mesh):
            mesh_path = p.GetPath().pathString
            schemas.define_collision_properties(mesh_path, col_props)
            schemas.define_mesh_collision_properties(mesh_path, mesh_props)
            if physics_material_path is not None:
                bind_physics_material(mesh_path, physics_material_path)

    return prim


@configclass
class GeoMindSceneCfg(InteractiveSceneCfg):
    """GeoMind scene configured from scene USD structure."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    furniture_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/furniture_table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{GEOMIND_ASSETS_ROOT}/table/table_1.usda",
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    furniture_chair = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/furniture_chair",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{GEOMIND_ASSETS_ROOT}/layout/chair_1.usda",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    container_basket = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/container_basket",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{GEOMIND_ASSETS_ROOT}/objects/basket_1.usda",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    decoration_lamp = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/decoration_lamp",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{GEOMIND_ASSETS_ROOT}/layout/lamp_1.usda",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    decoration_blanket = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/decoration_blanket",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{GEOMIND_ASSETS_ROOT}/layout/blanket_1.usda",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{GEOMIND_ASSETS_ROOT}/ground/wood_ground_1.usda",
            scale=(1.0, 1.0, 1.0),)
        # spawn=sim_utils.GroundPlaneCfg(
        #     color=(0.34, 0.28, 0.22),
        #     size=(12.0, 12.0),
        #     physics_material=TABLE_PHYSICS_MATERIAL,
        # ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=AMBIENT_LIGHT_COLOR,
            intensity=0.0,
            enable_color_temperature=False,
            color_temperature=6500.0,
            visible_in_primary_ray=False,
        ),
    )

    key_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/key_light",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[1.0, -1.0, 2.4]),
        spawn=sim_utils.SphereLightCfg(
            intensity=KEY_LIGHT_INTENSITY,
            radius=0.3,
            color=KEY_LIGHT_COLOR,
            enable_color_temperature=True,
            color_temperature=4600.0,
        ),
    )

    fill_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/fill_light",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-1.0, 1.0, 1.9]),
        spawn=sim_utils.SphereLightCfg(
            intensity=FILL_LIGHT_INTENSITY,
            radius=0.45,
            color=FILL_LIGHT_COLOR,
            enable_color_temperature=True,
            color_temperature=7200.0,
        ),
    )

    def _apply_asset_cfg(self, cfg: AssetBaseCfg | None, xform: dict[str, tuple[float, ...]], usd_path: str) -> None:
        if cfg is None:
            return
        cfg.init_state.pos = list(xform["pos"])
        cfg.init_state.rot = list(xform["rot"])
        cfg.spawn.usd_path = usd_path
        cfg.spawn.scale = tuple(xform["scale"])

    def _wall_attr_name(self, wall_name: str) -> str:
        return f"scene_wall_{re.sub(r'[^0-9a-zA-Z_]+', '_', wall_name.lower())}"

    def _build_wall_cfg(self, wall_spec: scene_utils.WallSpec) -> AssetBaseCfg:
        return AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{wall_spec.name}",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=list(scene_utils.resolve_wall_runtime_position(wall_spec)),
                rot=list(wall_spec.orientation),
            ),
            spawn=sim_utils.CuboidCfg(
                size=scene_utils.resolve_wall_runtime_dimensions(wall_spec),
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=WALL_COLOR, roughness=0.8),
            ),
        )

    def _build_lamp_light_cfg(self, position: tuple[float, float, float], preset: LightingPreset) -> AssetBaseCfg:
        return AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/lamp_light",
            init_state=AssetBaseCfg.InitialStateCfg(pos=list(position)),
            spawn=sim_utils.SphereLightCfg(
                intensity=preset.lamp_intensity,
                radius=0.06,
                treat_as_point=True,
                color=PRACTICAL_LAMP_COLOR,
                enable_color_temperature=True,
                color_temperature=2800.0,
            ),
        )

    def _apply_lighting_preset(self, preset: LightingPreset, variant: LightingPoolVariant | None = None) -> None:
        if variant is None:
            self.key_light.spawn.intensity = preset.key_intensity
            self.fill_light.spawn.intensity = preset.fill_intensity
            self.key_light.spawn.color = KEY_LIGHT_COLOR
            self.fill_light.spawn.color = FILL_LIGHT_COLOR
            self.key_light.spawn.color_temperature = 4600.0
            self.fill_light.spawn.color_temperature = 7200.0
            return

        self.key_light.spawn.intensity = preset.key_intensity * variant.key_scale
        self.fill_light.spawn.intensity = preset.fill_intensity * variant.fill_scale
        self.key_light.spawn.color = variant.key_color
        self.fill_light.spawn.color = variant.fill_color
        self.key_light.spawn.enable_color_temperature = False
        self.fill_light.spawn.enable_color_temperature = False
        self.key_light.spawn.color_temperature = float(variant.key_temperature)
        self.fill_light.spawn.color_temperature = float(variant.fill_temperature)

    def _prune_transient_scene_attrs(self) -> None:
        for attr_name in ("lighting_mode", "_lighting_mode"):
            if attr_name in self.__dict__:
                delattr(self, attr_name)

    def _configure_scene_lighting(
        self,
        profile: SceneProfile,
        preset: LightingPreset,
        variant: LightingPoolVariant | None = None,
    ) -> None:
        if variant is not None and variant.ambient_scale >= 0.45:
            self.light.spawn.intensity = preset.ambient_intensity * variant.ambient_scale
            self.light.spawn.color = variant.ambient_color
            self.light.spawn.enable_color_temperature = False
            self.light.spawn.color_temperature = float(variant.ambient_temperature)
        else:
            self.light.spawn.intensity = 0.0

        table_center = (0.0, 0.0)
        table_top = 0.75
        if profile.table_name:
            table_xform = profile.xforms[profile.table_name]
            table_center = (table_xform["pos"][0], table_xform["pos"][1])
            resolved_table_top = asset_utils.resolve_tabletop_height(
                profile.table_name,
                position_z=table_xform["pos"][2],
                scale_z=table_xform["scale"][2],
            )
            if resolved_table_top is not None:
                table_top = resolved_table_top

        key_dx, key_dy, key_dz = (0.0, 0.0, 0.0)
        fill_dx, fill_dy, fill_dz = (0.0, 0.0, 0.0)
        lamp_dz = 0.0
        lamp_color = PRACTICAL_LAMP_COLOR
        lamp_temperature = 2800.0
        lamp_scale = 1.0
        if variant is not None:
            key_dx, key_dy, key_dz = variant.key_delta_xyz
            fill_dx, fill_dy, fill_dz = variant.fill_delta_xyz
            lamp_dz = float(variant.lamp_delta_z)
            lamp_color = variant.lamp_color
            lamp_temperature = float(variant.lamp_temperature)
            lamp_scale = float(variant.lamp_scale)

        self.key_light.init_state.pos = [table_center[0] + 1.1 + key_dx, table_center[1] - 1.0 + key_dy, table_top + 1.75 + key_dz]
        self.fill_light.init_state.pos = [table_center[0] - 1.0 + fill_dx, table_center[1] + 0.85 + fill_dy, table_top + 1.35 + fill_dz]

        if profile.lamp_name:
            lamp_xform = profile.xforms[profile.lamp_name]
            lamp_top = asset_utils.resolve_asset_world_top(
                "decoration",
                profile.lamp_name,
                position_z=lamp_xform["pos"][2],
                scale_z=lamp_xform["scale"][2],
            )
            lamp_light_z = (lamp_top - 0.12) if lamp_top is not None else (lamp_xform["pos"][2] + 0.2)
            self.lamp_light = self._build_lamp_light_cfg(
                (lamp_xform["pos"][0], lamp_xform["pos"][1], lamp_light_z + lamp_dz),
                preset,
            )
            self.lamp_light.spawn.intensity = preset.lamp_intensity * lamp_scale
            self.lamp_light.spawn.color = lamp_color
            self.lamp_light.spawn.enable_color_temperature = variant is None
            self.lamp_light.spawn.color_temperature = lamp_temperature
        else:
            self.lamp_light = None

    def apply_scene(self, scene_id: int, lighting_mode: str | None = None, hide_basket: bool = False) -> None:
        self._prune_transient_scene_attrs()
        preset = resolve_lighting_preset(lighting_mode)
        variant_info = resolve_lighting_pool_variant(scene_id=scene_id, lighting_mode=lighting_mode)
        variant = variant_info[1] if variant_info is not None else None
        profile = get_scene_profile(scene_id)
        self._apply_lighting_preset(preset, variant=variant)

        if profile.table_name and profile.table_usd_path:
            self._apply_asset_cfg(self.furniture_table, profile.xforms[profile.table_name], profile.table_usd_path)
            self.furniture_table.spawn.func = spawn_usd_static_mesh_collision
            self.furniture_table.spawn._geomind_physics_material = TABLE_PHYSICS_MATERIAL
        else:
            self.furniture_table = None

        if profile.chair_name and profile.chair_usd_path:
            self._apply_asset_cfg(self.furniture_chair, profile.xforms[profile.chair_name], profile.chair_usd_path)
        else:
            self.furniture_chair = None

        if not hide_basket and profile.basket_name and profile.basket_usd_path:
            self._apply_asset_cfg(
                self.container_basket, profile.xforms[profile.basket_name], profile.basket_usd_path
            )
            self.container_basket.spawn.func = spawn_usd_static_mesh_collision
        else:
            self.container_basket = None

        if profile.lamp_name and profile.lamp_usd_path:
            self._apply_asset_cfg(self.decoration_lamp, profile.xforms[profile.lamp_name], profile.lamp_usd_path)
        else:
            self.decoration_lamp = None

        if profile.blanket_name and profile.blanket_usd_path:
            self._apply_asset_cfg(
                self.decoration_blanket, profile.xforms[profile.blanket_name], profile.blanket_usd_path
            )
        else:
            self.decoration_blanket = None

        if profile.ground_name and profile.ground_usd_path:
            if env_flag("GEOMIND_DISABLE_GROUND_MDL", default=False):
                ground_xform = profile.xforms[profile.ground_name]
                self.ground = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/Ground",
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=[
                            float(ground_xform["pos"][0]),
                            float(ground_xform["pos"][1]),
                            float(ground_xform["pos"][2]) - (GROUND_CUBOID_THICKNESS * 0.5),
                        ],
                        rot=list(ground_xform["rot"]),
                    ),
                    spawn=sim_utils.CuboidCfg(
                        size=(GROUND_PLANE_SIZE[0], GROUND_PLANE_SIZE[1], GROUND_CUBOID_THICKNESS),
                        collision_props=CollisionPropertiesCfg(collision_enabled=True),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GROUND_PLANE_COLOR, roughness=0.9),
                    ),
                )
            else:
                self._apply_asset_cfg(self.ground, profile.xforms[profile.ground_name], profile.ground_usd_path)
        else:
            self.ground = None

        self._configure_scene_lighting(profile, preset, variant=variant)

        for attr_name in [name for name in list(self.__dict__.keys()) if name.startswith("scene_wall_")]:
            delattr(self, attr_name)
        for wall_spec in profile.wall_specs:
            setattr(self, self._wall_attr_name(wall_spec.name), self._build_wall_cfg(wall_spec))
        self._prune_transient_scene_attrs()

    def __post_init__(self):
        super().__post_init__()
        self.apply_scene(TEST_SCENE_ID, lighting_mode=DEFAULT_LIGHTING_MODE)
