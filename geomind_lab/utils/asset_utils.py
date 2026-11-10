from __future__ import annotations
from .portable_paths import geomind_assets_root, repo_assets_root, resolve_asset_paths

import json
import os
import re
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_assets_root() -> Path:
    return geomind_assets_root()


def resolve_assets_catalog_path() -> Path:
    env_override = os.environ.get("GEOMIND_LAB_ASSET_CATALOG_PATH")
    if env_override:
        return Path(env_override)

    candidates = (
        repo_assets_root() / "assets_catalog_with_metadata.json",
        repo_assets_root() / "scene_profiles" / "assets_catalog_with_metadata.json",
        repo_root() / "assets_catalog_with_metadata.json",
        Path("/home/elm/projects/geomind/assets_catalog_with_metadata.json"),
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def load_annotated_asset_table(asset_table_path: str | Path) -> dict[str, Any]:
    path = Path(asset_table_path)
    if not path.is_file():
        return {}
    return resolve_asset_paths(json.loads(path.read_text(encoding="utf-8")))


GEOMIND_ASSETS_ROOT = str(resolve_assets_root())
ASSET_CATALOG_PATH = resolve_assets_catalog_path()

_asset_table_payload = load_annotated_asset_table(ASSET_CATALOG_PATH)
assets_table: dict[str, list[dict[str, Any]]] = dict(_asset_table_payload.get("assets", {}))
metadata_table: dict[str, Any] = dict(_asset_table_payload.get("metadata", {}))

tables_asset_list = assets_table.get("table", [])
container_asset_list = assets_table.get("container", [])
decoration_asset_list = assets_table.get("decoration", [])
furniture_asset_list = assets_table.get("furniture", [])
objects_asset_list = assets_table.get("catch_object", [])
table_heights = metadata_table.get("table_heights", {})


def get_asset_info_by_name(category: str, asset_name: str) -> dict[str, Any] | None:
    asset_list = assets_table.get(category, [])
    for asset_info in asset_list:
        if asset_info.get("name") == asset_name:
            return asset_info
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_table_height_key(table_name: str) -> str | None:
    match = re.search(r"_(\d+)$", table_name)
    if not match:
        return None
    return f"TABLE_{match.group(1)}"


def get_table_height_by_name(table_name: str) -> float | None:
    key = get_table_height_key(table_name)
    if key is None:
        return None
    return _as_float(table_heights.get(key))


def get_asset_vertical_extent(asset_info: dict[str, Any]) -> tuple[float, float] | None:
    z_min = _as_float(asset_info.get("z_min_offset"))
    dims = asset_info.get("dimensions", {}) if isinstance(asset_info, dict) else {}
    dim_z = _as_float(dims.get("z")) if isinstance(dims, dict) else None
    if z_min is None or dim_z is None:
        return None
    return (z_min, z_min + dim_z)


def compute_world_vertical_extent(
    asset_info: dict[str, Any],
    position_z: float,
    scale_z: float = 1.0,
) -> tuple[float, float] | None:
    local_extent = get_asset_vertical_extent(asset_info)
    if local_extent is None:
        return None
    world_extent = [float(position_z) + bound * float(scale_z) for bound in local_extent]
    return (min(world_extent), max(world_extent))


def resolve_asset_world_top(
    category: str,
    asset_name: str,
    *,
    position_z: float,
    scale_z: float | None = None,
) -> float | None:
    asset_info = get_asset_info_by_name(category, asset_name)
    if asset_info is None:
        return None
    world_extent = compute_world_vertical_extent(asset_info, position_z, 1.0 if scale_z is None else scale_z)
    if world_extent is None:
        return None
    return world_extent[1]


def resolve_tabletop_height(
    table_name: str,
    *,
    position_z: float | None = None,
    scale_z: float | None = None,
) -> float | None:
    if position_z is not None:
        world_top = resolve_asset_world_top("table", table_name, position_z=position_z, scale_z=scale_z)
        if world_top is not None:
            return world_top
    return get_table_height_by_name(table_name)


def get_ground_plane_info(name: str) -> str:
    """Resolve ground asset path by extension preference."""
    ground_dir = Path(GEOMIND_ASSETS_ROOT) / "ground"
    usda_path = ground_dir / f"{name}.usda"
    usd_path = ground_dir / f"{name}.usd"

    if usda_path.is_file():
        return str(usda_path)
    if usd_path.is_file():
        return str(usd_path)
    return str(usda_path)
