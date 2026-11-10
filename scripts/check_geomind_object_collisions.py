#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.utils import asset_utils, scene_utils


SCENE_CATEGORIES = ("function", "object", "container", "decoration")
DEFAULT_LOG_DIR = REPO_ROOT / "logs"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline GeoMind object overlap checker. This script reads scene USDA files, "
            "builds world-space object AABBs, checks pairwise overlap, prints per-scene "
            "results, and writes a JSON log."
        )
    )
    parser.add_argument(
        "--scene-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional scene ids to check. If omitted, all scenes under GEOMIND_ASSETS_ROOT/stage are checked.",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path(asset_utils.GEOMIND_ASSETS_ROOT) / "stage",
        help="Scene root directory. Defaults to GEOMIND_ASSETS_ROOT/stage.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory used to save the collision report JSON log.",
    )
    parser.add_argument(
        "--collision-eps",
        type=float,
        default=1e-6,
        help="Minimum overlap thickness required on each axis to count as a collision.",
    )
    return parser


def _coerce_tuple(value: Any, length: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)) and len(value) >= length:
        return tuple(float(item) for item in value[:length])
    return default


def _node_to_xform(node: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    attrs = node.get("attrs", {}) if isinstance(node, dict) else {}
    return {
        "pos": _coerce_tuple(attrs.get("translate"), 3, (0.0, 0.0, 0.0)),
        "rot": _coerce_tuple(attrs.get("orient"), 4, (0.0, 0.0, 0.0, 1.0)),
        "scale": _coerce_tuple(attrs.get("scale"), 3, (1.0, 1.0, 1.0)),
    }


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


def _object_prefix(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def _resolve_scene_usd_path(scene_root: Path, scene_id: int) -> Path:
    new_path = scene_root / str(scene_id) / f"{scene_id}_new.usda"
    if new_path.is_file():
        return new_path
    old_path = scene_root / str(scene_id) / f"{scene_id}.usda"
    if old_path.is_file():
        return old_path
    raise FileNotFoundError(f"Scene USD file not found for scene_id={scene_id}: {new_path} or {old_path}")


def _iter_scene_ids(scene_root: Path) -> list[int]:
    scene_ids: list[int] = []
    for path in sorted(scene_root.iterdir()):
        if path.is_dir() and path.name.isdigit():
            scene_ids.append(int(path.name))
    return scene_ids


def _quat_xyzw_to_rot_matrix(quat: tuple[float, float, float, float]) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

    x /= norm
    y /= norm
    z /= norm
    w /= norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _rotate_vector(
    rot: tuple[tuple[float, float, float], ...],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        rot[0][0] * vec[0] + rot[0][1] * vec[1] + rot[0][2] * vec[2],
        rot[1][0] * vec[0] + rot[1][1] * vec[1] + rot[1][2] * vec[2],
        rot[2][0] * vec[0] + rot[2][1] * vec[1] + rot[2][2] * vec[2],
    )


def _compute_world_aabb(
    *,
    position: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
    scale: tuple[float, float, float],
    dimensions: dict[str, Any],
    z_min_offset: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    dim_x = float(dimensions["x"])
    dim_y = float(dimensions["y"])
    dim_z = float(dimensions["z"])
    local_corners = (
        (-0.5 * dim_x, -0.5 * dim_y, z_min_offset),
        (-0.5 * dim_x, -0.5 * dim_y, z_min_offset + dim_z),
        (-0.5 * dim_x, 0.5 * dim_y, z_min_offset),
        (-0.5 * dim_x, 0.5 * dim_y, z_min_offset + dim_z),
        (0.5 * dim_x, -0.5 * dim_y, z_min_offset),
        (0.5 * dim_x, -0.5 * dim_y, z_min_offset + dim_z),
        (0.5 * dim_x, 0.5 * dim_y, z_min_offset),
        (0.5 * dim_x, 0.5 * dim_y, z_min_offset + dim_z),
    )
    rot = _quat_xyzw_to_rot_matrix(rotation_xyzw)
    world_corners: list[tuple[float, float, float]] = []

    for corner in local_corners:
        scaled_corner = (
            corner[0] * float(scale[0]),
            corner[1] * float(scale[1]),
            corner[2] * float(scale[2]),
        )
        rotated_corner = _rotate_vector(rot, scaled_corner)
        world_corners.append(
            (
                float(position[0]) + rotated_corner[0],
                float(position[1]) + rotated_corner[1],
                float(position[2]) + rotated_corner[2],
            )
        )

    mins = tuple(min(point[axis] for point in world_corners) for axis in range(3))
    maxs = tuple(max(point[axis] for point in world_corners) for axis in range(3))
    return mins, maxs


def _overlap_interval(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return min(a_max, b_max) - max(a_min, b_min)


def _check_aabb_overlap(
    a_min: tuple[float, float, float],
    a_max: tuple[float, float, float],
    b_min: tuple[float, float, float],
    b_max: tuple[float, float, float],
    *,
    collision_eps: float,
) -> dict[str, Any] | None:
    overlap_xyz = (
        _overlap_interval(a_min[0], a_max[0], b_min[0], b_max[0]),
        _overlap_interval(a_min[1], a_max[1], b_min[1], b_max[1]),
        _overlap_interval(a_min[2], a_max[2], b_min[2], b_max[2]),
    )
    if any(axis_overlap <= float(collision_eps) for axis_overlap in overlap_xyz):
        return None
    return {
        "overlap_xyz": [float(value) for value in overlap_xyz],
        "intersection_volume": float(overlap_xyz[0] * overlap_xyz[1] * overlap_xyz[2]),
    }


def _build_object_record(name: str, xform: dict[str, tuple[float, ...]]) -> dict[str, Any]:
    asset_name = _object_prefix(name)
    asset_info = asset_utils.get_asset_info_by_name("catch_object", asset_name)
    if asset_info is None:
        raise KeyError(f"Object asset metadata not found for '{asset_name}'")

    aabb_min, aabb_max = _compute_world_aabb(
        position=xform["pos"],
        rotation_xyzw=xform["rot"],
        scale=xform["scale"],
        dimensions=asset_info["dimensions"],
        z_min_offset=float(asset_info["z_min_offset"]),
    )

    return {
        "name": name,
        "asset_name": asset_name,
        "usd_path": str(asset_info.get("usd_path", "")),
        "position": [float(value) for value in xform["pos"]],
        "rotation_xyzw": [float(value) for value in xform["rot"]],
        "scale": [float(value) for value in xform["scale"]],
        "dimensions_raw": {
            "x": float(asset_info["dimensions"]["x"]),
            "y": float(asset_info["dimensions"]["y"]),
            "z": float(asset_info["dimensions"]["z"]),
        },
        "z_min_offset_raw": float(asset_info["z_min_offset"]),
        "aabb_min": [float(value) for value in aabb_min],
        "aabb_max": [float(value) for value in aabb_max],
    }


def inspect_scene(scene_id: int, scene_root: Path, *, collision_eps: float) -> dict[str, Any]:
    scene_path = _resolve_scene_usd_path(scene_root, scene_id)
    nodes = scene_utils.parse_usda_scene(scene_path)
    structure = _load_scene_structure(nodes)

    object_records: list[dict[str, Any]] = []
    skipped_objects: list[dict[str, Any]] = []
    for name, node in sorted(structure.get("object", {}).items()):
        xform = _node_to_xform(node)
        try:
            object_records.append(_build_object_record(name, xform))
        except Exception as exc:
            skipped_objects.append({"name": name, "reason": str(exc)})

    collision_pairs: list[dict[str, Any]] = []
    for object_a, object_b in combinations(object_records, 2):
        overlap = _check_aabb_overlap(
            tuple(object_a["aabb_min"]),
            tuple(object_a["aabb_max"]),
            tuple(object_b["aabb_min"]),
            tuple(object_b["aabb_max"]),
            collision_eps=collision_eps,
        )
        if overlap is None:
            continue
        collision_pairs.append(
            {
                "object_a": object_a["name"],
                "object_b": object_b["name"],
                **overlap,
            }
        )

    return {
        "scene_id": int(scene_id),
        "scene_usd_path": str(scene_path),
        "num_objects": len(object_records),
        "num_skipped_objects": len(skipped_objects),
        "has_collisions": bool(collision_pairs),
        "collision_pairs": collision_pairs,
        "objects": object_records,
        "skipped_objects": skipped_objects,
    }


def _format_scene_summary(scene_result: dict[str, Any]) -> str:
    prefix = f"scene {scene_result['scene_id']}:"
    if scene_result.get("error"):
        return f"{prefix} failed to inspect"
    if not scene_result["has_collisions"]:
        return (
            f"{prefix} no object collisions "
            f"(objects={scene_result['num_objects']}, skipped={scene_result['num_skipped_objects']})"
        )

    pair_text = ", ".join(
        f"{pair['object_a']} <-> {pair['object_b']}" for pair in scene_result["collision_pairs"]
    )
    return (
        f"{prefix} object collisions found "
        f"(objects={scene_result['num_objects']}, skipped={scene_result['num_skipped_objects']}): {pair_text}"
    )


def main() -> int:
    args = build_argparser().parse_args()
    scene_root = args.scene_root.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()

    if not scene_root.is_dir():
        raise FileNotFoundError(f"Scene root does not exist: {scene_root}")

    scene_ids = sorted(set(args.scene_ids)) if args.scene_ids else _iter_scene_ids(scene_root)
    if not scene_ids:
        raise RuntimeError(f"No scene ids found under: {scene_root}")

    results: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        try:
            scene_result = inspect_scene(scene_id, scene_root, collision_eps=args.collision_eps)
        except Exception as exc:
            scene_result = {
                "scene_id": int(scene_id),
                "scene_usd_path": "",
                "num_objects": 0,
                "num_skipped_objects": 0,
                "has_collisions": False,
                "collision_pairs": [],
                "objects": [],
                "skipped_objects": [],
                "error": str(exc),
            }
        results.append(scene_result)
        print(_format_scene_summary(scene_result), flush=True)
        if scene_result.get("error"):
            print(f"  error: {scene_result['error']}", flush=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"geomind_object_collision_report_{timestamp}.json"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scene_root": str(scene_root),
        "collision_method": "world_aabb_overlap",
        "quaternion_order_assumption": "xyzw",
        "collision_eps": float(args.collision_eps),
        "num_scenes_checked": len(results),
        "num_scenes_with_collisions": sum(1 for result in results if result["has_collisions"]),
        "num_scene_errors": sum(1 for result in results if result.get("error")),
        "results": results,
    }
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print("")
    print(f"log saved to: {log_path}")
    print(
        "summary: "
        f"checked={payload['num_scenes_checked']}, "
        f"colliding_scenes={payload['num_scenes_with_collisions']}, "
        f"scene_errors={payload['num_scene_errors']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
