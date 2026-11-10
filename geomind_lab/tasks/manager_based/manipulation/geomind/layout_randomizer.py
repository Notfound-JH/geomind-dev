# Copyright (c) 2024-2025, The GeoMind Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structure-preserving randomization of GeoMind runtime object layouts.

Two independent stages, each with its own on/off switch and its own RNG seed:

``jitter``
    Small per-rigid-group XY perturbation.  How the perturbation is applied
    depends on the task family: layouts whose ground truth depends on exact
    geometry (squares, collinear trends, grids, 1-D orderings) are re-built
    from structural parameters instead of being perturbed independently, so
    the task semantics and the protocol-resolved GT stay intact.

``shift``
    A single global translation applied to every randomized object.  The
    translation magnitude is clipped against a dynamic bounding box so that
    the whole scene (including derived GT landing points) stays inside the
    reachable tabletop region.

Both stages are disabled by default so existing benchmark numbers remain
reproducible.

Environment variables
---------------------
``GEOMIND_LAYOUT_JITTER_ENABLE``   ``0``/``1`` (default ``0``)
``GEOMIND_LAYOUT_JITTER_SEED``     integer seed; random when unset
``GEOMIND_LAYOUT_JITTER_XY``       jitter half-range in metres (default ``0.02``)
``GEOMIND_LAYOUT_SHIFT_ENABLE``    ``0``/``1`` (default ``0``)
``GEOMIND_LAYOUT_SHIFT_SEED``      integer seed; random when unset
``GEOMIND_LAYOUT_SHIFT_MAX``       max translation in metres (default ``0.15``)
``GEOMIND_LAYOUT_MIN_CLEARANCE``   min pairwise clearance in metres (default ``0.025``)
``GEOMIND_LAYOUT_BOUNDS``          ``x_lo,x_hi,y_lo,y_hi`` safe region
``GEOMIND_LAYOUT_MAX_ATTEMPTS``    rejection-sampling attempts (default ``256``)
"""

from __future__ import annotations

import math
import os
import random
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from geomind_lab.utils import asset_utils

__all__ = ["LayoutRandomizationConfig", "randomize_runtime_object_layout"]


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------

_ENV_JITTER_ENABLE = "GEOMIND_LAYOUT_JITTER_ENABLE"
_ENV_JITTER_SEED = "GEOMIND_LAYOUT_JITTER_SEED"
_ENV_JITTER_XY = "GEOMIND_LAYOUT_JITTER_XY"
_ENV_SHIFT_ENABLE = "GEOMIND_LAYOUT_SHIFT_ENABLE"
_ENV_SHIFT_SEED = "GEOMIND_LAYOUT_SHIFT_SEED"
_ENV_SHIFT_MAX = "GEOMIND_LAYOUT_SHIFT_MAX"
_ENV_MIN_CLEARANCE = "GEOMIND_LAYOUT_MIN_CLEARANCE"
_ENV_BOUNDS = "GEOMIND_LAYOUT_BOUNDS"
_ENV_MAX_ATTEMPTS = "GEOMIND_LAYOUT_MAX_ATTEMPTS"

_DEFAULT_JITTER_XY = 0.02
_DEFAULT_SHIFT_MAX = 0.15
_DEFAULT_MIN_CLEARANCE = 0.025
# Widest authored layout spans x in [0.158, 0.622] and y in [-0.182, 0.273]
# once footprints are included, so the safe region has to be at least that big.
_DEFAULT_BOUNDS = (0.15, 0.68, -0.32, 0.34)
_DEFAULT_MAX_ATTEMPTS = 256

# Pairs that already sit closer than ``min_clearance`` in the authored layout are
# allowed to stay close, but may not get meaningfully tighter than they were.
_CLEARANCE_SHRINK_RATIO = 0.85

_CONTAINER_PART_ROLES = frozenset({"container_floor", "container_wall"})

# Objects whose *relative* placement carries the stress semantics of a layout.
# They follow the structural frame exactly and never receive extra jitter.
_SEMANTIC_COMPANION_ROLES = frozenset({"obstacle", "clutter"})
_SEMANTIC_COMPANION_TOKENS = ("barrier", "decoy", "lure", "occlusion", "occluder")
_HIGH_DIVERSITY_SPATIAL_VARIANTS = frozenset({"place_directional", "place_inside", "place_on_top"})


@dataclass(frozen=True)
class LayoutRandomizationConfig:
    """Explicit settings for callers that should not mutate process environment."""

    jitter_enabled: bool
    jitter_xy: float
    jitter_seed: int
    shift_enabled: bool
    shift_max: float
    shift_seed: int
    min_clearance: float
    bounds: tuple[float, float, float, float]
    max_attempts: int


# Retain the internal annotation name for existing tools importing it.
_Config = LayoutRandomizationConfig


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw, 0)
    except ValueError:
        return default


def _env_seed(name: str) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return secrets.randbits(64)
    try:
        return int(raw, 0)
    except ValueError:
        return (hash(raw) & 0xFFFFFFFFFFFFFFFF) ^ 0x9E3779B97F4A7C15


def _env_bounds(name: str, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    parts = [chunk.strip() for chunk in raw.replace(";", ",").split(",") if chunk.strip()]
    if len(parts) != 4:
        return default
    try:
        x_lo, x_hi, y_lo, y_hi = (float(value) for value in parts)
    except ValueError:
        return default
    if x_lo >= x_hi or y_lo >= y_hi:
        return default
    return (x_lo, x_hi, y_lo, y_hi)


def _load_config() -> _Config:
    return _Config(
        jitter_enabled=_env_flag(_ENV_JITTER_ENABLE, False),
        jitter_xy=max(0.0, _env_float(_ENV_JITTER_XY, _DEFAULT_JITTER_XY)),
        jitter_seed=_env_seed(_ENV_JITTER_SEED),
        shift_enabled=_env_flag(_ENV_SHIFT_ENABLE, False),
        shift_max=max(0.0, _env_float(_ENV_SHIFT_MAX, _DEFAULT_SHIFT_MAX)),
        shift_seed=_env_seed(_ENV_SHIFT_SEED),
        min_clearance=max(0.0, _env_float(_ENV_MIN_CLEARANCE, _DEFAULT_MIN_CLEARANCE)),
        bounds=_env_bounds(_ENV_BOUNDS, _DEFAULT_BOUNDS),
        max_attempts=max(1, _env_int(_ENV_MAX_ATTEMPTS, _DEFAULT_MAX_ATTEMPTS)),
    )


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------

_Vec = tuple[float, float]


def _sub(a: _Vec, b: _Vec) -> _Vec:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: _Vec, b: _Vec) -> _Vec:
    return (a[0] + b[0], a[1] + b[1])


def _scale(a: _Vec, k: float) -> _Vec:
    return (a[0] * k, a[1] * k)


def _norm(a: _Vec) -> float:
    return math.hypot(a[0], a[1])


def _unit(theta: float) -> _Vec:
    return (math.cos(theta), math.sin(theta))


def _perp(a: _Vec) -> _Vec:
    return (-a[1], a[0])


def _cross(a: _Vec, b: _Vec) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _dot(a: _Vec, b: _Vec) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _frame_coords(point: _Vec, origin: _Vec, e1: _Vec, e2: _Vec, scale: float) -> _Vec:
    """Express ``point`` in the local frame ``(origin; e1, e2)`` normalized by ``scale``."""
    delta = _sub(point, origin)
    inv = 1.0 / scale if scale > 1e-9 else 0.0
    return (_dot(delta, e1) * inv, _dot(delta, e2) * inv)


def _frame_point(coords: _Vec, origin: _Vec, e1: _Vec, e2: _Vec, scale: float) -> _Vec:
    return _add(origin, _add(_scale(e1, coords[0] * scale), _scale(e2, coords[1] * scale)))


# --------------------------------------------------------------------------------------
# runtime object introspection
# --------------------------------------------------------------------------------------


def _object_name(runtime_object: Any) -> str:
    if not isinstance(runtime_object, Mapping):
        return ""
    return str(runtime_object.get("name", "")).strip()


def _object_role(runtime_object: Any) -> str:
    if not isinstance(runtime_object, Mapping):
        return ""
    return str(runtime_object.get("role", "")).strip().lower()


def _object_xy(runtime_object: Any) -> _Vec | None:
    if not isinstance(runtime_object, Mapping):
        return None
    pose = runtime_object.get("pose")
    if not isinstance(pose, Mapping):
        return None
    xy = pose.get("xy")
    if not isinstance(xy, Sequence) or isinstance(xy, (str, bytes)) or len(xy) < 2:
        return None
    try:
        return (float(xy[0]), float(xy[1]))
    except (TypeError, ValueError):
        return None


def _set_object_xy(runtime_object: Any, xy: _Vec) -> None:
    if not isinstance(runtime_object, dict):
        return
    pose = runtime_object.get("pose")
    if not isinstance(pose, dict):
        pose = {}
        runtime_object["pose"] = pose
    pose["xy"] = [float(xy[0]), float(xy[1])]


def _object_footprint_radius(runtime_object: Any) -> float:
    """Half of the larger horizontal extent, used for clearance bookkeeping."""
    if not isinstance(runtime_object, Mapping):
        return 0.05
    kind = str(runtime_object.get("kind", "")).strip()
    try:
        if kind == "primitive_cuboid":
            size = runtime_object.get("size", ())
            if isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) >= 2:
                return 0.5 * max(abs(float(size[0])), abs(float(size[1])))
        elif kind in {"primitive_sphere", "primitive_cylinder", "primitive_cone"}:
            return abs(float(runtime_object.get("radius", 0.0)))
        elif kind == "usd":
            category = str(runtime_object.get("asset_category", "")).strip()
            key = str(runtime_object.get("asset_key", "")).strip()
            info = asset_utils.get_asset_info_by_name(category, key) if category and key else None
            if isinstance(info, Mapping):
                dims = info.get("dimensions")
                if isinstance(dims, Mapping):
                    scale_x, scale_y = _object_scale_xy(runtime_object)
                    return 0.5 * max(
                        abs(float(dims.get("x", 0.0)) * scale_x),
                        abs(float(dims.get("y", 0.0)) * scale_y),
                    )
    except (TypeError, ValueError):
        return 0.05
    return 0.05


def _object_scale_xy(runtime_object: Mapping[str, Any]) -> tuple[float, float]:
    scale = runtime_object.get("scale")
    if isinstance(scale, (int, float)):
        return (float(scale), float(scale))
    if isinstance(scale, Sequence) and not isinstance(scale, (str, bytes)) and len(scale) >= 2:
        try:
            return (abs(float(scale[0])), abs(float(scale[1])))
        except (TypeError, ValueError):
            return (1.0, 1.0)
    return (1.0, 1.0)


def _is_semantic_companion(runtime_object: Any) -> bool:
    if _object_role(runtime_object) in _SEMANTIC_COMPANION_ROLES:
        return True
    name = _object_name(runtime_object).lower()
    return any(token in name for token in _SEMANTIC_COMPANION_TOKENS)


# --------------------------------------------------------------------------------------
# rigid grouping
# --------------------------------------------------------------------------------------


def _build_rigid_groups(runtime_objects: Sequence[Any], gt_spec: Mapping[str, Any]) -> list[list[int]]:
    """Group indices that must move together.

    Multi-part containers (an open box assembled from a floor plus four walls) are
    declared explicitly by ``gt_spec``; the object ``role`` acts as a fallback.
    Everything else forms its own single-member group.
    """
    name_to_index: dict[str, int] = {}
    for index, runtime_object in enumerate(runtime_objects):
        name = _object_name(runtime_object)
        if name and name not in name_to_index:
            name_to_index[name] = index

    part_names: list[str] = []
    for key in ("container_floor_object_id", "container_reference_object_id"):
        value = str(gt_spec.get(key, "")).strip()
        if value:
            part_names.append(value)
    walls = gt_spec.get("container_wall_object_ids")
    if isinstance(walls, Sequence) and not isinstance(walls, (str, bytes)):
        part_names.extend(str(wall).strip() for wall in walls)
    for runtime_object in runtime_objects:
        if _object_role(runtime_object) in _CONTAINER_PART_ROLES:
            part_names.append(_object_name(runtime_object))

    container_indices = sorted({name_to_index[name] for name in part_names if name in name_to_index})

    groups: list[list[int]] = []
    grouped: set[int] = set()
    if len(container_indices) >= 2:
        groups.append(container_indices)
        grouped.update(container_indices)
    for index in range(len(runtime_objects)):
        if index not in grouped:
            groups.append([index])
    return groups


# --------------------------------------------------------------------------------------
# strategy selection
# --------------------------------------------------------------------------------------

_STRATEGY_BY_RESOLVER: dict[str, str] = {
    # 1-D ordering: GT depends only on the ordering along an axis.
    "rank_after_intervention": "rank_line",
    "initial_rank": "rank_line",
    "rank_initial_layout": "rank_line",
    "nearest_planar_left_of_reference": "rank_line",
    # Geometric inference: GT is a closed-form function of the anchors.
    "square_completion": "square",
    "perfect_midpoint": "midpoint",
    "midpoint_placement": "midpoint",
    "point_symmetry": "point_symmetry",
    "isometric_extension": "extension",
    "extension_placement": "extension",
    "pca_trend_continuation": "trend",
    "grid_rowcol_slot": "grid",
    # Selection families have very narrow acceptance bands and an
    # ``expected_target`` hard check: keep their relative geometry frozen.
    "directional_anchor_selection": "frozen",
    "nearest_anchor_selection": "frozen",
    "visual_spatial_constraint_selection": "frozen",
    "exclusion_nearest_constraint_selection": "frozen",
    # Purely relative placement targets: safe to jitter freely.
    "directional_region_placement": "free",
    "containment_placement": "free",
    "support_top_placement": "free",
    "move_to_object_proximity": "free",
}


def _resolve_strategy(gt_spec: Mapping[str, Any], protocol: Mapping[str, Any]) -> str:
    """Select a planner without relaxing non-placement task semantics."""
    if (
        str(protocol.get("family", "")).strip() == "spatial_placement"
        and str(protocol.get("variant", "")).strip() in _HIGH_DIVERSITY_SPATIAL_VARIANTS
    ):
        return "spatial_free"
    key = str(gt_spec.get("resolver", "") or gt_spec.get("rule_type", "")).strip()
    return _STRATEGY_BY_RESOLVER.get(key, "free")


# --------------------------------------------------------------------------------------
# planning context
# --------------------------------------------------------------------------------------


@dataclass
class _Context:
    runtime_objects: Sequence[Any]
    groups: list[list[int]]
    origin: list[_Vec]
    """Authored centroid of every group."""
    radius: list[float]
    """Largest member footprint radius per group."""
    index_of_name: dict[str, int]
    group_of_index: dict[int, int]
    protocol: Mapping[str, Any]
    gt_spec: Mapping[str, Any]
    cfg: _Config
    rng: random.Random
    bounds: tuple[float, float, float, float]
    """Safe region, widened when the authored layout already exceeds ``cfg.bounds``."""
    companion_groups: set[int] = field(default_factory=set)

    def group_of_name(self, name: str) -> int | None:
        index = self.index_of_name.get(name)
        if index is None:
            return None
        return self.group_of_index.get(index)


def _anchor_object_ids(protocol: Mapping[str, Any]) -> list[str]:
    anchors = protocol.get("anchors")
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
        return []
    result: list[str] = []
    for anchor in anchors:
        if isinstance(anchor, Mapping):
            object_id = str(anchor.get("object_id", "")).strip()
            if object_id:
                result.append(object_id)
    return result


def _anchor_by_role(protocol: Mapping[str, Any], role: str) -> str:
    anchors = protocol.get("anchors")
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
        return ""
    for anchor in anchors:
        if isinstance(anchor, Mapping) and str(anchor.get("role", "")).strip() == role:
            return str(anchor.get("object_id", "")).strip()
    return ""


# --------------------------------------------------------------------------------------
# structural jitter strategies
#
# Every strategy returns the new centroid of each group plus any derived
# ground-truth landing point that must also stay inside the safe region.
# Returning ``None`` means the layout does not match the expected structure and
# the caller should fall back to the frozen strategy.
# --------------------------------------------------------------------------------------

_Plan = tuple[list[_Vec], list[_Vec]]


def _companion_transform(
    ctx: _Context,
    centroids: list[_Vec],
    structural_groups: set[int],
    old_frame: tuple[_Vec, _Vec, _Vec, float],
    new_frame: tuple[_Vec, _Vec, _Vec, float],
    *,
    extra_jitter: bool = True,
) -> None:
    """Carry non-structural groups along with the structural frame.

    The companion keeps its authored coordinates in the structural frame, so
    decoys stay just off the solution, barriers stay next to the approach
    corridor and distractors keep their visual arrangement.
    """
    old_origin, old_e1, old_e2, old_scale = old_frame
    new_origin, new_e1, new_e2, new_scale = new_frame
    jitter = ctx.cfg.jitter_xy * 0.5 if extra_jitter else 0.0
    for group_index, group in enumerate(ctx.groups):
        if group_index in structural_groups:
            continue
        coords = _frame_coords(ctx.origin[group_index], old_origin, old_e1, old_e2, old_scale)
        moved = _frame_point(coords, new_origin, new_e1, new_e2, new_scale)
        if jitter > 0.0 and group_index not in ctx.companion_groups:
            moved = (
                moved[0] + ctx.rng.uniform(-jitter, jitter),
                moved[1] + ctx.rng.uniform(-jitter, jitter),
            )
        centroids[group_index] = moved


def _plan_frozen(ctx: _Context) -> _Plan:
    """Keep every relative offset exactly as authored."""
    return list(ctx.origin), []


def _plan_free(ctx: _Context) -> _Plan:
    """Independent per-group jitter; used when GT is purely relative."""
    jitter = ctx.cfg.jitter_xy
    centroids: list[_Vec] = []
    for group_index, origin in enumerate(ctx.origin):
        if jitter <= 0.0 or group_index in ctx.companion_groups:
            centroids.append(origin)
            continue
        centroids.append(
            (
                origin[0] + ctx.rng.uniform(-jitter, jitter),
                origin[1] + ctx.rng.uniform(-jitter, jitter),
            )
        )
    return centroids, []


def _plan_spatial_free(ctx: _Context) -> _Plan | None:
    """Uniformly place each rigid group in the reachable tabletop workspace.

    This planner is intentionally restricted to the simple spatial-placement
    family.  Their ground truth is defined relative to the sampled support,
    container, or anchor, so independent group positions preserve task meaning.
    A group can contain several physical pieces (for example, container walls),
    and is translated as one unit.
    """
    x_lo, x_hi, y_lo, y_hi = ctx.bounds
    centroids: list[_Vec] = []
    for group_index, radius in enumerate(ctx.radius):
        min_x, max_x = x_lo + radius, x_hi - radius
        min_y, max_y = y_lo + radius, y_hi - radius
        if min_x > max_x or min_y > max_y:
            return None
        for _ in range(ctx.cfg.max_attempts):
            candidate = (ctx.rng.uniform(min_x, max_x), ctx.rng.uniform(min_y, max_y))
            if all(
                _norm(_sub(candidate, placed)) >= radius + ctx.radius[placed_index] + ctx.cfg.min_clearance
                for placed_index, placed in enumerate(centroids)
            ):
                centroids.append(candidate)
                break
        else:
            return None
    extra_points: list[_Vec] = []
    gt_resolver = str(ctx.gt_spec.get("resolver", "")).strip()
    if gt_resolver == "move_to_object_proximity":
        anchor_id = str(ctx.gt_spec.get("anchor_object_id", "")).strip()
        anchor_group = ctx.group_of_name(anchor_id)
        target_zone = ctx.gt_spec.get("target_zone", {})
        if anchor_group is None or not isinstance(target_zone, Mapping):
            return None
        offset = target_zone.get("center_offset_from_anchor", ())
        half_extents = target_zone.get("half_extents", ())
        if not (
            isinstance(offset, Sequence)
            and not isinstance(offset, (str, bytes))
            and len(offset) >= 2
            and isinstance(half_extents, Sequence)
            and not isinstance(half_extents, (str, bytes))
            and len(half_extents) >= 2
        ):
            return None
        center = (
            centroids[anchor_group][0] + float(offset[0]),
            centroids[anchor_group][1] + float(offset[1]),
        )
        hx, hy = abs(float(half_extents[0])), abs(float(half_extents[1]))
        extra_points = [
            (center[0] - hx, center[1] - hy),
            (center[0] - hx, center[1] + hy),
            (center[0] + hx, center[1] - hy),
            (center[0] + hx, center[1] + hy),
        ]
    return centroids, extra_points


def _plan_rank_line(ctx: _Context) -> _Plan:
    """Rescale the gaps along the ordering axis, never reordering anything.

    The ground truth of every reindex / identity-tracking task is a *rank* along
    ``order_axis``.  Scaling each consecutive gap by a positive factor keeps the
    ordering — and therefore the ground truth — mathematically invariant, while
    still changing every absolute coordinate.
    """
    axis = str(ctx.gt_spec.get("order_axis", "y")).strip().lower()
    primary = 1 if axis != "x" else 0
    secondary = 1 - primary
    jitter = ctx.cfg.jitter_xy
    if jitter <= 0.0:
        return list(ctx.origin), []

    order = sorted(range(len(ctx.groups)), key=lambda gi: ctx.origin[gi][primary])
    centroids: list[list[float]] = [list(origin) for origin in ctx.origin]

    previous = order[0]
    for current in order[1:]:
        gap = ctx.origin[current][primary] - ctx.origin[previous][primary]
        if gap <= 1e-9:
            # Co-located along the ordering axis (e.g. an off-line distractor):
            # keep the authored offset so the ordering stays well defined.
            centroids[current][primary] = centroids[previous][primary] + gap
        else:
            ratio = ctx.rng.uniform(0.88, 1.18)
            scaled = max(gap * ratio, max(0.03, gap * _CLEARANCE_SHRINK_RATIO))
            centroids[current][primary] = centroids[previous][primary] + scaled
        previous = current

    # The off-axis coordinate does not participate in ranking, but keeping it
    # tame protects the "nearest planar left neighbour" resolvers.
    lateral = jitter * 0.5
    for group_index in range(len(ctx.groups)):
        if group_index in ctx.companion_groups:
            continue
        centroids[group_index][secondary] += ctx.rng.uniform(-lateral, lateral)

    return [(value[0], value[1]) for value in centroids], []


def _plan_midpoint(ctx: _Context) -> _Plan | None:
    """Resample the anchor pair as (midpoint, separation, orientation).

    ``GT = (A + B) / 2`` holds for *any* pair, so the pair may be freely rotated
    and stretched; only the reachability of the midpoint constrains us.
    """
    anchors = _anchor_object_ids(ctx.protocol)
    if len(anchors) != 2:
        return None
    ga, gb = ctx.group_of_name(anchors[0]), ctx.group_of_name(anchors[1])
    if ga is None or gb is None:
        return None

    a, b = ctx.origin[ga], ctx.origin[gb]
    old_mid = _scale(_add(a, b), 0.5)
    old_length = _norm(_sub(b, a))
    if old_length <= 1e-6:
        return None
    old_dir = _scale(_sub(b, a), 1.0 / old_length)

    jitter = ctx.cfg.jitter_xy
    length = old_length * ctx.rng.uniform(0.85, 1.15)
    length = max(length, 0.10)
    theta = math.atan2(old_dir[1], old_dir[0]) + ctx.rng.uniform(-0.31, 0.31)
    new_dir = _unit(theta)
    new_mid = (
        old_mid[0] + ctx.rng.uniform(-jitter, jitter),
        old_mid[1] + ctx.rng.uniform(-jitter, jitter),
    )

    centroids = list(ctx.origin)
    centroids[ga] = _add(new_mid, _scale(new_dir, -0.5 * length))
    centroids[gb] = _add(new_mid, _scale(new_dir, 0.5 * length))
    _companion_transform(
        ctx,
        centroids,
        {ga, gb},
        (old_mid, old_dir, _perp(old_dir), old_length),
        (new_mid, new_dir, _perp(new_dir), length),
    )
    target_object = ctx.protocol.get("target_object", {})
    target_id = str(target_object.get("object_id", "")).strip() if isinstance(target_object, Mapping) else ""
    if not target_id:
        target_id = str(ctx.protocol.get("pre_attached_obj", "")).strip()
    target_group = ctx.group_of_name(target_id) if target_id else None
    if target_group is not None and target_group not in {ga, gb}:
        old_lateral = _dot(_sub(ctx.origin[target_group], old_mid), _perp(old_dir))
        lateral_sign = -1.0 if old_lateral < 0.0 else 1.0
        lateral_offset = lateral_sign * max(abs(old_lateral), 0.12)
        centroids[target_group] = _add(new_mid, _scale(_perp(new_dir), lateral_offset))
    return centroids, [new_mid]


def _plan_point_symmetry(ctx: _Context) -> _Plan | None:
    """Resample (centre, reference offset); ``GT = 2 * centre - reference``."""
    center_id = _anchor_by_role(ctx.protocol, "symmetry_center")
    reference_id = _anchor_by_role(ctx.protocol, "reference")
    anchors = _anchor_object_ids(ctx.protocol)
    if not center_id and len(anchors) == 2:
        center_id, reference_id = anchors[0], anchors[1]
    gc, gr = ctx.group_of_name(center_id), ctx.group_of_name(reference_id)
    if gc is None or gr is None:
        return None

    center, reference = ctx.origin[gc], ctx.origin[gr]
    old_vec = _sub(reference, center)
    old_length = _norm(old_vec)
    if old_length <= 1e-6:
        return None
    old_dir = _scale(old_vec, 1.0 / old_length)

    jitter = ctx.cfg.jitter_xy
    length = max(old_length * ctx.rng.uniform(0.85, 1.12), 0.10)
    theta = math.atan2(old_dir[1], old_dir[0]) + ctx.rng.uniform(-0.35, 0.35)
    new_dir = _unit(theta)
    new_center = (
        center[0] + ctx.rng.uniform(-jitter, jitter),
        center[1] + ctx.rng.uniform(-jitter, jitter),
    )
    new_reference = _add(new_center, _scale(new_dir, length))
    mirror = _add(new_center, _scale(new_dir, -length))

    centroids = list(ctx.origin)
    centroids[gc] = new_center
    centroids[gr] = new_reference
    _companion_transform(
        ctx,
        centroids,
        {gc, gr},
        (center, old_dir, _perp(old_dir), old_length),
        (new_center, new_dir, _perp(new_dir), length),
    )
    return centroids, [mirror]


def _plan_extension(ctx: _Context) -> _Plan | None:
    """Resample (base, step); ``GT = base + 2 * step`` stays collinear."""
    anchors = _anchor_object_ids(ctx.protocol)
    if len(anchors) != 2:
        return None
    ga, gb = ctx.group_of_name(anchors[0]), ctx.group_of_name(anchors[1])
    if ga is None or gb is None:
        return None

    base, second = ctx.origin[ga], ctx.origin[gb]
    old_vec = _sub(second, base)
    old_length = _norm(old_vec)
    if old_length <= 1e-6:
        return None
    old_dir = _scale(old_vec, 1.0 / old_length)

    jitter = ctx.cfg.jitter_xy
    length = max(old_length * ctx.rng.uniform(0.85, 1.12), 0.10)
    theta = math.atan2(old_dir[1], old_dir[0]) + ctx.rng.uniform(-0.26, 0.26)
    new_dir = _unit(theta)
    new_base = (
        base[0] + ctx.rng.uniform(-jitter, jitter),
        base[1] + ctx.rng.uniform(-jitter, jitter),
    )
    new_second = _add(new_base, _scale(new_dir, length))
    landing = _add(new_base, _scale(new_dir, 2.0 * length))

    centroids = list(ctx.origin)
    centroids[ga] = new_base
    centroids[gb] = new_second
    _companion_transform(
        ctx,
        centroids,
        {ga, gb},
        (base, old_dir, _perp(old_dir), old_length),
        (new_base, new_dir, _perp(new_dir), length),
    )
    return centroids, [landing]


def _plan_square(ctx: _Context) -> _Plan | None:
    """Rebuild the open square from (shared corner, side length, orientation).

    Independent jitter would turn the three anchors into a generic
    parallelogram, which contradicts the instruction ("form a square").
    Re-deriving the two adjacent corners keeps the right angle and the equal
    side lengths exact while still randomizing size, orientation and position.
    """
    shared_id = _anchor_by_role(ctx.protocol, "shared_corner")
    adjacent_ids = [
        str(anchor.get("object_id", "")).strip()
        for anchor in ctx.protocol.get("anchors", [])
        if isinstance(anchor, Mapping) and str(anchor.get("role", "")).strip() == "adjacent_corner"
    ]
    anchors = _anchor_object_ids(ctx.protocol)
    if not shared_id and len(anchors) == 3:
        shared_id, adjacent_ids = anchors[0], anchors[1:]
    if not shared_id or len(adjacent_ids) != 2:
        return None

    ga = ctx.group_of_name(shared_id)
    gb = ctx.group_of_name(adjacent_ids[0])
    gc = ctx.group_of_name(adjacent_ids[1])
    if ga is None or gb is None or gc is None:
        return None

    corner = ctx.origin[ga]
    edge_b = _sub(ctx.origin[gb], corner)
    edge_c = _sub(ctx.origin[gc], corner)
    old_side = 0.5 * (_norm(edge_b) + _norm(edge_c))
    if old_side <= 1e-6:
        return None
    old_theta = math.atan2(edge_b[1], edge_b[0])
    handedness = 1.0 if _cross(edge_b, edge_c) >= 0.0 else -1.0

    jitter = ctx.cfg.jitter_xy
    side = max(old_side * ctx.rng.uniform(0.92, 1.08), 0.10)
    theta = old_theta + ctx.rng.uniform(-0.18, 0.18)
    e1 = _unit(theta)
    e2 = _unit(theta + handedness * 0.5 * math.pi)
    new_corner = (
        corner[0] + ctx.rng.uniform(-0.5 * jitter, 0.5 * jitter),
        corner[1] + ctx.rng.uniform(-0.5 * jitter, 0.5 * jitter),
    )

    centroids = list(ctx.origin)
    centroids[ga] = new_corner
    centroids[gb] = _add(new_corner, _scale(e1, side))
    centroids[gc] = _add(new_corner, _scale(e2, side))
    missing_corner = _add(new_corner, _add(_scale(e1, side), _scale(e2, side)))

    old_e1 = _scale(edge_b, 1.0 / max(_norm(edge_b), 1e-9))
    old_e2 = _unit(math.atan2(old_e1[1], old_e1[0]) + handedness * 0.5 * math.pi)
    _companion_transform(
        ctx,
        centroids,
        {ga, gb, gc},
        (corner, old_e1, old_e2, old_side),
        (new_corner, e1, e2, side),
    )
    return centroids, [missing_corner]


def _plan_trend(ctx: _Context) -> _Plan | None:
    """Apply a similarity transform (rotate + uniform scale + translate) to the scene.

    The evaluator fits a PCA axis through the anchors and extrapolates one
    average step.  A similarity transform maps that construction onto itself,
    so the new ground truth is exactly the transformed ground truth.  Crucially
    it also preserves every anchor's *residual* off the fitted axis, which is
    what separates the ``low_noise`` variants from the ``weak_destruction``
    controls -- rebuilding a clean line would silently make the high-noise
    controls easier.
    """
    anchors = _anchor_object_ids(ctx.protocol)
    if len(anchors) < 3:
        return None
    group_indices: list[int] = []
    for name in anchors:
        group = ctx.group_of_name(name)
        if group is None:
            return None
        group_indices.append(group)

    start = ctx.origin[group_indices[0]]
    end = ctx.origin[group_indices[-1]]
    span = _sub(end, start)
    old_length = _norm(span)
    if old_length <= 1e-6:
        return None
    old_dir = _scale(span, 1.0 / old_length)
    steps = len(anchors) - 1

    jitter = ctx.cfg.jitter_xy
    length = max(old_length * ctx.rng.uniform(0.85, 1.18), 0.14)
    theta = math.atan2(old_dir[1], old_dir[0]) + ctx.rng.uniform(-0.26, 0.26)
    new_dir = _unit(theta)
    new_start = (
        start[0] + ctx.rng.uniform(-jitter, jitter),
        start[1] + ctx.rng.uniform(-jitter, jitter),
    )

    old_frame = (start, old_dir, _perp(old_dir), old_length)
    new_frame = (new_start, new_dir, _perp(new_dir), length)

    centroids = list(ctx.origin)
    _companion_transform(ctx, centroids, set(), old_frame, new_frame, extra_jitter=False)

    # Approximate continuation point; only used to keep the landing zone reachable.
    landing_source = _add(end, _scale(span, 1.0 / steps))
    landing = _frame_point(_frame_coords(landing_source, *old_frame), *new_frame)
    return centroids, [landing]


def _plan_grid(ctx: _Context) -> _Plan | None:
    """Rebuild an axis-aligned grid from (origin, row pitch, column pitch).

    The resolver quantizes coordinates with ``grid_quant_step`` to recover the
    row/column of each cell, so the grid must stay axis aligned and the pitches
    must stay comfortably above the quantization step.
    """
    row_axis = 0 if str(ctx.gt_spec.get("row_axis", "x")).strip().lower() == "x" else 1
    col_axis = 1 - row_axis
    try:
        num_rows = int(ctx.gt_spec.get("grid_num_rows", 0))
        num_cols = int(ctx.gt_spec.get("grid_num_cols", 0))
    except (TypeError, ValueError):
        return None
    if num_rows < 1 or num_cols < 1:
        return None

    cells = [gi for gi, group in enumerate(ctx.groups) if len(group) == 1]
    if len(cells) != num_rows * num_cols:
        return None

    row_labels, row_centers = _cluster_axis([ctx.origin[gi][row_axis] for gi in cells], num_rows)
    col_labels, col_centers = _cluster_axis([ctx.origin[gi][col_axis] for gi in cells], num_cols)
    if row_labels is None or col_labels is None:
        return None

    quant_step = abs(float(ctx.gt_spec.get("grid_quant_step", 0.0) or 0.0))
    min_pitch = max(4.0 * quant_step, 0.08)
    old_row_pitch = _mean_pitch(row_centers)
    old_col_pitch = _mean_pitch(col_centers)

    jitter = ctx.cfg.jitter_xy
    row_pitch = max(old_row_pitch * ctx.rng.uniform(0.85, 1.18), min_pitch) if num_rows > 1 else 0.0
    col_pitch = max(old_col_pitch * ctx.rng.uniform(0.85, 1.18), min_pitch) if num_cols > 1 else 0.0
    origin_row = row_centers[0] + ctx.rng.uniform(-jitter, jitter)
    origin_col = col_centers[0] + ctx.rng.uniform(-jitter, jitter)

    # A per-cell wobble stays far below the quantization step so the recovered
    # row/column indices cannot change.
    wobble = min(jitter, quant_step * 0.25) if quant_step > 0.0 else jitter * 0.25

    centroids = list(ctx.origin)
    for position, group_index in enumerate(cells):
        point = [0.0, 0.0]
        point[row_axis] = origin_row + row_labels[position] * row_pitch + ctx.rng.uniform(-wobble, wobble)
        point[col_axis] = origin_col + col_labels[position] * col_pitch + ctx.rng.uniform(-wobble, wobble)
        centroids[group_index] = (point[0], point[1])
    return centroids, []


def _mean_pitch(centers: Sequence[float]) -> float:
    if len(centers) < 2:
        return 0.0
    return (centers[-1] - centers[0]) / (len(centers) - 1)


def _cluster_axis(values: Sequence[float], count: int) -> tuple[list[int] | None, list[float]]:
    """Split ``values`` into ``count`` ordered clusters at the widest gaps."""
    if count <= 0 or len(values) < count:
        return None, []
    order = sorted(range(len(values)), key=lambda i: values[i])
    if count == 1:
        return [0] * len(values), [sum(values) / len(values)]

    gaps = sorted(
        ((values[order[k + 1]] - values[order[k]], k) for k in range(len(order) - 1)),
        key=lambda item: item[0],
        reverse=True,
    )
    cuts = sorted(index for _, index in gaps[: count - 1])

    labels = [0] * len(values)
    centers: list[float] = []
    start = 0
    for label, cut in enumerate([*cuts, len(order) - 1]):
        members = order[start : cut + 1]
        if not members:
            return None, []
        for member in members:
            labels[member] = label
        centers.append(sum(values[member] for member in members) / len(members))
        start = cut + 1
    return labels, centers


_PLANNERS = {
    "frozen": _plan_frozen,
    "free": _plan_free,
    "spatial_free": _plan_spatial_free,
    "rank_line": _plan_rank_line,
    "midpoint": _plan_midpoint,
    "point_symmetry": _plan_point_symmetry,
    "extension": _plan_extension,
    "square": _plan_square,
    "trend": _plan_trend,
    "grid": _plan_grid,
}


# --------------------------------------------------------------------------------------
# global translation
# --------------------------------------------------------------------------------------


def _sample_global_shift(
    centroids: Sequence[_Vec],
    radii: Sequence[float],
    extra_points: Sequence[_Vec],
    bounds: tuple[float, float, float, float],
    cfg: _Config,
    rng: random.Random,
) -> _Vec:
    """Sample a translation, clipped so the whole scene stays inside ``bounds``.

    Derived ground-truth landing points participate in the bounding box, which
    is what keeps mirrored / extrapolated targets reachable.
    """
    xs_lo = [centroid[0] - radius for centroid, radius in zip(centroids, radii)]
    xs_hi = [centroid[0] + radius for centroid, radius in zip(centroids, radii)]
    ys_lo = [centroid[1] - radius for centroid, radius in zip(centroids, radii)]
    ys_hi = [centroid[1] + radius for centroid, radius in zip(centroids, radii)]
    for point in extra_points:
        xs_lo.append(point[0])
        xs_hi.append(point[0])
        ys_lo.append(point[1])
        ys_hi.append(point[1])
    if not xs_lo:
        return (0.0, 0.0)

    x_lo, x_hi, y_lo, y_hi = bounds
    tx_lo, tx_hi = x_lo - min(xs_lo), x_hi - max(xs_hi)
    ty_lo, ty_hi = y_lo - min(ys_lo), y_hi - max(ys_hi)

    theta = rng.uniform(0.0, 2.0 * math.pi)
    radius = cfg.shift_max * math.sqrt(rng.uniform(0.0, 1.0))
    return (
        _clamp(radius * math.cos(theta), tx_lo, tx_hi),
        _clamp(radius * math.sin(theta), ty_lo, ty_hi),
    )


def _clamp(value: float, low: float, high: float) -> float:
    if low > high:
        # The scene is already wider than the safe region on this axis;
        # translating it further can only make things worse.
        return 0.0
    return max(low, min(high, value))


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------


def _required_clearance(original_distance: float, min_clearance: float) -> float:
    """Clearance floor for a pair of objects.

    Some authored layouts intentionally place objects closer than
    ``min_clearance`` (a cube staged right next to its target box, the direct
    left neighbour of an anchor).  Those pairs are only required not to get
    meaningfully tighter than the annotation made them.
    """
    if original_distance >= min_clearance:
        return max(min_clearance, original_distance * _CLEARANCE_SHRINK_RATIO)
    return original_distance * _CLEARANCE_SHRINK_RATIO


def _validate_clearance(ctx: _Context, centroids: Sequence[_Vec]) -> bool:
    """Pairwise clearance between rigid groups.

    Translation-invariant, so this only depends on the jitter stage.
    """
    for gi in range(len(ctx.groups)):
        offset_i = _sub(centroids[gi], ctx.origin[gi])
        for gj in range(gi + 1, len(ctx.groups)):
            offset_j = _sub(centroids[gj], ctx.origin[gj])
            for a in ctx.groups[gi]:
                pos_a = _member_xy(ctx, a)
                for b in ctx.groups[gj]:
                    pos_b = _member_xy(ctx, b)
                    original = _norm(_sub(pos_a, pos_b))
                    updated = _norm(_sub(_add(pos_a, offset_i), _add(pos_b, offset_j)))
                    if updated + 1e-9 < _required_clearance(original, ctx.cfg.min_clearance):
                        return False
    return True


def _layout_extent(
    ctx: _Context,
    centroids: Sequence[_Vec],
    extra_points: Sequence[_Vec],
) -> tuple[float, float, float, float]:
    xs_lo = [centroid[0] - radius for centroid, radius in zip(centroids, ctx.radius)]
    xs_hi = [centroid[0] + radius for centroid, radius in zip(centroids, ctx.radius)]
    ys_lo = [centroid[1] - radius for centroid, radius in zip(centroids, ctx.radius)]
    ys_hi = [centroid[1] + radius for centroid, radius in zip(centroids, ctx.radius)]
    for point in extra_points:
        xs_lo.append(point[0])
        xs_hi.append(point[0])
        ys_lo.append(point[1])
        ys_hi.append(point[1])
    return (min(xs_lo), max(xs_hi), min(ys_lo), max(ys_hi))


def _validate_bounds(ctx: _Context, centroids: Sequence[_Vec], extra_points: Sequence[_Vec]) -> bool:
    """The layout sits inside the safe region at its absolute position."""
    x_lo, x_hi, y_lo, y_hi = ctx.bounds
    ex_lo, ex_hi, ey_lo, ey_hi = _layout_extent(ctx, centroids, extra_points)
    return x_lo <= ex_lo + 1e-9 and ex_hi <= x_hi + 1e-9 and y_lo <= ey_lo + 1e-9 and ey_hi <= y_hi + 1e-9


def _validate_extent(ctx: _Context, centroids: Sequence[_Vec], extra_points: Sequence[_Vec]) -> bool:
    """The layout is small enough that *some* translation keeps it in bounds.

    Checking the extent rather than the absolute position keeps the jitter
    stage independent of the shift stage.
    """
    x_lo, x_hi, y_lo, y_hi = ctx.bounds
    ex_lo, ex_hi, ey_lo, ey_hi = _layout_extent(ctx, centroids, extra_points)
    return (ex_hi - ex_lo) <= (x_hi - x_lo) + 1e-9 and (ey_hi - ey_lo) <= (y_hi - y_lo) + 1e-9


def _member_xy(ctx: _Context, member: int) -> _Vec:
    xy = _object_xy(ctx.runtime_objects[member])
    return xy if xy is not None else (0.0, 0.0)


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------


def randomize_runtime_object_layout(
    runtime_objects: list[Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    layout_key: str = "",
    config: LayoutRandomizationConfig | None = None,
    verbose: bool | None = None,
) -> dict[str, Any] | None:
    """Randomize the XY poses of ``runtime_objects`` in place.

    Returns a record describing what was applied, or ``None`` when both stages
    are disabled or no valid sample could be found (in which case the authored
    layout is left untouched). Explicit ``config`` bypasses environment settings;
    existing callers retain the environment-controlled behavior.
    """
    cfg = _load_config() if config is None else config
    if not isinstance(cfg, LayoutRandomizationConfig):
        raise TypeError("config must be a LayoutRandomizationConfig")
    if not cfg.jitter_enabled and not cfg.shift_enabled:
        return None
    if not isinstance(runtime_objects, list) or not runtime_objects:
        return None

    protocol = protocol if isinstance(protocol, Mapping) else {}
    gt_spec = protocol.get("gt_spec")
    gt_spec = gt_spec if isinstance(gt_spec, Mapping) else {}

    placed = [index for index, item in enumerate(runtime_objects) if _object_xy(item) is not None]
    if not placed:
        return None

    groups = [group for group in _build_rigid_groups(runtime_objects, gt_spec) if all(i in placed for i in group)]
    if not groups:
        return None

    index_of_name: dict[str, int] = {}
    for index in placed:
        name = _object_name(runtime_objects[index])
        if name and name not in index_of_name:
            index_of_name[name] = index
    group_of_index = {member: gi for gi, group in enumerate(groups) for member in group}

    origins: list[_Vec] = []
    radii: list[float] = []
    companion_groups: set[int] = set()
    for group_index, group in enumerate(groups):
        points = [_object_xy(runtime_objects[member]) or (0.0, 0.0) for member in group]
        origins.append(
            (
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            )
        )
        radius = 0.0
        for member, point in zip(group, points):
            reach = _norm(_sub(point, origins[group_index])) + _object_footprint_radius(runtime_objects[member])
            radius = max(radius, reach)
        radii.append(radius)
        if any(_is_semantic_companion(runtime_objects[member]) for member in group):
            companion_groups.add(group_index)

    strategy = _resolve_strategy(gt_spec, protocol)
    jitter_rng = random.Random(cfg.jitter_seed)
    shift_rng = random.Random(cfg.shift_seed)

    # A handful of authored layouts already span slightly more than the nominal
    # safe region.  Widening the region to contain them keeps those layouts
    # randomizable instead of dead-locking the rejection sampler, while layouts
    # that fit stay strictly bounded by the configured region.
    bounds = (
        min(cfg.bounds[0], min(o[0] - r for o, r in zip(origins, radii))),
        max(cfg.bounds[1], max(o[0] + r for o, r in zip(origins, radii))),
        min(cfg.bounds[2], min(o[1] - r for o, r in zip(origins, radii))),
        max(cfg.bounds[3], max(o[1] + r for o, r in zip(origins, radii))),
    )
    # Unlike legacy layouts, these three simple placement tasks should never
    # expand past the explicit reachable tabletop workspace.
    if strategy == "spatial_free":
        bounds = cfg.bounds

    ctx = _Context(
        runtime_objects=runtime_objects,
        groups=groups,
        origin=origins,
        radius=radii,
        index_of_name=index_of_name,
        group_of_index=group_of_index,
        protocol=protocol,
        gt_spec=gt_spec,
        cfg=cfg,
        rng=jitter_rng,
        bounds=bounds,
        companion_groups=companion_groups,
    )

    planner = _PLANNERS.get(strategy, _plan_free)
    applied_strategy = "shift_only"
    centroids: list[_Vec] = list(origins)
    extra_points: list[_Vec] = []

    # Stage 1 -- jitter.  Only ``jitter_rng`` is consumed here, and acceptance
    # only depends on translation-invariant properties, so the result is
    # completely independent of the shift stage and its seed.
    if cfg.jitter_enabled:
        applied_strategy = strategy
        for _ in range(cfg.max_attempts):
            plan = planner(ctx)
            if plan is None:
                # Structure did not match the annotation: degrade to a pure
                # translation rather than risking the ground truth.
                planner = _plan_frozen
                applied_strategy = "frozen"
                plan = _plan_frozen(ctx)
            candidate, points = plan
            if strategy != "spatial_free" and not _validate_clearance(ctx, candidate):
                continue
            fits = (
                _validate_extent(ctx, candidate, points)
                if cfg.shift_enabled
                else _validate_bounds(ctx, candidate, points)
            )
            if not fits:
                continue
            centroids, extra_points = candidate, points
            break
        else:
            centroids, extra_points = list(origins), []
            applied_strategy = "frozen"

    # Stage 2 -- global translation.  The bounding box is eroded against the
    # safe region before sampling, so the draw always lands on a valid pose and
    # never has to be rejected; only ``shift_rng`` is consumed.
    shift: _Vec = (0.0, 0.0)
    if cfg.shift_enabled:
        shift = _sample_global_shift(centroids, radii, extra_points, bounds, cfg, shift_rng)
        centroids = [_add(centroid, shift) for centroid in centroids]

    for group_index, group in enumerate(groups):
        offset = _sub(centroids[group_index], origins[group_index])
        for member in group:
            _set_object_xy(runtime_objects[member], _add(_member_xy(ctx, member), offset))

    record = {
        "layout_key": str(layout_key),
        "strategy": applied_strategy,
        "resolver": str(gt_spec.get("resolver", "") or gt_spec.get("rule_type", "")),
        "jitter_enabled": cfg.jitter_enabled,
        "jitter_seed": cfg.jitter_seed,
        "jitter_xy": cfg.jitter_xy,
        "shift_enabled": cfg.shift_enabled,
        "shift_seed": cfg.shift_seed,
        "shift_max": cfg.shift_max,
        "global_shift": [shift[0], shift[1]],
        "min_clearance": cfg.min_clearance,
        "bounds": list(bounds),
        "rigid_groups": [[_object_name(runtime_objects[member]) for member in group] for group in groups],
    }
    if _env_flag("GEOMIND_LAYOUT_VERBOSE", True) if verbose is None else verbose:
        parts = []
        for obj in runtime_objects:
            xy = _object_xy(obj)
            if xy is None:
                continue
            parts.append(f"{_object_name(obj)}({xy[0]:+.3f},{xy[1]:+.3f})")
        print(
            f"[layout-randomizer] {layout_key} strategy={applied_strategy} "
            f"jitter={'on' if cfg.jitter_enabled else 'off'}(seed={cfg.jitter_seed}) "
            f"shift={'on' if cfg.shift_enabled else 'off'}(seed={cfg.shift_seed}) "
            f"global_shift=({shift[0]:+.4f},{shift[1]:+.4f})",
            flush=True,
        )
        print(f"[layout-randomizer] xy: {', '.join(parts)}", flush=True)
    return record
