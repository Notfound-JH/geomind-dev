"""Deterministic, axis-aligned XY placement without simulator dependencies.

All returned coordinates are object *centres* in the supplied world frame.
Grid rows increase along world X and columns along world Y; list order is row
major. Lines use world Y unless axis='x'. The T has its crossbar at positive X
and its stem towards negative X. Objects retain their source orientation.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from numbers import Real

from .request import RequestError, validate_request


class LayoutError(ValueError):
    """The requested arrangement cannot be placed within the support bounds."""


def _number(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise LayoutError(f"{field} must be a finite number.")
    if value < 0 or (positive and value == 0):
        raise LayoutError(f"{field} must be {'positive' if positive else 'nonnegative'}.")
    return float(value)


def _vector(value: object, length: int, field: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise LayoutError(f"{field} must contain {length} finite numbers.")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(item):
            raise LayoutError(f"{field} must contain {length} finite numbers.")
        result.append(float(item))
    return result


def generate_positions(
    layout: Mapping,
    count: int,
    bounds_xy: tuple[float, float, float, float],
    footprint_xy: tuple[float, float],
    seed: int = 0,
    clearance_m: float = 0.025,
    edge_margin_m: float = 0.03,
) -> list[list[float]]:
    """Generate every requested XY centre or raise :class:`LayoutError`.

    ``bounds_xy=(xmin,xmax,ymin,ymax)`` is a rectangular support region and
    ``footprint_xy=(width_x,width_y)`` is the complete axis-aligned asset bound.
    A requested spacing is a centre-to-centre distance. Defaults use the asset
    footprint plus clearance. A ring uses circumscribed circles to conservatively
    separate rectangular assets. Random rejection sampling enforces rectangular
    separation and any explicit minimum centre distance, with a finite budget;
    sampling failure raises an error even if a denser packing might exist.
    No layout silently drops objects or rotates them to make them fit.
    """
    try:
        checked = validate_request({"scene_id": 1, "object": {"asset": "blue_can", "count": count}, "layout": layout, "seed": seed})
    except RequestError as exc:
        raise LayoutError(str(exc)) from exc
    spec = checked["layout"]
    xmin, xmax, ymin, ymax = _vector(bounds_xy, 4, "bounds_xy")
    sx, sy = _vector(footprint_xy, 2, "footprint_xy")
    if xmin >= xmax or ymin >= ymax:
        raise LayoutError("bounds_xy needs xmin < xmax and ymin < ymax.")
    if sx <= 0 or sy <= 0:
        raise LayoutError("footprint_xy dimensions must be positive.")
    clearance = _number(clearance_m, "clearance_m")
    margin = _number(edge_margin_m, "edge_margin_m")
    low_x, high_x = xmin + margin + sx / 2, xmax - margin - sx / 2
    low_y, high_y = ymin + margin + sy / 2, ymax - margin - sy / 2
    eps = 1e-12
    if low_x > high_x + eps or low_y > high_y + eps:
        raise LayoutError("The asset footprint and edge margin do not fit the support bounds.")
    centre_x, centre_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    half_x, half_y = max(0.0, (high_x - low_x) / 2), max(0.0, (high_y - low_y) / 2)
    kind = spec["type"]
    explicit_spacing = spec.get("spacing_m")

    def spacing(minimum: float) -> float:
        if explicit_spacing is not None and explicit_spacing + eps < minimum:
            raise LayoutError(f"spacing_m={explicit_spacing:g} is too small; footprint plus clearance requires at least {minimum:g} m.")
        return explicit_spacing if explicit_spacing is not None else minimum

    positions: list[list[float]]
    if kind == "grid":
        rows, cols = spec["rows"], spec["cols"]
        minimum = max(sx + clearance if rows > 1 else 0.0, sy + clearance if cols > 1 else 0.0)
        step = spacing(minimum)
        positions = [[centre_x + (row - (rows - 1) / 2) * step, centre_y + (col - (cols - 1) / 2) * step] for row in range(rows) for col in range(cols)]
    elif kind == "line":
        axis = spec["axis"]
        step = spacing((sx if axis == "x" else sy) + clearance if count > 1 else 0.0)
        positions = [[centre_x + (index - (count - 1) / 2) * step if axis == "x" else centre_x, centre_y + (index - (count - 1) / 2) * step if axis == "y" else centre_y] for index in range(count)]
    elif kind == "t_shape":
        step = spacing(max(sx, sy) + clearance)
        positions = [[centre_x + step * x, centre_y + step * y] for x, y in ((1, -1), (1, 0), (1, 1), (0, 0), (-1, 0))]
    elif kind == "ring":
        # Chords between adjacent centres must exceed the full circumscribed
        # footprint diameter. This remains safe for any rectangular aspect ratio.
        step = spacing(math.hypot(sx, sy) + clearance if count > 1 else 0.0)
        if count == 1:
            if "radius_m" in spec:
                raise LayoutError("A one-object ring has no radius; omit radius_m.")
            positions = [[centre_x, centre_y]]
        else:
            minimum_radius = step / (2 * math.sin(math.pi / count))
            radius = spec.get("radius_m", minimum_radius)
            if radius + eps < minimum_radius:
                raise LayoutError(f"radius_m is too small; this ring requires at least {minimum_radius:g} m.")
            if radius > min(half_x, half_y) + eps:
                raise LayoutError("The ring's circular envelope does not fit the support bounds.")
            positions = [[centre_x + radius * math.cos(2 * math.pi * index / count), centre_y + radius * math.sin(2 * math.pi * index / count)] for index in range(count)]
    else:  # random; request validation rejects every other type.
        if explicit_spacing is not None:
            spacing(max(sx, sy) + clearance if count > 1 else 0.0)
        # Inflate each footprint by half the gap on every side. These rectangles
        # must have disjoint interiors, giving a cheap necessary area bound.
        packed_area = count * (sx + clearance) * (sy + clearance)
        available_area = (high_x - low_x + sx + clearance) * (high_y - low_y + sy + clearance)
        if packed_area > available_area + eps:
            raise LayoutError("Requested random objects cannot fit the support area with this footprint and clearance.")
        rng = random.Random(seed)
        positions = []
        for _ in range(max(2000, count * 1000)):
            candidate = [rng.uniform(low_x, high_x), rng.uniform(low_y, high_y)]
            if all(
                (abs(candidate[0] - other[0]) + eps >= sx + clearance or abs(candidate[1] - other[1]) + eps >= sy + clearance)
                and (explicit_spacing is None or math.dist(candidate, other) + eps >= explicit_spacing)
                for other in positions
            ):
                positions.append(candidate)
                if len(positions) == count:
                    break
        if len(positions) != count:
            raise LayoutError(f"Random sampling could place only {len(positions)}/{count} objects; enlarge the bounds, reduce the count, or choose a structured layout.")

    if any(x < low_x - eps or x > high_x + eps or y < low_y - eps or y > high_y + eps for x, y in positions):
        raise LayoutError(f"The {kind} layout does not fit after footprint and edge margins are included.")
    return positions
