"""Adapt the existing runtime layout randomizer to measured assembly geometry.

The grid uses the existing semantic grid planner, including its independently
seeded jitter and shift stages. Other layouts currently receive a rigid global
translation through that same runtime entry point: their authored line, ring or
T geometry is preserved instead of being passed to a free-object jitter planner.
This module changes XY centres only. The exporter separately applies yaw and
checks the resulting mesh bounds, pair gaps, furniture and tabletop support.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

from geomind_lab.tasks.manager_based.manipulation.geomind.layout_randomizer import (
    LayoutRandomizationConfig,
    randomize_runtime_object_layout,
)

from .layouts import LayoutError


def _finite_sequence(value, count, label):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != count:
        raise LayoutError(f"{label} must contain {count} finite numbers")
    if any(isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(item) for item in value):
        raise LayoutError(f"{label} must contain {count} finite numbers")
    return [float(item) for item in value]


def _nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise LayoutError(f"{label} must be a finite nonnegative number")
    return float(value)


def randomize_positions(
    positions: Sequence[Sequence[float]],
    *,
    layout: Mapping,
    footprint_xy: tuple[float, float],
    bounds_xy: tuple[float, float, float, float],
    seed: int,
    config: Mapping | None,
) -> tuple[list[list[float]], dict | None]:
    """Return seeded XY positions and the actual legacy randomization record.

    ``bounds_xy`` is the physical tabletop region after its edge margin has been
    removed, not an already footprint-eroded centre region. ``footprint_xy`` is
    the conservative full XY extent, including the intended yaw envelope.
    Disabled randomization returns copied positions and ``None``. Failure to
    apply an enabled grid planner raises instead of silently keeping a grid.
    """
    original = [_finite_sequence(point, 2, "position") for point in positions]
    if not original:
        raise LayoutError("randomization requires at least one position")
    if config is None:
        return original, None
    if not isinstance(config, Mapping) or not isinstance(config.get("enabled", False), bool):
        raise LayoutError("randomization config must be a mapping with boolean enabled")
    if not config.get("enabled", False):
        return original, None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise LayoutError("randomization seed must be an integer")
    bounds = tuple(_finite_sequence(bounds_xy, 4, "bounds_xy"))
    sx, sy = _finite_sequence(footprint_xy, 2, "footprint_xy")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3] or min(sx, sy) <= 0:
        raise LayoutError("randomization requires ordered bounds and a positive footprint")
    jitter = _nonnegative(config.get("jitter_xy_m", 0.02), "jitter_xy_m")
    shift = _nonnegative(config.get("shift_max_m", 0.15), "shift_max_m")
    kind = layout.get("type")
    if kind not in {"grid", "line", "ring", "t_shape", "random"}:
        raise LayoutError(f"Unsupported randomized layout: {kind}")

    # The legacy radius is a half-maximum XY extent. A measured cuboid proxy is
    # used only for that planner's footprint bookkeeping; no proxy is exported.
    radius = max(sx, sy) * 0.5
    for x, y in original:
        if not (bounds[0] + radius <= x <= bounds[1] - radius and bounds[2] + radius <= y <= bounds[3] - radius):
            raise LayoutError("Authored layout exceeds the explicit randomization bounds")
    objects = [
        {"name": f"assembly_object_{index:03d}", "kind": "primitive_cuboid", "size": [sx, sy, 0.01], "pose": {"xy": list(point)}}
        for index, point in enumerate(original)
    ]
    local_jitter = kind == "grid"
    if local_jitter:
        rows, cols = layout.get("rows"), layout.get("cols")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (rows, cols)) or rows * cols != len(objects):
            raise LayoutError("Randomized grid rows*cols must equal the object count")
        row_x = [original[row * cols][0] for row in range(rows)]
        col_y = [original[col][1] for col in range(cols)]
        if any(row_x[index] >= row_x[index + 1] for index in range(rows - 1)) or any(col_y[index] >= col_y[index + 1] for index in range(cols - 1)) or any(
            abs(point[0] - row_x[index // cols]) > 1e-8 or abs(point[1] - col_y[index % cols]) > 1e-8
            for index, point in enumerate(original)
        ):
            raise LayoutError("Authored grid must be axis aligned in increasing row-major order")
        protocol = {"gt_spec": {"resolver": "grid_rowcol_slot", "grid_num_rows": rows, "grid_num_cols": cols, "row_axis": "x", "grid_quant_step": 0.02}}
    else:
        protocol = {"gt_spec": {}}
    settings = LayoutRandomizationConfig(
        jitter_enabled=local_jitter,
        jitter_xy=jitter,
        jitter_seed=seed,
        shift_enabled=shift > 0,
        shift_max=shift,
        shift_seed=seed ^ 0x9E3779B9,
        min_clearance=0.025,
        bounds=bounds,
        max_attempts=256,
    )
    if not local_jitter and shift == 0:
        return original, {
            "enabled": True, "backend": "runtime_layout_randomizer", "strategy": "xy_disabled",
            "jitter_enabled": False, "shift_enabled": False, "jitter_seed": settings.jitter_seed,
            "shift_seed": settings.shift_seed, "jitter_xy": jitter, "shift_max": shift,
            "global_shift": [0.0, 0.0], "bounds": list(bounds),
            "source_positions_xy_m": original, "positions_xy_m": [list(point) for point in original],
            "local_jitter_policy": "disabled_to_preserve_structure", "layout_type": kind,
        }
    record = randomize_runtime_object_layout(
        objects, protocol=protocol, layout_key=f"scene_assembly_{kind}", config=settings, verbose=False,
    )
    expected_strategy = "grid" if local_jitter else "shift_only"
    if record is None or record.get("strategy") != expected_strategy:
        raise LayoutError(f"Legacy randomizer did not apply {expected_strategy}; refusing a silent fallback")
    result = [list(item["pose"]["xy"]) for item in objects]
    if any(not (bounds[0] + radius - 1e-9 <= x <= bounds[1] - radius + 1e-9 and bounds[2] + radius - 1e-9 <= y <= bounds[3] - radius + 1e-9) for x, y in result):
        raise LayoutError("Legacy randomization exceeded the explicit tabletop bounds")
    record.update({
        "enabled": True,
        "backend": "runtime_layout_randomizer",
        "layout_type": kind,
        "source_positions_xy_m": original,
        "positions_xy_m": result,
        "local_jitter_policy": "legacy_grid" if local_jitter else "disabled_to_preserve_structure",
    })
    if local_jitter:
        record["cell_indices"] = [[index // cols, index % cols] for index in range(len(objects))]
        record["grid_quant_step_m"] = 0.02
        record["cell_wobble_bound_m"] = min(jitter, 0.02 * 0.25)
    return result, record
