"""Optional in-viewer debug visualization for spatial-placement success regions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.common import get_asset_pose_w
from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.terminations import (
    _gt_spec_dict,
    _resolve_object_name,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _resolve_directional_zone(env: "ManagerBasedRLEnv") -> tuple[str, float, float, float, float] | None:
    cfg = getattr(env, "cfg", None)
    if cfg is None or not bool(getattr(cfg, "viz_directional_placement_zone", True)):
        return None

    protocol = dict(getattr(cfg, "task_protocol", {}) or {})
    gt_spec = _gt_spec_dict(protocol)
    if str(gt_spec.get("resolver", "")).strip() != "directional_region_placement":
        return None

    anchor_name = _resolve_object_name(protocol, gt_spec_key="anchor_object_id", protocol_key="anchor_object")
    if not anchor_name or str(anchor_name) not in env.scene.keys():
        return None

    target_zone = gt_spec.get("target_zone", {})
    if not isinstance(target_zone, dict):
        return None
    center_offset = target_zone.get("center_offset_from_anchor", (0.0, 0.0))
    half_extents = target_zone.get("half_extents", (0.0, 0.0))
    if not (isinstance(center_offset, (list, tuple)) and len(center_offset) >= 2):
        return None
    if not (isinstance(half_extents, (list, tuple)) and len(half_extents) >= 2):
        return None

    hx = _safe_float(half_extents[0], 0.0)
    hy = _safe_float(half_extents[1], 0.0)
    if hx <= 0.0 or hy <= 0.0:
        return None

    ox = _safe_float(center_offset[0], 0.0)
    oy = _safe_float(center_offset[1], 0.0)
    return str(anchor_name), ox, oy, hx, hy


def _collect_zone_rects(
    env: "ManagerBasedRLEnv",
    anchor_name: str,
    ox: float,
    oy: float,
    hx: float,
    hy: float,
) -> list[tuple[int, float, float, float, float, float]]:
    try:
        anchor_pos_w, _ = get_asset_pose_w(env, anchor_name)
    except Exception:
        return []

    num_envs = int(getattr(env, "num_envs", 1) or 1)
    try:
        num_anchor_envs = int(anchor_pos_w.shape[0])
    except Exception:
        num_anchor_envs = num_envs
    max_viz = min(num_envs, num_anchor_envs, 8)

    rects: list[tuple[int, float, float, float, float, float]] = []
    for env_id in range(max_viz):
        ax = float(anchor_pos_w[env_id, 0].item())
        ay = float(anchor_pos_w[env_id, 1].item())
        az = float(anchor_pos_w[env_id, 2].item()) + 0.012
        rects.append((env_id, ax + ox, ay + oy, az, hx, hy))
    return rects


def _draw_directional_zone_debug_lines(rects: list[tuple[int, float, float, float, float, float]]) -> None:
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    try:
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()

        rgba = (0.15, 0.95, 0.25, 1.0)
        line_width = 4.0

        starts: list[tuple[float, float, float]] = []
        ends: list[tuple[float, float, float]] = []
        for _env_id, cx, cy, z, hx, hy in rects:
            x0, y0 = cx - hx, cy - hy
            x1, y1 = cx + hx, cy - hy
            x2, y2 = cx + hx, cy + hy
            x3, y3 = cx - hx, cy + hy

            p0 = (x0, y0, z)
            p1 = (x1, y1, z)
            p2 = (x2, y2, z)
            p3 = (x3, y3, z)

            starts.extend([p0, p1, p2, p3])
            ends.extend([p1, p2, p3, p0])

        if starts:
            colors = [rgba] * len(starts)
            sizes = [line_width] * len(starts)
            draw.draw_lines(starts, ends, colors, sizes)
    except Exception:
        return


def draw_directional_placement_success_zone_debug(env: "ManagerBasedRLEnv") -> None:
    """Draw the XY success rectangle (``gt_spec.target_zone``) as a flat border in world space.

    Uses Isaac Sim debug lines when ``env.sim.has_gui()`` is true.
    """
    zone = _resolve_directional_zone(env)
    if zone is None:
        return

    anchor_name, ox, oy, hx, hy = zone
    rects = _collect_zone_rects(env, anchor_name, ox, oy, hx, hy)
    if not rects:
        return

    sim = getattr(env, "sim", None)
    if sim is None or not sim.has_gui():
        return

    _draw_directional_zone_debug_lines(rects)
