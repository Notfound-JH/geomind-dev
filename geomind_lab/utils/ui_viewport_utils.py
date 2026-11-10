from __future__ import annotations

import json
import re


def parse_ui_viewport_camera_map(raw_value: str | None) -> dict[int, str] | None:
    """Parse a JSON mapping from 1-based viewport index to camera name or camera prim path."""
    if raw_value is None:
        return None

    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("ui viewport camera map must be a JSON object.")

    resolved: dict[int, str] = {}
    for key, value in parsed.items():
        viewport_index = int(key)
        if viewport_index < 1:
            raise ValueError(f"Viewport index must be >= 1. Received: {viewport_index}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Viewport {viewport_index} must map to a non-empty camera name or prim path.")
        resolved[viewport_index] = value.strip()
    return resolved


def configure_env_camera_viewports(env, viewport_camera_map: dict[int, str] | None) -> None:
    """Bind UI viewport windows to selected GeoMind cameras.

    The default ``Viewport`` window is always kept alive and visible.
    If ``viewport_camera_map`` is None, no extra camera windows are created.
    Otherwise, it must be a dict like ``{1: "table_cam", 2: "wrist_cam"}``,
    where each entry creates an additional viewport window.
    """
    import omni.ui
    from omni.kit.viewport.utility import get_viewport_from_window_name

    default_viewport_window = omni.ui.Workspace.get_window("Viewport")
    if default_viewport_window is not None:
        default_viewport_window.visible = True

    default_viewport = get_viewport_from_window_name("Viewport")
    if default_viewport is None:
        raise RuntimeError("Unable to acquire the default 'Viewport' window.")

    if viewport_camera_map is None:
        return
    if not isinstance(viewport_camera_map, dict):
        raise TypeError("ui_viewport_camera_map must be a dict[int, str] or None.")
    if not viewport_camera_map:
        raise ValueError("ui_viewport_camera_map cannot be empty. Use None to keep the default single viewport.")

    ordered_indices = sorted(viewport_camera_map)
    expected_indices = list(range(1, len(ordered_indices) + 1))
    if ordered_indices != expected_indices:
        raise ValueError(
            f"ui_viewport_camera_map keys must be contiguous 1-based indices {expected_indices}. "
            f"Received: {ordered_indices}"
        )

    from omni.kit.viewport.utility import create_viewport_window
    from pxr import Sdf

    for viewport_index in ordered_indices:
        camera_prim_path = _resolve_camera_prim_path(env, viewport_camera_map[viewport_index])

        window_name = f"GeoMind View {viewport_index}"
        viewport_window = omni.ui.Workspace.get_window(window_name)
        if viewport_window is None:
            create_viewport_window(
                name=window_name,
                width=480,
                height=360,
                position_x=60 + (viewport_index - 1) * 40,
                position_y=60 + (viewport_index - 1) * 40,
                camera_path=Sdf.Path(camera_prim_path),
            )
        else:
            viewport_window.visible = True

        viewport = get_viewport_from_window_name(window_name)
        if viewport is None:
            raise RuntimeError(f"Unable to acquire viewport window '{window_name}'.")
        viewport.set_active_camera(camera_prim_path)


def _resolve_camera_prim_path(env, camera_ref: str) -> str:
    if not isinstance(camera_ref, str) or not camera_ref.strip():
        raise TypeError(f"Viewport camera mapping must point to a non-empty string. Received: {camera_ref!r}")

    camera_ref = camera_ref.strip()
    if camera_ref.startswith("/"):
        return _resolve_env_camera_prim_path(camera_ref)

    sensors = env.scene.sensors
    if camera_ref not in sensors:
        available_names = ", ".join(sorted(sensors.keys()))
        raise ValueError(
            f"Unknown camera name '{camera_ref}' in ui_viewport_camera_map. Available scene sensors: {available_names}"
        )

    sensor_cfg = getattr(sensors[camera_ref], "cfg", None)
    prim_path = getattr(sensor_cfg, "prim_path", None)
    if not prim_path:
        raise ValueError(f"Camera sensor '{camera_ref}' does not expose a valid prim_path.")
    return _resolve_env_camera_prim_path(prim_path)


def _resolve_env_camera_prim_path(prim_path: str) -> str:
    # UI viewport binding needs one concrete camera prim, so default to env_0 instead of the env regex.
    return re.sub(r"/env_\.\*", "/env_0", prim_path)
