from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as PoseUtils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _to_env_tensor(env: "ManagerBasedRLEnv", value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=env.device, dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32, device=env.device)


def _positive_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0.0 else None


def _unit_axis(value) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        axis_x, axis_y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    norm = (axis_x * axis_x + axis_y * axis_y) ** 0.5
    if norm <= 1.0e-9:
        return None
    return axis_x / norm, axis_y / norm


def goal_tolerance_distance(delta_xy: torch.Tensor, tolerance=None) -> torch.Tensor:
    """XY goal distance rescaled so ``<= pos_r`` matches the annotated tolerance shape.

    Geometric-inference layouts budget ``success_tolerance_l2`` *along* the rule's axis and the
    much tighter ``success_tolerance_orthogonal`` perpendicular to it, so the accepted region is
    an ellipse rather than a circle: an isotropic ``pos_r`` would accept a block sitting a full
    ``pos_r`` off the inferred line. Goals without ``axis``/``orthogonal_r`` keep the plain L2
    distance, and the two agree exactly when ``orthogonal_r == pos_r``.
    """
    tolerance = tolerance if isinstance(tolerance, dict) else {}
    radius = _positive_float(tolerance.get("pos_r"))
    orthogonal_r = _positive_float(tolerance.get("orthogonal_r"))
    axis = _unit_axis(tolerance.get("axis"))
    if radius is None or orthogonal_r is None or axis is None:
        return torch.linalg.vector_norm(delta_xy, dim=1)

    along = delta_xy[:, 0] * axis[0] + delta_xy[:, 1] * axis[1]
    perpendicular = -delta_xy[:, 0] * axis[1] + delta_xy[:, 1] * axis[0]
    normalized = torch.sqrt((along / radius) ** 2 + (perpendicular / orthogonal_r) ** 2)
    return normalized * radius


def get_asset_pose_w(env: "ManagerBasedRLEnv", asset_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns world pose tensors for either rigid assets or AssetBaseCfg extras."""
    asset = env.scene[asset_name]

    if hasattr(asset, "data") and hasattr(asset.data, "root_pos_w"):
        pos_w = asset.data.root_pos_w
        quat_w = getattr(asset.data, "root_quat_w", None)
        if quat_w is None:
            quat_w = torch.zeros((pos_w.shape[0], 4), dtype=torch.float32, device=env.device)
            quat_w[:, 0] = 1.0
        return pos_w, quat_w

    if hasattr(asset, "get_world_poses"):
        pos_w, quat_w = asset.get_world_poses()
        pos_w = _to_env_tensor(env, pos_w)
        quat_w = _to_env_tensor(env, quat_w) if quat_w is not None else None
        if pos_w.ndim == 1:
            pos_w = pos_w.unsqueeze(0)
        if quat_w is None:
            quat_w = torch.zeros((pos_w.shape[0], 4), dtype=torch.float32, device=env.device)
            quat_w[:, 0] = 1.0
        elif quat_w.ndim == 1:
            quat_w = quat_w.unsqueeze(0)
        return pos_w, quat_w

    raise TypeError(f"Unsupported scene entity type for pose query: '{asset_name}' ({type(asset)!r})")


def get_asset_dimensions(env: "ManagerBasedRLEnv", asset_name: str) -> tuple[float, float, float] | None:
    resolver = getattr(env.cfg, "_resolve_object_runtime_dimensions", None)
    if not callable(resolver):
        return None
    resolved = resolver(asset_name)
    if resolved is None:
        return None
    return tuple(float(value) for value in resolved)


def get_asset_linear_speed_w(env: "ManagerBasedRLEnv", asset_name: str) -> torch.Tensor | None:
    asset = env.scene[asset_name]
    if hasattr(asset, "data") and hasattr(asset.data, "root_lin_vel_w"):
        return asset.data.root_lin_vel_w[:, :3]
    return None


def get_asset_angular_speed_w(env: "ManagerBasedRLEnv", asset_name: str) -> torch.Tensor | None:
    asset = env.scene[asset_name]
    if hasattr(asset, "data") and hasattr(asset.data, "root_ang_vel_w"):
        return asset.data.root_ang_vel_w[:, :3]
    return None


def rotation_matrix_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
    if quat_w.ndim == 1:
        quat_w = quat_w.unsqueeze(0)
    return PoseUtils.matrix_from_quat(quat_w)
