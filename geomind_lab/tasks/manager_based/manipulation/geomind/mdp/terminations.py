from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import SceneEntityCfg

from .common import (
    get_asset_angular_speed_w,
    get_asset_dimensions,
    get_asset_linear_speed_w,
    get_asset_pose_w,
    goal_tolerance_distance,
    rotation_matrix_from_quat,
)
from .observations import object_grasped

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _vector_tensor(
    env: "ManagerBasedRLEnv",
    value: Any,
    *,
    length: int,
    default: tuple[float, ...],
) -> torch.Tensor:
    resolved = list(default[:length])
    if isinstance(value, (list, tuple)):
        for idx in range(min(length, len(value))):
            resolved[idx] = _safe_float(value[idx], resolved[idx])
    return torch.tensor(resolved, dtype=torch.float32, device=env.device)


def _task_protocol_dict(task_protocol: dict[str, Any] | None) -> dict[str, Any]:
    return dict(task_protocol) if isinstance(task_protocol, dict) else {}


def _gt_spec_dict(task_protocol: dict[str, Any] | None) -> dict[str, Any]:
    protocol = _task_protocol_dict(task_protocol)
    gt_spec = protocol.get("gt_spec", {})
    return dict(gt_spec) if isinstance(gt_spec, dict) else {}


def _resolve_object_name(
    task_protocol: dict[str, Any] | None,
    *,
    gt_spec_key: str,
    protocol_key: str,
) -> str:
    protocol = _task_protocol_dict(task_protocol)
    gt_spec = _gt_spec_dict(task_protocol)

    object_name = str(gt_spec.get(gt_spec_key, "")).strip()
    if object_name:
        return object_name

    payload = protocol.get(protocol_key, {})
    if isinstance(payload, dict):
        object_name = str(payload.get("object_id", "")).strip()
    return object_name


def _object_released(
    env: "ManagerBasedRLEnv",
    object_name: str,
) -> torch.Tensor:
    return ~object_grasped(
        env,
        robot_cfg=SceneEntityCfg("robot"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg(object_name),
    ).to(torch.bool)


def _object_stable(
    env: "ManagerBasedRLEnv",
    object_name: str,
    *,
    max_linear_speed: float,
    max_angular_speed: float,
) -> torch.Tensor:
    linear_velocity = get_asset_linear_speed_w(env, object_name)
    angular_velocity = get_asset_angular_speed_w(env, object_name)
    if linear_velocity is None or angular_velocity is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    linear_speed = torch.linalg.vector_norm(linear_velocity, dim=1)
    angular_speed = torch.linalg.vector_norm(angular_velocity, dim=1)
    return (linear_speed <= float(max_linear_speed)) & (angular_speed <= float(max_angular_speed))


def _end_effector_pos_w(env: "ManagerBasedRLEnv", ee_frame_name: str = "ee_frame") -> torch.Tensor | None:
    if ee_frame_name not in env.scene.keys():
        return None
    ee_frame = env.scene[ee_frame_name]
    data = getattr(ee_frame, "data", None)
    target_pos_w = getattr(data, "target_pos_w", None)
    if target_pos_w is None:
        return None
    return target_pos_w[:, 0, :]


def _move_to_object_proximity_reached(
    env: "ManagerBasedRLEnv",
    *,
    task_protocol: dict[str, Any] | None,
) -> torch.Tensor:
    gt_spec = _gt_spec_dict(task_protocol)
    from geomind_lab.mimic.envs.task_contract_logic import protocol_gt_positions
    dynamic_target = protocol_gt_positions(env, task_protocol or {})
    target_name = _resolve_object_name(task_protocol, gt_spec_key="target_object_id", protocol_key="target_object")
    if dynamic_target is None and (not target_name or target_name not in env.scene.keys()):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    ee_pos_w = _end_effector_pos_w(env, str(gt_spec.get("ee_frame_name", "ee_frame") or "ee_frame"))
    if ee_pos_w is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if dynamic_target is None:
        target_pos_w, _ = get_asset_pose_w(env, target_name)
        valid_target = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        target_pos_w, valid_target = dynamic_target
    target_offset = _vector_tensor(
        env,
        gt_spec.get("target_offset", (0.0, 0.0, 0.0)),
        length=3,
        default=(0.0, 0.0, 0.0),
    )
    target_point_w = target_pos_w + target_offset.unsqueeze(0)
    radius = _safe_float(
        gt_spec.get("success_radius", gt_spec.get("proximity_radius", gt_spec.get("target_proximity_radius"))),
        0.085,
    )
    if bool(gt_spec.get("xy_only", False)):
        distance = torch.linalg.vector_norm(ee_pos_w[:, :2] - target_point_w[:, :2], dim=1)
    else:
        distance = torch.linalg.vector_norm(ee_pos_w - target_point_w, dim=1)
    return valid_target & (distance <= float(radius))


def _projection_overlap_ratio(
    *,
    target_center_xy: torch.Tensor,
    target_size_xy: torch.Tensor,
    support_center_xy: torch.Tensor,
    support_size_xy: torch.Tensor,
) -> torch.Tensor:
    target_half = 0.5 * target_size_xy
    support_half = 0.5 * support_size_xy

    target_min = target_center_xy - target_half
    target_max = target_center_xy + target_half
    support_min = support_center_xy - support_half
    support_max = support_center_xy + support_half

    overlap = torch.clamp(torch.minimum(target_max, support_max) - torch.maximum(target_min, support_min), min=0.0)
    overlap_area = overlap[:, 0] * overlap[:, 1]
    target_area = torch.clamp(target_size_xy[:, 0] * target_size_xy[:, 1], min=1.0e-8)
    return overlap_area / target_area


def _directional_region_placement_reached(
    env: "ManagerBasedRLEnv",
    *,
    task_protocol: dict[str, Any] | None,
) -> torch.Tensor:
    gt_spec = _gt_spec_dict(task_protocol)
    target_name = _resolve_object_name(task_protocol, gt_spec_key="target_object_id", protocol_key="target_object")
    anchor_name = _resolve_object_name(task_protocol, gt_spec_key="anchor_object_id", protocol_key="anchor_object")
    if not target_name or not anchor_name:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if target_name not in env.scene.keys() or anchor_name not in env.scene.keys():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    target_zone = gt_spec.get("target_zone", {})
    if not isinstance(target_zone, dict):
        target_zone = {}
    center_offset = _vector_tensor(
        env,
        target_zone.get("center_offset_from_anchor", (0.0, 0.0)),
        length=2,
        default=(0.0, 0.0),
    )
    half_extents = _vector_tensor(
        env,
        target_zone.get("half_extents", (0.0, 0.0)),
        length=2,
        default=(0.0, 0.0),
    )

    target_pos_w, _ = get_asset_pose_w(env, target_name)
    anchor_pos_w, _ = get_asset_pose_w(env, anchor_name)
    target_xy = target_pos_w[:, :2]
    center_xy = anchor_pos_w[:, :2] + center_offset.unsqueeze(0)

    if bool(gt_spec.get("xy_aabb_fully_inside_zone", False)):
        dims = get_asset_dimensions(env, target_name)
        if dims is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        half_obj_x = 0.5 * float(dims[0])
        half_obj_y = 0.5 * float(dims[1])
        eff_half = half_extents - torch.tensor([half_obj_x, half_obj_y], dtype=torch.float32, device=env.device)
        if bool(torch.any(eff_half <= 1.0e-6)):
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        inside_zone = torch.all(torch.abs(target_xy - center_xy) <= eff_half.unsqueeze(0), dim=1)
    else:
        inside_zone = torch.all(torch.abs(target_xy - center_xy) <= half_extents.unsqueeze(0), dim=1)

    min_sep = gt_spec.get("min_anchor_center_xy_distance", None)
    if min_sep is not None:
        min_dist = _safe_float(min_sep, 0.0)
        anchor_xy = anchor_pos_w[:, :2]
        sep = torch.linalg.vector_norm(target_xy - anchor_xy, dim=1)
        inside_zone &= sep >= min_dist

    max_lin = gt_spec.get("max_linear_speed_after_settle", None)
    max_ang = gt_spec.get("max_angular_speed_after_settle", None)
    if max_lin is not None and max_ang is not None:
        inside_zone &= _object_stable(
            env,
            target_name,
            max_linear_speed=_safe_float(max_lin, 0.05),
            max_angular_speed=_safe_float(max_ang, 0.4),
        )

    if bool(gt_spec.get("must_be_released", False)):
        inside_zone &= _object_released(env, target_name)
    return inside_zone


def _support_top_placement_reached(
    env: "ManagerBasedRLEnv",
    *,
    task_protocol: dict[str, Any] | None,
) -> torch.Tensor:
    gt_spec = _gt_spec_dict(task_protocol)
    target_name = _resolve_object_name(task_protocol, gt_spec_key="target_object_id", protocol_key="target_object")
    support_name = _resolve_object_name(task_protocol, gt_spec_key="support_object_id", protocol_key="support_object")
    if not target_name or not support_name:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if target_name not in env.scene.keys() or support_name not in env.scene.keys():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    target_dims = get_asset_dimensions(env, target_name)
    support_dims = get_asset_dimensions(env, support_name)
    if target_dims is None or support_dims is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    target_size = torch.tensor(target_dims, dtype=torch.float32, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    support_size = torch.tensor(support_dims, dtype=torch.float32, device=env.device).unsqueeze(0).expand(env.num_envs, -1)

    target_pos_w, _ = get_asset_pose_w(env, target_name)
    support_pos_w, _ = get_asset_pose_w(env, support_name)

    support_top_z = support_pos_w[:, 2] + 0.5 * support_size[:, 2]
    target_bottom_z = target_pos_w[:, 2] - 0.5 * target_size[:, 2]
    support_gap_abs = torch.abs(target_bottom_z - support_top_z)
    overlap_ratio = _projection_overlap_ratio(
        target_center_xy=target_pos_w[:, :2],
        target_size_xy=target_size[:, :2],
        support_center_xy=support_pos_w[:, :2],
        support_size_xy=support_size[:, :2],
    )

    contact_required = bool(gt_spec.get("contact_required", False))
    support_required = bool(gt_spec.get("support_required", False))
    must_be_released = bool(gt_spec.get("must_be_released", False))
    max_support_gap = _safe_float(gt_spec.get("max_support_gap"), 0.015)
    min_overlap_ratio = _safe_float(gt_spec.get("min_support_projection_overlap_ratio"), 0.72)
    max_linear_speed = _safe_float(gt_spec.get("max_linear_speed_after_settle"), 0.05)
    max_angular_speed = _safe_float(gt_spec.get("max_angular_speed_after_settle"), 0.4)

    object_above_support = target_pos_w[:, 2] > (support_top_z + 0.25 * target_size[:, 2])
    support_contact = (support_gap_abs <= max_support_gap) & (overlap_ratio >= min_overlap_ratio)
    released = _object_released(env, target_name)
    stable = _object_stable(
        env,
        target_name,
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )

    success = stable
    if must_be_released:
        success &= released
    if contact_required:
        success &= support_contact
    if support_required:
        success &= object_above_support & support_contact
    return success


def _containment_placement_reached(
    env: "ManagerBasedRLEnv",
    *,
    task_protocol: dict[str, Any] | None,
) -> torch.Tensor:
    gt_spec = _gt_spec_dict(task_protocol)
    target_name = _resolve_object_name(task_protocol, gt_spec_key="target_object_id", protocol_key="target_object")
    container_reference_name = str(gt_spec.get("container_reference_object_id", "")).strip()
    if not target_name or not container_reference_name:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if target_name not in env.scene.keys() or container_reference_name not in env.scene.keys():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    target_dims = get_asset_dimensions(env, target_name)
    if target_dims is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    target_pos_w, target_quat_w = get_asset_pose_w(env, target_name)
    reference_pos_w, reference_quat_w = get_asset_pose_w(env, container_reference_name)

    reference_rot = rotation_matrix_from_quat(reference_quat_w)
    target_rot = rotation_matrix_from_quat(target_quat_w)
    relative_rot = torch.matmul(reference_rot.transpose(1, 2), target_rot)
    relative_center = torch.matmul(
        reference_rot.transpose(1, 2),
        (target_pos_w - reference_pos_w).unsqueeze(-1),
    ).squeeze(-1)

    target_half_size = torch.tensor(
        [0.5 * float(target_dims[0]), 0.5 * float(target_dims[1]), 0.5 * float(target_dims[2])],
        dtype=torch.float32,
        device=env.device,
    ).unsqueeze(0).expand(env.num_envs, -1)
    projected_half_extents = torch.matmul(torch.abs(relative_rot), target_half_size.unsqueeze(-1)).squeeze(-1)

    interior_half_extents_xy = _vector_tensor(
        env,
        gt_spec.get("interior_half_extents_xy", (0.0, 0.0)),
        length=2,
        default=(0.0, 0.0),
    )
    interior_floor_top_offset = _safe_float(gt_spec.get("interior_floor_top_offset"), 0.0)
    opening_plane_z_offset = _safe_float(gt_spec.get("opening_plane_z_offset"), 0.0)
    xy_inner_margin = _safe_float(gt_spec.get("xy_inner_margin"), 0.0)
    max_floor_gap = _safe_float(gt_spec.get("max_floor_gap"), 0.02)
    max_floor_penetration = _safe_float(gt_spec.get("max_floor_penetration"), 0.004)
    min_top_clearance = _safe_float(gt_spec.get("min_top_clearance_below_opening"), 0.0)
    max_linear_speed = _safe_float(gt_spec.get("max_linear_speed_after_settle"), 0.05)
    max_angular_speed = _safe_float(gt_spec.get("max_angular_speed_after_settle"), 0.4)
    must_be_released = bool(gt_spec.get("must_be_released", False))

    available_half_extents_xy = interior_half_extents_xy.unsqueeze(0) - projected_half_extents[:, :2] - xy_inner_margin
    has_non_negative_room = torch.all(available_half_extents_xy >= 0.0, dim=1)
    xy_inside = has_non_negative_room & torch.all(
        torch.abs(relative_center[:, :2]) <= available_half_extents_xy + 1.0e-6,
        dim=1,
    )

    target_bottom_local = relative_center[:, 2] - projected_half_extents[:, 2]
    target_top_local = relative_center[:, 2] + projected_half_extents[:, 2]
    floor_gap = target_bottom_local - interior_floor_top_offset
    top_clearance = opening_plane_z_offset - target_top_local

    near_floor = floor_gap <= max_floor_gap
    not_penetrating_floor = floor_gap >= -max_floor_penetration
    below_opening = top_clearance >= min_top_clearance
    released = _object_released(env, target_name)
    stable = _object_stable(
        env,
        target_name,
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )

    success = xy_inside & near_floor & not_penetrating_floor & below_opening & stable
    if must_be_released:
        success &= released
    return success


def asset_height_below_minimum(
    env: "ManagerBasedRLEnv",
    minimum_height: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    asset_pos_w, _ = get_asset_pose_w(env, asset_cfg.name)
    return asset_pos_w[:, 2] < minimum_height


def goal_sequence_reached(
    env: "ManagerBasedRLEnv",
    goal_sequence: list[dict] | None = None,
    xy_only: bool = True,
    task_protocol: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Checks resolver-backed placement success first, then falls back to the legacy goal_sequence XY check."""
    gt_spec = _gt_spec_dict(task_protocol)
    resolver = str(gt_spec.get("resolver", "")).strip()

    if resolver == "directional_region_placement":
        return _directional_region_placement_reached(env, task_protocol=task_protocol)
    if resolver in {"move_to_object", "move_to_object_proximity", "object_proximity"}:
        return _move_to_object_proximity_reached(env, task_protocol=task_protocol)
    if resolver == "support_top_placement":
        return _support_top_placement_reached(env, task_protocol=task_protocol)
    if resolver == "containment_placement":
        return _containment_placement_reached(env, task_protocol=task_protocol)

    if not goal_sequence:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    goals = goal_sequence[0].get("goals", [])
    if not goals:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    success = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    scene_keys = set(env.scene.keys())
    for goal in goals:
        object_name = goal.get("object_id")
        if object_name not in scene_keys:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        asset_pos_w, _ = get_asset_pose_w(env, object_name)
        target_xy = torch.tensor(goal.get("xy", [0.0, 0.0]), dtype=torch.float32, device=env.device).unsqueeze(0)
        tolerance_spec = goal.get("tolerance", {})
        tolerance = float(tolerance_spec.get("pos_r", 0.06))

        if xy_only:
            dist = goal_tolerance_distance(asset_pos_w[:, :2] - target_xy, tolerance_spec)
        else:
            target = torch.cat([target_xy, asset_pos_w[:, 2:3]], dim=1)
            dist = torch.linalg.vector_norm(asset_pos_w - target, dim=1)
        success &= dist <= tolerance

    return success


def _truthy_geomind_ignore_success_termination() -> bool:
    raw = str(os.environ.get("GEOMIND_IGNORE_SUCCESS_TERMINATION", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def goal_sequence_reached_for_done_term(
    env: "ManagerBasedRLEnv",
    goal_sequence: list[dict] | None = None,
    xy_only: bool = True,
    task_protocol: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Success DoneTerm only: when GEOMIND_IGNORE_SUCCESS_TERMINATION is set, never fire success so Isaac does not reset mid-rollout. Observations still use goal_sequence_reached."""
    if _truthy_geomind_ignore_success_termination():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return goal_sequence_reached(env, goal_sequence, xy_only, task_protocol)
