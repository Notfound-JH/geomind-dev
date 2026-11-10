from __future__ import annotations

from typing import Any
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .common import get_asset_pose_w


def _resolve_object_cfgs(object_cfgs: list[SceneEntityCfg] | None) -> list[SceneEntityCfg]:
    if object_cfgs:
        return list(object_cfgs)
    return []


def cube_positions_in_world_frame(
    env: "ManagerBasedRLEnv",
    object_cfgs: list[SceneEntityCfg] | None = None,
) -> torch.Tensor:
    cfgs = _resolve_object_cfgs(object_cfgs)
    if not cfgs:
        return torch.zeros((env.num_envs, 0), device=env.device)
    positions = [get_asset_pose_w(env, cfg.name)[0] for cfg in cfgs]
    return torch.cat(positions, dim=1)


def cube_orientations_in_world_frame(
    env: "ManagerBasedRLEnv",
    object_cfgs: list[SceneEntityCfg] | None = None,
) -> torch.Tensor:
    cfgs = _resolve_object_cfgs(object_cfgs)
    if not cfgs:
        return torch.zeros((env.num_envs, 0), device=env.device)
    orientations = [get_asset_pose_w(env, cfg.name)[1] for cfg in cfgs]
    return torch.cat(orientations, dim=1)


def object_obs(
    env: "ManagerBasedRLEnv",
    object_cfgs: list[SceneEntityCfg] | None = None,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    cfgs = _resolve_object_cfgs(object_cfgs)
    if not cfgs:
        return torch.zeros((env.num_envs, 0), device=env.device)

    positions: list[torch.Tensor] = []
    orientations: list[torch.Tensor] = []
    for cfg in cfgs:
        pos_w, quat_w = get_asset_pose_w(env, cfg.name)
        positions.append(pos_w)
        orientations.append(quat_w)

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    gripper_to_objects = [pos - ee_pos_w for pos in positions]

    if len(cfgs) == 3:
        pair_indices = [(0, 1), (1, 2), (0, 2)]
    else:
        pair_indices = [(i, j) for i in range(len(cfgs)) for j in range(i + 1, len(cfgs))]
    object_deltas = [positions[i] - positions[j] for i, j in pair_indices]

    object_pose_features: list[torch.Tensor] = []
    for pos, quat in zip(positions, orientations):
        object_pose_features.append(pos - env.scene.env_origins)
        object_pose_features.append(quat)

    return torch.cat((*object_pose_features, *gripper_to_objects, *object_deltas), dim=1)


def object_grasped(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w, _ = get_asset_pose_w(env, object_cfg.name)

    end_effector_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos_w - end_effector_pos_w, dim=1)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        return torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"
        grasped = torch.logical_and(
            pose_diff < diff_threshold,
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[0]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            )
            > env.cfg.gripper_threshold,
        )
        grasped = torch.logical_and(
            grasped,
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[1]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            )
            > env.cfg.gripper_threshold,
        )
        return grasped

    return pose_diff < diff_threshold


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


def task_goal_xy_error(
    env: "ManagerBasedRLEnv",
    goal_sequence: list[dict] | None = None,
    task_protocol: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Returns stacked XY error vectors for resolver-backed placements or legacy XY goals."""
    gt_spec = _gt_spec_dict(task_protocol)
    resolver = str(gt_spec.get("resolver", "")).strip()
    if resolver in {"move_to_object", "move_to_object_proximity", "object_proximity"}:
        from geomind_lab.mimic.envs.task_contract_logic import protocol_gt_positions
        dynamic_target = protocol_gt_positions(env, task_protocol or {})
        if dynamic_target is not None and "ee_frame" in env.scene.keys():
            target_pos_w, valid = dynamic_target
            error = target_pos_w[:, :2] - env.scene["ee_frame"].data.target_pos_w[:, 0, :2]
            return torch.where(valid.unsqueeze(1), error, torch.zeros_like(error))
        target_name = _resolve_object_name(task_protocol, gt_spec_key="target_object_id", protocol_key="target_object")
        if target_name in env.scene.keys() and "ee_frame" in env.scene.keys():
            target_pos_w, _ = get_asset_pose_w(env, target_name)
            ee_frame: FrameTransformer = env.scene["ee_frame"]
            ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
            return target_pos_w[:, :2] - ee_pos_w[:, :2]

    if resolver == "directional_region_placement":
        target_name = _resolve_object_name(task_protocol, gt_spec_key="target_object_id", protocol_key="target_object")
        anchor_name = _resolve_object_name(task_protocol, gt_spec_key="anchor_object_id", protocol_key="anchor_object")
        if target_name in env.scene.keys() and anchor_name in env.scene.keys():
            target_pos_w, _ = get_asset_pose_w(env, target_name)
            anchor_pos_w, _ = get_asset_pose_w(env, anchor_name)
            target_zone = gt_spec.get("target_zone", {})
            if not isinstance(target_zone, dict):
                target_zone = {}
            center_offset = target_zone.get("center_offset_from_anchor", (0.0, 0.0))
            if isinstance(center_offset, (list, tuple)) and len(center_offset) >= 2:
                center_xy = anchor_pos_w[:, :2] + torch.tensor(
                    [float(center_offset[0]), float(center_offset[1])],
                    dtype=torch.float32,
                    device=env.device,
                ).unsqueeze(0)
                return center_xy - target_pos_w[:, :2]

    if resolver == "support_top_placement":
        target_name = _resolve_object_name(task_protocol, gt_spec_key="target_object_id", protocol_key="target_object")
        support_name = _resolve_object_name(task_protocol, gt_spec_key="support_object_id", protocol_key="support_object")
        if target_name in env.scene.keys() and support_name in env.scene.keys():
            target_pos_w, _ = get_asset_pose_w(env, target_name)
            support_pos_w, _ = get_asset_pose_w(env, support_name)
            return support_pos_w[:, :2] - target_pos_w[:, :2]

    if resolver == "containment_placement":
        target_name = _resolve_object_name(task_protocol, gt_spec_key="target_object_id", protocol_key="target_object")
        reference_name = str(gt_spec.get("container_reference_object_id", "")).strip()
        if target_name in env.scene.keys() and reference_name in env.scene.keys():
            target_pos_w, _ = get_asset_pose_w(env, target_name)
            reference_pos_w, _ = get_asset_pose_w(env, reference_name)
            return reference_pos_w[:, :2] - target_pos_w[:, :2]

    if not goal_sequence:
        return torch.zeros((env.num_envs, 2), device=env.device)

    first_step = goal_sequence[0]
    goals = first_step.get("goals", [])
    if not goals:
        return torch.zeros((env.num_envs, 2), device=env.device)

    errors: list[torch.Tensor] = []
    scene_keys = set(env.scene.keys())
    for goal in goals:
        object_name = goal.get("object_id")
        if object_name not in scene_keys:
            continue
        xy_target = torch.tensor(goal.get("xy", [0.0, 0.0]), dtype=torch.float32, device=env.device).unsqueeze(0)
        asset_pos_w, _ = get_asset_pose_w(env, object_name)
        xy_current = asset_pos_w[:, :2]
        errors.append(xy_target - xy_current)

    if not errors:
        return torch.zeros((env.num_envs, 2), device=env.device)
    return torch.cat(errors, dim=1)


def object_xy(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Returns object XY position in world frame."""
    asset_pos_w, _ = get_asset_pose_w(env, object_cfg.name)
    return asset_pos_w[:, :2]
