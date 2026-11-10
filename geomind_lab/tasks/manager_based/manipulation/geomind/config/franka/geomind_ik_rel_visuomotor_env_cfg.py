from __future__ import annotations

import math
import os
from dataclasses import dataclass

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_from_matrix

from geomind_lab.utils import asset_utils

from .franka_env_cfg import FrankaGeoMindEnvCfg
from ... import geomind_env_cfg


DEFAULT_TABLE_TOP_HEIGHT = 0.7552
LOOK_AT_HEIGHT_ABOVE_TABLE = 0.0448
TABLE_VIEW_CAMERA_HEIGHT_ABOVE_TABLE = 0.4
ROOM_VIEW_CAMERA_HEIGHT_ABOVE_TABLE = 1.7448


@dataclass(frozen=True)
class CameraSpec:
    name: str
    position_xy: tuple[float, float]
    look_at_xy: tuple[float, float]
    height_above_table: float
    resolution: tuple[int, int]


DEFAULT_CAMERAS: list[CameraSpec] = [
    CameraSpec("front_camera", (1.0, 0.0), (0.3, 0.0), ROOM_VIEW_CAMERA_HEIGHT_ABOVE_TABLE, (224, 224)),
    CameraSpec("left_camera", (0.35, 1.0), (0.3, 0.0), ROOM_VIEW_CAMERA_HEIGHT_ABOVE_TABLE, (224, 224)),
    CameraSpec("right_camera", (0.35, -1.0), (0.3, 0.0), ROOM_VIEW_CAMERA_HEIGHT_ABOVE_TABLE, (224, 224)),
]
CAMERA_OBS_NAMES = ("table_cam", "wrist_cam", "front_camera", "left_camera", "right_camera")
CAMERA_SENSOR_NAMES = (*CAMERA_OBS_NAMES, "scene_cam", "ego_view_camera")


def _active_camera_names() -> set[str]:
    raw = str(os.environ.get("GEOMIND_ACTIVE_CAMERAS", "")).strip()
    if not raw or raw.lower() in {"all", "*"}:
        return set(CAMERA_SENSOR_NAMES)
    names = {name.strip() for name in raw.split(",") if name.strip()}
    return {name for name in names if name in set(CAMERA_SENSOR_NAMES)}


def _quat_from_rpy_deg(rpy_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (math.radians(v) for v in rpy_deg)
    quat = quat_from_euler_xyz(
        torch.tensor([roll], dtype=torch.float32),
        torch.tensor([pitch], dtype=torch.float32),
        torch.tensor([yaw], dtype=torch.float32),
    )[0]
    return tuple(quat.tolist())


def look_at_quat_ros(pos, target, up=(0.0, 0.0, 1.0)):
    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z = target - pos
    z = z / (np.linalg.norm(z) + 1e-12)

    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-12)

    y = np.cross(z, x)
    x = -x
    y = -y

    rotation = np.stack([x, y, z], axis=1).astype(np.float32)
    quat = quat_from_matrix(torch.from_numpy(rotation))
    return tuple(quat.numpy().tolist())


def _resolve_scene_table_top_height(scene_profile) -> float:
    if scene_profile is None or not scene_profile.table_name:
        return DEFAULT_TABLE_TOP_HEIGHT

    table_xform = scene_profile.xforms.get(scene_profile.table_name)
    if table_xform is None:
        resolved_height = asset_utils.resolve_tabletop_height(scene_profile.table_name)
        return DEFAULT_TABLE_TOP_HEIGHT if resolved_height is None else resolved_height

    resolved_height = asset_utils.resolve_tabletop_height(
        scene_profile.table_name,
        position_z=table_xform["pos"][2],
        scale_z=table_xform["scale"][2],
    )
    return DEFAULT_TABLE_TOP_HEIGHT if resolved_height is None else resolved_height


def _target_from_table_height(target_xy: tuple[float, float], table_top_height: float) -> tuple[float, float, float]:
    return (target_xy[0], target_xy[1], table_top_height + LOOK_AT_HEIGHT_ABOVE_TABLE)


def _camera_cfg_from_spec(spec: CameraSpec, table_top_height: float) -> CameraCfg:
    height, width = spec.resolution
    camera_pos = (
        spec.position_xy[0],
        spec.position_xy[1],
        table_top_height + spec.height_above_table,
    )
    look_at = _target_from_table_height(spec.look_at_xy, table_top_height)
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(
            pos=camera_pos,
            rot=look_at_quat_ros(camera_pos, target=look_at),
            convention="ros",
        ),
    )


@configclass
class VisuomotorObservationsCfg:
    """Policy observations with both proprioception and RGB inputs."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=geomind_env_cfg.stack_mdp.last_action)
        joint_pos = ObsTerm(func=geomind_env_cfg.stack_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=geomind_env_cfg.stack_mdp.joint_vel_rel)
        object = ObsTerm(
            func=geomind_env_cfg.geomind_mdp.object_obs,
            params={"object_cfgs": [], "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        eef_pos = ObsTerm(func=geomind_env_cfg.stack_mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=geomind_env_cfg.stack_mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=geomind_env_cfg.stack_mdp.gripper_pos)
        goal_error = ObsTerm(func=geomind_env_cfg.geomind_mdp.task_goal_xy_error, params={"goal_sequence": []})
        table_cam = ObsTerm(
            func=geomind_env_cfg.stack_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=geomind_env_cfg.stack_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            for spec in DEFAULT_CAMERAS:
                setattr(
                    self,
                    spec.name,
                    ObsTerm(
                        func=geomind_env_cfg.stack_mdp.image,
                        params={"sensor_cfg": SceneEntityCfg(spec.name), "data_type": "rgb", "normalize": False},
                    ),
                )
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class VisuomotorSkillgenObservationsCfg:
    """Visuomotor observations for SkillGen-style recording, including subtask signals."""

    @configclass
    class PolicyCfg(VisuomotorObservationsCfg.PolicyCfg):
        pass

    @configclass
    class SubtaskCfg(ObsGroup):
        success = ObsTerm(func=geomind_env_cfg.geomind_mdp.goal_sequence_reached, params={"goal_sequence": []})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class FrankaGeoMindIKRelVisuomotorEnvCfg(FrankaGeoMindEnvCfg):
    """Franka IK-Rel GeoMind config with visuomotor observations for VLA."""

    observations: VisuomotorObservationsCfg = VisuomotorObservationsCfg()
    task_mode = "vla"

    def __post_init__(self):
        super().__post_init__()
        table_top_height = _resolve_scene_table_top_height(self._scene_profile)
        active_cameras = _active_camera_names()

        if "wrist_cam" in active_cameras:
            self.scene.wrist_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
                update_period=0.0,
                height=224,
                width=224,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=20.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.13, 0.0, -0.15),
                    rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                    convention="ros",
                ),
            )
        cam_pos = (1.05, 0.0, table_top_height + TABLE_VIEW_CAMERA_HEIGHT_ABOVE_TABLE)
        look_at = _target_from_table_height((0.3, 0.0), table_top_height)
        quat = look_at_quat_ros(cam_pos, look_at)

        if "table_cam" in active_cameras:
            self.scene.table_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/table_cam",
                update_period=0.0,
                height=224,
                width=224,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=cam_pos,
                    rot=quat,
                    convention="ros",
                ),
            )

        cam_pos = (1.85, 0.0, 2.22)
        look_at = _target_from_table_height((0.3, 0.0), table_top_height)
        quat = look_at_quat_ros(cam_pos, look_at)

        if "scene_cam" in active_cameras:
            self.scene.scene_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/scene_cam",
                update_period=0.0,
                height=1024,
                width=1024,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=cam_pos,
                    rot=quat,
                    convention="ros",
                ),
            )


        ego_cam_pos = (0.12, 0.0, table_top_height + TABLE_VIEW_CAMERA_HEIGHT_ABOVE_TABLE)
        look_at = _target_from_table_height((0.35, 0.0), table_top_height)
        ego_quat = look_at_quat_ros(ego_cam_pos, look_at)

        if "ego_view_camera" in active_cameras:
            self.scene.ego_view_camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/ego_view_camera",
                update_period=0.0,
                height=224,
                width=224,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=10.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=ego_cam_pos,
                    rot=ego_quat,
                    convention="ros",
                ),
            )
        for spec in DEFAULT_CAMERAS:
            if spec.name in active_cameras:
                setattr(self.scene, spec.name, _camera_cfg_from_spec(spec, table_top_height))

        self.observations.policy.goal_error.params["goal_sequence"] = self.task_spec["goal_sequence"]
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        self.image_obs_list = [name for name in CAMERA_OBS_NAMES if name in active_cameras]
        for camera_name in CAMERA_OBS_NAMES:
            if camera_name not in active_cameras and hasattr(self.observations.policy, camera_name):
                delattr(self.observations.policy, camera_name)


@configclass
class FrankaGeoMindIKRelVisuomotorSkillgenEnvCfg(FrankaGeoMindIKRelVisuomotorEnvCfg):
    """SkillGen variant with subtask observations and task success signal."""

    observations: VisuomotorSkillgenObservationsCfg = VisuomotorSkillgenObservationsCfg()
    task_mode = "skillgen"

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.goal_error.params["goal_sequence"] = self.task_spec["goal_sequence"]
        self.observations.subtask_terms.success.params["goal_sequence"] = self.task_spec["goal_sequence"]
