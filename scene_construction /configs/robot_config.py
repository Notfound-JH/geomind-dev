from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Type

import numpy as np
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    load_supported_lula_kinematics_solver_config,
    load_supported_motion_policy_config,
)
from isaacsim.storage.native import get_assets_root_path

from siferbench.robots import FR3
from siferbench.controllers.fr3controller.fr3pickplacecontroller import FR3PickPlaceController

@dataclass(frozen=True)
class EndEffectorProbeConfig:
    prim_path_name: str
    name: str


@dataclass(frozen=True)
class GripperConfig:
    joint_names: Tuple[str, ...]
    opened_positions: np.ndarray
    closed_positions: np.ndarray
    action_deltas: np.ndarray
    prim_path_name: str


@dataclass(frozen=True)
class ControlGains:
    kps: np.ndarray
    kds: np.ndarray


@dataclass(frozen=True)
class MotionLimits:
    max_velocity: float
    max_acceleration: float
    max_effort: float

@dataclass(frozen=True)
class ControllerConfig: 
    controller_class: Type[Any]
    

@dataclass(frozen=True)
class RobotConfig:
    robot_class: Type[Any]
    name: str
    prim_path: str
    usd_path: str
    rmp_config: Mapping[str, Any]
    kinematics_config: Mapping[str, Any]
    end_effector_probe: EndEffectorProbeConfig
    gripper: GripperConfig
    default_gains: ControlGains
    default_params: MotionLimits
    controller: ControllerConfig
    def as_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for compatibility with legacy config callers."""
        return {
            "robot_class": self.robot_class,
            "robot_config": {
                "name": self.name,
                "prim_path": self.prim_path,
                "usd_path": self.usd_path,
                "rmp_config": self.rmp_config,
                "kinematics_config": self.kinematics_config,
                "end_effector_probe": asdict(self.end_effector_probe),
                "gripper": asdict(self.gripper),
            },
            "default_gains": asdict(self.default_gains),
            "default_params": asdict(self.default_params),
            "controller": asdict(self.controller),
        }


def _build_fr3_config(stage_units: float, is_dict: bool = True) -> Dict[str, Any]:
    assets_root = Path(get_assets_root_path())
    usd_path = assets_root / "Isaac" / "Robots" / "Franka" / "FR3" / "fr3.usd"
    robotconfig = RobotConfig(
            robot_class=FR3,
            controller=ControllerConfig(controller_class=FR3PickPlaceController),
            name="fr3",
            prim_path="/World/FR3",
            usd_path=str(usd_path),
            rmp_config=load_supported_motion_policy_config("FR3", "RMPflow"),
            kinematics_config=load_supported_lula_kinematics_solver_config("FR3"),
            end_effector_probe=EndEffectorProbeConfig(
                prim_path_name="fr3_hand_tcp",
                name="fr3_ee_probe",
            ),
            gripper=GripperConfig(
                joint_names=("fr3_finger_joint1", "fr3_finger_joint2"),
                opened_positions=np.array([0.04, 0.04]) / stage_units,
                closed_positions=np.array([0.01, 0.01]),
                action_deltas=np.array([0.01, 0.01]) / stage_units,
                prim_path_name="fr3_rightfinger",
            ),
            default_gains=ControlGains(
                kps=np.array([1e15] * 7),
                kds=np.array([1e13] * 7),
            ),
            default_params=MotionLimits(
                max_velocity=2.0,
                max_acceleration=5.0,
                max_effort=87.0,
            ),
        )
    if is_dict:
        return robotconfig.as_dict()
    else:
        return robotconfig


def build_robot_configs() -> Dict[str, Dict[str, Any]]:
    """Create the robots configuration dictionary."""
    stage_units = get_stage_units()
    return {"FR3": _build_fr3_config(stage_units)}


robots_config = build_robot_configs()


__all__ = [
    "ControlGains",
    "EndEffectorProbeConfig",
    "GripperConfig",
    "MotionLimits",
    "RobotConfig",
    "build_robot_configs",
    "robots_config",
]
