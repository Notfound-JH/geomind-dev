"""Rigid wrist-camera mounts for every native robot, with lazy USD validation.

Positions and quaternions are relative to the named rigid hand body. They are
never world poses or target-tracking extrinsics. Cameras have no collision or
mass geometry. These are simulator viewing profiles, not measured hardware
camera-to-hand calibrations.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re


_PROFILES = {
    'franka': {'parent_link': 'panda_hand', 'tcp_z_m': .107},
    'fr3': {'parent_link': 'fr3_hand', 'tcp_z_m': .1034},
    'franka_robotiq': {
        'parent_link': 'Robotiq_2F_85_edit/Robotiq_2F_85/base_link',
        'tcp_z_m': .14,
    },
    'piper': {'parent_link': 'gripper_base', 'tcp_z_m': .138},
    'piper_dual': {'parent_link': 'gripper_base', 'tcp_z_m': .138},
}

SUPPORTED_WRIST_CAMERA_ROBOTS = tuple(_PROFILES)


def _root_path(value):
    if not isinstance(value, str) or not re.fullmatch(r'(?:/[A-Za-z_][A-Za-z0-9_]*)+', value):
        raise ValueError('Robot root must be an absolute USD prim path without variants or traversal')
    return value


def _optical_quaternion(position, target_z):
    """ROS +Z looks past the fingertips; ROS +X follows hand-local +Y.

    The camera is on hand-local +X, so the view tilts toward the hand centre.
    The +90 degree optical roll presents the local-Y finger opening laterally.
    """
    angle = math.atan2(position[0], target_z - position[2])
    c = math.cos(angle * .5) / math.sqrt(2.)
    s = -math.sin(angle * .5) / math.sqrt(2.)
    return [c, s, s, c]


def wrist_camera_specs(robot_name, robot_roots):
    """Return JSON-safe local camera specifications for all of a robot's arms.

    ``robot_roots`` is ``{'left': '/World/Robot'}`` for a single arm, or also
    includes ``'right': '/World/RobotRight'`` for ``piper_dual``. Both dual arms
    always receive a camera, independently of the action's active-arm choice.
    Runtime must create the camera beneath ``parent_prim_path`` and retain the
    supplied local offset (``CameraCfg.OffsetCfg(..., convention='ros')``).
    """
    if robot_name not in _PROFILES:
        raise ValueError(f'Unsupported wrist-camera robot: {robot_name}')
    arms = ('left', 'right') if robot_name == 'piper_dual' else ('left',)
    if not isinstance(robot_roots, Mapping) or set(robot_roots) != set(arms):
        raise ValueError(f'{robot_name} requires exactly these robot roots: {arms}')
    roots = {arm: _root_path(robot_roots[arm]) for arm in arms}
    if len(set(roots.values())) != len(arms):
        raise ValueError('Each arm needs a distinct robot root')
    profile = _PROFILES[robot_name]
    dual = robot_name == 'piper_dual'
    # The dual tool pose approaches 20 degrees below horizontal. Its local +X
    # points downward, unlike the single-arm down-facing poses. Mount on -X to
    # see over the hand and use the opposite optical roll to keep image-up up.
    # The Robotiq rear offset puts the camera behind the forearm for a
    # 90-degree top grasp. A fixed forward side mount keeps the same rigid
    # hand child and optical roll while avoiding that self-occlusion.
    position = [-.13 if dual else .13, 0., .02 if robot_name == 'franka_robotiq' else -.15]
    target_z = profile['tcp_z_m'] + .15
    quaternion = _optical_quaternion(position, target_z)
    if dual:
        w, x, y, z = quaternion
        # Fixed 180-degree optical roll, canonical positive-w quaternion.
        # ROS +Z and the physical mount stay unchanged by this roll.
        quaternion = [z, -y, x, -w]
    result = []
    for arm in arms:
        parent = roots[arm] + '/' + profile['parent_link']
        result.append({
            'name': 'wrist_' + arm if len(arms) == 2 else 'wrist',
            'arm': arm,
            'robot_name': robot_name,
            'robot_root_prim_path': roots[arm],
            'parent_prim_path': parent,
            'prim_path': parent + '/wrist_cam',
            'position_m': list(position),
            'quaternion_wxyz_ros': list(quaternion),
            'optical_target_in_parent_m': [0., 0., target_z],
            'tcp_in_parent_m': [0., 0., profile['tcp_z_m']],
            'focal_length_mm': 16.,
            'horizontal_aperture_mm': 24.,
            'clipping_range_m': [.015, 20.],
            'convention': 'ros',
            'calibration_scope': 'Authored simulator hand-local mount; not a measured hardware extrinsic calibration.',
            'mount_policy': 'Rigid child of the actual hand/gripper body; no world-pose tracking or physics geometry.',
            'profile_version': 2 if dual or robot_name == 'franka_robotiq' else 1,
        })
    return deepcopy(result)


def validate_wrist_camera_parents(stage, specs):
    """Fail loudly when an asset's real rigid hand topology changed.

    This read-only check is optional for CPU callers and intended before runtime
    spawning. USD is imported here only; importing profiles never starts Isaac.
    """
    from pxr import UsdGeom, UsdPhysics

    records = []
    for spec in specs:
        parent = stage.GetPrimAtPath(spec['parent_prim_path'])
        if not parent.IsValid() or not parent.IsA(UsdGeom.Xformable):
            raise ValueError(f"Missing wrist-camera hand prim: {spec['parent_prim_path']}")
        if not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Wrist-camera parent is not a rigid hand body: {parent.GetPath()}")
        if parent.IsInstanceProxy() or parent.IsInstance():
            raise ValueError(f"Wrist-camera parent cannot receive a local camera child: {parent.GetPath()}")
        records.append({'name': spec['name'], 'parent_prim_path': str(parent.GetPath()),
                        'rigid_body': True, 'instance_proxy': False})
    return records
