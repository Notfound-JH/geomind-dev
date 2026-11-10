"""Asset-specific grasp commands and measured-state predicates (no simulator imports).

Robotiq commands are radians, Panda commands are metres. The 2F-85 opening
model is derived from the local USD's outer-knuckle pivot, inner-finger pivot
and pad meshes; it is kinematics, not an asserted contact-force calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


ROBOTIQ_OPEN_GAP_M = .0870968475937844
ROBOTIQ_OUTER_PIVOT_YZ_M = (-.0306, .05466)
ROBOTIQ_INNER_PIVOT_YZ_M = (-.06776, .09809)
ROBOTIQ_PAD_Z_BOUNDS_M = (.11041999608278298, .148420184850693)
ROBOTIQ_TCP_Z_M = .14
ROBOTIQ_CLOSED_LIMIT_RAD = math.radians(47.)


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be finite and positive')
    return float(value)


def robotiq_pad_translation_yz_m(angle_rad):
    """Parallel pad translation from its authored open pose, hand-local Y/Z."""
    angle = float(angle_rad)
    if not math.isfinite(angle) or not -.005 <= angle <= ROBOTIQ_CLOSED_LIMIT_RAD + .005:
        raise ValueError('Robotiq outer-knuckle angle is outside the local USD limits')
    y = ROBOTIQ_INNER_PIVOT_YZ_M[0] - ROBOTIQ_OUTER_PIVOT_YZ_M[0]
    z = ROBOTIQ_INNER_PIVOT_YZ_M[1] - ROBOTIQ_OUTER_PIVOT_YZ_M[1]
    return (y * (math.cos(angle) - 1.) + z * math.sin(angle),
            -y * math.sin(angle) + z * (math.cos(angle) - 1.))


def robotiq_gap_m(angle_rad):
    return max(0., ROBOTIQ_OPEN_GAP_M - 2. * robotiq_pad_translation_yz_m(angle_rad)[0])


def robotiq_angle_for_gap_rad(gap_m):
    """Invert the monotonic opening curve inside the actual 0..47 degree limit."""
    gap = _positive(gap_m, 'Robotiq requested gap')
    if gap > ROBOTIQ_OPEN_GAP_M:
        raise ValueError('Requested gap exceeds the measured Robotiq pad opening')
    lo, hi = 0., ROBOTIQ_CLOSED_LIMIT_RAD
    for _ in range(60):
        mid = (lo + hi) * .5
        if robotiq_gap_m(mid) > gap:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * .5


@dataclass(frozen=True)
class GripperProfile:
    embodiment: str
    command_joint_names: tuple[str, ...]
    measurement_joint_names: tuple[str, ...]
    joint_position_unit: str
    open_target: float
    max_object_width_m: float

    def _positions(self, positions):
        values = tuple(float(v) for v in positions)
        if len(values) != len(self.measurement_joint_names) or not all(math.isfinite(v) for v in values):
            raise ValueError('Expected one finite measured position per gripper measurement joint')
        return values

    def close_target(self, width_m):
        width = _positive(width_m, 'Measured gripper-axis width')
        if self.embodiment == 'franka':
            # Preserve the existing Panda contact preload and clipping.
            return max(.001, min(.03, width * .5 - .0015))
        if width > self.max_object_width_m:
            raise ValueError('Asset width leaves insufficient opening clearance for Robotiq')
        # A 1.5mm allowance per pad keeps a finite contact preload. This is a
        # position grasp heuristic; actual lift and release still decide success.
        return robotiq_angle_for_gap_rad(max(.001, width - .003))

    def hold_target_plan(self, width_m, mass_kg=.1, *, measured_outer_angle_rad=None,
                         drive_stiffness=17., aux_opening_torque_nm=None):
        """Estimate a post-lift joint target, without claiming measured force.

        The caller may refresh this model from measured outer-joint mean and
        inner-drive errors during a verified Pick lift's hold phase. Approach,
        initial grasp/lift and all placement stages retain ``close_target``.
        Actual lift, velocities, references, limits and dwell remain the only
        physical acceptance criteria. This method does not execute control.
        """
        close = self.close_target(width_m)
        if self.embodiment == 'franka':
            return {'kind': 'existing_panda_close', 'embodiment': self.embodiment,
                    'joint_position_unit': 'm', 'target': close, 'target_m': close,
                    'target_rad': None, 'force_model_applied': False,
                    'force_sensor_used': False, 'physical_hold_verified': False}
        mass = _positive(mass_kg, 'Held body mass')
        stiffness = _positive(drive_stiffness, 'Gripper drive stiffness')
        zero_contact = robotiq_angle_for_gap_rad(float(width_m))
        if measured_outer_angle_rad is None:
            contact = zero_contact
            contact_source = 'zero_geometric_preload_gap_inverse'
        else:
            value = measured_outer_angle_rad
            if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                    or not 0. <= value <= ROBOTIQ_CLOSED_LIMIT_RAD + .005):
                raise ValueError('Measured outer-joint mean must be finite and within hardware/measurement tolerance')
            contact = float(value)
            contact_source = 'supplied_measured_outer_joint_mean'
        lever_y = ROBOTIQ_INNER_PIVOT_YZ_M[0] - ROBOTIQ_OUTER_PIVOT_YZ_M[0]
        lever_z = ROBOTIQ_INNER_PIVOT_YZ_M[1] - ROBOTIQ_OUTER_PIVOT_YZ_M[1]
        jacobian = -lever_y * math.sin(contact) + lever_z * math.cos(contact)
        if not math.isfinite(jacobian) or jacobian <= 0:
            raise ValueError('Invalid inward-pad Jacobian at the supplied contact angle')
        if aux_opening_torque_nm is None:
            auxiliary = .4 * contact
            auxiliary_source = 'default_two_inner_joint_spring_estimate'
        else:
            value = aux_opening_torque_nm
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError('Auxiliary opening torque estimate must be finite and nonnegative')
            auxiliary = float(value)
            auxiliary_source = 'supplied_joint_spring_error_estimate'
        gravity = 9.81
        safety = 2.
        friction = .825
        normal_force = mass * gravity * safety / (2. * friction)
        contact_torque = 2. * normal_force * jacobian
        torque = auxiliary + contact_torque
        unconstrained = contact + torque / stiffness
        target = min(ROBOTIQ_CLOSED_LIMIT_RAD, max(0., unconstrained))
        return {'kind': 'robotiq_post_lift_static_joint_spring_model_v1',
                'embodiment': self.embodiment, 'joint_position_unit': 'rad',
                'target': target, 'target_rad': target,
                'width_input_m': float(width_m), 'mass_kg': mass,
                'contact_angle_rad': contact, 'contact_angle_source': contact_source,
                'zero_geometric_contact_angle_rad': zero_contact,
                'measured_outer_angle_rad': None if measured_outer_angle_rad is None else float(measured_outer_angle_rad),
                'pad_inward_jacobian_m_per_rad': jacobian,
                'estimated_normal_force_per_pad_n': normal_force,
                'aux_opening_torque_nm': auxiliary, 'aux_opening_torque_source': auxiliary_source,
                'drive_stiffness_nm_per_rad': stiffness,
                'estimated_contact_torque_nm': contact_torque, 'estimated_total_drive_torque_nm': torque,
                'unclamped_target_rad': unconstrained, 'hardware_clamped': target != unconstrained,
                'hardware_limits_rad': [0., ROBOTIQ_CLOSED_LIMIT_RAD],
                'original_close_target_rad': close, 'capped_by_original_close': False,
                'model_assumptions': {
                    'gravity_m_s2': gravity, 'static_load_safety_factor': safety,
                    'effective_static_friction_coefficient_assumed': friction,
                    'friction_assumption': 'average of object 1.15 and unbound-pad/default 0.5; caller must verify the actual materials',
                    'default_aux_opening_stiffness_nm_per_rad': .4,
                    'auxiliary_assumption': 'sum of two inner-joint spring biases with absolute unit mimic ratios',
                    'balance': 'tau = auxiliary_opening_torque + 2 * per_pad_normal_force * inward_pad_jacobian',
                    'scope': 'quasi-static spring/geometry estimate; ignores unmeasured compliance, dynamic contact and vertical pad-friction torque'},
                'force_model_applied': True, 'force_sensor_used': False,
                'physical_hold_verified': False}

    def hold_target(self, width_m, mass_kg=.1, *, measured_outer_angle_rad=None,
                    drive_stiffness=17., aux_opening_torque_nm=None):
        """Return the phase model's joint target in this profile's native unit."""
        return self.hold_target_plan(width_m, mass_kg,
            measured_outer_angle_rad=measured_outer_angle_rad, drive_stiffness=drive_stiffness,
            aux_opening_torque_nm=aux_opening_torque_nm)['target']

    def measured_gap_m(self, positions):
        q = self._positions(positions)
        if self.embodiment == 'franka':
            return sum(q)
        # Each outer knuckle determines one side's inward pad translation.
        return max(0., ROBOTIQ_OPEN_GAP_M - sum(robotiq_pad_translation_yz_m(v)[0] for v in q))

    def fingers_open(self, positions):
        q = self._positions(positions)
        if self.embodiment == 'franka':
            return all(v > .03 for v in q)
        return all(-.005 <= v <= .04 for v in q) and abs(q[0] - q[1]) <= .04

    def fingers_closed(self, positions, *, width_m):
        q = self._positions(positions)
        if self.embodiment == 'franka':
            return all(v < .038 for v in q)
        width = _positive(width_m, 'Measured gripper-axis width')
        return (all(.05 <= v <= ROBOTIQ_CLOSED_LIMIT_RAD + .005 for v in q)
                and abs(q[0] - q[1]) <= .08
                and self.measured_gap_m(q) <= width + .006)

    def grasp_tcp_height_offset_m(self, action, height_m, *, width_m, asset_kind=None,
                                  contact_angle_geometry=False):
        height = _positive(height_m, 'Measured object height')
        if type(contact_angle_geometry) is not bool:
            raise ValueError('contact_angle_geometry must be boolean')
        if contact_angle_geometry and (self.embodiment != 'franka_robotiq' or asset_kind != 'primitive_cuboid'):
            raise ValueError('Contact-angle height is scoped to Robotiq primitive cuboids')
        if action not in {'PickLiftSelectedObject', 'PickPlaceObject'}:
            raise ValueError('Unsupported assembled grasp action')
        if self.embodiment == 'franka':
            if action == 'PickPlaceObject':
                return .008
            return -min(.008, max(0., height * .5 - .015))
        preload_angle = self.close_target(width_m)
        # A rigid cuboid stops pad closure at its geometric width; the preload
        # position target is a spring command, not the attained pad pose. Keep
        # the original four-millimetre table clearance, using the contact pose.
        angle = robotiq_angle_for_gap_rad(width_m) if contact_angle_geometry else preload_angle
        dz = robotiq_pad_translation_yz_m(angle)[1]
        pad_low, pad_high = (v + dz for v in ROBOTIQ_PAD_Z_BOUNDS_M)
        # Centre the pad at the object. For short objects raise the TCP enough
        # that the distal pad edge stays 4mm above the supporting table.
        offset=max((pad_low + pad_high) * .5 - ROBOTIQ_TCP_Z_M,
                   pad_high - ROBOTIQ_TCP_Z_M - height * .5 + .004)
        if asset_kind=='primitive_cylinder':
            # A tall constant-width shaft otherwise hits the proximal plastic
            # linkage before the pads close. Pinch its upper sidewall, keeping
            # the top rim 4mm below the proximal edge of the closed pad.
            offset=max(offset,height*.5+pad_low-ROBOTIQ_TCP_Z_M+.004)
        return offset

    def as_dict(self):
        result = {'embodiment': self.embodiment,
                  'command_joint_names': list(self.command_joint_names),
                  'measurement_joint_names': list(self.measurement_joint_names),
                  'joint_position_unit': self.joint_position_unit,
                  'open_target': self.open_target, 'max_object_width_m': self.max_object_width_m,
                  'contact_preload_per_pad_m': .0015}
        if self.embodiment == 'franka_robotiq':
            result.update({'model': 'local_usd_parallel_linkage_v1',
                           'hold_target_model': {
                               'kind': 'post_lift_static_joint_spring_estimate',
                               'static_load_safety_factor': 2.,
                               'effective_static_friction_coefficient_assumed': .825,
                               'default_aux_opening_stiffness_nm_per_rad': .4,
                               'default_drive_stiffness_nm_per_rad': 17.,
                               'contact_base': 'supplied measured outer-joint mean, otherwise zero-preload gap inverse',
                               'target_limit': 'hardware only; not capped by original close_target',
                               'scope': 'Pick hold after verified lift only; close_target and all placement stages unchanged',
                               'force_sensor_used': False},
                           'outer_knuckle_limits_rad': [0., ROBOTIQ_CLOSED_LIMIT_RAD],
                           'open_pad_gap_m': ROBOTIQ_OPEN_GAP_M,
                           'outer_pivot_yz_m': list(ROBOTIQ_OUTER_PIVOT_YZ_M),
                           'inner_pivot_yz_m': list(ROBOTIQ_INNER_PIVOT_YZ_M),
                           'open_pad_z_bounds_m': list(ROBOTIQ_PAD_Z_BOUNDS_M),
                           'tcp_z_m': ROBOTIQ_TCP_Z_M,
                           'open_angle_max_rad': .04, 'closed_angle_min_rad': .05,
                           'closed_outer_coupling_tolerance_rad': .08,
                           'closed_gap_width_allowance_m': .006,
                           'mimic': {'joint': 'right_outer_knuckle_joint', 'reference': 'finger_joint',
                                     'usd_gearing': -1., 'expected_outer_position_ratio': 1.},
                           'width_model_scope': 'parallel inner-pad kinematics; not a contact-force calibration'})
        return result


def gripper_profile(embodiment):
    if embodiment == 'franka':
        joints = ('panda_finger_joint1', 'panda_finger_joint2')
        return GripperProfile(embodiment, joints, joints, 'm', .04, .075)
    if embodiment == 'franka_robotiq':
        return GripperProfile(embodiment, ('finger_joint',),
                              ('finger_joint', 'right_outer_knuckle_joint'), 'rad', 0., .080)
    raise ValueError(f'Unsupported assembled contact gripper: {embodiment}')
