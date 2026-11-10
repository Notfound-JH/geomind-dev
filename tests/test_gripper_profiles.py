"""CPU tests for asset geometry/units; these do not assert a physical grasp."""
import math
from pathlib import Path
import unittest

from geomind.episodes.gripper_profiles import (
    ROBOTIQ_CLOSED_LIMIT_RAD, ROBOTIQ_OPEN_GAP_M, ROBOTIQ_PAD_Z_BOUNDS_M,
    gripper_profile, robotiq_angle_for_gap_rad, robotiq_gap_m,
    robotiq_pad_translation_yz_m,
)


class GripperProfileTests(unittest.TestCase):
    def test_tall_cylinder_rim_stays_below_proximal_pad_with_sidewall_overlap(self):
        p=gripper_profile('franka_robotiq');height=.09;width=.052
        offset=p.grasp_tcp_height_offset_m('PickLiftSelectedObject',height,width_m=width,
                                           asset_kind='primitive_cylinder')
        dz=robotiq_pad_translation_yz_m(p.close_target(width))[1]
        pad_low,pad_high=[z+dz for z in ROBOTIQ_PAD_Z_BOUNDS_M]
        top_in_hand_z=.14+offset-height/2
        self.assertGreaterEqual(top_in_hand_z,pad_low+.004-1e-12)
        world_pad_bottom=offset+.14-pad_high
        world_pad_top=offset+.14-pad_low
        self.assertGreaterEqual(world_pad_bottom+height/2,.004)
        overlap=min(world_pad_top,height/2)-max(world_pad_bottom,-height/2)
        self.assertGreater(overlap,.02)
        self.assertGreater(offset,p.grasp_tcp_height_offset_m('PickLiftSelectedObject',height,width_m=width)+.02)
        panda=gripper_profile('franka')
        self.assertEqual(panda.grasp_tcp_height_offset_m('PickPlaceObject',height,width_m=width,
                                                       asset_kind='primitive_cylinder'),.008)

    def test_hold_model_virtual_work_aux_bias_mass_and_measured_contact_base(self):
        p=gripper_profile('franka_robotiq');width=.045;qc=.422;stiffness=17.;aux=.1688
        plan=p.hold_target_plan(width,mass_kg=.1,measured_outer_angle_rad=qc,
                                drive_stiffness=stiffness,aux_opening_torque_nm=aux)
        # Independent virtual-work estimate: one motor supplies both pads.
        jacobian=.03716*math.sin(qc)+.04343*math.cos(qc)
        normal=.1*9.81*2/(2*.825)
        expected=qc+(aux+2*normal*jacobian)/stiffness
        self.assertAlmostEqual(plan['target_rad'],expected,places=14)
        self.assertAlmostEqual(plan['estimated_normal_force_per_pad_n'],1.1890909090909092,places=12)
        self.assertAlmostEqual(plan['pad_inward_jacobian_m_per_rad'],jacobian,places=14)
        self.assertEqual(plan['contact_angle_rad'],qc)
        self.assertEqual(plan['contact_angle_source'],'supplied_measured_outer_joint_mean')
        self.assertFalse(plan['force_sensor_used']);self.assertFalse(plan['physical_hold_verified'])
        self.assertEqual(p.hold_target(width,.1,measured_outer_angle_rad=qc,
                         drive_stiffness=stiffness,aux_opening_torque_nm=aux),plan['target_rad'])

    def test_hold_model_fallback_hardware_clamp_and_original_close_are_separate(self):
        p=gripper_profile('franka_robotiq');width=.0512;plan=p.hold_target_plan(width)
        self.assertAlmostEqual(robotiq_gap_m(plan['contact_angle_rad']),width,places=12)
        self.assertAlmostEqual(plan['aux_opening_torque_nm'],.4*plan['contact_angle_rad'],places=14)
        self.assertAlmostEqual(p.close_target(width),.39269624346524823,places=14)
        # A measured narrower contact can require further closing than the
        # initial width heuristic. Only actual hardware limits may cap it.
        narrow=p.hold_target_plan(width,measured_outer_angle_rad=.7,aux_opening_torque_nm=0.)
        self.assertGreater(narrow['target_rad'],p.close_target(width))
        self.assertFalse(narrow['capped_by_original_close'])
        saturated=p.hold_target_plan(width,mass_kg=100.,measured_outer_angle_rad=.7)
        self.assertTrue(saturated['hardware_clamped'])
        self.assertEqual(saturated['target_rad'],ROBOTIQ_CLOSED_LIMIT_RAD)
        zero_aux=p.hold_target_plan(width,aux_opening_torque_nm=0.)
        self.assertEqual(zero_aux['aux_opening_torque_nm'],0.)
        panda=gripper_profile('franka')
        self.assertEqual(panda.hold_target(width),panda.close_target(width))
        self.assertFalse(panda.hold_target_plan(width)['force_model_applied'])
        self.assertNotIn('hold_target_model',panda.as_dict())

    def test_hold_model_rejects_invalid_measurements_and_parameters(self):
        p=gripper_profile('franka_robotiq')
        for kwargs in [{'mass_kg':0},{'mass_kg':True},{'mass_kg':float('nan')},
                       {'drive_stiffness':0},{'drive_stiffness':float('inf')},
                       {'aux_opening_torque_nm':-.01},{'aux_opening_torque_nm':True},
                       {'measured_outer_angle_rad':-.01},{'measured_outer_angle_rad':1.},
                       {'measured_outer_angle_rad':float('nan')}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                p.hold_target_plan(.045,**kwargs)

    def test_panda_keeps_existing_metre_targets_and_predicates(self):
        p = gripper_profile('franka')
        self.assertEqual(p.joint_position_unit, 'm')
        self.assertEqual(p.open_target, .04)
        self.assertAlmostEqual(p.close_target(.0512), .0241)
        self.assertEqual(p.close_target(.002), .001)
        self.assertEqual(p.close_target(.2), .03)
        self.assertTrue(p.fingers_open([.04, .04]))
        self.assertFalse(p.fingers_open([.04, .02]))
        self.assertTrue(p.fingers_closed([.025, .025], width_m=.0512))
        self.assertFalse(p.fingers_closed([.04, .025], width_m=.0512))
        self.assertEqual(p.grasp_tcp_height_offset_m('PickPlaceObject', .07, width_m=.05), .008)
        self.assertEqual(p.grasp_tcp_height_offset_m('PickLiftSelectedObject', .07, width_m=.05), -.008)

    def test_robotiq_command_is_one_angular_drive_and_two_measured_outer_joints(self):
        p = gripper_profile('franka_robotiq')
        self.assertEqual(p.command_joint_names, ('finger_joint',))
        self.assertEqual(p.measurement_joint_names, ('finger_joint', 'right_outer_knuckle_joint'))
        self.assertEqual(p.joint_position_unit, 'rad')
        self.assertEqual(p.open_target, 0.)
        self.assertAlmostEqual(ROBOTIQ_CLOSED_LIMIT_RAD, math.radians(47))
        self.assertEqual(p.as_dict()['mimic']['expected_outer_position_ratio'], 1.)

    def test_gap_curve_matches_known_linkage_displacement_and_is_invertible(self):
        # Independent right-triangle FK of the two authored pivot positions.
        # At 20 degrees the left inner pivot moves +17.09496mm in Y.
        self.assertAlmostEqual(robotiq_gap_m(math.radians(20)), .05290693352132553, places=12)
        self.assertAlmostEqual(robotiq_gap_m(0.), ROBOTIQ_OPEN_GAP_M, places=12)
        self.assertEqual(robotiq_gap_m(ROBOTIQ_CLOSED_LIMIT_RAD), 0.)
        previous = ROBOTIQ_OPEN_GAP_M
        for angle in [.1, .2, .4, .6, .8]:
            gap = robotiq_gap_m(angle)
            self.assertLess(gap, previous)
            self.assertAlmostEqual(robotiq_angle_for_gap_rad(gap), angle, places=12)
            previous = gap

    def test_blue_can_close_target_uses_measured_diameter_and_pad_centre(self):
        p = gripper_profile('franka_robotiq')
        q = p.close_target(.0512)
        self.assertGreater(q, math.radians(22))
        self.assertLess(q, math.radians(23))
        self.assertAlmostEqual(robotiq_gap_m(q), .0482, places=12)
        offset = p.grasp_tcp_height_offset_m('PickPlaceObject', .0698, width_m=.0512)
        self.assertLess(abs(offset), .001)
        dz = robotiq_pad_translation_yz_m(q)[1]
        # Down-facing hand: world pad centre = TCP + .14 - local pad centre.
        self.assertAlmostEqual(offset + .14 - (sum(ROBOTIQ_PAD_Z_BOUNDS_M) / 2 + dz), 0.)
        self.assertEqual(p.grasp_tcp_height_offset_m('PickLiftSelectedObject', .0698, width_m=.0512), offset)

    def test_short_object_target_keeps_distal_pad_above_support(self):
        p = gripper_profile('franka_robotiq')
        height = .02
        offset = p.grasp_tcp_height_offset_m('PickPlaceObject', height, width_m=.0512)
        dz = robotiq_pad_translation_yz_m(p.close_target(.0512))[1]
        distal_height_above_table = height / 2 + offset + .14 - (ROBOTIQ_PAD_Z_BOUNDS_M[1] + dz)
        self.assertAlmostEqual(distal_height_above_table, .004, places=12)

    def test_robotiq_predicates_check_both_actual_outer_joint_states(self):
        p = gripper_profile('franka_robotiq')
        self.assertTrue(p.fingers_open([0., 0.]))
        self.assertFalse(p.fingers_open([0., .2]))
        self.assertFalse(p.fingers_closed([.04, .04], width_m=.0512))
        q = robotiq_angle_for_gap_rad(.0512)
        self.assertTrue(p.fingers_closed([q, q], width_m=.0512))
        self.assertFalse(p.fingers_closed([q, q - .12], width_m=.0512))
        self.assertFalse(p.fingers_closed([.1, .1], width_m=.0512))
        self.assertAlmostEqual(p.measured_gap_m([q, q]), .0512, places=12)

    def test_invalid_measurements_and_unsupported_embodiments_are_rejected(self):
        p = gripper_profile('franka_robotiq')
        for value in [0., -.1, True, float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                p.close_target(value)
        with self.assertRaises(ValueError):
            p.close_target(.081)
        for q in [[0.], [0., float('nan')], [0., 0., 0.]]:
            with self.subTest(q=q), self.assertRaises(ValueError):
                p.fingers_open(q)
        with self.assertRaises(ValueError):
            gripper_profile('piper')

    @unittest.skipUnless(Path('/home/elm/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd').is_file(),
                         'Local Robotiq USD is not installed')
    def test_local_usd_geometry_limits_and_mimic_match_profile(self):
        from pxr import Gf, Usd, UsdGeom
        s = Usd.Stage.Open('/home/elm/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd')
        s.SetEditTarget(s.GetSessionLayer())
        s.GetDefaultPrim().GetVariantSet('Gripper').SetVariantSelection('Robotiq_2F_85')
        root = '/panda/Robotiq_2F_85_edit/Robotiq_2F_85/'
        joint = s.GetPrimAtPath(root + 'Joints/finger_joint')
        self.assertEqual(joint.GetAttribute('physics:lowerLimit').Get(), 0.)
        self.assertEqual(joint.GetAttribute('physics:upperLimit').Get(), 47.)
        mimic = s.GetPrimAtPath(root + 'Joints/right_outer_knuckle_joint')
        self.assertEqual(mimic.GetAttribute('physxMimicJoint:rotZ:gearing').Get(), -1.)
        self.assertEqual(str(mimic.GetRelationship('physxMimicJoint:rotZ:referenceJoint').GetTargets()[0]), str(joint.GetPath()))
        axis = Gf.Rotation(joint.GetAttribute('physics:localRot0').Get()).TransformDir(Gf.Vec3d(0., 0., 1.))
        self.assertAlmostEqual(axis[0], -1., places=6)
        pivot = joint.GetAttribute('physics:localPos0').Get()
        inner = s.GetPrimAtPath(root + 'Joints/left_inner_finger_joint').GetAttribute('physics:localPos0').Get()
        self.assertAlmostEqual(pivot[1], -.0306, places=7)
        self.assertAlmostEqual(pivot[2], .05466, places=7)
        self.assertAlmostEqual(inner[1], -.06776, places=7)
        self.assertAlmostEqual(inner[2], .09809, places=7)
        cache = UsdGeom.XformCache()
        hand_inverse = cache.GetLocalToWorldTransform(s.GetPrimAtPath('/panda/panda_hand')).GetInverse()
        pad_points = []
        for prim in Usd.PrimRange.Stage(s, Usd.TraverseInstanceProxies()):
            if (prim.IsA(UsdGeom.Mesh) and '/left_inner_finger/' in str(prim.GetPath())
                    and 'fingertipsstep' in prim.GetName()):
                transform = cache.GetLocalToWorldTransform(prim) * hand_inverse
                pad_points.extend(transform.Transform(Gf.Vec3d(v)) for v in UsdGeom.Mesh(prim).GetPointsAttr().Get())
        self.assertGreater(len(pad_points), 0)
        self.assertAlmostEqual(-2 * max(v[1] for v in pad_points), ROBOTIQ_OPEN_GAP_M, places=10)
        self.assertAlmostEqual(min(v[2] for v in pad_points), ROBOTIQ_PAD_Z_BOUNDS_M[0], places=10)
        self.assertAlmostEqual(max(v[2] for v in pad_points), ROBOTIQ_PAD_Z_BOUNDS_M[1], places=10)


if __name__ == '__main__':
    unittest.main()
