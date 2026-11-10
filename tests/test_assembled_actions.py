"""CPU geometry/scoring checks, not robot physics executions."""
import math
import unittest

import numpy as np

from geomind.episodes.assembled_actions import (
    finger_close_target_m,grasp_tcp_height_offset_m,grasp_control_width,lift_sample_valid,object_geometry,placement_sample_valid,sample_metrics,
)


class AssembledActionsTests(unittest.TestCase):
    def test_rotated_square_closing_uses_face_width_without_rewriting_measured_width(self):
        cube=np.asarray([[x,y,z] for x in [-.0138,.0138] for y in [-.0138,.0138] for z in [-.0135,.0135]])
        plan=grasp_control_width(.03156,cube,'primitive_cuboid')
        self.assertAlmostEqual(plan['width_m'],.0276)
        self.assertEqual(plan['measured_world_axis_width_m'],.03156)
        rectangle=cube.copy();rectangle[:,0]*=2
        self.assertEqual(grasp_control_width(.03156,rectangle,'primitive_cuboid')['width_m'],.03156)
        self.assertEqual(grasp_control_width(.03156,cube,'primitive_cylinder')['width_m'],.03156)

    def test_pick_grasp_depth_respects_short_objects_and_preserves_place_grasp(self):
        self.assertEqual(grasp_tcp_height_offset_m('PickLiftSelectedObject',.02),0.)
        self.assertAlmostEqual(grasp_tcp_height_offset_m('PickLiftSelectedObject',.036),-.003)
        self.assertEqual(grasp_tcp_height_offset_m('PickLiftSelectedObject',.0698),-.008)
        for height in [.02,.036,.0698]:
            self.assertEqual(grasp_tcp_height_offset_m('PickPlaceObject',height),.008)
        for invalid in [0,-.01,float('nan'),float('inf'),True]:
            with self.subTest(height=invalid),self.assertRaises(ValueError):
                grasp_tcp_height_offset_m('PickLiftSelectedObject',invalid)

    def test_measured_width_keeps_nonzero_bounded_contact_preload(self):
        # Blue-can diameter ~51.2mm: preload is 1.5mm below the measured
        # half-width, rather than commanding the fingers completely shut.
        target=finger_close_target_m(.0512)
        self.assertGreater(target,.024)
        self.assertLess(target,.0245)
        self.assertLess(target,.0512/2)
        self.assertEqual(finger_close_target_m(.002),.001)
        self.assertEqual(finger_close_target_m(.2),.03)
        for invalid in [0,-.1,float('nan'),float('inf'),True]:
            with self.subTest(width=invalid),self.assertRaises(ValueError):
                finger_close_target_m(invalid)

    def fixture(self):
        # The origin is 3cm to one side of the true measured AABB centre.
        points=np.asarray([[x,y,z] for x in [.01,.05] for y in [-.02,.02] for z in [-.03,.03]])
        states=np.zeros((3,13));states[:,3]=1
        states[:,:3]=[[-.23,0,.63],[-.03,0,.63],[.17,0,.63]]
        initial=np.asarray([[-.2,0,.63],[0,0,.63],[.2,0,.63]])
        return points,states,initial,states[:,3:7].copy()

    def test_measured_geometry_is_not_the_rigid_origin(self):
        points,states,initial,_=self.fixture()
        centres,low,high=object_geometry(points,states)
        np.testing.assert_allclose(centres,initial,atol=1e-12)
        np.testing.assert_allclose(low[:,2],.6,atol=1e-12)
        self.assertAlmostEqual(high[1,0],.02)

    def test_lift_requires_actual_gt_object_motion_and_preserved_references(self):
        points,states,initial,quaternions=self.fixture()
        stationary=sample_metrics(points,states,initial,quaternions,1,.6,initial[1])
        self.assertFalse(lift_sample_valid(stationary,correct_object=True,fingers_closed=True,gripper_distance_m=0.))
        states[1,2]+=.15
        lifted=sample_metrics(points,states,initial,quaternions,1,.6,initial[1])
        self.assertTrue(lift_sample_valid(lifted,correct_object=True,fingers_closed=True,gripper_distance_m=.008))
        self.assertFalse(lift_sample_valid(lifted,correct_object=False,fingers_closed=True,gripper_distance_m=.008))
        states[0,1]+=.006
        disturbed=sample_metrics(points,states,initial,quaternions,1,.6,initial[1])
        self.assertFalse(lift_sample_valid(disturbed,correct_object=True,fingers_closed=True,gripper_distance_m=.008))

    def test_wrong_object_lift_does_not_pass_expected_object_score(self):
        points,states,initial,quaternions=self.fixture()
        states[2,2]+=.15
        metrics=sample_metrics(points,states,initial,quaternions,1,.6,initial[1])
        self.assertEqual(metrics['lift_height_m'],0.)
        self.assertAlmostEqual(metrics['reference_max_displacement_m'],.15)
        self.assertFalse(lift_sample_valid(metrics,correct_object=False,fingers_closed=True,gripper_distance_m=.008))

    def test_hold_requires_closed_fingers_proximity_and_object_velocity(self):
        points,states,initial,quaternions=self.fixture();states[1,2]+=.15
        metrics=sample_metrics(points,states,initial,quaternions,1,.6,initial[1])
        self.assertFalse(lift_sample_valid(metrics,correct_object=True,fingers_closed=False,gripper_distance_m=.008))
        self.assertFalse(lift_sample_valid(metrics,correct_object=True,fingers_closed=True,gripper_distance_m=.1))
        states[1,7]=.02
        moving=sample_metrics(points,states,initial,quaternions,1,.6,initial[1])
        self.assertFalse(lift_sample_valid(moving,correct_object=True,fingers_closed=True,gripper_distance_m=.008))

    def test_placement_requires_released_supported_object_at_frozen_target(self):
        points,states,initial,quaternions=self.fixture();target=[.07,.06,.63]
        states[1,:3]=[.04,.06,.63]
        placed=sample_metrics(points,states,initial,quaternions,1,.6,target)
        self.assertTrue(placement_sample_valid(placed,fingers_open=True,support_samples_passed=True))
        self.assertFalse(placement_sample_valid(placed,fingers_open=False,support_samples_passed=True))
        self.assertFalse(placement_sample_valid(placed,fingers_open=True,support_samples_passed=False))
        states[1,0]+=.08
        wrong=sample_metrics(points,states,initial,quaternions,1,.6,target)
        self.assertAlmostEqual(wrong['position_error_m'],.08)
        self.assertFalse(placement_sample_valid(wrong,fingers_open=True,support_samples_passed=True))

    def test_actual_rotation_changes_support_even_when_center_hits_target(self):
        points,states,initial,quaternions=self.fixture();target=[0,0,.63]
        # Rotate about Y by 90 degrees. The new half-height is 2cm, not 3cm.
        states[1,:3]=[0,0,.66];states[1,3:7]=[math.sqrt(.5),0,math.sqrt(.5),0]
        metrics=sample_metrics(points,states,initial,quaternions,1,.6,target)
        self.assertLess(metrics['position_error_m'],1e-12)
        self.assertAlmostEqual(metrics['support_height_error_m'],.01,places=12)
        self.assertFalse(placement_sample_valid(metrics,fingers_open=True,support_samples_passed=True))

    def test_nonfinite_geometry_and_bad_quaternions_are_rejected(self):
        points,states,_,_=self.fixture();states[0,3:7]=0
        with self.assertRaisesRegex(ValueError,'quaternion'):object_geometry(points,states)
        points,states,_,_=self.fixture();states[1,8]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):object_geometry(points,states)


if __name__=='__main__':unittest.main()
