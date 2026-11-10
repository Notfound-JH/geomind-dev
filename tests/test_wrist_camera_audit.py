"""Numerical camera mount evidence tests; no physical execution claim."""
from copy import deepcopy
import math
import unittest

from geomind.tools.verify_wrist_cameras import audit_dual_image_orientation,audit_frame_poses,pose_residuals


def pose():
    q=math.sqrt(.5)
    return {'position_w_m':[1.,2.1,3.], 'quaternion_wxyz_ros':[q,0.,0.,q],
            'parent_position_w_m':[1.,2.,3.], 'parent_quaternion_wxyz':[q,0.,0.,q],
            'mount_position_m':[.1,0.,0.], 'mount_quaternion_wxyz_ros':[1.,0.,0.,0.]}


class WristCameraAuditTests(unittest.TestCase):
    def test_known_quarter_turn_parent_maps_local_x_to_world_y(self):
        distance,angle=pose_residuals(pose())
        self.assertLess(distance,1e-12);self.assertLess(angle,1e-7)

    def test_quaternion_sign_ambiguity_is_equivalent(self):
        value=pose();value['quaternion_wxyz_ros']=[-x for x in value['quaternion_wxyz_ros']]
        self.assertLess(pose_residuals(value)[1],1e-7)

    def test_a_fixed_observer_cannot_pass_as_moving_wrist(self):
        first=pose();last=deepcopy(first);last['parent_position_w_m'][0]+=.1
        with self.assertRaisesRegex(ValueError,'does not follow'):
            audit_frame_poses([{'camera_pose':first},{'camera_pose':last}],active=True)

    def test_world_translation_with_fixed_mount_is_accepted(self):
        first=pose();last=deepcopy(first);last['parent_position_w_m'][0]+=.1;last['position_w_m'][0]+=.1
        result=audit_frame_poses([{'camera_pose':first},{'camera_pose':last}],active=True)
        self.assertAlmostEqual(result['max_camera_displacement_m'],.1)

    def test_idle_dual_arm_mount_need_not_move_but_active_arm_must(self):
        rows=[{'camera_pose':pose()},{'camera_pose':pose()}]
        self.assertTrue(audit_frame_poses(rows,active=False)['fixed_local_mount_verified'])
        with self.assertRaisesRegex(ValueError,'did not move'):
            audit_frame_poses(rows,active=True)

    def test_nonfinite_or_nonunit_recorded_poses_are_rejected(self):
        value=pose();value['position_w_m'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Invalid position'):pose_residuals(value)
        value=pose();value['parent_quaternion_wxyz']=[2.,0.,0.,0.]
        with self.assertRaisesRegex(ValueError,'Invalid quaternion'):pose_residuals(value)

    def test_dual_roll_audit_rejects_a_rigid_but_upside_down_camera(self):
        # Rotation about optical X: the two horizontal views have opposite
        # image-up; both could otherwise satisfy the fixed-mount audit.
        half=math.sqrt(.5)
        upright=[{'camera_pose':{'quaternion_wxyz_ros':[half,-half,0.,0.]}}]
        self.assertTrue(audit_dual_image_orientation(upright)['upright_image_verified'])
        inverted=[{'camera_pose':{'quaternion_wxyz_ros':[half,half,0.,0.]}}]
        with self.assertRaisesRegex(ValueError,'inverted'):
            audit_dual_image_orientation(inverted)

    def test_dual_roll_uses_projected_up_and_reports_unobservable_vertical_frames(self):
        # Identity points straight up, so gravity cannot identify its image roll.
        vertical={'camera_pose':{'quaternion_wxyz_ros':[1.,0.,0.,0.]}}
        with self.assertRaisesRegex(ValueError,'unobservable'):
            audit_dual_image_orientation([vertical])
        half=math.sqrt(.5)
        upright={'camera_pose':{'quaternion_wxyz_ros':[half,-half,0.,0.]}}
        result=audit_dual_image_orientation([vertical,upright])
        self.assertEqual(result['near_vertical_frames'],1)
        self.assertEqual(result['observable_frames'],1)
        self.assertAlmostEqual(result['min_projected_gravity_up_cosine'],1.)

    def test_steep_downward_pitch_is_not_confused_with_an_upside_down_image(self):
        angle=-math.pi+.2
        q=[math.cos(angle/2),math.sin(angle/2),0.,0.]
        result=audit_dual_image_orientation([{'camera_pose':{'quaternion_wxyz_ros':q}}])
        # At this pitch image-up has only a small Z component but zero roll.
        self.assertLess(result['min_image_up_world_z'],.2)
        self.assertAlmostEqual(result['min_projected_gravity_up_cosine'],1.)


if __name__=='__main__':unittest.main()
