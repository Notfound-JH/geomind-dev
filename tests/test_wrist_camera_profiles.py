import unittest

import numpy as np

from geomind.episodes.wrist_cameras import (
    SUPPORTED_WRIST_CAMERA_ROBOTS, validate_wrist_camera_parents, wrist_camera_specs,
)


def rotate(q, v):
    w = q[0];xyz = np.asarray(q[1:]);v = np.asarray(v)
    return v + 2 * w * np.cross(xyz, v) + 2 * np.cross(xyz, np.cross(xyz, v))


class WristCameraProfiles(unittest.TestCase):
    def test_all_native_robots_have_rigid_local_optical_mounts(self):
        self.assertEqual(set(SUPPORTED_WRIST_CAMERA_ROBOTS), {'franka','fr3','franka_robotiq','piper','piper_dual'})
        for robot in SUPPORTED_WRIST_CAMERA_ROBOTS:
            roots={'left':'/World/Robot'}
            if robot=='piper_dual':roots['right']='/World/RobotRight'
            specs=wrist_camera_specs(robot,roots)
            self.assertEqual(len(specs),len(roots))
            for spec in specs:
                with self.subTest(robot=robot,arm=spec['arm']):
                    q=spec['quaternion_wxyz_ros']
                    self.assertAlmostEqual(sum(x*x for x in q),1.,places=13)
                    direction=np.asarray(spec['optical_target_in_parent_m'])-spec['position_m']
                    direction/=np.linalg.norm(direction)
                    np.testing.assert_allclose(rotate(q,[0,0,1]),direction,atol=1e-12)
                    np.testing.assert_allclose(rotate(q,[1,0,0]),[0,-1 if robot=='piper_dual' else 1,0],atol=1e-12)
                    self.assertGreater(spec['optical_target_in_parent_m'][2],spec['tcp_in_parent_m'][2])
                    self.assertEqual(spec['prim_path'],spec['parent_prim_path']+'/wrist_cam')
                    self.assertTrue(spec['parent_prim_path'].startswith(roots[spec['arm']]+'/'))
                    self.assertGreater(spec['clipping_range_m'][0],0.)

    def test_dual_mounts_are_independent_and_not_selected_by_active_arm(self):
        result=wrist_camera_specs('piper_dual',{'right':'/World/R','left':'/World/L'})
        self.assertEqual([x['name'] for x in result],['wrist_left','wrist_right'])
        self.assertEqual([x['parent_prim_path'] for x in result],['/World/L/gripper_base','/World/R/gripper_base'])
        result[0]['position_m'][0]=99
        self.assertEqual(result[1]['position_m'][0],-.13)
        fresh=wrist_camera_specs('piper_dual',{'left':'/World/L','right':'/World/R'})
        self.assertEqual(fresh[0]['position_m'][0],-.13)

    def test_dual_inclined_tools_have_an_above_hand_upright_view(self):
        from geomind.episodes.robot_profiles import PIPER_DUAL_TOOL_QUATERNIONS
        specs=wrist_camera_specs('piper_dual',{'left':'/World/L','right':'/World/R'})
        for spec in specs:
            hand=PIPER_DUAL_TOOL_QUATERNIONS[spec['arm']]
            optical=spec['quaternion_wxyz_ros']
            position_delta=rotate(hand,spec['position_m'])
            world_forward=rotate(hand,rotate(optical,[0,0,1]))
            world_up=rotate(hand,rotate(optical,[0,-1,0]))
            with self.subTest(arm=spec['arm']):
                self.assertEqual(spec['profile_version'],2)
                self.assertGreater(position_delta[2],.17)
                self.assertLess(world_forward[2],-.59)
                self.assertGreater(world_up[2],.80)

    def test_single_arm_profiles_keep_the_existing_mount_and_version(self):
        for robot in ('franka','fr3','piper'):
            spec=wrist_camera_specs(robot,{'left':'/World/Robot'})[0]
            self.assertEqual(spec['position_m'],[.13,0.,-.15])
            self.assertEqual(spec['profile_version'],1)

    def test_robotiq_forward_side_mount_looks_past_fingertips(self):
        spec=wrist_camera_specs('franka_robotiq',{'left':'/World/Robot'})[0]
        self.assertEqual(spec['position_m'],[.13,0.,.02])
        self.assertEqual(spec['profile_version'],2)
        # Both down-facing grasp yaws put the camera ahead of the forearm.
        for tool in ([0.,1.,0.,0.],[0.,2**-.5,2**-.5,0.]):
            self.assertLess(rotate(tool,spec['position_m'])[2],0.)
        to_object=np.array(spec['tcp_in_parent_m'])-spec['position_m']
        optical=rotate(spec['quaternion_wxyz_ros'],[0.,0.,1.])
        angle=np.degrees(np.arccos(np.dot(to_object/np.linalg.norm(to_object),optical)))
        half_fov=np.degrees(np.arctan(spec['horizontal_aperture_mm']/(2*spec['focal_length_mm'])))
        self.assertLess(angle,half_fov-5.)

    def test_robotiq_uses_real_gripper_housing_not_empty_panda_hand(self):
        spec=wrist_camera_specs('franka_robotiq',{'left':'/World/Robot'})[0]
        self.assertEqual(spec['parent_prim_path'],'/World/Robot/Robotiq_2F_85_edit/Robotiq_2F_85/base_link')

    def test_unknown_or_incomplete_robot_roots_are_rejected(self):
        for robot,roots in [('unknown',{'left':'/World/Robot'}),('piper_dual',{'left':'/World/Robot'}),
                            ('franka',{'left':'/World/Robot','right':'/World/R'}),
                            ('piper_dual',{'left':'/World/Robot','right':'/World/Robot'}),
                            ('franka',{'left':'relative/Robot'}),('franka',{'left':'/World/../Robot'}),
                            ('franka',{'left':'/World/Robot/'}),('franka',{})]:
            with self.subTest(robot=robot,roots=roots),self.assertRaises(ValueError):wrist_camera_specs(robot,roots)

    def test_runtime_parent_validation_is_read_only_and_rejects_missing_or_nonrigid_parents(self):
        from pxr import Usd,UsdGeom,UsdPhysics
        stage=Usd.Stage.CreateInMemory();spec=wrist_camera_specs('franka',{'left':'/World/Robot'})
        with self.assertRaisesRegex(ValueError,'Missing wrist-camera'):validate_wrist_camera_parents(stage,spec)
        p=UsdGeom.Xform.Define(stage,spec[0]['parent_prim_path']).GetPrim()
        with self.assertRaisesRegex(ValueError,'not a rigid'):validate_wrist_camera_parents(stage,spec)
        UsdPhysics.RigidBodyAPI.Apply(p);before=stage.GetRootLayer().ExportToString()
        records=validate_wrist_camera_parents(stage,spec)
        self.assertEqual(records[0]['parent_prim_path'],str(p.GetPath()))
        self.assertEqual(stage.GetRootLayer().ExportToString(),before)
        p.SetInstanceable(True)
        # A prim without a reference is merely marked instanceable, not an
        # actual instance; the checker uses IsInstance/IsInstanceProxy instead.
        validate_wrist_camera_parents(stage,spec)


if __name__=='__main__':unittest.main()
