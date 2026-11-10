"""Opt-in must preserve validated grasp targets and reject unintended scope."""
from copy import deepcopy
import unittest

from geomind.episodes.assembled_actions import post_lift_grip_policy
from geomind.episodes.gripper_profiles import gripper_profile


class CanonicalHoldPolicyTests(unittest.TestCase):
    def test_existing_recipes_do_not_opt_in_or_change_their_targets(self):
        for protocol in ({}, {'module_name':'canonical_extension'}, {'module_name':'anchor_reference'},
                         {'module_name':'canonical_extension','grip_hold_policy':'mass_spring_estimate'}):
            original = deepcopy(protocol)
            for robot in ('franka','franka_robotiq'):
                result = post_lift_grip_policy(protocol,robot_name=robot,asset_kind='primitive_cuboid',
                    action='PickLiftSelectedObject',initial_close_target=.49423809443639255)
                self.assertEqual(result['policy'],'mass_spring_estimate')
                self.assertIsNone(result['target'])
                self.assertFalse(result['explicit_opt_in'])
                self.assertEqual(protocol,original)

    def test_opt_in_reuses_original_width_specific_close_commands_exactly(self):
        p = gripper_profile('franka_robotiq')
        protocol = {'module_name':'canonical_extension','grip_hold_policy':'retain_initial_close_target'}
        for width in (.025,.04,.055,.075):
            close = p.close_target(width)
            result = post_lift_grip_policy(protocol,robot_name='franka_robotiq',asset_kind='primitive_cuboid',
                action='PickLiftSelectedObject',initial_close_target=close)
            self.assertEqual(result['target'],close)
            self.assertEqual(result['target_rad'],close)
            self.assertTrue(result['explicit_opt_in'])
            self.assertFalse(result['force_model_applied'])
            self.assertFalse(result['direct_force_command'])
            self.assertFalse(result['gripper_stiffness_changed'])
            self.assertFalse(result['physical_hold_verified'])

    def test_requesting_unimplemented_scope_fails_instead_of_silent_fallback(self):
        p = {'module_name':'canonical_extension','grip_hold_policy':'retain_initial_close_target'}
        cases=[(dict(p,module_name='anchor_reference'),'franka_robotiq','primitive_cuboid','PickLiftSelectedObject'),
               (p,'franka','primitive_cuboid','PickLiftSelectedObject'),
               (p,'franka_robotiq','primitive_cylinder','PickLiftSelectedObject'),
               (p,'franka_robotiq','primitive_cuboid','PickPlaceObject')]
        for protocol,robot,kind,action in cases:
            with self.subTest(robot=robot,kind=kind,action=action),self.assertRaisesRegex(ValueError,'scoped'):
                post_lift_grip_policy(protocol,robot_name=robot,asset_kind=kind,action=action,
                    initial_close_target=.49423809443639255)

    def test_unknown_policies_and_invalid_original_commands_are_rejected(self):
        for value in ('force_hold',False,[],None):
            with self.subTest(policy=value),self.assertRaisesRegex(ValueError,'Unknown'):
                post_lift_grip_policy({'module_name':'canonical_extension','grip_hold_policy':value},
                    robot_name='franka_robotiq',asset_kind='primitive_cuboid',action='PickLiftSelectedObject',
                    initial_close_target=.49423809443639255)
        for value in (False,-.1,float('nan'),float('inf')):
            with self.subTest(target=value),self.assertRaisesRegex(ValueError,'positive'):
                post_lift_grip_policy({'module_name':'canonical_extension','grip_hold_policy':'retain_initial_close_target'},
                    robot_name='franka_robotiq',asset_kind='primitive_cuboid',action='PickLiftSelectedObject',
                    initial_close_target=value)


if __name__ == '__main__':
    unittest.main()
