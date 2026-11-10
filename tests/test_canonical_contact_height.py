"""Independent geometry checks for the proposed canonical-only height fix.

Run against the applied patch as a normal unittest, or use the companion
candidate runner without changing any repository runtime source.
"""
import math
import unittest

from geomind.episodes.gripper_profiles import gripper_profile


def independent_pad_bounds_for_width(width):
    # Literal measured pivots/mesh bounds from the already validated robot
    # asset; this does not call production gap-inverse/translation helpers.
    opening = .0870968475937844
    y, z = -.06776 - (-.0306), .09809 - .05466
    def gap(angle):
        inward = y*(math.cos(angle)-1.) + z*math.sin(angle)
        return opening - 2*inward
    lo, hi = 0., math.radians(47.)
    for _ in range(80):
        theta = (lo+hi)/2
        if gap(theta) > width: lo = theta
        else: hi = theta
    theta = (lo+hi)/2
    dz = -y*math.sin(theta) + z*(math.cos(theta)-1.)
    return .11041999608278298 + dz, .148420184850693 + dz


class CanonicalContactHeightTests(unittest.TestCase):
    def test_contact_pad_clearance_is_still_at_least_four_millimetres(self):
        profile = gripper_profile('franka_robotiq')
        for width in (.025, .04, .055, .075):
            for height in (.035, .04, .06, .10):
                with self.subTest(width=width, height=height):
                    offset = profile.grasp_tcp_height_offset_m('PickLiftSelectedObject', height,
                        width_m=width, asset_kind='primitive_cuboid', contact_angle_geometry=True)
                    pad_low, pad_high = independent_pad_bounds_for_width(width)
                    center_z = .14 + offset
                    table_clearance = height/2 + center_z - pad_high
                    overlap = min(pad_high, center_z+height/2) - max(pad_low, center_z-height/2)
                    self.assertGreaterEqual(table_clearance, .004-1e-12)
                    self.assertGreater(overlap, .025)
                    self.assertAlmostEqual(offset,
                        max((pad_low+pad_high)/2-.14, pad_high-.14-height/2+.004), places=12)

    def test_recorded_branch_height_correction_is_geometry_derived(self):
        profile = gripper_profile('franka_robotiq')
        height = .04015789434231867
        legacy = profile.grasp_tcp_height_offset_m('PickLiftSelectedObject', height,
            width_m=.04, asset_kind='primitive_cuboid')
        contact = profile.grasp_tcp_height_offset_m('PickLiftSelectedObject', height,
            width_m=.04, asset_kind='primitive_cuboid', contact_angle_geometry=True)
        self.assertAlmostEqual(contact-legacy, -.00034644422082513726, places=12)
        self.assertLess(contact, legacy)

    def test_legacy_default_profile_and_panda_commands_are_unchanged(self):
        robotiq = gripper_profile('franka_robotiq')
        for height in (.035, .04, .10):
            legacy = robotiq.grasp_tcp_height_offset_m('PickLiftSelectedObject', height,
                width_m=.04, asset_kind='primitive_cuboid')
            explicit = robotiq.grasp_tcp_height_offset_m('PickLiftSelectedObject', height,
                width_m=.04, asset_kind='primitive_cuboid', contact_angle_geometry=False)
            self.assertEqual(legacy, explicit)
        panda = gripper_profile('franka')
        self.assertEqual(panda.grasp_tcp_height_offset_m('PickPlaceObject', .04, width_m=.04), .008)
        self.assertAlmostEqual(panda.grasp_tcp_height_offset_m('PickLiftSelectedObject', .04, width_m=.04), -.005)

    def test_opt_in_is_restricted_to_robotiq_cuboids_and_boolean(self):
        profile = gripper_profile('franka_robotiq')
        with self.assertRaisesRegex(ValueError, 'scoped'):
            profile.grasp_tcp_height_offset_m('PickLiftSelectedObject', .04, width_m=.04,
                asset_kind='primitive_cylinder', contact_angle_geometry=True)
        with self.assertRaisesRegex(ValueError, 'scoped'):
            gripper_profile('franka').grasp_tcp_height_offset_m('PickLiftSelectedObject', .04,
                width_m=.04, asset_kind='primitive_cuboid', contact_angle_geometry=True)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            profile.grasp_tcp_height_offset_m('PickLiftSelectedObject', .04, width_m=.04,
                asset_kind='primitive_cuboid', contact_angle_geometry=1)


if __name__ == '__main__':
    unittest.main()
