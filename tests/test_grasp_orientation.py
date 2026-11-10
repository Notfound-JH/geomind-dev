"""Synthetic CPU box geometry only; no claimed gripper calibration or PhysX."""
from copy import deepcopy
import json
import math
import unittest

import numpy as np

from geomind.scene_assembly.grasp_orientation import plan_grasp_orientation


def bounds(centres, sizes):
    centres=np.asarray(centres,dtype=float);half=np.asarray(sizes,dtype=float)*.5
    return centres,centres-half,centres+half


class GraspOrientationTests(unittest.TestCase):
    def test_canonical_preference_requires_the_explicit_measured_margin(self):
        centres,low,high=bounds([[0.,0.,.02],[0.,.1,.02]],[[.0276]*3]*2)
        args={'open_gripper_half_xy_m':[.01,.07]}
        self.assertEqual(plan_grasp_orientation(centres,low,high,0,**args)['yaw_degrees'],90)
        clear=plan_grasp_orientation(centres,low,high,0,**args,canonical_clearance_margin_m=.01)
        self.assertEqual(clear['yaw_degrees'],0)
        self.assertGreaterEqual(clear['minimum_reference_clearance_m'],.01)
        self.assertEqual(clear['canonical_clearance_margin_m'],.01)
        self.assertEqual(plan_grasp_orientation(centres,low,high,0,**args,
                         canonical_clearance_margin_m=.02)['yaw_degrees'],90)
        for invalid in [-.01,True,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):
                plan_grasp_orientation(centres,low,high,0,**args,canonical_clearance_margin_m=invalid)

    def test_adjacent_y_row_28mm_apart_27p6mm_objects_prefers_jaw_x(self):
        centres,low,high=bounds([[0.,0.,.02],[0.,.028,.02],[0.,-.028,.02]],[[.0276]*3]*3)
        originals=[array.copy() for array in (centres,low,high)]
        result=plan_grasp_orientation(centres,low,high,0,open_gripper_half_xy_m=[.01,.07])
        self.assertEqual(result['yaw_degrees'],90);self.assertEqual(result['grip_width_axis'],0)
        self.assertAlmostEqual(result['grip_width_m'],.0276)
        self.assertAlmostEqual(result['minimum_reference_clearance_m'],.0042)
        self.assertTrue(result['selected_has_positive_reference_clearance'])
        zero,ninety=result['candidate_orientations']
        self.assertEqual(zero['blocked_reference_indices'],[1,2]);self.assertLess(zero['minimum_reference_clearance_m'],0.)
        self.assertEqual(ninety['blocked_reference_indices'],[])
        np.testing.assert_allclose(result['tool_quat_wxyz'],[0.,math.sqrt(.5),math.sqrt(.5),0.],atol=1e-15)
        for current,original in zip((centres,low,high),originals):np.testing.assert_array_equal(current,original)
        json.dumps(result,allow_nan=False)

    def test_rectangular_target_uses_width_of_the_selected_jaw_axis(self):
        centres,low,high=bounds([[0.,0.,.02],[0.,.028,.02]],[[.06,.025,.04],[.0276]*3])
        result=plan_grasp_orientation(centres,low,high,0,open_gripper_half_xy_m=[.01,.07])
        self.assertEqual(result['grip_width_axis'],0);self.assertAlmostEqual(result['grip_width_m'],.06)
        centres,low,high=bounds([[0.,0.,.02],[.028,0.,.02]],[[.06,.025,.04],[.0276]*3])
        result=plan_grasp_orientation(centres,low,high,0,open_gripper_half_xy_m=[.01,.07])
        self.assertEqual(result['yaw_degrees'],0);self.assertEqual(result['grip_width_axis'],1)
        self.assertAlmostEqual(result['grip_width_m'],.025)

    def test_no_reference_and_equal_clearance_ties_prefer_yaw_zero(self):
        centres,low,high=bounds([[0.,0.,.02]],[[.06,.025,.04]])
        result=plan_grasp_orientation(centres,low,high,0,open_gripper_half_xy_m=[.01,.07])
        self.assertEqual(result['yaw_degrees'],0);self.assertIsNone(result['minimum_reference_clearance_m'])
        self.assertFalse(result['all_candidates_blocked']);self.assertEqual(result['tool_quat_wxyz'],[0.,1.,0.,0.])
        centres,low,high=bounds([[0.,0.,.02],[.2,.2,.02]],[[.03]*3]*2)
        result=plan_grasp_orientation(centres,low,high,0,open_gripper_half_xy_m=[.02,.02])
        self.assertEqual(result['yaw_degrees'],0)
        self.assertAlmostEqual(result['minimum_reference_clearance_m'],math.sqrt(2)*.165)

    def test_blocked_alternatives_remain_explicitly_blocked_without_a_safety_claim(self):
        centres,low,high=bounds([[0.,0.,.02],[.01,.01,.02]],[[.03]*3,[.06]*3])
        result=plan_grasp_orientation(centres,low,high,0,open_gripper_half_xy_m=[.01,.07])
        self.assertTrue(result['all_candidates_blocked']);self.assertFalse(result['selected_has_positive_reference_clearance'])
        self.assertLess(result['minimum_reference_clearance_m'],0.)
        self.assertTrue(any('contact-free certificate' in text for text in result['limitations']))

    def test_invalid_shapes_nonfinite_bounds_and_footprint_are_rejected(self):
        centres,low,high=bounds([[0.,0.,.02]],[[.03]*3])
        for malformed in ([[0.,0.]], [], [['0','0','.02']], [[0.,float('nan'),.02]]):
            with self.assertRaises(ValueError):plan_grasp_orientation(malformed,low,high,0,open_gripper_half_xy_m=[.01,.07])
        for footprint in ([0.,.07],[-.01,.07],[.01,float('inf')],[.01],['.01','.07']):
            with self.assertRaises(ValueError):plan_grasp_orientation(centres,low,high,0,open_gripper_half_xy_m=footprint)
        for index in (-1,1,True):
            with self.assertRaises(ValueError):plan_grasp_orientation(centres,low,high,index,open_gripper_half_xy_m=[.01,.07])
        with self.assertRaisesRegex(ValueError,'rectangular'):
            plan_grasp_orientation(centres,low,high,0,open_gripper_half_xy_m=[.01,.07],footprint_shape='unknown')
        with self.assertRaisesRegex(ValueError,'positive dimensions'):
            plan_grasp_orientation(centres,high,low,0,open_gripper_half_xy_m=[.01,.07])
        with self.assertRaisesRegex(ValueError,'midpoints'):
            plan_grasp_orientation(centres+.01,low,high,0,open_gripper_half_xy_m=[.01,.07])


if __name__=='__main__':unittest.main()
