import math
import unittest
from copy import deepcopy

from geomind.episodes.extended_geometry_tasks import TASK_IDS, build_spec, solve_spec, check_spec


class ExtendedGeometryTests(unittest.TestCase):
    def positions(self, spec):
        return {o['name']: [*o['pose']['xy'], .95285] for o in spec['runtime_objects']}

    def test_analytic_diagonal_projection_and_reflection(self):
        for task, expected in [('ProjectOntoLine_v0', [.424, .036]),
                               ('ReflectAcrossLine_v0', [.5155255323, -.0250170215])]:
            spec=build_spec(task); result=solve_spec(spec,self.positions(spec))
            for actual, value in zip(result['point_xy'],expected):
                self.assertAlmostEqual(actual,value,places=9)
            self.assertLess(abs(result['calculation']['perpendicular_dot_residual_m2']),1e-12)
            self.assertGreater(result['calculation']['minimum_reference_clearance_m'],.10)
            self.assertEqual(spec['protocol']['target_object']['object_id'],'green_cube')
            self.assertEqual(spec['protocol']['variant'],'TabletopPlane')
            self.assertTrue(check_spec(spec,self.positions(spec),result)['success'])
            wrong=deepcopy(result);wrong['point_xy'][0]+=.03
            self.assertFalse(check_spec(spec,self.positions(spec),wrong)['success'])

    def test_translation_rotation_and_identity_permutation_preserve_relation(self):
        for task in TASK_IDS:
            spec=build_spec(task);positions=self.positions(spec)
            original=solve_spec(spec,positions)['point_xy'];angle=.47;c,s=math.cos(angle),math.sin(angle)
            transform=lambda xy:[c*xy[0]-s*xy[1]+.13,s*xy[0]+c*xy[1]-.27]
            rotated={name:[*transform(xyz),xyz[2]] for name,xyz in reversed(list(positions.items()))}
            result=solve_spec(spec,rotated)['point_xy']
            for actual,expected in zip(result,transform(original)):self.assertAlmostEqual(actual,expected,places=12)

    def test_different_measurements_recompute_goal_and_do_not_use_authored_answer(self):
        spec=build_spec('ProjectOntoLine_v0');positions=self.positions(spec)
        original=solve_spec(spec,positions)['point_xy']
        positions['green_cube'][0]-=.048
        positions['green_cube'][1]-=.072
        actual=solve_spec(spec,positions)['point_xy']
        self.assertGreater(math.dist(original,actual),.02)

    def test_ambiguous_degenerate_or_overlapping_geometry_is_rejected(self):
        spec=build_spec('ProjectOntoLine_v0');positions=self.positions(spec)
        invalid=deepcopy(positions);invalid['blue_cube']=list(invalid['red_cube'])
        with self.assertRaisesRegex(ValueError,'Degenerate'):solve_spec(spec,invalid)
        invalid=deepcopy(positions);invalid['green_cube']=[.4,0.,.95]
        with self.assertRaisesRegex(ValueError,'too close'):solve_spec(spec,invalid)
        invalid=deepcopy(positions);invalid['green_cube']=[.1535899411,-.0090599608,.95]
        with self.assertRaisesRegex(ValueError,'overlaps'):solve_spec(spec,invalid)


    def test_authored_projection_is_off_midpoint_at_hand_calculated_t_point65(self):
        # A=(.32,-.12), B=(.48,.12); A+.65(B-A)=(.424,.036).
        # The green point is Q+(-.0915255323,.0610170215), a perpendicular offset.
        spec=build_spec('ProjectOntoLine_v0');positions=self.positions(spec)
        result=solve_spec(spec,positions)
        self.assertAlmostEqual(result['calculation']['line_parameter'],.65,places=9)
        midpoint=[.4,0.]
        self.assertGreater(math.dist(result['point_xy'],midpoint),.043)
        self.assertGreater(math.dist(result['point_xy'],midpoint),.01)
        self.assertFalse(check_spec(spec,positions,{'point_xy':midpoint})['success'])

    def test_midpoint_centred_point_symmetry_is_not_line_reflection(self):
        spec=build_spec('ReflectAcrossLine_v0');positions=self.positions(spec)
        # Independent literal 2*Q-P, with Q=(.424,.036).
        result=solve_spec(spec,positions)
        for actual,expected in zip(result['point_xy'],[.5155255323,-.0250170215]):
            self.assertAlmostEqual(actual,expected,places=9)
        wrong=[.8-positions['green_cube'][0],-positions['green_cube'][1]]
        self.assertGreater(math.dist(result['point_xy'],wrong),.086)
        self.assertFalse(check_spec(spec,positions,{'point_xy':wrong})['success'])

    def test_fixed_references_changed_source_requires_changed_both_geometry_goals(self):
        for task in TASK_IDS:
            spec=build_spec(task);positions=self.positions(spec)
            first=solve_spec(spec,positions)
            # Same endpoints, same perpendicular source offset; only along-axis
            # source location changes by (-.048,-.072), moving the foot to t=.35.
            # Both members satisfy the >=30mm midpoint-quality separation.
            alternate=deepcopy(positions)
            alternate['green_cube'][0]-=.048;alternate['green_cube'][1]-=.072
            second=solve_spec(spec,alternate)
            self.assertGreater(math.dist(first['point_xy'],second['point_xy']),.043)
            self.assertAlmostEqual(second['calculation']['line_parameter'],.35,places=9)
            self.assertAlmostEqual(first['calculation']['source_offset_m'],second['calculation']['source_offset_m'],places=10)
            self.assertFalse(check_spec(spec,alternate,first)['success'])
            self.assertTrue(check_spec(spec,alternate,second)['success'])

    def test_measured_midpoint_shortcut_geometry_is_rejected_by_solver_and_checker(self):
        for task in TASK_IDS:
            spec=build_spec(task);positions=self.positions(spec)
            positions['green_cube']=[.3084744677,.0610170215,.95285]
            with self.assertRaisesRegex(ValueError,'Midpoint shortcut'):
                solve_spec(spec,positions)
            mathematically_correct=([.4,0.] if task.startswith('Project') else
                                    [.4915255323,-.0610170215])
            checked=check_spec(spec,positions,{'point_xy':mathematically_correct})
            self.assertFalse(checked['success'])
            self.assertFalse(checked['source_position_necessity_screen']['passed'])

    def test_quality_margin_cannot_be_disabled_to_accept_midpoint_shortcut(self):
        spec=build_spec('ProjectOntoLine_v0');positions=self.positions(spec)
        spec['protocol']['gt_spec']['minimum_projection_midpoint_separation_m']=0.
        with self.assertRaisesRegex(ValueError,'30mm'):solve_spec(spec,positions)
        checked=check_spec(spec,positions,{'point_xy':[.424,.036]})
        self.assertFalse(checked['success'])
