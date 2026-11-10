"""Temporal contract tests use synthetic measurements, never physical evidence."""
from copy import deepcopy
import unittest

import numpy as np

from geomind.episodes.extended_temporal_tasks import (
    TASK_IDS, build_spec, check_spec, draw_context, plan_spec, solve_spec,
)


def fixture(task_id):
    spec=build_spec(task_id)
    before={obj['name']:[*obj['pose']['xy'],.95] for obj in spec['runtime_objects']}
    plan=plan_spec(spec,before)
    after=deepcopy(before)
    for name,xy in plan['actual_movement']['waypoints'][-1]['positions_xy_m'].items():after[name][:2]=xy
    return spec,before,after,plan


class ExtendedTemporalTasksTests(unittest.TestCase):
    def test_all_three_specs_are_deterministic_natural_operations(self):
        for name in TASK_IDS:
            first=build_spec(name,seed=3);second=build_spec(name,seed=3)
            self.assertEqual(first,second)
            self.assertNotIn('报告',first['instruction'])
            self.assertEqual(first['protocol']['module_name'],'canonical_extension')
            self.assertEqual(first['extension_layout']['policy'],'rigid_translation')
            self.assertTrue(4<=len(first['runtime_objects'])<=6)

    def test_motion_commands_are_continuous_collision_screened_and_not_gt(self):
        for name in TASK_IDS:
            spec,before,_,plan=fixture(name)
            self.assertFalse(plan['planned_event_is_ground_truth'])
            movement=plan['actual_movement'];self.assertEqual(len(movement['waypoints']),41)
            self.assertEqual(movement['duration_s'],4.)
            self.assertGreater(movement['geometric_minimum_command_centre_distance_m'],.065)
            for first,last in zip(movement['waypoints'],movement['waypoints'][1:]):
                for identity in first['positions_xy_m']:
                    self.assertLess(np.linalg.norm(np.asarray(first['positions_xy_m'][identity])-last['positions_xy_m'][identity]),.012)
            if name.startswith('Track'):
                self.assertEqual(plan['mechanism'],'historical_identity_tracking')

    def test_displacement_selection_uses_measured_history_not_event_role(self):
        spec,before,_,_=fixture('DetectSceneChange_v0')
        # The plan moves item_a. This independent measured example moves item_c.
        after=deepcopy(before);after['item_c'][1]+=.12
        result=solve_spec(spec,after,before)
        self.assertEqual(result['selected'],'item_c')
        self.assertTrue(check_spec(spec,after,result,before)['success'])
        self.assertFalse(check_spec(spec,after,{'selected':'item_a'},before)['success'])

    def test_displacement_rejects_ambiguous_threshold_or_multiple_changes(self):
        spec,before,_,_=fixture('DetectSceneChange_v0')
        for delta in (.03,.032):
            after=deepcopy(before);after['item_a'][1]+=delta
            with self.assertRaisesRegex(ValueError,'ambiguous'):solve_spec(spec,after,before)
        after=deepcopy(before);after['item_a'][1]+=.10;after['item_b'][1]+=.10
        with self.assertRaisesRegex(ValueError,'Exactly one'):solve_spec(spec,after,before)

    def test_current_vacancy_is_measured_and_declares_history_boundary(self):
        spec,before,after,_=fixture('UpdateEmptySlot_v0')
        result=solve_spec(spec,after,before)
        # Slot1 was occupied before transfer to the initially empty slot3.
        np.testing.assert_allclose(result['point_xy'],[.42,-.10],atol=1e-12)
        self.assertEqual(result['calculation']['current_empty_slot'],1)
        self.assertEqual(result['calculation']['initial_empty_slot'],3)
        self.assertFalse(result['calculation']['history_necessary_for_answer'])
        self.assertTrue(check_spec(spec,after,result,before)['success'])
        self.assertFalse(check_spec(spec,after,{'point_xy':[.24,.10]},before)['success'])

    def test_vacancy_rejects_no_event_and_objects_between_slots(self):
        spec,before,after,_=fixture('UpdateEmptySlot_v0')
        with self.assertRaisesRegex(ValueError,'did not change'):solve_spec(spec,before,before)
        after['occupant_b'][1]-=.08
        with self.assertRaisesRegex(ValueError,'one measured slot'):solve_spec(spec,after,before)
        self.assertFalse(check_spec(spec,after,{'point_xy':[.42,-.10]},before)['success'])

    def test_vacancy_frame_is_anchored_to_measured_reference(self):
        spec,before,after,_=fixture('UpdateEmptySlot_v0')
        shift=np.asarray([.017,-.013])
        shifted=lambda state:{n:[*(np.asarray(p[:2])+shift),p[2]] for n,p in state.items()}
        result=solve_spec(spec,shifted(after),shifted(before))
        np.testing.assert_allclose(result['point_xy'],np.asarray([.42,-.10])+shift)
        self.assertTrue(check_spec(spec,shifted(after),result,shifted(before))['success'])

    def test_origin_uses_initial_region_instead_of_current_region(self):
        spec,before,after,_=fixture('TrackObjectOriginClutter_v0')
        result=solve_spec(spec,after,before)
        self.assertEqual(result['selected'],'item_a')
        self.assertEqual(result['calculation']['current_region_occupant'],'item_d')
        self.assertTrue(check_spec(spec,after,result,before)['success'])
        self.assertFalse(check_spec(spec,after,{'selected':'item_d'},before)['success'])

    def test_same_terminal_state_with_changed_history_changes_source_answer(self):
        spec,before,after,_=fixture('TrackObjectOriginClutter_v0')
        alternate=deepcopy(before);alternate['item_a'],alternate['item_b']=alternate['item_b'],alternate['item_a']
        first=solve_spec(spec,after,before);second=solve_spec(spec,after,alternate)
        self.assertEqual(first['selected'],'item_a');self.assertEqual(second['selected'],'item_b')
        self.assertTrue(check_spec(spec,after,second,alternate)['success'])
        self.assertFalse(check_spec(spec,after,first,alternate)['success'])

    def test_origin_rejects_no_physical_mix_or_ambiguous_initial_source(self):
        spec,before,after,_=fixture('TrackObjectOriginClutter_v0')
        with self.assertRaisesRegex(ValueError,'did not physically leave'):solve_spec(spec,before,before)
        before['item_b']=list(before['item_a'])
        with self.assertRaisesRegex(ValueError,'exactly one'):solve_spec(spec,after,before)

    def test_history_is_required_for_temporal_event_verification(self):
        for name in TASK_IDS:
            spec,_,after,_=fixture(name)
            with self.assertRaisesRegex(ValueError,'history'):solve_spec(spec,after)
            self.assertFalse(check_spec(spec,after,{},None)['success'])

    def test_context_frames_have_no_collision_or_rigid_geometry(self):
        try:
            from pxr import Usd,UsdPhysics
        except ImportError:
            self.skipTest('USD unavailable')
        for name,count in (('UpdateEmptySlot_v0',4),('TrackObjectOriginClutter_v0',1)):
            spec,before,_,_=fixture(name);stage=Usd.Stage.CreateInMemory()
            objects=[{'object_id':identity,'position_w_m':point} for identity,point in before.items()]
            paths=draw_context(stage,spec,objects,.93)
            self.assertEqual(len(paths),count)
            for path in paths:
                prim=stage.GetPrimAtPath(path);self.assertEqual(prim.GetTypeName(),'BasisCurves')
                self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
                self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))


if __name__=='__main__':unittest.main()
