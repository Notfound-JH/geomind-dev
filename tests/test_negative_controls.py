"""CPU planning tests; no robot execution or negative-control result is claimed."""
from copy import deepcopy
import json
import unittest

import numpy as np

from geomind.scene_assembly.negative_controls import (
    NegativeControlPlanningError, choose_wrong_placement_point, plan_wrong_placement_point,
)
from geomind.scene_assembly.usd import supported_xy


def box(size):
    return np.asarray([[x,y,z] for x in [-size[0]/2,size[0]/2]
                       for y in [-size[1]/2,size[1]/2] for z in [-size[2]/2,size[2]/2]])


def table(half=.5):
    return {'height_m': .6, 'surface_triangles': np.asarray([
        [[-half,-half,.6],[half,-half,.6],[half,half,.6]],
        [[-half,-half,.6],[half,half,.6],[-half,half,.6]],
    ])}


def workspace():
    return {'frame':'world','bounds_m':{'x':[-.5,.5],'y':[-.5,.5],'z':[.5,1.2]}}


class NegativeControlTests(unittest.TestCase):
    def test_explicit_offsets_first_uses_valid_offset_and_preserves_gt_and_states(self):
        states=self.states()[:1];before=states.copy();gt=self.gt();gt_before=deepcopy(gt)
        plan=plan_wrong_placement_point(gt,table(),box([.045]*3),states,0,workspace(),candidate_order='offsets_first')
        self.assertEqual(plan['policy']['candidate_order'],'offsets_first')
        np.testing.assert_allclose(plan['point_m'],[.08,0.,.6225])
        self.assertEqual(plan['candidates'][-1]['label'],'return_initial_supported_position')
        self.assertTrue(plan['candidates'][0]['support_9_samples_passed'])
        self.assertEqual(gt,gt_before);np.testing.assert_array_equal(states,before)

    def test_offsets_first_still_rejects_reference_collisions(self):
        plan=plan_wrong_placement_point(self.gt(),table(),box([.045]*3),self.states(),0,workspace(),candidate_order='offsets_first')
        self.assertIn('retained_object_clearance',self.offset_record(plan,[.08,0.])['rejection_reasons'])
        self.assertTrue(plan['candidates'][plan['selected_candidate_index']]['accepted'])
        with self.assertRaisesRegex(ValueError,'Unknown negative placement'):
            plan_wrong_placement_point(self.gt(),table(),box([.045]*3),self.states(),0,workspace(),candidate_order='random')

    def offset_record(self,plan,xy):
        return next(record for record in plan['candidates']
                    if record['source']=='frozen_ground_truth' and np.allclose(record['offset_xy_m'],xy,atol=1e-12,rtol=0))

    def states(self):
        states=np.zeros((3,13));states[:,3]=1.
        states[:,:3]=[[0.,-.18,.6225],[-.08,0.,.6225],[.08,0.,.6225]]
        return states

    def gt(self,z=.6225):
        return {'type':'Point','position_m':[0.,0.,z],'frame':'world'}

    def test_midpoint_old_wrong_point_hits_reference_and_supported_empty_alternative_is_recorded(self):
        states=self.states();plan=plan_wrong_placement_point(self.gt(),table(),box([.045]*3),states,0,workspace())
        old_wrong=self.offset_record(plan,[.08,0.])
        self.assertIn('retained_object_clearance',old_wrong['rejection_reasons'])
        self.assertIn(2,old_wrong['blocked_body_indices'])
        chosen=plan['point_m']
        self.assertGreaterEqual(np.linalg.norm(np.asarray(chosen)-self.gt()['position_m']),.05)
        self.assertEqual(sum(record['selected'] for record in plan['candidates']),1)
        self.assertGreater(len(plan['candidates']),20)
        self.assertTrue(all('rejection_reasons' in record for record in plan['candidates']))
        selected=plan['candidates'][plan['selected_candidate_index']]
        self.assertTrue(selected['support_9_samples_passed'])
        self.assertGreaterEqual(selected['minimum_retained_object_aabb_gap_m'],.06-1e-9)
        json.dumps(plan,allow_nan=False)

    def test_topology_wrong_point_rests_on_table_instead_of_original_elevated_support(self):
        states=self.states();clouds=[box([.04,.04,.06]),box([.10,.10,.10]),box([.04]*3)]
        plan=plan_wrong_placement_point(self.gt(z=.85),table(),clouds,states,0,workspace())
        self.assertAlmostEqual(plan['point_m'][2],.63)
        self.assertNotEqual(plan['point_m'][2],.85)

    def test_relation_representative_point_is_supported_but_full_contract_is_not_interpreted(self):
        relation=dict(self.gt(z=.85),type='Relation',relation='containment_placement',
                      contract={'private_test_marker':'caller validates full contract'})
        plan=plan_wrong_placement_point(relation,table(),box([.04]*3),self.states(),0,workspace())
        self.assertEqual(plan['frozen_ground_truth'],relation)
        self.assertEqual(plan['policy']['source_gt_type'],'Relation')
        self.assertAlmostEqual(plan['point_m'][2],.62)
        for invalid_type in ['Object','Other','relation']:
            invalid=dict(relation,type=invalid_type)
            with self.subTest(kind=invalid_type),self.assertRaisesRegex(ValueError,'Point or Relation'):
                choose_wrong_placement_point(invalid,table(),box([.04]*3),self.states(),0,workspace())

    def test_all_nine_footprint_samples_are_required_at_curved_or_cut_table_edge(self):
        surface={'height_m':.6,'surface_triangles':np.asarray([[[-.3,-.3,.6],[.3,-.3,.6],[-.3,.3,.6]]])}
        gt=self.gt();gt['position_m']=[-.08,-.01,.62]
        states=self.states()[:1];states[0,:3]=[-.1,-.1,.62]
        self.assertTrue(supported_xy(surface,[0.,-.01],[0.,0.]))
        plan=plan_wrong_placement_point(gt,surface,box([.04]*3),states,0,workspace())
        first_offset=self.offset_record(plan,[.08,0.])
        self.assertFalse(first_offset['support_9_samples_passed'])
        self.assertIn('footprint_lacks_table_support',first_offset['rejection_reasons'])

    def test_full_bbox_must_fit_workspace_even_when_candidate_centre_is_inside(self):
        bounds=workspace();bounds['bounds_m']['x']=[-.1,.1]
        states=self.states()[:1]
        plan=plan_wrong_placement_point(self.gt(),table(),box([.06,.04,.04]),states,0,bounds)
        self.assertFalse(self.offset_record(plan,[.08,0.])['full_body_in_workspace'])
        self.assertFalse(self.offset_record(plan,[-.08,0.])['full_body_in_workspace'])
        self.assertTrue(self.offset_record(plan,[0.,.08])['full_body_in_workspace'])

    def test_retained_fixture_aabb_is_checked_when_supplied(self):
        surface=table();surface['retained_obstacles']=[{'min':[.06,-.02,.6],'max':[.10,.02,.9]}]
        plan=plan_wrong_placement_point(self.gt(),surface,box([.04]*3),self.states()[:1],0,workspace())
        self.assertIn('retained_fixture_clearance',self.offset_record(plan,[.08,0.])['rejection_reasons'])
        self.assertEqual(plan['policy']['retained_fixture_count'],1)

    def test_no_available_point_raises_with_all_candidates_preserved(self):
        with self.assertRaises(NegativeControlPlanningError) as raised:
            choose_wrong_placement_point(self.gt(),table(.03),box([.04]*3),self.states()[:1],0,workspace())
        plan=raised.exception.diagnostics
        self.assertIsNone(plan['point_m'])
        self.assertGreater(len(plan['candidates']),20)
        self.assertTrue(all(not candidate['accepted'] for candidate in plan['candidates']))
        json.dumps(plan,allow_nan=False)

    def test_planning_is_deterministic_and_does_not_modify_query_state_or_gt(self):
        gt=self.gt();states=self.states();saved_gt=deepcopy(gt);saved_states=states.copy();collector={}
        one=choose_wrong_placement_point(gt,table(),box([.045]*3),states,0,workspace(),diagnostics=collector)
        two=choose_wrong_placement_point(gt,table(),box([.045]*3),states,0,workspace())
        self.assertEqual(one,two);self.assertEqual(gt,saved_gt);np.testing.assert_array_equal(states,saved_states)
        self.assertEqual(one,collector['point_m'])

    def test_initial_empty_place_is_first_candidate_using_measured_bbox_not_rigid_origin(self):
        states=self.states()[:1];states[0,:3]=[-.2,-.18,.62]
        points=box([.04]*3)+[.03,.01,0.]
        gt=self.gt(z=.82);gt['position_m'][:2]=[.2,.2]
        plan=plan_wrong_placement_point(gt,table(),points,states,0,workspace())
        selected=plan['candidates'][plan['selected_candidate_index']]
        self.assertEqual(plan['selected_candidate_index'],0)
        self.assertEqual(selected['label'],'return_initial_supported_position')
        self.assertEqual(selected['source'],'measured_query_body_bbox_center')
        np.testing.assert_allclose(plan['point_m'],[-.17,-.17,.62],atol=1e-12,rtol=0)
        self.assertEqual(plan['policy']['bounded_gt_offset_count'],32)
        self.assertEqual(len(plan['candidates']),33)

    def test_initial_place_near_gt_is_rejected_then_original_offsets_remain_available(self):
        states=self.states()[:1];states[0,:3]=[.001,0.,.62]
        plan=plan_wrong_placement_point(self.gt(),table(),box([.04]*3),states,0,workspace())
        initial=plan['candidates'][0]
        self.assertEqual(initial['label'],'return_initial_supported_position')
        self.assertIn('too_close_to_frozen_gt',initial['rejection_reasons'])
        selected=plan['candidates'][plan['selected_candidate_index']]
        self.assertEqual(selected['source'],'frozen_ground_truth')
        self.assertTrue(selected['accepted'])


if __name__=='__main__':
    unittest.main()
