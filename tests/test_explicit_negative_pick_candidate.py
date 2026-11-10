"""Pin the intended wrong-object control without changing task GT or geometry."""
import math
import unittest

from geomind.episodes.extended_static_tasks import TASK_IDS, build_spec, solve_spec
from geomind.tools.run_episode_physics import _negative_pick_object_id


class ExplicitNegativePickCandidateTests(unittest.TestCase):
    def test_legacy_identity_survives_alias_and_candidate_order_permutations(self):
        for mapping, choices, expected in [
            ({'obj_a':'n1','obj_b':'n5','obj_c':'n2'}, ['obj_c','obj_a'], 'obj_a'),
            ({'obj_z':'n1','obj_a':'n2','obj_m':'n5'}, ['obj_a','obj_z'], 'obj_z'),
        ]:
            with self.subTest(mapping=mapping):
                self.assertEqual(_negative_pick_object_id(choices,expected_object_id=next(k for k,v in mapping.items() if v=='n5'),
                    object_id_map=mapping,legacy_object_id='n1'),expected)

    def test_no_private_identity_preserves_existing_first_choice(self):
        self.assertEqual(_negative_pick_object_id(['obj_z','obj_a'],expected_object_id='obj_gt'),'obj_z')

    def test_missing_or_ambiguous_identity_cannot_fall_back(self):
        for mapping in [{'obj_a':'n2'}, {'obj_a':'n1','obj_b':'n1'}]:
            with self.subTest(mapping=mapping),self.assertRaisesRegex(ValueError,'exactly one alias'):
                _negative_pick_object_id(['obj_a','obj_b'],expected_object_id='obj_gt',object_id_map=mapping,legacy_object_id='n1')

    def test_grasp_ineligible_or_gt_body_cannot_be_chosen(self):
        for mapping in [{'too_wide':'n1'}, {'obj_gt':'n1'}]:
            with self.subTest(mapping=mapping),self.assertRaisesRegex(ValueError,'eligible non-GT'):
                _negative_pick_object_id(['obj_other'],expected_object_id='obj_gt',object_id_map=mapping,legacy_object_id='n1')

    def test_malformed_private_identity_or_mapping_is_rejected(self):
        for identity in ['',False,1,[]]:
            with self.subTest(identity=identity),self.assertRaisesRegex(ValueError,'nonempty string'):
                _negative_pick_object_id(['obj_a'],expected_object_id='obj_gt',object_id_map={'obj_a':'n1'},legacy_object_id=identity)
        with self.assertRaisesRegex(ValueError,'identity map'):
            _negative_pick_object_id(['obj_a'],expected_object_id='obj_gt',legacy_object_id='n1')

    def test_only_path_task_requests_pinned_negative_identity(self):
        for task_id in TASK_IDS:
            protocol=build_spec(task_id)['protocol']
            self.assertEqual(protocol.get('negative_control_object_id'),'n1' if task_id=='SelectByPathDistance_v0' else None)

    def test_path_wrong_leaf_is_euclidean_farthest_but_not_graph_farthest(self):
        for seed in (0,1,27):
            spec=build_spec('SelectByPathDistance_v0',seed=seed)
            for shift in (0.,.08):
                positions={key:[value['position'][0]+shift,value['position'][1]] for key,value in spec['object_positions'].items()}
                selected=solve_spec(spec,positions)['selected']
                self.assertEqual(selected,'n5')
                root=positions['n0']
                euclidean=max(spec['protocol']['candidate_object_ids'],key=lambda key:math.dist(root,positions[key]))
                self.assertEqual(euclidean,spec['protocol']['negative_control_object_id'])
                self.assertNotEqual(selected,euclidean)


if __name__=='__main__':unittest.main()
