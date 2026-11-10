"""Known answers and condition necessity for bounded canonical static tasks."""
from copy import deepcopy
import unittest

from geomind.episodes.extended_static_tasks import TASK_IDS, build_spec, check_spec, solve_spec


def positions(spec):
    return {obj['name']: [*obj['pose']['xy'], .95] for obj in spec['runtime_objects']}


class ExtendedStaticContracts(unittest.TestCase):
    def test_independent_hand_labeled_fixture_answers(self):
        # Labels follow authored diagrams/inequalities, not a production solve.
        known = {'SelectByBranchPosition_v0': 'n4', 'SelectByRegion_v0': 'n2',
                 'SelectByPathDistance_v0': 'n5', 'SelectByTwoRelations_v0': 'n2',
                 'SelectByIndirectReference_v0': 'n3'}
        for task_id, expected in known.items():
            spec = build_spec(task_id); p = positions(spec)
            with self.subTest(task=task_id):
                actual = solve_spec(spec, p)
                self.assertEqual(actual['selected'], expected)
                self.assertTrue(check_spec(spec, p, {'selected': expected})['success'])
                wrong = next(obj for obj in spec['protocol']['candidate_object_ids'] if obj != expected)
                self.assertFalse(check_spec(spec, p, {'selected': wrong})['success'])
                self.assertFalse(check_spec(spec, p, actual)['production_solver_called'])

    def test_shared_translation_and_object_permutation_preserve_semantic_answer(self):
        for task_id in TASK_IDS:
            spec = build_spec(task_id); p = positions(spec)
            actual = solve_spec(spec, p)['selected']
            p = {name: [value[0]+.31, value[1]-.17, value[2]+.8]
                 for name, value in reversed(list(p.items()))}
            spec['runtime_objects'].reverse()
            self.assertEqual(solve_spec(spec, p)['selected'], actual)
            self.assertTrue(check_spec(spec, p, {'selected': actual})['success'])

    def test_many_seeded_fixtures_keep_clearance_and_condition_margins(self):
        for seed in range(30):
            for task_id in TASK_IDS:
                spec = build_spec(task_id, seed=seed); p = positions(spec)
                result = solve_spec(spec, p)
                self.assertTrue(check_spec(spec, p, result)['success'])
                self.assertFalse(spec['layout_randomization_contract']['independent_object_jitter_allowed'])
                for index, (name, point) in enumerate(p.items()):
                    for other in list(p.values())[index+1:]:
                        self.assertGreater(max(abs(point[i]-other[i]) for i in (0, 1)), .06)

    def test_branch_uses_topology_and_ordinal_not_world_order(self):
        spec = build_spec('SelectByBranchPosition_v0'); p = positions(spec)
        # Swap coordinates while retaining graph identities: arrows, not X/Y,
        # determine the branch and rank.
        p['n3'], p['n4'] = p['n4'], p['n3']
        result = solve_spec(spec, p)
        self.assertEqual(result['selected'], 'n4')
        self.assertEqual(result['calculation']['without_branch_mark_candidates'], ['n4','n6'])
        self.assertEqual(result['calculation']['without_ordinal_candidates'], ['n3','n4'])
        bad = deepcopy(spec); bad['protocol']['gt_spec']['directed_edges'].pop()
        with self.assertRaisesRegex(ValueError, 'tree'):
            solve_spec(bad, p)
        bad = deepcopy(spec); bad['protocol']['gt_spec']['ordinal_after_mark'] = 3
        with self.assertRaisesRegex(ValueError, 'ordinal'):
            solve_spec(bad, p)

    def test_graph_path_farthest_differs_from_euclidean_farthest(self):
        spec = build_spec('SelectByPathDistance_v0'); result = solve_spec(spec, positions(spec))
        self.assertEqual(result['selected'], 'n5')
        self.assertEqual(result['calculation']['shortest_path_distance']['n5'], 4.)
        self.assertEqual(result['calculation']['euclidean_farthest'], 'n1')
        bad = deepcopy(spec); bad['protocol']['gt_spec']['undirected_edges'] = [['n0', name, 1.] for name in ['n1','n2','n3','n4','n5']]
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            solve_spec(bad, positions(bad))
        bad = deepcopy(spec); bad['protocol']['gt_spec']['undirected_edges'].pop()
        with self.assertRaisesRegex(ValueError, 'unreachable'):
            solve_spec(bad, positions(bad))

    def test_region_rejects_boundary_and_multiple_interior_candidates(self):
        spec = build_spec('SelectByRegion_v0'); p = positions(spec)
        p['n3'][0] = p['n0'][0]
        with self.assertRaisesRegex(ValueError, 'boundary'):
            solve_spec(spec, p)
        p = positions(spec); p['n3'] = [p['n2'][0]+.045, p['n2'][1], .95]
        with self.assertRaisesRegex(ValueError, 'one candidate'):
            solve_spec(spec, p)

    def test_two_relations_each_condition_is_necessary(self):
        spec = build_spec('SelectByTwoRelations_v0'); p = positions(spec)
        evidence = solve_spec(spec, p)['calculation']
        self.assertEqual(set(evidence['without_right_candidates']), {'n2','n4'})
        self.assertEqual(set(evidence['without_front_candidates']), {'n2','n3'})
        # Move one single-condition distractor across the other boundary. The
        # intersection is still unique, but removing a condition now suffices.
        p['n4'][1] = p['n1'][1]-.1
        with self.assertRaisesRegex(ValueError, 'own distractor'):
            solve_spec(spec, p)

    def test_indirect_two_hops_cannot_be_collapsed_to_direct_distance(self):
        spec = build_spec('SelectByIndirectReference_v0'); p = positions(spec)
        evidence = solve_spec(spec, p)['calculation']
        self.assertEqual(evidence['intermediate_reference'], 'n1')
        self.assertEqual(evidence['direct_root_shortcut_target'], 'n4')
        self.assertEqual(evidence['alternative_intermediate_targets'], {'n2':'n5'})
        p['n2'] = list(p['n1'])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            solve_spec(spec, p)

    def test_missing_positions_and_nonfinite_states_are_rejected(self):
        spec = build_spec(TASK_IDS[0]); p = positions(spec); p.pop('n0')
        with self.assertRaisesRegex(ValueError, 'exactly'):
            solve_spec(spec, p)
        p = positions(spec); p['n0'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            solve_spec(spec, p)


class StaticContextVisuals(unittest.TestCase):
    def test_visual_marks_keep_expected_graph_and_region_without_colliders(self):
        try:
            from pxr import Usd, UsdPhysics
        except ImportError:
            self.skipTest('USD bindings unavailable')
        from geomind.episodes.extended_static_tasks import draw_context
        for task_id in TASK_IDS:
            stage = Usd.Stage.CreateInMemory(); spec = build_spec(task_id)
            objects = [{'object_id': key, 'position_w_m': point} for key, point in positions(spec).items()]
            paths = draw_context(stage, spec, objects, .93)
            if task_id != 'SelectByIndirectReference_v0': self.assertTrue(paths)
            self.assertFalse(any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse()))
            # Redraw is deterministic and replaces only this module's context.
            self.assertEqual(draw_context(stage, spec, objects, .93), paths)


if __name__ == '__main__':
    unittest.main()
