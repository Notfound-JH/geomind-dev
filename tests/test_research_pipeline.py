import tempfile
from pathlib import Path
import unittest
from geomind.episodes import ContractError
from geomind.tools.research_pipeline import parse_request, prepare
from geomind.episodes.resolver import load_json

class PipelineTests(unittest.TestCase):
    def request(self, **kw):
        return parse_request(dict(task_family='GeometricInference', output='unused', backend='semantic', artifacts=['gt','observations','events'], interaction_mode='diagnostic', **kw))

    def test_operation_request_does_not_fall_back_to_coordinate_task(self):
        raw = dict(task_family='GeometricInference', scene_id='semantic_table',
                   seed=0, count=2, backend='isaaclab', timeout_s=420,
                   output='outputs/research/next_base')
        with self.assertRaisesRegex(ContractError, 'missing grasp, transport'):
            parse_request(raw)
        # Naming a geometry task must not silently opt into the old diagnostic chain.
        with self.assertRaisesRegex(ContractError, 'diagnostic only'):
            parse_request(dict(raw, task_id='ProjectOntoLine'))
        diagnostic = parse_request(dict(raw, interaction_mode='diagnostic'))
        self.assertEqual(diagnostic['task_id'], 'ProjectOntoLine')
        self.assertEqual(diagnostic['scene_id'], 'semantic_table')

    def test_scene_updating_defaults_to_physical_grasp(self):
        with tempfile.TemporaryDirectory() as d:
            r=parse_request(dict(task_family='SceneUpdating',output=d,backend='isaaclab',count=2))
            self.assertEqual(r['interaction_mode'],'manipulation')
            self.assertTrue(prepare(r,Path(d))['passed'])
            for p in (Path(d)/'configs/episodes').glob('DetectSceneChange*.json'):
                self.assertEqual(load_json(p)['physical_execution']['action'],'PickLiftChangedObject')

    def test_selection_and_full_coverage_rejection(self):
        self.assertEqual(self.request()['task_id'], 'ProjectOntoLine')
        with self.assertRaisesRegex(ContractError, 'Full family unsupported'):
            self.request(coverage='all')

    def test_no_silent_substitution(self):
        for kw in [dict(scene_id='scene17'), dict(condition='Aug'), dict(time_protocol='tracking'),
                   dict(task_id='InferMidpoint'), dict(layout='unknown'), dict(augmentation='translate_x_0.03m')]:
            with self.subTest(kw=kw), self.assertRaises(ContractError): self.request(**kw)
        with self.assertRaisesRegex(ContractError, 'CPU cannot'):
            parse_request(dict(task_family='GeometricInference',output='x',backend='semantic',interaction_mode='diagnostic'))

    def test_paired_translation_gt_and_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            r=self.request(condition='Aug',augmentation='translate_x_0.03m',count=2)
            self.assertTrue(prepare(r,out)['passed'])
            for p in (out/'paired_base').glob('*.json'):
                base=load_json(p); ep=base['episode']
                aug=load_json(out/'configs/layouts'/f"{ep['layout_id']}.json")
                for a,b in zip(base['layout']['objects'],aug['objects']):
                    self.assertAlmostEqual(b['position_m'][0]-a['position_m'][0],.03)
                    self.assertEqual(b['position_m'][1:],a['position_m'][1:])
                result=load_json(out/'semantic'/p.name)
                self.assertAlmostEqual(result['semantic_gt']['ground_truth']['position_m'][0]-base['expected_answer']['position_m'][0],.03)

    def test_dynamic_visibility_and_history(self):
        with tempfile.TemporaryDirectory() as d:
            r=parse_request(dict(task_family='SceneUpdating',output=d,backend='semantic',artifacts=['gt','events','observations']))
            prepare(r,Path(d)); result=load_json(Path(d)/'semantic/DetectSceneChange_v0.json')
            self.assertEqual(len(result['observations']),6)
            self.assertTrue(result['evidence']['history'])
            for obs in result['observations']:
                self.assertNotIn('ground_truth',obs)
                if not obs['scene_visible']: self.assertEqual(obs['objects'],[])

if __name__ == '__main__': unittest.main()
