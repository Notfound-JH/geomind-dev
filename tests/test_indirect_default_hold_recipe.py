"""Recipe opt-in must change provenance without changing semantics or layout."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from geomind.episodes.canonical_extensions import task_ids, module_for
from geomind.episodes.extended_static_tasks import build_spec
from geomind.episodes.registered_tasks import load_registration, plan_temporal
from geomind.episodes.resolver import EpisodeResolver, Snapshot, solve


class IndirectDefaultHoldRecipeTests(unittest.TestCase):
    def test_all_recipes_restore_default_mass_spring(self):
        for task in task_ids():
            spec = module_for(task).build_spec(task)
            self.assertNotIn('grip_hold_policy',spec['protocol'])

    def test_native_schema_accepts_policy_and_gt_and_instruction_stay_identical(self):
        plan = load_registration('SelectByIndirectReference_v0',seed=17)
        spec = plan['task_spec']
        self.assertNotIn('grip_hold_policy',spec['protocol'])
        positions = {o['name']:[*o['pose']['xy'],.95] for o in spec['runtime_objects']}
        mapping = {f'opaque_{index:02d}':name for index,name in enumerate(reversed(list(positions)))}
        parameters = {'frame':'world','registered_task_id':'SelectByIndirectReference_v0',
            'legacy_task_spec':spec,'object_id_map':mapping,'temporal_plan':plan_temporal(spec,positions,17)}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('task_definitions','scenes','layouts','episodes'):(root/name).mkdir()
            task={'task_id':'SelectByIndirectReference_v0','domain':'SpatialReasoning','family':'ConstraintComposition'}
            scene={'scene_id':'cpu_scene','usd_path':'cpu_scene.usda','robot':'not_configured',
                'workspace':{'frame':'world','axes':{'right':'+x','forward':'+y','up':'+z'},
                    'bounds_m':{'x':[0.,1.],'y':[-1.,1.],'z':[0.,2.]}},'cameras':['cpu_no_render']}
            layout={'layout_id':'cpu_layout','frame':'world','objects':[
                {'id':alias,'asset':'brown_block','label':'物体'+str(index),'position_m':positions[name]}
                for index,(alias,name) in enumerate(mapping.items())],
                'randomization':{'position_jitter_m':0.}}
            episode={'schema_version':3,'episode_id':'indirect_demo','task_id':task['task_id'],
                'scene_id':scene['scene_id'],'layout_id':layout['layout_id'],'parameters':parameters,
                'response':'SelectObject','seed':17}
            def write(folder,name,value):
                (root/folder/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False))
            write('task_definitions',task['task_id'],task);write('scenes',scene['scene_id'],scene)
            write('layouts',layout['layout_id'],layout);write('episodes',episode['episode_id'],episode)
            enabled=EpisodeResolver(root).resolve('indirect_demo')
            states={alias:tuple(positions[name]) for alias,name in mapping.items()}
            current=Snapshot('cpu_recipe_check',enabled.contract_hash,'world',0,states)
            first=solve(enabled,current)
            episode['parameters']['legacy_task_spec']['protocol']['grip_hold_policy']='retain_initial_close_target'
            write('episodes',episode['episode_id'],episode)
            default=EpisodeResolver(root).resolve('indirect_demo')
            second=solve(default,Snapshot('cpu_recipe_check',default.contract_hash,'world',0,states))
            self.assertNotEqual(enabled.contract_hash,default.contract_hash)
            self.assertEqual(first['ground_truth'],second['ground_truth'])
            self.assertEqual(first['instruction'],second['instruction'])
            self.assertEqual(mapping[first['ground_truth']['object_id']],'n3')
            self.assertEqual(enabled.layout,default.layout)

    def test_fixture_two_hops_objects_and_query_are_unchanged(self):
        spec=build_spec('SelectByIndirectReference_v0',seed=17)
        self.assertEqual(spec['protocol']['gt_spec']['root_reference'],'n0')
        self.assertEqual(spec['protocol']['gt_spec']['intermediate_candidates'],['n1','n2'])
        self.assertEqual(spec['protocol']['gt_spec']['distance_margin_m'],.015)
        self.assertEqual({o['name']:o['pose']['xy'] for o in spec['runtime_objects']},
            {'n0':[.24,-.06],'n1':[.36,-.03],'n2':[.52,.20],
             'n3':[.39,-.12],'n4':[.22,-.19],'n5':[.55,.10]})
        self.assertEqual(spec['instruction'],
            '先找离红色标记最近的蓝色标记，再拿起离那个蓝色标记最近的棕色方块。')


if __name__=='__main__':
    unittest.main()
