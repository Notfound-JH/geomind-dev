"""CPU synthetic contracts; no robot or physics execution is represented here."""
from pathlib import Path
import tempfile
import unittest

from geomind.episodes import EpisodeResolver
from geomind.episodes.resolver import load_json
from geomind.scene_assembly.robot_task_config import prepare_robot_configs
from geomind.scene_assembly.task_binding import bind_and_solve

import test_assembled_task_binding as binding_fixtures


class AssembledRobotConfigTests(unittest.TestCase):
    def build(self,root,spec_index):
        fixture=binding_fixtures.TestAssembledTaskBinding()
        source,_=fixture.fixture(root)
        gt=root/'gt';result=bind_and_solve(source,fixture.specs()[spec_index],gt)
        # The parent pipeline owns task_result; this CPU fixture mirrors that
        # handoff without fabricating any physical robot execution results.
        import json
        (gt/'task_result.json').write_text(json.dumps(result))
        robot={'embodiment':'franka','base_position_w_m':[-.4,0,.6]}
        action='PickLiftSelectedObject' if spec_index==0 else 'PickPlaceObject'
        return gt,source,result,robot,{'action':action}

    def test_scene_uses_existing_prims_and_preserves_semantic_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);gt,source,result,robot,execution=self.build(root,0)
            original=(gt/'configs/scenes/assembled_scene.json').read_bytes()
            converted=prepare_robot_configs(gt,robot,execution,root/'robot_configs')
            resolved=EpisodeResolver(converted['root']).resolve(converted['episode_id'])
            profile=resolved.scene['physics']
            self.assertEqual(profile['scene_mode'],'AssembledScene')
            self.assertEqual(profile['frame_origin_w_m'],[0.,0.,0.])
            self.assertEqual(profile['settle_drop_height_m'],0)
            self.assertEqual(profile['assembly_manifest_path'],str(source))
            self.assertEqual(len(profile['assembly_object_bindings']),4)
            self.assertEqual({b['object_id'] for b in profile['assembly_object_bindings']},{o['id'] for o in result['objects']})
            self.assertTrue(all(b['prim_path'].startswith('/World/AssembledObjects/') for b in profile['assembly_object_bindings']))
            self.assertEqual(resolved.layout['randomization']['position_jitter_m'],0)
            self.assertEqual(resolved.episode['physical_execution']['action'],'PickLiftSelectedObject')
            self.assertNotEqual(converted['contract_hash'],result['contract_hash'])
            self.assertEqual((gt/'configs/scenes/assembled_scene.json').read_bytes(),original)
            evidence=load_json(Path(converted['binding_path']))
            self.assertFalse(evidence['task_action_physics_executed'])
            self.assertFalse(evidence['objects_respawned'])

    def test_place_uses_semantic_moving_alias_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);gt,_,result,robot,execution=self.build(root,1)
            dest=root/'robot_configs';converted=prepare_robot_configs(gt,robot,execution,dest)
            resolved=EpisodeResolver(converted['root']).resolve(converted['episode_id'])
            self.assertEqual(resolved.episode['physical_execution']['moving_object'],result['operation']['moving_object'])
            self.assertEqual(resolved.episode['physical_execution']['dwell_steps'],60)
            with self.assertRaisesRegex(ValueError,'overwrite'):
                prepare_robot_configs(gt,robot,execution,dest)


if __name__=='__main__':unittest.main()
