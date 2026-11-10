"""CPU-only assembly binding contracts; synthetic evidence is not PhysX execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from geomind.episodes import ContractError
from geomind.episodes.isaaclab_physics import load_assembled_source
from geomind.episodes.physical_contract import physics_profile,execution_profile,validate_physics,validate_execution


class AssembledRobotContractTests(unittest.TestCase):
    def fixture(self, root):
        points=np.asarray([[x,y,z] for x in [.01,.05] for y in [-.02,.02] for z in [-.03,.03]])
        np.savez_compressed(root/'asset_body_vertices.npz',points_m=points)
        states=np.zeros((600,2,13));states[:,:,3]=1
        states[:,:,:3]=[[.1,0,.63],[.3,0,.63]]
        paths=['/World/AssembledObjects/blue_can_001','/World/AssembledObjects/blue_can_002']
        np.savez_compressed(root/'object_states.npz',data=states,prim_paths=paths)
        (root/'scene.usda').write_text('#usda 1.0\n# CPU fixture only\n')
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        manifest={'status':'assembled_physics_passed','offline_audit':{'passed':True},
                  'scene_usda':str(root/'scene.usda'),'asset_body_vertices':str(root/'asset_body_vertices.npz'),
                  'table':{'height_m':.6},'physics':{'sha256':{'object_states.npz':digest(root/'object_states.npz')}},
                  'provenance':{'scene_usda_sha256':digest(root/'scene.usda'),
                                'asset_body_vertices_sha256':digest(root/'asset_body_vertices.npz')}}
        (root/'manifest.json').write_text(json.dumps(manifest))
        profile=physics_profile();profile.pop('table_size_m')
        profile.update(scene_mode='AssembledScene',frame_origin_w_m=[0,0,0],settle_drop_height_m=0,
            assembly_manifest_path=str(root/'manifest.json'),assembly_manifest_sha256=digest(root/'manifest.json'),
            asset_body_vertices_path=str(root/'asset_body_vertices.npz'),asset_body_vertices_sha256=digest(root/'asset_body_vertices.npz'),
            tabletop_height_m=.6,assembly_object_bindings=[
                {'object_id':name,'prim_path':path,'root_position_w_m':state[:3].tolist(),'root_orientation_wxyz':state[3:7].tolist()}
                for name,path,state in zip(['obj_002','obj_001'],paths,states[-1])])
        scene={'robot':'franka','usd_path':str(root/'scene.usda'),'physics':profile,'workspace':{'frame':'world'}}
        layout={'frame':'world','randomization':{'position_jitter_m':0},'objects':[
            {'id':'obj_002','position_m':[.13,0,.63]},{'id':'obj_001','position_m':[.33,0,.63]}]}
        return SimpleNamespace(scene=scene,layout=layout)

    def test_existing_body_identity_root_pose_and_aabb_center_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            resolved=self.fixture(Path(directory));validate_physics(resolved.scene)
            manifest,cloud,bindings=load_assembled_source(resolved)
            self.assertEqual(bindings[0]['object_id'],'obj_002')
            self.assertEqual(bindings[0]['prim_path'],'/World/AssembledObjects/blue_can_001')
            self.assertAlmostEqual(float(cloud[:,0].mean()),.03)
            self.assertEqual(manifest['scene_usda'],resolved.scene['usd_path'])

    def test_second_jitter_unknown_prim_or_changed_root_pose_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original=self.fixture(Path(directory))
            for change in ('jitter','prim','pose','center'):
                resolved=deepcopy(original)
                if change=='jitter':resolved.layout['randomization']['position_jitter_m']=.01
                if change=='prim':resolved.scene['physics']['assembly_object_bindings'][0]['prim_path']='/World/AssembledObjects/other'
                if change=='pose':resolved.scene['physics']['assembly_object_bindings'][0]['root_position_w_m'][0]+=.01
                if change=='center':resolved.layout['objects'][0]['position_m'][0]+=.03
                with self.subTest(change=change),self.assertRaises(ContractError):load_assembled_source(resolved)

    def test_artifact_changes_cannot_be_loaded_as_original_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);resolved=self.fixture(root)
            (root/'scene.usda').write_text('changed')
            with self.assertRaisesRegex(ContractError,'USD hash changed'):load_assembled_source(resolved)

    def test_world_frame_zero_drop_and_unique_bindings_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            original=self.fixture(Path(directory))
            for change in ('origin','drop','duplicates','quaternion'):
                scene=deepcopy(original.scene)
                if change=='origin':scene['physics']['frame_origin_w_m']=[.1,0,0]
                if change=='drop':scene['physics']['settle_drop_height_m']=.02
                if change=='duplicates':scene['physics']['assembly_object_bindings'][1]=deepcopy(scene['physics']['assembly_object_bindings'][0])
                if change=='quaternion':scene['physics']['assembly_object_bindings'][0]['root_orientation_wxyz']=[0,0,0,0]
                with self.subTest(change=change),self.assertRaises(ContractError):validate_physics(scene)

    def test_new_actions_support_explicit_panda_and_robotiq_manipulation(self):
        with tempfile.TemporaryDirectory() as directory:
            resolved=self.fixture(Path(directory));scene=resolved.scene
            episode={'task_id':'SelectByOrder','parameters':{},'physical_execution':dict(execution_profile(),action='PickLiftSelectedObject')}
            validate_execution(episode,scene)
            robotiq=deepcopy(scene);robotiq['robot']='franka_robotiq'
            validate_execution(episode,robotiq)
            for robot in ['fr3','piper','piper_dual']:
                altered=deepcopy(scene);altered['robot']=robot
                if robot=='piper_dual':episode['physical_execution']['active_arm']='left'
                else:episode['physical_execution'].pop('active_arm',None)
                validate_physics(altered)
                with self.subTest(robot=robot),self.assertRaises(ContractError):validate_execution(episode,altered)
                diagnostic=deepcopy(episode);diagnostic['physical_execution']['action']='ReachAboveTarget'
                validate_execution(diagnostic,altered)

    def test_placement_cannot_move_reference_or_unknown_object(self):
        with tempfile.TemporaryDirectory() as directory:
            scene=self.fixture(Path(directory)).scene
            episode={'task_id':'InferMidpoint','parameters':{'endpoints':['obj_001','obj_002']},
                'physical_execution':dict(execution_profile(),action='PickPlaceObject',moving_object='obj_003')}
            with self.assertRaisesRegex(ContractError,'Unknown assembled moving'):validate_execution(episode,scene)
            episode['physical_execution']['moving_object']='obj_001'
            with self.assertRaisesRegex(ContractError,'Cannot move a task reference'):validate_execution(episode,scene)


if __name__=='__main__':unittest.main()
