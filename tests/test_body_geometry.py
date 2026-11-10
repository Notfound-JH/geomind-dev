"""Mixed-body CPU geometry fixtures; no simulator or physical evidence."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from geomind.episodes.assembled_actions import object_geometry
from geomind.episodes.isaaclab_physics import load_assembled_source
from geomind.scene_assembly.audit import score_window
from geomind.scene_assembly.body_geometry import load_body_clouds, points_for


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BodyGeometryTests(unittest.TestCase):
    def clouds(self):
        a = np.asarray([[x,y,z] for x in [.01,.05] for y in [-.02,.02] for z in [-.03,.03]])
        b = np.asarray([[x,y,z] for x in [-.05,.05] for y in [-.015,.015] for z in [-.01,.01]] + [[0,0,-.01],[0,0,.01]])
        return a, b

    def fixture(self, root):
        a, b = self.clouds()
        path = root/'asset_body_vertices.npz'
        np.savez_compressed(path, a=a, b=b, unused=np.asarray([123]))
        return {'asset_body_vertices':str(path), 'objects':[
                    {'prim_path':'/World/AssembledObjects/a','body_points_key':'a'},
                    {'prim_path':'/World/AssembledObjects/b','body_points_key':'b'}],
                'provenance':{'asset_body_vertices_sha256':digest(path)}}

    def states(self):
        states = np.zeros((2,13));states[:,3] = 1.
        states[:,:3] = [[-.1,0,.63],[.2,.1,.65]]
        states[1,3:7] = [math.sqrt(.5),0,math.sqrt(.5),0]
        return states

    def test_mixed_cloud_keys_and_vertex_counts_follow_manifest_order(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.fixture(Path(directory))
            a, b = self.clouds()
            clouds = load_body_clouds(manifest)
            self.assertEqual([len(v) for v in clouds], [8,10])
            np.testing.assert_array_equal(clouds[0], a)
            np.testing.assert_array_equal(clouds[1], b)
            manifest['objects'].reverse()
            np.testing.assert_array_equal(load_body_clouds(manifest)[0], b)

    def test_shared_legacy_points_array_and_default_archive_key_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);a,_ = self.clouds();path = root/'asset_body_vertices.npz'
            np.savez_compressed(path, points_m=a)
            manifest = {'asset_body_vertices':str(path),'objects':[{},{}],
                        'provenance':{'asset_body_vertices_sha256':digest(path)}}
            loaded = load_body_clouds(manifest)
            np.testing.assert_array_equal(loaded[0], a)
            np.testing.assert_array_equal(loaded[1], a)
            np.testing.assert_array_equal(points_for(a, 7), a)
            np.testing.assert_array_equal(points_for(a.tolist(), 7), a)

    def test_complete_archive_hash_rejects_changes_even_to_unused_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);manifest = self.fixture(root)
            load_body_clouds(manifest)
            a,b = self.clouds()
            np.savez_compressed(root/'asset_body_vertices.npz',a=a,b=b,unused=np.asarray([456]))
            with self.assertRaisesRegex(ValueError,'hash changed'):
                load_body_clouds(manifest)

    def test_invalid_key_or_vertices_are_rejected_after_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);manifest = self.fixture(root)
            manifest['objects'][0]['body_points_key'] = 'missing'
            with self.assertRaisesRegex(ValueError,'key is missing'):
                load_body_clouds(manifest)
            manifest['objects'][0]['body_points_key'] = 'a'
            a,b = self.clouds();a[0,0] = np.nan
            np.savez_compressed(root/'asset_body_vertices.npz',a=a,b=b)
            manifest['provenance']['asset_body_vertices_sha256'] = digest(root/'asset_body_vertices.npz')
            with self.assertRaisesRegex(ValueError,'finite Nx3'):
                load_body_clouds(manifest)

    def test_actual_rotation_and_asymmetric_origins_use_each_objects_own_shape(self):
        clouds = list(self.clouds());states = self.states()
        centres,low,high = object_geometry(clouds, states)
        np.testing.assert_allclose(centres,[[-.07,0,.63],[.2,.1,.65]],atol=1e-12)
        np.testing.assert_allclose(low,[[-.09,-.02,.6],[.19,.085,.6]],atol=1e-12)
        np.testing.assert_allclose(high,[[-.05,.02,.66],[.21,.115,.7]],atol=1e-12)
        manifest = {'table':{'height_m':.6,'bounds_xy_m':[-.5,.5,-.5,.5]},'objects':[
            {'position_w_m':s[:3].tolist(),'orientation_wxyz':s[3:7].tolist()} for s in states]}
        history = np.repeat(states[None], 100, axis=0)
        self.assertEqual(score_window(history,manifest,clouds),100)
        self.assertEqual(score_window(history,manifest,clouds[::-1]),0)
        with self.assertRaisesRegex(ValueError,'count mismatch'):
            object_geometry(clouds[:1],states)

    def test_runtime_clouds_are_reordered_by_prim_binding_not_manifest_position(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);manifest = self.fixture(root);states = self.states()
            paths = [v['prim_path'] for v in manifest['objects']]
            np.savez_compressed(root/'object_states.npz',data=np.repeat(states[None],600,axis=0),prim_paths=paths)
            (root/'scene.usda').write_text('#usda 1.0\n# CPU fixture only\n')
            manifest.update(status='assembled_physics_passed',offline_audit={'passed':True},
                            scene_usda=str(root/'scene.usda'),table={'height_m':.6},
                            physics={'sha256':{'object_states.npz':digest(root/'object_states.npz')}})
            manifest['provenance']['scene_usda_sha256'] = digest(root/'scene.usda')
            manifest['objects'].reverse()
            (root/'manifest.json').write_text(json.dumps(manifest))
            profile = {'assembly_manifest_path':str(root/'manifest.json'),
                       'assembly_manifest_sha256':digest(root/'manifest.json'),
                       'asset_body_vertices_path':str(root/'asset_body_vertices.npz'),
                       'asset_body_vertices_sha256':digest(root/'asset_body_vertices.npz'),
                       'tabletop_height_m':.6,'assembly_object_bindings':[
                           {'object_id':identity,'prim_path':path,'root_position_w_m':state[:3].tolist(),
                            'root_orientation_wxyz':state[3:7].tolist()}
                           for identity,path,state in zip(['obj_a','obj_b'],paths,states)]}
            resolved = SimpleNamespace(scene={'physics':profile,'usd_path':str(root/'scene.usda')},
                layout={'frame':'world','randomization':{'position_jitter_m':0},'objects':[
                    {'id':'obj_a','position_m':[-.07,0,.63]},{'id':'obj_b','position_m':[.2,.1,.65]}]})
            _,clouds,bindings = load_assembled_source(resolved)
            self.assertEqual([b['object_id'] for b in bindings],['obj_a','obj_b'])
            self.assertEqual([len(v) for v in clouds],[8,10])
            np.testing.assert_array_equal(clouds[0],self.clouds()[0])
            np.testing.assert_array_equal(clouds[1],self.clouds()[1])


if __name__ == '__main__':
    unittest.main()
