"""Portability contracts: isolated prefixes, preserved metadata, worker identity."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]

def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,REPO/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

paths=load('_portable_paths_under_test','geomind_lab/utils/portable_paths.py')
installer=load('_portable_installer_under_test','scripts/configure_portable_assets.py')


def sha(data):return hashlib.sha256(data).hexdigest()


def fixture(root):
    bundle=root/'transfer prefix with spaces/geomind-assets-v1';repo=root/'code clone';repo.mkdir()
    for folder in ['geomind_assets','grscenes_home/Materials','isaacsim_assets','material_cache/nvidia',
                   'repo_assets/assets/scene_profiles','repo_assets/assets/scenes','repo_assets/scene_atlas']:
        (bundle/folder).mkdir(parents=True,exist_ok=True)
    original=b'{"assets": {"catch_object": [{"name":"box", "usd_path":"/old/host/box.usda"}]}}\n'
    values={
      'geomind_assets/box.usda':b'#usda 1.0\ndef Xform "box" {}\n',
      'repo_assets/assets/scenes/index.json':b'{}\n',
      'repo_assets/scene_atlas/dist/index.html':b'<html>atlas</html>\n',
      'repo_assets/assets/scene_profiles/assets_catalog_with_metadata.json':json.dumps({'assets':{'catch_object':[{'name':'box','usd_path':'bundle://geomind_assets/box.usda'}]}}).encode(),
    }
    entries=[]
    for relative,data in values.items():
        f=bundle/relative;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
        entries.append({'path':relative,'size_bytes':len(data),'sha256':sha(data),
          'source_sha256':sha(original) if relative.endswith('assets_catalog_with_metadata.json') else sha(data)})
    (bundle/'asset_bundle_manifest.json').write_text(json.dumps({'schema_version':1,'bundle_name':'geomind-assets-v1','files':entries}))
    return bundle,repo,original


class PortablePathsTest(unittest.TestCase):
    def test_bundle_rebinding_is_explicit_and_does_not_reuse_old_host(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{},clear=True):
            bundle,repo,_=fixture(Path(temp))
            with patch.object(paths,'repo_root',return_value=repo),patch.dict(os.environ,{'GEOMIND_ASSET_BUNDLE_ROOT':str(bundle)}):
                self.assertEqual(paths.resolve_asset_path('/home/elm/projects/geomind_assets/box.usda'),str(bundle/'geomind_assets/box.usda'))
                self.assertEqual(paths.repo_assets_root(),bundle/'repo_assets/assets')
                self.assertEqual(paths.scene_atlas_root(),repo/'.geomind_local/scene_atlas')
                self.assertEqual(paths.resolve_asset_path('bundle://geomind_assets/box.usda',require_exists=True),str(bundle/'geomind_assets/box.usda'))
    def test_unknown_metadata_and_urls_are_not_guessed(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(paths.resolve_asset_path('https://example.org/unrelated.usd'),'https://example.org/unrelated.usd')
            self.assertEqual(paths.resolve_asset_path('/home/elm/unrelated/user_file'),'/home/elm/unrelated/user_file')
    def test_asset_and_repo_uri_traversal_is_rejected(self):
        for value in ['bundle://../../outside','repo://../outside']:
            with self.subTest(value=value),self.assertRaises(ValueError):paths.resolve_asset_path(value)
    def test_config_rebinding_preserves_original_data_and_non_paths(self):
        original={'source':'bundle://geomind_assets/box.usda','label':'brown block','values':[1,2,True]};saved=deepcopy(original)
        result=paths.resolve_asset_paths(original)
        self.assertEqual(original,saved);self.assertEqual(result['values'],original['values']);self.assertNotEqual(result['source'],original['source'])


class PortableConfigureTest(unittest.TestCase):
    def test_clean_clone_links_and_rebound_catalog_are_relocatable(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,repo,_=fixture(Path(temp));receipt=installer.configure(bundle,repo,verify_all=True)
            self.assertTrue((repo/'assets').is_symlink());self.assertTrue((repo/'scene_atlas').is_symlink())
            data=json.loads((repo/'.geomind_local/assets_catalog.json').read_text())
            self.assertEqual(data['assets']['catch_object'][0]['usd_path'],str(bundle/'geomind_assets/box.usda'))
            self.assertTrue(receipt['full_file_sha256_verified']);self.assertFalse(receipt['physical_execution_performed'])
            self.assertIn(str(bundle),(repo/'.env.geomind').read_text())
    def test_tracked_metadata_is_preserved_while_missing_subtrees_are_linked(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,repo,original=fixture(Path(temp));catalog=repo/'assets/scene_profiles/assets_catalog_with_metadata.json'
            catalog.parent.mkdir(parents=True);catalog.write_bytes(original)
            installer.configure(bundle,repo)
            self.assertEqual(catalog.read_bytes(),original);self.assertFalse(catalog.is_symlink())
            self.assertTrue((repo/'assets/scenes').is_symlink())
    def test_existing_changed_metadata_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,repo,_=fixture(Path(temp));catalog=repo/'assets/scene_profiles/assets_catalog_with_metadata.json'
            catalog.parent.mkdir(parents=True);catalog.write_bytes(b'User maintained metadata\n')
            with self.assertRaisesRegex(ValueError,'preserved without overwrite'):installer.configure(bundle,repo)
            self.assertEqual(catalog.read_bytes(),b'User maintained metadata\n')
    def test_missing_or_changed_extraction_fails_before_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,repo,_=fixture(Path(temp));(bundle/'geomind_assets/box.usda').unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete'):installer.configure(bundle,repo)
            self.assertFalse((repo/'.env.geomind').exists())
    def test_actual_gallery_publish_leaves_asset_bundle_unchanged(self):
        from geomind.scene_assembly.gallery import publish_local
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle,repo,_=fixture(root);installer.configure(bundle,repo)
            source=bundle/'repo_assets/scene_atlas/dist/index.html';before=sha(source.read_bytes())
            output=root/'assembly';output.mkdir()
            manifest={'request':{'scene_id':65,'object':{'count':1,'asset':'box'},'layout':{'type':'random'}},
                      'scene':{'scene_name':'test'},'status':'assembled_physics_unverified','physics':{'executed':False},
                      'scene_usda':str(output/'scene.usda')}
            (output/'manifest.json').write_text(json.dumps(manifest));(output/'request.json').write_text('{}')
            with patch.dict(os.environ,{'GEOMIND_ASSET_BUNDLE_ROOT':str(bundle),'GEOMIND_SCENE_ATLAS_ROOT':str(repo/'.geomind_local/scene_atlas')}):
                self.assertIsNotNone(publish_local(output))
            self.assertEqual(sha(source.read_bytes()),before)
            self.assertFalse((bundle/'repo_assets/scene_atlas/dist/assets/assemblies.json').exists())
            self.assertTrue((repo/'.geomind_local/scene_atlas/dist/assets/assemblies.json').is_file())
    def test_manifest_cannot_escape_asset_root(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,repo,_=fixture(Path(temp));m=bundle/'asset_bundle_manifest.json';data=json.loads(m.read_text());data['files'][0]['path']='../other';m.write_text(json.dumps(data))
            with self.assertRaises(ValueError):installer.configure(bundle,repo)


class PortableInterpreterTest(unittest.TestCase):
    def setUp(self):self.worker=load('_portable_worker_under_test','geomind/tools/assemble_scene.py')
    def sdk(self,root,name,version='5.1.0-test'):
        prefix=root/name;exe=prefix/'bin/python';exe.parent.mkdir(parents=True);exe.touch()
        sdk=prefix/'lib/python3.11/site-packages/isaacsim';(sdk/'extscache/omni.usd.libs-test').mkdir(parents=True)
        (sdk/'VERSION').write_text(version);return exe
    def test_cli_then_env_then_current_sdk_priority(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{},clear=True):
            root=Path(temp);cli=self.sdk(root,'cli');env=self.sdk(root,'env');current=self.sdk(root,'current')
            with patch.object(self.worker.sys,'executable',str(current)),patch.dict(os.environ,{'GEOMIND_ISAAC_PYTHON':str(env)}):
                self.assertEqual(self.worker._python_environment(str(cli))[0],str(cli))
                self.assertEqual(self.worker._python_environment()[0],str(env))
            with patch.object(self.worker.sys,'executable',str(current)):
                self.assertEqual(self.worker._python_environment()[0],str(current))
    def test_missing_or_45_selection_never_falls_back_to_another_env(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{},clear=True):
            root=Path(temp);good=self.sdk(root,'good');old=self.sdk(root,'old','4.5.0')
            with patch.object(self.worker.sys,'executable',str(good)):
                with self.assertRaisesRegex(ValueError,'unavailable'):self.worker._python_environment(str(root/'missing'))
                with self.assertRaisesRegex(ValueError,'IsaacSim5.1'):self.worker._python_environment(str(old))
    def test_explicit_cli_interpreter_reaches_export_settle_and_robot_stage(self):
        batch=load('_portable_batch_under_test','geomind/tools/regenerate_registered_tasks.py')
        with tempfile.TemporaryDirectory() as temp:
            case=Path(temp)/'case';selected='/selected/isaacsim51/bin/python'
            plan={'registration':{'task_id':'SelectByRegion_v0','response':'SelectObject'},'source_hashes':{},'task_spec':{'protocol':{'module_name':'canonical_extension'}}}
            manifest={'table':{'height_m':.93},'scene':{'scene_id':65},'scene_usda':str(case/'assembly/scene.usda')}
            def worker(kind,*args):
                (case/'assembly').mkdir(exist_ok=True)
                return {'report':manifest if kind=='export' else {'passed':True},'report_path':str(case/'assembly/manifest.json'),
                        'exit_code':0,'bounded_cleanup_after_report':False}
            with patch('geomind.episodes.registered_tasks.load_registration',return_value=plan),patch.object(batch,'_snapshot',return_value={}), \
                 patch.object(batch,'_finish_provenance',return_value={'passed':True}),patch.object(batch,'_worker',side_effect=worker) as workers, \
                 patch('geomind.scene_assembly.audit.audit',return_value={'passed':True}), \
                 patch('geomind.scene_assembly.registered_binding.bind_and_solve',return_value={'status':'solved','ground_truth':{'type':'Object','object_id':'obj_1'}}), \
                 patch('geomind.scene_assembly.robot_execution.execute_robot_stage',return_value={'executed':True,'passed':True,'report':{'results':[]}}) as robot, \
                 patch.object(batch,'_action_passed',return_value=True):
                result=batch._run_case('SelectByRegion_v0',case,scene_id=65,seed=0,timeout_s=420,max_steps=12000,negative_controls=True,prepare_only=False,isaac_python=selected,command=['test'])
            self.assertTrue(result['task_completed'],result)
            self.assertEqual([call.args[3] for call in workers.call_args_list],[selected,selected])
            self.assertEqual(robot.call_args.kwargs['isaac_python'],selected)

if __name__=='__main__':unittest.main()
