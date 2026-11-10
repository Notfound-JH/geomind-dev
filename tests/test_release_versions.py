import json
from pathlib import Path
import tempfile
import unittest
from geomind.core.registry import registry
from geomind.episodes import EpisodeResolver, ContractError
from geomind.tools.generate_spatial_suite import generate

ROOT = Path(__file__).resolve().parents[1] / 'geomind/configs'

class TestReleaseVersions(unittest.TestCase):
    def test_original_registrations_are_one_and_native_are_two(self):
        original = [*ROOT.glob('tasks/*.json'), *ROOT.glob('vla_tasks/*.json')]
        self.assertEqual(len(original),59)
        for p in original:
            self.assertEqual(json.loads(p.read_text())['release_version'],'1.0')
        for p in ROOT.glob('episodes/*.json'):
            self.assertEqual(json.loads(p.read_text())['release_version'],'2.0')
        self.assertEqual(registry.get('ProjectOntoLine').release_version,'1.0')

    def test_generator_persists_native_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=generate(root,[0])
            resolver=EpisodeResolver(root)
            for case in manifest['cases']:
                self.assertEqual(resolver.resolve(case['episode_id']).episode['release_version'],'2.0')
            path=root/'episodes/ProjectOntoLine_v0.json'
            payload=json.loads(path.read_text());payload['release_version']=2.0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ContractError,'release 2.0'):
                resolver.resolve('ProjectOntoLine_v0')
