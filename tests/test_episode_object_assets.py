from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from geomind.episodes import EpisodeResolver, ContractError
from geomind.episodes.object_assets import catalog_descriptor, validate_asset
from geomind.tools.generate_spatial_suite import generate, verify


class TestObjectAssets(unittest.TestCase):
    def test_descriptor_rejects_invalid_extent_and_reference(self):
        spec = {'usd_path': 'object.usd', 'scale': [.5]*3, 'dimensions_m': [.04]*3,
                'bottom_offset_m': -.02, 'sha256': 'a'*64,
                'reference_point': 'RigidBodyOrigin', 'collision': 'convexHull'}
        validate_asset(spec)
        for key, value in [('dimensions_m', [-1,.04,.04]), ('bottom_offset_m', float('nan')),
                           ('reference_point', 'unknown'), ('collision', 'none')]:
            invalid = deepcopy(spec); invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_asset(invalid)

    def test_descriptor_uses_catalog_metadata_and_real_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'object.usd'; path.write_bytes(b'asset')
            metadata = {'usd_path': str(path), 'dimensions': {'x': .1,'y': .2,'z': .3}, 'z_min_offset': -.15}
            with patch('geomind_lab.utils.asset_utils.get_asset_info_by_name', return_value=metadata):
                spec = catalog_descriptor('example', .5)
            self.assertEqual(spec['dimensions_m'], [.05,.1,.15])
            self.assertEqual(spec['bottom_offset_m'], -.075)
            self.assertEqual(len(spec['sha256']),64)

    def test_catalog_suite_updates_labels_heights_and_event_consistently(self):
        def fake(name, scale):
            return {'usd_path': name+'.usd','scale':[scale]*3,'dimensions_m':[.04,.05,.06],
                    'bottom_offset_m': -.03,'sha256':'a'*64,
                    'reference_point':'RigidBodyOrigin','collision':'convexHull'}
        with tempfile.TemporaryDirectory() as directory, patch(
                'geomind.episodes.object_assets.catalog_descriptor', side_effect=fake):
            root=Path(directory)
            manifest=generate(root,[0],object_set='catalog')
            self.assertTrue(verify(root,manifest)['passed'])
            resolved=EpisodeResolver(root).resolve('DetectSceneChange_v0')
            self.assertEqual([o['asset'] for o in resolved.layout['objects']],
                             ['blue_can','cube_can','yellow_bottle'])
            self.assertTrue(all(o['position_m'][2] == .03 for o in resolved.layout['objects']))
            self.assertEqual(resolved.protocol['phases'][1]['events'][0]['position_m'][2],.03)
            self.assertNotIn('red block',verify(root,manifest)['results'][-1]['instruction'])
