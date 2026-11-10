"""Camera review fixtures must use the table rather than legacy flying slabs."""
import json
from pathlib import Path
import tempfile
import unittest

from geomind.tools.verify_wrist_cameras import ground_diagnostic_layout


class TabletopFixtureTests(unittest.TestCase):
    def test_variant_preserves_xy_and_original_but_changes_episode_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            files={
                'episodes/camera.json':{'layout_id':'old3d','scene_id':'room'},
                'scenes/room.json':{'physics':{'object_size_m':[.03,.03,.03]}},
                'layouts/old3d.json':{'layout_id':'old3d','objects':[
                    {'id':'a','position_m':[.1,.2,.2]}, {'id':'b','position_m':[-.1,.3,.3]}],
                    'randomization':{'position_jitter_m':0.}},
            }
            for name,value in files.items():
                path=root/name;path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(value))
            change=ground_diagnostic_layout(root,'camera')
            self.assertFalse(change['helper_supports_allowed'])
            self.assertEqual(json.loads((root/'layouts/old3d.json').read_text()),files['layouts/old3d.json'])
            variant=json.loads((root/'layouts/old3d_camera_tabletop.json').read_text())
            self.assertEqual([x['position_m'] for x in variant['objects']],[[.1,.2,.015],[-.1,.3,.015]])
            self.assertEqual(json.loads((root/'episodes/camera.json').read_text())['layout_id'],variant['layout_id'])

    def test_dual_fixture_uses_validated_base_without_changing_objects_xy_or_home(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            files={
                'episodes/camera.json':{'layout_id':'old','scene_id':'room'},
                'scenes/room.json':{'robot':'piper_dual','physics':{
                    'object_size_m':[.03,.03,.03],'robot_base_w_m':[-.11,-.22,.93]}},
                'layouts/old.json':{'layout_id':'old','objects':[{'id':'a','position_m':[.1,.2,.2]}],
                                    'randomization':{'position_jitter_m':0.}},
            }
            for name,value in files.items():
                path=root/name;path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(value))
            change=ground_diagnostic_layout(root,'camera')
            actual=json.loads((root/'scenes/room.json').read_text())
            self.assertAlmostEqual(actual['physics']['robot_base_w_m'][0],-.01)
            self.assertEqual(actual['physics']['robot_base_w_m'][1:],[-.22,.93])
            self.assertEqual(change['robot_mount_change']['translation_w_m'],[.10,0.,0.])
            self.assertEqual(change['positions_m']['a'],[.1,.2,.015])

