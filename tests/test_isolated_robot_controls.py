import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from geomind.scene_assembly.robot_execution import _execute_isolated_controls


class TestIsolatedControls(unittest.TestCase):
    def execute(self, root, *, missing_control=False):
        originals={}
        commands=[]
        def worker(argv, **kwargs):
            commands.append(argv)
            dest=Path(argv[argv.index('--output')+1]);dest.mkdir(parents=True)
            negative='--negative-only' in argv
            identity='negative' if negative else 'positive'
            result={'negative_control':negative,'verification_passed':True,'success':not negative}
            report={'results':[] if negative and missing_control else [result],
                    'physical_execution_verified':True,'run_token':identity}
            path=dest/'report.json';path.write_text(json.dumps(report));originals[path]=path.read_bytes()
            for kind in ('evidence','observations'):
                evidence=dest/kind;evidence.mkdir();(evidence/(identity+'.json')).write_text(identity)
            return SimpleNamespace(returncode=0)
        dest=root/'robot_physics'
        with patch('geomind.scene_assembly.robot_execution.subprocess.run',side_effect=worker):
            _execute_isolated_controls(['python','--output',str(dest)],root,dest,{},420)
        return dest,originals,commands

    def test_each_case_starts_a_new_worker_and_raw_reports_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            dest,originals,commands=self.execute(Path(folder))
            self.assertEqual(len(commands),2)
            self.assertNotIn('--negative-only',commands[0])
            self.assertIn('--negative-only',commands[1])
            self.assertNotEqual(commands[0][2],commands[1][2])
            for path,data in originals.items():self.assertEqual(path.read_bytes(),data)
            self.assertEqual((dest/'evidence/negative.json').read_text(),'negative')
            self.assertEqual((dest/'observations/negative.json').read_text(),'negative')
            self.assertTrue(json.loads((dest/'report.json').read_text())['physical_execution_verified'])

    def test_positive_cannot_hide_a_missing_control_in_merged_report(self):
        with tempfile.TemporaryDirectory() as folder:
            dest,_,_=self.execute(Path(folder),missing_control=True)
            self.assertFalse(json.loads((dest/'report.json').read_text())['physical_execution_verified'])
