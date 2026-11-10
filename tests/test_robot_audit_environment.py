"""Offline verification must use selected SDK USD bindings, without Kit/GPU."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BASE=Path(os.environ.get('GEOMIND_AUDIT_CANDIDATE_SOURCE',Path(__file__).resolve().parents[1]))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,BASE/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

robot=load('geomind.scene_assembly._audit_environment_robot','geomind/scene_assembly/robot_execution.py')
cli=load('geomind.tools._audit_environment_cli','geomind/tools/audit_assembled_robot.py')


class RobotAuditEnvironmentTest(unittest.TestCase):
    def test_audit_launches_selected_sdk_in_usd_only_cpu_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);physics=output/'robot_physics';physics.mkdir()
            expected={'passed':True,'mode':'independent offline replay; no physical re-execution','cases':[{'success':True}]}
            def run(command,**kwargs):
                (physics/'assembled_robot_audit.json').write_text(json.dumps(expected))
                self.assertEqual(command,['/chosen/sdk/python','-m','geomind.tools.audit_assembled_robot','--run',str(output)])
                self.assertEqual(kwargs['env']['PYTHONPATH'],'sdk/USD/path')
                self.assertEqual(kwargs['env']['GEOMIND_OFFLINE_AUDIT_USD_CHILD'],'1')
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                self.assertEqual(kwargs['timeout'],120)
                return SimpleNamespace(returncode=0)
            with patch('geomind.tools.assemble_scene._python_environment',return_value=('/chosen/sdk/python',{'PYTHONPATH':'sdk/USD/path'})) as select,patch.object(robot.subprocess,'run',side_effect=run):
                result=robot._audit_in_sdk_environment(output,isaac_python='/explicit/python')
            select.assert_called_once_with('/explicit/python',usd_only=True)
            self.assertEqual(result,expected)
            record=json.loads((output/'robot_audit_command.json').read_text())
            self.assertFalse(record['GPU_requested']);self.assertFalse(record['physical_reexecution'])
    def test_failed_child_cannot_accept_a_stale_passing_report(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);physics=output/'robot_physics';physics.mkdir()
            report=physics/'assembled_robot_audit.json';report.write_text('{"passed":true}')
            before=report.read_bytes()
            with patch('geomind.tools.assemble_scene._python_environment',return_value=('/sdk/python',{})),patch.object(robot.subprocess,'run',return_value=SimpleNamespace(returncode=1)):
                result=robot._audit_in_sdk_environment(output)
            self.assertFalse(result['passed']);self.assertEqual(report.read_bytes(),before)
            self.assertTrue((physics/'assembled_robot_audit_failure.json').is_file())
    def test_missing_report_or_timeout_is_not_a_success(self):
        for timeout in [False,True]:
            with self.subTest(timeout=timeout),tempfile.TemporaryDirectory() as temp:
                output=Path(temp);(output/'robot_physics').mkdir()
                error=subprocess.TimeoutExpired(['audit'],120) if timeout else None
                with patch('geomind.tools.assemble_scene._python_environment',return_value=('/sdk/python',{})),patch.object(robot.subprocess,'run',return_value=SimpleNamespace(returncode=0),side_effect=error):
                    result=robot._audit_in_sdk_environment(output)
                self.assertFalse(result['passed'])
    def test_pipeline_never_calls_the_parent_pxrlib_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);manifest=output/'assembly.json'
            manifest.write_text(json.dumps({'table':{'bounds_xy_m':[0,1,0,1],'height_m':.9}}))
            (output/'result.json').write_text(json.dumps({'assembly_manifest':str(manifest)}))
            request={'robot':{'embodiment':'franka_robotiq'},'execution':{'action':'PickLiftSelectedObject','timeout_s':420,'negative_controls':True}}
            prepared={'root':str(output/'robot_configs'),'episode_id':'assembled_task'}
            def physical(*args):
                destination=output/'robot_physics';destination.mkdir()
                (destination/'report.json').write_text(json.dumps({'results':[{}],'physical_execution_verified':True}))
                return 0
            with patch('geomind.scene_assembly.robot_task_config.prepare_robot_configs',return_value=prepared),patch('geomind.tools.assemble_scene._python_environment',return_value=('/sdk/python',{})),patch.object(robot,'_execute_isolated_controls',side_effect=physical),patch.object(robot,'_audit_in_sdk_environment',return_value={'passed':True}) as audit_child,patch('geomind.tools.audit_assembled_robot.audit',side_effect=ModuleNotFoundError('pxr')) as parent_audit:
                result=robot.execute_robot_stage(output,request,isolate_negative_controls=True,isaac_python='/explicit/sdk/python')
            self.assertTrue(result['passed']);parent_audit.assert_not_called()
            audit_child.assert_called_once_with(output,isaac_python='/explicit/sdk/python')


class OfflineAuditCliTest(unittest.TestCase):
    def test_direct_cli_selects_sdk_usd_child_and_preserves_return_code(self):
        with patch.dict(os.environ,{},clear=True),patch('geomind.tools.assemble_scene._python_environment',return_value=('/chosen/python',{})) as select,patch.object(cli.subprocess,'run',return_value=SimpleNamespace(returncode=7)) as launch,patch.object(cli,'audit',side_effect=AssertionError('No parent audit')):
            code=cli.main(['--run','/tmp/case','--isaac-python','/selected/python'])
        self.assertEqual(code,7);select.assert_called_once_with('/selected/python',usd_only=True)
        self.assertEqual(launch.call_args.args[0],['/chosen/python','-m','geomind.tools.audit_assembled_robot','--run','/tmp/case'])
        self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
    def test_usd_child_audits_once_without_recursion_and_returns_false_as_failure(self):
        with patch.dict(os.environ,{'GEOMIND_OFFLINE_AUDIT_USD_CHILD':'1'},clear=True),patch.object(cli,'audit',return_value={'passed':False}) as audit,patch.object(cli.subprocess,'run',side_effect=AssertionError('No recursive launch')):
            self.assertEqual(cli.main(['--run','/tmp/case']),1)
        audit.assert_called_once_with(Path('/tmp/case/robot_configs'),Path('/tmp/case/robot_physics'))

if __name__=='__main__':unittest.main()
