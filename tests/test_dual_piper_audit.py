import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from geomind.episodes import ContractError
from geomind.tools.audit_dual_piper import audit_dual

class TestDualPiperAudit(unittest.TestCase):
    def test_natural_tool_directions_are_mirrored_and_downward(self):
        from geomind.episodes.robot_profiles import PIPER_DUAL_TOOL_QUATERNIONS
        directions = []
        for side in ['left','right']:
            w,x,y,z = PIPER_DUAL_TOOL_QUATERNIONS[side]
            self.assertAlmostEqual(w*w+x*x+y*y+z*z, 1.)
            direction = np.array([2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)])
            self.assertAlmostEqual(direction[2], -np.sin(np.deg2rad(20)))
            directions.append(direction)
        np.testing.assert_allclose(directions[0], directions[1]*[1,-1,1], atol=1e-7)

    def fixture(self, root, drift=0.):
        trace = root/'physics/traces';trace.mkdir(parents=True)
        cases=[];n=9
        for identity,active in [('left_case','left'),('right_case','right')]:
            data=np.zeros((3,2+6*n));data[:,0]=[1,2,3]
            idle=1 if active=='left' else 0
            data[-1,2+3*n*idle+n]=drift
            np.savez(trace/f'{identity}_dual_controls.npz',data=data,arm_names=['left','right'],
                joint_names=[f'joint{i}' for i in range(9)],joint_limits=np.tile([-3.,3.],(2,n,1)),active_arm=active,dt_s=1/120)
            cases.append(dict(episode_id=identity,negative_control=False,physics_steps=3,active_arm=active,arm_count=2))
        (root/'physics/report.json').write_text(json.dumps(dict(physical_execution_verified=True,results=cases)))

    def test_both_arms_with_idle_hold_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            self.assertTrue(audit_dual(root)['passed'])

    def test_passed_primary_task_does_not_hide_idle_arm_disturbance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,.05)
            with self.assertRaisesRegex(ContractError,'Idle arm'):
                audit_dual(root)
