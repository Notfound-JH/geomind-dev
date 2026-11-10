"""CPU regressions for failures observed in the registered Robotiq campaign."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from geomind.episodes.isaaclab_physics import IsaacLabEpisode
from geomind.scene_assembly.physics import mesh_bounds_from_rigid_pose
from geomind.scene_assembly.audit import measured_bounds


class RegisteredBoundaryTests(unittest.TestCase):
    def test_respond_settling_keeps_every_actual_video_clock_boundary(self):
        for ratio in [1,2]:
            with self.subTest(dt_s=1/(120*ratio)):
                self._check_video_clock(ratio)

    def _check_video_clock(self,ratio):
        env=IsaacLabEpisode.__new__(IsaacLabEpisode)
        env.dt=1/(120*ratio)
        env.closed=False;env.is_assembled=True;env.protocol_decimation=20*ratio
        env.physics_step_index=140*ratio;env.step_index=1;env._video_deferred_tick=None
        env.targets={'obj_001':'old event force'};env.history={}
        env.resolved=SimpleNamespace(task={'task_id':'TrackMovingObject_v0'},protocol={'phases':[
            {'id':'observe','duration_steps':1,'scene_visible':True},
            {'id':'delay','duration_steps':1,'events':[],'scene_visible':True},
            {'id':'respond','allow_response':True,'scene_visible':True}]})
        captured=list(range(120*ratio,140*ratio+1,4*ratio))
        def capture(**kwargs):
            if env.physics_step_index%(4*ratio):return
            self.assertEqual(env.physics_step_index,captured[-1]+4*ratio)
            captured.append(env.physics_step_index)
        def tick(**kwargs):
            env.physics_step_index+=1
            if env.physics_step_index!=env._video_deferred_tick:capture()
        env._capture_video=capture;env._physics_tick=tick;env._sync_positions=lambda:None
        env.snapshot=lambda:env.physics_step_index;env.observation=lambda:{'allow_response':True}
        with patch('geomind.episodes.isaaclab_physics.solve',return_value={'verified':'measured'}) as solver:
            env.step()
        self.assertEqual(captured,list(range(120*ratio,280*ratio+1,4*ratio)))
        self.assertEqual(env.targets,{})
        self.assertEqual(solver.call_args.args[1],280*ratio)

    def test_near_identity_raw_quaternion_preserves_small_measured_tilt(self):
        cloud=np.array([[x,y,z] for x in [-.027,.027] for y in [-.027,.027] for z in [-.045,.045]])
        q=np.array([1.00000008,.00003,.00007,-.00004])
        original=q.copy();position=[.3,.1,.95]
        measured=mesh_bounds_from_rigid_pose(cloud,position,q)
        independent=measured_bounds(cloud,np.asarray(position),q)
        np.testing.assert_allclose(measured,independent,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(q,original)

if __name__=='__main__':unittest.main()
