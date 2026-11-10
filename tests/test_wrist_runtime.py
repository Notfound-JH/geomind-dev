import unittest
from unittest.mock import Mock
from geomind.episodes.isaaclab_physics import IsaacLabEpisode


class WristRecordingCleanupTests(unittest.TestCase):
    def test_failure_in_one_view_still_closes_every_open_writer(self):
        env=object.__new__(IsaacLabEpisode)
        env.recorder=Mock();env.recorder.close.side_effect=ValueError('incomplete table stream')
        left,right=Mock(),Mock();env.wrist_recorders={'wrist_left':left,'wrist_right':right}
        with self.assertRaisesRegex(ValueError,'incomplete table stream'):
            env.finish_recording()
        left.close.assert_called_once();right.close.assert_called_once()
        self.assertIsNone(env.recorder);self.assertEqual(env.wrist_recorders,{})
        self.assertIsNone(env.finish_recording())

    def test_each_view_keeps_its_own_report_and_common_frame_clock(self):
        env=object.__new__(IsaacLabEpisode)
        base={'frame_count':5,'start_physics_tick':120,'last_physics_tick':136}
        env.recorder=Mock();env.recorder.close.return_value=dict(base,observer='external.mp4')
        left,right=Mock(),Mock()
        left.close.return_value=dict(base,observer='left.mp4')
        right.close.return_value=dict(base,observer='right.mp4')
        env.wrist_recorders={'wrist_left':left,'wrist_right':right}
        result=env.finish_recording()
        self.assertEqual(result['observer'],'external.mp4')
        self.assertEqual(result['wrist_cameras']['wrist_left']['observer'],'left.mp4')
        self.assertEqual(result['wrist_cameras']['wrist_right']['observer'],'right.mp4')
        self.assertTrue(result['all_views_share_physics_ticks'])
