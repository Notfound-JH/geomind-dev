from pathlib import Path
import json
import tempfile
import unittest

from geomind.tools.dashboard import Dashboard, save_json


class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / 'outputs/example'
        self.run.mkdir(parents=True)
        save_json(self.run / 'report.json', {'physical_execution_verified': True, 'positive_cases': 1,
            'positive_successes': 1, 'results': [{'episode_id': 'Example_v0', 'task_id': 'Example',
            'success': True, 'verification_passed': True, 'physical': {'position_error_m': .002}}]})
        (self.run / 'video.mp4').write_bytes(b'fake test-only data')
        (self.root / 'outputs/example.zip').write_bytes(b'archive')
        self.config = self.root / 'geomind/configs/task_definitions/Example.json'
        save_json(self.config, {'task_id': 'Example', 'domain': 'SpatialReasoning', 'family': 'Geometry'})
        self.dashboard = Dashboard(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_live_config_changes_and_new_results(self):
        state = self.dashboard.state()
        self.assertEqual(len(state['runs']), 1)
        save_json(self.config, {'task_id': 'Example', 'domain': 'SpatialMemory', 'family': 'Recall'})
        state = self.dashboard.state()
        self.assertTrue(any('修改 task_definitions' in e['title'] for e in state['changes']))
        self.assertEqual(state['config']['task_definitions'][0]['domain'], 'SpatialMemory')

    def test_preview_does_not_delete_and_confirmation_required(self):
        plan = self.dashboard.plan('example')
        self.assertEqual(plan['file_count'], 3)
        self.assertTrue(self.run.exists())
        with self.assertRaisesRegex(ValueError, '确认文字'):
            self.dashboard.delete(plan['token'], 'wrong')
        self.assertTrue(self.run.exists())

    def test_confirmed_cleanup_preserves_summary_and_configuration(self):
        plan = self.dashboard.plan('example')
        self.dashboard.delete(plan['token'], 'example')
        self.assertFalse(self.run.exists())
        self.assertFalse((self.root / 'outputs/example.zip').exists())
        self.assertTrue(self.config.exists())
        state = Dashboard(self.root).state()
        self.assertTrue(state['runs'][0]['deleted'])
        self.assertEqual(state['runs'][0]['cases'][0]['task'], 'Example')
        self.assertIsNone(state['runs'][0]['cases'][0]['observer'])
        with self.assertRaises(ValueError):
            self.dashboard.delete(plan['token'], 'example')

    def test_changed_files_cancel_cleanup(self):
        plan = self.dashboard.plan('example')
        (self.run / 'new.npz').write_bytes(b'new data')
        with self.assertRaisesRegex(ValueError, '发生变化'):
            self.dashboard.delete(plan['token'], 'example')
        self.assertTrue(self.run.exists())

    def test_outside_paths_and_symlinks_rejected(self):
        for name in ('../example', '..', '/tmp/example'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.dashboard.plan(name)
        external = self.root / 'keep.txt'
        external.write_text('keep')
        (self.run / 'link').symlink_to(external)
        with self.assertRaisesRegex(ValueError, '符号链接'):
            self.dashboard.plan('example')
        self.assertTrue(external.exists())

    def test_unreported_probe_is_visible_without_claiming_success(self):
        (self.run / 'report.json').unlink()
        (self.run / 'cases').mkdir()
        (self.run / 'traces').mkdir()
        state = self.dashboard.state()
        self.assertEqual(len(state['runs']), 1)
        self.assertFalse(state['runs'][0]['verified'])
        self.dashboard.plan('example')

    def test_expired_plan_rejected(self):
        plan = self.dashboard.plan('example')
        self.dashboard.plans[plan['token']]['expires'] = 0
        with self.assertRaisesRegex(ValueError, '过期'):
            self.dashboard.delete(plan['token'], 'example')
        self.assertTrue(self.run.exists())

    def test_media_cannot_escape_outputs(self):
        self.assertIsNone(self.dashboard.path_url(self.config))
        self.assertIsNone(self.dashboard.path_url('/etc/passwd'))
        self.assertEqual(self.dashboard.path_url(self.run / 'video.mp4'), '/media/example/video.mp4')

    def test_legacy_report_keeps_version_chinese_and_metric(self):
        report = json.loads((self.run / 'report.json').read_text())
        case = report['results'][0]
        case.update(release_version='1.0', instruction_zh='把方块放在中点。', error_label='方块中点位置误差',
                    camera_videos=[{'name':'table_cam','path':str(self.run / 'video.mp4')}])
        save_json(self.run / 'report.json', report)
        summary=self.dashboard.state()['runs'][0]
        self.assertEqual(summary['video_count'],1)
        self.assertEqual(summary['cases'][0]['release_version'],'1.0')
        self.assertEqual(summary['cases'][0]['instruction_zh'],'把方块放在中点。')
        self.assertEqual(summary['cases'][0]['error_label'],'方块中点位置误差')
