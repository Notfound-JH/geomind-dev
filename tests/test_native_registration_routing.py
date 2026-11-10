"""Prevent versioned manipulation registrations falling back to diagnostics.

All episode factories are mocked; this verifies API routing without a GPU or
simulation, preserving explicit-root, bare-name and semantic native APIs.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from geomind.core import loader as current_loader


class RegistrationRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.loader=current_loader

    def assert_registration_route(self,task,**kwargs):
        with patch('geomind.episodes.create_episode_env',return_value=object()) as episode:
            with self.assertRaisesRegex(ValueError,'measured assembly pipeline'):
                self.loader.create_env(task,backend='isaaclab',**kwargs)
            episode.assert_not_called()

    def test_versioned_projection_cannot_silently_run_reach_diagnostic(self):
        self.assert_registration_route('ProjectOntoLine_v0')

    def test_versioned_reflection_cannot_silently_run_reach_diagnostic(self):
        self.assert_registration_route('ReflectAcrossLine_v0')

    def test_versioned_scene_change_cannot_silently_use_another_default_episode(self):
        self.assert_registration_route('DetectSceneChange_v0')

    def test_null_root_is_not_an_explicit_native_episode_choice(self):
        self.assert_registration_route('ProjectOntoLine_v0',root=None)

    def test_explicit_native_episode_root_is_preserved_for_all_three(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'episodes').mkdir()
            for name in ('ProjectOntoLine','ReflectAcrossLine','DetectSceneChange'):
                (root/'episodes'/(name+'_v0.json')).write_text(json.dumps({'task_id':name}))
                sentinel=object()
                with self.subTest(task=name),patch('geomind.episodes.create_episode_env',return_value=sentinel) as episode:
                    self.assertIs(self.loader.create_env(name+'_v0',backend='isaaclab',root=root),sentinel)
                    self.assertEqual(episode.call_args.args,(name+'_v0',))
                    self.assertEqual(episode.call_args.kwargs['root'],root)

    def test_bare_native_release_two_names_are_preserved(self):
        for name in ('ProjectOntoLine','ReflectAcrossLine','DetectSceneChange'):
            sentinel=object()
            with self.subTest(task=name),patch('geomind.episodes.create_episode_env',return_value=sentinel) as episode:
                self.assertIs(self.loader.create_env(name,backend='isaaclab'),sentinel)
                self.assertEqual(episode.call_args.args,(name,))

    def test_unregistered_seeded_native_episode_id_is_preserved(self):
        sentinel=object()
        with patch('geomind.episodes.create_episode_env',return_value=sentinel) as episode:
            self.assertIs(self.loader.create_env('ProjectOntoLine_s1_v0',backend='isaaclab'),sentinel)
            self.assertEqual(episode.call_args.args,('ProjectOntoLine_s1_v0',))

    def test_explicit_semantic_episode_api_is_preserved(self):
        sentinel=object()
        with patch('geomind.episodes.create_episode_env',return_value=sentinel) as episode:
            self.assertIs(self.loader.create_env('ProjectOntoLine_v0',backend='semantic'),sentinel)
            self.assertEqual(episode.call_args.kwargs['backend'],'semantic')


if __name__=='__main__':unittest.main()
