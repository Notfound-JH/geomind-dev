"""Regression tests for physically grounded wrist-camera diagnostic evidence."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geomind.tools.audit_wrist_scene import audit_scene, audit_support_states, oriented_half_extents


def surface():
    return {'prim_path': '/World/RegisteredScene/furniture_table', 'height_m': .9,
            'bounds_xy_m': [-1., 1., -1., 1.],
            'surface_triangles': [[[-1., -1., .9], [1., -1., .9], [1., 1., .9]],
                                  [[-1., -1., .9], [1., 1., .9], [-1., 1., .9]]]}


def states():
    data = np.zeros((121, 15))
    data[:, 0] = np.arange(1, 122)
    data[:, 4] = .915
    data[:, 5] = 1.
    return data


def audit(data, **kwargs):
    options = {'dimensions_m': [.03, .03, .03], 'support_surface': surface(), 'settled_tick': 61}
    options.update(kwargs)
    return audit_support_states(data, ['block'], 1/120, **options)


class WristSceneAuditTests(unittest.TestCase):
    def test_motionless_cube_on_real_mesh_passes(self):
        result = audit(states())
        self.assertTrue(result['real_table_support_verified'])
        self.assertAlmostEqual(result['objects']['block']['maximum_bottom_gap_m'], 0.)
        self.assertEqual(result['final_stable_samples'], 60)
        self.assertAlmostEqual(result['final_stable_duration_s'], .5)

    def test_original_floating_platform_case_cannot_pass_due_to_low_velocity(self):
        data = states()
        data[:, 4] += .16451
        with self.assertRaisesRegex(ValueError, 'elevated'):
            audit(data)

    def test_orientation_is_used_to_compute_bottom_and_footprint(self):
        data = states()
        # Rotate a 6x4x2 cm block by 90 degrees around Y: vertical half size
        # becomes 3 cm, which a naive size[2]/2 calculation would misread.
        data[:, 4] = .93
        data[:, 5:9] = [math.sqrt(.5), 0., math.sqrt(.5), 0.]
        result = audit(data, dimensions_m=[.06, .04, .02])
        self.assertAlmostEqual(result['objects']['block']['final_bottom_gap_m'], 0.)
        np.testing.assert_allclose(oriented_half_extents(data[0, 5:9], [.06, .04, .02]), [.01, .02, .03])

    def test_pre_reset_drop_is_allowed_but_action_time_floating_is_rejected(self):
        data = states()
        data[:60, 4] += .02
        audit(data)
        data[61, 4] += .02
        with self.assertRaisesRegex(ValueError, 'elevated'):
            audit(data)

    def test_cube_center_inside_table_does_not_imply_footprint_support(self):
        data = states()
        data[:, 2] = .99
        with self.assertRaisesRegex(ValueError, 'footprint'):
            audit(data)

    def test_mesh_support_checked_in_addition_to_table_aabb(self):
        table = surface()
        # Bounds claim a square but measured mesh covers its left half only.
        for triangle in table['surface_triangles']:
            for vertex in triangle:
                vertex[0] = (vertex[0] - 1.) / 2
        data = states()
        data[:, 2] = .5
        with self.assertRaisesRegex(ValueError, 'tabletop mesh'):
            audit(data, support_surface=table)

    def test_final_window_requires_stability_and_full_duration(self):
        data = states()
        data[-2, 9] = .1
        with self.assertRaisesRegex(ValueError, 'not stable'):
            audit(data)
        with self.assertRaisesRegex(ValueError, 'half a second'):
            audit(states(), settled_tick=100)

    def test_incomplete_and_invalid_trace_states_rejected(self):
        data = states()
        data[-1, 0] += 1
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            audit(data)
        data = states()
        data[-1, 5] = 2.
        with self.assertRaisesRegex(ValueError, 'Non-unit'):
            audit(data)

    def test_report_api_rejects_helpers_and_saves_independent_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            physics = Path(directory)
            (physics/'traces').mkdir()
            np.savez(physics/'traces/reach_objects.npz', data=states(), object_ids=['block'], dt_s=1/120)
            case = {'episode_id': 'reach', 'robot_name': 'piper_dual', 'trace': '/old/location/reach.npz',
                    'physical': {'action': 'ReachAboveTarget'}, 'query_physical_clock': {'current_tick': 61},
                    'scene_provenance': {'support_surface': surface(),
                        'helper_supports': {'enabled': False, 'prim_paths': []},
                        'diagnostic_objects': {'dimensions_m': [.03]*3, 'support_policy': 'real_table_only'}}}
            report = {'results': [case]}
            (physics/'report.json').write_text(json.dumps(report))
            self.assertTrue(audit_scene(physics)['passed'])
            self.assertTrue((physics/'wrist_scene_audit.json').exists())
            broken = deepcopy(report)
            broken['results'][0]['scene_provenance']['helper_supports']['enabled'] = True
            (physics/'report.json').write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, 'helper supports'):
                audit_scene(physics)


if __name__ == '__main__':
    unittest.main()
