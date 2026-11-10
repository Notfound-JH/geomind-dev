"""Independently check real-table support in native wrist-camera diagnostics.

This narrow audit uses recorded rigid-body states and measured runtime table
bounds. It does not treat a low velocity, or a successful reaching action, as
evidence that an object rests on the table. It is intentionally unsuitable for
grasping episodes, whose objects should leave the table during the action.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def oriented_half_extents(quaternions, dimensions_m):
    """World AABB half extents of cuboids from wxyz root quaternions."""
    q = np.asarray(quaternions, dtype=float)
    size = np.asarray(dimensions_m, dtype=float)
    _require(q.shape[-1:] == (4,) and np.isfinite(q).all(), "Invalid object quaternion")
    _require(size.shape == (3,) and np.isfinite(size).all() and (size > 0).all(),
             "Invalid cuboid dimensions")
    norms = np.linalg.norm(q, axis=-1)
    _require(np.all(np.abs(norms - 1.) < 2e-4), "Non-unit object quaternion")
    w, x, y, z = np.moveaxis(q / norms[..., None], -1, 0)
    rotation = np.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    return np.abs(rotation) @ (size / 2.)


def _require_footprint_on_mesh(centers_xy, half_extents_xy, triangles):
    """Check conservative footprint samples with an independent cross-sign test."""
    triangles = np.asarray(triangles, dtype=float)
    _require(triangles.ndim == 3 and triangles.shape[1:] == (3, 3)
             and len(triangles) and np.isfinite(triangles).all(),
             "Missing measured tabletop surface triangles")
    triangles = triangles[:, :, :2]
    offsets = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
    points = (centers_xy[..., None, :] + half_extents_xy[..., None, :]*offsets).reshape(-1, 2)
    edges = np.roll(triangles, -1, axis=1) - triangles
    area2 = np.abs(edges[:, 0, 0]*(-edges[:, 2, 1]) - edges[:, 0, 1]*(-edges[:, 2, 0]))
    triangles, edges = triangles[area2 > 1e-12], edges[area2 > 1e-12]
    _require(len(triangles), "Degenerate measured tabletop surface triangles")
    for start in range(0, len(points), 512):
        vectors = points[start:start+512, None, None, :] - triangles[None]
        crosses = edges[None, :, :, 0]*vectors[..., 1] - edges[None, :, :, 1]*vectors[..., 0]
        inside = np.all(crosses >= -1e-9, axis=-1) | np.all(crosses <= 1e-9, axis=-1)
        _require(np.all(np.any(inside, axis=-1)), "Object footprint leaves the measured tabletop mesh")


def audit_support_states(data, object_ids, dt_s, *, dimensions_m, support_surface,
                         settled_tick, gap_tolerance_m=.002,
                         stable_speed_tolerance_m_s=.001, stable_duration_s=.5):
    """Recompute support and settled speeds directly from an NPZ state table."""
    data = np.asarray(data, dtype=float)
    ids = [str(identity) for identity in object_ids]
    _require(ids and len(set(ids)) == len(ids), "Missing or duplicate object identities")
    _require(data.ndim == 2 and data.shape[1] == 2 + 13*len(ids) and len(data) > 1,
             "Unexpected native object-state trace shape")
    _require(np.isfinite(data).all(), "Nonfinite object state")
    _require(np.isfinite(dt_s) and dt_s > 0, "Invalid physics timestep")
    ticks = data[:, 0]
    _require(np.all(np.diff(ticks) == 1), "Object-state ticks are not consecutive")
    _require(np.isfinite(settled_tick) and settled_tick >= ticks[0], "Invalid settled-state tick")
    _require(support_surface.get('prim_path'), "Missing runtime support mesh prim")
    height = float(support_surface['height_m'])
    bounds = np.asarray(support_surface['bounds_xy_m'], dtype=float)
    _require(np.isfinite(height) and bounds.shape == (4,) and np.isfinite(bounds).all()
             and bounds[0] < bounds[1] and bounds[2] < bounds[3], "Invalid runtime support bounds")
    after_reset = ticks >= settled_tick
    _require(after_reset.any(), "No post-reset object states")
    minimum_samples = int(np.ceil(stable_duration_s / dt_s))
    stable = ticks > ticks[-1] - minimum_samples
    _require(int(np.sum(stable & after_reset)) >= minimum_samples,
             "Need at least half a second of settled object states")
    states = data[:, 2:].reshape(len(data), len(ids), 13)
    half = oriented_half_extents(states[..., 3:7], dimensions_m)
    bottoms = states[..., 2] - half[..., 2]
    gaps = bottoms - height
    footprint = np.stack([states[..., 0] - half[..., 0], states[..., 0] + half[..., 0],
                          states[..., 1] - half[..., 1], states[..., 1] + half[..., 1]], axis=-1)
    _require(np.all(footprint[after_reset, :, 0] >= bounds[0])
             and np.all(footprint[after_reset, :, 1] <= bounds[1])
             and np.all(footprint[after_reset, :, 2] >= bounds[2])
             and np.all(footprint[after_reset, :, 3] <= bounds[3]),
             "Object footprint leaves the measured real table bounds")
    _require_footprint_on_mesh(states[after_reset, :, :2], half[after_reset, :, :2],
                               support_surface['surface_triangles'])
    _require(np.all(np.abs(gaps[after_reset]) <= gap_tolerance_m),
             "Object bottom is elevated above or penetrates the real table")
    speeds = np.linalg.norm(states[..., 7:10], axis=-1)
    _require(np.all(speeds[stable] <= stable_speed_tolerance_m_s),
             "Objects are not stable over the final half second")
    return {
        'real_table_support_verified': True,
        'support_surface': {key: value for key, value in support_surface.items()
                            if key != 'surface_triangles'},
        'measured_tabletop_triangle_count': len(support_surface['surface_triangles']),
        'mesh_footprint_samples_per_state': 9,
        'gap_tolerance_m': gap_tolerance_m,
        'stable_speed_tolerance_m_s': stable_speed_tolerance_m_s,
        'post_reset_samples': int(after_reset.sum()),
        'final_stable_samples': int(stable.sum()),
        'final_stable_duration_s': float(stable.sum()*dt_s),
        'objects': {
            identity: {
                'minimum_bottom_gap_m': float(gaps[after_reset, i].min()),
                'maximum_bottom_gap_m': float(gaps[after_reset, i].max()),
                'final_bottom_gap_m': float(gaps[-1, i]),
                'final_window_minimum_bottom_gap_m': float(gaps[stable, i].min()),
                'final_window_maximum_bottom_gap_m': float(gaps[stable, i].max()),
                'final_window_maximum_linear_speed_m_s': float(speeds[stable, i].max()),
                'final_position_w_m': states[-1, i, :3].tolist(),
            } for i, identity in enumerate(ids)
        },
    }


def audit_scene(physics):
    """Audit all reaching diagnostic cases and save a JSON result on success."""
    physics = Path(physics).resolve()
    report = json.loads((physics/'report.json').read_text())
    _require(report.get('results'), "No native execution results to audit")
    cases = []
    for case in report['results']:
        _require(case['physical']['action'] == 'ReachAboveTarget',
                 "Table-only wrist audit is restricted to reaching diagnostics")
        provenance = case['scene_provenance']
        objects = provenance['diagnostic_objects']
        helpers = provenance['helper_supports']
        _require(helpers.get('enabled') is False and helpers.get('prim_paths') == []
                 and objects.get('support_policy') == 'real_table_only',
                 "Floating helper supports are enabled or unspecified")
        trace_path = Path(case['trace'])
        local = physics/'traces'/(trace_path.stem + '_objects.npz')
        path = local if local.is_file() else trace_path.with_name(trace_path.stem + '_objects.npz')
        with np.load(path, allow_pickle=False) as trace:
            checked = audit_support_states(
                trace['data'], trace['object_ids'], float(trace['dt_s']),
                dimensions_m=objects['dimensions_m'], support_surface=provenance['support_surface'],
                settled_tick=case['query_physical_clock']['current_tick'])
        cases.append({'episode_id': case['episode_id'], 'robot_name': case['robot_name'],
                      'negative_control': case.get('negative_control', False),
                      'helper_supports_disabled': True, **checked})
    result = {'passed': True, 'cases': cases,
              'scope': 'Recorded cuboid poses, measured real-table bounds, support gaps, '
                       'and final half-second velocities; no grasp or model claim'}
    (physics/'wrist_scene_audit.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physics', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_scene(args.physics), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
