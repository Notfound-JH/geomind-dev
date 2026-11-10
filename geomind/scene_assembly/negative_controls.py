"""Deterministic wrong-placement controls from real query-time body geometry.

This plans an empty, supported wrong landing point. It does not certify robot
reachability, a collision-free transport path, or physical success/failure.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from geomind.scene_assembly.audit import measured_bounds
from geomind.scene_assembly.body_geometry import points_for
from geomind.scene_assembly.usd import supported_xy


MIN_WRONG_DISTANCE_M = .05
RETAINED_OBJECT_CLEARANCE_M = .06
TABLE_EDGE_MARGIN_M = .005
# After the emptied initial-position candidate, preserve the original +80mm X
# point as the first offset, followed by fixed seed-independent alternatives.
_OFFSETS_XY_M = tuple(
    offset for radius in (.08, .10, .12, .16, .20)
    for offset in ((radius, 0.), (-radius, 0.), (0., radius), (0., -radius))
) + tuple(
    (sx * radius, sy * radius)
    for radius in (.08, .12, .16) for sx, sy in ((1., 1.), (1., -1.), (-1., 1.), (-1., -1.))
)


class NegativeControlPlanningError(ValueError):
    """No legal candidate; ``diagnostics`` preserves every attempted point."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bounds(states, points):
    states = np.asarray(states, dtype=float)
    _require(states.ndim == 2 and states.shape[1] == 13 and len(states) > 0
             and np.isfinite(states).all(), 'Expected finite query-time Nx13 rigid states')
    _require(np.allclose(np.linalg.norm(states[:, 3:7], axis=1), 1., atol=1e-4, rtol=0),
             'Invalid query-time object quaternion')
    low, high = [], []
    for index, state in enumerate(states):
        lo, hi = measured_bounds(points_for(points, index), state[:3], state[3:7])
        low.append(lo); high.append(hi)
    return np.asarray(low), np.asarray(high)


def _aabb_distance(low, high, other_low, other_high):
    gaps = np.maximum(0., np.maximum(low - other_high, other_low - high))
    return float(np.linalg.norm(gaps))


def plan_wrong_placement_point(gt, table, points, states, moving_index, workspace, *, candidate_order='initial_first'):
    """Return a point plus all candidate checks, without changing GT or state.

    ``table`` is the result of ``measure_table`` and must contain its actual
    ``surface_triangles``. Optional ``retained_obstacles`` entries contain
    world ``min``/``max`` AABBs of retained fixtures. All nonmoving bodies in
    ``states`` are checked independently of that optional fixture list.
    ``workspace`` is the native world-frame workspace dictionary.
    """
    _require(candidate_order in {'initial_first','offsets_first'}, 'Unknown negative placement candidate order')
    _require(isinstance(gt, dict) and gt.get('type') in {'Point', 'Relation'}
             and gt.get('frame') == 'world', 'Placement control requires a world-frame Point or Relation GT')
    target = np.asarray(gt.get('position_m'), dtype=float)
    _require(target.shape == (3,) and np.isfinite(target).all(), 'Invalid frozen placement GT')
    _require(isinstance(workspace, dict) and workspace.get('frame') == 'world',
             'Negative-control workspace must use the world frame')
    limits = np.asarray([workspace['bounds_m'][axis] for axis in ('x', 'y', 'z')], dtype=float)
    _require(limits.shape == (3, 2) and np.isfinite(limits).all() and np.all(limits[:, 1] > limits[:, 0]),
             'Invalid negative-control workspace bounds')
    triangles = np.asarray(table.get('surface_triangles'), dtype=float)
    _require(triangles.ndim == 3 and triangles.shape[1:] == (3, 3) and len(triangles) > 0
             and np.isfinite(triangles).all(), 'Actual tabletop support triangles are required')
    table_height = float(table['height_m'])
    _require(np.isfinite(table_height), 'Invalid measured tabletop height')
    low, high = _bounds(states, points)
    _require(type(moving_index) is int and 0 <= moving_index < len(low), 'Invalid moving-body index')
    half = (high[moving_index] - low[moving_index]) * .5
    _require(np.all(half > 0), 'Moving body has no finite positive volume')
    references = [index for index in range(len(low)) if index != moving_index]
    fixtures = table.get('retained_obstacles', [])
    _require(isinstance(fixtures, list), 'Retained fixture AABBs must be a list')
    fixture_bounds = []
    for fixture in fixtures:
        lo = np.asarray(fixture['min'], dtype=float); hi = np.asarray(fixture['max'], dtype=float)
        _require(lo.shape == hi.shape == (3,) and np.isfinite(lo).all()
                 and np.isfinite(hi).all() and np.all(hi >= lo), 'Invalid retained fixture AABB')
        fixture_bounds.append((lo, hi))
    initial_xy = ((low[moving_index] + high[moving_index]) * .5)[:2]
    candidates = [{'label': 'return_initial_supported_position',
                   'source': 'measured_query_body_bbox_center', 'xy': initial_xy}]
    candidates.extend({'label': 'bounded_gt_offset', 'source': 'frozen_ground_truth',
                       'xy': target[:2] + np.asarray(offset)} for offset in _OFFSETS_XY_M)
    if candidate_order=='offsets_first':
        candidates=candidates[1:]+candidates[:1]
    records = []
    for candidate_index, candidate in enumerate(candidates):
        xy = np.asarray(candidate['xy'], dtype=float)
        dx, dy = xy - target[:2]
        point = np.asarray([xy[0], xy[1], table_height + half[2]])
        lo = point - half; hi = point + half
        distance = float(np.linalg.norm(point - target))
        in_workspace = bool(np.all(lo >= limits[:, 0] - 1e-9) and np.all(hi <= limits[:, 1] + 1e-9))
        support = bool(supported_xy(table, point[:2], half[:2] + TABLE_EDGE_MARGIN_M))
        body_distances = {index: _aabb_distance(lo, hi, low[index], high[index]) for index in references}
        fixture_distances = [_aabb_distance(lo, hi, flo, fhi) for flo, fhi in fixture_bounds]
        blocked_bodies = [index for index, gap in body_distances.items() if gap < RETAINED_OBJECT_CLEARANCE_M - 1e-9]
        blocked_fixtures = [index for index, gap in enumerate(fixture_distances) if gap < RETAINED_OBJECT_CLEARANCE_M - 1e-9]
        reasons = []
        if distance < MIN_WRONG_DISTANCE_M: reasons.append('too_close_to_frozen_gt')
        if not in_workspace: reasons.append('full_body_outside_workspace')
        if not support: reasons.append('footprint_lacks_table_support')
        if blocked_bodies: reasons.append('retained_object_clearance')
        if blocked_fixtures: reasons.append('retained_fixture_clearance')
        records.append({'candidate_index': candidate_index, 'offset_xy_m': [dx, dy],
                        'label': candidate['label'], 'source': candidate['source'],
                        'point_m': point.tolist(), 'wrong_distance_m': distance,
                        'full_body_in_workspace': in_workspace, 'support_9_samples_passed': support,
                        'minimum_retained_object_aabb_gap_m': min(body_distances.values()) if body_distances else None,
                        'blocked_body_indices': blocked_bodies, 'blocked_fixture_indices': blocked_fixtures,
                        'accepted': not reasons, 'rejection_reasons': reasons, 'selected': False})
    valid = [record for record in records if record['accepted']]
    plan = {'schema_version': 1, 'kind': 'supported_empty_wrong_placement_v1',
            'frozen_ground_truth': deepcopy(gt),
            'point_m': valid[0]['point_m'] if valid else None,
            'selected_candidate_index': valid[0]['candidate_index'] if valid else None,
            'candidates': records,
            'policy': {'minimum_wrong_distance_m': MIN_WRONG_DISTANCE_M,
                       'retained_object_aabb_clearance_m': RETAINED_OBJECT_CLEARANCE_M,
                       'table_edge_margin_m': TABLE_EDGE_MARGIN_M, 'support_samples': 9,
                       'source_gt_type': gt['type'],
                       'z_policy': 'measured_table_height_plus_query_time_moving_world_AABB_halfheight',
                       'retained_body_count': len(references), 'retained_fixture_count': len(fixtures),
                       'candidate_count': len(records), 'bounded_gt_offset_count': len(_OFFSETS_XY_M),
                       'initial_position_candidate': 'measured query-time moving world AABB centre XY; original moving body is excluded from retained objects',
                       'candidate_order':candidate_order,
                       'selection': ('first_accepted_in_fixed_order; measured initial position first, then original bounded offsets'
                                     if candidate_order=='initial_first' else
                                     'first_accepted_in_fixed_order; original bounded offsets first, then measured initial position')},
            'robot_reachability_verified': False, 'contact_free_certified': False,
            'physical_execution_performed': False}
    if not valid:
        raise NegativeControlPlanningError('No supported empty wrong-placement point in the bounded candidate set', plan)
    valid[0]['selected'] = True
    return plan


def choose_wrong_placement_point(gt, table, points, states, moving_index, workspace, *, diagnostics=None):
    """Return XYZ; optionally copy the complete plan into ``diagnostics``."""
    plan = plan_wrong_placement_point(gt, table, points, states, moving_index, workspace)
    if diagnostics is not None:
        _require(isinstance(diagnostics, dict), 'Diagnostics collector must be a dictionary')
        diagnostics.update(plan)
    return list(plan['point_m'])
