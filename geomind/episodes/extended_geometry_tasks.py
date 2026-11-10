"""Tabletop manipulation variants of projection and line reflection.

The public native tasks still support their original infinite 3-D line
contracts. These registrations explicitly restrict the demonstration to the
horizontal support plane so the inferred point is a legal placement goal.
"""
from __future__ import annotations

from copy import deepcopy
import math


TASK_IDS = ('ProjectOntoLine_v0', 'ReflectAcrossLine_v0')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cube(name, xy, color, label):
    return {'name': name, 'kind': 'primitive_cuboid', 'size': [.04, .04, .04],
            'object_category': '方块', 'role': 'task_object',
            'visual_color': color, 'pose': {'xy': list(xy), 'yaw': 0., 'z_policy': 'tabletop'},
            'physics': {'mass': .05, 'rigid_body': {'rigid_body_enabled': True}},
            'public_label': label}


def build_spec(task_id, scene_id=65, seed=0):
    _require(task_id in TASK_IDS, f'Unsupported extended geometry task: {task_id}')
    _require(type(seed) is int and seed >= 0, 'Seed must be a nonnegative integer')
    name = task_id.removesuffix('_v0')
    # A tilted axis and an oblique offset avoid axis-aligned arithmetic clues.
    # The perpendicular foot is at t=.65, rather than the endpoint midpoint.
    # This defeats a source-ignoring midpoint / midpoint-centred symmetry rule.
    # Both demonstrated goals lie well inside the scene-65 table workspace.
    objects = [_cube('red_cube', [.32, -.12], [.85, .08, .06], '红色方块'),
               _cube('blue_cube', [.48, .12], [.06, .16, .85], '蓝色方块'),
               _cube('green_cube', [.3324744677, .0970170215], [.08, .68, .12], '绿色方块')]
    instruction = ('把绿色方块移到红、蓝方块连成的直线上，尽量少移动。'
                   if name == 'ProjectOntoLine' else
                   '以红、蓝方块连成的直线为对称轴，把绿色方块移到桌面上与它对称的位置。')
    return {'instruction': instruction, 'runtime_objects': objects,
            'object_positions': {o['name']: {'position': [*o['pose']['xy'], .02]} for o in objects},
            'protocol': {'module_name': 'canonical_extension', 'variant': 'TabletopPlane',
                         'query_text': instruction,
                         'target_object': {'object_id': 'green_cube'},
                         'pre_attached_obj': 'green_cube',
                         'object_semantics': {
                             'red_cube': {'color': '红色', 'shape': '方块'},
                             'blue_cube': {'color': '蓝色', 'shape': '方块'},
                             'green_cube': {'color': '绿色', 'shape': '方块'}},
                         'gt_spec': {'resolver': name, 'variant': 'TabletopPlane',
                                     'point_object_id': 'green_cube',
                                     'line_object_ids': ['red_cube', 'blue_cube'],
                                     'minimum_line_length_m': .10,
                                     'minimum_offset_m': .06,
                                     'minimum_projection_midpoint_separation_m': .03,
                                     'minimum_goal_reference_clearance_m': .08}},
            'native_task_id': task_id,
            'definition_scope': {
                'variant': 'TabletopPlane', 'support_plane': 'MeasuredHorizontalTabletop',
                'reference_point': 'WorldAxisAlignedBoundingBoxCenter',
                'line_extent': 'Infinite',
                'preserves_public_3d_native_contract': True,
                'requires_real_pick_transport_release_and_stability': True},
            'scene_id': scene_id, 'seed': seed}


def solve_spec(spec, positions, initial=None):
    """Solve XY from measured centres; Z is supplied by measured table support."""
    gt = spec['protocol']['gt_spec']; name = gt['resolver']
    _require(name in {x.removesuffix('_v0') for x in TASK_IDS}, 'Unknown geometry resolver')
    _require(gt.get('variant') == 'TabletopPlane', 'Expected explicit tabletop-plane variant')
    point_id = gt['point_object_id']; ids = gt['line_object_ids']
    _require(len(ids) == 2 and len(set([point_id, *ids])) == 3, 'Line and moving point must be distinct')
    _require({point_id, *ids} <= positions.keys(), 'Missing measured geometry object')
    p, a, b = ([float(v) for v in positions[key][:2]] for key in [point_id, *ids])
    _require(all(math.isfinite(v) for xy in (p, a, b) for v in xy), 'Nonfinite measured position')
    d = [b[i]-a[i] for i in range(2)]; length2 = sum(x*x for x in d)
    _require(length2 >= gt['minimum_line_length_m']**2, 'Degenerate or undersized reference line')
    fraction = sum((p[i]-a[i])*d[i] for i in range(2))/length2
    foot = [a[i]+fraction*d[i] for i in range(2)]
    offset = math.dist(p, foot)
    _require(offset >= gt['minimum_offset_m'], 'Moving object is too close to reference line')
    midpoint = [(a[i]+b[i])*.5 for i in range(2)]
    midpoint_separation = math.dist(foot, midpoint)
    minimum_midpoint_separation = gt.get('minimum_projection_midpoint_separation_m', .03)
    _require(math.isfinite(minimum_midpoint_separation) and minimum_midpoint_separation >= .03,
             'Midpoint quality margin must remain at least 30mm')
    _require(midpoint_separation >= minimum_midpoint_separation,
             'Midpoint shortcut makes the source-position condition unnecessary within action tolerance')
    target = foot if name == 'ProjectOntoLine' else [2*foot[i]-p[i] for i in range(2)]
    clearance = min(math.dist(target, q) for q in (a, b))
    _require(clearance >= gt['minimum_goal_reference_clearance_m'], 'Goal overlaps a reference object')
    displacement = [p[i]-foot[i] for i in range(2)]
    return {'point_xy': target, 'calculation': {
        'resolver': name, 'variant': 'TabletopPlane', 'line_parameter': fraction,
        'line_length_m': math.sqrt(length2), 'projection_xy_m': foot,
        'source_offset_m': offset, 'minimum_reference_clearance_m': clearance,
        'projection_midpoint_separation_m': midpoint_separation,
        'minimum_projection_midpoint_separation_m': minimum_midpoint_separation,
        'midpoint_centred_reflection_shortcut_error_m': 2*midpoint_separation,
        'midpoint_shortcut_rejected': True,
        'source_point_xy_m': p, 'line_endpoints_xy_m': [a, b],
        'perpendicular_dot_residual_m2': sum(displacement[i]*d[i] for i in range(2)),
        'goal_derived_from': 'Actual measured current object centres; planned positions not used',
        'minimum_displacement_unique': name == 'ProjectOntoLine',
        'reflection_equal_distance_to_axis': name == 'ReflectAcrossLine'}}


def check_spec(spec, positions, solution, initial=None):
    """Independent geometric identities, without calling the production solver."""
    contract=spec['protocol']['gt_spec'];name=contract['resolver']
    p=[float(v) for v in positions[contract['point_object_id']][:2]]
    a,b=([float(v) for v in positions[k][:2]] for k in contract['line_object_ids'])
    q=[float(v) for v in solution['point_xy']]
    dx,dy=b[0]-a[0],b[1]-a[1];norm=math.hypot(dx,dy)
    _require(norm>0 and all(math.isfinite(v) for v in q),'Invalid independent geometry input')
    signed=lambda xy: (dx*(xy[1]-a[1])-dy*(xy[0]-a[0]))/norm
    # Separate scalar identity: distance between the perpendicular foot and
    # endpoint midpoint is the source's absolute along-axis midpoint offset.
    # This does not call the solver or construct its projected point.
    mx,my=(a[0]+b[0])*.5,(a[1]+b[1])*.5
    midpoint_separation=abs((p[0]-mx)*dx+(p[1]-my)*dy)/norm
    minimum_midpoint_separation=contract.get('minimum_projection_midpoint_separation_m',.03)
    midpoint_quality=bool(math.isfinite(minimum_midpoint_separation)
                          and minimum_midpoint_separation>=.03
                          and midpoint_separation>=minimum_midpoint_separation)
    if name=='ProjectOntoLine':
        collinear=abs(signed(q))
        perpendicular=abs((p[0]-q[0])*dx+(p[1]-q[1])*dy)/norm
        residuals={'target_line_distance_m':collinear,'displacement_parallel_residual_m':perpendicular}
    elif name=='ReflectAcrossLine':
        midpoint=[(p[i]+q[i])*.5 for i in range(2)]
        residuals={'midpoint_line_distance_m':abs(signed(midpoint)),
                   'displacement_parallel_residual_m':abs((p[0]-q[0])*dx+(p[1]-q[1])*dy)/norm,
                   'signed_distance_opposition_residual_m':abs(signed(p)+signed(q))}
    else:raise ValueError('Unknown independent geometry contract')
    tolerance=1e-7
    return {'success':midpoint_quality and all(value<=tolerance for value in residuals.values()),
            'independent_geometric_residuals_m':residuals,'tolerance_m':tolerance,
            'source_position_necessity_screen':{
                'projection_midpoint_separation_m':midpoint_separation,
                'minimum_projection_midpoint_separation_m':minimum_midpoint_separation,
                'midpoint_centred_reflection_shortcut_error_m':2*midpoint_separation,
                'passed':midpoint_quality},
            'independent_method':'Line determinant, displacement dot product and signed-side identities',
            'production_solver_called':False}

