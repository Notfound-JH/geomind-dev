"""Bounded, measured-state contracts for five canonical selection tasks.

These contracts use the existing registered assembly/export/action pipeline.
The graph variants deliberately define their topology in visible table marks;
nearness is never substituted for graph connectivity. All object names below
are evaluator-side identities and are rebound to neutral public aliases.
"""
from __future__ import annotations

from copy import deepcopy
import heapq
import math
import random


TASK_IDS = tuple(name + '_v0' for name in (
    'SelectByBranchPosition', 'SelectByRegion', 'SelectByPathDistance',
    'SelectByTwoRelations', 'SelectByIndirectReference'))

_BROWN = [.63, .48, .33]
_COLORS = {'brown': _BROWN, 'red': [.85, .10, .08],
           'blue': [.08, .25, .83], 'yellow': [.94, .75, .08]}
_COLOR_NAMES = {'brown': '棕色', 'red': '红色', 'blue': '蓝色', 'yellow': '黄色'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _name(task_id):
    name = str(task_id).removesuffix('_v0')
    _require(name + '_v0' in TASK_IDS, 'Unsupported extended static task')
    return name


def _cube(name, xy, color='brown'):
    return {'name': name, 'kind': 'primitive_cuboid', 'object_category': '方块',
            'size': [.04, .04, .04], 'visual_color': list(_COLORS[color]),
            'physics': {'mass': .05, 'collision': {'contact_offset': .001, 'rest_offset': 0.},
                        'rigid_body': {'rigid_body_enabled': color == 'brown'}},
            'pose': {'xy': list(xy), 'yaw': 0., 'z_policy': 'tabletop'},
            'role': 'candidate' if color == 'brown' else 'reference'}


def build_spec(task_id, scene_id=65, seed=0):
    """Build one explicit Base variant with deterministic list permutation.

    There is no per-object jitter: graph/rectangle and condition margins are
    part of the authored structure. The exporter applies one shared XY offset.
    """
    name = _name(task_id)
    _require(type(seed) is int and seed >= 0, 'seed must be nonnegative integer')
    _require(type(scene_id) is int and 1 <= scene_id <= 102, 'Invalid scene ID')
    if name == 'SelectByBranchPosition':
        objects = [_cube('n0', [.21, 0.], 'red'), _cube('n1', [.31, 0.]),
                   _cube('n2', [.39, .14], 'blue'), _cube('n3', [.48, .14]),
                   _cube('n4', [.57, .14]), _cube('n5', [.42, -.15]),
                   _cube('n6', [.56, -.15])]
        candidates = ['n1', 'n3', 'n4', 'n5', 'n6']
        gt = {'root': 'n0', 'branch_mark': 'n2', 'ordinal_after_mark': 2,
              'directed_edges': [['n0', 'n1'], ['n1', 'n2'], ['n2', 'n3'],
                                 ['n3', 'n4'], ['n1', 'n5'], ['n5', 'n6']]}
        instruction = '从红色起点沿箭头走，在分叉处走蓝色标记所在的分支，拿起蓝色标记之后的第二个棕色方块。'
        variant = 'RootedDirectedTreeMarkedBranchOrdinal'
    elif name == 'SelectByRegion':
        objects = [_cube('n0', [.29, -.11], 'blue'), _cube('n1', [.52, .16], 'yellow'),
                   _cube('n2', [.41, .04]), _cube('n3', [.21, .04]),
                   _cube('n4', [.58, -.12]), _cube('n5', [.41, .25])]
        candidates = ['n2', 'n3', 'n4', 'n5']
        gt = {'diagonal_marks': ['n0', 'n1'], 'boundary_margin_m': .025,
              'region_type': 'WorldXYAxisAlignedRectangle'}
        instruction = '拿起桌面长方形框内的棕色方块；蓝色和黄色标记是这个框的两个对角。'
        variant = 'VisibleRectangleInterior'
    elif name == 'SelectByPathDistance':
        objects = [_cube('n0', [.22, 0.], 'red'), _cube('n1', [.56, -.12]),
                   _cube('n2', [.24, .21]), _cube('n3', [.42, .23]),
                   _cube('n4', [.52, .07]), _cube('n5', [.36, .07])]
        candidates = ['n1', 'n2', 'n3', 'n4', 'n5']
        gt = {'root': 'n0', 'edge_weight_policy': 'UnitHop', 'query': 'farthest',
              'undirected_edges': [['n0', 'n1', 1.], ['n0', 'n2', 1.],
                                   ['n2', 'n3', 1.], ['n3', 'n4', 1.], ['n4', 'n5', 1.]],
              'distance_tie_tolerance': .01}
        instruction = '沿桌面连线，每段算一步，拿起从红色起点走过去所需步数最多的棕色方块。'
        variant = 'VisibleUnitEdgeShortestPathFarthest'
    elif name == 'SelectByTwoRelations':
        objects = [_cube('n0', [.31, .04], 'red'), _cube('n1', [.49, .15], 'blue'),
                   _cube('n2', [.43, .27]), _cube('n3', [.49, -.10]),
                   _cube('n4', [.23, .25]), _cube('n5', [.23, -.14])]
        candidates = ['n2', 'n3', 'n4', 'n5']
        gt = {'right_reference': 'n0', 'front_reference': 'n1',
              'right_axis': 'x', 'front_axis': 'y', 'relation_margin_m': .035}
        instruction = '以橙色箭头所指为前方，拿起既在红色标记右边、又在蓝色标记前方的棕色方块。'
        variant = 'IndependentRightAndFrontIntersection'
    else:
        objects = [_cube('n0', [.24, -.06], 'red'), _cube('n1', [.36, -.03], 'blue'),
                   _cube('n2', [.52, .20], 'blue'), _cube('n3', [.39, -.12]),
                   _cube('n4', [.22, -.19]), _cube('n5', [.55, .10])]
        candidates = ['n3', 'n4', 'n5']
        # Uniform candidate geometry permits pad-centred contact with table clearance.
        # All three candidates remain identical; references and XY relations stay fixed.
        for obj in objects:
            if obj['name'] in candidates:
                obj['size'] = [.04, .04, .05]
        gt = {'root_reference': 'n0', 'intermediate_candidates': ['n1', 'n2'],
              'distance_axes': ['x', 'y'], 'distance_margin_m': .015}
        instruction = '先找离红色标记最近的蓝色标记，再拿起离那个蓝色标记最近的棕色方块。'
        variant = 'TwoHopNearestReferenceThenCandidate'
    rng = random.Random(seed + 62419)
    rng.shuffle(objects)
    semantics = {obj['name']: {'color': _COLOR_NAMES[next(key for key, value in _COLORS.items()
                                                   if value == obj['visual_color'])], 'shape': '方块'}
                 for obj in objects}
    gt = {'resolver': name, 'definition_version': '1.0', 'bounded_variant': variant, **gt}
    protocol = {'module_name': 'canonical_extension', 'family': 'canonical_extension',
                'variant': variant, 'task_version': 'base', 'query_text': instruction,
                'candidate_object_ids': candidates, 'object_semantics': semantics, 'gt_spec': gt}
    if name == 'SelectByPathDistance':
        # Private evaluator control: keep the Euclidean-farthest leaf as the
        # wrong pick across alias permutations and shared layout translations.
        protocol['negative_control_object_id'] = 'n1'
    return {'task_name': name, 'instruction': instruction, 'goal_sequence': [],
            'runtime_objects': objects, 'protocol': protocol,
            'object_positions': {obj['name']: {'position': obj['pose']['xy']} for obj in objects},
            'layout_randomization_contract': {'kind': 'SharedXYTranslation', 'seed': seed,
                'applied': False, 'translation_m': [0., 0.], 'independent_object_jitter_allowed': False,
                'preserves': ['graph topology', 'rectangle membership', 'relational margins', 'candidate identity']},
            'quality_contract': {'instruction_difficulty': 'Base, one bounded graph/region/composition rule',
                'visual_context_required': name in {'SelectByBranchPosition', 'SelectByRegion', 'SelectByPathDistance', 'SelectByTwoRelations'},
                'solver_uses_actual_positions': True, 'selection_requires_true_grasp': True}}


def _positions(spec, positions):
    names = {obj['name'] for obj in spec['runtime_objects']}
    _require(set(positions) == names, 'Measured positions must cover exactly the authored objects')
    result = {}
    for key, value in positions.items():
        value = value.get('position') if isinstance(value, dict) else value
        _require(isinstance(value, (list, tuple)) and len(value) in (2, 3)
                 and all(type(x) in (int, float) and math.isfinite(x) for x in value),
                 'Nonfinite or invalid measured positions')
        result[key] = list(map(float, value))
    return result


def _contract(spec, positions):
    protocol = spec['protocol']; gt = protocol['gt_spec']; name = _name(gt['resolver'])
    _require(protocol.get('module_name') == 'canonical_extension', 'Wrong static protocol module')
    p = _positions(spec, positions); candidates = protocol['candidate_object_ids']
    _require(isinstance(candidates, list) and candidates and len(set(candidates)) == len(candidates)
             and set(candidates) <= set(p), 'Invalid candidate object set')
    return name, gt, p, candidates


def _one(values, message):
    _require(len(values) == 1, message)
    return values[0]


def _tree(gt, p):
    root = gt['root']; edges = gt['directed_edges']
    _require(root in p and isinstance(edges, list) and len(edges) == len(p)-1, 'Incomplete rooted tree')
    children = {key: [] for key in p}; parent = {}
    for edge in edges:
        _require(isinstance(edge, list) and len(edge) == 2, 'Malformed directed edge')
        a, b = edge
        _require(a in p and b in p and a != b and b not in parent and b != root, 'Invalid tree parent/root')
        children[a].append(b); parent[b] = a
    seen = set(); todo = [root]
    while todo:
        node = todo.pop()
        _require(node not in seen, 'Cyclic rooted tree')
        seen.add(node); todo += children[node]
    _require(seen == set(p), 'Rooted tree is disconnected')
    return children, parent


def _graph(gt, p):
    _require(gt.get('edge_weight_policy') == 'UnitHop', 'Unsupported edge-weight semantics')
    root = gt['root']; _require(root in p, 'Missing graph root')
    graph = {key: [] for key in p}; seen = set()
    for edge in gt['undirected_edges']:
        _require(isinstance(edge, list) and len(edge) == 3, 'Malformed weighted edge')
        a, b, weight = edge
        _require(a in p and b in p and a != b and weight == 1., 'Invalid unit edge')
        pair = tuple(sorted((a, b)))
        _require(pair not in seen, 'Duplicate graph edge'); seen.add(pair)
        graph[a].append((b, float(weight))); graph[b].append((a, float(weight)))
    return graph


def _near(candidates, reference, p, margin):
    distances = sorted((math.dist(p[name][:2], p[reference][:2]), name) for name in candidates)
    _require(len(distances) >= 2 and distances[1][0]-distances[0][0] > margin,
             'Nearest reference is ambiguous within margin')
    return distances[0][1], dict((name, distance) for distance, name in distances)


def solve_spec(spec, positions, initial=None):
    """Solve from actual measured state, without any authored expected answer."""
    name, gt, p, candidates = _contract(spec, positions)
    evidence = {'definition_version': gt['definition_version'], 'bounded_variant': gt['bounded_variant'],
                'measured_current_positions_used': True, 'planned_positions_used_as_observations': False}
    if name == 'SelectByBranchPosition':
        children, parents = _tree(gt, p); mark = gt['branch_mark']
        _require(mark in p and mark in parents and len(children[parents[mark]]) >= 2,
                 'Blue mark must identify one branch at a real fork')
        node = mark; ordered = []
        while children[node]:
            _require(len(children[node]) == 1, 'Marked branch forks again without a branch rule')
            node = children[node][0]
            if node in candidates: ordered.append(node)
        rank = gt['ordinal_after_mark']
        _require(type(rank) is int and 1 <= rank <= len(ordered), 'Branch ordinal is outside branch')
        selected = ordered[rank-1]
        without_mark = []
        pending = [(child, 0) for child in children[parents[mark]]]
        while pending:
            node, count = pending.pop(); count += node in candidates
            if count == rank:
                without_mark.append(node)
            else:
                pending += [(child, count) for child in children[node]]
        _require(len(set(without_mark)) >= 2 and len(ordered) >= 2,
                 'Branch marker or ordinal is unnecessary for this fixture')
        evidence.update(marked_branch_candidates=ordered, selected_ordinal=rank,
                        root=gt['root'], fork=parents[mark], topology_is_not_euclidean_order=True)
        evidence.update(without_branch_mark_candidates=sorted(set(without_mark)),
                        without_ordinal_candidates=ordered, branch_and_ordinal_both_necessary=True)
    elif name == 'SelectByRegion':
        _require(gt.get('region_type') == 'WorldXYAxisAlignedRectangle', 'Unsupported region geometry')
        a, b = gt['diagonal_marks']; _require(a in p and b in p and a != b, 'Missing diagonal marks')
        low = [min(p[a][i], p[b][i]) for i in (0, 1)]
        high = [max(p[a][i], p[b][i]) for i in (0, 1)]
        margin = gt['boundary_margin_m']; _require(margin > 0 and min(high[i]-low[i] for i in (0, 1)) > 2*margin,
                                                    'Degenerate or too-small region')
        signed = {obj: min(p[obj][i]-low[i] for i in (0, 1)) for obj in candidates}
        for obj in candidates:
            signed[obj] = min(signed[obj], *(high[i]-p[obj][i] for i in (0, 1)))
            _require(abs(signed[obj]) > margin, 'Region candidate is ambiguous near boundary')
        selected = _one([obj for obj in candidates if signed[obj] > margin], 'Region must contain one candidate')
        evidence.update(region_low_xy_m=low, region_high_xy_m=high, signed_boundary_distance_m=signed,
                        without_region_constraint_candidates=list(candidates))
    elif name == 'SelectByPathDistance':
        graph = _graph(gt, p); distances = {key: math.inf for key in p}; distances[gt['root']] = 0.
        todo = [(0., gt['root'])]
        while todo:
            distance, node = heapq.heappop(todo)
            if distance != distances[node]: continue
            for other, weight in graph[node]:
                if distance+weight < distances[other]:
                    distances[other] = distance+weight; heapq.heappush(todo, (distances[other], other))
        _require(all(math.isfinite(x) for x in distances.values()), 'Path graph contains unreachable objects')
        ordered = sorted(candidates, key=lambda obj: distances[obj], reverse=True)
        _require(len(ordered) >= 2 and distances[ordered[0]]-distances[ordered[1]] > gt['distance_tie_tolerance'],
                 'Farthest path candidate is ambiguous')
        selected = ordered[0]
        euclidean = max(candidates, key=lambda obj: math.dist(p[obj][:2], p[gt['root']][:2]))
        _require(euclidean != selected, 'Path fixture does not distinguish path and Euclidean distance')
        evidence.update(shortest_path_distance=distances, euclidean_farthest=euclidean,
                        path_vs_euclidean_target_differs=True)
    elif name == 'SelectByTwoRelations':
        _require(gt.get('right_axis') == 'x' and gt.get('front_axis') == 'y', 'Unsupported relation axes')
        r, f = gt['right_reference'], gt['front_reference']; margin = gt['relation_margin_m']
        _require(r in p and f in p and r != f and margin > 0, 'Invalid relation references')
        dx = {obj: p[obj][0]-p[r][0] for obj in candidates}
        dy = {obj: p[obj][1]-p[f][1] for obj in candidates}
        _require(all(abs(v) > margin for v in [*dx.values(), *dy.values()]), 'Ambiguous directional boundary')
        right = {obj for obj in candidates if dx[obj] > margin}; front = {obj for obj in candidates if dy[obj] > margin}
        selected = _one(sorted(right & front), 'Two-relation intersection must be unique')
        _require(len(right) >= 2 and len(front) >= 2 and right-front and front-right,
                 'Each spatial condition must have its own distractor')
        evidence.update(right_candidates=sorted(right), front_candidates=sorted(front),
                        without_right_candidates=sorted(front), without_front_candidates=sorted(right),
                        each_condition_necessary=True, right_offset_m=dx, front_offset_m=dy)
    else:
        margin = gt['distance_margin_m']; blues = gt['intermediate_candidates']; root = gt['root_reference']
        _require(root in p and len(blues) >= 2 and len(set(blues)) == len(blues)
                 and set(blues) <= set(p) and not set(blues) & set(candidates), 'Invalid intermediate references')
        intermediate, first_distances = _near(blues, root, p, margin)
        selected, second_distances = _near(candidates, intermediate, p, margin)
        direct, _ = _near(candidates, root, p, margin)
        alternatives = {blue: _near(candidates, blue, p, margin)[0] for blue in blues if blue != intermediate}
        _require(direct != selected and any(value != selected for value in alternatives.values()),
                 'Indirect-reference fixture permits a one-hop shortcut')
        evidence.update(intermediate_reference=intermediate, first_hop_distances_m=first_distances,
                        second_hop_distances_m=second_distances, direct_root_shortcut_target=direct,
                        alternative_intermediate_targets=alternatives, both_hops_necessary=True)
    return {'selected': selected, 'calculation': evidence}


def check_spec(spec, positions, solution, initial=None):
    """Check the selected identity through separate constraint calculations.

    This routine never invokes solve_spec. Known-answer and malformed-contract
    tests provide further independent evidence beyond this shared schema check.
    """
    name, gt, p, candidates = _contract(spec, positions)
    selected = solution.get('selected') if isinstance(solution, dict) else solution
    _require(selected in candidates, 'Selected identity is not a task candidate')
    expected = []
    if name == 'SelectByBranchPosition':
        # Reverse ancestor walks, rather than production's forward branch walk.
        edges = gt['directed_edges']; _require(len(edges) == len(p)-1, 'Incomplete rooted tree')
        parents = {b: a for a, b in edges}; root, mark = gt['root'], gt['branch_mark']
        _require(len(parents) == len(edges) and root not in parents and mark in parents,
                 'Invalid reverse tree')
        fork = parents[mark]
        _require(sum(a == fork for a, b in edges) >= 2, 'Marked reference lacks fork')
        without_mark = []
        for obj in candidates:
            chain = [obj]; current = obj
            while current != root:
                _require(current in parents and parents[current] not in chain, 'Invalid rooted ancestry')
                current = parents[current]; chain.append(current)
            if mark in chain and sum(v in candidates for v in chain[:chain.index(mark)]) == gt['ordinal_after_mark']:
                expected.append(obj)
            if fork in chain and sum(v in candidates for v in chain[:chain.index(fork)]) == gt['ordinal_after_mark']:
                without_mark.append(obj)
        _require(len(without_mark) >= 2, 'Branch mark is unnecessary')
    elif name == 'SelectByRegion':
        a, b = (p[key] for key in gt['diagonal_marks']); margin = gt['boundary_margin_m']
        for obj in candidates:
            q = p[obj]
            distances = [q[i]-min(a[i], b[i]) for i in (0, 1)] + [max(a[i], b[i])-q[i] for i in (0, 1)]
            _require(abs(min(distances)) > margin, 'Region boundary ambiguity')
            if all(distance > margin for distance in distances): expected.append(obj)
    elif name == 'SelectByPathDistance':
        # Bellman-Ford relaxation is independent of production Dijkstra's queue.
        _graph(gt, p); distances = {obj: math.inf for obj in p}; distances[gt['root']] = 0.
        for _ in range(len(p)-1):
            for a, b, weight in gt['undirected_edges']:
                old_a, old_b = distances[a], distances[b]
                distances[a] = min(old_a, old_b+weight); distances[b] = min(old_b, old_a+weight)
        _require(all(math.isfinite(x) for x in distances.values()), 'Unreachable graph candidate')
        maximum = max(distances[obj] for obj in candidates)
        expected = [obj for obj in candidates if maximum-distances[obj] <= gt['distance_tie_tolerance']]
        _require(max(candidates, key=lambda obj: math.dist(p[obj][:2], p[gt['root']][:2])) not in expected,
                 'Path fixture has Euclidean shortcut')
    elif name == 'SelectByTwoRelations':
        r, f = p[gt['right_reference']], p[gt['front_reference']]; margin = gt['relation_margin_m']
        flags = {obj: (p[obj][0]-r[0] > margin, p[obj][1]-f[1] > margin) for obj in candidates}
        _require(all(abs(p[obj][0]-r[0]) > margin and abs(p[obj][1]-f[1]) > margin for obj in candidates),
                 'Ambiguous relation boundary')
        expected = [obj for obj, pair in flags.items() if pair == (True, True)]
        _require(any(pair == (True, False) for pair in flags.values()) and any(pair == (False, True) for pair in flags.values()),
                 'A relation condition is unnecessary')
    else:
        # Compare all squared distances, avoiding the production distance sort.
        root = gt['root_reference']; blues = gt['intermediate_candidates']; margin = gt['distance_margin_m']
        squared = lambda a, b: sum((p[a][i]-p[b][i])**2 for i in (0, 1))
        nearest_blue = [a for a in blues if all(math.sqrt(squared(a, root))+margin < math.sqrt(squared(b, root)) for b in blues if b != a)]
        blue = _one(nearest_blue, 'First hop is ambiguous')
        expected = [a for a in candidates if all(math.sqrt(squared(a, blue))+margin < math.sqrt(squared(b, blue)) for b in candidates if b != a)]
        direct = min(candidates, key=lambda a: squared(a, root))
        alternatives = [min(candidates, key=lambda a: squared(a, b)) for b in blues if b != blue]
        _require(len(expected) == 1 and direct != expected[0] and any(a != expected[0] for a in alternatives),
                 'Indirect-reference condition is unnecessary')
    _require(len(expected) == 1, 'Independent constraint check has no unique answer')
    return {'success': selected == expected[0], 'expected_selected': expected[0],
            'independent_algorithm': {'SelectByBranchPosition': 'reverse_ancestor_rank',
                'SelectByRegion': 'four_halfplane_constraints', 'SelectByPathDistance': 'BellmanFord',
                'SelectByTwoRelations': 'per_object_boolean_constraints',
                'SelectByIndirectReference': 'pairwise_squared_distance_inequalities'}[name],
            'production_solver_called': False}


def draw_context(stage, spec, objects, height):
    """Draw observable graph/region/frame marks, without physical geometry."""
    from pxr import Gf, UsdGeom
    gt = spec['protocol']['gt_spec']; name = _name(gt['resolver'])
    points = {obj['object_id']: obj.get('bbox_center_w_m', obj['position_w_m']) for obj in objects}
    root = '/World/TaskContext/ExtendedStatic'
    if stage.GetPrimAtPath(root): stage.RemovePrim(root)
    UsdGeom.Xform.Define(stage, root); paths = []
    def line(label, coordinates, color, width=.003):
        path = root+'/'+label; curve = UsdGeom.BasisCurves.Define(stage, path)
        curve.CreateTypeAttr('linear'); curve.CreateCurveVertexCountsAttr([len(coordinates)])
        curve.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(height)+.0025) for x, y in coordinates])
        curve.CreateWidthsAttr([width]); curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        paths.append(path)
    def arrow(label, start, end):
        dx, dy = end[0]-start[0], end[1]-start[1]; length = math.hypot(dx, dy)
        _require(length > .06, 'Graph vertices too close for visible edge')
        ux, uy = dx/length, dy/length
        low = [start[0]+ux*.026, start[1]+uy*.026]; high = [end[0]-ux*.026, end[1]-uy*.026]
        line(label, [low, high], [.96, .55, .08])
        tip = [low[i]+.65*(high[i]-low[i]) for i in (0, 1)]
        line(label+'_head', [[tip[0]-ux*.012-uy*.007, tip[1]-uy*.012+ux*.007], tip,
                            [tip[0]-ux*.012+uy*.007, tip[1]-uy*.012-ux*.007]], [.96, .55, .08])
    if name == 'SelectByBranchPosition':
        for i, (a, b) in enumerate(gt['directed_edges']): arrow(f'Edge_{i:02d}', points[a], points[b])
    elif name == 'SelectByPathDistance':
        for i, (a, b, _) in enumerate(gt['undirected_edges']):
            dx, dy = points[b][0]-points[a][0], points[b][1]-points[a][1]; length = math.hypot(dx, dy)
            _require(length > .06, 'Graph vertices too close for visible edge')
            line(f'Edge_{i:02d}', [[points[a][0]+dx/length*.026, points[a][1]+dy/length*.026],
                                 [points[b][0]-dx/length*.026, points[b][1]-dy/length*.026]], [.96, .55, .08])
    elif name == 'SelectByRegion':
        a, b = [points[obj] for obj in gt['diagonal_marks']]
        lo = [min(a[i], b[i]) for i in (0, 1)]; hi = [max(a[i], b[i]) for i in (0, 1)]
        line('Rectangle', [lo, [hi[0], lo[1]], hi, [lo[0], hi[1]], lo], [.05, .65, .68], .004)
    elif name == 'SelectByTwoRelations':
        x = min(point[0] for point in points.values())-.055
        y = min(point[1] for point in points.values())-.06
        arrow('Forward', [x, y-.02], [x, y+.07])
    return paths
