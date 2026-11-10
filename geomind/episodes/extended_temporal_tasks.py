"""Measured-state contracts for the remaining canonical temporal tasks.

These functions supply task specifications to the existing registered assembly
pipeline. Motion samples are PD scene-event commands, never observations or GT.
The current variants are deliberately bounded: displacement change, one current
vacancy among four visible slots, and one source-region identity through a
visible cyclic mix. Structured identity observations are not a visual benchmark.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np


TASK_IDS = ('DetectSceneChange_v0', 'UpdateEmptySlot_v0', 'TrackObjectOriginClutter_v0')
_GRAY = [.52, .54, .57]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cube(name, xy, color, *, size=.04, role='candidate'):
    return {'name': name, 'kind': 'primitive_cuboid', 'object_category': 'cube',
            'role': role, 'size': [size]*3, 'pose': {'xy': list(xy), 'yaw': 0., 'z_policy': 'tabletop'},
            'visual_color': list(color), 'mass': .08,
            'physics': {'mass': .08, 'rigid_body': {'rigid_body_enabled': True,
                'kinematic_enabled': False}, 'material': {'static_friction': 1.15,
                'dynamic_friction': .96, 'restitution': 0.}}}


def _points(value):
    _require(isinstance(value, dict) and value, 'Measured object positions are required')
    result={}
    for name, point in value.items():
        if isinstance(point, dict):point=point.get('position')
        a=np.asarray(point, dtype=float)
        _require(a.shape in ((2,), (3,)) and np.isfinite(a).all(), 'Positions must be finite XY or XYZ')
        result[name]=a[:2]
    return result


def _contract(spec):
    gt=spec.get('protocol', {}).get('gt_spec', {})
    _require(gt.get('resolver') in {'DetectSceneChange','UpdateEmptySlot','TrackObjectOrigin'},
             'Unsupported canonical temporal resolver')
    return gt


def _unique_source(gt, positions):
    center=positions[gt['region_anchor']]+np.asarray(gt['region_center_offset_xy_m'])
    half=np.asarray(gt['region_half_extents_xy_m'])
    inside=[name for name in gt['candidate_object_ids']
            if np.all(np.abs(positions[name]-center) <= half-gt['region_boundary_margin_m'])]
    _require(len(inside)==1, 'Source region must contain exactly one candidate with a boundary margin')
    return inside[0], center


def _slot_positions(gt, positions):
    return positions[gt['slot_anchor']]+np.asarray(gt['slot_offsets_xy_m'])


def _occupancy(gt, positions):
    slots=_slot_positions(gt, positions)
    occupants={}
    for name in gt['occupant_object_ids']:
        distances=np.linalg.norm(slots-positions[name],axis=1)
        choices=np.flatnonzero(distances <= gt['slot_center_tolerance_m'])
        _require(len(choices)==1, 'Each slot occupant must be at one measured slot centre')
        slot=int(choices[0])
        _require(slot not in occupants, 'A slot cannot contain two occupants')
        occupants[slot]=name
    empty=sorted(set(range(len(slots)))-set(occupants))
    _require(len(empty)==1, 'Exactly one current slot must be empty')
    return slots, occupants, empty[0]


def build_spec(task_id, scene_id=65, seed=0):
    _require(task_id in TASK_IDS, 'Unknown canonical temporal task')
    _require(type(seed) is int and seed>=0, 'Seed must be a nonnegative integer')
    _require(type(scene_id) is int and 1<=scene_id<=102, 'Invalid scene id')
    if task_id=='DetectSceneChange_v0':
        colors=([.85,.12,.12],[.12,.28,.85],[.12,.67,.22],[.93,.73,.10],[.58,.30,.15])
        names=['item_a','item_b','item_c','item_d','item_e']
        xy=[(.25,-.18),(.40,-.18),(.55,-.18),(.25,.12),(.55,.12)]
        objects=[_cube(n,p,c) for n,p,c in zip(names,xy,colors)]
        gt={'resolver':'DetectSceneChange','candidate_object_ids':names,
            'change_threshold_m':.03,'ambiguity_margin_m':.006,
            'maximum_unchanged_drift_m':.012}
        actor=names[seed%len(names)]
        intervention={'duration_s':4.,'event_object':actor,
                      'translation_xy_m':[0.,.20 if seed%len(names)<3 else -.20]}
        instruction='记住桌上方块的位置，等画面恢复后，把刚才被移动的那个方块拿起来。'
        variant='UniqueDisplacementOverThreeCentimetres'
    elif task_id=='UpdateEmptySlot_v0':
        objects=[_cube('slot_reference',(.18,-.26),[.83,.84,.85],size=.025,role='context_anchor'),
                 _cube('occupant_a',(.24,-.10),[.85,.12,.12]),
                 _cube('occupant_b',(.42,-.10),[.12,.28,.85]),
                 _cube('occupant_c',(.24,.10),[.12,.67,.22]),
                 _cube('brown_block',(.22,.26),[.52,.28,.13],role='moving_object')]
        gt={'resolver':'UpdateEmptySlot','slot_anchor':'slot_reference',
            'slot_offsets_xy_m':[[.06,.16],[.24,.16],[.06,.36],[.24,.36]],
            'slot_center_tolerance_m':.025,'occupant_object_ids':['occupant_a','occupant_b','occupant_c'],
            'target_object_id':'brown_block','candidate_object_ids':['occupant_a','occupant_b','occupant_c']}
        intervention={'duration_s':4.,'event_object':'occupant_b','destination_slot_index':3}
        instruction='看清四个格子的占用情况。方块移动结束后，把棕色方块放到现在空着的那个格子里。'
        variant='CurrentVacancyAfterOneOccupancyTransfer'
    else:
        names=['item_a','item_b','item_c','item_d']
        xy=[(.30,-.15),(.48,-.15),(.48,.15),(.30,.15)]
        objects=[_cube(n,p,_GRAY) for n,p in zip(names,xy)]
        objects.append(_cube('blue_reference',(.17,-.25),[.10,.28,.88],size=.025,role='context_anchor'))
        gt={'resolver':'TrackObjectOrigin','candidate_object_ids':names,'region_anchor':'blue_reference',
            'region_center_offset_xy_m':[.13,.10],'region_half_extents_xy_m':[.055,.055],
            'region_boundary_margin_m':.008,'minimum_source_displacement_m':.08,
            'minimum_final_candidate_separation_m':.065}
        intervention={'duration_s':4.,'cycle_object_ids':names}
        instruction='看着这些灰色方块移动，结束后，把最初在蓝色方框里的那个灰色方块拿起来。'
        variant='InitiallyUniqueBlueRegionSourceThroughVisibleCycle'
    protocol={'module_name':'canonical_extension','gt_spec':gt,'intervention':intervention,
              'temporal_required':True,'candidate_object_ids':list(gt['candidate_object_ids']),
              'query_text':instruction,
              'history_requirement':'observe.End source identity' if gt['resolver']=='TrackObjectOrigin' else
                  'observe.End displacement comparison' if gt['resolver']=='DetectSceneChange' else
                  'Current occupancy suffices; history only verifies that an occupancy event occurred'}
    if gt['resolver']=='UpdateEmptySlot':protocol['target_object']={'object_id':'brown_block'}
    semantics={obj['name']:{'shape':'方块','color':
                '灰色' if task_id.startswith('TrackObjectOrigin') and obj['name'].startswith('item_') else
                '棕色' if obj['name']=='brown_block' else
                '蓝色' if obj['name']=='blue_reference' else '彩色'} for obj in objects}
    protocol['object_semantics']=semantics
    return {'instruction':instruction,'runtime_objects':objects,'protocol':protocol,
            'canonical_variant':variant,'canonical_definition_version':'1.0',
            'extension_layout':{'policy':'rigid_translation','safe_shift_m':.02,
                'reason':'Preserve slot/source geometry and nonintersecting event paths'},
            'input_contract':{'structured_identity_visible':True,'model_evaluated':False,
                'source_identity_label_visible_at_query':False if gt['resolver']=='TrackObjectOrigin' else None,
                'visual_identity_tracking_performance_claimed':False}}


def plan_spec(spec, positions, seed=0):
    gt=_contract(spec);actual=_points(positions)
    _require({o['name'] for o in spec['runtime_objects']}==set(actual), 'Motion plan identities differ')
    intervention=spec['protocol']['intervention'];starts={};ends={}
    if gt['resolver']=='DetectSceneChange':
        name=intervention['event_object'];starts[name]=actual[name]
        ends[name]=actual[name]+np.asarray(intervention['translation_xy_m'])
    elif gt['resolver']=='UpdateEmptySlot':
        slots,_,vacancy=_occupancy(gt,actual)
        _require(vacancy==intervention['destination_slot_index'],'Slot transfer must enter the initial vacancy')
        name=intervention['event_object'];starts[name]=actual[name];ends[name]=slots[vacancy]
    else:
        names=intervention['cycle_object_ids'];_unique_source(gt,actual)
        for i,name in enumerate(names):starts[name]=actual[name];ends[name]=actual[names[(i+1)%len(names)]]
    waypoints=[]
    for i in range(41):
        t=i/40.;smooth=t*t*(3-2*t)
        waypoints.append({'progress':t,'positions_xy_m':{
            name:(start+(ends[name]-start)*smooth).tolist() for name,start in starts.items()}})
    # This is a geometric command-path screen. Actual object separations and
    # support are checked from PhysX trajectories by the delivery pipeline.
    minimum=float('inf')
    for waypoint in waypoints:
        all_positions={name:waypoint['positions_xy_m'].get(name,p) for name,p in actual.items()}
        names=list(all_positions)
        for i,a in enumerate(names):
            for b in names[i+1:]:minimum=min(minimum,math.dist(all_positions[a],all_positions[b]))
    _require(minimum>.065,'Authored event paths have insufficient centre clearance')
    return {'required':True,'reference_key':'observe.End','seed':seed,
            'planned_event_is_ground_truth':False,'requires_measured_initial_and_current':True,
            'mechanism':'historical_identity_tracking' if gt['resolver']=='TrackObjectOrigin' else 'state_reindexing',
            'object_names':list(actual),'initial_positions_m':{n:p.tolist() for n,p in actual.items()},
            'script_name':gt['resolver'],'intervention':deepcopy(intervention),
            'actual_movement':{'motion_mode':'canonical_smoothstep_translation',
                'motion_function':__name__+'.plan_spec','duration_s':intervention['duration_s'],
                'waypoints':waypoints,'object_names':list(starts),
                'planned_only':True,'requires_runtime_measurement':True,
                'geometric_minimum_command_centre_distance_m':minimum,
                'event_policy':'External PD scene events only; never robot manipulation or observed GT'}}


def solve_spec(spec, positions, initial=None):
    gt=_contract(spec);actual=_points(positions)
    _require(initial is not None,'Measured observe.End history is required for temporal event validation')
    before=_points(initial)
    _require(set(before)==set(actual),'Initial/current identities differ')
    resolver=gt['resolver']
    if resolver=='DetectSceneChange':
        distances={n:float(np.linalg.norm(actual[n]-before[n])) for n in gt['candidate_object_ids']}
        threshold=gt['change_threshold_m'];margin=gt['ambiguity_margin_m']
        _require(all(abs(d-threshold)>margin for d in distances.values()),'Change is ambiguous at threshold')
        changed=[n for n,d in distances.items() if d>threshold]
        _require(len(changed)==1,'Exactly one measured changed candidate is required')
        _require(all(d<=gt['maximum_unchanged_drift_m'] for n,d in distances.items() if n!=changed[0]),
                 'Unchanged candidate drift violates the unique-movement contract')
        return {'selected':changed[0],'calculation':{'measured_displacements_m':distances,
                'change_threshold_m':threshold,'planned_positions_used_as_gt':False}}
    if resolver=='UpdateEmptySlot':
        slots,occupants,vacancy=_occupancy(gt,actual)
        _,_,old_vacancy=_occupancy(gt,before)
        _require(vacancy!=old_vacancy,'Occupancy event did not change the vacant slot')
        return {'point_xy':slots[vacancy].tolist(),'calculation':{
            'initial_empty_slot':old_vacancy,'current_empty_slot':vacancy,
            'measured_occupancy':{str(k):v for k,v in occupants.items()},
            'measured_slot_centres_xy_m':slots.tolist(),
            'history_necessary_for_answer':False,'history_used_for_event_validation':True,
            'planned_positions_used_as_gt':False}}
    selected,center=_unique_source(gt,before)
    moved=float(np.linalg.norm(actual[selected]-before[selected]))
    _require(moved>=gt['minimum_source_displacement_m'],'Source candidate did not physically leave its source')
    names=gt['candidate_object_ids']
    separation=min(math.dist(actual[a],actual[b]) for i,a in enumerate(names) for b in names[i+1:])
    _require(separation>=gt['minimum_final_candidate_separation_m'],'Mixed candidates overlap or are ambiguous')
    current_in_source,_=_unique_source(gt,actual)
    _require(current_in_source!=selected,'Final source-region lookup would solve the initial-origin query')
    return {'selected':selected,'calculation':{'initial_source_region_center_xy_m':center.tolist(),
            'measured_source_displacement_m':moved,'current_region_occupant':current_in_source,
            'minimum_final_candidate_separation_m':separation,'history_necessary_for_answer':True,
            'planned_positions_used_as_gt':False,'stable_structured_ids_available':True,
            'visual_tracking_performance_claimed':False}}


def check_spec(spec, positions, solution, initial=None):
    """Independent direct predicates; deliberately never call ``solve_spec``."""
    try:
        gt=_contract(spec);now=_points(positions);old=_points(initial)
        _require(set(now)==set(old),'History identities differ')
        selected=solution.get('selected');point=solution.get('point_xy')
        if gt['resolver']=='DetectSceneChange':
            candidate=selected
            _require(candidate in gt['candidate_object_ids'],'Wrong candidate identity')
            moved=math.dist(now[candidate],old[candidate])
            _require(moved>gt['change_threshold_m']+gt['ambiguity_margin_m'],'Selected candidate did not change')
            for name in gt['candidate_object_ids']:
                if name!=candidate:
                    _require(math.dist(now[name],old[name])<=gt['maximum_unchanged_drift_m'],
                             'More than one candidate changed')
            return {'success':True,'independent_method':'Direct per-body historical displacement inequalities'}
        if gt['resolver']=='UpdateEmptySlot':
            offsets=gt['slot_offsets_xy_m'];anchor=now[gt['slot_anchor']]
            proposed=np.asarray(point,dtype=float)
            _require(proposed.shape==(2,) and np.isfinite(proposed).all(),'Invalid proposed slot point')
            proposed_indices=[i for i,p in enumerate(offsets) if math.dist(anchor+np.asarray(p),proposed)<=.001]
            _require(len(proposed_indices)==1,'Proposed point is not a slot centre')
            free=[];initial_free=[]
            for i,offset in enumerate(offsets):
                center=anchor+np.asarray(offset)
                occupants=[n for n in gt['occupant_object_ids'] if math.dist(now[n],center)<=gt['slot_center_tolerance_m']]
                _require(len(occupants)<=1,'Duplicate slot occupancy')
                if not occupants:free.append(i)
                initial_center=old[gt['slot_anchor']]+np.asarray(offset)
                if not any(math.dist(old[n],initial_center)<=gt['slot_center_tolerance_m'] for n in gt['occupant_object_ids']):initial_free.append(i)
            _require(free==proposed_indices and len(initial_free)==1 and initial_free!=free,'Current vacancy or occupancy update is incorrect')
            _require(all(any(math.dist(now[n],anchor+np.asarray(p))<=gt['slot_center_tolerance_m'] for p in offsets)
                         for n in gt['occupant_object_ids']),'An occupant is between slots')
            return {'success':True,'independent_method':'Direct proposed-slot emptiness and complementary slot coverage'}
        _require(selected in gt['candidate_object_ids'],'Wrong source candidate identity')
        center=old[gt['region_anchor']]+np.asarray(gt['region_center_offset_xy_m'])
        half=gt['region_half_extents_xy_m'];margin=gt['region_boundary_margin_m']
        matches=[]
        for name in gt['candidate_object_ids']:
            if all(abs(old[name][axis]-center[axis])<=half[axis]-margin for axis in range(2)):matches.append(name)
        _require(matches==[selected],'Selected identity did not originate uniquely in blue region')
        _require(math.dist(old[selected],now[selected])>=gt['minimum_source_displacement_m'],'Source did not move')
        for i,a in enumerate(gt['candidate_object_ids']):
            for b in gt['candidate_object_ids'][i+1:]:
                _require(math.dist(now[a],now[b])>=gt['minimum_final_candidate_separation_m'],'Candidates overlap')
        current_center=now[gt['region_anchor']]+np.asarray(gt['region_center_offset_xy_m'])
        current_members=[n for n in gt['candidate_object_ids']
                         if all(abs(now[n][axis]-current_center[axis])<=half[axis]-margin for axis in range(2))]
        _require(len(current_members)==1 and current_members[0]!=selected,
                 'Current-region shortcut was not rejected or final region membership is ambiguous')
        return {'success':True,'independent_method':'Initial source rectangle membership and actual movement/spacing predicates'}
    except (ValueError,KeyError,TypeError) as exc:
        return {'success':False,'reason':str(exc),'independent_method':'Direct measured-state predicates'}


def draw_context(stage, spec, objects, height):
    """Draw visible slot/source outlines without collision or support geometry."""
    from pxr import Gf, UsdGeom
    gt=_contract(spec)
    if gt['resolver']=='DetectSceneChange':return []
    positions={obj.get('object_id',obj.get('name')):obj.get('position_w_m',obj.get('bbox_center_w_m')) for obj in objects}
    actual=_points(positions);paths=[]
    if gt['resolver']=='UpdateEmptySlot':
        frames=[(center,[.053,.053],[.85,.73,.18]) for center in _slot_positions(gt,actual)]
    else:
        center=actual[gt['region_anchor']]+np.asarray(gt['region_center_offset_xy_m'])
        frames=[(center,gt['region_half_extents_xy_m'],[.10,.35,.95])]
    for i,(center,half,color) in enumerate(frames):
        path=f'/World/TaskContext/CanonicalFrame_{i:02d}'
        curve=UsdGeom.BasisCurves.Define(stage,path)
        curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([5])
        xy=[center+np.array([x*half[0],y*half[1]]) for x,y in ((-1,-1),(1,-1),(1,1),(-1,1),(-1,-1))]
        curve.CreatePointsAttr([Gf.Vec3f(float(p[0]),float(p[1]),height+.003) for p in xy])
        curve.CreateWidthsAttr([.003]);curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        paths.append(path)
    return paths
