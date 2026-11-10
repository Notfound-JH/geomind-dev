"""Bounded canonical task adapters composed into the existing episode pipeline.

The taxonomy remains authoritative. These modules supply concrete contracts,
not another task registry or a replacement simulator.
"""
from __future__ import annotations

from copy import deepcopy
import importlib


_MODULES = ('extended_static_tasks', 'extended_temporal_tasks', 'extended_geometry_tasks')


def modules():
    return [importlib.import_module('geomind.episodes.' + name) for name in _MODULES]


def task_ids():
    return tuple(task for module in modules() for task in module.TASK_IDS)


def module_for(task_id):
    for module in modules():
        if task_id in module.TASK_IDS:
            return module
    raise ValueError(f'No canonical extension contract for {task_id!r}')


def build_spec(task_id, scene_id=65, seed=0):
    spec=module_for(task_id).build_spec(task_id, scene_id=scene_id, seed=seed)
    spec['extension_task_id']=task_id
    return spec


def response(task_id):
    return 'ReportPoint' if task_id in {'ProjectOntoLine_v0', 'ReflectAcrossLine_v0', 'UpdateEmptySlot_v0'} else 'SelectObject'


def temporal_required(spec):
    return bool(spec['protocol'].get('temporal_required', False))


def plan_temporal(spec, positions, seed):
    module=module_for(spec['extension_task_id'])
    if temporal_required(spec):
        return module.plan_spec(spec, positions, seed)
    return {'required':False, 'reference_key':'observe.End', 'seed':seed,
            'planned_event_is_ground_truth':False, 'requires_measured_initial_and_current':False}


def solve(spec, positions, initial=None):
    return module_for(spec['extension_task_id']).solve_spec(spec, positions, initial=initial)


def check(spec, positions, solution, initial=None):
    return module_for(spec['extension_task_id']).check_spec(spec, positions, solution, initial=initial)


def draw_context(stage, spec, objects, height):
    module=module_for(spec['extension_task_id'])
    if hasattr(module, 'draw_context'):
        return module.draw_context(stage, spec, objects, height)
    return []


def translate_layout(runtime_objects, seed):
    """One deterministic structure-preserving sample, never independent jitter."""
    import random
    rng=random.Random(seed+4117)
    delta=[rng.uniform(-.012,.012),rng.uniform(-.012,.012)]
    before={obj['name']:list(obj['pose']['xy']) for obj in runtime_objects}
    for obj in runtime_objects:
        obj['pose']['xy']=[before[obj['name']][i]+delta[i] for i in range(2)]
    return {'kind':'canonical_extension_rigid_xy_translation','seed':seed,
            'translation_xy_m':delta,'original_positions_xy_m':before,
            'positions_xy_m':{obj['name']:list(obj['pose']['xy']) for obj in runtime_objects},
            'preserved':'Object scale, identity, topology, relative positions, margins and declared task rule',
            'independent_jitter':False,'resampled':False}
