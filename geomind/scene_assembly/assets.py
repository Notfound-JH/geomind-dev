"""Resolve local object descriptors without importing Isaac Sim or executing old scripts."""
from __future__ import annotations
from geomind_lab.utils.portable_paths import repo_assets_root, resolve_asset_path

from pathlib import Path
import json

from geomind_lab.utils import asset_utils

ROOT = Path(__file__).resolve().parents[2]


def resolve_asset(name: str) -> dict:
    """Return the catalog scale and wxyz orientation plus a checked local path.

    Source ``orientations`` follow the existing SingleRigidPrim convention
    (wxyz), even where the old configuration named its constants differently.
    Geometry is measured after these transforms, not inferred from text names.
    """
    if name=='bin':
        raise ValueError('bin has an unverified legacy scale for its metre-based visual model; register a correct rigid/collision descriptor before assembly')
    entry = None
    category = None
    for group in ('catch_object', 'container'):
        for item in asset_utils.assets_table.get(group, []):
            if item['name'] == name and not name.startswith('cabinet_'):
                entry, category = dict(item), group
                break
    if entry is None and name in {'cup_1','cup_2','pot_1','pot_2'}:
        scale = .012 if name in {'cup_2','pot_2'} else .01
        rotations = {'cup_1':[.0019,.9643,.271,-.2635], 'cup_2':[0,1,0,0]}
        entry = {'name':name,'usd_path':str(Path(asset_utils.GEOMIND_ASSETS_ROOT)/'layout'/f'{name}.usda'),
                 'scale':[scale]*3,'orientations':[rotations.get(name,[0,0,0,1])]}
        category = 'portable_decoration'
    if entry is None and name in {'basket_4','box_1','box_2','box_3','blue_tin','tin'}:
        # File-only candidates do not have a trusted scale descriptor. Explicitly
        # reject rather than silently invent dimensions or use a unit guess.
        raise ValueError(f'{name} exists as a candidate file but has no registered scale/orientation; register it before assembly')
    if entry is None:
        raise ValueError(f'Unknown interactive asset: {name}')
    source = Path(resolve_asset_path(entry['usd_path']))
    path = source
    if not path.is_file() and '/Assets/ArchVis/' in str(path):
        path = Path(resolve_asset_path(str(path)))
    if not path.is_file():
        raise ValueError(f'Asset file is unavailable: {source}')
    orientation = [float(x) for x in entry.get('orientations',[[1,0,0,0]])[0]]
    norm = sum(v*v for v in orientation)**.5
    if norm < 1e-8:
        raise ValueError(f'Invalid orientation for asset {name}')
    return {'name':name, 'category':category, 'usd_path':str(path.resolve()),
            'catalog_path':str(source), 'scale':[float(x) for x in entry['scale']],
            'orientation_wxyz':[v/norm for v in orientation],
            'path_remapped':source != path, 'descriptor_source':str(asset_utils.ASSET_CATALOG_PATH)}


def resolve_scene(scene_id: int) -> dict:
    path = repo_assets_root()/'scenes/tabletop_cleared_v1'/str(scene_id)/f'{scene_id}.usda'
    if not path.is_file():
        raise ValueError(f'Cleared scene {scene_id} has not been saved: {path}')
    name = None
    index_path = path.parents[1]/'scene_index.json'
    if index_path.is_file():
        rows = json.loads(index_path.read_text())['scenes']
        name = next((row['scene_name'] for row in rows if row['scene_id']==scene_id), None)
    return {'scene_id':scene_id,'scene_name':name,'usd_path':str(path.resolve())}
