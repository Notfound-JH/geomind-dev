"""Explicit catalog USD descriptors for native episodes; poses refer to rigid-body origins."""
import hashlib
from pathlib import Path
from .resolver import _check, _fields, _number, _position


def catalog_descriptor(name, scale):
    from geomind_lab.utils.asset_utils import get_asset_info_by_name
    asset = get_asset_info_by_name('catch_object', name)
    _check(asset is not None, f'Unknown catalog asset: {name}')
    path = Path(asset['usd_path'])
    _check(path.is_file(), f'Asset missing: {path}')
    spec = {'usd_path': str(path), 'scale': [scale]*3,
            'dimensions_m': [asset['dimensions'][axis]*scale for axis in ('x','y','z')],
            'bottom_offset_m': asset['z_min_offset']*scale,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'reference_point': 'RigidBodyOrigin', 'collision': 'convexHull'}
    validate_asset(spec)
    return spec


def validate_asset(spec):
    _fields(spec, {'usd_path','scale','dimensions_m','bottom_offset_m','sha256','reference_point','collision'})
    _check(isinstance(spec['usd_path'], str) and bool(spec['usd_path']), 'Missing USD path')
    for key in ('scale', 'dimensions_m'):
        _check(all(0 < v < 100 for v in _position(spec[key])), f'Invalid asset {key}')
    _check(_number(spec['bottom_offset_m']) and -1 < spec['bottom_offset_m'] <= 0, 'Invalid asset bottom offset')
    _check(isinstance(spec['sha256'], str) and len(spec['sha256']) == 64, 'Invalid asset hash')
    _check(spec['reference_point'] == 'RigidBodyOrigin', 'Unsupported asset reference point')
    _check(spec['collision'] == 'convexHull', 'Unsupported asset collision')


def spawn_catalog_object(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files import from_files
    from isaaclab.sim import schemas
    from pxr import Usd, UsdGeom, UsdPhysics
    prim = from_files.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    # Use one rigid root with an explicit origin, irrespective of authored nested bodies.
    for child in Usd.PrimRange(prim):
        if child.IsInstance():
            child.SetInstanceable(False)
    for child in Usd.PrimRange(prim):
        if child != prim and child.HasAPI(UsdPhysics.RigidBodyAPI):
            child.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if child.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr('convexHull')
    schemas.define_rigid_body_properties(str(prim.GetPath()), cfg._body_properties)
    schemas.define_mass_properties(str(prim.GetPath()), cfg._mass_properties)
    return prim
