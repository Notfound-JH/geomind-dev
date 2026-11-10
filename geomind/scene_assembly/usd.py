"""USD composition and geometric validation for initial scene assemblies.

Imports pxr only in the export worker. The exported layer references the already
cleared room, with rigid task objects in a new namespace. No original is edited.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import random

import numpy as np

from .assets import ROOT, resolve_asset, resolve_scene
from .layouts import generate_positions
from .request import validate_request


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _world_yaw_orientation(quaternion, degrees):
    """Compose a bounded world-Z yaw with the catalog's wxyz orientation."""
    w,x,y,z=quaternion
    half=math.radians(degrees)*.5;c=math.cos(half);s=math.sin(half)
    return [c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]


def _positions_have_clearance(positions, footprint, clearance=.025, *, circular_envelope=False):
    if circular_envelope:
        diameter=math.hypot(*footprint)+clearance
        return all(math.dist(a,b)+1e-9>=diameter for i,a in enumerate(positions) for b in positions[i+1:])
    return all(abs(a[0]-b[0])+1e-9>=footprint[0]+clearance or
               abs(a[1]-b[1])+1e-9>=footprint[1]+clearance
               for i,a in enumerate(positions) for b in positions[i+1:])


def _triangles(stage, root):
    from pxr import Gf, Usd, UsdGeom
    result = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or counts is None or indices is None:
            continue
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        pts = np.asarray(points,dtype=float)
        pts = np.c_[pts, np.ones(len(pts))] @ matrix
        offset = 0
        triangles = []
        for count in counts:
            ids = indices[offset:offset+count]
            offset += count
            for i in range(1,count-1):
                triangles.append([ids[0],ids[i],ids[i+1]])
        if triangles:
            tri = pts[np.asarray(triangles),:3]
            cross = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            norm = np.linalg.norm(cross,axis=1)
            mask = (norm>1e-10) & (np.abs(cross[:,2])/np.maximum(norm,1e-12)>.995)
            for t,a in zip(tri[mask], np.abs(cross[mask,2])*.5):
                if np.ptp(t[:,2]) < .004:
                    result.append((float(np.mean(t[:,2])),float(a),t,str(prim.GetPath())))
    return result


def measure_table(stage, prim_path=None):
    """Choose the largest near-horizontal mesh surface, preferring its top.

    The largest room/table AABB may include feet or monitors. Measuring triangle
    planes avoids interpreting a monitor top as a tabletop. Conservative support
    samples later reject a grid extending beyond a curved tabletop edge.
    """
    import re
    from pxr import UsdGeom, Usd
    tables = ([stage.GetPrimAtPath(prim_path)] if prim_path is not None else
              [p for p in stage.Traverse() if re.fullmatch(r'table_\d+',p.GetName()) and p.IsA(UsdGeom.Xformable)])
    if prim_path is not None and (not tables[0].IsValid() or not tables[0].IsA(UsdGeom.Xformable)):
        raise ValueError(f'Invalid explicit table prim: {prim_path}')
    if len(tables)!=1:
        raise ValueError(f'Expected one table in the cleared scene; found {len(tables)}')
    table=tables[0]
    triangles=_triangles(stage,table)
    named_top=[r for r in triangles if any(token in r[3].lower() for token in ('table_top','tabletop'))]
    if named_top:
        triangles=named_top
    groups=defaultdict(list)
    for row in triangles:
        groups[round(row[0]/.003)].append(row)
    if not groups:
        raise ValueError('No near-horizontal table mesh surface found')
    areas={k:sum(t[1] for t in v) for k,v in groups.items()}
    maximum=max(areas.values())
    candidates=[(k,v) for k,v in groups.items() if areas[k]>=maximum*.45]
    _,chosen=max(candidates,key=lambda pair:sum(t[0]*t[1] for t in pair[1])/sum(t[1] for t in pair[1]))
    pts=np.concatenate([x[2] for x in chosen])
    z=sum(t[0]*t[1] for t in chosen)/sum(t[1] for t in chosen)
    bounds=[float(pts[:,0].min()),float(pts[:,0].max()),float(pts[:,1].min()),float(pts[:,1].max())]
    return {'prim_path':str(table.GetPath()),'height_m':float(z),'bounds_xy_m':bounds,
            'surface_area_m2':sum(t[1] for t in chosen),'mesh_paths':sorted({t[3] for t in chosen}),
            'surface_triangles':np.asarray([t[2] for t in chosen])}


def supported_xy(table,xy,half):
    """Check center, edge midpoints and corners against the tabletop triangles."""
    triangles=table['surface_triangles'][:,:,:2]
    a=triangles[:,0];b=triangles[:,1];c=triangles[:,2]
    v0=b-a;v1=c-a
    den=v0[:,0]*v1[:,1]-v0[:,1]*v1[:,0]
    for dx in [-half[0],0,half[0]]:
        for dy in [-half[1],0,half[1]]:
            v2=np.asarray([xy[0]+dx,xy[1]+dy])-a
            u=(v2[:,0]*v1[:,1]-v2[:,1]*v1[:,0])/np.where(abs(den)>1e-12,den,np.nan)
            v=(v0[:,0]*v2[:,1]-v0[:,1]*v2[:,0])/np.where(abs(den)>1e-12,den,np.nan)
            if not np.any((u>=-1e-5)&(v>=-1e-5)&(u+v<=1+1e-5)):
                return False
    return True


def _background_obstacles(stage,table_path):
    """Conservative AABBs of retained furniture/fixtures, excluding the table."""
    from pxr import UsdGeom
    obstacles=[]
    world=stage.GetPrimAtPath('/World')
    for child in world.GetChildren():
        candidates=list(child.GetChildren()) if child.GetName() in {'function','decoration','container','object'} else [child]
        for prim in candidates:
            if str(prim.GetPath())==table_path or not prim.IsA(UsdGeom.Xformable):continue
            if prim.GetName().lower().startswith(('wood_ground','physicsground','wall')):continue
            try:lo,hi=_bounds(stage,prim)
            except ValueError:continue
            if all(b>a for a,b in zip(lo,hi)):
                obstacles.append({'prim_path':str(prim.GetPath()),'min':lo,'max':hi})
    return obstacles


def _hits_obstacle(xy,dimensions,z_min,z_max,obstacles):
    lo=[xy[0]-dimensions[0]*.5,xy[1]-dimensions[1]*.5,z_min]
    hi=[xy[0]+dimensions[0]*.5,xy[1]+dimensions[1]*.5,z_max]
    return any(all(hi[i]>o['min'][i]+.001 and lo[i]<o['max'][i]-.001 for i in range(3)) for o in obstacles)


def _set_transform(prim,scale,orientation,translation):
    from pxr import Gf, UsdGeom
    xform=UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    def operation(name,create):
        attr=prim.GetAttribute(name)
        return UsdGeom.XformOp(attr) if attr else create()
    translate=operation('xformOp:translate',xform.AddTranslateOp)
    orient=operation('xformOp:orient',xform.AddOrientOp)
    scaling=operation('xformOp:scale',xform.AddScaleOp)
    translate.Set(Gf.Vec3d(*translation))
    quaternion=Gf.Quatd(orientation[0],Gf.Vec3d(*orientation[1:]))
    if orient.GetPrecision()==UsdGeom.XformOp.PrecisionFloat:
        quaternion=Gf.Quatf(quaternion)
    orient.Set(quaternion)
    scaling.Set(Gf.Vec3d(*scale))
    xform.SetXformOpOrder([translate,orient,scaling])


def _bounds(stage,prim):
    from pxr import Usd, UsdGeom
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
    b=cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo=list(b.GetMin());hi=list(b.GetMax())
    if not all(math.isfinite(x) for x in lo+hi):
        raise ValueError('Asset has invalid bounds')
    return lo,hi


def _body_points(stage,root,orientation):
    """Scaled mesh vertices in a rotation-only rigid frame, for measured poses."""
    from pxr import Usd,UsdGeom,Gf
    points=[]
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility()==UsdGeom.Tokens.invisible:
            continue
        raw=UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if raw is None:continue
        matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        values=np.asarray(raw,dtype=float)
        points.append((np.c_[values,np.ones(len(values))]@matrix)[:,:3])
    if not points:raise ValueError('No mesh vertices found for body-frame measurement')
    rot=np.asarray(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(orientation[0],Gf.Vec3d(*orientation[1:])))))
    return np.concatenate(points)@rot.T


def _rigid_object(prim):
    from pxr import Usd, UsdGeom, UsdPhysics
    for child in Usd.PrimRange(prim):
        if child!=prim and child.HasAPI(UsdPhysics.RigidBodyAPI):
            child.RemoveAPI(UsdPhysics.RigidBodyAPI)
    body=UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(False)
    body.CreateVelocityAttr((0,0,0));body.CreateAngularVelocityAttr((0,0,0))
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(.10)
    meshes=[]
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr('convexHull')
            meshes.append(str(child.GetPath()))
    if not meshes:
        raise ValueError('The requested asset has no mesh collision geometry')
    return meshes


def assemble(request, output):
    from pxr import Usd, Sdf, Gf, UsdGeom, UsdPhysics
    request=validate_request(request)
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    destination=output/'scene.usda'
    if destination.exists():
        raise ValueError(f'Refusing to overwrite existing assembly: {destination}')
    scene=resolve_scene(request['scene_id']);asset=resolve_asset(request['object']['asset'])
    source=Usd.Stage.Open(scene['usd_path'])
    table=measure_table(source)
    obstacles=_background_obstacles(source,table['prim_path'])
    source_active={str(p.GetPath()) for p in source.Traverse()}
    layer=Sdf.Layer.CreateNew(str(destination))
    layer.subLayerPaths=[scene['usd_path']]
    stage=Usd.Stage.Open(layer)
    for key in ['startTimeCode','endTimeCode','timeCodesPerSecond','framesPerSecond','defaultPrim','upAxis','metersPerUnit']:
        if source.GetRootLayer().pseudoRoot.HasInfo(key):
            stage.SetMetadata(key,source.GetRootLayer().pseudoRoot.GetInfo(key))
    # Keep room/table furniture fixed while the newly added props can settle.
    frozen=[]
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            api=UsdPhysics.RigidBodyAPI(p)
            api.CreateKinematicEnabledAttr(True)
            api.CreateVelocityAttr((0,0,0));api.CreateAngularVelocityAttr((0,0,0))
            frozen.append(str(p.GetPath()))
    UsdGeom.Xform.Define(stage,'/World/AssembledObjects')
    paths=[];collision_paths=[]
    for i in range(request['object']['count']):
        path=f'/World/AssembledObjects/{asset["name"]}_{i+1:03d}'
        prim=stage.DefinePrim(path,'Xform')
        prim.GetReferences().AddReference(asset['usd_path'])
        _set_transform(prim,asset['scale'],asset['orientation_wxyz'],[0,0,0])
        paths.append(path)
    body_points=_body_points(stage,stage.GetPrimAtPath(paths[0]),asset['orientation_wxyz'])
    np.savez_compressed(output/'asset_body_vertices.npz',points_m=body_points)
    randomization=request.get('randomization',{})
    randomized=randomization.get('enabled',False)
    yaw_rng=random.Random(request['seed']+2000003)
    yaw_degrees=[yaw_rng.uniform(-randomization['yaw_deg'],randomization['yaw_deg']) if randomized else 0.0 for _ in paths]
    orientations=[_world_yaw_orientation(asset['orientation_wxyz'],yaw) for yaw in yaw_degrees]
    object_geometry=[]
    for path,orientation in zip(paths,orientations):
        prim=stage.GetPrimAtPath(path)
        _set_transform(prim,asset['scale'],orientation,[0,0,0])
        lo,hi=_bounds(stage,prim)
        object_geometry.append({'min':lo,'max':hi,'dimensions':[b-a for a,b in zip(lo,hi)],
                                'center':[(a+b)*.5 for a,b in zip(lo,hi)]})
    dimensions=[max(g['dimensions'][axis] for g in object_geometry) for axis in range(3)]
    if not all(.001<=d<=.75 for d in dimensions):
        raise ValueError(f'Asset transformed dimensions are unsupported for tabletop assembly: {dimensions}')
    bounds=table['bounds_xy_m']
    # Place bounding-box centres, then compensate asymmetric rigid-body origins.
    positions=None;sampling_attempt=0;randomization_record=None
    layout=dict(request['layout'])
    # The legacy placer uses a 13cm comfort pitch. Use it as the randomized
    # baseline where it fits, while explicit spacing remains the user's base.
    if randomized and layout['type'] in {'grid','line','t_shape'} and 'spacing_m' not in layout:
        comfort=max(.13,max(dimensions[:2])+.025)
        try:
            generate_positions(dict(layout,spacing_m=comfort),request['object']['count'],
                bounds_xy=tuple(bounds),footprint_xy=tuple(dimensions[:2]),seed=request['seed'])
        except ValueError:
            pass
        else:
            layout['spacing_m']=comfort
    for attempt in range(256 if request['layout']['type']=='random' else 1):
        candidate=generate_positions(layout,request['object']['count'],bounds_xy=tuple(bounds),
            footprint_xy=tuple(dimensions[:2]),seed=request['seed']+attempt)
        if randomized:
            from .randomization import randomize_positions
            candidate,randomization_record=randomize_positions(candidate,layout=layout,
                footprint_xy=tuple(dimensions[:2]),bounds_xy=(bounds[0]+.03,bounds[1]-.03,bounds[2]+.03,bounds[3]-.03),
                seed=request['seed'],config=randomization)
        clearance_ok=_positions_have_clearance(candidate,dimensions[:2],circular_envelope=layout['type']=='ring')
        support_ok=all(supported_xy(table,xy,[d*.5 for d in dimensions[:2]]) for xy in candidate)
        obstacles_ok=all(not _hits_obstacle(xy,dimensions,table['height_m']+.001,table['height_m']+.001+dimensions[2],obstacles) for xy in candidate)
        if clearance_ok and support_ok and obstacles_ok:
            positions=candidate;sampling_attempt=attempt;break
    if positions is None:
        raise ValueError('Layout lacks tabletop support or intersects a retained fixture; reduce count/spacing or choose another scene')
    poses=[]
    for index,(path,xy,geometry,orientation) in enumerate(zip(paths,positions,object_geometry,orientations)):
        translation=[xy[0]-geometry['center'][0],xy[1]-geometry['center'][1],table['height_m']-geometry['min'][2]+.001]
        prim=stage.GetPrimAtPath(path)
        _set_transform(prim,asset['scale'],orientation,translation)
        collision_paths.extend(_rigid_object(prim))
        b0,b1=_bounds(stage,prim)
        pose={'prim_path':path,'position_w_m':translation,'orientation_wxyz':orientation,
            'bbox_center_xy_m':list(xy),'bbox_min_w_m':b0,'bbox_max_w_m':b1,'yaw_degrees':yaw_degrees[index]}
        if layout['type']=='grid':pose['grid_cell']={'row':index//layout['cols']+1,'col':index%layout['cols']+1}
        poses.append(pose)
    layer.customLayerData={'generator':'geomind.tools.assemble_scene','scene_id':request['scene_id'],
                           'asset':asset['name'],'layout':request['layout']['type'],'seed':request['seed']}
    layer.Save()
    reopened=Usd.Stage.Open(str(destination))
    actual=[p for p in reopened.GetPrimAtPath('/World/AssembledObjects').GetChildren() if p.IsActive()]
    if len(actual)!=request['object']['count']:
        raise ValueError('Saved object count did not survive reopening')
    if not source_active.issubset({str(p.GetPath()) for p in reopened.Traverse()}):
        raise ValueError('Assembly unexpectedly removed part of the cleared background')
    for pose in poses:
        b0,b1=_bounds(reopened,reopened.GetPrimAtPath(pose['prim_path']))
        if max(abs(a-b) for a,b in zip(b0+ b1,pose['bbox_min_w_m']+pose['bbox_max_w_m']))>1e-8:
            raise ValueError('Saved geometry changed after reopening')
    public_table={k:v for k,v in table.items() if k!='surface_triangles'}
    manifest={'schema_version':1,'kind':'initial_scene_assembly','status':'assembled_physics_unverified',
        'request':request,'scene':scene,'asset':asset,'table':public_table,'objects':poses,
        'layout_frame':'world_xy; grid rows along +X, columns along +Y',
        'geometry_validation':{'saved_reopened':True,'object_count':len(actual),'footprint_support_samples_passed':True,
            'tabletop_method':'largest near-horizontal triangle plane; 9 support samples per object',
            'minimum_clearance_m':.025,'collision_approximation':'convexHull per mesh',
            'retained_fixture_aabb_checks_passed':True,'sampling_attempt':sampling_attempt,
            'effective_sampling_seed':request['seed']+sampling_attempt},
        'frozen_background_rigid_bodies':frozen,'collision_mesh_count':len(collision_paths),
        'asset_body_vertices':str(output/'asset_body_vertices.npz'),
        'physics':{'executed':False,'passed':False,'notes':'Initial scene generated; no robot action or physics settling executed'},
        'scene_usda':str(destination),'provenance':{'cleared_scene_sha256':digest(scene['usd_path']),
            'asset_sha256':digest(asset['usd_path']),'scene_usda_sha256':digest(destination),
            'asset_body_vertices_sha256':digest(output/'asset_body_vertices.npz'),
            'used_usd_layers_sha256':{str(x.realPath):digest(x.realPath) for x in stage.GetUsedLayers() if x.realPath},
            'used_usd_layers':[str(x.realPath) for x in stage.GetUsedLayers() if x.realPath]}}
    if randomized:
        manifest['layout_randomization']={**randomization_record,'enabled':True,'yaw_degrees':yaw_degrees,
            'yaw_seed':request['seed']+2000003,'yaw_max_degrees':randomization['yaw_deg'],
            'orientation_method':'bounded world-Z yaw composed with first catalog orientation',
            'base_layout':layout,'conservative_footprint_xy_m':dimensions[:2]}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest
