#!/usr/bin/env python3
"""Render original USDA scenes for the local atlas, with MDL search paths.

Read-only source assets; camera, lighting and path repairs are session-layer edits.
No timeline/physics steps, robot, or task modifications. GPU jobs run sequentially.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MATERIALS = ROOT.parent / 'InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes/Materials'
old = os.environ.get('MDL_SYSTEM_PATH', '')
LOCAL_MDL = ROOT/'outputs/scene_atlas_materials_v1/material_cache/nvidia'
os.environ['MDL_SYSTEM_PATH'] = str(MATERIALS) + os.pathsep + str(LOCAL_MDL) + (os.pathsep + old if old else '')
parser = argparse.ArgumentParser()
parser.add_argument('--ids', required=True, help='Comma-separated scene IDs')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--resolution', type=int, default=768)
parser.add_argument('--source-root',type=Path,default=ROOT.parent/'geomind_assets/stage')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

from geomind.core.envs.isaaclab_native_delegate import _IsaacLabAppRuntime
_IsaacLabAppRuntime.ensure_started(headless=True, enable_cameras=True)
app = _IsaacLabAppRuntime._simulation_app
import carb
import omni.usd
import omni.replicator.core as rep
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
import numpy as np
from PIL import Image

cache_mapping_path = ROOT/'outputs/scene_atlas_materials_v1/material_cache/mapping.json'
texture_defaults = json.loads(cache_mapping_path.read_text()) if cache_mapping_path.exists() else {}

module_mapping_path = cache_mapping_path.with_name('modules.json')
local_modules = json.loads(module_mapping_path.read_text()) if module_mapping_path.exists() else {}

settings = carb.settings.get_settings()
settings.set('/renderer/mdl/searchPaths/system', os.environ['MDL_SYSTEM_PATH'])
settings.set('/rtx/post/histogram/enabled', False)
settings.set('/rtx/post/tonemap/filmIso', 100.0)


def repair_assets(stage):
    repairs, missing = [], []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if not value or not value.path:
                continue
            raw = value.path
            if raw in local_modules:
                attr.Set(Sdf.AssetPath(local_modules[raw]))
                repairs.append({'attribute':str(attr.GetPath()),'from':raw,'to':local_modules[raw]})
                continue
            # Preserve valid resolved assets and remote built-in materials.
            if value.resolvedPath and Path(value.resolvedPath).is_file():
                continue
            name = Path(raw).name
            candidates = [MATERIALS / name, MATERIALS / 'Textures' / name]
            if '/Assets/ArchVis/' in raw:
                candidates.append(Path('/home/elm/isaacsim_assets/Assets/Isaac/4.5/NVIDIA/Assets/ArchVis') / raw.split('/Assets/ArchVis/',1)[1])
            if '/isaacsim_assets/' in raw:
                candidates.append(Path('/home/elm/isaacsim_assets') / raw.split('/isaacsim_assets/', 1)[1])
            replacement = next((p for p in candidates if p.is_file()), None)
            if replacement:
                attr.Set(Sdf.AssetPath(str(replacement)))
                repairs.append({'attribute': str(attr.GetPath()), 'from': raw, 'to': str(replacement)})
            elif not raw.startswith(('http:', 'https:', 'omniverse:')):
                missing.append({'attribute': str(attr.GetPath()), 'path': raw})
    for prim in stage.Traverse():
        source = prim.GetAttribute('info:mdl:sourceAsset')
        value = source.Get() if source else None
        if value and value.path in texture_defaults:
            for param, texture in texture_defaults[value.path].items():
                prim.CreateAttribute('inputs:'+param, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture['path']))
                repairs.append({'attribute':str(prim.GetPath())+'.inputs:'+param,'from':texture['url'],'to':texture['path']})
    return repairs, missing


def run(n):
    start = time.monotonic()
    print(f'ATLAS_START {n}', flush=True)
    source = args.source_root.resolve() / f'{n}/{n}_new.usda'
    if not source.exists():
        source = source.with_name(f'{n}.usda')
    ctx = omni.usd.get_context()
    print(f'ATLAS_PREPARE {source}', flush=True)
    overlay = args.output / f'scene-{n}-preview.usda'
    layer = Sdf.Layer.CreateNew(str(overlay.resolve()))
    layer.subLayerPaths = [str(source)]
    stage = Usd.Stage.Open(layer)
    stage.SetEditTarget(layer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.0)
    repairs, missing = repair_assets(stage)
    print(f'ATLAS_MATERIALS {n} repairs={len(repairs)} missing={len(missing)}', flush=True)
    # Use a cutaway camera view; never write visibility changes to source files.
    hidden = []
    for prim in stage.Traverse():
        if prim.GetName().lower().startswith(('wall', 'ceiling')) and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(str(prim.GetPath()))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    import re
    tables = [p for p in stage.Traverse() if re.fullmatch(r'table_\d+', p.GetName()) and p.IsA(UsdGeom.Xformable)]
    if not tables:
        raise RuntimeError('No table found for camera framing')
    bounds = cache.ComputeWorldBound(tables[0]).ComputeAlignedRange()
    lo, hi = bounds.GetMin(), bounds.GetMax()
    center = [(lo[i] + hi[i]) / 2 for i in range(3)]
    target = [center[0], center[1], float(hi[2]) - .12]
    size = max(float(hi[0]-lo[0]), float(hi[1]-lo[1]), 1.2)
    eye = [center[0]+size*1.10, center[1]-size*1.35, float(hi[2])+size*1.15]
    camera = UsdGeom.Camera.Define(stage, '/AtlasPreviewCamera')
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateHorizontalApertureAttr(30.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 1000))
    transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0,0,1)).GetInverse()
    UsdGeom.Xformable(camera).AddTransformOp().Set(transform)
    light = UsdLux.DomeLight.Define(stage, '/AtlasPreviewFill')
    light.CreateIntensityAttr(700)
    light.CreateColorAttr(Gf.Vec3f(1, .98, .95))
    camera_path = str(camera.GetPath())
    layer.Save()
    stage = None
    print(f'ATLAS_OPEN_PREPARED {n}', flush=True)
    ctx.open_stage(str(overlay.resolve()))
    for _ in range(8):
        app.update()
    product = rep.create.render_product(camera_path, (args.resolution, args.resolution))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([product])
    # Compile materials and let RTX finish loading texture resources before capture.
    for _ in range(35):
        app.update()
    for _ in range(45):
        app.update()
    data = np.asarray(rgb.get_data())[:, :, :3]
    if data.shape != (args.resolution, args.resolution, 3) or float(data.std()) < 2:
        raise RuntimeError(f'Invalid render {data.shape}')
    output = args.output / f'scene-{n}.png'
    Image.fromarray(data).save(output)
    rgb.detach([product.path])
    product.destroy()
    result = {'id':n, 'source':str(source), 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'preview_layer':str(overlay.resolve()), 'image':str(output), 'image_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'mdl_path':str(MATERIALS), 'asset_repairs':repairs, 'unresolved_assets':missing,
        'hidden_walls':hidden, 'camera_eye':eye, 'camera_target':target, 'resolution':args.resolution,
        'physics_executed':False, 'elapsed_s':round(time.monotonic()-start,2)}
    (args.output/f'scene-{n}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(f'ATLAS_RENDER_OK {n} repairs={len(repairs)} unresolved={len(missing)} seconds={result["elapsed_s"]}',flush=True)
    return result

results = []
try:
    for n in map(int, args.ids.split(',')):
        try:
            results.append(run(n))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results.append({'id':n,'error':str(exc)})
        (args.output/'summary.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
finally:
    # Native Kit shutdown can hang; the parent job enforces an independent bound.
    print('ATLAS_RENDER_DONE', flush=True)
    app.close()
