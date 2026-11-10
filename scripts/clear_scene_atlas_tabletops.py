#!/usr/bin/env python3
"""Save 102 non-destructive USD scene variants with manipulable props removed.

Uses USD active=false overrides, which remove rendering AND physics participation.
Original environments remain sublayers; source assets are never edited.
"""
import argparse,hashlib,json,re,time
from pathlib import Path
from pxr import Sdf,Usd,UsdGeom

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT.parent/'geomind_assets/stage'
OUT=ROOT/'assets/scenes/tabletop_cleared_v1'
EVIDENCE=ROOT/'outputs/scene_atlas_cleared_v1'
MATERIALS=ROOT.parent/'InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes/Materials'
MODULES=json.loads((ROOT/'outputs/scene_atlas_materials_v1/material_cache/modules.json').read_text())
TASK_PREFIXES={'blue_can','red_can','low_can','cube_can','yellow_bottle','white_bottle','brown_box','red_box','rubiks_cube'}
CONTAINER_PREFIXES=('basket','bowl','plate','container')
PORTABLE_PREFIXES=('cup','pot','plant','book','vase')


def stem(name):return re.sub(r'_\d+$','',name)
def is_under(path,parents):return any(path==p or path.startswith(p+'/') for p in parents)
def bbox(cache,p):
 b=cache.ComputeWorldBound(p).ComputeAlignedRange()
 return {'min':list(b.GetMin()),'max':list(b.GetMax())}

def targets(stage):
 world=stage.GetPrimAtPath('/World')
 if not world:raise RuntimeError('Missing /World')
 cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
 tables=[p for p in stage.Traverse() if re.fullmatch(r'table_\d+',p.GetName()) and p.IsA(UsdGeom.Xformable)]
 if len(tables)!=1:raise RuntimeError(f'Expected one table, got {len(tables)}')
 table=tables[0];tb=bbox(cache,table);candidates=[];removed=[]
 for p in world.GetChildren():
  if p.GetName() in ['object','container']:
   children=[str(c.GetPath()) for c in p.GetChildren()]
   if children:removed.append({'path':str(p.GetPath()),'reason':'task_objects' if p.GetName()=='object' else 'operation_containers','items':children})
  elif p.GetName() in ['decoration','function']:
   candidates.extend(p.GetChildren())
  else:candidates.append(p)
 for p in candidates:
  name=p.GetName();path=str(p.GetPath());base=stem(name)
  if base in TASK_PREFIXES or base.startswith(CONTAINER_PREFIXES):
   removed.append({'path':path,'reason':'ungrouped_task_object_or_container','items':[path]});continue
  if not base.startswith(PORTABLE_PREFIXES):continue
  b=bbox(cache,p);lo,hi=b['min'],b['max'];center=[(a+c)/2 for a,c in zip(lo,hi)]
  on_table=(all(tb['min'][i]-.02<=center[i]<=tb['max'][i]+.02 for i in [0,1])
      and tb['max'][2]-.10<=lo[2]<=tb['max'][2]+.15 and hi[2]>=tb['max'][2]-.03
      and max(hi[i]-lo[i] for i in range(3))<=.8)
  if on_table:removed.append({'path':path,'reason':'portable_decoration_on_tabletop','items':[path],'bbox':b})
 return table,removed,tb


def repair_materials(stage):
 repairs=[]
 for p in stage.Traverse():
  for a in p.GetAttributes():
   if a.GetTypeName()!=Sdf.ValueTypeNames.Asset:continue
   v=a.Get()
   if not v or not v.path:continue
   raw=v.path
   replacement=Path(MODULES[raw]) if raw in MODULES else None
   if replacement is None:
    if v.resolvedPath and Path(v.resolvedPath).is_file():continue
    candidates=[MATERIALS/Path(raw).name,MATERIALS/'Textures'/Path(raw).name]
    if '/Assets/ArchVis/' in raw:candidates.append(Path('/home/elm/isaacsim_assets/Assets/Isaac/4.5/NVIDIA/Assets/ArchVis')/raw.split('/Assets/ArchVis/',1)[1])
    if '/isaacsim_assets/' in raw:candidates.append(Path('/home/elm/isaacsim_assets')/raw.split('/isaacsim_assets/',1)[1])
    replacement=next((x for x in candidates if x.is_file()),None)
   if replacement:
    a.Set(Sdf.AssetPath(str(replacement)))
    repairs.append({'attribute':str(a.GetPath()),'from':raw,'to':str(replacement)})
 return repairs


def save_scene(n):
 source=ASSETS/str(n)/f'{n}_new.usda'
 if not source.exists():source=source.with_name(f'{n}.usda')
 original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
 original=Usd.Stage.Open(str(source))
 table,removed,tb=targets(original);paths=[r['path'] for r in removed]
 source_paths={str(p.GetPath()) for p in original.Traverse()}
 expected={p for p in source_paths if not is_under(p,paths)}
 folder=OUT/str(n);folder.mkdir(parents=True,exist_ok=False)
 destination=folder/f'{n}.usda'
 layer=Sdf.Layer.CreateNew(str(destination))
 layer.subLayerPaths=[str(source)]
 layer.customLayerData={'variant':'TabletopCleared','source_scene_id':n,'source_sha256':original_hash,'removal_mode':'active_false_render_and_physics'}
 edited=Usd.Stage.Open(layer);UsdGeom.SetStageUpAxis(edited,UsdGeom.GetStageUpAxis(original));UsdGeom.SetStageMetersPerUnit(edited,UsdGeom.GetStageMetersPerUnit(original))
 for key in ['startTimeCode','endTimeCode','timeCodesPerSecond','framesPerSecond','defaultPrim','colorManagementSystem','colorConfiguration','renderSettingsPrimPath']:
  if original.GetRootLayer().pseudoRoot.HasInfo(key):edited.SetMetadata(key,original.GetRootLayer().pseudoRoot.GetInfo(key))
 for path in paths:edited.OverridePrim(path).SetActive(False)
 repairs=repair_materials(edited)
 layer.Save()
 # Reopen saved layer independently, rather than checking only in-memory changes.
 reopened=Usd.Stage.Open(str(destination));actual={str(p.GetPath()) for p in reopened.Traverse()}
 if actual!=expected:raise RuntimeError(f'Unexpected retained/removed prims: {actual.symmetric_difference(expected)}')
 for path in paths:
  p=reopened.GetPrimAtPath(path)
  if p and p.IsActive():raise RuntimeError('Removed object is still active')
 actual_table=reopened.GetPrimAtPath(table.GetPath())
 if not actual_table or not actual_table.IsActive():raise RuntimeError('Table was removed')
 if UsdGeom.Xformable(table).ComputeLocalToWorldTransform(Usd.TimeCode.Default())!=UsdGeom.Xformable(actual_table).ComputeLocalToWorldTransform(Usd.TimeCode.Default()):raise RuntimeError('Table transform changed')
 # No source edits are made; verify content still matches the pre-edit fingerprint.
 if hashlib.sha256(source.read_bytes()).hexdigest()!=original_hash:raise RuntimeError('Original source changed')
 original_json=source.parent/f'{n}.json'
 meta=json.loads(original_json.read_text())
 (folder/'source_scene.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
 meta.update(scene_id=n,variant='TabletopCleared',layout='EmptyTabletop',num_of_objects=0,object_asset=None,object_positions={},container={'basket':0,'basket_position':[],'plate':0,'plate_position':[],'bowl':0,'bowl_position':[]},source_usda=str(source),usda_path=str(destination))
 (folder/f'{n}.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
 record={'id':n,'source':str(source),'source_sha256':original_hash,'output':str(destination),'output_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'removed':removed,'removed_item_count':sum(len(r['items']) for r in removed),'table_path':str(table.GetPath()),'table_bbox':tb,'retained_prim_count':len(actual),'asset_repairs':repairs,'saved_reopened_verified':True,'original_unchanged':True,'physics_executed':False}
 (folder/'removal_report.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
 return record


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--ids',default=','.join(map(str,range(1,103))));a=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);EVIDENCE.mkdir(parents=True,exist_ok=True)
 records=[]
 for n in map(int,a.ids.split(',')):
  print(f'CLEAR_START {n}',flush=True)
  try:
   row=save_scene(n);records.append(row)
   print(f'CLEAR_SAVED {n} removed_items={row["removed_item_count"]} kept_prims={row["retained_prim_count"]}',flush=True)
  except Exception as exc:
   import traceback;traceback.print_exc();records.append({'id':n,'error':str(exc)})
  (EVIDENCE/'save_report.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
 if any('error' in r for r in records):raise SystemExit(1)
 print('CLEAR_DONE',len(records),flush=True)
if __name__=='__main__':main()
