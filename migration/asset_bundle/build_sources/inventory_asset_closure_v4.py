"""Read-only asset closure audit. Does not mutate original USD/data or run GPU."""
from pathlib import Path
import json,re,time,collections
from pxr import UsdUtils
ROOT=Path('/home/elm/projects/geomind_lab')
OUT=ROOT/'outputs/research/portable_delivery_v1'
BASES={
 'repo_assets/assets':ROOT/'assets',
 'repo_assets/scene_atlas':ROOT/'scene_atlas',
 'curobo_assets/curobo/fr3_description':Path('/home/elm/projects/IsaacLab/curobo/src/curobo/content/assets/robot/fr3_description'),
 'curobo_assets/nvidia-curobo/fr3_description':Path('/home/elm/projects/IsaacLab/src/nvidia-curobo/src/curobo/content/assets/robot/fr3_description'),
 'geomind_assets':Path('/home/elm/projects/geomind_assets'),
 'grscenes_home':Path('/home/elm/projects/InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes'),
 'isaacsim_assets':Path('/home/elm/isaacsim_assets'),
 'material_cache':ROOT/'outputs/scene_atlas_materials_v1/material_cache',
}
ASSET_RE=re.compile(rb'@@@([^\r\n]*?)@@@|@([^@\r\n]+)@')
TEXT_RE=re.compile(r'["\']([^"\'\r\n]+\.(?:png|jpe?g|tga|exr|hdr|bmp|dds|mdl|usd[ac]?))["\']',re.I)
IMPORT_RE=re.compile(r'\bimport\s+(?:\*\s+from\s+)?([^;\s]+)\s*;')
BUILTIN={'anno','math','state','tex','df','base','limits','scene','debug','distilling_support','std'}
USING_RE=re.compile(r'\busing\s+([^;\s]+)\s+import')
seen={};queue=collections.deque();edges=[];missing=[];remote=[];mdl_modules=[];sdk_provided=[]

def logical(path):
 path=path.resolve()
 for name,base in sorted(BASES.items(),key=lambda p:-len(str(p[1]))):
  if path.is_relative_to(base): return name+'/'+path.relative_to(base).as_posix()
 return None

def add(path,reason):
 p=Path(path)
 if not p.is_file(): return False
 p=p.resolve()
 if p in seen:return True
 name=logical(p)
 if name is None:
  missing.append({'owner':reason,'reference':str(p),'reason':'outside_declared_asset_roots'});return False
 seen[p]={'source':str(p),'archive_path':name,'bytes':p.stat().st_size,'reason':reason}
 queue.append(p);return True

def resolve(ref,owner):
 ref=ref.replace('\\\\','/')
 if ref.startswith('file:'): ref=ref[5:]
 if ref in {'OmniPBR.mdl','OmniGlass.mdl'}:
  sdk_provided.append({'owner':str(owner),'reference':ref,'provider':'IsaacSim5.1 standard MDL library'});return []
 candidates=[]
 if ref.startswith(('http://','https://','omniverse://')):
  for token in ('/Assets/Isaac/4.5/','/Assets/Isaac/5.1/'):
   if token in ref:candidates.append(BASES['isaacsim_assets']/('Assets/Isaac/'+token.split('/')[3]+'/'+ref.split(token,1)[1]))
  if '/Assets/ArchVis/' in ref:candidates.append(BASES['isaacsim_assets']/'Assets/Isaac/4.5/NVIDIA/Assets/ArchVis'/ref.split('/Assets/ArchVis/',1)[1])
  if '/Assets/DigitalTwin/' in ref:candidates.append(BASES['isaacsim_assets']/'Assets/Isaac/5.1/NVIDIA/Assets/DigitalTwin'/ref.split('/Assets/DigitalTwin/',1)[1])
  if '/Materials/' in ref:
   rel=ref.split('/Materials/',1)[1]
   for ver in ('5.1','4.5'):candidates.append(BASES['isaacsim_assets']/f'Assets/Isaac/{ver}/NVIDIA/Materials'/rel)
  if 'ov-materials' in ref and '/Materials/' in ref:
   candidates.append(BASES['material_cache']/'nvidia'/ref.split('/Materials/',1)[1])
  if not candidates:remote.append({'owner':str(owner),'reference':ref});return []
 else:
  p=Path(ref)
  candidates.append(p if p.is_absolute() else owner.parent/p)
  if not p.is_absolute():
   candidates.extend([BASES['grscenes_home']/'Materials'/p,BASES['grscenes_home']/p])
  if '/Assets/ArchVis/' in ref:candidates.append(BASES['isaacsim_assets']/'Assets/Isaac/4.5/NVIDIA/Assets/ArchVis'/ref.split('/Assets/ArchVis/',1)[1])
  if '/Assets/DigitalTwin/' in ref:candidates.append(BASES['isaacsim_assets']/'Assets/Isaac/5.1/NVIDIA/Assets/DigitalTwin'/ref.split('/Assets/DigitalTwin/',1)[1])
  for old,name in [('/home/projects/geomind_assets','geomind_assets'),('/home/elm/projects/geomind/geomind_assets','geomind_assets')]:
   if ref.startswith(old+'/'):candidates.append(BASES[name]/ref[len(old)+1:])
 for p in candidates:
  if p.is_file():return [p.resolve()]
  if '<UDIM>' in str(p):
   tiles=list(p.parent.glob(p.name.replace('<UDIM>','[0-9][0-9][0-9][0-9]')))
   if tiles:return [x.resolve() for x in tiles if x.is_file()]
 missing.append({'owner':str(owner),'reference':ref,'candidates':[str(x) for x in candidates]})
 return []

# Full current GEOMIND data. Old backup scene copies are excluded unless an
# explicit dependency refers to them. No caches, raw SDK or unrelated models.
for p in BASES['geomind_assets'].rglob('*'):
 if p.is_file() and not {'stage_old','.thumbs','stage_camera_images'}&set(p.relative_to(BASES['geomind_assets']).parts):add(p,'geomind_runtime_tree')
for kind in ['repo_assets/assets','repo_assets/scene_atlas','curobo_assets/curobo/fr3_description','curobo_assets/nvidia-curobo/fr3_description']:
 for p in BASES[kind].rglob('*'):
  if p.is_file() and '.thumbs' not in p.parts:add(p,'repo_and_optional_curobo_asset_tree')
for p in BASES['material_cache'].rglob('*'):
 if p.is_file():add(p,'material_cache_full_modules')
robot_roots=[
 'Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd',
 'Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd',
 'Assets/Isaac/4.5/Isaac/Robots/Franka/FR3/fr3.usd',
]
legacy_text=(ROOT/'scene_construction /configs/asset_config.py').read_text()
for path in re.findall(r'asset_path\s*\+\s*["\']([^"\']+\.usd[ac]?)["\']',legacy_text):
 add(BASES['isaacsim_assets']/('Assets/Isaac/4.5'+path),'legacy_scene_construction_asset')
for p in robot_roots:add(BASES['isaacsim_assets']/p,'five_embodiment_robot_root')
# Public geometry examples use these three YCB assets.
for name in ('002_master_chef_can','010_potted_meat_can','006_mustard_bottle'):
 add(BASES['isaacsim_assets']/f'Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/{name}.usd','native_episode_public_geometry')
# Catalog paths can point outside the GEOMIND data tree.
cat=json.loads((ROOT/'assets/scene_profiles/assets_catalog_with_metadata.json').read_text())
for group,values in cat['assets'].items():
 for x in values:
  for p in resolve(x['usd_path'],ROOT/'assets/scene_profiles/assets_catalog_with_metadata.json'):add(p,'catalog:'+x['name'])
started=time.time();processed=0
while queue:
 owner=queue.popleft();processed+=1;refs=set()
 suffix=owner.suffix.lower()
 try:
  if suffix=='.usda':
   with owner.open('rb') as f:
    for line in f:
     if b'@' in line:refs.update(next(v for v in pair if v).decode('utf-8') for pair in ASSET_RE.findall(line))
  elif suffix in {'.usd','.usdc'}:
   groups=UsdUtils.ExtractExternalReferences(str(owner))
   refs.update(x for group in groups for x in group)
  elif suffix=='.mdl':
   text=owner.read_text(errors='replace');refs.update(TEXT_RE.findall(text))
   imports_text=re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)
   for token in [*IMPORT_RE.findall(imports_text),*USING_RE.findall(imports_text)]:
    if token=='*':continue
    if token in {'OmniPBR','OmniGlass'} or token.startswith(('::OmniPBR','::OmniGlass')):
     sdk_provided.append({'owner':str(owner),'reference':token,'provider':'IsaacSim5.1 standard MDL library'});continue
    base=token.lstrip(':').split('::')[0].lstrip('.')
    if base not in BUILTIN:
     if token.startswith(('::OmniSurface::','::nvidia::')):sdk_provided.append({'owner':str(owner),'reference':token,'provider':'IsaacSim5.1 standard MDL library'})
     elif token.startswith('.'):
      parts=token.split('::');dots=parts.pop(0);folder=owner.parent
      for _ in range(max(0,len(dots)-1)):folder=folder.parent
      if parts[-1]=='*':parts.pop()
      candidate=folder.joinpath(*parts).with_suffix('.mdl')
      if candidate.is_file():
       edges.append({'owner':logical(owner),'reference':token,'target':logical(candidate)});add(candidate,'relative_mdl_import:'+logical(owner))
      else:mdl_modules.append({'owner':str(owner),'module':token,'candidate':str(candidate)})
     else:mdl_modules.append({'owner':str(owner),'module':token})
  else:continue
  for ref in sorted(refs):
   for p in resolve(ref,owner):
    edges.append({'owner':logical(owner),'reference':ref,'target':logical(p)})
    add(p,'dependency:'+logical(owner))
 except Exception as e:missing.append({'owner':str(owner),'reason':f'{type(e).__name__}: {e}'})
 if processed%100==0:print(json.dumps({'processed':processed,'files':len(seen),'pending':len(queue),'seconds':round(time.time()-started,1)}),flush=True)
summary=collections.defaultdict(lambda:{'files':0,'bytes':0})
for x in seen.values():
 k=x['archive_path'].split('/')[0];summary[k]['files']+=1;summary[k]['bytes']+=x['bytes']
manifest={'kind':'portable_asset_closure_audit','read_only':True,'root_directories':{k:str(v) for k,v in BASES.items()},'root_policy':'Full GEOMIND runtime data excluding stage_old/.thumbs/stage_camera_images; all repo assets; cached material modules; robot and public native episode roots; recursive USD and texture references','files':sorted(seen.values(),key=lambda x:x['archive_path']),'summary':dict(summary),'total_files':len(seen),'total_bytes':sum(x['bytes'] for x in seen.values()),'edges':edges,'missing':missing,'remote':remote,'sdk_provided':sdk_provided,'mdl_modules_requiring_installed_IsaacSim_or_local_module_resolution':mdl_modules,'elapsed_s':time.time()-started,'limitations':['No physical execution or reference rewriting; binary USD referenced asset paths use UsdUtils.ExtractExternalReferences','MDL module imports audited separately; standard modules are supplied by installed IsaacSim5.1, not the asset ZIP','Missing inactive/deprecated references remain explicit and must be resolved or documented before a complete bundle claim']}
(OUT/'asset_closure_inventory_v4.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ['total_files','total_bytes','summary','elapsed_s']}),flush=True)
print('missing',len(missing),'remote',len(remote),'mdl imports',len(mdl_modules),flush=True)
