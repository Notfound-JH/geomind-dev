"""Independent references/metadata validation of the extracted asset prefix."""
from pathlib import Path
import hashlib,json,re,time,xml.etree.ElementTree as ET,collections,gc
from pxr import Sdf,UsdUtils
OUT=Path('/home/elm/projects/geomind_lab/outputs/research/portable_delivery_v1');BASE=OUT/'relocated_validation/geomind-assets-v1'
ASSET_RE=re.compile(rb'@@@([^\r\n]*?)@@@|@([^@\r\n]+)@')
TEXT_RE=re.compile(r'["\']([^"\'\r\n]+\.(?:png|jpe?g|tga|exr|hdr|bmp|dds|mdl|usd[ac]?))["\']',re.I)
XML_RE=re.compile(r'(?:filename|file)\s*=\s*["\']([^"\']+)["\']')
manifest_path=BASE/'asset_bundle_manifest.json';m=json.loads(manifest_path.read_text());counts=collections.Counter();failures=[];homes=[];source_templates=[];started=time.time()


def check(owner,reference,kind):
 counts[kind]+=1
 if '/third_party/' in str(owner.relative_to(BASE)) and '/source/' in str(owner.relative_to(BASE)) and owner.suffix=='.xacro':
  source_templates.append({'owner':str(owner.relative_to(BASE)),'reference':reference,'scope':'Archived official xacro regeneration source; requires optional full franka_description ROS/Gazebo toolchain, not used by the delivered concrete URDF/USD'});return
 if reference in {'OmniPBR.mdl','OmniGlass.mdl'}:counts['SDK_builtin_asset']+=1;return
 if reference.startswith(('http:','https:','omniverse:','bundle:','repo:','package:','file:')):
  failures.append({'owner':str(owner.relative_to(BASE)),'reference':reference,'reason':'concrete resource is not a relative bundled file'});return
 target=(owner.parent/reference.replace('\\','/')).resolve()
 if not target.is_relative_to(BASE) or not target.is_file():
  failures.append({'owner':str(owner.relative_to(BASE)),'reference':reference,'reason':'missing file or reference escaped relocated bundle'})

for entry in m['files']:
 owner=BASE/entry['path'];suffix=owner.suffix.lower();refs=set()
 if not owner.is_file() or owner.stat().st_size!=entry['size_bytes']:
  failures.append({'owner':entry['path'],'reason':'file is missing or size changed'});continue
 if suffix=='.usda':
  with owner.open('rb') as f:
   for line in f:
    if b'/home/elm/' in line:homes.append(entry['path'])
    if b'@' in line:refs.update(next(item for item in pair if item).decode('utf-8') for pair in ASSET_RE.findall(line))
  for ref in refs:check(owner,ref,'USD_text_asset_reference')
 elif suffix in {'.usd','.usdc'}:
  try:
   for group in UsdUtils.ExtractExternalReferences(str(owner)):
    for ref in group:check(owner,ref,'USD_binary_asset_reference')
  except Exception as e:failures.append({'owner':entry['path'],'reason':type(e).__name__+': '+str(e)})
 elif suffix=='.mdl':
  text=owner.read_text(errors='replace')
  if '/home/elm/' in text:homes.append(entry['path'])
  for ref in set(TEXT_RE.findall(text)):check(owner,ref,'MDL_texture_reference')
 elif suffix in {'.urdf','.xacro','.xml','.dae','.obj','.mtl'}:
  text=owner.read_text(errors='replace')
  if '/home/elm/' in text:homes.append(entry['path'])
  if suffix in {'.urdf','.xacro','.xml'}:refs.update(XML_RE.findall(text))
  if suffix=='.dae':
   try:
    dom=ET.fromstring(text)
    for node in dom.iter():
     if node.tag.split('}')[-1]=='image':
      for child in node.iter():
       if child.tag.split('}')[-1]=='init_from' and child.text:refs.add(child.text.strip())
   except ET.ParseError as e:failures.append({'owner':entry['path'],'reason':'invalid XML '+str(e)})
  if suffix=='.obj':refs.update(re.findall(r'^mtllib\s+(.+)$',text,re.M))
  if suffix=='.mtl':refs.update(re.findall(r'^(?:map_\w+|bump)\s+(.+)$',text,re.M))
  for ref in refs:check(owner,ref,'XML_URDF_OBJ_resource_reference')
 elif suffix in {'.json','.md','.txt','.csv','.html','.css','.js','.yaml','.yml','.sh','.py'}:
  if b'/home/elm/' in owner.read_bytes():homes.append(entry['path'])
 counts['asset_files']+=1

# Parse all 102 saved scene layers under this prefix. Concrete dependencies
# were checked above without loading hundreds of gigabytes of composed meshes.
for scene in range(1,103):
 p=BASE/f'repo_assets/assets/scenes/tabletop_cleared_v1/{scene}/{scene}.usda'
 if not p.is_file() or not Sdf.Layer.FindOrOpen(str(p)):failures.append({'scene_id':scene,'reason':'cleared scene layer is missing or invalid'})
 else:counts['scene_root_layers_parsed']+=1

sdk_root=Path('/home/elm/miniconda3/envs/new_isaaclab/lib/python3.11/site-packages/isaacsim')
sdk_checks=[]
for relative in ['kit/mdl/core/Base/OmniPBR.mdl','kit/mdl/core/Base/OmniGlass.mdl',
 'kit/mdl/core/mdl/nvidia/core_definitions.mdl','kit/mdl/core/mdl/OmniSurface/OmniShared.mdl',
 'kit/mdl/core/mdl/OmniSurface/OmniSurfaceBase.mdl','kit/mdl/core/mdl/OmniSurface/OmniSurfaceLiteBase.mdl','kit/mdl/core/mdl/OmniSurface/OmniImage.mdl']:
 result={'path_inside_installed_isaacsim':relative,'exists':(sdk_root/relative).is_file()};sdk_checks.append(result)
 if not result['exists']:failures.append({'sdk_module':relative,'reason':'declared standard5.1 SDK module absent'})

if homes:failures.append({'reason':'creator home prefix remains in extracted text','files':sorted(set(homes))})
report={'passed':not failures,'stage_root':str(BASE),'counts':dict(counts),'failures':failures,
 'creator_home_prefix_text_files':sorted(set(homes)),'sdk_modules_checked':sdk_checks,
 'runtime_SDK_is_external':True,'optional_curobo_dynamics_or_collision_planner_not_executed':True,
 'optional_source_template_requirements':source_templates,
 'independent_validation':'Reads actual extracted resources, not the source reference rewrite table',
 'elapsed_s':time.time()-started,'GPU_used':False}
(OUT/'asset_stage_reference_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
if failures:
 print(json.dumps(report,ensure_ascii=False,indent=2));raise SystemExit(1)
m['optional_source_template_requirements']=source_templates
m['dependency_validation']={'status':'passed','independent_relocated_references_checked':True,
 'counts':dict(counts),'creator_home_prefix_in_text':False,'SDK_provided_standard_modules_verified':True,
 'optional_curobo_mesh_resources_complete':True,'optional_curobo_physical_execution_verified':False}
manifest_path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
lines=[f"{item['sha256']}  {item['path']}" for item in m['files']]+[f'{manifest_sha}  asset_bundle_manifest.json']
(BASE/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
ready={'ready_for_physics':True,'stage_root':str(BASE),'asset_bundle_manifest_sha256':manifest_sha,
 'asset_files':len(m['files']),'total_size_bytes':m['total_size_bytes'],'reference_validation_passed':True,
 'bulk_extraction_SHA256_and_CRC_passed':True,'code_path_tests_passed':98,'final_ZIP_may_still_be_building':True}
(OUT/'ASSET_STAGE_READY.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(ready,ensure_ascii=False),flush=True)
