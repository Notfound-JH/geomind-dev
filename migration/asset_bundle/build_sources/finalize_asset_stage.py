"""Finalize immutable relocated assets; official supplements stay traceable."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,datetime
OUT=Path('/home/elm/projects/geomind_lab/outputs/research/portable_delivery_v1')
spec=importlib.util.spec_from_file_location('asset_builder',OUT/'build_asset_zip.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
audit=json.loads((OUT/'asset_closure_inventory_final.json').read_text());normal=builder.Normalizer(audit)
bulk=json.loads((OUT/'asset_zip_bulk_manifest.json').read_text());stage=OUT/'relocated_validation/geomind-assets-v1';rows={x['path']:dict(x) for x in bulk['files']}

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()

def json_normalize(value):
 if isinstance(value,dict):
  result={}
  for key,item in value.items():
   changed=normal.json_data(key)
   if changed in result:raise ValueError('Metadata alias normalization collapsed distinct keys')
   result[changed]=json_normalize(item)
  return result
 if isinstance(value,list):return [json_normalize(item) for item in value]
 return normal.json_data(value)

# Source hash dictionaries have path keys too, not only string values.
for key,row in rows.items():
 p=stage/key
 if p.suffix.lower()=='.json' and b'/home/elm/' in p.read_bytes():
  data=(json.dumps(json_normalize(json.loads(p.read_text())),ensure_ascii=False,indent=2)+'\n').encode()
  p.write_bytes(data);row.update(size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),reference_paths_normalized=True)

supp_root=OUT/'supplemental_fr3_meshes';supp=json.loads((supp_root/'supplement_manifest.json').read_text())
if not supp.get('passed'):raise ValueError('Official FR3 supplement did not pass its provenance/geometry checks')
for item in supp['files']:
 source=supp_root/item['local_relative_path'];key=builder.safe_relative(item['logical_path']);target=stage/key
 if key in rows:raise ValueError('Supplement collides with immutable bulk data: '+key)
 if source.stat().st_size!=item['size_bytes'] or digest(source)!=item['sha256']:raise ValueError('Supplement changed before copy: '+key)
 target.parent.mkdir(parents=True,exist_ok=True);data=source.read_bytes()
 if target.suffix.lower()=='.json':data=(json.dumps(json_normalize(json.loads(data)),ensure_ascii=False,indent=2)+'\n').encode()
 if b'/home/elm/' in data:data=normal.metadata(data.decode('utf-8')).encode()
 target.write_bytes(data)
 rows[key]={'path':key,'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
  'source_sha256':item.get('source_sha256',item['sha256']),'source_size_bytes':item['size_bytes'],
  'reference_paths_normalized':data!=source.read_bytes(),'official_source_url':item.get('source_url'),
  'official_source_commit':supp['official_commit'],'transformation':item.get('transformation','provenance metadata copied')}

sdk=[]
for value in audit.get('sdk_provided',[]):
 item={'reference':value['reference'],'provider':value['provider']}
 if item not in sdk:sdk.append(item)
manifest={'schema_version':1,'bundle_name':'geomind-assets-v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'purpose':'Relocatable GEOMIND scenes, catalogs, robot and material dependencies; no simulator executable or model weights',
 'files':sorted(rows.values(),key=lambda x:x['path']),'total_asset_files':len(rows),
 'total_size_bytes':sum(x['size_bytes'] for x in rows.values()),
 'roots':['geomind_assets','grscenes_home','isaacsim_assets','material_cache','repo_assets/assets','repo_assets/scene_atlas','curobo_assets'],
 'supported_scene_ids':list(range(1,103)),
 'catalog_coverage':{'runtime_catalog_assets':64,'preserved_authoring_full_catalog_assets':70},
 'robot_embodiments':['franka','franka_robotiq','fr3','piper','piper_dual'],
 'sdk_provided':sdk,'required_runtime':{'isaacsim':'5.1.0.0','python':'3.11','SDK_not_included':True},
 'normalization':{'usd_and_texture_references':'Relative to each owner file within bundle',
  'json_asset_paths':'bundle:// URIs; configure writes a machine-local catalog',
  'historical_atlas_metadata':'Origin paths rebound; copied atlas is presentation data, not a new raw physics proof',
  'urdf_and_collada':'Concrete mesh/image references checked; official supplement DAE declaration position normalized without changing XML DOM values'},
 'external_original_files_modified':False,'contains_GPU_execution':False,
 'excluded_non_asset_components':['Nested Git metadata','credentials and secret directories','cache thumbnails','stage_old backup scenes','IsaacSim executable/SDK','unrelated full GRScenes model collection','training/model checkpoints'],
 'optional_planner_scope':{'FR3_missing_meshes_restored':19,'official_commit':supp['official_commit'],
  'original_custom_joint_limits_preserved':True,'curobo_physical_execution_verified':False},
 'dependency_validation':{'status':'pending_final_reference_validation'}}
(stage/'asset_bundle_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(OUT/'asset_stage_finalization.json').write_text(json.dumps({'stage_root':str(stage),'asset_files':len(rows),'total_size_bytes':manifest['total_size_bytes'],'manifest_sha256':digest(stage/'asset_bundle_manifest.json'),'official_supplement_files':supp['total_files'],'ready_for_physics':False},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'stage_root':str(stage),'files':len(rows),'bytes':manifest['total_size_bytes']},ensure_ascii=False))
