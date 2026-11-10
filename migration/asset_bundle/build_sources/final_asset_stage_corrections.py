from pathlib import Path
import json,hashlib,shutil
OUT=Path('/home/elm/projects/geomind_lab/outputs/research/portable_delivery_v1');BASE=OUT/'relocated_validation/geomind-assets-v1'
manifest_path=BASE/'asset_bundle_manifest.json';m=json.loads(manifest_path.read_text());rows={x['path']:x for x in m['files']}
audit=json.loads((OUT/'asset_closure_inventory_v4.json').read_text());source_rows={x['archive_path']:x for x in audit['files']}
references={x['target'] for x in audit['edges'] if '.thumbs/' in x['target']}
whitelist=[]
for key in sorted(references):
 source=Path(source_rows[key]['source']);data=source.read_bytes()
 if not data.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Declared thumbnail is not a PNG resource: '+key)
 target=BASE/key;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);digest=hashlib.sha256(data).hexdigest()
 rows[key]={'path':key,'size_bytes':len(data),'sha256':digest,'source_sha256':digest,'source_size_bytes':len(data),
  'reference_paths_normalized':False,'asset_role':'Explicit MDL thumbnail resource; not an unreferenced cache'}
 whitelist.append(key)
views=[]
for key,row in rows.items():
 if key.startswith('repo_assets/scene_atlas/') and key.endswith('.json'):
  p=BASE/key
  try:d=json.loads(p.read_text())
  except (ValueError,UnicodeError):continue
  if not isinstance(d,dict):continue
  if any(token in key for token in ('/assembled_tasks','/assemblies')):
   d['portable_view_only']=True;d['original_source_sha256']=row['source_sha256']
   d['portable_metadata_notice']='Path-rebased UI metadata copy. This is not the original hash-bound PhysX evidence. Original source files were not modified; the verified canonical27 raw evidence is delivered separately.'
   data=(json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode();p.write_bytes(data)
   row.update(size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),reference_paths_normalized=True,
    portable_view_only=True,original_source_sha256=row['source_sha256']);views.append(key)
m['files']=sorted(rows.values(),key=lambda x:x['path']);m['total_asset_files']=len(rows);m['total_size_bytes']=sum(x['size_bytes'] for x in rows.values())
m['declared_thumbnail_assets']=whitelist
m['portable_ui_metadata_views']={'files':views,'original_physics_evidence':False,'original_host_files_modified':False,
 'notice':'Historical atlas records are relocated presentation metadata, not newly accepted robot results or raw physics evidence.'}
m['excluded_non_asset_components']=[x.replace('cache thumbnails','unreferenced cache thumbnails') for x in m['excluded_non_asset_components']]
manifest_path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
print('Included explicit MDL preview PNGs',len(whitelist),'UI metadata views marked',len(views),'total',len(rows),m['total_size_bytes'])
