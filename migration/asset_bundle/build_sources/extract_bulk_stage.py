"""Extract and hash the immutable bulk ZIP at a different prefix, without GPU."""
from pathlib import Path,PurePosixPath
import hashlib,json,time,zipfile
OUT=Path('/home/elm/projects/geomind_lab/outputs/research/portable_delivery_v1')
DEST=OUT/'relocated_validation'
meta=json.loads((OUT/'asset_zip_bulk_manifest.json').read_text());entries={x['path']:x for x in meta['files']}
root=DEST/meta['bundle_name'];root.mkdir(parents=True,exist_ok=True)
started=time.time();checked=0;size=0;old_prefix=[]
with zipfile.ZipFile(OUT/'geomind-assets-v1.partial.zip') as archive:
 for info in archive.infolist():
  name=PurePosixPath(info.filename)
  if name.is_absolute() or '..' in name.parts or name.parts[0]!=meta['bundle_name']:raise ValueError('Unsafe ZIP extraction path')
  relative='/'.join(name.parts[1:]);expected=entries[relative];target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
  digest=hashlib.sha256();length=0;previous=b''
  with archive.open(info) as source,target.open('wb') as dest:
   for chunk in iter(lambda:source.read(4*1024*1024),b''):
    if b'/home/elm/' in previous+chunk:old_prefix.append(relative)
    digest.update(chunk);dest.write(chunk);length+=len(chunk);previous=chunk[-16:]
  if length!=expected['size_bytes'] or digest.hexdigest()!=expected['sha256']:raise ValueError('Extraction hash mismatch '+relative)
  checked+=1;size+=length
  if checked%100==0:print(json.dumps({'files':checked,'bytes':size,'seconds':round(time.time()-started,1)}),flush=True)
report={'status':'bulk_extracted_and_sha256_verified','stage_root':str(root),'files':checked,'bytes':size,
 'elapsed_s':time.time()-started,'zip_CRC_verified_by_reader':True,'creator_home_prefix_found':sorted(set(old_prefix)),
 'final_bundle_ready':False,'pending':'Official FR3 supplemental meshes and final internal manifest/reference validation'}
(OUT/'asset_stage_bulk_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False),flush=True)
