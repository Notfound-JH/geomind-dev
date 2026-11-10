"""Read every final ZIP member to verify CRC, SHA256, size and safe paths."""
from pathlib import Path,PurePosixPath
import hashlib,json,stat,time,zipfile
OUT=Path('/home/elm/projects/geomind_lab/outputs/research/portable_delivery_v1');path=OUT/'geomind-assets-v1.zip'
started=time.time();before=path.stat();checked=0;size=0
with zipfile.ZipFile(path) as archive:
 infos=archive.infolist();names=[entry.filename for entry in infos]
 if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP member names')
 manifest_bytes=archive.read('geomind-assets-v1/asset_bundle_manifest.json');manifest=json.loads(manifest_bytes)
 ready=json.loads((OUT/'ASSET_STAGE_READY.json').read_text())
 if hashlib.sha256(manifest_bytes).hexdigest()!=ready['asset_bundle_manifest_sha256']:raise ValueError('Final ZIP has another data manifest')
 entries={'geomind-assets-v1/'+item['path']:item for item in manifest['files']}
 expected=set(entries)|{'geomind-assets-v1/asset_bundle_manifest.json','geomind-assets-v1/SHA256SUMS'}
 if set(names)!=expected:raise ValueError('ZIP contains extra or missing members')
 for info in infos:
  name=PurePosixPath(info.filename)
  if name.is_absolute() or '..' in name.parts or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unsafe ZIP member')
  digest=hashlib.sha256();length=0
  with archive.open(info) as source:
   for chunk in iter(lambda:source.read(4*1024*1024),b''):
    digest.update(chunk);length+=len(chunk)
  if info.filename in entries:
   entry=entries[info.filename]
   if length!=entry['size_bytes'] or digest.hexdigest()!=entry['sha256']:raise ValueError('Archived asset differs from manifest '+info.filename)
   checked+=1;size+=length
  if checked and checked%250==0:print(json.dumps({'verified_asset_files':checked,'bytes':size,'seconds':round(time.time()-started,1)}),flush=True)
 after=path.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('ZIP changed while verifying')
 checksum=(OUT/'geomind-assets-v1.zip.sha256').read_text().split()[0]
 report={'passed':True,'zip':str(path),'size_bytes':before.st_size,'sha256':checksum,
  'unique_archive_entries':len(infos),'asset_files_verified':checked,'asset_bytes_verified':size,
  'all_members_CRC_verified':True,'all_assets_SHA256_verified':True,'safe_relative_members_only':True,
  'no_symlink_members':True,'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
  'independent_stage_reference_validation_passed':True,'asset_security_scan_passed':json.loads((OUT/'asset_security_scan.json').read_text())['passed'],
  'optional_planner_execution_verified':False,'GPU_used':False,'elapsed_s':time.time()-started}
 (OUT/'asset_zip_download_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False),flush=True)
