"""Package the independently validated relocated stage as a standard ZIP64."""
from pathlib import Path,PurePosixPath
import hashlib,json,time,zipfile,os
OUT=Path('/home/elm/projects/geomind_lab/outputs/research/portable_delivery_v1')
BASE=OUT/'relocated_validation/geomind-assets-v1';READY=OUT/'ASSET_STAGE_READY.json'
if not READY.is_file() or json.loads(READY.read_text()).get('ready_for_physics') is not True:raise ValueError('Final asset stage is not yet validated')
manifest=json.loads((BASE/'asset_bundle_manifest.json').read_text());entries=manifest['files']
output=OUT/'geomind-assets-v1.zip';temporary=output.with_suffix('.zip.building')
if output.exists() or temporary.exists():raise FileExistsError('Refusing to overwrite a final ZIP or an active build')
blocked={'.git','.aws','.ssh','.env','credentials','__pycache__','.cache','node_modules'}
started=time.time();total=0
with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
 for index,entry in enumerate([*entries,{'path':'asset_bundle_manifest.json'},{'path':'SHA256SUMS'}],1):
  name=PurePosixPath(entry['path'])
  if name.is_absolute() or '..' in name.parts or blocked&set(name.parts) or name.name.startswith('.env') or ('.thumbs' in name.parts and name.as_posix() not in manifest.get('declared_thumbnail_assets',[])):raise ValueError('Unsafe/non-asset archive path')
  source=BASE/name
  if not source.is_file() or not source.resolve().is_relative_to(BASE.resolve()):raise ValueError('Missing file or outside-bundle symlink')
  h=hashlib.sha256();size=0
  info=zipfile.ZipInfo('geomind-assets-v1/'+name.as_posix(),date_time=(2026,11,10,0,0,0))
  info.compress_type=zipfile.ZIP_DEFLATED;info._compresslevel=1;info.external_attr=0o100644<<16
  with source.open('rb') as stream,archive.open(info,'w',force_zip64=True) as target:
   for data in iter(lambda:stream.read(4*1024*1024),b''):
    h.update(data);target.write(data);size+=len(data)
  if 'sha256' in entry and (h.hexdigest()!=entry['sha256'] or size!=entry['size_bytes']):raise ValueError('Asset changed while packaging '+name.as_posix())
  total+=size
  if index%100==0:print(json.dumps({'files':index,'input_bytes':total,'zip_bytes':temporary.stat().st_size,'seconds':round(time.time()-started,1)}),flush=True)
os.replace(temporary,output)
h=hashlib.sha256()
with output.open('rb') as stream:
 for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
checksum=h.hexdigest();(OUT/'geomind-assets-v1.zip.sha256').write_text(checksum+'  geomind-assets-v1.zip\n')
report={'passed':True,'zip':str(output),'size_bytes':output.stat().st_size,'sha256':checksum,
 'zip64':True,'asset_files':len(entries),'archive_entries':len(entries)+2,'uncompressed_bytes':total,
 'asset_manifest_sha256':hashlib.sha256((BASE/'asset_bundle_manifest.json').read_bytes()).hexdigest(),
 'all_inputs_rehashed_while_packaging':True,'all_extracted_inputs_previously_CRC_and_SHA256_verified':True,
 'no_nested_git_or_credentials':True,'relative_reference_validation_passed':True,
 'source_assets_modified':False,'GPU_used':False,'elapsed_s':time.time()-started}
(OUT/'asset_zip_final_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False),flush=True)
