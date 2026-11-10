#!/usr/bin/env python3
"""Portably repackage an already extracted GEOMIND asset bundle; stdlib only."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,os,zipfile

def repack(base,output):
 base=Path(base).resolve();output=Path(output).resolve();manifest=json.loads((base/'asset_bundle_manifest.json').read_text())
 if manifest.get('bundle_name')!='geomind-assets-v1':raise ValueError('Unsupported bundle')
 temporary=output.with_suffix(output.suffix+'.building')
 if output.exists() or temporary.exists():raise FileExistsError('Refusing to overwrite ZIP/build')
 entries=[*manifest['files'],{'path':'asset_bundle_manifest.json'},{'path':'SHA256SUMS'}]
 names=[entry['path'] for entry in entries]
 if len(names)!=len(set(names)):raise ValueError('Duplicate manifest path')
 with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
  for entry in entries:
   name=PurePosixPath(entry['path']);source=(base/str(name)).resolve()
   if name.is_absolute() or '..' in name.parts or not source.is_relative_to(base) or not source.is_file():raise ValueError('Unsafe/missing resource')
   info=zipfile.ZipInfo(manifest['bundle_name']+'/'+str(name),date_time=(2026,11,10,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info._compresslevel=1;info.external_attr=0o100644<<16
   h=hashlib.sha256();length=0
   with source.open('rb') as stream,archive.open(info,'w',force_zip64=True) as target:
    for chunk in iter(lambda:stream.read(4*1024*1024),b''):target.write(chunk);h.update(chunk);length+=len(chunk)
   if 'sha256' in entry and (h.hexdigest()!=entry['sha256'] or length!=entry['size_bytes']):raise ValueError('Changed resource '+str(name))
 os.replace(temporary,output);h=hashlib.sha256()
 with output.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 output.with_suffix(output.suffix+'.sha256').write_text(h.hexdigest()+'  '+output.name+'\n')
 return {'zip':str(output),'sha256':h.hexdigest(),'resource_files_verified':len(manifest['files'])}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();print(json.dumps(repack(a.assets_root,a.output),indent=2))
