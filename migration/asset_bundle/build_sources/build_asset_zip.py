"""Create a ZIP64 of normalized asset copies; never modify source assets."""
from pathlib import Path,PurePosixPath
import collections,datetime,hashlib,json,os,posixpath,re,tempfile,time,zipfile
from pxr import Sdf,UsdUtils
ROOT=Path('/home/elm/projects/geomind_lab');OUT=ROOT/'outputs/research/portable_delivery_v1'
AUDIT=OUT/'asset_closure_inventory_final.json'
BUNDLE='geomind-assets-v1'
ZIP=OUT/(BUNDLE+'.partial.zip')
ASSET_RE=re.compile(rb'@@@([^\r\n]*?)@@@|@([^@\r\n]+)@')
TEXT_SUFFIXES={'.usda','.mdl','.json','.xml','.urdf','.xacro','.dae','.obj','.mtl','.txt','.md','.csv','.yaml','.yml','.sh','.py','.html','.css','.js'}
SDK_ASSETS={'OmniPBR.mdl','OmniGlass.mdl'}


def safe_relative(name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('Unsafe archive path: '+name)
 return p.as_posix()


class Normalizer:
 def __init__(self,audit):
  self.audit=audit;self.entries={x['archive_path']:x for x in audit['files']}
  self.sources={str(Path(x['source']).resolve()):x['archive_path'] for x in audit['files']}
  self.edge={}
  for item in [*audit['edges'],*audit.get('xml_edges',[])]:self.edge[(item['owner'],item['reference'])]=item['target']
  self.owner_edges=collections.defaultdict(list)
  for (name,reference),target in self.edge.items():self.owner_edges[name].append((reference,target))
  self.roots=[(value,key) for key,value in audit['root_directories'].items()]
  self.roots.sort(key=lambda pair:-len(pair[0]))
  self.deferred=[]
  for row in audit.get('xml_missing',[]):
   if row.get('reference'):
    target=posixpath.normpath(posixpath.join(posixpath.dirname(row['owner']),row['reference']))
    if target.startswith('../'):raise ValueError('Deferred XML resource escaped bundle')
    self.edge[(row['owner'],row['reference'])]=target
    self.deferred.append({'owner':row['owner'],'reference':row['reference'],'target':target})

 def target(self,raw,owner=None):
  if (owner,raw) in self.edge:return self.edge[(owner,raw)]
  value=raw[5:] if raw.startswith('file:') else raw
  if value in self.sources:return self.sources[value]
  for source,logical in self.roots:
   if value==source or value.startswith(source+'/'):return logical+value[len(source):]
  if '/Assets/ArchVis/' in value:
   candidate='isaacsim_assets/Assets/Isaac/4.5/NVIDIA/Assets/ArchVis/'+value.split('/Assets/ArchVis/',1)[1]
   if candidate in self.entries:return candidate
  if '/Assets/DigitalTwin/' in value:
   candidate='isaacsim_assets/Assets/Isaac/5.1/NVIDIA/Assets/DigitalTwin/'+value.split('/Assets/DigitalTwin/',1)[1]
   if candidate in self.entries:return candidate
  if '/Materials/' in value and value.startswith(('http://','https://')):
   for version in ('5.1','4.5'):
    candidate=f'isaacsim_assets/Assets/Isaac/{version}/NVIDIA/Materials/'+value.split('/Materials/',1)[1]
    if candidate in self.entries:return candidate
  return None

 def reference(self,raw,owner):
  if raw in SDK_ASSETS:return raw
  target=self.target(raw,owner)
  if target is None:raise ValueError(f'Concrete asset reference was not in audited closure: {owner}: {raw}')
  return posixpath.relpath(target,posixpath.dirname(owner))

 def metadata(self,text):
  # Only compatibility origin strings are changed; actual USD/MDL resource
  # references have already been mapped using the independently audited graph.
  for source,logical in self.roots:
   text=text.replace(source,'bundle://'+logical)
  text=text.replace('/home/elm/projects/geomind_lab/','repo://')
  text=text.replace('/home/elm/projects/geomind/scene_construction/','repo://scene_construction /')
  text=text.replace('/home/elm/','historical-host://')
  return text

 def json_data(self,value):
  if isinstance(value,dict):return {key:self.json_data(item) for key,item in value.items()}
  if isinstance(value,list):return [self.json_data(item) for item in value]
  if isinstance(value,str):
   target=self.target(value)
   return 'bundle://'+target if target else self.metadata(value)
  return value

 def line(self,data,owner,suffix):
  if suffix=='.usda' and b'@' in data:
   def replace(match):
    raw=next(item for item in match.groups() if item).decode('utf-8')
    value=self.reference(raw,owner).encode('utf-8')
    return b'@'+value+b'@'
   data=ASSET_RE.sub(replace,data)
  if suffix in {'.mdl','.xml','.urdf','.xacro','.dae','.obj','.mtl'}:
   # XML resource references and MDL texture strings retain original geometry
   # and serialization, only their declared paths change.
   for old,target in self.owner_edges[owner]:
    if not old.startswith(('::','.::','..::')):
     new=posixpath.relpath(target,posixpath.dirname(owner))
     data=data.replace(old.encode('utf-8'),new.encode('utf-8'))
  if b'/home/elm/' in data:data=self.metadata(data.decode('utf-8')).encode('utf-8')
  return data


def _chunks(path,size=4*1024*1024):
 with path.open('rb') as f:
  yield from iter(lambda:f.read(size),b'')


def build_bulk():
 audit=json.loads(AUDIT.read_text());normal=Normalizer(audit)
 if audit['missing'] or audit['remote'] or audit['mdl_modules_requiring_installed_IsaacSim_or_local_module_resolution']:
  raise ValueError('Concrete USD/MDL closure must be complete before ZIP construction')
 if ZIP.exists():raise FileExistsError('Refusing to overwrite partial asset ZIP')
 entries=[];started=time.time();source_total=0;normalized_total=0;modified=[]
 with tempfile.TemporaryDirectory(prefix='asset_normalized_',dir=OUT) as tmpdir:
  temp=Path(tmpdir)
  with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
   for index,item in enumerate(audit['files'],1):
    owner=safe_relative(item['archive_path']);source=Path(item['source']);suffix=source.suffix.lower()
    original=hashlib.sha256();normalized=hashlib.sha256();size=0;changed=False
    name=BUNDLE+'/'+owner
    info=zipfile.ZipInfo(name,date_time=(2026,11,10,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
    info.external_attr=(0o100644<<16);info._compresslevel=1
    # Binary USDs with relative references are copied byte-for-byte, retaining
    # their existing descriptor SHA. Re-encode only when an actual path changes.
    use_path=source
    if suffix in {'.usd','.usdc'}:
     all_refs=[ref for group in UsdUtils.ExtractExternalReferences(str(source)) for ref in group]
     needs=any(ref not in SDK_ASSETS and posixpath.normpath(normal.reference(ref,owner))!=posixpath.normpath(ref) for ref in all_refs)
     if needs:
      layer=Sdf.Layer.FindOrOpen(str(source))
      clone=Sdf.Layer.CreateAnonymous('portable.usda');clone.ImportFromString(layer.ExportToString())
      UsdUtils.ModifyAssetPaths(clone,lambda value:normal.reference(value,owner))
      text=normal.metadata(clone.ExportToString());clone.ImportFromString(text)
      use_path=temp/('normalized'+suffix);clone.Export(str(use_path));changed=True
    with archive.open(info,'w',force_zip64=True) as dest:
     if suffix=='.json':
      raw=source.read_bytes();original.update(raw)
      try:data=(json.dumps(normal.json_data(json.loads(raw.decode('utf-8-sig'))),ensure_ascii=False,indent=2)+'\n').encode('utf-8')
      except (json.JSONDecodeError,UnicodeDecodeError):data=normal.line(raw,owner,suffix)
      normalized.update(data);dest.write(data);size=len(data);changed=changed or data!=raw
     elif suffix in TEXT_SUFFIXES:
      with source.open('rb') as f:
       for line in f:
        original.update(line);data=normal.line(line,owner,suffix);normalized.update(data);dest.write(data);size+=len(data);changed=changed or data!=line
     else:
      if use_path!=source:
       for chunk in _chunks(source):original.update(chunk)
      for chunk in _chunks(use_path):
       if use_path==source:original.update(chunk)
       normalized.update(chunk);dest.write(chunk);size+=len(chunk)
    if use_path!=source:use_path.unlink()
    row={'path':owner,'size_bytes':size,'sha256':normalized.hexdigest(),'source_sha256':original.hexdigest(),
         'source_size_bytes':source.stat().st_size,'reference_paths_normalized':changed}
    entries.append(row);source_total+=source.stat().st_size;normalized_total+=size
    if changed:modified.append(owner)
    if index%50==0:
     print(json.dumps({'files':index,'total_files':len(audit['files']),'input_GB':round(source_total/1e9,3),'zip_GB':round(ZIP.stat().st_size/1e9,3),'elapsed_s':round(time.time()-started,1)}),flush=True)
    if index%100==0:(OUT/'asset_zip_bulk_progress.json').write_text(json.dumps({'status':'building','files':index,'total_files':len(audit['files']),'input_bytes':source_total,'zip_bytes':ZIP.stat().st_size,'elapsed_s':time.time()-started},indent=2)+'\n')
 report={'status':'bulk_complete_pending_final_validation','zip':str(ZIP),'bundle_name':BUNDLE,'files':entries,
  'source_total_bytes':source_total,'normalized_total_bytes':normalized_total,'modified_files':modified,
  'deferred_xml_resources':normal.deferred,'elapsed_s':time.time()-started,'source_assets_modified':False,'GPU_used':False}
 (OUT/'asset_zip_bulk_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['status','source_total_bytes','normalized_total_bytes','elapsed_s']},ensure_ascii=False),flush=True)

if __name__=='__main__':build_bulk()
