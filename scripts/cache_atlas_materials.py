"""Cache exact NVIDIA MDL texture defaults for read-only preview overrides."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,re,shutil,urllib.request,urllib.parse
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/scene_atlas_materials_v1/material_cache'
OUT.mkdir(parents=True,exist_ok=True)
base='https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/Automotive/Wood/'
mapping={}
for name in ['Wood_Grain_Raw_02','Wood_Grain_Raw_03','Wood_Grain_Raw_04','Wood_Grain_Raw_05']:
 url=base+name+'.mdl'
 source=OUT/(name+'.mdl.txt')
 text=source.read_text() if source.exists() else urllib.request.urlopen(url,timeout=20).read().decode()
 source.write_text(text)
 overrides={}
 for param,rel in re.findall(r'^\s*(\w+):\s*texture_2d\("([^"]+)"',text,re.M):
  texurl=urllib.parse.urljoin(url,rel)
  path=OUT/Path(rel).name
  overrides[param]={'url':texurl,'path':str(path)}
 mapping[url]=overrides
jobs={v['path']:v['url'] for overrides in mapping.values() for v in overrides.values()}
def download(item):
 path,url=Path(item[0]),item[1]
 if not path.exists():
  print('DOWNLOAD',path.name,flush=True)
  with urllib.request.urlopen(url,timeout=45) as r, path.with_suffix(path.suffix+'.partial').open('wb') as f:shutil.copyfileobj(r,f)
  path.with_suffix(path.suffix+'.partial').rename(path)
 return str(path),hashlib.sha256(path.read_bytes()).hexdigest()
with ThreadPoolExecutor(max_workers=5) as pool:hashes=dict(pool.map(download,jobs.items()))
for overrides in mapping.values():
 for v in overrides.values():v['sha256']=hashes[v['path']]
(OUT/'mapping.json').write_text(json.dumps(mapping,indent=2))
print('CACHE_DONE',len(mapping),len(jobs),flush=True)
