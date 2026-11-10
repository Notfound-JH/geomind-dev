"""Mirror the four original NVIDIA wood modules and exact dependency layout."""
from pathlib import Path
import json,urllib.request,urllib.parse
ROOT=Path(__file__).resolve().parents[1]
cache=ROOT/'outputs/scene_atlas_materials_v1/material_cache'
modules=cache/'nvidia'
base='https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/'
mapping=json.loads((cache/'mapping.json').read_text())
result={}
for url,textures in mapping.items():
 rel=url.split('/Materials/2023_1/',1)[1]
 dest=modules/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 dest.write_text((cache/(Path(rel).name+'.txt')).read_text())
 result[url]=str(dest)
 for tex in textures.values():
  path=modules/tex['url'].split('/Materials/2023_1/',1)[1]
  path.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists():path.symlink_to(tex['path'])
template=modules/'Automotive/Templates/OmniUber_Automotive.mdl'
template.parent.mkdir(parents=True,exist_ok=True)
if not template.exists():template.write_bytes(urllib.request.urlopen(base+'Automotive/Templates/OmniUber_Automotive.mdl',timeout=30).read())
(cache/'modules.json').write_text(json.dumps(result,indent=2))
print('LOCAL_MODULES_READY',len(result))
