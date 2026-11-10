#!/usr/bin/env python3
"""Replace only wall geometry/material blocks; back up sources and record seeded assignments."""
import argparse,hashlib,json,random,re,shutil
from pathlib import Path
from pxr import Sdf,Usd,UsdGeom,UsdShade
ROOT=Path(__file__).resolve().parents[1]
def span(text,start):
 a=text.index('{',start); depth=1;i=a+1
 while depth:
  depth+=(text[i]=='{')-(text[i]=='}');i+=1
 return i

def rewrite(text,name):
 candidate=(ROOT/'outputs/wall_texture_comparison'/f'{name}.usda').read_text()
 m=re.search(r'def Material "WallCandidate"',candidate); material=candidate[m.start():span(candidate,m.end())]
 # Candidate layers contain paths relative to outputs/wall_texture_comparison.
 material=re.sub(r'@([^@]+)@',lambda m:'@'+str((ROOT/'outputs/wall_texture_comparison'/m[1]).resolve())+'@',material)
 edits=[]
 wall_materials=set()
 for wall in ['Wall','Wall_1','Wall_2','Wall_3']:
  wm=re.search(r'def (?:Cube|Mesh) "'+wall+'"',text);assert wm,wall
  wb=text[wm.start():span(text,wm.end())]
  binding=re.search(r'rel material:binding\s*=\s*<([^>]+)>',wb);assert binding,wall
  wall_materials.add(binding[1].rsplit('/',1)[-1])
 for mat in sorted(wall_materials):
  m=re.search(r'def Material "'+mat+'"',text); assert m,mat
  edits.append((m.start(),span(text,m.end()),material.replace('WallCandidate',mat)))
 pts=[(-2.5,-2.5,-2.5),(2.5,-2.5,-2.5),(2.5,2.5,-2.5),(-2.5,2.5,-2.5),(-2.5,-2.5,2.5),(2.5,-2.5,2.5),(2.5,2.5,2.5),(-2.5,2.5,2.5)]
 faces=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
 for wall in ['Wall','Wall_1','Wall_2','Wall_3']:
  m=re.search(r'def Cube "'+wall+'"',text);assert m,wall
  end=span(text,m.end());b=text[m.start():end]; scale=tuple(map(float,re.search(r'xformOp:scale = \(([^)]+)\)',b)[1].split(',')))
  assert re.search(r'double size = 5\b',b)
  uv=[]
  for face in faces:
   axes=[a for a in range(3) if len({pts[i][a] for i in face})>1]
   uv.extend(tuple((pts[i][a]+2.5)*scale[a]/2 for a in axes) for i in face)
  b=b.replace('def Cube','def Mesh',1); b=re.sub(r'\s*double size = 5\b','',b)
  attrs='\n            point3f[] points = '+str(pts)+'\n            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\n            int[] faceVertexIndices = '+str([i for f in faces for i in f])+'\n            uniform token subdivisionScheme = "none"\n            texCoord2f[] primvars:st = '+str(uv)+' (\n                interpolation = "faceVarying"\n            )\n'
  b=b[:b.rfind('}')]+attrs+'        }'; edits.append((m.start(),end,b))
 outside=[];changed=[];cursor=0
 for start,end,b in sorted(edits):
  outside.extend([text[cursor:start],'<WALL_EDIT>']);changed.extend([text[cursor:start],b]);cursor=end
 outside.append(text[cursor:]);changed.append(text[cursor:])
 return ''.join(changed),hashlib.sha256(''.join(outside).encode()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=int,default=20261110);ap.add_argument('--start',type=int,default=1);ap.add_argument('--end',type=int,default=102);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 assert not (args.output/'manifest.json').exists(),'Use a fresh output directory'
 rng=random.Random(args.seed); choices=['beige_wall_001']*51+['beige_wall_002']*51;rng.shuffle(choices);records=[]
 # Preflight every file before writing any source.
 pending=[]
 for n,name in enumerate(choices,1):
  if not args.start<=n<=args.end:continue
  folder=ROOT.parent/'geomind_assets/stage'/str(n);files=[p for p in (folder/f'{n}.usda',folder/f'{n}_new.usda') if p.exists()];assert files
  for p in files:
   before=p.read_text();after,other_hash=rewrite(before,name); layer=Sdf.Layer.CreateAnonymous();assert layer.ImportFromString(after)
   temp=args.output/'prepared'/str(n)/p.name;temp.parent.mkdir(parents=True,exist_ok=True);temp.write_text(after)
   print(f'PREFLIGHT {n} {p.name}',flush=True)
   pending.append((n,name,p,temp,hashlib.sha256(before.encode()).hexdigest(),hashlib.sha256(after.encode()).hexdigest(),other_hash))
 for n,name,p,temp,before_hash,after_hash,other_hash in pending:
  backup=args.output/'backup'/str(n)/p.name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,backup);shutil.copyfile(temp,p)
  records.append({'scene_id':n,'material':name,'source':str(p),'backup':str(backup.resolve()),'before_sha256':before_hash,'after_sha256':after_hash,'unchanged_outside_wall_blocks_sha256':other_hash})
 (args.output/'manifest.json').write_text(json.dumps({'seed':args.seed,'scene_count':args.end-args.start+1,'source_files':len(records),'distribution':{k:choices[args.start-1:args.end].count(k) for k in set(choices)},'tile_size_m':2,'records':records},indent=2))
 print(f'Updated {len(records)} source files in selected scene range.')
if __name__=='__main__':main()
