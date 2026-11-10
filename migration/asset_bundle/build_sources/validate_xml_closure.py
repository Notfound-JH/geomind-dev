from pathlib import Path
import json,re,xml.etree.ElementTree as ET
ROOT=Path('/home/elm/projects/geomind_lab');OUT=ROOT/'outputs/research/portable_delivery_v1'
d=json.loads((OUT/'asset_closure_inventory_v4.json').read_text());source_to_logical={str(Path(x['source']).resolve()):x['archive_path'] for x in d['files']}
logical_to_source={x['archive_path']:Path(x['source']) for x in d['files']};edges=[];missing=[];templates=[]
refs_re=re.compile(r'(?:filename|file)\s*=\s*["\']([^"\']+)["\']')
for f in d['files']:
 owner=Path(f['source']);suffix=owner.suffix.lower()
 if suffix not in {'.urdf','.xacro','.xml','.dae','.obj','.mtl'}:continue
 text=owner.read_text(errors='replace');refs=set()
 if suffix in {'.urdf','.xacro','.xml'}:refs.update(refs_re.findall(text))
 if suffix=='.dae':
  try:
   root=ET.fromstring(text)
   for node in root.iter():
    if node.tag.split('}')[-1]=='image':
     for child in node.iter():
      if child.tag.split('}')[-1]=='init_from' and child.text:refs.add(child.text.strip())
  except ET.ParseError as e:missing.append({'owner':f['archive_path'],'reason':'XML parse: '+str(e)})
 if suffix=='.obj':refs.update(re.findall(r'^mtllib\s+(.+)$',text,re.M))
 if suffix=='.mtl':refs.update(re.findall(r'^(?:map_\w+|bump)\s+(.+)$',text,re.M))
 for ref in sorted(refs):
  value=ref.replace('\\','/');candidates=[]
  if 'agx_arm_urdf/' in value:
   candidates.append(ROOT/'assets/robots/agilex_arm_source'/value.split('agx_arm_urdf/',1)[1])
  elif value.startswith('package://'):
   bits=value[len('package://'):].split('/',1);package=bits[0];relative=bits[1] if len(bits)>1 else ''
   if package in {'piper_description','piper'}:candidates.append(ROOT/'assets/robots/agilex_arm_source/piper'/relative)
   for parent in owner.parents:
    if parent.name==package:candidates.append(parent/relative)
  elif '$(' in value or '${' in value:
   templates.append({'owner':f['archive_path'],'reference':ref,'scope':'runtime xacro macro, not a concrete file reference'});continue
  else:candidates.append(owner.parent/value)
  matches=[p.resolve() for p in candidates if p.is_file()]
  if not matches:missing.append({'owner':f['archive_path'],'reference':ref,'candidates':[str(x) for x in candidates]});continue
  path=matches[0];logical=source_to_logical.get(str(path))
  if logical is None:
   missing.append({'owner':f['archive_path'],'reference':ref,'reason':'resolved concrete XML resource absent from bundle manifest','source':str(path)});continue
  edges.append({'owner':f['archive_path'],'reference':ref,'target':logical})
d['xml_edges']=edges;d['xml_missing']=missing;d['xml_templates']=templates
(OUT/'asset_closure_inventory_final.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'xml_reference_edges':len(edges),'missing':missing,'macro_references':templates},ensure_ascii=False,indent=2))
