"""Add the saved cleared variants and verified renders to the existing atlas."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1];SITE=ROOT/'scene_atlas/dist';OUT=ROOT/'outputs/scene_atlas_cleared_v1';SOURCE=ROOT/'assets/scenes/tabletop_cleared_v1'
backup=OUT/'original_site'
if not backup.exists():
 backup.mkdir(parents=True)
 for name in ['index.html','app.js','style.css','scenes.json','data.js']:shutil.copyfile(SITE/name,backup/name)
scenes=json.loads((backup/'scenes.json').read_text());renders={}
for folder in [OUT/'render_probe',*sorted(OUT.glob('render_batch_*'))]:
 if not folder.is_dir():continue
 for p in folder.glob('scene-*.json'):
  r=json.loads(p.read_text());renders[r['id']]=r
saved=[]
for s in scenes:
 p=SOURCE/str(s['id'])/'removal_report.json'
 if not p.exists():continue
 d=json.loads(p.read_text());saved.append(d)
 cleared={'usda':d['output'],'file_size':Path(d['output']).stat().st_size,'removed_item_count':d['removed_item_count'],'saved_reopened_verified':d['saved_reopened_verified'],'thumbnail':None,'removal_mode':'USD active=false overrides remove rendering and physics; source assets retained as sublayer'}
 if s['id'] in renders:
  r=renders[s['id']];image=Path(r['image'])
  assert hashlib.sha256(image.read_bytes()).hexdigest()==r['image_sha256']
  name=f"assets/scene-{s['id']}-cleared-v1.png";shutil.copyfile(image,SITE/name);cleared['thumbnail']=name
 s['cleared']=cleared
name_index=SOURCE/'scene_index.json'
if name_index.exists():
 names={r['scene_id']:r for r in json.loads(name_index.read_text())['scenes']}
 for s in scenes:
  if s['id'] in names:
   n=names[s['id']];s.update(scene_name=n['scene_name'],environment_group=n['environment_group'],environment_group_size=n['group_size'])
(SITE/'scenes.json').write_text(json.dumps(scenes,ensure_ascii=False));(SITE/'data.js').write_text('window.SCENES = '+json.dumps(scenes,ensure_ascii=False)+';\n')
summary={'saved_scenes':len(saved),'rendered_scenes':len(renders),'removed_items':sum(d['removed_item_count'] for d in saved),'originals_unchanged':all(d['original_unchanged'] for d in saved),'saved_files_reopened_verified':all(d['saved_reopened_verified'] for d in saved),'output_root':str(SOURCE),'pending_save':[s['id'] for s in scenes if not s.get('cleared')],'pending_render':[s['id'] for s in scenes if s['id'] not in renders]}
(OUT/'delivery.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
