"""Preserve original stage timing metadata in already-saved cleared variants."""
from pathlib import Path
from pxr import Sdf
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'assets/scenes/tabletop_cleared_v1';OUT=ROOT/'outputs/scene_atlas_cleared_v1'
backup=OUT/'usda_before_metadata';backup.mkdir(exist_ok=True)
keys=['startTimeCode','endTimeCode','timeCodesPerSecond','framesPerSecond','defaultPrim','upAxis','metersPerUnit','colorManagementSystem','colorConfiguration','renderSettingsPrimPath']
rows=[]
for n in range(1,103):
 report=SOURCE/str(n)/'removal_report.json';d=json.loads(report.read_text());path=Path(d['output'])
 before=backup/f'{n}.usda'
 if not before.exists():shutil.copyfile(path,before)
 original=Sdf.Layer.OpenAsAnonymous(d['source'],metadataOnly=True);layer=Sdf.Layer.FindOrOpen(str(path));metadata={}
 for key in keys:
  if original.pseudoRoot.HasInfo(key):
   value=original.pseudoRoot.GetInfo(key);layer.pseudoRoot.SetInfo(key,value);metadata[key]=str(value)
 layer.Save();fresh=Sdf.Layer.OpenAsAnonymous(str(path),metadataOnly=True)
 for key in metadata:assert fresh.pseudoRoot.GetInfo(key)==original.pseudoRoot.GetInfo(key)
 d.update(output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stage_metadata_preserved=True,stage_metadata=metadata)
 report.write_text(json.dumps(d,ensure_ascii=False,indent=2));rows.append(d)
 for folder in [OUT/'render_probe',*sorted(OUT.glob('render_batch_*'))]:
  rpath=folder/f'scene-{n}.json'
  if rpath.exists():
   r=json.loads(rpath.read_text());assert hashlib.sha256(before.read_bytes()).hexdigest()==r['source_sha256']
   r.update(source_snapshot=str(before),current_saved_source_sha256=d['output_sha256'],post_render_metadata_patch='Original timing/up-axis/unit metadata preserved; no geometry, material, active state, camera or default-time pose changed.')
   rpath.write_text(json.dumps(r,ensure_ascii=False,indent=2))
(SOURCE/'manifest.json').write_text(json.dumps({'variant':'TabletopCleared','scene_count':len(rows),'removed_items':sum(d['removed_item_count'] for d in rows),'scenes':rows},ensure_ascii=False,indent=2))
(OUT/'metadata_audit.json').write_text(json.dumps({'verified':len(rows),'all_passed':True,'keys_checked':keys,'time_codes_per_second_values':sorted(set(d['stage_metadata'].get('timeCodesPerSecond') for d in rows))},indent=2))
print('STAGE_METADATA_VERIFIED',len(rows))
