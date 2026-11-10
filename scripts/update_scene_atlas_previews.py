#!/usr/bin/env python3
"""Install verified renders into local static atlas; back up the old index/images."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/scene_atlas_materials_v1'
SITE = ROOT/'scene_atlas/dist'
BACKUP = OUT/'original_atlas'
if not BACKUP.exists():
    BACKUP.mkdir(parents=True)
    shutil.copytree(SITE/'assets', BACKUP/'assets')
    for name in ['scenes.json','data.js']:
        shutil.copyfile(SITE/name, BACKUP/name)
scenes=json.loads((BACKUP/'scenes.json').read_text())
evidence={}
for folder in [OUT/'retry_local_mdl_probe',*sorted(OUT.glob('batch_*'))]:
    if not folder.is_dir():continue
    for p in folder.glob('scene-*.json'):
        record=json.loads(p.read_text())
        image=Path(record.get('image',''))
        if record.get('error') or not image.is_file():continue
        if hashlib.sha256(image.read_bytes()).hexdigest()!=record['image_sha256']:
            raise RuntimeError(f'Image hash mismatch: {image}')
        evidence[record['id']]=record
for s in scenes:
    if s['id'] not in evidence:continue
    r=evidence[s['id']]
    name=f"assets/scene-{s['id']}-mdl-v1.png"
    shutil.copyfile(r['image'],SITE/name)
    s.update(original_thumbnail=s['thumbnail'],thumbnail=name,preview_kind='mdl_render',preview_file=Path(r['source']).name,
        preview_source=r['source'],render_resolution=r['resolution'],
        render_unresolved_assets=len(r['unresolved_assets']),
        render_notes='MDL材质路径已加载；静态RTX渲染；墙体剖视隐藏；额外环境补光；未执行物理仿真。')
(SITE/'scenes.json').write_text(json.dumps(scenes,ensure_ascii=False))
(SITE/'data.js').write_text('window.SCENES = '+json.dumps(scenes,ensure_ascii=False)+';\n')
report={'total':len(scenes),'rendered':len(evidence),'pending':[s['id'] for s in scenes if s['id'] not in evidence],
    'rendered_with_unresolved_assets':[n for n,r in evidence.items() if r['unresolved_assets']],
    'material_path':str(ROOT.parent/'InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes/Materials'),
    'originals_backup':str(BACKUP)}
(OUT/'delivery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
