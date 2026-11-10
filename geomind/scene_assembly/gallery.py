"""Update the existing local atlas with actual assembly evidence; no uploads."""
from geomind_lab.utils.portable_paths import scene_atlas_root
from pathlib import Path
import hashlib
import json
import shutil
import shlex

from .assets import ROOT


def publish_local(output):
    output=Path(output).resolve()
    site=scene_atlas_root()/'dist'
    if not site.is_dir():return None
    manifest=json.loads((output/'manifest.json').read_text());request=manifest['request']
    identity=output.name+'-'+hashlib.sha256(str(output).encode()).hexdigest()[:8]
    target=site/'assets/assemblies'/identity;target.mkdir(parents=True,exist_ok=True)
    urls={}
    for key in ['preview','detail_preview','video']:
        path=manifest.get(key) or manifest.get('physics',{}).get(key)
        if path and Path(path).is_file():
            filename=Path(path).name;shutil.copyfile(path,target/filename)
            urls[key]='assets/assemblies/'+identity+'/'+filename
    shutil.copyfile(output/'manifest.json',target/'manifest.json')
    command=shlex.join(['python','-m','geomind.tools.assemble_scene','--request',str(output/'request.json'),
        '--output',str(output.with_name(output.name+'_replay'))]+(['--verify-physics'] if manifest['physics'].get('executed') else []))
    layout=request['layout'];shape=(f"{layout['rows']}x{layout['cols']} grid" if layout['type']=='grid' else layout['type'])
    instruction=manifest.get('instruction') or f"Place {request['object']['count']} {request['object']['asset']} in {shape} in scene {request['scene_id']}."
    status={'assembled_physics_passed':'初始布局稳定性通过','assembled_physics_failed':'初始布局稳定性未通过',
        'assembled_physics_unverified':'初始布局已生成，物理未执行','assembled_audit_failed':'离线复核未通过'}.get(manifest['status'],manifest['status'])
    physics=dict(manifest.get('physics',{}))
    physics['notes']='实际 PhysX 检查的是初始布局落稳、数量和位置关系；本次没有机器人抓放动作。' if physics.get('executed') else '初始场景已另存，物理稳定性检查未执行。'
    record={'id':identity,'instruction':instruction,'scene_id':request['scene_id'],'scene_name':manifest['scene'].get('scene_name'),
        'asset':request['object']['asset'],'count':request['object']['count'],'layout':layout,
        'seed':request.get('seed',0),'randomization':request.get('randomization',{}),
        'layout_randomization':manifest.get('layout_randomization',{}),
        'physics':physics,'usda_path':manifest['scene_usda'],'manifest_path':str(output/'manifest.json'),
        'manifest_url':'assets/assemblies/'+identity+'/manifest.json','command':command,'status':status,**urls}
    index=site/'assets/assemblies.json'
    rows=json.loads(index.read_text()).get('assemblies',[]) if index.exists() else []
    rows=[r for r in rows if r.get('id')!=identity];rows.insert(0,record)
    def order(row):
        randomized=(row.get('randomization',{}).get('enabled') is True
                    or row.get('layout_randomization',{}).get('enabled') is True)
        return (row.get('physics',{}).get('passed') is not True,not randomized)
    rows.sort(key=order)
    payload={'schema_version':1,'assemblies':rows}
    temporary=index.with_suffix('.json.tmp');temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2));temporary.replace(index)
    return 'http://127.0.0.1:8780/assembly.html'
