#!/usr/bin/env python3
"""Bounded serial render jobs; retain all failed output and worker logs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/scene_atlas_materials_v1'
PYTHON = '/home/elm/miniconda3/envs/new_isaaclab/bin/python'
verified = OUT/'retry_local_mdl_probe'
existing = {int(p.stem.split('-')[-1]) for p in verified.glob('scene-*.json')}
ids = [n for n in range(1, 103) if n not in existing]
status=[]
for i in range(0,len(ids),12):
    group=ids[i:i+12]
    name=f'batch_{i//12+1:02d}'
    out=OUT/name
    if out.exists():
        raise RuntimeError(f'Output already exists: {out}')
    out.mkdir(parents=True)
    command=[PYTHON,str(ROOT/'scripts/render_scene_atlas.py'),'--ids',','.join(map(str,group)),'--output',str(out),'--resolution','768']
    print(f'BATCH_START {name}: {group}',flush=True)
    start=time.monotonic()
    with (out/'worker.log').open('w') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=420)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code='timeout'
    good=[n for n in group if (out/f'scene-{n}.json').is_file()]
    status.append({'batch':name,'scene_ids':group,'completed':good,'worker_exit_code':code,'elapsed_s':round(time.monotonic()-start,1)})
    (OUT/'batch_status.json').write_text(json.dumps(status,indent=2))
    print(f'BATCH_DONE {name}: {len(good)}/{len(group)} exit={code}',flush=True)
print('BATCH_ALL_DONE',flush=True)
