"""Render saved cleared scene variants serially, retaining all bounded worker logs."""
from pathlib import Path
import json,os,signal,subprocess,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/scene_atlas_cleared_v1';SOURCE=ROOT/'assets/scenes/tabletop_cleared_v1'
ids=[n for n in range(1,103) if n not in [17,65,100]];status=[]
for i in range(0,len(ids),12):
 group=ids[i:i+12];name=f'render_batch_{i//12+1:02d}';out=OUT/name
 if out.exists():raise RuntimeError(f'Output exists: {out}')
 deadline=time.monotonic()+1200
 while not all((SOURCE/str(n)/'removal_report.json').exists() for n in group):
  if time.monotonic()>deadline:raise RuntimeError(f'Saved files not ready: {group}')
  time.sleep(2)
 out.mkdir();print('BATCH_START',name,group,flush=True)
 command=['/home/elm/miniconda3/envs/new_isaaclab/bin/python',str(ROOT/'scripts/render_scene_atlas.py'),'--ids',','.join(map(str,group)),'--source-root',str(SOURCE),'--output',str(out),'--resolution','768']
 with (out/'worker.log').open('w') as log:
  p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=420)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   code='timeout'
 good=[n for n in group if (out/f'scene-{n}.json').is_file()]
 status.append({'batch':name,'scene_ids':group,'completed':good,'exit_code':code});(OUT/'render_status.json').write_text(json.dumps(status,indent=2))
 print('BATCH_DONE',name,len(good),'/',len(group),'exit',code,flush=True)
print('CLEARED_RENDER_ALL_DONE',flush=True)
