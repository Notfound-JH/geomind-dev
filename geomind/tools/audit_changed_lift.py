"""Offline state/action audit of physical change detection followed by grasp/lift."""
import json
from pathlib import Path
import numpy as np
from geomind.episodes import EpisodeResolver
from geomind.episodes.resolver import _check
from geomind.tools.migrate_episode_configs import write


def audit(root, output):
    report=json.loads((output/'report.json').read_text()); results=[]
    for c in report['results']:
        p=c['physical']; path=Path(c['trace']); tr=np.load(path)['data']
        objects=np.load(path.with_name(path.stem+'_objects.npz'));ids=objects['object_ids'].tolist();data=objects['data'];s=data[:,2:].reshape(len(data),len(ids),13)
        ctrl=np.load(c['control_trace']);u=ctrl['data'];n=len(ctrl['joint_names']);lim=ctrl['joint_limits']
        _check(len(tr)==len(data)==len(u) and np.isfinite(s).all() and np.isfinite(u).all(),'Invalid trace')
        _check(np.all(np.diff(data[:,0])==1) and np.array_equal(data[:,0],tr[:,0]),'Clock mismatch')
        _check(np.all(u[:,:n]>=lim[:,0]-1e-5) and np.all(u[:,:n]<=lim[:,1]+1e-5),'Control limit')
        _check(np.all(u[:,n:2*n]>=lim[:,0]-.005) and np.all(u[:,n:2*n]<=lim[:,1]+.005),'Measured joint limit')
        ep=EpisodeResolver(root).resolve(c['episode_id']);origin=np.array(ep.scene['physics']['frame_origin_w_m'])
        evidence=json.loads((output/'evidence'/f'{path.stem}.json').read_text());query=int(evidence['physical_clock']['current_tick'])-1
        reference=next(iter(evidence['history'].values()))['positions_m'];positions=evidence['current']['positions_m']
        shifts={key:float(np.linalg.norm(np.array(positions[key])-reference[key])) for key in ids}
        winners=[key for key,d in shifts.items() if d>ep.episode['parameters']['change_threshold_m']]
        _check(len(winners)==1 and winners[0]==c['ground_truth']['object_id'],'GT displacement mismatch')
        i=ids.index(winners[0]);_check(np.allclose(s[query,i,:3]-origin,positions[winners[0]],atol=2e-7),'Query mismatch')
        for j,key in enumerate(ids):
            _check(np.allclose(s[:,j,:3],tr[:,13+3*j:16+3*j],atol=2e-7),'Trace mismatch')
        history_ticks=evidence['physical_clock']['history_ticks']
        for key,tick in history_ticks.items():
            for j,identity in enumerate(ids):
                _check(np.allclose(s[int(tick)-1,j,:3]-origin,evidence['history'][key]['positions_m'][identity],atol=2e-7),'History mismatch')
        after=data[:,0]>p['hold_start_tick'];a=s[after];heights=a[:,i,2]-s[query,i,2];speeds=np.linalg.norm(a[:,i,7:10],axis=1)
        refs=[j for j in range(len(ids)) if j!=i];drift=np.linalg.norm(a[:,refs,:3]-s[query,refs,:3],axis=2).max(axis=1)
        fingers=[j for j,name in enumerate(ctrl['joint_names'].tolist()) if 'finger_joint' in name];closed=np.all(u[after,n:2*n][:,fingers]<.025,axis=1)
        good=(heights>.06)&(speeds<.01)&(drift<=.005)&closed;dwell=0
        for ok in good[::-1]:
            if not ok:break
            dwell+=1
        selected=ids.index(p['selected_object_id']);lift_mask=data[:,1]==6
        selected_lift=float(s[lift_mask,selected,2][-1]-s[query,selected,2]) if lift_mask.any() else 0.
        success=bool(p['selected_object_id']==winners[0] and selected_lift>.06 and dwell>=60 and all(x['reached'] for x in p['stages']))
        _check(success==p['success'] and dwell==p['dwell_steps'],'Physical outcome mismatch')
        _check(abs(float(heights[-1])-p['lift_height_m'])<2e-6,'Lift disagreement')
        obs=json.loads((output/'observations'/f'{path.stem}.json').read_text())['observations']
        _check(all(x['objects']==[] for x in obs if not x['scene_visible']),'Hidden state leaked')
        results.append({'episode_id':c['episode_id'],'negative_control':c['negative_control'],'success':success,'dwell':dwell,'lift_height_m':float(heights[-1]),'displacements_m':shifts,'rows':len(tr),'joint_limits_checked':True,'history_query_matched':True})
    result={'passed':bool(results),'mode':'offline recomputation; not physical re-execution','cases':results}
    write(output/'changed_lift_audit.json',result);return result
