"""Independently recompute placement acceptance from measured PhysX exports."""
import argparse
import json
from pathlib import Path
import numpy as np
from geomind.episodes import EpisodeResolver
from geomind.episodes.resolver import _check
from geomind.tools.migrate_episode_configs import write


def audit(root, output):
    report=json.loads((output/'report.json').read_text()); results=[]
    for case in report['results']:
        _check(case['physical']['action']=='PickPlaceMidpoint','Not a placement trace')
        path=Path(case['trace']); data=np.load(path)['data']
        controls=np.load(case['control_trace']); u=controls['data']; n=len(controls['joint_names']);limits=controls['joint_limits']
        objects=np.load(path.with_name(path.stem+'_objects.npz')); states=objects['data']; ids=objects['object_ids'].tolist()
        _check(len(states)==len(data)==len(u),'Trace length mismatch')
        _check(np.isfinite(states).all() and np.isfinite(u).all(),'Nonfinite states')
        _check(np.all(np.diff(states[:,0])==1),'Physical clock gap')
        _check(np.allclose(states[:,0],data[:,0]),'Object clock mismatch')
        _check(np.all(u[:,:n]>=limits[:,0]-1e-5) and np.all(u[:,:n]<=limits[:,1]+1e-5),'Joint action outside limits')
        _check(np.all(u[:,n:2*n]>=limits[:,0]-.005) and np.all(u[:,n:2*n]<=limits[:,1]+.005),'Measured joints outside limits')
        ep=EpisodeResolver(root).resolve(case['episode_id']);origin=np.array(ep.scene['physics']['frame_origin_w_m'])
        evidence=json.loads((output/'evidence'/f'{path.stem}.json').read_text());query=int(case['query_physical_clock']['current_tick'])-1
        world=states[:,2:].reshape(len(states),len(ids),13);moving=ids.index(ep.episode['physical_execution']['moving_object'])
        refs=[ids.index(x) for x in ep.episode['parameters']['endpoints']]
        for i,key in enumerate(ids):
            _check(np.allclose(world[query,i,:3]-origin,evidence['current']['positions_m'][key],atol=2e-7),'Query/evidence mismatch')
            _check(np.allclose(world[:,i,:3],data[:,13+3*i:16+3*i],atol=2e-7),'Object state/trace mismatch')
        target=world[query,refs,:3].mean(axis=0)
        _check(np.allclose(target,case['physical']['target_object_w_m'],atol=2e-7),'Frozen GT mismatch')
        after=states[:,0]>case['physical']['release_tick'];s=world[after,moving];refmove=np.linalg.norm(world[after][:,refs,:3]-world[query,refs,:3],axis=2).max(axis=1)
        err=np.linalg.norm(s[:,:3]-target,axis=1);speed=np.linalg.norm(s[:,7:10],axis=1);angular=np.linalg.norm(s[:,10:13],axis=1)
        support=np.abs(s[:,2]-origin[2]-.015)
        finger_ids=[i for i,x in enumerate(controls['joint_names'].tolist()) if 'finger_joint' in x]
        opened=np.all(u[after,n:2*n][:,finger_ids]>.03,axis=1)
        valid=(err<=.01)&(speed<=.01)&(angular<=.1)&(support<=.004)&(refmove<=.005)&opened
        dwell=0
        for ok in valid[::-1]:
            if not ok:break
            dwell+=1
        lift_rows=states[:,1]==6
        lift=float(world[lift_rows,moving,2].max()-world[query,moving,2]) if lift_rows.any() else 0.
        reached=all(x['reached'] for x in case['physical']['stages'])
        success=bool(reached and lift>.06 and dwell>=60)
        _check(success==case['physical']['success'],'Recomputed placement outcome disagrees')
        _check(abs(err[-1]-case['physical']['position_error_m'])<2e-6,'Final object error disagrees')
        results.append({'episode_id':case['episode_id'],'negative_control':case['negative_control'],'success':success,'rows':len(data),'dwell':dwell,'lift_height_m':lift,'object_error_m':float(err[-1]),'reference_displacement_m':float(refmove[-1]),'support_error_m':float(support[-1]),'joints_checked':True})
    result={'passed':bool(results),'mode':'offline recomputation of measured physical execution','cases':results}
    write(output/'placement_audit.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(audit(args.run/'configs',args.run/'physics'),indent=2))
