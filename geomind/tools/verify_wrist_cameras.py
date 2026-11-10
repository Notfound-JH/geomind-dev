"""Prepare, run and independently audit wrist-camera diagnostics for every native arm.

ReachAboveTarget cases validate mounts and rendered motion, not grasp capability.
All workers use the existing native backend and a bounded serial supervisor.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCES={
    'franka':('fr3_panda_target_v1/panda','ProjectOntoLine_v0','franka','left'),
    'fr3':('fr3_panda_target_matched_home_v2/fr3','ProjectOntoLine_v0','fr3','left'),
    'franka_robotiq':('panda_robotiq_target_v1/robotiq','ProjectOntoLine_v0','franka_robotiq','left'),
    'piper':('piper_target_v1','ProjectOntoLine_v0','piper','left'),
    'piper_dual_left':('piper_dual_edge_v3','ProjectOntoLine_v0','piper_dual','left'),
    'piper_dual_right':('piper_dual_edge_v3','ProjectOntoLine_s1_v0','piper_dual','right'),
}


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def ground_diagnostic_layout(config_root, episode):
    """Create a named tabletop camera variant; preserve the historical 3D case."""
    root=Path(config_root);episode_path=root/'episodes'/f'{episode}.json'
    entry=read(episode_path);original=read(root/'layouts'/f'{entry["layout_id"]}.json')
    scene=read(root/'scenes'/f'{entry["scene_id"]}.json')
    require(not any(obj.get('physical_asset') for obj in original['objects']),
            'Tabletop camera diagnostic expects primitive cuboids')
    half=scene['physics']['object_size_m'][2]/2
    layout=deepcopy(original);layout['layout_id']=original['layout_id']+'_camera_tabletop'
    before={obj['id']:list(obj['position_m']) for obj in layout['objects']}
    for obj in layout['objects']:obj['position_m'][2]=half
    require(layout['randomization']['position_jitter_m']==0,'Camera fixture must not jitter off the table')
    write(root/'layouts'/f'{layout["layout_id"]}.json',layout)
    entry['layout_id']=layout['layout_id'];write(episode_path,entry)
    mount_change=None
    if scene.get('robot')=='piper_dual':
        # The historical edge-mounted Cartesian path selects a wrist-limit
        # branch after lowering the target. The +10cm camera-fixture variant
        # was separately verified in both actual PhysX arms (v8).
        before=list(scene['physics']['robot_base_w_m'])
        after=list(before);after[0]+=.10
        scene['physics']['robot_base_w_m']=after
        write(root/'scenes'/f'{entry["scene_id"]}.json',scene)
        mount_change={'before_position_w_m':before,'after_position_w_m':after,
                      'translation_w_m':[.10,0.,0.],
                      'reason':'Validated tabletop camera fixture avoids the historical Cartesian wrist-limit branch',
                      'preserved':'Object XYZ, mirrored Y, height, home joints, tool orientation and execution criteria'}
    # Historical semantic evidence is not evidence for the modified geometry.
    # Resolve/solve again at runtime from the physically settled object states.
    return {'variant':'camera_tabletop','original_layout_id':original['layout_id'],
            'layout_id':layout['layout_id'],'original_positions_m':before,
            'positions_m':{obj['id']:obj['position_m'] for obj in layout['objects']},
            'robot_mount_change':mount_change,
            'preserved':'XY, objects, seed, embodiment and reach semantics',
            'changed':'All cube centres at half-height above the real table; GT must be recomputed',
            'helper_supports_allowed':False}


def prepare(output):
    from geomind.episodes import EpisodeResolver
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    require(not (output/'cases.json').exists(),'Refusing to overwrite the prepared case index')
    cases=[]
    for name,(source,episode,robot,arm) in SOURCES.items():
        case=output/name;require(not case.exists(),f'Refusing existing case {case}')
        source_root=ROOT/'outputs/research'/source/'configs'
        shutil.copytree(source_root,case/'configs')
        placement=ground_diagnostic_layout(case/'configs',episode)
        resolved=EpisodeResolver(case/'configs').resolve(episode)
        require(resolved.scene['scene_id']=='registered_scene_65','Diagnostic replaced registered scene 65')
        require(resolved.scene['physics']['scene_mode']=='RegisteredScene','Procedural fallback is forbidden')
        require(resolved.scene['robot']==robot,'Source embodiment mismatch')
        require(resolved.episode['physical_execution']['action']=='ReachAboveTarget','Expected camera-motion diagnostic')
        require(resolved.episode['physical_execution'].get('active_arm','left')==arm,'Source active arm mismatch')
        request={'case':name,'source_config_root':str(source_root),'config_root':str(case/'configs'),
                 'episode_id':episode,'robot':robot,'active_arm':arm,'scene_id':65,
                 'scope':'wrist-camera mounting and actual motion/rendering diagnostic; not grasp capability',
                 'camera_width':640,'camera_height':480,'video_fps':30,'timeout_s':420,
                 'placement_change':placement,
                 'source_config_sha256':{p.relative_to(source_root).as_posix():sha(p) for p in sorted(source_root.rglob('*.json'))}}
        if placement['robot_mount_change'] is not None:
            request['robot_mount_change']=placement['robot_mount_change']
        write(case/'request.json',request);cases.append(request)
    result={'cases':cases,'gpu_concurrency':1,'worker_timeout_s':420,'physics_executed':False,
            'model_evaluated':False,'scope':'all five native embodiments; six active-arm runs'}
    write(output/'cases.json',result)
    return result


def run_case(output,name):
    from geomind.tools.assemble_scene import _python_environment
    from geomind.tools.research_pairs import snapshot,check_provenance
    output=Path(output).resolve();case=output/name;request=read(case/'request.json')
    require(not (case/'physics').exists(),'Refusing to overwrite physical evidence')
    require(not (case/'provenance.json').exists(),'Refusing to overwrite launch provenance')
    python,environment=_python_environment()
    command=[python,'-m','geomind.tools.run_episode_physics','--root',request['config_root'],
        '--episode',request['episode_id'],'--output',str(case/'physics'),'--timeout-s','420','--video',
        '--video-width',str(request['camera_width']),'--video-height',str(request['camera_height']),
        '--video-fps',str(request['video_fps']),'--camera-eye','1.7','0','1.85','--camera-look-at','.4','0','1.12']
    extra=['geomind/tools/verify_wrist_cameras.py','geomind/tools/run_episode_physics.py',
           'geomind/tools/audit_wrist_scene.py',
           'geomind_lab/tasks/manager_based/manipulation/geomind/geomind_scene_cfg.py',
           'geomind_lab/utils/scene_utils.py',
           'tests/test_wrist_camera_audit.py','tests/test_episode_video.py']
    extra += [str(p.relative_to(ROOT)) for p in [ROOT/'tests/test_wrist_camera_profiles.py'] if p.is_file()]
    extra += [p.relative_to(ROOT).as_posix() for p in (ROOT/'geomind/scene_assembly').glob('*.py')]
    snapshot(case,command,index_filename='request.json',extra_paths=extra)
    write(case/'command.json',{'argv':command,'timeout_s':420,'gpu_concurrency':1})
    with (case/'worker.log').open('w') as log:
        worker=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=460)
    result={'case':name,'worker_returncode':worker.returncode,'provenance_check':check_provenance(case),
            'physical_execution_verified':False,'camera_audit_passed':False,'scene_audit_passed':False}
    if (case/'physics/report.json').exists():
        result['physical_execution_verified']=bool(read(case/'physics/report.json').get('physical_execution_verified'))
        try:
            result['camera_audit']=audit(case/'physics')
            result['camera_audit_passed']=result['camera_audit']['passed']
        except (ValueError,KeyError,OSError) as exc:
            result['camera_audit_failure']=f'{type(exc).__name__}: {exc}'
        try:
            from geomind.tools.audit_wrist_scene import audit_scene
            result['scene_audit']=audit_scene(case/'physics')
            result['scene_audit_passed']=result['scene_audit']['passed']
        except (ValueError,KeyError,OSError) as exc:
            result['scene_audit_failure']=f'{type(exc).__name__}: {exc}'
    result['passed']=bool(worker.returncode==0 and result['physical_execution_verified']
                          and result['camera_audit_passed'] and result['scene_audit_passed']
                          and result['provenance_check']['passed'])
    write(case/'result.json',result)
    return result


def _q(value):
    a=np.asarray(value,dtype=float)
    require(a.shape==(4,) and np.isfinite(a).all() and abs(np.linalg.norm(a)-1)<2e-4,'Invalid quaternion')
    return a/np.linalg.norm(a)


def _v(value):
    a=np.asarray(value,dtype=float);require(a.shape==(3,) and np.isfinite(a).all(),'Invalid position')
    return a


def _multiply(a,b):
    aw,ax,ay,az=a;bw,bx,by,bz=b
    return np.asarray([aw*bw-ax*bx-ay*by-az*bz,
        aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw])


def _rotate(q,v):
    # Independent Hamilton-product calculation; does not call runtime helpers.
    return _multiply(_multiply(q,np.r_[0.,v]),q*np.array([1.,-1.,-1.,-1.]))[1:]


def pose_residuals(pose):
    camera_position=_v(pose['position_w_m']);camera_orientation=_q(pose['quaternion_wxyz_ros'])
    parent_position=_v(pose['parent_position_w_m']);parent_orientation=_q(pose['parent_quaternion_wxyz'])
    mount_position=_v(pose['mount_position_m']);mount_orientation=_q(pose['mount_quaternion_wxyz_ros'])
    position=parent_position+_rotate(parent_orientation,mount_position)
    orientation=_multiply(parent_orientation,mount_orientation)
    return float(np.linalg.norm(position-camera_position)),float(2*np.arccos(np.clip(abs(np.dot(orientation,camera_orientation)),0.,1.)))


def audit_frame_poses(rows, *, active):
    require(len(rows)>1,'Need multiple actual rendered wrist frames')
    positions=[];quaternions=[];errors=[];local=[]
    for row in rows:
        pose=row['camera_pose'];errors.append(pose_residuals(pose))
        positions.append(_v(pose['position_w_m']));quaternions.append(_q(pose['quaternion_wxyz_ros']))
        local.append(np.r_[_v(pose['mount_position_m']),_q(pose['mount_quaternion_wxyz_ros'])])
    require(np.allclose(local,local[0],rtol=0,atol=1e-8),'Camera local mount changed during execution')
    max_position,max_angle=np.max(errors,axis=0)
    require(max_position<=2e-4 and max_angle<=2e-3,'Camera world pose does not follow parent/local mount')
    displacement=float(np.max(np.linalg.norm(np.asarray(positions)-positions[0],axis=1)))
    rotation=float(max(2*np.arccos(np.clip(abs(np.dot(q,quaternions[0])),0.,1.)) for q in quaternions))
    if active:require(displacement>.002 or rotation>.02,'Active wrist camera did not move with the arm')
    return {'frames':len(rows),'max_mount_position_residual_m':float(max_position),
            'max_mount_rotation_residual_rad':float(max_angle),'max_camera_displacement_m':displacement,
            'max_camera_rotation_rad':rotation,'active_wrist':active,'fixed_local_mount_verified':True}


def audit_dual_image_orientation(rows):
    """Check image roll from measured ROS poses, independently of mount profiles.

    Image-up is optical -Y. Compare it with gravity-up projected into the image
    plane; a plain world-Z check becomes ill-conditioned when looking straight
    down. Skip only those near-vertical frames and require an observable horizon
    somewhere in this inclined-tool diagnostic. This is a diagnostic viewing
    contract, not a restriction on every possible robot/tool orientation.
    """
    cosines=[];up_z=[];near_vertical=0
    for row in rows:
        q=_q(row['camera_pose']['quaternion_wxyz_ros'])
        forward=_rotate(q,[0.,0.,1.]);up=_rotate(q,[0.,-1.,0.])
        projected_up=np.array([0.,0.,1.])-forward[2]*forward
        norm=float(np.linalg.norm(projected_up));up_z.append(float(up[2]))
        if norm<.1:
            near_vertical+=1
            continue
        cosines.append(float(np.clip(np.dot(up,projected_up/norm),-1.,1.)))
    require(cosines,'Dual wrist image roll is unobservable in every frame')
    require(min(cosines)>=.5,'Dual wrist image is inverted or excessively rolled relative to gravity-up')
    return {'upright_image_verified':True,'observable_frames':len(cosines),
            'near_vertical_frames':near_vertical,'min_image_up_world_z':min(up_z),
            'min_projected_gravity_up_cosine':min(cosines),
            'max_roll_from_projected_gravity_up_deg':float(np.degrees(np.arccos(min(cosines))))}


def _media(video):
    import imageio.v2 as imageio
    for key in ('observer','agent','frames'):
        require(sha(video[key])==video['sha256'][key],f'Changed camera artifact {key}')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_frames,avg_frame_rate','-of','json',video['observer']]))['streams'][0]
    require(int(probe['nb_frames'])==video['frame_count'],'Video frame count disagrees with metadata')
    require([probe['width'],probe['height']]==[video['width'],video['height']],'Video dimensions disagree')
    result=subprocess.run(['ffmpeg','-v','error','-i',video['observer'],'-f','null','-'],capture_output=True,text=True)
    require(result.returncode==0 and not result.stderr.strip(),'Camera video failed complete decode')
    first=np.asarray(imageio.imread(video['first_frame']));last=np.asarray(imageio.imread(video['last_frame']))
    require(float(first.std())>1 and float(last.std())>1,'Camera returned blank images')
    return first,{'probe':probe,'full_observer_decode_passed':True,'first_rgb_std':float(first.std()),
                 'last_rgb_std':float(last.std()),'first_last_mean_pixel_difference':float(np.mean(np.abs(first.astype(float)-last.astype(float))))}


def audit(physics):
    from geomind.episodes.wrist_cameras import wrist_camera_specs
    physics=Path(physics).resolve();report=read(physics/'report.json');cases=[]
    require(report.get('results'),'No native execution results to audit')
    for case in report['results']:
        robot=case['robot_name'];active=case.get('active_arm','left')
        roots={'left':'/World/Robot'}
        if robot=='piper_dual':roots['right']='/World/RobotRight'
        expected={s['name']:s for s in wrist_camera_specs(robot,roots)}
        video=case['video'];wrists=video.get('wrist_cameras',{})
        require(set(wrists)==set(expected),'Missing, duplicate or extra wrist camera views')
        table_rows=[json.loads(x) for x in Path(video['frames']).read_text().splitlines()]
        table_first,table_media=_media(video)
        views={}
        for name,spec in expected.items():
            recording=wrists[name]
            require(recording['calibration']['mount']['parent_prim_path']==spec['parent_prim_path'],'Wrong parent arm/link')
            require(recording['calibration']['mount']['prim_path']==spec['prim_path'],'Camera is not mounted below expected link')
            rows=[json.loads(x) for x in Path(recording['frames']).read_text().splitlines()]
            require([r['physics_tick'] for r in rows]==[r['physics_tick'] for r in table_rows],'Wrist and external video clocks differ')
            require([r['scene_visible_to_agent'] for r in rows]==[r['scene_visible_to_agent'] for r in table_rows],'View visibility masks differ')
            require(all(np.allclose(r['camera_pose']['mount_position_m'],spec['position_m'],atol=1e-8,rtol=0) for r in rows),'Recorded mount offset differs from saved profile')
            require(all(abs(float(np.dot(_q(r['camera_pose']['mount_quaternion_wxyz_ros']),_q(spec['quaternion_wxyz_ros']))))>=1-1e-8 for r in rows),'Recorded mount orientation differs from saved profile')
            pose=audit_frame_poses(rows,active=spec['arm']==active)
            if robot=='piper_dual':
                pose['image_orientation']=audit_dual_image_orientation(rows)
            first,media=_media(recording)
            if first.shape==table_first.shape:
                difference=float(np.mean(np.abs(first.astype(float)-table_first.astype(float))))
                require(difference>1,'Wrist view duplicates the fixed observer image')
            else:difference=None
            pose.update(media=media,first_frame_observer_difference=difference,mount=deepcopy(spec))
            pose['hidden_frames']=sum(not row['scene_visible_to_agent'] for row in rows)
            pose['masking_scope']='All view visibility clocks agree; no hidden-frame pixel test when this static diagnostic has no hidden phase.'
            views[name]=pose
        cases.append({'episode_id':case['episode_id'],'robot':robot,'active_arm':active,
                      'negative_control':case['negative_control'],'views':views,'table_media':table_media})
    result={'passed':True,'cases':cases,'scope':'independent parent/local-to-camera pose recomposition, dynamic motion, rendered pixels, media hashes and clocks; no model or grasp-capability claim'}
    write(physics/'wrist_camera_audit.json',result)
    return result


def gallery(output):
    output=Path(output).resolve();index=read(output/'cases.json');cards=[];summary=[]
    for entry in index['cases']:
        name=entry['case'];case=output/name;result=read(case/'result.json') if (case/'result.json').exists() else {}
        report=read(case/'physics/report.json') if (case/'physics/report.json').exists() else {}
        status='已验证' if result.get('passed') else '未通过' if result else '待执行'
        clips=[]
        for row in report.get('results',[]):
            video=row.get('video') or {}
            for camera,view in {'外部视角':video,**video.get('wrist_cameras',{})}.items():
                if not view.get('observer'):continue
                path=Path(view['observer']).relative_to(output).as_posix();poster=Path(view['first_frame']).relative_to(output).as_posix()
                clips.append(f'<figure><video controls muted playsinline preload="metadata" poster="{html.escape(poster)}" src="{html.escape(path)}"></video><figcaption>{html.escape(camera)}</figcaption></figure>')
        links=f'<a href="{name}/request.json">配置来源</a>'
        if (case/'physics/wrist_camera_audit.json').exists():links+=f' · <a href="{name}/physics/wrist_camera_audit.json">挂载与录像核验</a>'
        cards.append(f'<article><h2>{html.escape(name)} <small>{status}</small></h2><p>场景 65 · 活动臂 {entry["active_arm"]} · 末端到达诊断</p><div class="views">{"".join(clips)}</div><p>{links}</p></article>')
        summary.append({'case':name,'passed':bool(result.get('passed')),'status':status})
    count=sum(r['passed'] for r in summary)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>机械臂手腕相机验证</title><style>body{{margin:0;background:#f4f3ee;color:#222d29;font:15px/1.7 system-ui,sans-serif}}main{{max-width:1400px;margin:auto;padding:36px 5vw}}a{{color:#365b46}}h1{{font-size:32px}}article{{background:#fafbf7;border:1px solid #d9ddd3;padding:22px;margin:26px 0;border-radius:8px}}small{{font-size:13px;color:#365b46}}.views{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:18px}}figure{{margin:0;min-width:0}}video{{width:100%;background:#ddd}}figcaption{{font-size:12px;color:#727970}}.note{{background:#e6eadd;padding:18px}}p{{color:#59655e}}</style><main><a href="../">GEOMIND 场景与任务</a><h1>所有机械臂的手腕相机</h1><p>5 种已接入配置，双臂各自挂载，6 个活动臂执行实例。</p><p class="note">当前 {count} / 6 个挂载验证通过。实际腕部视角随末端运动，和外部相机按相同物理时钟录制。这里运行的是末端到达诊断，用于检查挂载与录像，不代表新增抓放能力或视觉模型评测。</p>{''.join(cards)}<p><a href="summary.json">完整状态</a></p></main></html>'''
    (output/'index.html').write_text(page);write(output/'summary.json',{'passed':count,'requested':6,'cases':summary,'model_evaluated':False})
    return {'passed':count,'requested':6,'page':str(output/'index.html')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('prepare','run','gallery'):
        item=sub.add_parser(mode);item.add_argument('--output',type=Path,required=True)
        if mode=='run':item.add_argument('--cases',nargs='+',choices=list(SOURCES),default=list(SOURCES))
    item=sub.add_parser('audit');item.add_argument('--physics',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='prepare':result=prepare(args.output)
    elif args.mode=='audit':result=audit(args.physics)
    elif args.mode=='gallery':result=gallery(args.output)
    else:
        result=[run_case(args.output,name) for name in args.cases]
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if all(r['passed'] for r in result) else 1
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
