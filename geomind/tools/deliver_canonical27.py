"""Build a 27-type evidence gallery from accepted old and explicit new runs.

This command reads original evidence and verifies its hashes. It does not run
physics or rerun historical solvers under today's changed source tree. A newer
explicit retry directory wins even when it contains a failure or partial run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path

from geomind.core.registry import GeoMindRegistry
from geomind.core.taxonomy import CANONICAL_TASK_FAMILIES, FAMILY_LABELS
from geomind.episodes.canonical_extensions import task_ids as new_task_ids
from geomind.episodes.registered_tasks import REGISTERED_TASK_IDS
from geomind.tools.deliver_registered_tasks import saved_integrity
from geomind.tools.audit_negative_pick import audit_negative_picks


ROOT=Path(__file__).resolve().parents[2]
LABELS={
    'SelectByOrder':'按顺序选择','SelectByGridIndex':'按网格索引选择',
    'SelectByBranchPosition':'按分支位置选择','SelectByRegion':'按区域选择',
    'SelectByDirection':'按方向选择','SelectByDistance':'按距离选择',
    'SelectByAdjacency':'按邻接关系选择','SelectByPathDistance':'按路径距离选择',
    'PlaceByTopology':'按拓扑关系放置','SelectWithinSubset':'子集内选择',
    'SelectWithVisualCue':'结合视觉线索选择','SelectAfterExclusion':'排除后选择',
    'SelectByTwoRelations':'双关系选择','SelectByIndirectReference':'间接指代选择',
    'InferMidpoint':'中点推断','InferPointSymmetry':'中心对称推断',
    'ReflectAcrossLine':'轴对称','ExtendEqualSpacing':'等间距延伸',
    'ProjectOntoLine':'直线投影','CompleteSquare':'补全正方形',
    'ContinueSpatialPattern':'空间模式续接','DetectSceneChange':'检测场景变化',
    'UpdateObjectOrder':'更新物体顺序','UpdateEmptySlot':'更新空位',
    'TrackMovingObject':'追踪移动物体','TrackObjectSwap':'追踪物体交换',
    'TrackObjectOrigin':'追溯物体来源'}


def read(path, default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def esc(value):
    return html.escape(str(value),quote=True)


def select_new_versions(batch,retries=()):
    roots=[Path(batch).resolve(),*(Path(p).resolve() for p in retries)]
    if len(roots)!=len(set(roots)) or not all(p.is_dir() for p in roots):
        raise ValueError('New batch/retry roots must be existing, distinct and explicitly ordered')
    selected=[]
    for task in new_task_ids():
        versions=[{'root':str(root),'case':str(root/task),'order':order}
                  for order,root in enumerate(roots) if (root/task).is_dir()]
        selected.append({'task_id':task,'versions':versions,'selected':versions[-1] if versions else None})
    return roots,selected


def verify_raw_hashes(case, hashes):
    """Verify only hash-bound evidence inside this selected case directory."""
    case=Path(case).resolve();bad=[]
    if not isinstance(hashes,dict) or not hashes:
        return {'passed':False,'mismatches':['missing_raw_hash_manifest'],'files':0}
    for name,expected in hashes.items():
        path=Path(name)
        path=(path if path.is_absolute() else case/path).resolve()
        if not path.is_relative_to(case) or not path.is_file() or digest(path)!=expected:
            bad.append(name)
    return {'passed':not bad,'mismatches':bad,'files':len(hashes)}


def negative_pick_stages_complete(row):
    physical=row.get('physical',{})
    if physical.get('action')!='PickLiftSelectedObject':
        return physical.get('action')=='PickPlaceObject'
    stages=physical.get('stages',[]);by_stage={entry.get('stage'):entry for entry in stages}
    return bool(physical.get('grasp_verified') is True
        and physical.get('selected_object_id')!=physical.get('expected_object_id')
        and len(by_stage)==len(stages) and set(range(2,7))<=set(by_stage)
        and all(by_stage[label].get('reached') is True for label in range(2,7)))


def geometry_shortcut_row_quality(row,evidence):
    """Independently separate actual line goals from endpoint-midpoint shortcuts.

    Inputs are the recorded query state and frozen GT. No current task builder
    or production solver is used, so historical evidence remains unchanged.
    """
    try:
        calculation=row['provenance']['calculation'];name=calculation['resolver']
        if name not in {'ProjectOntoLine','ReflectAcrossLine'}:
            raise ValueError('Expected a projection or line-reflection calculation')
        params=row['provenance']['parameters'];contract=params['legacy_task_spec']['protocol']['gt_spec']
        current=evidence['current'];positions=current['positions_m'];mapping=params['object_id_map']
        if current['instance_id']!=row['instance_id']:
            raise ValueError('Geometry query state belongs to another physical instance')
        def xy(value):
            if (not isinstance(value,(list,tuple)) or len(value)<2
                    or not all(type(v) in (int,float) and math.isfinite(v) for v in value[:2])):
                raise ValueError('Invalid finite measured geometry vector')
            return [float(v) for v in value[:2]]
        def measured(legacy):
            aliases=[alias for alias,identity in mapping.items() if identity==legacy]
            if len(aliases)!=1:raise ValueError('Geometry identity has no unique measured alias')
            return xy(positions[aliases[0]])
        line=contract['line_object_ids']
        if len(line)!=2 or len(set([*line,contract['point_object_id']]))!=3:
            raise ValueError('Geometry source and line endpoints must be distinct')
        a,b=[measured(identity) for identity in line];p=measured(contract['point_object_id'])
        q=xy(row['ground_truth']['position_m'])
        if row['ground_truth']['type']!='Point':raise ValueError('Expected frozen point GT')
        tolerance=row['physical']['position_tolerance_m']
        if type(tolerance) not in (int,float) or not math.isfinite(tolerance) or tolerance<=0:
            raise ValueError('Invalid physical position tolerance')
        d=[b[i]-a[i] for i in range(2)];length2=sum(value*value for value in d)
        if length2<=0:raise ValueError('Degenerate measured reference line')
        fraction=sum((p[i]-a[i])*d[i] for i in range(2))/length2
        foot=[a[i]+fraction*d[i] for i in range(2)]
        midpoint=[(a[i]+b[i])*.5 for i in range(2)]
        expected=foot if name=='ProjectOntoLine' else [2*foot[i]-p[i] for i in range(2)]
        shortcut=midpoint if name=='ProjectOntoLine' else [2*midpoint[i]-p[i] for i in range(2)]
        threshold=max(.020,2*tolerance)
        saved_endpoints=calculation['line_endpoints_xy_m']
        epsilon=1e-7
        checks={
            'saved_calculation_bound_to_measured_query':len(saved_endpoints)==2
                and all(math.dist(xy(saved),actual)<=epsilon for saved,actual in zip(saved_endpoints,(a,b)))
                and math.dist(xy(calculation['source_point_xy_m']),p)<=epsilon
                and calculation.get('planned_positions_used_as_observations') is False,
            'frozen_gt_matches_independent_line_geometry':math.dist(q,expected)<=epsilon
                and math.dist(xy(row['physical']['frozen_ground_truth']['position_m']),q)<=epsilon,
            'saved_projection_matches_independent_line_geometry':math.dist(xy(calculation['projection_xy_m']),foot)<=epsilon
                and abs(float(calculation['line_parameter'])-fraction)*math.sqrt(length2)<=epsilon,
            'projection_distinct_from_endpoint_midpoint':math.dist(foot,midpoint)>threshold,
            'target_distinct_from_midpoint_shortcut':math.dist(q,shortcut)>threshold,
        }
        return {'passed':all(checks.values()),'resolver':name,'negative_control':bool(row.get('negative_control')),
            'checks':checks,'measured_line_parameter':fraction,
            'projection_midpoint_separation_m':math.dist(foot,midpoint),
            'target_midpoint_shortcut_separation_m':math.dist(q,shortcut),
            'required_shortcut_separation_m':threshold,'position_tolerance_m':tolerance,
            'independent_gt_residual_m':math.dist(q,expected),
            'failed_checks':[key for key,value in checks.items() if not value],
            'scope':'Measured query geometry; endpoint midpoint for projection and point symmetry about that midpoint for reflection'}
    except (KeyError,ValueError,TypeError,IndexError,OverflowError) as exc:
        return {'passed':False,'negative_control':bool(row.get('negative_control')),
                'failure':f'{type(exc).__name__}: {exc}'}


def geometry_shortcut_quality(report,task_id=None,*,source=None):
    expected=str(task_id or '').removesuffix('_v0')
    names={'ProjectOntoLine','ReflectAcrossLine'}
    rows=report.get('results',[])
    applicable=expected in names or any(row.get('provenance',{}).get('calculation',{}).get('resolver') in names for row in rows)
    if not applicable:return {'applicable':False,'passed':True,'cases':[]}
    cases=[]
    for row in rows:
        try:
            if source is None:raise ValueError('Missing original measured query evidence source')
            trace=Path(row['trace']).resolve();source=Path(source).resolve()
            if not trace.is_relative_to(source):raise ValueError('Geometry trace is outside selected case')
            path=source/'robot_physics/evidence'/(trace.stem+'.json')
            evidence=read(path)
            if evidence is None:raise ValueError('Missing original measured geometry query state')
            checked=geometry_shortcut_row_quality(row,evidence)
            if expected in names and checked.get('resolver')!=expected:checked['passed']=False
            cases.append(checked)
        except (KeyError,ValueError,TypeError,OSError,json.JSONDecodeError) as exc:
            cases.append({'passed':False,'negative_control':bool(row.get('negative_control')),
                          'failure':f'{type(exc).__name__}: {exc}'})
    return {'applicable':True,'passed':bool(cases) and sum(not row.get('negative_control') for row in rows)==1
            and all(case.get('passed') is True for case in cases),'cases':cases,
            'criterion':'Each measured projection differs from endpoint midpoint, and each target from its midpoint shortcut, by more than max(20mm, twice physical position tolerance)'}


def new_completion_gates(result,initial,initial_audit,report,audit,integrity,raw,wrist_audit=None,negative_pick_audit=None,geometry_quality_audit=None):
    rows=report.get('results',[])
    positives=[r for r in rows if not r.get('negative_control')]
    negatives=[r for r in rows if r.get('negative_control')]
    picks=[r for r in negatives if r.get('physical',{}).get('action')=='PickLiftSelectedObject']
    places=[r for r in negatives if r.get('physical',{}).get('action')=='PickPlaceObject']
    negative_pick_audit=negative_pick_audit or {}
    pick_commands_valid=(not picks or negative_pick_audit.get('passed') is True
                         and negative_pick_audit.get('applicable_negative_pick_cases')==len(picks))
    audited_negatives=[row for row in audit.get('cases',[]) if row.get('negative_control')]
    place_commands_valid=(not places or len(audited_negatives)==len(places)
        and all(row.get('command_execution_verified_recomputed') is True
                and (row.get('point_negative_control_valid') is True or row.get('topology_negative_control_valid') is True)
                for row in audited_negatives))
    return {
        'task_completed_recorded':result.get('task_completed') is True,
        'initial_physics_verified':result.get('initial_layout_physics_verified') is True and initial.get('passed') is True,
        'initial_independent_audit_passed':initial_audit.get('passed') is True,
        'task_action_physics_verified':result.get('task_action_physics_verified') is True,
        'runtime_gt_verified':result.get('gt_verified') is True and bool(positives)
            and all(r.get('independent_gt_check',{}).get('success') is True
                    and bool(r.get('physical',{}).get('frozen_ground_truth')) for r in positives),
        'worker_physical_execution_verified':report.get('physical_execution_verified') is True,
        'positive_actual_action_passed':len(positives)==1
            and all(r.get('success') is True and r.get('physical',{}).get('success') is True
                    and r.get('physical',{}).get('grasp_verified') is True for r in positives),
        'actual_negative_controls_correctly_rejected':bool(negatives)
            and all(r.get('success') is False and r.get('verification_passed') is True for r in negatives),
        'negative_command_stages_complete':bool(negatives) and all(negative_pick_stages_complete(r) for r in negatives),
        'negative_command_execution_valid':bool(negatives) and len(picks)+len(places)==len(negatives)
            and pick_commands_valid and place_commands_valid,
        'independent_raw_state_robot_audit_passed':audit.get('passed') is True,
        'audited_raw_bytes_unchanged':raw.get('passed') is True,
        'saved_source_snapshot_complete':integrity.get('saved_source_snapshot_complete') is True,
        'saved_config_hashes_complete':integrity.get('saved_config_hashes_complete') is True,
        'run_end_provenance_passed':result.get('provenance_check',{}).get('passed') is True,
        'actual_wrist_camera_audit_passed':(wrist_audit or {}).get('passed') is True,
        'geometry_task_quality_passed':(geometry_quality_audit if geometry_quality_audit is not None
                                        else geometry_shortcut_quality(report)).get('passed') is True,
    }


def _link_case(output,case,kind):
    case=Path(case).resolve()
    alias=Path('evidence')/kind/(case.parent.name+'_'+hashlib.sha256(str(case.parent).encode()).hexdigest()[:8])/case.name
    target=output/alias;target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():
        if target.resolve()!=case:raise ValueError('Refusing to change an existing evidence alias')
    elif target.exists():raise ValueError(f'Evidence destination is not a symlink: {target}')
    else:target.symlink_to(case,target_is_directory=True)
    return alias.as_posix()


def _url(source,alias,value):
    if not value:return None
    path=Path(value);path=(path if path.is_absolute() else source/path).resolve()
    if not path.is_relative_to(source) or not path.is_file():return None
    return alias+'/'+path.relative_to(source).as_posix()


def _media(source,alias,row):
    videos=[];video=row.get('video') or {}
    for name,view in [('外部视角',video),*video.get('wrist_cameras',{}).items()]:
        clip=_url(source,alias,view.get('observer'));poster=_url(source,alias,view.get('first_frame'))
        if clip:videos.append({'name':name,'src':clip,'agent_src':_url(source,alias,view.get('agent')),
                               'poster':poster,'frame_count':view.get('frame_count'),
                               'fps':view.get('fps'),'hashes':view.get('sha256',{})})
    return videos


def _numeric_metrics(row):
    p=row.get('physical',{});dt=row.get('query_physical_clock',{}).get('dt_s')
    return {key:p.get(key) for key in ('position_error_m','lift_height_m','reference_max_displacement_m',
            'expected_lift_height_m','support_height_error_m','motion_steps','dwell_steps','required_dwell_steps','termination_reason')} | {
            'stable_duration_s':p['dwell_steps']*dt if dt and 'dwell_steps' in p else None,
            'physics_dt_s':dt,'action':p.get('action'),'success':row.get('success')}


def _media_integrity(source,report,*,require_wrist):
    problems=[];streams=0
    rows=report.get('results',[])
    if not rows:problems.append('missing_recorded_cases')
    for row in rows:
        video=row.get('video') or {};wrists=video.get('wrist_cameras',{})
        if require_wrist and not wrists:problems.append('missing_actual_wrist_stream')
        for name,view in [('external',video),*wrists.items()]:
            streams+=1
            for key in ('observer','agent','frames'):
                value=view.get(key);expected=view.get('sha256',{}).get(key)
                path=Path(value).resolve() if value else None
                if (not expected or path is None or not path.is_relative_to(source)
                        or not path.is_file() or digest(path)!=expected):
                    problems.append(f'{row.get("episode_id")}/{name}/{key}')
    return {'passed':not problems,'streams':streams,'mismatches':problems}


def inspect_case(output,task,selection,*,accepted_old=None):
    if selection is None:
        return {'task_id':task,'kind':'new','passed':False,'status':'未执行','issues':['missing_case'],
                'videos':[],'versions':[],'instruction':'尚未生成本项实际指令。'}
    source=Path(selection['case']).resolve();kind='old' if accepted_old is not None else 'new'
    alias=_link_case(output,source,kind)
    result=read(source/'result.json',{}) or {};report=read(source/'robot_physics/report.json',{}) or {}
    manifest=read(source/'assembly/manifest.json',{}) or {};initial=read(source/'assembly/physics_report.json',{}) or {}
    audit=read(source/'robot_physics/assembled_robot_audit.json',{}) or {}
    wrist_audit=read(source/'robot_physics/wrist_camera_audit.json',{}) or {}
    initial_audit=read(source/'assembly/offline_audit.json',{}) or manifest.get('offline_audit',{})
    integrity=saved_integrity(source)
    media_integrity=_media_integrity(source,report,require_wrist=kind=='new')
    initial_raw=verify_raw_hashes(source,{'assembly/'+name:expected
        for name,expected in initial.get('sha256',{}).items()})
    if accepted_old is not None:
        historical_audit=accepted_old.get('new_robot_audit',{}).get('audit',{})
        raw=verify_raw_hashes(source,historical_audit.get('raw_sha256'))
        gates={**accepted_old.get('qualification',{}).get('gates',{}),
               'accepted_historical_delivery':accepted_old.get('task_completed') is True,
               'saved_source_snapshot_complete':integrity.get('saved_source_snapshot_complete') is True,
               'saved_config_hashes_complete':integrity.get('saved_config_hashes_complete') is True,
               'accepted_raw_bytes_unchanged':raw['passed']}
    else:
        raw=verify_raw_hashes(source,audit.get('raw_sha256'))
        negative_pick_audit=audit_negative_picks(source,report)
        geometry_quality_audit=geometry_shortcut_quality(report,task,source=source)
        gates=new_completion_gates(result,initial,initial_audit,report,audit,integrity,raw,wrist_audit,negative_pick_audit,geometry_quality_audit)
    gates['actual_recorded_media_intact']=media_integrity['passed']
    gates['initial_raw_evidence_intact']=initial_raw['passed']
    passed=bool(gates) and all(value is True for value in gates.values())
    rows=report.get('results',[])
    instruction=next((r.get('instruction') for r in rows if not r.get('negative_control') and r.get('instruction')),None)
    task_result=read(source/'task_result.json',{}) or {}
    instruction=instruction or task_result.get('instruction') or result.get('task',{}).get('instruction')
    plan=read(source/'registered_plan.json',{}) or {}
    instruction=instruction or plan.get('task_spec',{}).get('instruction') or '尚未生成本项实际指令。'
    videos=[{'negative_control':bool(row.get('negative_control')),'instruction':row.get('instruction'),
             'metrics':_numeric_metrics(row),'views':_media(source,alias,row),
             'success':row.get('success'),'verification_passed':row.get('verification_passed')}
            for row in rows]
    if kind=='new':
        for video,row in zip(videos,rows):
            if row.get('negative_control'):
                video['negative_command_execution_valid']=gates['negative_command_execution_valid']
                if row.get('physical',{}).get('action')=='PickLiftSelectedObject':
                    checked=next((item for item in negative_pick_audit['cases'] if item.get('episode_id')==row.get('episode_id')),None)
                    if checked:
                        video['selected_wrong_object_command_audit']=checked
                        video['metrics']['selected_wrong_object_lift_height_m']=checked.get('selected_body_lift_at_stage6_end_m')
                        video['metrics']['selected_wrong_object_hold_duration_s']=checked.get('trailing_hold_duration_s')
    links=[]
    for name,path in [('原始执行结果','result.json'),('实际物理报告','robot_physics/report.json'),
                      ('物理状态独立审计','robot_physics/assembled_robot_audit.json'),
                      ('腕相机随动与时钟审计','robot_physics/wrist_camera_audit.json'),
                      ('初始落稳审计','assembly/offline_audit.json'),('组装与资产','assembly/manifest.json'),
                      ('运行时查询GT','robot_physics/evidence/assembled_task.json'),
                      ('公开初始输入','observations/initial.json'),('源码与配置清单','provenance.json'),
                      ('原始请求','request.json'),('执行命令','robot_command.json')]:
        if (source/path).is_file():links.append({'label':name,'href':alias+'/'+path})
    settle=_url(source,alias,manifest.get('video') or initial.get('video'))
    walls=manifest.get('visualization',{}).get('walls')
    if walls is None:
        runtime=next((row for row in rows if not row.get('negative_control')),{}).get('scene_provenance',{})
        walls=runtime.get('visualization',{}).get('walls','未单独声明；保持原录像')
    return {'task_id':task,'kind':kind,'source':str(source),'alias':alias,'passed':passed,
            'status':'历史物理证据通过' if passed and kind=='old' else '本次物理验证通过' if passed else
                '几何质量未通过，保留原始动作证据' if gates.get('geometry_task_quality_passed') is False else result.get('status','未完成'),
            'issues':[key for key,value in gates.items() if value is not True], 'gates':gates,
            'instruction':instruction,'videos':videos,'initial_settling_video':settle,'links':links,
            'scene_id':result.get('scene_id',manifest.get('scene_id',65)),
            'seed':result.get('seed',manifest.get('request',{}).get('seed')),
            'layout':plan.get('registration',{}).get('layout',manifest.get('request',{}).get('layout')),
            'assets':sorted({obj.get('asset_name',obj.get('kind','object')) for obj in manifest.get('objects',[])}),
            'wall_visibility':walls,'integrity':integrity,'raw_integrity':raw,
            'media_integrity':media_integrity,'initial_raw_integrity':initial_raw,
            'negative_command_execution_valid':gates.get('negative_command_execution_valid') if kind=='new' else None,
            'negative_pick_command_audit':negative_pick_audit if kind=='new' else {
                'scope':'Historical accepted qualification retained; the new selected-body command audit was not applied to these old20 instances'},
            'geometry_task_quality':geometry_quality_audit if kind=='new' else {'applicable':False,'scope':'Historical accepted qualification retained'},
            'failure':result.get('failure'),'source_snapshot_id':integrity.get('source_snapshot_id'),
            'configuration_id':integrity.get('configuration_id'),
            'original_source_version_reused':kind=='old','current_code_physics_reexecuted':kind=='new'}


def _video_html(video):
    poster=f' poster="{esc(video["poster"])}"' if video.get('poster') else ''
    agent=f' · <a href="{esc(video["agent_src"])}">公开输入遮挡版</a>' if video.get('agent_src') else ''
    return f'<figure><video controls muted playsinline preload="metadata" src="{esc(video["src"])}"{poster}></video><figcaption>{esc(video["name"])} · 研究者完整视图 · <a href="{esc(video["src"])}" download>下载原片</a>{agent}</figcaption></figure>'


def _metrics_html(metrics):
    values=[]
    wrong='selected_wrong_object_lift_height_m' in metrics
    for key,label,scale,unit in [('position_error_m','位置误差',1000,'mm'),('lift_height_m','报告中的选中物体抬升' if wrong else '抬升',1000,'mm'),
                                ('support_height_error_m','支撑高度误差',1000,'mm'),('stable_duration_s','语义成功驻留' if wrong else '稳定',1,'s')]:
        if isinstance(metrics.get(key),(int,float)):
            values.append(f'{label} {metrics[key]*scale:.3f} {unit}')
    for key,label,scale,unit in [('selected_wrong_object_lift_height_m','实测错误物体抬升',1000,'mm'),
                                ('selected_wrong_object_hold_duration_s','实测错误抓取持稳',1,'s')]:
        if isinstance(metrics.get(key),(int,float)):values.append(f'{label} {metrics[key]*scale:.3f} {unit}')
    if wrong and isinstance(metrics.get('expected_lift_height_m'),(int,float)):
        values.append(f'语义目标物体最终抬升 {metrics["expected_lift_height_m"]*1000:.3f} mm')
    return ' · '.join(values)


def _instance_html(row):
    media=[]
    for case in row['videos']:
        title='错误目标／落点对照' if case['negative_control'] else '实际任务动作'
        rejection=' · 评分正确拒绝' if case['negative_control'] and case.get('verification_passed') and not case.get('success') else ''
        if case['negative_control'] and row['kind']=='new':
            rejection+=(' · 错误命令实际执行通过' if case.get('negative_command_execution_valid') else
                        ' · 错误命令未实际完成，不能计为有效对照')
        content=f'<h4>{title}{rejection}</h4><p class="caption">{esc(_metrics_html(case["metrics"]))}</p><div class="views">'+''.join(_video_html(v) for v in case['views'])+'</div>'
        media.append('<details class="negative"><summary>查看错误对照录像与指标</summary>'+content+'</details>' if case['negative_control'] else content)
    links=' · '.join(f'<a href="{esc(link["href"])}">{esc(link["label"])}</a>' for link in row.get('links',[]))
    versions=''.join(f'<li><a href="{esc(v["href"])}">{esc(v["root_name"])}</a>：{esc(v.get("status","未完成"))}</li>' for v in row.get('versions',[]))
    initial=''
    if row.get('initial_settling_video'):
        initial='<details><summary>初始场景落稳录像（独立于机器人任务动作）</summary><video controls muted preload="none" src="'+esc(row['initial_settling_video'])+'"></video></details>'
    errors='<p class="failure">未通过：'+esc('、'.join(row['issues']))+'</p>' if row['issues'] else ''
    if row.get('failure'):errors+='<p class="failure">'+esc(row['failure'])+'</p>'
    geometry=''
    if row.get('geometry_task_quality',{}).get('applicable'):
        geometry='<details><summary>几何任务质量：与中点捷径的实测间距</summary><pre>'+esc(json.dumps(row['geometry_task_quality'],ensure_ascii=False,indent=2))+'</pre></details>'
    return (f'<details class="instance" open><summary><b>{esc(row["task_id"])}</b> <span class="badge {"pass" if row["passed"] else "pending"}">{esc(row["status"])}</span></summary>'
            f'<p class="instruction">{esc(row["instruction"])}</p>'
            f'<p class="caption">{"历史保存版本，未在本次重执行" if row["kind"]=="old" else "本次新增实例"} · 场景 {esc(row.get("scene_id","—"))} · seed {esc(row.get("seed","—"))} · 布局 {esc(row.get("layout","—"))} · 墙体 {esc(row.get("wall_visibility","—"))}</p>'
            f'<p class="caption">资产：{esc(" / ".join(row.get("assets",[])) or "尚未生成")}</p>{errors}{geometry}{"".join(media) or "<p>尚无实际任务录像。</p>"}{initial}'
            '<details><summary>证据、版本和验收门槛</summary>'
            f'<p>{links}</p><p class="caption">源码快照 {esc(row.get("source_snapshot_id","—"))}<br>配置 {esc(row.get("configuration_id","—"))}</p>'
            '<p class="caption">当前源码可能已演进；历史通过依据保存的运行源码、配置、原始状态与原结束时来源检查。</p>'
            f'<ul>{versions}</ul><pre>{esc(json.dumps(row.get("gates",{}),ensure_ascii=False,indent=2))}</pre></details></details>')


def build(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    old_path=Path(args.old_delivery).resolve();old=read(old_path)
    if not old or {row['task_id'] for row in old.get('cases',[])}!=set(REGISTERED_TASK_IDS):
        raise ValueError('Old accepted delivery must contain the exact historical 20 instances')
    roots,new=select_new_versions(args.new_batch,args.retry)
    registry=GeoMindRegistry();registry.load();instances=[]
    for entry in old['cases']:
        row=inspect_case(output,entry['task_id'],entry.get('selected'),accepted_old=entry)
        row['versions']=[]
        for version in entry.get('versions',[]):
            source=Path(version['case']);alias=_link_case(output,source,'history')
            row['versions'].append({'root_name':source.parent.name,'source':str(source),
                                    'href':alias+'/result.json','status':version.get('original_status')})
        instances.append(row)
    for entry in new:
        row=inspect_case(output,entry['task_id'],entry['selected']);row['versions']=[]
        for version in entry['versions']:
            source=Path(version['case']);alias=_link_case(output,source,'history')
            record=('/result.json' if (source/'result.json').is_file() else
                    '/request.json' if (source/'request.json').is_file() else '/')
            row['versions'].append({'root_name':source.parent.name,'source':str(source),
                                    'href':alias+record,
                                    'status':read(source/'result.json',{}).get('status','未完成')})
        instances.append(row)
    groups={name:[] for names in CANONICAL_TASK_FAMILIES.values() for name in names}
    for row in instances:
        public=registry.get(row['task_id']).public_task_id
        if public not in groups:raise ValueError(f'Unrecognized canonical type {public}')
        row['canonical_task_id']=public;groups[public].append(row)
    completed=sum(bool(rows) and all(row['passed'] for row in rows) for rows in groups.values())
    counts={'canonical_task_types':27,'completed_canonical_task_types':completed,
            'expected_instances':30,'selected_instances':sum(bool(row.get('source')) for row in instances),
            'physically_passed_instances':sum(row['passed'] for row in instances),
            'historical_instances':20,'new_instances':10,
            'retained_retry_versions':sum(max(0,len(row['versions'])-1) for row in instances if row['kind']=='new')}
    summary={'schema_version':1,'kind':'canonical_27_evidence_delivery','created_at_utc':datetime.now(timezone.utc).isoformat(),
             'counts':counts,'status':'complete' if completed==27 and counts['physically_passed_instances']==30 else 'incomplete',
             'old_delivery':str(old_path),'old_delivery_sha256':digest(old_path),
             'new_roots_in_selection_order':[str(root) for root in roots],
             'selection_policy':'Newest explicitly supplied new case directory wins, including incomplete and failed retries',
             'instances':instances,'canonical_groups':{name:[row['task_id'] for row in rows] for name,rows in groups.items()},
             'model_evaluated':False,'browser_interaction_verified':False,
             'input_modality':'StructuredStateWithSceneImage; the public agent video uses the recorded visibility mask',
             'scope':'Historical20 preserved at their saved versions; new10 actual native robot actions. No claim that all30 ran under one latest controller or all old clips have wrist views.'}
    write(output/'summary.json',summary)
    sections=[];nav=[]
    for family,names in CANONICAL_TASK_FAMILIES.items():
        nav.append(f'<a href="#{family}">{esc(FAMILY_LABELS[family])}</a>')
        cards=[]
        for name in names:
            rows=groups[name];passed=bool(rows) and all(row['passed'] for row in rows)
            cards.append(f'<article id="{name}" data-search="{esc(name+" "+LABELS[name])}"><div class="heading"><h3>{esc(LABELS[name])}<small>{name}</small></h3><span class="badge {"pass" if passed else "pending"}">{"物理验证通过" if passed else "未完成"}</span></div>'+''.join(_instance_html(row) for row in rows)+'</article>')
        sections.append(f'<section id="{family}"><h2>{esc(FAMILY_LABELS[family])} <small>{len(names)} 种任务</small></h2>'+''.join(cards)+'</section>')
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GEOMIND · 27 类任务证据</title><style>
:root{--bg:#f4f3ee;--paper:#fafbf7;--ink:#222d29;--muted:#727970;--line:#d9ddd3;--accent:#365b46}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.8 system-ui,sans-serif}a{color:var(--accent)}header{padding:22px 5vw;border-bottom:1px solid var(--line)}main{max-width:1450px;margin:auto;padding:28px 5vw 50px}h1{font-size:34px}h2{font-size:23px;margin:36px 0 12px}h3{font-size:21px;margin:0}small{font-size:12px;font-weight:400;color:var(--muted)}h3 small{display:block}.lead,.caption{color:var(--muted)}.summary{background:#e3ecd8;border:1px solid #c7d5b7;padding:18px 22px;margin:22px 0}nav{display:flex;flex-wrap:wrap;gap:14px}article{background:var(--paper);border:1px solid var(--line);padding:22px;border-radius:8px;margin:22px 0}.heading{display:flex;justify-content:space-between;gap:12px;align-items:center}.badge{font-size:11px;border-radius:20px;padding:4px 10px;white-space:nowrap}.pass{background:#dfebd4;color:#365b46}.pending{background:#eee2c6;color:#60451a}.instruction{font-size:16px;color:#222d29}.caption{font-size:12px}.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:16px}figure{margin:0}video{display:block;width:100%;background:#d8d8d1;max-height:640px}figcaption{font-size:12px;color:var(--muted)}details{margin-top:18px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}.failure{color:#a13b2c}.instance>summary{padding:4px 0}.negative{background:#efede5;padding:12px}input{font:inherit;padding:9px 12px;width:min(100%,500px);margin:20px 0;border:1px solid var(--line);background:white}footer{color:var(--muted);font-size:12px}@media(max-width:700px){h1{font-size:27px}.heading{align-items:flex-start}article{padding:16px}.views{grid-template-columns:minmax(0,1fr)}}
</style></head><body><header><a href="../registered_robotiq_v2/delivery.html">原 20 个实例交付页 ↗</a></header><main>'''
    page+=f'<h1>27 类机器人任务</h1><p class="lead">自然操作指令、实际场景、机器人动作与原始验收证据。</p><div class="summary"><b>{completed} / 27 种任务物理通过 · {counts["physically_passed_instances"]} / 30 个实例通过</b><br>原 20 个实例覆盖 17 种类型；本次新增 10 种。顺序选择的 First／Last／Second 和拓扑放置的 Inside／OnTop 在各自类型中保留独立实例。</div>'
    page+='<p class="caption">旧 20 项保留其原版本、场景可见性与实际录像，未在本次重执行，也不补造旧录像的腕部视角。新实例显示实际外部与腕部录像。默认播放器为研究者完整视图；各视角另有公开输入遮挡版链接，隐藏阶段不会作为 agent 可见视频。初始落稳、任务动作和错误对照分开记录。输入为结构化状态加场景图像；这里只验证 oracle 指令执行与有界任务实现，不代表视觉模型或一般任务能力。</p>'
    page+='<nav>'+''.join(nav)+'</nav><input id="search" aria-label="搜索任务" placeholder="搜索中文任务名或 Task ID">'+''.join(sections)
    page+='<footer><p><a href="summary.json">逐项结果、验收门槛与全部来源</a></p><p>通过只依据原始物理执行、独立状态审计与保存证据；新失败不会回退成旧成功。录像中的墙体状态按每例原始 manifest 展示。</p></footer></main><script>document.querySelector("#search").addEventListener("input",e=>{const q=e.target.value.trim().toLowerCase();document.querySelectorAll("article[data-search]").forEach(card=>card.hidden=!card.dataset.search.toLowerCase().includes(q));});</script></body></html>'
    (output/'index.html').write_text(page)
    if args.require_complete and summary['status']!='complete':
        raise ValueError(f'Incomplete physical delivery: {completed}/27 types, {counts["physically_passed_instances"]}/30 instances; partial page and failures retained')
    return {'status':summary['status'],'counts':counts,'webpage':str(output/'index.html')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-delivery',type=Path,default=ROOT/'outputs/research/registered_robotiq_v2/delivery.json')
    parser.add_argument('--new-batch',type=Path,required=True)
    parser.add_argument('--retry',type=Path,action='append',default=[],help='Explicitly ordered newer batch; newest case always wins')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/research/canonical27_delivery_v1')
    parser.add_argument('--require-complete',action='store_true')
    result=build(parser.parse_args());print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
