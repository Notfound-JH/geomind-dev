"""Build the wrist-camera scene-correction review from explicit run directories.

The page only reads recorded reports; it does not execute physics, overwrite
historical results, or claim that offline checks are fresh physical execution.
"""
from pathlib import Path
import argparse,html,json

REPO=Path(__file__).resolve().parents[2]
NAMES={'franka':'Franka Panda','fr3':'Franka FR3','franka_robotiq':'Panda + Robotiq 2F-85',
       'piper':'Piper 单臂','piper_dual_left':'Piper 双臂 · 左臂运动','piper_dual_right':'Piper 双臂 · 右臂运动'}
CAMERAS={'observer':'外部视角','wrist':'手腕视角','wrist_left':'左手腕视角','wrist_right':'右手腕视角'}

def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def esc(value):return html.escape(str(value),quote=True)


def previous_case_summary(source):
    """Describe the saved checks without promoting camera passes to action passes."""
    result=read(source/'result.json')
    report=read(source/'physics/report.json') if (source/'physics/report.json').is_file() else {}
    failures=[]
    if report.get('failure'):failures.append(str(report['failure']))
    for row in report.get('results',[]):
        physical=row.get('physical',{})
        if physical.get('success') is False:
            fields=[str(physical.get('termination_reason','未通过'))]
            if physical.get('position_error_m') is not None:
                fields.append(f"位置误差 {physical['position_error_m']*1000:.2f} mm")
            if physical.get('orientation_error_rad') is not None:
                fields.append(f"朝向误差 {physical['orientation_error_rad']:.3f} rad")
            if physical.get('dwell_steps') is not None:
                fields.append(f"驻留 {physical['dwell_steps']} / {physical.get('required_dwell_steps','?')} 步")
            failures.append(f"{physical.get('action','动作')}："+'，'.join(fields))
    for key in ['camera_audit_failure','scene_audit_failure']:
        if result.get(key):failures.append(str(result[key]))
    checks={'physical_execution_verified':bool(result.get('physical_execution_verified')),
            'camera_audit_passed':bool(result.get('camera_audit_passed')),
            'scene_audit_passed':bool(result.get('scene_audit_passed')),
            'failures':failures}
    text=('旧版相机核验'+('通过' if checks['camera_audit_passed'] else '未通过')
          +'、物体支撑核验'+('通过' if checks['scene_audit_passed'] else '未通过')
          +'；实际动作'+('通过' if checks['physical_execution_verified'] else '未通过')+'。')
    if failures:text+='原始结果：'+'；'.join(failures)+'。'
    return checks,text


def robot_mount_note(source):
    request=read(source/'request.json') if (source/'request.json').is_file() else {}
    change=request.get('robot_mount_change')
    if not change:return '',None
    before=change['before_position_w_m'];after=change['after_position_w_m']
    shift=change['translation_w_m']
    text=(f"双臂安装配置修正：两侧基座相对原桌边安装沿世界 +X 前移 {shift[0]*100:.0f} cm"
          f"（基座 X：{before[0]:.2f} → {after[0]:.2f} m），Y 间距与安装高度保持。"
          '这是本次独立诊断配置的调整；落在桌面的物体 XYZ、查询语义、初始关节角、工具朝向、关节限位、时间步、接近路径规则及验收容差与预算均保留。')
    return '<p class="caption">'+esc(text)+'</p>',change


def scene_metrics(report, audit):
    positives=[row for row in report.get('results',[]) if not row.get('negative_control')]
    walls=[asset for row in positives for asset in row.get('scene_provenance',{}).get('assets',[])
           if asset.get('name','').startswith('source_wall_')]
    materials_ok=bool(walls) and all(wall.get('active') and wall.get('bound_material_path')
        and wall.get('surface_textures') and all(t.get('exists') for t in wall['surface_textures'])
        for wall in walls)
    objects=[{'episode_id':case['episode_id'],'object_id':identity,**state}
             for case in audit.get('cases',[]) if not case.get('negative_control')
             for identity,state in case.get('objects',{}).items()]
    geometry_ok=bool(walls) and all(wall.get('active') and wall.get('collision_enabled')
        and wall.get('geometry_type') in {'Mesh','Cube'} for wall in walls)
    return {'wall_count':len(walls),'wall_materials_verified':materials_ok,
            'wall_geometry_verified':geometry_ok,
            'source_walls':walls,'support_audit_passed':bool(audit.get('passed')),
            'objects':objects,
            'maximum_absolute_bottom_gap_m':max((max(abs(s['minimum_bottom_gap_m']),abs(s['maximum_bottom_gap_m'])) for s in objects),default=None),
            'maximum_final_linear_speed_m_s':max((s['final_window_maximum_linear_speed_m_s'] for s in objects),default=None),
            'helper_supports_disabled':bool(audit.get('cases')) and all(case.get('helper_supports_disabled') for case in audit.get('cases',[]))}


def correction_details(metrics, orientation):
    walls=metrics['source_walls']
    materials=sorted({Path(t['asset_path']).name for wall in walls for t in wall.get('surface_textures',[])
                      if '_diff_' in t.get('asset_path','')})
    wall_text=f"{metrics['wall_count']} 面源场景墙体 · 材质纹理{'完整' if metrics['wall_materials_verified'] else '未完整验证'}"
    support='尚无落稳支撑审计。'
    if metrics['objects']:
        support=(f"最大绝对底面高度差 <b>{metrics['maximum_absolute_bottom_gap_m']*1000:.3f} mm</b> · "
                 f"末尾 0.5 秒最大速度 <b>{metrics['maximum_final_linear_speed_m_s']*1000:.4f} mm/s</b> · "
                 f"辅助支撑{'已关闭' if metrics['helper_supports_disabled'] else '未确认关闭'}")
    rows=''.join(f"<tr><td>{esc(s['object_id'])}</td><td>{s['minimum_bottom_gap_m']*1000:.3f} ～ {s['maximum_bottom_gap_m']*1000:.3f} mm</td><td>{s['final_window_maximum_linear_speed_m_s']*1000:.4f} mm/s</td></tr>" for s in metrics['objects'])
    orientation_text=''
    if orientation:
        orientation_text='<p>'+ ' · '.join(f"{esc(CAMERAS.get(name,name))}：{'正向通过' if value.get('upright_image_verified') else '未验证'}，最大滚转 {value['max_roll_from_projected_gravity_up_deg']:.2f}°" for name,value in orientation.items())+'</p>'
    return (f'<p>{wall_text}'+(' · '+esc(' / '.join(materials)) if materials else '')+'</p>'
            +f'<p class="metrics">{support}</p>'+orientation_text
            +('<div class="scroll"><table><thead><tr><th>物体</th><th>动作期间底面与实测桌面高度差</th><th>最终稳定窗口最大速度</th></tr></thead><tbody>'+rows+'</tbody></table></div>' if rows else ''))

def link_directory(root,name,source):
    path=root/'evidence'/name;path.parent.mkdir(exist_ok=True)
    if path.is_symlink():
        if path.resolve()!=source.resolve():raise ValueError('Refusing to silently change an existing evidence source: '+str(path))
    elif path.exists():raise ValueError('Evidence alias already exists: '+str(path))
    else:path.symlink_to(source.resolve(),target_is_directory=True)
    return path

def media_url(path,source,alias):
    return f'evidence/{alias}/'+Path(path).resolve().relative_to(source.resolve()).as_posix()

def video_views(row,source,alias):
    video=row.get('video') or {};views=[]
    for name,item in [('observer',video),*video.get('wrist_cameras',{}).items()]:
        if not item.get('observer'):continue
        clip=media_url(item['observer'],source,alias);poster=media_url(item['first_frame'],source,alias)
        views.append(f'<figure><video controls muted playsinline preload="metadata" src="{esc(clip)}" poster="{esc(poster)}"></video><figcaption><b>{esc(CAMERAS.get(name,name))}</b><a href="{esc(clip)}" download>下载原片 ↓</a></figcaption></figure>')
    return '<div class="views">'+''.join(views)+'</div>'

def card(root,source,entry):
    alias=entry['case'];link_directory(root,alias,source)
    result=read(source/'result.json') if (source/'result.json').is_file() else {}
    report=read(source/'physics/report.json') if (source/'physics/report.json').is_file() else {}
    audit=read(source/'physics/wrist_camera_audit.json') if (source/'physics/wrist_camera_audit.json').is_file() else {}
    scene_audit=read(source/'physics/wrist_scene_audit.json') if (source/'physics/wrist_scene_audit.json').is_file() else {}
    scene=scene_metrics(report,scene_audit)
    passed=bool(result.get('passed') and result.get('scene_audit_passed') and audit.get('passed')
                and scene['support_audit_passed'] and scene['wall_count']==4
                and scene['wall_materials_verified'] and scene['wall_geometry_verified']
                and scene['helper_supports_disabled'])
    rows=[r for r in report.get('results',[]) if not r.get('negative_control')]
    mount_note,mount_change=robot_mount_note(source)
    videos=''.join(video_views(row,source,alias) for row in rows)
    pose=[]
    orientation={}
    for row in audit.get('cases',[]):
        for name,view in row.get('views',{}).items():
            if view.get('image_orientation'):orientation[name]=view['image_orientation']
            pose.append(f'<tr><td>{esc(CAMERAS.get(name,name))}</td><td>{view["max_mount_position_residual_m"]*1000:.4f} mm</td><td>{view["max_camera_displacement_m"]*100:.2f} cm</td><td>{esc(view["mount"]["parent_prim_path"])}</td></tr>')
    if alias.startswith('piper_dual'):
        passed=bool(passed and len(orientation)==2 and all(v.get('upright_image_verified') for v in orientation.values()))
    details=('''<details><summary>查看真实挂载与录像核验</summary><div class="scroll"><table><thead><tr><th>视角</th><th>挂载位置残差</th><th>相机运动范围</th><th>挂载位置</th></tr></thead><tbody>'''+''.join(pose)+'</tbody></table></div>' if pose else '<details><summary>查看执行记录</summary>')
    links=[f'<a href="evidence/{alias}/request.json">配置来源</a>']
    for label,path in [('执行结果','result.json'),('原始物理报告','physics/report.json'),('相机独立核验','physics/wrist_camera_audit.json'),('物体支撑与速度审计','physics/wrist_scene_audit.json'),('源码与配置清单','provenance.json')]:
        if (source/path).exists():links.append(f'<a href="evidence/{alias}/{path}">{label}</a>')
    text=f'''<article data-diagnostic="{esc(alias)}"><div class="heading"><h2>{esc(NAMES[alias])}</h2><span class="badge {'pass' if passed else 'pending'}">{'场景、支撑与相机验证通过' if passed else '未完成验证'}</span></div><p class="caption">实际场景 65 · 末端到达诊断 · {'右臂' if entry['active_arm']=='right' else '左臂'}运动</p>{mount_note}{videos or '<p>尚无本次完成的录像。</p>'}{correction_details(scene,orientation)}{details}<p class="links">{' '.join(links)}</p></details></article>'''
    return text,{'case':alias,'source':str(source),'passed':passed,'scene':scene,'dual_image_orientation':orientation,
                'robot_mount_change':mount_change,
                'wrist_streams':sum(len(r.get('video',{}).get('wrist_cameras',{})) for r in rows)}

def grasp_card(root,source):
    required=['result.json','robot_physics/report.json','robot_physics/assembled_robot_audit.json','robot_physics/wrist_camera_audit.json']
    if not all((source/name).is_file() for name in required):
        return ('<article class="featured"><div class="heading"><h2>CompleteSquare · 居中 Robotiq 实际抓放</h2><span class="badge pending">等待本次实际录像与验收</span></div><p>计划在本次相机挂载检查通过后，重新执行居中 Robotiq 抓放并同步录制手腕视角。</p></article>',
                {'source':str(source),'passed':False,'status':'pending','action':'PickPlaceObject'})
    alias='complete_square';link_directory(root,alias,source)
    result=read(source/'result.json');report=read(source/'robot_physics/report.json')
    robot_audit=read(source/'robot_physics/assembled_robot_audit.json')
    camera_audit=read(source/'robot_physics/wrist_camera_audit.json')
    passed=bool(result.get('task_completed') and robot_audit.get('passed') and camera_audit.get('passed'))
    rows=[r for r in report['results'] if not r['negative_control']]
    if len(rows)!=1:raise ValueError('Expected one CompleteSquare positive case')
    row=rows[0];p=row['physical'];dt=row['query_physical_clock']['dt_s']
    text=f'''<article class="featured"><div class="heading"><h2>CompleteSquare · 居中 Robotiq 实际抓放</h2><span class="badge {'pass' if passed else 'pending'}">{'抓放与手腕录像通过' if passed else '未完成验证'}</span></div><p>把棕色方块放到三个彩色方块缺失的顶点，让四个方块构成正方形。</p>{video_views(row,source,alias)}<p class="metrics">落点误差 <b>{p['position_error_m']*1000:.2f} mm</b> · 松爪后稳定 <b>{p['dwell_steps']*dt:.2f} 秒</b></p><p class="links"><a href="evidence/{alias}/robot_physics/assembled_robot_audit.json">抓放独立验收</a><a href="evidence/{alias}/robot_physics/wrist_camera_audit.json">手腕相机核验</a><a href="evidence/{alias}/result.json">完整执行结果</a></p></article>'''
    return text,{'source':str(source),'passed':passed,'action':'PickPlaceObject','position_error_m':p['position_error_m']}

def build(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    diagnostics=args.diagnostics_root.resolve();index=read(diagnostics/'cases.json')
    overrides={}
    for selection in args.case_root:
        name,separator,path=selection.partition('=')
        if not separator or name not in NAMES or name in overrides:
            raise ValueError('--case-root must be an unrepeated known case=absolute-or-relative-case-directory')
        source=Path(path).resolve()
        if not source.is_dir():raise ValueError('Case source directory does not exist: '+str(source))
        overrides[name]=source
    cards=[];cases=[]
    for entry in index['cases']:
        source=overrides.get(entry['case'],diagnostics/entry['case'])
        text,result=card(output,source,entry);cards.append(text);cases.append(result)
    earlier=[];earlier_rows=[]
    if args.previous_root:
        previous_root=args.previous_root.resolve()
        for name in NAMES:
            previous=previous_root/name
            if not (previous/'result.json').is_file():continue
            alias='previous_'+previous_root.name+'_'+name;link_directory(output,alias,previous)
            reason='旧版相机挂载检查通过；缺少墙体，诊断物体采用抬高的辅助支撑，双臂腕相机画面倒置。其通过范围仅保留为历史记录。'
            earlier.append({'case':name,'source':str(previous),'passed':bool(read(previous/'result.json').get('passed')),
                            'superseded_for_scene_review':True,'reason':reason})
            earlier_rows.append(f'<li><a href="evidence/{alias}/physics/report.json">{esc(previous_root.name)} / {esc(NAMES[name])}</a>：{esc(reason)} <a href="evidence/{alias}/physics/wrist_camera_audit.json">原相机审计</a></li>')
    for name,source in overrides.items():
        previous=diagnostics/name
        if source==previous or not (previous/'result.json').is_file():continue
        result=read(previous/'result.json')
        alias='previous_'+diagnostics.name+'_'+name;link_directory(output,alias,previous)
        validation,reason=previous_case_summary(previous)
        replacement_request=read(source/'request.json') if (source/'request.json').is_file() else {}
        if replacement_request.get('robot_mount_change'):
            reason+='落到桌面的目标对应原桌边基座的路径进入 joint4 限位分支；本次独立配置将双臂基座前移 10 cm 后重新执行，原失败保留。'
        else:reason+='保留被本次录制替代的旧版视角与原始结果。'
        earlier.append({'case':name,'source':str(previous),'passed':bool(result.get('passed')),
                        'replacement':str(source),'reason':reason,'previous_validation':validation})
        earlier_rows.append(f'<li><a href="evidence/{alias}/result.json">{esc(diagnostics.name)} / {esc(NAMES[name])}</a>：{esc(reason)} <a href="evidence/{alias}/physics/wrist_camera_audit.json">原挂载验收</a></li>')
    grasp_html='';grasp=None
    if args.grasp_run:
        grasp_html,grasp=grasp_card(output,args.grasp_run.resolve())
    failed=[];failure_rows=[]
    causes={'wrist_cameras_v1':'读取了静态 USD 外参','wrist_cameras_v2':'最新相机位姿选项仍读取静态 USD',
        'wrist_cameras_v3':'Fabric 索引类型不兼容，录像未完成',
        'wrist_cameras_v4':'Fabric 读取初始化把静态 USD 位姿写入了动态相机'}
    for history in args.history:
        history=history.resolve()
        for result_path in sorted(history.glob('*/result.json')):
            result=read(result_path)
            if result.get('passed'):continue
            source=result_path.parent;alias=history.name+'_'+source.name;link_directory(output,alias,source)
            report=read(source/'physics/report.json') if (source/'physics/report.json').exists() else {}
            failure=report.get('failure') or result.get('camera_audit_failure') or '未通过'
            cause=causes.get(history.name,'原始失败保留')
            failed.append({'source':str(source),'physical_action_passed':result.get('physical_execution_verified'),
                           'camera_audit_passed':result.get('camera_audit_passed'),'failure':failure,'cause':cause})
            failure_rows.append(f'<li><a href="evidence/{alias}/result.json">{esc(history.name)} / {esc(source.name)}</a>：{esc(cause)}。<small>{esc(failure)}</small></li>')
    count=sum(c['passed'] for c in cases)
    if not args.allow_partial:
        if count!=6:raise ValueError('Final gallery requires all six diagnostic cases to pass')
        if args.grasp_run and not grasp['passed']:raise ValueError('Final gallery requires wrist grasp replay to pass')
    status={'diagnostics_root':str(diagnostics),'completed_diagnostics':count,'requested_diagnostics':6,
        'supported_native_embodiments':5,'distinct_mounts_across_embodiments':6,'diagnostic_cases':cases,
        'grasp_replay':grasp,'retained_failed_attempts':failed,'model_evaluated':False,
        'case_source_overrides':{name:str(source) for name,source in overrides.items()},
        'retained_previous_mount_validations':earlier,
        'corrections':['source walls with authored materials','real-table-only object support with no raised helper pads','upright dual Piper optical mount and measured roll check'],
        'calibration_scope':'authored simulation mounts; no physical camera-to-hand hardware calibration',
        'browser_interaction_verified':False}
    write(output/'summary.json',status)
    failures=('<details class="history"><summary>保留的实现失败记录（'+str(len(failed))+' 次）</summary><p>相机外参现在通过只读取仿真层级变换获得。之前的静态读回、索引错误和读取初始化副作用分别保留了原始失败报告。</p><ul>'+''.join(failure_rows)+'</ul></details>') if failed else ''
    if earlier_rows:
        failures='<details class="history"><summary>保留的旧版场景、相机与动作结果</summary><ul>'+''.join(earlier_rows)+'</ul></details>'+failures
    corrections='<div class="history"><b>本轮检查三个问题</b><ul><li>墙体丢失：改为引用源场景四面墙的实际网格与材质。</li><li>物体浮空：关闭抬高的辅助支撑，让物体落在实测桌面；逐帧复算底面高度差，并检查最终 0.5 秒速度。</li><li>双臂腕相机倒置：使用新的固定局部挂载与光学滚转，按实测位姿检查画面上方向。</li></ul><p class="caption">下列通过状态同时要求墙体材质、真实桌面支撑、相机随动和双臂画面朝向验证。旧结果保留在页面末尾。</p></div>'
    headline='墙体、物体支撑与手腕视角已重新验证' if count==6 else '墙体、物体支撑与手腕视角验证进度'
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GEOMIND · 全部机械臂手腕相机</title><style>
:root{{--bg:#f4f3ee;--paper:#fafbf7;--ink:#222d29;--muted:#727970;--line:#d9ddd3;--accent:#365b46}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.8 system-ui,sans-serif}}a{{color:var(--accent)}}header{{height:80px;padding:0 5vw;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:15px}}header a{{text-decoration:none}}.brand{{font-size:22px;font-weight:700;letter-spacing:2px}}main{{max-width:1450px;margin:auto;padding:30px 5vw 50px}}h1{{font-size:34px;line-height:1.4;margin:18px 0}}h2{{font-size:21px;margin:0;font-weight:600}}.eyebrow{{font-size:10px;letter-spacing:2px;color:var(--accent)}}.lead{{font-size:15px;color:var(--muted)}}.summary{{padding:18px 22px;background:#e3ecd8;border:1px solid #c7d5b7;margin:25px 0}}article{{border:1px solid var(--line);border-radius:8px;background:var(--paper);padding:21px;margin:25px 0;min-width:0}}.heading{{display:flex;gap:12px;justify-content:space-between;align-items:center}}.badge{{font-size:11px;white-space:nowrap;border-radius:20px;padding:4px 10px}}.pass{{color:#365b46;background:#dfebd4}}.pending{{background:#eee2c6}}.caption{{color:var(--muted);font-size:12px}}.views{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,290px),1fr));gap:18px}}figure{{margin:0;min-width:0}}video{{display:block;width:100%;max-width:100%;aspect-ratio:4/3;background:#d8d8d1}}figcaption{{display:flex;justify-content:space-between;gap:12px;font-size:12px;margin-top:8px}}figcaption a{{font-size:11px}}.links{{display:flex;gap:14px;flex-wrap:wrap;font-size:12px}}details{{border-top:1px solid var(--line);margin-top:18px;padding-top:14px;font-size:12px}}summary{{cursor:pointer}}.scroll{{overflow-x:auto;margin-top:15px}}table{{width:100%;border-collapse:collapse;font-size:11px}}td,th{{text-align:left;padding:7px;border-bottom:1px solid var(--line)}}td:last-child{{font-family:monospace;overflow-wrap:anywhere}}.metrics b{{font-size:21px;color:var(--accent)}}.history{{background:#efece3;padding:18px 22px;border:1px solid var(--line)}}.controls{{display:flex;gap:10px;flex-wrap:wrap;margin:15px 0}}button{{font:inherit;padding:9px 15px;border:0;border-radius:4px;color:white;background:var(--accent);cursor:pointer}}button:focus-visible,a:focus-visible{{outline:3px solid #9cb680;outline-offset:3px}}footer{{font-size:11px;color:var(--muted);margin-top:30px}}@media(max-width:700px){{header{{height:65px}}main{{padding-top:22px}}h1{{font-size:27px}}h2{{font-size:18px}}.heading{{align-items:flex-start;flex-wrap:wrap}}article{{padding:15px}}.views{{grid-template-columns:minmax(0,1fr)}}}}
</style></head><body><header><a class="brand" href="../">GEOMIND</a><a href="../wrist_cameras_delivery_v1/">查看上一版录像 ↗</a></header><main><div class="eyebrow">ROBOT EMBODIMENTS / WRIST CAMERAS</div><h1>{headline}</h1><p class="lead">相机固定在各自手腕上，随末端运动。双臂 Piper 的左右手腕分别录制，并保留外部视角作对照。</p><div class="summary"><b>5 种配置 · 6 个独立挂载 · {count} / 6 个完整检查通过</b><br>每帧相机位姿由实际仿真状态读回，独立核对挂载关系、运动和录像时钟。下面均为本次实际录制。</div>{corrections}{grasp_html}<section><h2>各本体的挂载与运动检查</h2><p class="caption">这些是末端到达诊断，用于检查相机挂载与成像；不将 FR3 / Piper 的到达结果称作抓放能力。</p>{''.join(cards)}</section>{failures}<footer>本次未进行视觉模型评测。挂载参数为仿真配置，不是实物手眼标定。<p class="links"><a href="summary.json">结果与全部来源</a></p></footer></main><script>for(const card of document.querySelectorAll('article')){{const videos=[...card.querySelectorAll('video')];if(videos.length<2)continue;const controls=document.createElement('div');controls.className='controls';const play=document.createElement('button'),pause=document.createElement('button'),status=document.createElement('span');play.textContent='同步从头播放本组';pause.textContent='暂停';status.className='caption';status.setAttribute('role','status');play.onclick=async()=>{{videos.forEach(v=>{{v.pause();v.currentTime=0;}});const results=await Promise.allSettled(videos.map(v=>v.play()));status.textContent=results.some(r=>r.status==='rejected')?'请使用各视频自带的播放按钮。':'正在同步播放本组视角。';}};pause.onclick=()=>{{videos.forEach(v=>v.pause());status.textContent='已暂停。';}};controls.append(play,pause,status);card.querySelector('.views').before(controls);}}</script></body></html>'''
    (output/'index.html').write_text(page)
    print(json.dumps({'output':str(output),'diagnostics_passed':count,'grasp_passed':None if grasp is None else grasp['passed'],'retained_failures':len(failed)},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--diagnostics-root',type=Path,default=REPO/'outputs/research/wrist_cameras_v7')
    parser.add_argument('--output',type=Path,default=REPO/'outputs/research/wrist_cameras_delivery_v2')
    parser.add_argument('--case-root',action='append',default=[],metavar='CASE=CASE_DIRECTORY',help='Use an explicitly selected source for one case; records the previous result transparently.')
    parser.add_argument('--previous-root',type=Path,default=REPO/'outputs/research/wrist_cameras_v5');parser.add_argument('--grasp-run',type=Path);parser.add_argument('--history',type=Path,nargs='*',default=[])
    parser.add_argument('--allow-partial',action='store_true');build(parser.parse_args())
