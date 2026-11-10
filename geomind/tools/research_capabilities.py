"""Capability declarations and explicitly selected execution evidence.

Implemented recipes are bounded contracts, not proof of a model's ability or
an automatically passed physical instance. Historical evidence remains tied to
the original saved versions; new evidence uses explicitly ordered batch roots.
"""
import argparse
from pathlib import Path

from geomind.core.registry import GeoMindRegistry
from geomind.core.taxonomy import CANONICAL_TASK_FAMILIES
from geomind.episodes.canonical_extensions import task_ids as extension_ids
from geomind.episodes.resolver import TASKS, load_json
from geomind.episodes.physical_contract import PHYSICS_TASKS
from geomind.episodes.registered_tasks import supported_registered_task_ids
from geomind.tools.research_pairs import write, digest

REPO=Path(__file__).resolve().parents[2]
HISTORICAL_DELIVERY=REPO/'outputs/research/registered_robotiq_v2/delivery.json'
DEFAULT_NEW_BATCH=REPO/'outputs/research/canonical27_smoke_v1'
NOTES={
'SelectByOrder':'First/Last/Second remain three distinct registered instances of one type; original saved-version grasp evidence retained. Stable structured identities are not visual perception evidence.',
'SelectByGridIndex':'Configured grid row/column grounding and runtime bucket validation; original saved instance physically verified, not arbitrary-grid generalization.',
'SelectByBranchPosition':'Bounded rooted directed tree, visible arrows, marked branch and ordinal after the mark. Topology/branch/ordinal are explicit; no unrestricted graph traversal claim.',
'SelectByRegion':'Visible axis-aligned rectangle defined by two diagonal markers, unique interior candidate with boundary margin. Shared translation preserves membership; other region geometries are not implemented.',
'SelectByDirection':'Original directional-reference contract and saved physical instance retained; direction conventions remain task-specific.',
'SelectByDistance':'Original Semantic nearest-anchor registration has no separately registered Base parent; measured distance and original saved physical evidence retained.',
'SelectByAdjacency':'Original directed-chain predecessor contract and visible arrows; structure is not replaced by geometric nearest-neighbour selection.',
'SelectByPathDistance':'Visible undirected graph with unweighted edge-hop shortest path, unique farthest candidate and a Euclidean-distance counterexample. Weighted graphs are not claimed.',
'PlaceByTopology':'Inside/OnTop remain two distinct relation instances. Actual grasp, release, stable support/containment and valid wrong-placement controls were audited in saved versions.',
'SelectWithinSubset':'Original subset/ranking contract and saved physical evidence retained; no general condition-necessity claim for all old layouts.',
'SelectWithVisualCue':'Original material/shape plus spatial contract and saved physical evidence retained. Attribute metadata or oracle actions are not visual-model evaluation.',
'SelectAfterExclusion':'Original exclusion/nearest solver and saved physical instance retained; older inspected layouts contain redundant conditions, so general composition necessity remains unproven.',
'SelectByTwoRelations':'Bounded intersection of two independent directional relations. Each relation alone leaves distractors; deletion controls are checked, not inferred from unique full GT.',
'SelectByIndirectReference':'Bounded two-hop reference resolution with independently unique intermediate and final referents. Shortcut controls are tested; arbitrary chain depth is not claimed.',
'InferMidpoint':'Original registered midpoint placement and independent saved-version physical evidence retained; actual moving-object end state, not TCP arrival, defines success.',
'InferPointSymmetry':'Original 2c-p registration and saved physical placement retained; no arbitrary role-assignment generalization claim.',
'ReflectAcrossLine':'Explicit TabletopPlane manipulation variant with independent midpoint/orthogonality/signed-side identities. Original bare native InfiniteLine3D axial-reflection API is preserved separately.',
'ExtendEqualSpacing':'Original ordered-extension registration and saved physical placement/control evidence retained; fixed equal-spacing contract, not arbitrary extrapolation.',
'ProjectOntoLine':'Explicit TabletopPlane placement onto an infinite tilted line, with unique minimum-displacement and orthogonality checks. Original bare 3-D projection API and diagnostic evidence remain separate.',
'CompleteSquare':'Original square-completion contract and saved physical placement retained; explicit corner roles and accepted saved geometry, not arbitrary vertex assignment.',
'ContinueSpatialPattern':'Original declared trend family and saved physical placement retained; finite point sequences do not define unrestricted pattern rules.',
'DetectSceneChange':'Confirmed unique displacement-over-3cm variant from measured Observe.End/current states; real PD event, hidden public-video phase and actual selected-object grasp. Add/remove variants remain undefined.',
'UpdateObjectOrder':'Original intervention then measured current-state ordering with saved physical evidence; final-state solvability is acknowledged, not reported as demonstrated history necessity.',
'UpdateEmptySlot':'Bounded visible-slot occupancy update from measured object centres. Fill the unique current vacancy with a real picked object; current-state sufficient, not a history-necessity claim.',
'TrackMovingObject':'Original visible continuous motion/initial-identity contract and saved physical grasp evidence retained; structured identity access is explicit and no visual identity-switch metric is claimed.',
'TrackObjectSwap':'Original swap path and initial-identity contract with saved physical grasp evidence retained; no unrestricted occlusion/re-identification claim.',
'TrackObjectOrigin':'Bounded initial marked-source region followed by visible separated mixing paths, measured initial membership and movement checks. Counterfactual history checks exist; stable structured IDs preclude pure-visual memory claims.'}


def _read(path):
    path=Path(path)
    return load_json(path) if path.is_file() else {}


def _historical(registry):
    delivery=_read(HISTORICAL_DELIVERY);by_public={}
    for row in delivery.get('cases',[]):
        identity=row['task_id'];source=Path(row.get('selected',{}).get('case',''))
        protocol=_read(source/'registered_plan.json').get('task_spec',{}).get('protocol',{})
        temporal=(protocol.get('module_name')=='post_action_relocation'
                  and protocol.get('intervention',{}).get('script_name')!='static_scene_noop')
        item={'registration_id':identity,'case_directory':str(source),
              'task_completed_at_saved_version':row.get('task_completed') is True,
              'gt_verified_at_saved_version':row.get('gt_verified') is True,
              'original_qualification':row.get('qualification',{}),
              'source_snapshot_id':row.get('integrity',{}).get('source_snapshot_id'),
              'temporal_event_required':temporal,
              'temporal_script':protocol.get('intervention',{}).get('script_name'),
              'scope':'Accepted historical delivery at its saved source/config version; no fresh current-code physical execution'}
        by_public.setdefault(registry.get(identity).public_task_id,[]).append(item)
    return by_public


def _new_evidence(identity,roots):
    candidates=[root/identity for root in roots if (root/identity).is_dir()]
    if not candidates:return {'registration_id':identity,'status':'not_executed','task_completed':False,'selection_policy':'Latest explicitly listed case wins, including failure'}
    case=candidates[-1];result=_read(case/'result.json');report=_read(case/'robot_physics/report.json')
    positives=[r for r in report.get('results',[]) if not r.get('negative_control')]
    negatives=[r for r in report.get('results',[]) if r.get('negative_control')]
    # A semantic rejection is not an executed wrong-command control. Older
    # worker/auditor flags did not distinguish a failed wrong-object grasp.
    from geomind.tools.audit_negative_pick import audit_negative_picks
    selected_commands=audit_negative_picks(case,report)
    checks={
        'result_task_completed':result.get('task_completed') is True,
        'initial_layout_physics_verified':result.get('initial_layout_physics_verified') is True,
        'task_action_physics_verified':result.get('task_action_physics_verified') is True,
        'gt_verified':result.get('gt_verified') is True,
        'independent_gt_checked':bool(positives) and all(r.get('independent_gt_check',{}).get('success') is True for r in positives),
        'physical_execution_verified':report.get('physical_execution_verified') is True,
        'raw_state_offline_audit':_read(case/'robot_physics/assembled_robot_audit.json').get('passed') is True,
        'actual_negative_controls_rejected':bool(negatives) and all(r.get('verification_passed') is True and r.get('success') is False for r in negatives),
        'selected_wrong_pick_commands_valid':selected_commands.get('passed') is True,
        'run_end_provenance':result.get('provenance_check',{}).get('passed') is True}
    passed=all(checks.values())
    return {'registration_id':identity,'case_directory':str(case),'status':
            'action_recorded_passed' if passed else 'invalid_negative_pick_control'
            if selected_commands.get('passed') is False else result.get('status','physical_execution_pending'),
            'task_completed':passed,'checks':checks,'failure':result.get('failure'),'versions':[str(path) for path in candidates],
            'selected_wrong_pick_command_audit':selected_commands,
            'selection_policy':'Latest explicitly listed case wins, including failure',
            'scope':'Recorded per-instance physics/GT/audit/provenance flags; final delivery separately checks saved bytes, complete media and wrist-camera audits'}


def build(new_batches=None):
    definitions=[(str(p.relative_to(REPO)),load_json(p)) for p in sorted((REPO/'geomind/configs/tasks').glob('*.json'))]
    registry=GeoMindRegistry();registry.load();historical=_historical(registry)
    roots=[Path(root).resolve() for root in (new_batches if new_batches is not None else [DEFAULT_NEW_BATCH])]
    if len(roots)!=len(set(roots)):raise ValueError('Capability evidence roots must be distinct and explicitly ordered')
    if any(not root.is_dir() for root in roots):raise ValueError('Capability evidence batch roots must exist')
    supported=supported_registered_task_ids();new_ids=set(extension_ids());rows=[]
    for family,tasks in CANONICAL_TASK_FAMILIES.items():
        for task in tasks:
            configs=[(p,c) for p,c in definitions if c.get('task_id')==task]
            legacy=any(c.get('execution')=='LegacyIsaacLab' for _,c in configs)
            registered=[identity for identity in supported if registry.get(identity).public_task_id==task]
            native_registered=[identity for identity in registered if identity in new_ids]
            bare_native=task in PHYSICS_TASKS;native_recipe=None;recipe_hashes={}
            if native_registered:
                import importlib
                spec=registry.get(native_registered[0]);module=importlib.import_module(spec.semantic_module)
                compiled=module.build_spec(native_registered[0],scene_id=65,seed=0)
                source=Path(module.__file__);recipe_hashes[str(source.relative_to(REPO))]=digest(source.read_bytes())
                native_recipe={'module':spec.semantic_module,'variant':compiled['protocol'].get('variant'),
                               'registration_ids':native_registered,'runner':'geomind.tools.regenerate_registered_tasks',
                               'action':spec.meta['native_assembly']['action'],
                               'quality_scope':compiled.get('quality_contract',compiled.get('definition_scope',{}))}
            old_temporal=[entry['temporal_event_required'] for entry in historical.get(task,[])]
            events=any(old_temporal) or bool(native_registered and family in {'SceneUpdating','IdentityTracking'})
            mode=('history_comparison' if task=='DetectSceneChange' else 'continuous_identity_tracking' if family=='IdentityTracking'
                  else 'update_then_query_current_sufficient' if family=='SceneUpdating' or events and all(old_temporal)
                  else 'mixed_static_and_update_current_by_instance' if events else 'static_query')
            new_evidence=[_new_evidence(identity,roots) for identity in native_registered]
            evidence={'historical_saved_instances':historical.get(task,[]),'new_instances':new_evidence,
                      'new_batch_roots_in_selection_order':[str(root) for root in roots],
                      'model_evaluated':False,'registered_or_implemented_is_not_physical_pass':True}
            evidence_text=('; '.join(f"{entry['registration_id']}: {entry['status']}" for entry in new_evidence) if native_registered else
                f"{sum(e['task_completed_at_saved_version'] for e in historical.get(task,[]))}/{len(historical.get(task,[]))} historical saved instances accepted; not re-executed under current source")
            rows.append(dict(task_id=task,domain=registry.get(registered[0]).domain,family=family,
                definition_version='Canonical taxonomy with bounded operational scope defined by actual recipe or saved registered contract',
                definition_sha256=digest((REPO/'geomind/configs/task_definitions'/f'{task}.json').read_bytes()),
                config_sources=[p for p,_ in configs],legacy_declared=legacy,native_solver=bool(registered) or task in TASKS,
                native_generator=bool(registered) or bare_native,native_events=events,
                native_controller=bool(registered) or bare_native,native_execution_implemented=bool(registered) or bare_native,
                legacy_execution_declared=legacy,assembled_native_implemented=bool(registered),
                registration_kind='AssembledNative' if native_registered else 'ExistingRegisteredMeasuredAdapter',
                registered_manipulation_ids=registered,native_recipe=native_recipe,implementation_sources_sha256=recipe_hashes,
                bare_native_episode_api_preserved=task in TASKS,
                data_export='Native configs, evaluator/public observations, raw state/control NPZ, original external/wrist video where actually recorded',
                base='Bounded native recipe with shared structure-preserving layout translation' if native_registered else 'Existing registered task and seeded structure-preserving layout',
                aug='No new Aug claim; existing Semantic/Visual conditions retain their separate registrations',
                legacy_conditions=sorted({c.get('condition','unspecified') for _,c in configs}),
                legacy_variants=sorted({str(c.get('variant')) for _,c in configs if 'variant' in c}),
                time_mechanism=mode,gt_state='measured Observe.End + current' if task=='DetectSceneChange' else
                    'measured initial identity/source + actual terminal state' if family=='IdentityTracking' else 'measured current state',
                input_answer='structured measured geometry/history + scene image → point for actual placement' if family=='GeometricInference' or task in {'PlaceByTopology','UpdateEmptySlot'} else 'structured measured geometry/history + scene image → object identity for actual grasp',
                scenes=['registered_scene_65'] if native_registered else sorted({c['scene_id'] for _,c in configs if 'scene_id' in c}),
                layout=sorted({c['layout_id'] for _,c in configs if 'layout_id' in c}),
                backends=['isaaclab_native_assembled']+(['semantic_native_episode','isaaclab_native_episode'] if bare_native else [])+(['isaaclab_legacy'] if legacy else []),
                current_execution_evidence=evidence_text,execution_evidence=evidence,gap=NOTES[task]))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--new-batch',type=Path);parser.add_argument('--retry',type=Path,action='append',default=[])
    args=parser.parse_args()
    if args.retry and not args.new_batch:parser.error('--retry requires --new-batch')
    roots=[args.new_batch,*args.retry] if args.new_batch else None
    rows=build(roots);write(args.output,rows)
    md=['# 27项任务能力矩阵','',
        '更新：2026-11-10。27 是 canonical 类型数；原 20 个实例覆盖 17 类，新 10 个注册实例补齐剩余类型。完整集合共 30 个实例。',
        'A=新 AssembledNative 有界实现；R=已有注册任务的实测场景适配；/L=仍保留旧 manager 接口。实现标记不代表物理通过。',
        '原 20 项保留其原源码、配置与物理证据；新任务仅按明确列出的实际批次逐例报告。自然抓放、错误对照、运行时 GT 和独立原始状态验收分别检查；未接模型。',
        '新实现的任务定义、场景、生成与时间机制已接入现有原生链，不扩大为任意图、任意区域、任意关系组合或纯视觉记忆能力。原三项 bare 3-D/场景更新 API 另行保留。',
        '发现入口：`python -m geomind.tools.research_pipeline --assembly-options` 的 `registered_manipulation_tasks`；执行入口为 `python -m geomind.tools.regenerate_registered_tasks --tasks TASK_ID --output NEW_DIRECTORY --max-steps 12000`。静态请求入口仍只接受原四种 task schema。','',
        '|Task|Domain / Family|实现|时间 / GT|实际执行证据|范围与尚存边界|','|---|---|---|---|---|---|']
    for row in rows:
        status=('A' if row['registration_kind']=='AssembledNative' else 'R')+('/L' if row['legacy_declared'] else '')
        md.append(f"|{row['task_id']}|{row['domain']} / {row['family']}|{status}|{row['time_mechanism']}; {row['gt_state']}|{row['current_execution_evidence']}|{row['gap']}|")
    md.extend(['','JSON 保存纯任务定义 SHA256、实现源码 SHA256、具体注册 ID、原始证据路径、版本和逐项状态；新增物理通过计数以交付页严格门槛为准。',
        '历史证据：[20 项 Robotiq 交付记录](experiments/2026-11-10-registered-robotiq-regeneration.md)。新扩展的科学解释遵循实际输入合同；结构化 ID 和精确状态均不能冒充视觉模型成绩。'])
    args.output.with_suffix('.md').write_text('\n'.join(md)+'\n')


if __name__=='__main__':main()
