"""Adapt the existing twenty registered task contracts to measured episodes.

This module owns no geometric formula or task registry. Registration metadata,
legacy protocols and their existing CPU solvers remain authoritative. Planned
intervention positions are scene-event commands, never observations or GT.
"""
from __future__ import annotations
from geomind_lab.utils.portable_paths import asset_bundle_provenance

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
from types import FunctionType
from typing import Any, Mapping

from .resolver import ContractError


# The requested release-1.0 instances, including three distinct order contracts.
# This is an export boundary, not a replacement for the repository registry.
REGISTERED_TASK_IDS = (
    "CompleteSquare_v0", "ContinueSpatialPattern_v0", "ExtendEqualSpacing_v0",
    "InferMidpoint_v0", "InferPointSymmetry_v0", "PlaceByTopologyInside_v0",
    "PlaceByTopologyOnTop_v0", "SelectAfterExclusion_v0", "SelectByAdjacency_v0",
    "SelectByDirection_v0", "SelectByDistance_v0", "SelectByGridIndex_v0",
    "SelectByOrderFirst_v0", "SelectByOrderLast_v0", "SelectByOrderSecond_v0",
    "SelectWithVisualCue_v0", "SelectWithinSubset_v0", "TrackMovingObject_v0",
    "TrackObjectSwap_v0", "UpdateObjectOrder_v0",
)
INITIAL_REFERENCE = "observe.End"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _json(value: Any) -> Any:
    """Convert solver tensors/tuples to finite, portable JSON values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        value = {str(k): _json(v) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        value = [_json(v) for v in value]
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"Nonfinite or non-JSON registered-task evidence: {exc}") from exc


def _registration(task_id: str):
    _require(task_id in supported_registered_task_ids(), f"Unsupported registered instance: {task_id!r}")
    from geomind.core.resolver import resolve_task_request
    resolved = resolve_task_request(task_id)
    _require(resolved.resolved_task_id == task_id, "Registered instance identity changed")
    _require(resolved.spec.implementation_status == "implemented", "Registered task is not implemented")
    return resolved


def signature(task_id: str) -> tuple[str, str, str]:
    """Return native domain/family/response from an actual legacy registration."""
    spec = _registration(task_id).spec
    family = spec.task_family
    domain = spec.domain
    response = "ReportPoint" if family == "GeometricInference" or spec.public_task_id in {"PlaceByTopology", "UpdateEmptySlot"} else "SelectObject"
    return domain, family, response


def is_registered(task_id: str) -> bool:
    """Recognize only the twenty supported instance IDs, without imports/I/O."""
    return isinstance(task_id, str) and task_id in supported_registered_task_ids()


def supported_registered_task_ids():
    from .canonical_extensions import task_ids
    return REGISTERED_TASK_IDS + task_ids()


def temporal_required(task_spec: Mapping[str, Any]) -> bool:
    protocol = task_spec.get("protocol", {})
    if protocol.get('module_name') == 'canonical_extension':
        from .canonical_extensions import temporal_required as extension_temporal
        return extension_temporal(task_spec)
    return (protocol.get("module_name") == "post_action_relocation"
            and protocol.get("intervention", {}).get("script_name") != "static_scene_noop")


def task_reference_view(task_spec: Mapping[str, Any]) -> dict:
    """Describe the authored directional convention without changing its GT.

    Old registrations do not share one left/right view. These public reference
    directions are derived from their existing protocol, rather than inferred
    from the current camera or replaced by the native workspace's +X right.
    """
    if task_spec.get('task_reference'):
        return deepcopy(task_spec['task_reference'])
    protocol = task_spec.get("protocol", {})
    gt = protocol.get("gt_spec", {})
    resolver = gt.get("selection_resolver", gt.get("resolver"))
    result = {"frame": "world", "instruction_suffix": ""}
    if resolver == "structural_adjacency":
        result.update(kind="DirectedChain", direction="visible_arrows",
            instruction_suffix="（前驱按场景中箭头的方向判断。）")
    elif resolver == "grid_rowcol_slot":
        row, col = gt.get("row_axis", "x"), gt.get("col_axis", "y")
        result.update(kind="GridCoordinates", row_axis="+"+row, column_axis="+"+col,
            row_origin="minimum_"+row, column_origin="minimum_"+col,
            instruction_suffix=f"（行按世界{row.upper()}从小到大、列按世界{col.upper()}从小到大计数。）")
    elif resolver in {"directional_anchor_selection", "visual_spatial_constraint_selection"}:
        axis = gt.get("axis", "y")
        right = "-" if gt.get("direction") == "negative" else "+"
        result.update(kind="DirectionalReference", right_axis=right+axis,
            left_axis=("+" if right == "-" else "-")+axis, up_axis="+z",
            instruction_suffix=f"（本任务右方指世界{axis.upper()}{'减小' if right == '-' else '增大'}方向。）")
    elif resolver == "exclusion_nearest_constraint_selection":
        axis = gt.get("exclusion_axis", "x")
        result.update(kind="ExclusionOrder", left_axis="-"+axis, right_axis="+"+axis,
            instruction_suffix=f"（本任务最左指世界{axis.upper()}最小的一端。）")
    elif resolver in {"rank_initial_layout", "rank_after_intervention", "initial_rank"}:
        axis = gt.get("order_axis", "y")
        rank = str(gt.get("target_rank", ""))
        result.update(kind="OrderAxis", order_axis="+"+axis, left_axis="-"+axis,
                      right_axis="+"+axis, up_axis="+z")
        if rank not in {"middle", "center"}:
            result["instruction_suffix"] = f"（本任务从世界{axis.upper()}较小端向较大端视为从左到右。）"
    elif resolver == "nearest_anchor_selection":
        result.update(kind="PlanarDistance", distance_axes=deepcopy(gt.get("distance_axes", ["x", "y"])))
    elif resolver in {"containment_placement", "support_top_placement"}:
        result.update(kind="PlacementRelation", relation=resolver)
    else:
        result.update(kind="GeometricRelation")
    return result


def public_instruction(task_spec: Mapping[str, Any], *, from_table: bool = True) -> str:
    """Render the authored operation with its necessary public reference view."""
    protocol = task_spec.get("protocol", {})
    instruction = str(task_spec.get("instruction", protocol.get("query_text", "")))
    if from_table and protocol.get("module_name") in {"geometric_inference", "spatial_placement"}:
        instruction = instruction.replace("把手中的方块", "从桌面拿起方块，并把它")
        instruction = instruction.replace("Place the held block", "Pick up the block from the table and place it")
    suffix = task_reference_view(task_spec)["instruction_suffix"]
    return instruction + suffix if suffix and not instruction.endswith(suffix) else instruction


def _authored_task_spec(resolved, scene_id: int) -> dict:
    if resolved.spec.task_id not in REGISTERED_TASK_IDS:
        from .canonical_extensions import build_spec
        spec=build_spec(resolved.spec.task_id, scene_id, resolved.env_cfg.backend_kwargs.get('seed', 0))
        spec['public_task_contract']=deepcopy(resolved.env_cfg.backend_kwargs['task_contract'])
        return spec
    from geomind_lab.tasks.manager_based.manipulation.geomind import task_specs
    from geomind_lab.tasks.manager_based.manipulation.geomind.task_evolution import apply_task_contract
    spec = resolved.spec
    profile = task_specs._build_lightweight_scene_profile(spec.annotation_layout_key, scene_id)
    # load_task_spec has an environment-controlled randomization hook. Inject a
    # disabled hook into this invocation's globals, leaving the module and process
    # environment untouched. The exporter applies the real seeded planner once.
    original = task_specs.load_task_spec
    compile_authored = FunctionType(original.__code__,
        {**original.__globals__, "randomize_runtime_object_layout": lambda *args, **kwargs: None},
        original.__name__, original.__defaults__, original.__closure__)
    compile_authored.__kwdefaults__ = original.__kwdefaults__
    compiled = compile_authored(spec.annotation_layout_key, spec.annotation_task_index, profile)
    compiled = apply_task_contract(compiled, spec.canonical_task_id,
                                  resolved.env_cfg.backend_kwargs["task_contract"])
    return compiled


def load_registration(task_id: str, seed: int = 0, scene_id: int = 65) -> dict:
    """Compile one existing registration without randomizing or executing it."""
    _require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
    _require(type(scene_id) is int and 1 <= scene_id <= 102, "scene_id must be in 1..102")
    resolved = _registration(task_id)
    spec = resolved.spec
    if task_id not in REGISTERED_TASK_IDS:
        from .canonical_extensions import build_spec
        from geomind.core.taxonomy import TASK_TO_FAMILY
        compiled=build_spec(task_id, scene_id, seed)
        compiled['public_task_contract']=deepcopy(resolved.env_cfg.backend_kwargs['task_contract'])
        domain,family,response=signature(task_id)
        root=Path(__file__).resolve().parents[2]
        files=[root/'geomind/core'/name for name in ('registry.py','resolver.py','schema.py','episode_compat.py')]
        files += list((root/'geomind/episodes').glob('*.py'))
        files += [root/'geomind_lab/utils/portable_paths.py',root/'scripts/configure_portable_assets.py']
        bundle_data=asset_bundle_provenance()
        if bundle_data:files.append(Path(bundle_data['manifest_path']))
        files += [Path(resolved.config_path),root/'geomind/configs/task_definitions'/f'{spec.public_task_id}.json',
                  root/'geomind/configs/native/registrations'/f'{task_id}.json']
        return _json({'schema_version':1,'registration':{
            'task_id':task_id,'public_task_id':spec.public_task_id,'canonical_task_id':spec.public_task_id,
            'domain':domain,'family':family,'response':response,'condition':spec.condition,
            'layout':spec.layout,'parameters':deepcopy(spec.parameters),'scene_id':scene_id,
            'source_scene_id':scene_id,'annotation_layout_key':'canonical_extension',
            'annotation_task_index':0,'interface':spec.interface,'task_variant':spec.task_variant,
            'task_version':spec.task_version,'config_path':str(resolved.config_path),'seed':seed},
            'task_spec':compiled,'seed':seed,'temporal_required':temporal_required(compiled),
            'asset_bundle_provenance':bundle_data,
            'task_reference':task_reference_view(compiled),
            'asset_preserving':{'source':'New bounded canonical fixture; no predecessor runtime assets existed',
                'authored_runtime_assets_preserved':True,'object_mode_conversion_applied':False},
            'layout_randomization_policy':'One seeded rigid XY translation; no independent jitter or resampling',
            'source_hashes':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}})
    domain, family, response = signature(task_id)
    compiled = _authored_task_spec(resolved, scene_id)
    from geomind_lab.tasks.manager_based.manipulation.geomind import task_specs
    root = Path(__file__).resolve().parents[2]
    sources = [Path(resolved.config_path), task_specs.resolve_annotated_tasks_path(), Path(__file__), root/"geomind_lab/utils/portable_paths.py", root/"scripts/configure_portable_assets.py"]
    bundle_data=asset_bundle_provenance()
    if bundle_data:sources.append(Path(bundle_data["manifest_path"]))
    sources += [root / relative for relative in (
        "geomind/core/registry.py", "geomind/core/resolver.py", "geomind/core/schema.py",
        "geomind_lab/tasks/manager_based/manipulation/geomind/task_specs.py",
        "geomind_lab/tasks/manager_based/manipulation/geomind/task_evolution.py",
        "geomind_lab/tasks/manager_based/manipulation/geomind/layout_randomizer.py",
        "geomind_lab/mimic/utils/task_metadata_registry.py",
        "geomind_lab/mimic/envs/anchor_reference_logic.py",
        "geomind_lab/mimic/envs/constraint_selection_logic.py",
        "geomind_lab/mimic/envs/geometric_inference_logic.py",
        "geomind_lab/mimic/envs/state_reindexing_logic.py",
        "geomind_lab/mimic/envs/task_contract_logic.py",
        "geomind_lab/mimic/envs/post_action_relocation_motion.py",
        "geomind_lab/tasks/manager_based/manipulation/geomind/mdp/terminations.py",
        "geomind/configs/legacy/layouts/standard.json", "geomind/configs/legacy/layouts/vla.json",
    )]
    sources += sorted((root / "geomind/configs/layouts").glob("*.json"))
    registration = {"task_id": task_id, "public_task_id": spec.public_task_id,
        "canonical_task_id": spec.canonical_task_id, "domain": domain, "family": family,
        "response": response, "condition": spec.condition, "layout": spec.layout,
        "parameters": deepcopy(spec.parameters), "scene_id": scene_id,
        "source_scene_id": spec.scene_id, "annotation_layout_key": spec.annotation_layout_key,
        "annotation_task_index": spec.annotation_task_index, "interface": spec.interface,
        "task_variant": spec.task_variant, "task_version": spec.task_version,
        "config_path": str(resolved.config_path), "seed": seed}
    return _json({"schema_version": 1, "registration": registration,
        "task_spec": compiled, "seed": seed, "temporal_required": temporal_required(compiled),
        "asset_bundle_provenance":bundle_data,
        "task_reference": task_reference_view(compiled),
        "asset_preserving": {"authored_runtime_assets_preserved": True,
            "authored_materials_preserved": True, "object_mode_conversion_applied": False,
            "legacy_track_reindex_cuboid_scaling_preserved": True,
            "policy": "Use authored runtime descriptors and material overrides; no geometric/concrete substitution"},
        "layout_randomization_policy": "Exporter applies explicit seeded legacy planner once",
        "source_hashes": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources if p.is_file()}})


def _positions(positions: Mapping[str, Any]) -> dict[str, list[float]]:
    _require(isinstance(positions, Mapping) and bool(positions), "Missing measured object positions")
    result = {}
    for name, value in positions.items():
        point = value.get("position") if isinstance(value, Mapping) else value
        _require(isinstance(name, str) and bool(name), "Invalid object identity")
        _require(isinstance(point, (list, tuple)) and len(point) in (2, 3), "Position must have 2 or 3 coordinates")
        _require(all(type(x) in (int, float) and math.isfinite(x) for x in point), "Position must be finite")
        result[name] = [float(x) for x in point]
    return result


def _movement_paths(task_spec, initial_xy, names, plan):
    """Sample the existing portable motion functions; no scene state is written."""
    import torch
    from geomind_lab.mimic.envs.post_action_relocation_motion import (
        get_post_action_relocation_motion_function,
        resolve_post_action_relocation_motion_mode,
    )
    protocol = task_spec["protocol"]
    script = plan["script_name"]
    intervention = protocol.get("intervention", {})
    if script == "static_scene_noop":
        return {"motion_mode": None, "waypoints": [], "object_names": names,
                "planned_only": True, "requires_runtime_measurement": True}
    canonical = task_spec.get("task_metadata", {}).get("task_id")
    # Same portable motion policy as the legacy mimic environment. These choose
    # existing path functions, not task targets or GT formulae.
    defaults = {"track_single": "track_single_paired_arc",
                "track_multi": "track_multi_staggered_waypoint",
                "reindex_insert": "reindex_insert_shallow_arc",
                "reindex_neighbor": "reindex_neighbor_bezier_lanes",
                "reindex_subset": "reindex_subset_two_block_swap"}
    default = defaults.get(canonical, "track_single_paired_arc")
    mode = resolve_post_action_relocation_motion_mode(intervention, default=default)
    motion = get_post_action_relocation_motion_function(mode)
    if "step_object_indices" in plan:
        indices = plan["step_object_indices"]
        target = plan["step_target_xy"]
    elif "per_object_target_xy" in plan:
        indices = plan["initial_rank_indices"].clone()
        safe = indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 2)
        target = torch.gather(plan["per_object_target_xy"], 1, safe)
    else:
        indices = plan["moved_object_indices"].view(-1, 1)
        target = plan["target_xy"].unsqueeze(1)
    starts = torch.gather(initial_xy, 1, indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 2))
    waypoints = []
    # A runtime may execute these by the existing PD MoveObject events, but must
    # read its actual trace and validate the terminal intervention independently.
    for index in range(13):
        progress = index / 12
        points = motion(starts, target, indices, torch.tensor([progress], dtype=initial_xy.dtype),
                        object_names=names)
        waypoints.append({"progress": progress, "positions_xy_m": {
            names[int(indices[0, slot])]: points[0, slot].tolist()
            for slot in range(indices.shape[1]) if int(indices[0, slot]) >= 0}})
    return _json({"motion_mode": mode, "motion_function": f"geomind_lab.mimic.envs.post_action_relocation_motion.{motion.__name__}",
        "object_names": names, "slot_object_indices": indices,
        "start_xy_m": starts, "target_xy_m": target,
        "duration_s": float(intervention.get("duration_s", 2.0)), "waypoints": waypoints,
        "planned_only": True, "requires_runtime_measurement": True,
        "native_protocol_suggestion": {"reference": {"phase": "observe", "boundary": "End"},
            "phases": ["observe", "delay", "respond"], "event_type": "MoveObject",
            "event_policy": "PD body motion through sampled waypoints; event targets are not measured positions",
            "object_id_namespace": "legacy semantic IDs; map to native opaque IDs before execution"}})


def plan_temporal(task_spec: Mapping[str, Any], positions: Mapping[str, Any], seed: int = 0) -> dict:
    """Plan existing scene interventions from measured *initial* positions.

    The return value is privileged event data. Applying these targets is a
    separate runtime operation; this function never declares an event completed.
    """
    if task_spec.get('protocol',{}).get('module_name')=='canonical_extension':
        from .canonical_extensions import plan_temporal as extension_plan
        return extension_plan(task_spec, positions, seed)
    import torch
    from geomind_lab.mimic.envs import state_reindexing_logic as logic
    _require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
    protocol = task_spec.get("protocol", {})
    _require(isinstance(protocol, Mapping), "Missing task protocol")
    required = temporal_required(task_spec)
    result = {"required": required, "reference_key": INITIAL_REFERENCE,
              "seed": seed, "planned_event_is_ground_truth": False,
              "requires_measured_initial_and_current": required}
    if protocol.get("module_name") != "post_action_relocation":
        return result
    actual = _positions(positions)
    names = list(actual)
    subset = protocol.get("target_subset_spec", {}).get("object_ids", names)
    _require(isinstance(subset, list) and bool(subset) and set(subset) <= set(names), "Invalid active object subset")
    initial_xy = torch.tensor([[actual[name][:2] for name in names]], dtype=torch.float64)
    active = torch.tensor([[name in subset for name in names]], dtype=torch.bool)
    script = logic.resolve_state_reindexing_script_name(dict(protocol))
    axis = logic.resolve_post_action_relocation_order_axis(dict(protocol))
    is_tracking = protocol.get("track_name") in {"historical_identity_tracking", "object_identity_tracking"}
    kwargs = {"min_reorder_gap": float(protocol.get("intervention", {}).get("min_reorder_gap", .08)),
              "script_name": script, "order_axis": axis, "active_object_mask": active}
    swap_right = bool(random.Random(seed).getrandbits(1))
    forced = protocol.get("intervention", {}).get("force_neighbor_swap")
    if forced in {"left", "right"}:
        swap_right = forced == "right"
    if is_tracking:
        kwargs["swap_with_right_mask"] = torch.tensor([swap_right], dtype=torch.bool)
        plan = logic.build_historical_identity_tracking_intervention_plan(initial_xy, **kwargs)
    else:
        plan = logic.build_state_reindexing_intervention_plan(initial_xy, **kwargs)
    _require(bool(plan["script_is_supported"]), f"Unsupported registered intervention: {script}")
    result.update({"mechanism": "historical_identity_tracking" if is_tracking else "state_reindexing",
        "object_names": names, "initial_positions_m": actual, "script_name": script,
        "order_axis": axis, "intervention": deepcopy(protocol.get("intervention", {})),
        "swap_with_right": swap_right, "plan": plan,
        "actual_movement": _movement_paths(task_spec, initial_xy, names, plan)})
    return _json(result)


def validate_parameters(task_id: str, params: dict, scene: dict | None = None,
                        layout: dict | None = None) -> None:
    """Validate the registered adapter fields without replacing legacy schemas."""
    signature(task_id)
    _require(isinstance(params, dict), "Registered parameters must be an object")
    required = {"frame", "registered_task_id", "legacy_task_spec", "object_id_map", "temporal_plan"}
    optional = {"goal_z_m", "displacement_tolerance_m", "objects", "reference"}
    _require(required <= params.keys(), f"Missing registered parameters: {sorted(required - params.keys())}")
    _require(not params.keys() - required - optional, "Unknown registered parameters")
    _require(params["registered_task_id"] == task_id and params["frame"] == "world", "Registered identity/frame mismatch")
    task_spec = params["legacy_task_spec"]
    _require(isinstance(task_spec, dict) and isinstance(task_spec.get("protocol"), dict), "Missing compiled legacy protocol")
    registered = _registration(task_id)
    contract = task_spec.get("public_task_contract", {})
    _require(contract == registered.env_cfg.backend_kwargs["task_contract"],
             "Compiled public task contract differs from the selected registration")
    if task_id not in REGISTERED_TASK_IDS:
        _require(task_spec.get('extension_task_id')==task_id
                 and task_spec['protocol'].get('module_name')=='canonical_extension',
                 'Native assembled contract does not match its registered adapter')
        _require(task_spec['protocol'].get('gt_spec',{}).get('resolver')==registered.spec.public_task_id,
                 'Native assembled semantic resolver differs from registration')
    else:
        _require(task_spec["protocol"].get("module_name") == registered.env_cfg.layout_ref["annotation"]["protocol"].get("module_name"),
                 "Compiled protocol module differs from the selected registration")
    names = [obj.get("name") for obj in task_spec.get("runtime_objects", [])]
    mapping = params["object_id_map"]
    _require(isinstance(mapping, dict) and bool(mapping) and all(isinstance(k, str) and k and isinstance(v, str) and v for k,v in mapping.items()), "Invalid private identity map")
    _require(len(names) == len(set(names)) and set(mapping.values()) == set(names)
             and len(mapping.values()) == len(set(mapping.values())), "Identity map must bijectively cover compiled runtime objects")
    if layout is not None:
        _require(set(mapping) == {obj["id"] for obj in layout["objects"]}, "Identity map differs from episode objects")
    if scene is not None:
        _require(scene["workspace"]["frame"] == params["frame"], "Registered scene frame mismatch")
    plan = params["temporal_plan"]
    _require(isinstance(plan, dict) and plan.get("required") is temporal_required(task_spec), "Temporal plan does not match registered protocol")
    if plan["required"]:
        _require(plan.get("reference_key", INITIAL_REFERENCE) == INITIAL_REFERENCE, "Unknown temporal history reference")
        _require(plan.get("planned_event_is_ground_truth") is False, "A planned event cannot be GT")
        _require(params.get("reference") == {"phase": "observe", "boundary": "End"}, "Dynamic registered task requires observe.End history")
        _require(isinstance(params.get("objects"), list) and len(params["objects"]) == len(set(params["objects"]))
                 and set(params["objects"]) == set(mapping), "Dynamic registered task must identify all measured objects")
    for field in {"goal_z_m", "displacement_tolerance_m"} & params.keys():
        _require(type(params[field]) in (int, float) and math.isfinite(params[field]), f"Invalid {field}")
    _require(params.get("displacement_tolerance_m", .02) > 0, "Displacement tolerance must be positive")
    _json(params)


def _history_initial(resolved, current, history):
    _require(isinstance(history, Mapping) and INITIAL_REFERENCE in history,
             f"Measured initial history unavailable: {INITIAL_REFERENCE}")
    initial = history[INITIAL_REFERENCE]
    for field in ("instance_id", "contract_hash", "frame"):
        _require(getattr(initial, field, None) == getattr(current, field, None), f"Initial history {field} mismatch")
    _require(initial.step < current.step, "Initial history must precede measured current state")
    _require(set(initial.positions_m) == set(current.positions_m), "Initial/current identities differ")
    protocol = getattr(resolved, "protocol", None)
    _require(isinstance(protocol, Mapping) and isinstance(protocol.get("phases"), list)
             and len(protocol["phases"]) == 3, "Dynamic registered task requires a native temporal protocol")
    observe, delay, respond = protocol["phases"]
    _require(observe.get("id") == "observe" and observe.get("type") == "Observe"
             and delay.get("type") == "Delay" and respond.get("type") == "Respond",
             "Registered temporal phase boundaries differ from observe.End/Respond")
    _require(initial.step == observe["duration_steps"], "Initial history must be captured exactly at Observe.End")
    _require(current.step >= observe["duration_steps"] + delay["duration_steps"],
             "Registered GT requested before Respond after the intervention delay")
    return initial


def validate_relation_ground_truth(resolved, gt: dict) -> None:
    """Bind a private relation GT to the exact registered physical identities.

    ``position_m`` is an oracle action strategy. The original relation contract,
    not distance to that representative point, defines the legal end states.
    """
    _require(isinstance(gt, dict), "Relation GT must be an object")
    fields = {"type", "relation", "target_object_id", "reference_object_id", "contract",
              "position_m", "frame", "point_role"}
    _require(set(gt) == fields and gt["type"] == "Relation", "Invalid Relation GT schema")
    params = resolved.episode["parameters"]
    _require(is_registered(resolved.task["task_id"]), "Relation GT needs a supported registered task")
    protocol = params["legacy_task_spec"]["protocol"]
    _require(protocol.get("module_name") == "spatial_placement", "Relation GT belongs to a different task module")
    contract = protocol.get("gt_spec", {})
    relation = contract.get("resolver")
    _require(relation in {"containment_placement", "support_top_placement"}
             and gt["relation"] == relation and gt["contract"] == contract,
             "Relation GT differs from the original registered predicate")
    mapping = params["object_id_map"]
    _require(len(set(mapping.values())) == len(mapping), "Relation identity map must be bijective")
    inverse = {legacy: alias for alias, legacy in mapping.items()}
    target = contract.get("target_object_id") or protocol.get("target_object", {}).get("object_id")
    reference = (contract.get("support_object_id") if relation == "support_top_placement"
                 else contract.get("container_reference_object_id"))
    _require(target in inverse and reference in inverse and target != reference,
             "Missing or overlapping relation target/reference binding")
    _require(gt["target_object_id"] == inverse[target] and gt["reference_object_id"] == inverse[reference],
             "Relation GT target/reference aliases differ from the private binding")
    _require({gt["target_object_id"], gt["reference_object_id"]} <= {obj["id"] for obj in resolved.layout["objects"]},
             "Relation GT aliases are absent from the native layout")
    _require(gt["frame"] == params["frame"] == "world"
             and gt["point_role"] == "representative_action_goal", "Invalid relation representative frame/role")
    point = gt["position_m"]
    _require(isinstance(point, list) and len(point) == 3
             and all(type(value) in (int, float) and math.isfinite(value) for value in point),
             "Relation representative position must be finite world XYZ")
    execution = resolved.episode.get("physical_execution")
    if execution is not None:
        _require(execution.get("action") == "PickPlaceObject"
                 and execution.get("moving_object") == gt["target_object_id"],
                 "Physical moving object differs from the registered relation target")


def solve_registered(resolved, current, history=None) -> dict:
    """Call the existing GT solver on measured state, preserving task semantics."""
    task_id = resolved.task["task_id"]
    params = resolved.episode["parameters"]
    validate_parameters(task_id, params, resolved.scene, resolved.layout)
    mapping = params["object_id_map"]
    _require(set(current.positions_m) == set(mapping), "Measured identities differ from private bindings")
    if params["temporal_plan"]["required"]:
        _require(params["temporal_plan"].get("seed") == resolved.episode["seed"],
                 "Temporal plan seed differs from the episode seed")
    actual = _positions({mapping[alias]: point for alias, point in current.positions_m.items()})
    inverse = {name: alias for alias, name in mapping.items()}
    wrapped = {name: {"position": point} for name, point in actual.items()}
    task_spec = deepcopy(params["legacy_task_spec"])
    protocol = task_spec["protocol"]
    module = protocol.get("module_name")
    calculation = {"registered_task_id": task_id, "legacy_module": module,
                   "planned_positions_used_as_observations": False}
    # expected_target is an authored-instance assertion, not a rule. The new
    # measured scene is solved afresh; preserve that old assertion in evidence.
    expected = protocol.get("gt_spec", {}).pop("expected_target", None)
    if expected is not None:
        calculation["authored_expected_target"] = expected
    point_xy = None
    selected = None
    if module == 'canonical_extension':
        from .canonical_extensions import solve as extension_solve
        initial=None
        if temporal_required(task_spec):
            snapshot=_history_initial(resolved,current,history)
            initial={mapping[alias]:point for alias,point in snapshot.positions_m.items()}
            calculation.update(initial_reference=INITIAL_REFERENCE,initial_step=snapshot.step)
        solution=extension_solve(task_spec,actual,initial=initial)
        selected=solution.get('selected')
        point_xy=solution.get('point_xy')
        calculation.update(solution.get('calculation',{}))
        calculation['canonical_extension_solution']=solution
    elif module == "anchor_reference":
        from geomind_lab.mimic.envs.anchor_reference_logic import build_anchor_reference_problem_from_object_positions
        problem = build_anchor_reference_problem_from_object_positions(task_spec, wrapped)
        selected = problem["gt_object_id"]
        calculation["solver_result"] = problem
    elif module == "constraint_selection":
        from geomind_lab.mimic.envs.constraint_selection_logic import build_constraint_selection_problem_from_object_positions
        problem = build_constraint_selection_problem_from_object_positions(task_spec, wrapped)
        selected = problem["gt_object_id"]
        calculation["solver_result"] = problem
    elif module == "geometric_inference":
        from geomind_lab.mimic.envs.geometric_inference_logic import build_geometric_inference_problem_from_object_positions
        problem = build_geometric_inference_problem_from_object_positions(task_spec, wrapped)
        point_xy = problem["gt_pos_xy"]
        calculation["solver_result"] = problem
    elif module == "post_action_relocation":
        import torch
        from geomind_lab.mimic.envs import state_reindexing_logic as logic
        if temporal_required(task_spec):
            initial = _history_initial(resolved, current, history)
            initial_positions = {mapping[alias]: point for alias, point in initial.positions_m.items()}
            calculation["initial_reference"] = INITIAL_REFERENCE
            calculation["initial_step"] = initial.step
        else:
            initial_positions = actual
        planned = plan_temporal(task_spec, initial_positions, resolved.episode["seed"])
        names = planned["object_names"]
        plan = planned["plan"]
        tensor = lambda key, dtype: torch.tensor(plan[key], dtype=dtype)
        initial_xy = torch.tensor([[initial_positions[name][:2] for name in names]], dtype=torch.float64)
        post_xy = torch.tensor([[actual[name][:2] for name in names]], dtype=torch.float64)
        kwargs = dict(initial_xy=initial_xy, post_xy=post_xy,
            initial_rank_indices=tensor("initial_rank_indices", torch.long),
            displacement_tol=float(params.get("displacement_tolerance_m", .02)),
            script_name=planned["script_name"], target_rank=logic.resolve_state_reindexing_target_rank(protocol),
            order_axis=planned["order_axis"], active_object_mask=tensor("active_object_mask", torch.bool),
            planned_moved_mask=tensor("planned_moved_mask", torch.bool))
        if planned["mechanism"] == "historical_identity_tracking":
            kwargs["swap_with_right_mask"] = torch.tensor([planned["swap_with_right"]], dtype=torch.bool)
            result = logic.evaluate_historical_identity_tracking_terminal_state(**kwargs)
        else:
            kwargs.update(moved_object_indices=tensor("moved_object_indices", torch.long),
                          gt_spec=protocol["gt_spec"], object_names=names)
            result = logic.evaluate_state_reindexing_terminal_state(**kwargs)
        calculation["terminal_validation"] = _json(result)
        _require(bool(result["valid_episode"][0]), "Registered intervention/grid/order is invalid in measured current state")
        index = int(result["gt_object_indices"][0])
        _require(0 <= index < len(names), "Existing registered solver did not resolve an object")
        selected = names[index]
    elif module == "spatial_placement":
        gt_spec = protocol.get("gt_spec", {})
        resolver = gt_spec.get("resolver")
        _require(resolver in {"containment_placement", "support_top_placement"}, "Unsupported registered placement relation")
        reference = (gt_spec.get("container_reference_object_id") if resolver == "containment_placement"
                     else gt_spec.get("support_object_id"))
        _require(reference in actual, "Missing measured placement reference")
        point_xy = actual[reference][:2]
        calculation.update(relation_contract=deepcopy(gt_spec), relation_reference_object=inverse[reference],
            semantic_point_policy="Representative relation centre; physical success requires the original relation predicate",
            physical_predicate="geomind_lab.tasks.manager_based.manipulation.geomind.mdp.terminations.goal_sequence_reached")
    else:
        raise ContractError(f"No existing registered solver for module {module!r}")
    if selected is not None:
        _require(selected in inverse, "GT selects an object absent from measured bindings")
        gt = {"type": "Object", "object_id": inverse[selected]}
    else:
        target = protocol.get("target_object", {}).get("object_id") or protocol.get("pre_attached_obj")
        _require(target in actual and len(actual[target]) == 3, "Point task needs a measured moving-object centre")
        z = params.get("goal_z_m", actual[target][2])
        gt = {"type": "Point", "position_m": [float(point_xy[0]), float(point_xy[1]), float(z)], "frame": "world"}
        if module == "spatial_placement":
            gt.update(type="Relation", relation=calculation["relation_contract"]["resolver"],
                target_object_id=inverse[target], reference_object_id=calculation["relation_reference_object"],
                contract=deepcopy(calculation["relation_contract"]), point_role="representative_action_goal")
            validate_relation_ground_truth(resolved, gt)
        calculation["moving_object"] = inverse[target]
        calculation["goal_z_policy"] = "explicit_support_height" if "goal_z_m" in params else "measured_moving_object_centre"
    return _json({"instruction": public_instruction(task_spec),
                  "source_instruction": task_spec.get("instruction", protocol.get("query_text", "")),
                  "task_reference": task_reference_view(task_spec),
                  "reference": INITIAL_REFERENCE if temporal_required(task_spec) else "Current",
                  "ground_truth": gt, "calculation": calculation})


def check_extended_ground_truth(resolved, current, ground_truth, history=None):
    """Verify new task predicates independently of the production selection."""
    from .canonical_extensions import check
    params=resolved.episode['parameters'];spec=params['legacy_task_spec'];mapping=params['object_id_map']
    _require(spec['protocol']['module_name']=='canonical_extension','Not a canonical extension')
    positions={mapping[alias]:point for alias,point in current.positions_m.items()}
    initial=None
    if temporal_required(spec):
        old=_history_initial(resolved,current,history)
        initial={mapping[alias]:point for alias,point in old.positions_m.items()}
    solution=({'selected':mapping[ground_truth['object_id']]} if ground_truth['type']=='Object' else
              {'point_xy':ground_truth['position_m'][:2]})
    result=check(spec,positions,solution,initial=initial)
    _require(isinstance(result,dict) and type(result.get('success')) is bool,'Invalid independent extension check')
    return _json(result)
