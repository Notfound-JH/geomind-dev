"""Bind registered task contracts to hash-verified settled mixed assemblies.

Static GT uses measured geometry. Temporal contracts remain pending until a
runtime has observed and physically executed the authored scene intervention.
No planned waypoint is submitted to a solver as an observation here.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from geomind.episodes.registered_tasks import plan_temporal, public_instruction, signature, task_reference_view
from geomind.episodes.resolver import EpisodeResolver, Snapshot, load_json, solve
from geomind.episodes.runtime import evaluate_answer

from .task_binding import (
    _REFERENCE_DEFINITION, _load_measured_assembly, _require,
    _semantic_identity_binding, _write,
)


def _label(obj, descriptor, protocol):
    """Visible properties only: do not expose target/subset/anchor role names."""
    semantic = protocol.get("object_semantics", {}).get(descriptor["name"], {})
    shape = semantic.get("shape") or descriptor.get("object_category")
    if not shape:
        asset = descriptor.get("asset_key", "")
        shape = asset.rsplit("_", 1)[0] if asset else descriptor.get("kind", "object").removeprefix("primitive_")
    properties = [semantic[key] for key in ("color", "material") if semantic.get(key)]
    return f"{int(obj['id'].split('_')[1])}号{' '.join(properties + [str(shape)])}"


def _operation(response, task_spec, objects):
    if response == "SelectObject":
        return {"type": "pick"}
    protocol = task_spec["protocol"]
    moving = (protocol.get("target_object", {}).get("object_id")
              or protocol.get("gt_spec", {}).get("target_object_id")
              or protocol.get("pre_attached_obj"))
    by_legacy = {obj["legacy_object_id"]: obj for obj in objects}
    _require(moving in by_legacy, "Registered placement has no measured moving object")
    return {"type": "place", "moving_object": by_legacy[moving]["id"]}


def _goal_height(manifest, spec, objects, operation):
    """Choose physical centre height; the legacy solver retains its XY rule."""
    if operation["type"] != "place":
        return None, None
    by_alias = {obj["id"]: obj for obj in objects}
    by_legacy = {obj["legacy_object_id"]: obj for obj in objects}
    moving = by_alias[operation["moving_object"]]
    half_height = .5 * (moving["bbox_max_world_m"][2] - moving["bbox_min_world_m"][2])
    gt = spec["protocol"].get("gt_spec", {})
    resolver = gt.get("resolver")
    if resolver == "support_top_placement":
        support = by_legacy[gt["support_object_id"]]
        return support["bbox_max_world_m"][2] + half_height, {"kind": "support_top", "support_object": support["id"]}
    if resolver == "containment_placement":
        floor_name = gt.get("container_floor_object_id", gt["container_reference_object_id"])
        floor = by_legacy[floor_name]
        return floor["bbox_max_world_m"][2] + half_height, {"kind": "container_floor", "support_object": floor["id"]}
    return manifest["table"]["height_m"] + half_height, {"kind": "tabletop", "tabletop_height_m": manifest["table"]["height_m"]}


def _protocol(temporal, mapping, objects, event_interval_steps, terminal_settle_steps):
    _require(type(event_interval_steps) is int and event_interval_steps >= 1, "Event interval must be a positive integer")
    _require(type(terminal_settle_steps) is int and terminal_settle_steps >= 1, "Terminal settling must be a positive integer")
    inverse = {legacy: alias for alias, legacy in mapping.items()}
    positions = {obj["id"]: obj["position_m"] for obj in objects}
    waypoints = temporal["actual_movement"]["waypoints"]
    _require(len(waypoints) >= 2, "Temporal registration has no authored movement waypoints")
    events = []
    for index, waypoint in enumerate(waypoints[1:], start=1):
        for name, xy in waypoint["positions_xy_m"].items():
            alias = inverse[name]
            events.append({"type": "MoveObject", "object_id": alias,
                "after_steps": index * event_interval_steps,
                "position_m": [float(xy[0]), float(xy[1]), positions[alias][2]]})
    tracking = temporal["mechanism"] == "historical_identity_tracking"
    return {"protocol_id": "registered_intervention", "phases": [
        {"id": "observe", "type": "Observe", "scene_visible": True, "duration_steps": 1},
        {"id": "delay", "type": "Delay", "scene_visible": tracking,
         "duration_steps": (len(waypoints)-1) * event_interval_steps + terminal_settle_steps, "events": events},
        {"id": "respond", "type": "Respond", "scene_visible": True, "allow_response": True}],
        "query_at": "respond", "allow_history_replay": False}


def bind_and_solve(manifest_path, output, *, event_interval_steps=1, terminal_settle_steps=4):
    """Write native configs and a compatible task result from settled evidence.

    For the seven temporal instances, ``ground_truth`` is null and no GT file is
    written. Their native protocol is an execution plan, with GT deferred until
    actual runtime initial and post-intervention snapshots exist.
    """
    manifest, measured, source = _load_measured_assembly(manifest_path)
    registration = manifest.get("registration", {})
    name = registration.get("task_id")
    domain, family, response = signature(name)
    spec = deepcopy(manifest.get("registered_task_spec"))
    _require(isinstance(spec, dict) and bool(spec.get("runtime_objects")), "Registered assembly lacks its compiled task specification")
    descriptor_by_name = {obj["name"]: obj for obj in spec["runtime_objects"]}
    legacy_by_prim = {obj["prim_path"]: obj.get("object_id", obj.get("name")) for obj in manifest["objects"]}
    _require(set(legacy_by_prim.values()) == set(descriptor_by_name), "Assembly prim bindings differ from compiled registered objects")
    measured, _, _, presentation = _semantic_identity_binding(measured, source, {}, {"type": "pick"})
    mapping = {}
    for obj in measured:
        legacy = legacy_by_prim[obj["prim_path"]]
        obj["legacy_object_id"] = legacy
        obj["label"] = _label(obj, descriptor_by_name[legacy], spec["protocol"])
        mapping[obj["id"]] = legacy
    source["object_binding"] = measured
    source["identity_mapping"]["semantic_to_legacy"] = mapping
    source["identity_mapping"]["legacy_to_semantic"] = {legacy: alias for alias, legacy in mapping.items()}
    source["registered_sources_sha256"] = manifest.get("provenance", {}).get("registered_sources_sha256", {})
    positions = {obj["id"]: obj["position_m"] for obj in measured}
    legacy_positions = {mapping[alias]: point for alias, point in positions.items()}
    seed = manifest["request"]["seed"]
    temporal = plan_temporal(spec, legacy_positions, seed)
    operation = _operation(response, spec, measured)
    params = {"frame": "world", "registered_task_id": name, "legacy_task_spec": spec,
              "object_id_map": mapping, "temporal_plan": temporal}
    z, support = _goal_height(manifest, spec, measured, operation)
    if z is not None:
        params["goal_z_m"] = z
    if temporal["required"]:
        params.update(objects=list(mapping), reference={"phase": "observe", "boundary": "End"})
    output = Path(output).resolve()
    owned = [output / folder for folder in ("configs", "evaluator", "observations")]
    owned += [output / filename for filename in ("task_binding.json", "task_result.json")]
    _require(not any(path.exists() for path in owned), "Refusing to overwrite registered binding artifacts")
    root = output / "configs"
    for folder in ("task_definitions", "scenes", "layouts", "episodes", "protocols"):
        (root / folder).mkdir(parents=True, exist_ok=False)
    for folder in ("evaluator", "observations"):
        (output / folder).mkdir()
    bounds = manifest["table"]["bounds_xy_m"]
    z_values = [obj["position_m"][2] for obj in measured] + ([z] if z is not None else [])
    scene = {"scene_id": "assembled_scene", "usd_path": source["scene_usda_path"], "robot": "not_configured",
        "workspace": {"frame": "world", "axes": {"right": "+x", "forward": "+y", "up": "+z"},
            "bounds_m": {"x": bounds[:2], "y": bounds[2:], "z": [min(z_values)-.5, max(z_values)+.5]}},
        "cameras": ["assembly_preview"]}
    layout = {"layout_id": "assembled_layout", "frame": "world",
        "objects": [{key: obj[key] for key in ("id", "asset", "label", "position_m")} for obj in measured],
        "randomization": {"position_jitter_m": 0}}
    episode = {"schema_version": 3, "episode_id": "assembled_task", "task_id": name,
        "scene_id": scene["scene_id"], "layout_id": layout["layout_id"], "parameters": params,
        "response": response, "seed": seed}
    if response == "ReportPoint":
        episode["evaluation"] = {"position_tolerance_m": .001}
    if temporal["required"]:
        protocol = _protocol(temporal, mapping, measured, event_interval_steps, terminal_settle_steps)
        episode["protocol_id"] = protocol["protocol_id"]
        _write(root / "protocols/registered_intervention.json", protocol)
    _write(root / "task_definitions" / f"{name}.json", {"task_id": name, "domain": domain, "family": family})
    _write(root / "scenes/assembled_scene.json", scene)
    _write(root / "layouts/assembled_layout.json", layout)
    _write(root / "episodes/assembled_task.json", episode)
    resolved = EpisodeResolver(root).resolve("assembled_task")
    snapshot = Snapshot.from_dict({"instance_id": "assembly_" + source["states_sha256"][:24],
        "contract_hash": resolved.contract_hash, "frame": "world", "step": source["query_physics_tick"],
        "positions_m": positions})
    _write(output / "evaluator/initial_snapshot.json", snapshot.to_dict())
    _write(output / "evaluator/source_binding.json", source)
    _write(output / "evaluator/temporal_plan.json", temporal)
    gt = None
    solved = None
    checks = {"native_resolver_used": True,
        "existing_legacy_solvers_used": spec['protocol'].get('module_name')!='canonical_extension',
        "heterogeneous_body_clouds_hash_verified": True, "private_identity_mapping_complete": True,
        "layout_rerandomized": False, "objects_respawned": False,
        "independent_gt_agrees": None, "independent_gt_check_status": "not_performed",
        "planned_intervention_states_used_as_gt": False,
        "semantic_control_physics_executed": False}
    if not temporal["required"]:
        solved = solve(resolved, snapshot)
        gt = solved["ground_truth"]
        if spec['protocol'].get('module_name')=='canonical_extension':
            from geomind.episodes.registered_tasks import check_extended_ground_truth
            independent=check_extended_ground_truth(resolved,snapshot,gt)
            _require(independent['success'],'Independent canonical task check disagrees with solver')
            checks.update(existing_legacy_solvers_used=False,independent_gt_agrees=True,
                          independent_gt_check_status='measured independent task predicates',
                          independent_predicates=independent)
        correct = ({"object_id": gt["object_id"]} if gt["type"] == "Object"
                   else {"position_m": gt["position_m"], "frame": "world"})
        if gt["type"] == "Relation":
            alternate = {"position_m": positions[operation["moving_object"]], "frame": "world"}
            positive = evaluate_answer(resolved, solved, correct)
            diagnostic = evaluate_answer(resolved, solved, alternate)
            _require(positive.get("command_valid") is True and diagnostic.get("command_valid") is True
                     and positive.get("semantic_deferred_to_relation") is True
                     and diagnostic.get("semantic_deferred_to_relation") is True,
                     "Registered relation command diagnostics failed")
            checks.update(representative_command_valid=True, alternate_command_valid=True,
                semantic_task_success_deferred_to_relation=True,
                command_diagnostics_are_semantic_task_acceptance=False,
                correct_answer_accepted=None, wrong_answer_rejected=None)
            _write(output / "evaluator/command_diagnostics.json", {
                "scope": "command schema/frame/workspace diagnostics only; no relation task outcome",
                "representative_command": correct, "representative_validation": positive,
                "alternate_command": alternate, "alternate_validation": diagnostic})
        elif gt["type"] == "Object":
            wrong = {"object_id": next(alias for alias in mapping if alias != gt["object_id"])}
        else:
            wrong_point = list(gt["position_m"]); wrong_point[0] += .05
            wrong = {"position_m": wrong_point, "frame": "world"}
        if gt["type"] != "Relation":
            positive = evaluate_answer(resolved, solved, correct)
            negative = evaluate_answer(resolved, solved, wrong)
            _require(positive["success"] is True and negative["success"] is False, "Registered semantic answer controls failed")
            checks.update(correct_answer_accepted=True, wrong_answer_rejected=True, semantic_wrong_answer=wrong)
        _write(output / "evaluator/query_snapshot.json", snapshot.to_dict())
        _write(output / "evaluator/ground_truth.json", solved)
    source_instruction = spec.get("instruction", spec["protocol"].get("query_text", ""))
    instruction = public_instruction(spec)
    reference_view = task_reference_view(spec)
    public = {"instruction": instruction, "phase": "observe" if temporal["required"] else "query",
        "frame": "world", "axes": {"x": "+x", "y": "+y", "z": "+z"}, "units": "m",
        "input_modality": "structured_state_with_scene_image", "reference_point": "GeometricCenter",
        "task_reference": reference_view,
        "reference_point_definition": "WorldAxisAlignedBoundingBoxCenter",
        "input_contract": {"structured_object_positions_available": True,
            "image_contains_object_id_labels": False, "visual_only": False},
        "objects": [{"object_id": measured[index]["id"], "label": measured[index]["label"],
                     "position_m": measured[index]["position_m"]} for index in presentation],
        "scene_image_path": str(Path(manifest_path).resolve().parent / "detail_preview.png")}
    _write(output / "observations/initial.json", public)
    _write(output / "evaluator/semantic_checks.json", checks)
    result = {"schema_version": 1, "kind": "assembled_task_binding",
        "status": "gt_pending_runtime_task_action_unverified" if temporal["required"] else "gt_solved_task_action_unverified",
        "task_id": name, "registration": registration, "operation": operation,
        "instruction": instruction, "source_instruction": source_instruction,
        "task_reference": reference_view,
        "ground_truth": gt, "gt_pending_runtime": temporal["required"],
        "gt_verified": not temporal["required"], "native_config_root": str(root),
        "episode_id": "assembled_task", "contract_hash": resolved.contract_hash,
        "ground_truth_path": str(output / "evaluator/ground_truth.json") if gt is not None else None,
        "public_observation_path": str(output / "observations/initial.json"),
        "source_binding_path": str(output / "evaluator/source_binding.json"), "source": source,
        "objects": measured, "checks": checks,
        "gt_inside_declared_workspace": None if gt is None or gt["type"] not in {"Point", "Relation"} else all(
            scene["workspace"]["bounds_m"][axis][0] <= gt["position_m"][index] <= scene["workspace"]["bounds_m"][axis][1]
            for index, axis in enumerate(("x", "y", "z"))),
        "target_support_and_clearance": "pending_parent_validation", "goal_height_source": support,
        "semantic_position_tolerance_m": (None if gt is not None and gt["type"] == "Relation"
                                          else episode.get("evaluation", {}).get("position_tolerance_m")),
        "relation_acceptance": "deferred_to_original_measured_physical_predicate" if gt is not None and gt["type"] == "Relation" else None,
        "reference_point_definition": _REFERENCE_DEFINITION, "assembly_settling_physics_passed": True,
        "action_physics_executed": False, "task_completed": False, "model_evaluated": False,
        "notes": ("GT pending measured observe.End and actual post-intervention runtime snapshots; event plans are not evidence."
                  if temporal["required"] else
                  "Original relation constraints bound to measured objects; representative action point is a strategy, not a unique legal target. Task success requires physical relation validation."
                  if gt["type"] == "Relation" else "GT solved by existing legacy solver from settled measured geometry; robot action remains unverified.")}
    _write(output / "task_binding.json", result)
    _write(output / "task_result.json", result)
    return result
