"""Bind native task semantics to a measured, settled assembly without respawning it.

This creates task contracts and GT, not robot execution. Source files are read
only. The native semantic resolver receives measured geometric centres in world
coordinates, with no second round of layout randomization.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from geomind.episodes.resolver import EpisodeResolver, Snapshot, TASKS, load_json, solve
from geomind.episodes.runtime import evaluate_answer
from .audit import measured_bounds, score_window
from .body_geometry import load_body_clouds, points_for


SUPPORTED_TASKS = frozenset({"SelectByOrder", "InferMidpoint", "ProjectOntoLine", "ReflectAcrossLine"})
_COLUMNS = ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"]
_REFERENCE_DEFINITION = "Center of the world-axis-aligned bounding box of scaled mesh vertices at the measured pose; not a mass centroid."


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _source_file(source, value, name):
    path = Path(value)
    if not path.is_absolute():
        path = source / path
    path = path.resolve()
    _require(path.parent == source and path.name == name and path.is_file(), f"Expected assembly-local {name}")
    return path


def _load_measured_assembly(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    source = manifest_path.parent
    manifest = load_json(manifest_path)
    _require(manifest.get("kind") == "initial_scene_assembly", "Not an initial assembly manifest")
    _require(manifest.get("status") == "assembled_physics_passed", "Assembly has not passed physical settling")
    report_path = source / "physics_report.json"
    audit_path = source / "offline_audit.json"
    report = load_json(report_path)
    offline = load_json(audit_path)
    _require(manifest.get("physics") == report, "Manifest and physics report disagree")
    _require(manifest.get("offline_audit") == offline and offline.get("passed") is True, "Assembly offline audit missing or failed")
    _require(report.get("executed") is True and report.get("passed") is True, "Physical settling missing or failed")
    _require(report.get("backend") == "isaacsim_physx" and report.get("kind") == "initial_layout_settling", "Unsupported physical evidence")
    _require(report.get("physics_steps") == 600 and report.get("stable_steps", 0) >= 60, "Incomplete physical state history")
    warmup = report.get("warmup_steps")
    _require(type(warmup) is int and warmup >= 0, "Missing or invalid physics warmup steps")
    objects = manifest.get("objects")
    _require(isinstance(objects, list) and objects and report.get("object_count") == len(objects), "Object count mismatch")
    count = len(objects)
    _require(manifest["request"]["object"]["count"] == count, "Requested and instantiated object counts differ")
    paths = [obj["prim_path"] for obj in objects]
    identities = [path.rsplit("/", 1)[-1] for path in paths]
    _require(len(set(paths)) == count and len(set(identities)) == count, "Duplicate object bindings")
    _require(all(path.startswith("/World/AssembledObjects/") for path in paths), "Unexpected assembled object namespace")
    scene_path = _source_file(source, manifest["scene_usda"], "scene.usda")
    body_path = _source_file(source, manifest["asset_body_vertices"], "asset_body_vertices.npz")
    states_path = _source_file(source, report["state_trace"], "object_states.npz")
    _require(_digest(scene_path) == manifest["provenance"].get("scene_usda_sha256"), "Assembly USD hash changed")
    for path, expected in manifest["provenance"].get("used_usd_layers_sha256", {}).items():
        _require(_digest(path) == expected, f"Assembly USD layer changed: {path}")
    expected_body_hash = manifest["provenance"].get("asset_body_vertices_sha256")
    _require(isinstance(expected_body_hash, str) and len(expected_body_hash) == 64, "Original body-point hash is missing; regenerate the assembly")
    _require(_digest(body_path) == expected_body_hash, "Body-point hash changed")
    hashes = report.get("sha256", {})
    required_files = {"preview.png", "detail_preview.png", "settling.mp4", "object_states.npz", "asset_body_vertices.npz"}
    _require(required_files <= hashes.keys(), "Physical report lacks required artifact hashes")
    for name, expected in hashes.items():
        _require(Path(name).name == name and isinstance(expected, str) and len(expected) == 64, "Invalid artifact hash entry")
        path = _source_file(source, name, name)
        _require(_digest(path) == expected, f"Artifact hash changed: {name}")
    _require(hashes["asset_body_vertices.npz"] == expected_body_hash, "Body-point provenance disagrees")
    with np.load(states_path, allow_pickle=False) as archive:
        _require({"data", "dt_s", "columns", "prim_paths"} <= set(archive.files), "State archive schema incomplete")
        states = np.asarray(archive["data"], dtype=float)
        dt = np.asarray(archive["dt_s"], dtype=float)
        _require(dt.shape == () and np.isfinite(dt).all() and abs(float(dt) - 1 / 120) < 1e-12, "Unexpected physics timestep")
        _require(abs(report.get("dt_s", 0) - float(dt)) < 1e-12, "Reported timestep differs from raw states")
        _require(archive["columns"].tolist() == _COLUMNS, "Unexpected state column schema")
        _require(archive["prim_paths"].tolist() == paths, "State prim paths differ from manifest order")
    _require(states.shape == (600, count, 13) and np.isfinite(states).all(), "Invalid state trace shape or nonfinite values")
    _require(np.allclose(np.linalg.norm(states[:, :, 3:7], axis=2), 1, atol=1e-4), "Invalid measured quaternion norms")
    points = load_body_clouds(dict(manifest, asset_body_vertices=str(body_path)))
    dwell = score_window(states, manifest, points)
    _require(dwell == report["stable_steps"] == offline.get("stable_steps_recomputed"), "Raw trace no longer reproduces stable dwell")
    _require(offline.get("offline_wrong_layout_control_rejected") is True, "Assembly wrong-layout control was not rejected")
    measured = []
    for index, (identity, obj, state, metric) in enumerate(zip(identities, objects, states[-1], report.get("objects", []))):
        _require(metric.get("prim_path") == obj["prim_path"], "Physics metric object order differs")
        low, high = measured_bounds(points_for(points, index), state[:3], state[3:7])
        _require(np.allclose(low, metric["measured_bbox_min_w_m"], atol=1e-6) and np.allclose(high, metric["measured_bbox_max_w_m"], atol=1e-6), "Measured mesh bounds differ from physics report")
        asset_name = obj.get('asset_name', manifest['asset']['name'])
        asset_label = "蓝色罐" if asset_name == "blue_can" else asset_name
        measured.append({"id": identity, "label": f"{len(measured) + 1}号{asset_label}",
                         "asset": asset_name, "prim_path": obj["prim_path"],
                         "position_m": ((low + high) * .5).tolist(), "reference_point": "GeometricCenter",
                         "rigid_origin_world_m": state[:3].tolist(), "orientation_wxyz": state[3:7].tolist(),
                         "bbox_min_world_m": low.tolist(), "bbox_max_world_m": high.tolist()})
    _require(len(measured) == count, "Physics metrics are incomplete")
    binding = {"manifest_path": str(manifest_path), "manifest_sha256": _digest(manifest_path),
               "scene_usda_path": str(scene_path), "scene_usda_sha256": _digest(scene_path),
               "body_points_path": str(body_path), "body_points_sha256": expected_body_hash,
               "states_path": str(states_path), "states_sha256": _digest(states_path),
               "physics_report_sha256": _digest(report_path), "offline_audit_sha256": _digest(audit_path),
               "artifact_hashes_verified": hashes, "state_row": 599, "measured_physics_steps": 600,
               "warmup_steps": warmup, "query_physics_tick": warmup + 600,
               "dt_s": float(dt), "frame": "world", "reference_point": "GeometricCenter",
               "reference_point_definition": _REFERENCE_DEFINITION,
               "object_binding": measured, "layout_rerandomized": False, "objects_respawned": False}
    return manifest, measured, binding


def _validate_task(task_spec, identities):
    _require(isinstance(task_spec, dict) and set(task_spec) == {"task_id", "parameters", "operation"}, "Task binding requires task_id, parameters and operation only")
    name = task_spec["task_id"]
    _require(name in SUPPORTED_TASKS, "Unsupported assembled static task")
    _require(isinstance(task_spec["parameters"], dict), "Task parameters must be an object")
    params = deepcopy(task_spec["parameters"])
    _require("frame" not in params or params["frame"] == "world", "Binding frame is fixed to world")
    params["frame"] = "world"
    operation = deepcopy(task_spec["operation"])
    _require(isinstance(operation, dict) and {"type"} <= operation.keys() and operation.keys() <= {"type", "moving_object"}, "Invalid operation fields")
    _require(operation["type"] in {"pick", "place"}, "Unsupported operation")
    if name == "SelectByOrder":
        _require(operation["type"] == "pick", "Object selection requires pick operation")
    else:
        _require(operation["type"] == "place" and operation.get("moving_object") in identities, "Point tasks require a valid moving_object and place operation")
        references = set(params.get("endpoints", [])) | set(params.get("line", []))
        if "point" in params:
            references.add(params["point"])
        _require(operation["moving_object"] not in references, "Moving object must differ from all task references")
    if "moving_object" in operation:
        _require(operation["moving_object"] in identities, "Unknown moving_object")
    return name, params, operation


def _semantic_identity_binding(measured, source, params, operation):
    """Assign identities and presentation order without geometry/task-dependent RNG.

    Seeds are domain-separated derivatives of the raw-state archive hash. The
    privileged mapping preserves the physical request IDs; public/native task
    identities and labels contain only independent semantic numbers.
    """
    identity_seed = hashlib.sha256(("assembly_semantic_identity_v1:" + source["states_sha256"]).encode()).hexdigest()
    presentation_seed = hashlib.sha256(("assembly_public_presentation_v1:" + source["states_sha256"]).encode()).hexdigest()
    semantic_ids = [f"obj_{index + 1:03d}" for index in range(len(measured))]
    shuffled_ids = list(semantic_ids)
    random.Random(int(identity_seed, 16)).shuffle(shuffled_ids)
    # Avoid retaining the full source enumeration by chance for small scenes.
    if len(shuffled_ids) > 1 and shuffled_ids == semantic_ids:
        shuffled_ids = shuffled_ids[1:] + shuffled_ids[:1]
    physical_to_semantic = {obj["id"]: identity for obj, identity in zip(measured, shuffled_ids)}
    semantic_measured = []
    for obj in measured:
        semantic = deepcopy(obj)
        semantic["source_object_id"] = obj["id"]
        semantic["id"] = physical_to_semantic[obj["id"]]
        asset_label = "蓝色罐" if obj["asset"] == "blue_can" else obj["asset"]
        semantic["label"] = f"{int(semantic['id'].split('_')[1])}号{asset_label}"
        semantic_measured.append(semantic)
    mapped_params = deepcopy(params)
    for key in ("endpoints", "line", "candidates"):
        if key in mapped_params:
            _require(isinstance(mapped_params[key], list), f"{key} must be an object-ID list")
            _require(all(isinstance(obj, str) and obj in physical_to_semantic for obj in mapped_params[key]), f"Unknown physical object in {key}")
            mapped_params[key] = [physical_to_semantic[obj] for obj in mapped_params[key]]
    if "point" in mapped_params:
        _require(isinstance(mapped_params["point"], str) and mapped_params["point"] in physical_to_semantic, "Unknown physical point object")
        mapped_params["point"] = physical_to_semantic[mapped_params["point"]]
    mapped_operation = deepcopy(operation)
    if "moving_object" in mapped_operation:
        mapped_operation["moving_object"] = physical_to_semantic[mapped_operation["moving_object"]]
    order = list(range(len(semantic_measured)))
    random.Random(int(presentation_seed, 16)).shuffle(order)
    if len(order) > 1 and order == list(range(len(order))):
        order = order[1:] + order[:1]
    source["identity_mapping"] = {"physical_to_semantic": physical_to_semantic,
                                  "semantic_to_physical": {value: key for key, value in physical_to_semantic.items()},
                                  "identity_seed_sha256": identity_seed, "presentation_seed_sha256": presentation_seed,
                                  "public_presentation_ids": [semantic_measured[index]["id"] for index in order],
                                  "policy": "Independent state-hash-seeded ID permutation and public presentation; no geometry, task or target used to order objects."}
    source["object_binding"] = semantic_measured
    return semantic_measured, mapped_params, mapped_operation, order


def _independent_gt(name, params, positions):
    """Different arithmetic from production solve; also check geometric identities."""
    if name == "SelectByOrder":
        direction = -1 if params["from"] == "Right" else 1
        candidates = params["candidates"]
        ranks = {obj: sum(direction * positions[other][0] < direction * positions[obj][0] for other in candidates) + 1 for obj in candidates}
        matches = [obj for obj, rank in ranks.items() if rank == params["index"]]
        _require(len(matches) == 1, "Independent order target is ambiguous")
        return {"type": "Object", "object_id": matches[0]}
    if name == "InferMidpoint":
        a, b = (np.asarray(positions[obj]) for obj in params["endpoints"])
        midpoint = a + (b - a) * .5
        _require(np.allclose(midpoint - a, b - midpoint, atol=1e-12), "Independent midpoint property failed")
        return {"type": "Point", "position_m": midpoint.tolist(), "frame": "world"}
    a, b = (np.asarray(positions[obj]) for obj in params["line"])
    p = np.asarray(positions[params["point"]])
    unit = (b - a) / np.linalg.norm(b - a)
    perpendicular = np.cross(unit, np.cross(p - a, unit))
    projected = p - perpendicular
    _require(abs(float(np.dot(p - projected, unit))) < 1e-10, "Independent projection perpendicularity failed")
    point = projected if name == "ProjectOntoLine" else p - 2 * perpendicular
    return {"type": "Point", "position_m": point.tolist(), "frame": "world"}


def _instruction(name, params, operation, labels):
    if name == "SelectByOrder":
        direction = "左" if params["from"] == "Left" else "右"
        return f"在{', '.join(labels[obj] for obj in sorted(params['candidates']))}中，从{direction}数，抓起第{params['index']}个物体。"
    moving = labels[operation["moving_object"]]
    if name == "InferMidpoint":
        a, b = (labels[obj] for obj in params["endpoints"])
        return f"把{moving}放到{a}与{b}的正中间。"
    a, b = (labels[obj] for obj in params["line"])
    point = labels[params["point"]]
    if name == "ProjectOntoLine":
        target = f"{point}的几何中心向通过{a}与{b}中心的无限三维直线作正交投影的位置"
    else:
        target = f"{point}的几何中心绕通过{a}与{b}中心的无限三维直线旋转180度后的位置"
    return f"把{moving}放到{target}，使被放物体的几何中心到达该位置。"


def bind_and_solve(manifest_path, task_spec, output):
    """Write native configs, a privileged GT and separate public observation.

    The source must carry original body-cloud and raw-state artifact hashes.
    Robot-action execution and target support/clearance are not certified here.
    """
    manifest, measured, source = _load_measured_assembly(manifest_path)
    identities = [obj["id"] for obj in measured]
    name, params, operation = _validate_task(task_spec, identities)
    measured, params, operation, presentation_order = _semantic_identity_binding(measured, source, params, operation)
    identities = [obj["id"] for obj in measured]
    output = Path(output).resolve()
    owned = [output / name for name in ("configs", "evaluator", "observations", "task_binding.json")]
    _require(not any(path.exists() for path in owned), "Refusing to overwrite existing task binding artifacts")
    root = output / "configs"
    for folder in ("task_definitions", "scenes", "layouts", "episodes"):
        (root / folder).mkdir(parents=True, exist_ok=False)
    for folder in ("evaluator", "observations"):
        (output / folder).mkdir()
    bounds = manifest["table"]["bounds_xy_m"]
    z_values = [obj["position_m"][2] for obj in measured]
    scene = {"scene_id": "assembled_scene", "usd_path": source["scene_usda_path"], "robot": "not_configured",
             "workspace": {"frame": "world", "axes": {"right": "+x", "forward": "+y", "up": "+z"},
                           "bounds_m": {"x": bounds[:2], "y": bounds[2:], "z": [min(z_values) - .5, max(z_values) + .5]}},
             "cameras": ["assembly_preview"]}
    layout = {"layout_id": "assembled_layout", "frame": "world",
              "objects": [{key: obj[key] for key in ("id", "asset", "label", "position_m")} for obj in measured],
              "randomization": {"position_jitter_m": 0}}
    domain, family, response = TASKS[name]
    episode = {"schema_version": 3, "episode_id": "assembled_task", "task_id": name,
               "scene_id": scene["scene_id"], "layout_id": layout["layout_id"], "parameters": params,
               "response": response, "seed": manifest["request"]["seed"]}
    if response == "ReportPoint":
        episode["evaluation"] = {"position_tolerance_m": 1e-3}
    _write(root / "task_definitions" / f"{name}.json", {"task_id": name, "domain": domain, "family": family})
    _write(root / "scenes/assembled_scene.json", scene)
    _write(root / "layouts/assembled_layout.json", layout)
    _write(root / "episodes/assembled_task.json", episode)
    resolved = EpisodeResolver(root).resolve("assembled_task")
    positions = {obj["id"]: obj["position_m"] for obj in measured}
    snapshot = Snapshot.from_dict({"instance_id": "assembly_" + source["states_sha256"][:24],
                                   "contract_hash": resolved.contract_hash, "frame": "world", "step": source["query_physics_tick"],
                                   "positions_m": positions})
    solved = solve(resolved, snapshot)
    gt = solved["ground_truth"]
    independent = _independent_gt(name, params, positions)
    agreement = (gt == independent if gt["type"] == "Object" else
                 independent["type"] == "Point" and np.allclose(gt["position_m"], independent["position_m"], atol=1e-10, rtol=0))
    _require(agreement, "Native resolver and independent GT disagree")
    if gt["type"] == "Object":
        correct = {"object_id": gt["object_id"]}
        wrong = {"object_id": next((obj for obj in identities if obj != gt["object_id"]), "wrong_object")}
        if "moving_object" in operation:
            _require(operation["moving_object"] == gt["object_id"], "Requested moving_object differs from selected GT")
    else:
        correct = {"position_m": gt["position_m"], "frame": "world"}
        wrong_point = list(gt["position_m"]); wrong_point[0] += .05
        wrong = {"position_m": wrong_point, "frame": "world"}
    positive = evaluate_answer(resolved, solved, correct)
    negative = evaluate_answer(resolved, solved, wrong)
    _require(positive["success"] is True and negative["success"] is False, "Semantic answer controls failed")
    labels = {obj["id"]: obj["label"] for obj in measured}
    instruction = _instruction(name, params, operation, labels)
    public = {"instruction": instruction, "phase": "query", "frame": "world", "input_modality": "structured_state_with_scene_image",
              "axes": {"right": "+x", "forward": "+y", "up": "+z"}, "units": "m",
              "reference_point": "GeometricCenter", "reference_point_definition": "WorldAxisAlignedBoundingBoxCenter",
              "input_contract": {"structured_object_positions_available": True, "image_contains_object_id_labels": False, "visual_only": False},
              "objects": [{"object_id": measured[index]["id"], "label": measured[index]["label"], "position_m": measured[index]["position_m"]} for index in presentation_order],
              "scene_image_path": str(Path(manifest_path).resolve().parent / "detail_preview.png")}
    checks = {"native_resolver_used": True, "independent_gt_agrees": bool(agreement),
              "correct_answer_accepted": positive["success"], "wrong_answer_rejected": not negative["success"],
              "semantic_wrong_answer": wrong, "semantic_control_physics_executed": False}
    _write(output / "evaluator/query_snapshot.json", snapshot.to_dict())
    _write(output / "evaluator/ground_truth.json", solved)
    _write(output / "evaluator/independent_checks.json", {**checks, "independent_ground_truth": independent})
    _write(output / "evaluator/source_binding.json", source)
    _write(output / "observations/initial.json", public)
    inside = None if gt["type"] != "Point" else all(scene["workspace"]["bounds_m"][axis][0] <= gt["position_m"][i] <= scene["workspace"]["bounds_m"][axis][1] for i, axis in enumerate(("x", "y", "z")))
    result = {"schema_version": 1, "kind": "assembled_task_binding", "status": "gt_solved_task_action_unverified",
              "task_id": name, "operation": operation, "instruction": instruction, "ground_truth": gt,
              "native_config_root": str(root), "episode_id": "assembled_task", "contract_hash": resolved.contract_hash,
              "ground_truth_path": str(output / "evaluator/ground_truth.json"), "public_observation_path": str(output / "observations/initial.json"),
              "source_binding_path": str(output / "evaluator/source_binding.json"), "source": source, "objects": measured,
              "checks": checks, "gt_inside_declared_workspace": inside, "target_support_and_clearance": "pending_parent_validation",
              "semantic_position_tolerance_m": episode.get("evaluation", {}).get("position_tolerance_m"),
              "reference_point_definition": _REFERENCE_DEFINITION,
              "assembly_settling_physics_passed": True, "action_physics_executed": False, "model_evaluated": False,
              "task_completed": False, "notes": "GT from measured settled geometry. No robot or task-action physics execution. Point-target support, clearance and reachability need separate validation."}
    _write(output / "task_binding.json", result)
    return result
