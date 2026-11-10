"""Explicit physical profiles, validated without importing Isaac Sim."""
from __future__ import annotations

from copy import deepcopy

from .resolver import _check, _fields, _integer, _number, _position


PHYSICS_TASKS = frozenset({"DetectSceneChange", "ProjectOntoLine", "ReflectAcrossLine", "InferMidpoint", "SelectByOrder"})


def supports_physics_task(task_id):
    if task_id in PHYSICS_TASKS:return True
    from .resolver import is_registered_task
    return is_registered_task(task_id)

# Generator templates: serialized in full. The physical loader supplies no
# implicit parameter values when these fields are absent.
PROCEDURAL_TABLE_PHYSICS = {
    "scene_mode": "ProceduralTable",
    "dt_s": 1 / 120,
    "protocol_step_s": 0.5,
    "frame_origin_w_m": [0.5, 0.0, 0.75],
    "robot_base_w_m": [0.0, 0.0, 0.75],
    "table_size_m": [1.0, 1.0, 0.05],
    "object_size_m": [0.03, 0.03, 0.03],
    "object_mass_kg": 0.05,
    "settle_steps": 120,
    "settle_drop_height_m": 0.02,
    "settle_velocity_tolerance_m_s": 0.01,
    "event_stiffness_n_m": 30.0,
    "event_damping_ns_m": 2.5,
}

REACH_ABOVE_TARGET = {
    "action": "ReachAboveTarget",
    "target_offset_m": [0.0, 0.0, 0.10],
    "position_tolerance_m": 0.012,
    "orientation_tolerance_rad": 0.15,
    "dwell_steps": 20,
    "max_steps": 1800,
    "joint_step_limit_rad": 0.04,
    "approach_clearance_m": 0.15,
}


def physics_profile() -> dict:
    return deepcopy(PROCEDURAL_TABLE_PHYSICS)


def execution_profile() -> dict:
    return deepcopy(REACH_ABOVE_TARGET)


def validate_physics(scene: dict) -> None:
    physics = scene["physics"]
    registered = physics.get("scene_mode") == "RegisteredScene"
    assembled = physics.get("scene_mode") == "AssembledScene"
    required = set(PROCEDURAL_TABLE_PHYSICS)
    if registered or assembled:
        required.remove("table_size_m")
    if registered:
        required.update({"registered_scene_id", "lighting_mode", "hide_basket"})
    if assembled:
        required.update({"assembly_manifest_path", "assembly_manifest_sha256", "assembly_object_bindings",
                         "asset_body_vertices_path", "asset_body_vertices_sha256", "tabletop_height_m"})
    _fields(physics, required)
    if registered:
        _check(_integer(physics["registered_scene_id"], 1), "Invalid registered scene ID")
        _check(physics["lighting_mode"] in {"natural", "bright"}, "Unsupported lighting mode")
        _check(type(physics["hide_basket"]) is bool, "hide_basket must be boolean")
    _check(scene["robot"] in {"franka", "fr3", "franka_robotiq", "piper", "piper_dual"}, "Physical backend supports Panda and FR3, Robotiq and Piper")
    _check(physics["scene_mode"] in {"ProceduralTable", "RegisteredScene", "AssembledScene"}, "Unsupported physical scene mode")
    if assembled:
        _check(scene["workspace"]["frame"] == "world" and physics["frame_origin_w_m"] == [0, 0, 0],
               "AssembledScene requires the original world frame and zero frame origin")
        _check(physics["settle_drop_height_m"] == 0, "AssembledScene cannot add a reset drop height")
        for key in ("assembly_manifest_path", "asset_body_vertices_path"):
            _check(isinstance(physics[key], str) and bool(physics[key]), f"Missing {key}")
        for key in ("assembly_manifest_sha256", "asset_body_vertices_sha256"):
            _check(isinstance(physics[key], str) and len(physics[key]) == 64
                   and all(c in "0123456789abcdef" for c in physics[key]), f"Invalid {key}")
        _check(_number(physics["tabletop_height_m"]), "Invalid assembled tabletop height")
        bindings = physics["assembly_object_bindings"]
        _check(isinstance(bindings, list) and bool(bindings), "AssembledScene requires explicit existing-object bindings")
        for binding in bindings:
            _fields(binding, {"object_id", "prim_path", "root_position_w_m", "root_orientation_wxyz"})
            _check(isinstance(binding["object_id"], str) and bool(binding["object_id"]), "Invalid assembled object ID")
            _check(isinstance(binding["prim_path"], str) and binding["prim_path"].startswith("/World/AssembledObjects/"),
                   "Assembled binding must reference the original object namespace")
            _position(binding["root_position_w_m"])
            quat = binding["root_orientation_wxyz"]
            _check(isinstance(quat, list) and len(quat) == 4 and all(_number(v) for v in quat)
                   and abs(sum(v*v for v in quat)-1) < 1e-4, "Invalid assembled root quaternion")
        _check(len({b["object_id"] for b in bindings}) == len(bindings)
               and len({b["prim_path"] for b in bindings}) == len(bindings), "Duplicate assembled object binding")
    for name in ("dt_s", "protocol_step_s", "object_mass_kg", "event_stiffness_n_m", "event_damping_ns_m",
                 "settle_velocity_tolerance_m_s"):
        _check(_number(physics[name]) and physics[name] > 0, f"physics.{name} must be positive")
    ratio = physics["protocol_step_s"] / physics["dt_s"]
    _check(ratio >= 1 and abs(ratio - round(ratio)) < 1e-6, "Protocol step must be an integer number of physics steps")
    for name in ("frame_origin_w_m", "robot_base_w_m"):
        _position(physics[name])
    for name in (("object_size_m",) if registered or assembled else ("table_size_m", "object_size_m")):
        _check(all(v > 0 for v in _position(physics[name])), f"physics.{name} dimensions must be positive")
    _check(_integer(physics["settle_steps"], 1), "settle_steps must be positive integer")
    _check(_number(physics["settle_drop_height_m"]) and physics["settle_drop_height_m"] >= 0,
           "settle_drop_height_m must be nonnegative")


def validate_execution(episode: dict, scene: dict) -> None:
    _check("physics" in scene, "Physical execution requires scene.physics")
    execution = episode["physical_execution"]
    placement = execution.get("action") in {"PickPlaceMidpoint", "PickPlaceObject"}
    assembled_action = execution.get("action") in {"PickLiftSelectedObject", "PickPlaceObject"}
    dual = scene["robot"] == "piper_dual"
    _fields(execution, set(REACH_ABOVE_TARGET) | ({"moving_object"} if placement else set())
            | ({"active_arm"} if dual else set()))
    if dual:
        _check(execution["active_arm"] in {"left", "right"}, "Dual Piper requires left/right active_arm")
        _check(execution["action"] == "ReachAboveTarget", "Dual Piper currently supports target reach only")
    _check(supports_physics_task(episode['task_id']), "No native physical implementation for this task")
    _check(execution["action"] in {"ReachAboveTarget", "PickPlaceMidpoint", "PickLiftChangedObject", "PickLiftSelectedObject", "PickPlaceObject"}, "Unsupported physical action")
    if assembled_action:
        _check(scene["physics"]["scene_mode"] == "AssembledScene", "Assembled object actions require AssembledScene")
        _check(scene["robot"] in {"franka","franka_robotiq"}, "Assembled manipulation supports Panda or Panda with Robotiq")
        if execution["action"] == "PickLiftSelectedObject":
            from .resolver import task_signature
            _check(task_signature(episode['task_id'])[2]=='SelectObject', "Selected-object lift requires an object-selection task")
        else:
            from .resolver import task_signature
            _check(task_signature(episode['task_id'])[2]=='ReportPoint', "Object placement requires a supported point task")
            _check(execution["moving_object"] in {b["object_id"] for b in scene["physics"]["assembly_object_bindings"]},
                   "Unknown assembled moving object")
            refs = set(episode["parameters"].get("endpoints", [])) | set(episode["parameters"].get("line", []))
            if "point" in episode["parameters"]: refs.add(episode["parameters"]["point"])
            _check(execution["moving_object"] not in refs, "Cannot move a task reference")
    if execution["action"] == "PickLiftChangedObject":
        _check(episode["task_id"] == "DetectSceneChange", "Changed-object lift requires change semantics")
    if execution["action"] == "PickPlaceMidpoint":
        _check(scene["robot"] == "franka", "Placement currently supports Panda only")
        _check(episode["task_id"] == "InferMidpoint", "Placement requires midpoint semantics")
        _check(execution["moving_object"] not in episode["parameters"]["endpoints"], "Cannot move a reference")
    offset = _position(execution["target_offset_m"])
    _check(offset[2] > 0, "ReachAboveTarget requires positive height offset")
    for name in ("position_tolerance_m", "orientation_tolerance_rad", "joint_step_limit_rad", "approach_clearance_m"):
        _check(_number(execution[name]) and execution[name] > 0, f"physical_execution.{name} must be positive")
    for name in ("dwell_steps", "max_steps"):
        _check(_integer(execution[name], 1), f"physical_execution.{name} must be a positive integer")
    _check(execution["max_steps"] >= execution["dwell_steps"], "max_steps is shorter than dwell_steps")
