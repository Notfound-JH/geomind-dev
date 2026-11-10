from __future__ import annotations
from geomind_lab.utils.portable_paths import resolve_asset_paths

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Invalid configuration, unavailable evidence, or ambiguous ground truth."""


IMPLEMENTATION_HASH = hashlib.sha256(b"".join(
    source.name.encode() + b"\0" + source.read_bytes() + b"\0"
    for source in sorted(Path(__file__).parent.glob("*.py"))
)).hexdigest()


def load_json(path: str | Path) -> Any:
    """Reject duplicate JSON keys and nonstandard NaN/Infinity literals."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ContractError(f"Nonfinite JSON constant: {value}")

    try:
        return resolve_asset_paths(json.loads(Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object, parse_constant=invalid_constant))
    except (OSError, ValueError) as exc:
        raise ContractError(f"Cannot read {path}: {exc}") from exc


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _fields(value: Any, required: set[str], optional: set[str] | None = None) -> None:
    _check(isinstance(value, dict), "Expected a JSON object")
    _check(required <= value.keys(), f"Missing fields: {sorted(required - value.keys())}")
    extra = value.keys() - required - (optional or set())
    _check(not extra, f"Unknown fields: {sorted(extra)}")


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _position(value: Any) -> tuple[float, float, float]:
    _check(isinstance(value, (list, tuple)) and len(value) == 3, "Position must have 3 coordinates")
    _check(all(_number(v) for v in value), "Coordinates must be finite numbers")
    return tuple(float(v) for v in value)


def _id(value: Any) -> None:
    _check(isinstance(value, str) and bool(value) and all(
        c.isascii() and (c.isalnum() or c in "_-") for c in value
    ), f"Invalid ID: {value!r}")


def _fingerprint(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ContractError(f"Invalid JSON contract or nonfinite computation: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


# These are public task names. No historical task/protocol IDs are involved.
TASKS = {
    "SelectByOrder": ("SpatialReasoning", "TargetIdentification", "SelectObject"),
    "InferMidpoint": ("SpatialReasoning", "GeometricInference", "ReportPoint"),
    "RecallObjectLocation": ("SpatialMemory", "LocationMemory", "ReportPoint"),
    "ProjectOntoLine": ("SpatialReasoning", "GeometricInference", "ReportPoint"),
    "ReflectAcrossLine": ("SpatialReasoning", "GeometricInference", "ReportPoint"),
    "DetectSceneChange": ("SpatialMemory", "SceneUpdating", "SelectObject"),
}


def task_signature(task_id):
    if task_id in TASKS:return TASKS[task_id]
    from .registered_tasks import signature
    return signature(task_id)


def is_registered_task(task_id):
    if task_id in TASKS:return False
    from .registered_tasks import is_registered
    return is_registered(task_id)


@dataclass(frozen=True)
class ResolvedEpisode:
    """A validated contract. Callers must treat nested configuration as read-only."""

    episode: dict[str, Any]
    task: dict[str, Any]
    scene: dict[str, Any]
    layout: dict[str, Any]
    protocol: dict[str, Any] | None
    source_hashes: dict[str, str]
    contract_hash: str


class EpisodeResolver:
    """Resolve explicit references within one independent v3 config root."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _read(self, folder: str, identity: str, id_field: str) -> tuple[dict, str]:
        _id(identity)
        path = self.root / folder / f"{identity}.json"
        data = load_json(path)
        _check(isinstance(data, dict), f"Expected object in {path}")
        _check(data.get(id_field) == identity, f"ID mismatch in {path}")
        return data, _fingerprint(data)

    def resolve(self, episode_id: str) -> ResolvedEpisode:
        episode, eh = self._read("episodes", episode_id, "episode_id")
        _fields(episode, {"schema_version", "episode_id", "task_id", "scene_id", "layout_id",
                          "parameters", "response", "seed"}, {"protocol_id", "evaluation", "physical_execution", "release_version"})
        _check(type(episode["schema_version"]) is int and episode["schema_version"] == 3,
               "Expected schema_version=3")
        _check(episode.get("release_version", "2.0") == "2.0", "Native registrations belong to release 2.0")
        _check(_integer(episode["seed"]), "seed must be an explicit nonnegative integer")
        task_folder = "task_definitions" if (self.root / "task_definitions").is_dir() else "tasks"
        task, th = self._read(task_folder, episode["task_id"], "task_id")
        scene, sh = self._read("scenes", episode["scene_id"], "scene_id")
        layout, lh = self._read("layouts", episode["layout_id"], "layout_id")
        _fields(task, {"task_id", "domain", "family"})
        domain, family, response = task_signature(task["task_id"])
        _check((task["domain"], task["family"]) == (domain, family), "Task domain/family mismatch")
        _check(episode["response"] == response, f"This task currently requires response={response}")
        if "evaluation" in episode:
            _check(response == "ReportPoint", "Object selection uses exact identity evaluation")
            _fields(episode["evaluation"], {"position_tolerance_m"})
            tolerance = episode["evaluation"]["position_tolerance_m"]
            _check(_number(tolerance) and tolerance > 0, "Evaluation tolerance must be positive")
        self._validate_scene_layout(scene, layout)
        if "physical_execution" in episode:
            from .physical_contract import validate_execution
            validate_execution(episode, scene)
        self._validate_parameters(task["task_id"], episode["parameters"], scene, layout)
        hashes = {"episode": eh, "task": th, "scene": sh, "layout": lh,
                  "implementation": IMPLEMENTATION_HASH}
        protocol = None
        if is_registered_task(task['task_id']):
            if 'protocol_id' in episode:
                protocol,ph=self._read('protocols',episode['protocol_id'],'protocol_id')
                self._validate_protocol(protocol,episode['parameters'],task['task_id'],scene,layout)
                hashes['protocol']=ph
        elif domain == "SpatialReasoning":
            _check("protocol_id" not in episode, "Reasoning episodes cannot include temporal protocols")
        else:
            _check("protocol_id" in episode, "Memory episodes require protocol_id")
            protocol, ph = self._read("protocols", episode["protocol_id"], "protocol_id")
            self._validate_protocol(protocol, episode["parameters"], task["task_id"], scene, layout)
            hashes["protocol"] = ph
        return ResolvedEpisode(episode, task, scene, layout, protocol, hashes, _fingerprint(hashes))

    @staticmethod
    def _validate_scene_layout(scene: dict, layout: dict) -> None:
        _fields(scene, {"scene_id", "usd_path", "robot", "workspace", "cameras"}, {"physics"})
        if "physics" in scene:
            from .physical_contract import validate_physics
            validate_physics(scene)
        _check(all(isinstance(scene[k], str) and scene[k] for k in ("usd_path", "robot")),
               "Scene must identify USD and robot")
        _check(isinstance(scene["cameras"], list) and bool(scene["cameras"])
               and all(isinstance(c, str) and c for c in scene["cameras"]), "Invalid cameras")
        workspace = scene["workspace"]
        _fields(workspace, {"frame", "axes", "bounds_m"})
        _id(workspace["frame"])
        _check(workspace["axes"] == {"right": "+x", "forward": "+y", "up": "+z"},
               "Only right=+x, forward=+y, up=+z is implemented")
        _fields(workspace["bounds_m"], {"x", "y", "z"})
        for bounds in workspace["bounds_m"].values():
            _check(isinstance(bounds, list) and len(bounds) == 2
                   and all(_number(v) for v in bounds) and bounds[0] < bounds[1], "Invalid workspace bounds")
        _fields(layout, {"layout_id", "frame", "objects", "randomization"})
        _check(layout["frame"] == workspace["frame"], "Layout frame differs from workspace; transform unsupported")
        _check(isinstance(layout["objects"], list) and bool(layout["objects"]), "Layout requires objects")
        ids = set()
        for obj in layout["objects"]:
            _fields(obj, {"id", "asset", "position_m"}, {"label", "physical_asset"})
            if "physical_asset" in obj:
                from .object_assets import validate_asset
                validate_asset(obj["physical_asset"])
            if "label" in obj:
                _check(isinstance(obj["label"], str) and bool(obj["label"].strip()), "Invalid object label")
            _id(obj["id"])
            _check(obj["id"] not in ids, f"Duplicate object ID: {obj['id']}")
            ids.add(obj["id"])
            _check(isinstance(obj["asset"], str) and bool(obj["asset"]), "Missing object asset")
            position = _position(obj["position_m"])
            _check(all(workspace["bounds_m"][axis][0] <= position[i] <= workspace["bounds_m"][axis][1]
                       for i, axis in enumerate(("x", "y", "z"))), "Initial object position outside workspace")
        _fields(layout["randomization"], {"position_jitter_m"})
        jitter = layout["randomization"]["position_jitter_m"]
        _check(_number(jitter) and jitter >= 0, "Invalid position jitter")
        labels = [o.get("label", o["id"]) for o in layout["objects"]]
        _check(len(set(labels)) == len(labels), "Object labels must be unique")

    @staticmethod
    def _validate_parameters(name: str, params: dict, scene: dict, layout: dict) -> None:
        if is_registered_task(name):
            from .registered_tasks import validate_parameters
            validate_parameters(name,params,scene,layout)
            return
        fields = {
            "SelectByOrder": {"index", "from", "frame", "candidates", "tie_tolerance_m"},
            "InferMidpoint": {"endpoints", "frame"},
            "RecallObjectLocation": {"objects", "query_object", "frame", "reference"},
            "ProjectOntoLine": {"point", "line", "frame", "degeneracy_tolerance_m", "geometry"},
            "ReflectAcrossLine": {"point", "line", "frame", "degeneracy_tolerance_m", "geometry"},
            "DetectSceneChange": {"objects", "frame", "reference", "change_threshold_m", "ambiguity_tolerance_m"},
        }
        _fields(params, fields[name])
        _check(params["frame"] == scene["workspace"]["frame"], "Task frame differs from workspace")
        geometry_task = name in {"ProjectOntoLine", "ReflectAcrossLine"}
        key = {"SelectByOrder": "candidates", "InferMidpoint": "endpoints",
               "RecallObjectLocation": "objects", "DetectSceneChange": "objects",
               "ProjectOntoLine": "line", "ReflectAcrossLine": "line"}[name]
        objects = params[key]
        _check(isinstance(objects, list) and bool(objects) and all(isinstance(v, str) for v in objects),
               f"{key} must be a nonempty list of object IDs")
        _check(len(objects) == len(set(objects)), f"Duplicate objects in {key}")
        _check(set(objects) <= {o["id"] for o in layout["objects"]}, f"Unknown object in {key}")
        if geometry_task:
            _check(len(objects) == 2, "Line requires two distinct endpoints")
            _check(isinstance(params["point"], str) and params["point"] in {o["id"] for o in layout["objects"]},
                   "Unknown point object")
            _check(params["point"] not in objects, "Point object must differ from line endpoints")
            _check(params["geometry"] == "InfiniteLine3D", "Only InfiniteLine3D geometry is implemented")
            _check(_number(params["degeneracy_tolerance_m"]) and params["degeneracy_tolerance_m"] > 0,
                   "degeneracy_tolerance_m must be positive")
        elif name == "SelectByOrder":
            _check(_integer(params["index"], 1) and params["index"] <= len(objects), "index is out of range")
            _check(params["from"] in ("Left", "Right"), "from must be Left or Right")
            _check(_number(params["tie_tolerance_m"]) and params["tie_tolerance_m"] > 0,
                   "tie_tolerance_m must be positive")
        elif name == "InferMidpoint":
            _check(len(objects) == 2, "Midpoint requires exactly two distinct endpoints")
        else:
            if name == "RecallObjectLocation":
                _check(params["query_object"] in objects, "query_object must be among encoded objects")
            else:
                _check(_number(params["ambiguity_tolerance_m"]) and params["ambiguity_tolerance_m"] > 0,
                       "ambiguity_tolerance_m must be positive")
                _check(_number(params["change_threshold_m"])
                       and params["change_threshold_m"] > params["ambiguity_tolerance_m"],
                       "change_threshold_m must exceed ambiguity tolerance")
            _fields(params["reference"], {"phase", "boundary"})
            _check(params["reference"]["boundary"] == "End", "Only explicit phase End references are implemented")

    @staticmethod
    def _validate_protocol(protocol: dict, params: dict, name: str, scene: dict, layout: dict) -> None:
        _fields(protocol, {"protocol_id", "phases", "query_at", "allow_history_replay"})
        _check(protocol["allow_history_replay"] is False, "Only streaming memory protocols are implemented")
        phases = protocol["phases"]
        _check(isinstance(phases, list) and len(phases) == 3, "Expected Observe, Delay, Respond phases")
        for phase, kind in zip(phases, ("Observe", "Delay", "Respond")):
            required = {"id", "type", "scene_visible"}
            required.add("allow_response" if kind == "Respond" else "duration_steps")
            _fields(phase, required, {"events"} if kind == "Delay" else set())
            _id(phase["id"])
            _check(phase["type"] == kind, f"Expected phase type {kind}")
            visible = kind == "Observe" or ((name == "DetectSceneChange" or is_registered_task(name)) and kind == "Respond")
            if is_registered_task(name) and kind == "Delay":
                _check(type(phase["scene_visible"]) is bool, "Registered intervention visibility must be explicit")
            else:
                _check(phase["scene_visible"] is visible, "Phase visibility conflicts with task evidence requirements")
            if kind == "Respond":
                _check(phase["allow_response"] is True, "Respond must allow response")
            else:
                _check(_integer(phase["duration_steps"], 1), "Phase duration must be positive integer")
        _check(len({p["id"] for p in phases}) == 3, "Duplicate phase IDs")
        _check(protocol["query_at"] == phases[2]["id"], "Query must occur at Respond")
        _check(params["reference"]["phase"] == phases[0]["id"], "Reference must identify the Observe phase")
        events = phases[1].get("events", [])
        _check(isinstance(events, list), "events must be a list")
        objects = {o["id"] for o in layout["objects"]}
        for event in events:
            _fields(event, {"type", "object_id", "after_steps", "position_m"})
            _check(event["type"] == "MoveObject", "Unsupported event type")
            _check(isinstance(event["object_id"], str) and event["object_id"] in objects, "Unknown event object")
            _check(_integer(event["after_steps"], 1) and event["after_steps"] <= phases[1]["duration_steps"],
                   "Event time outside delay phase")
            position = _position(event["position_m"])
            _check(all(scene["workspace"]["bounds_m"][axis][0] <= position[i] <= scene["workspace"]["bounds_m"][axis][1]
                       for i, axis in enumerate(("x", "y", "z"))), "Event position outside workspace")
        identities = [(e["after_steps"], e["object_id"]) for e in events]
        _check(len(set(identities)) == len(identities), "Conflicting simultaneous events")


@dataclass(frozen=True)
class Snapshot:
    """Measured world state, supplied by a backend, never inferred from layout poses."""

    instance_id: str
    contract_hash: str
    frame: str
    step: int
    positions_m: dict[str, tuple[float, float, float]]

    @classmethod
    def from_dict(cls, data: dict) -> Snapshot:
        _fields(data, {"instance_id", "contract_hash", "frame", "step", "positions_m"})
        _id(data["instance_id"])
        _check(isinstance(data["contract_hash"], str) and len(data["contract_hash"]) == 64,
               "Invalid contract hash")
        _id(data["frame"])
        _check(_integer(data["step"]), "Snapshot step must be nonnegative integer")
        _check(isinstance(data["positions_m"], dict) and bool(data["positions_m"]), "Missing measured positions")
        for identity in data["positions_m"]:
            _id(identity)
        return cls(data["instance_id"], data["contract_hash"], data["frame"], data["step"],
                   {k: _position(v) for k, v in data["positions_m"].items()})

    def to_dict(self) -> dict:
        return {"instance_id": self.instance_id, "contract_hash": self.contract_hash,
                "frame": self.frame, "step": self.step,
                "positions_m": {k: list(v) for k, v in self.positions_m.items()}}


def _validate_snapshot(resolved: ResolvedEpisode, snapshot: Snapshot) -> None:
    # Revalidate even when a caller constructed the dataclass directly.
    Snapshot.from_dict(snapshot.to_dict())
    _check(snapshot.contract_hash == resolved.contract_hash, "Snapshot belongs to a different configuration")
    _check(snapshot.frame == resolved.scene["workspace"]["frame"], "Snapshot frame mismatch")
    _check(set(snapshot.positions_m) == {o["id"] for o in resolved.layout["objects"]},
           "Snapshot object identities differ from instantiated layout")


def solve(resolved: ResolvedEpisode, current: Snapshot, *,
          history: dict[str, Snapshot] | None = None) -> dict[str, Any]:
    """Generate instruction and semantic GT from the same validated inputs.

    Output is privileged evaluation data and must not be passed to the agent.
    This function does not run physics or determine physical action success.
    """
    # Catch accidental mutation of a previously resolved contract.
    sources = {"episode": resolved.episode, "task": resolved.task,
               "scene": resolved.scene, "layout": resolved.layout}
    if resolved.protocol is not None:
        sources["protocol"] = resolved.protocol
    hashes = {k: _fingerprint(v) for k, v in sources.items()}
    hashes["implementation"] = IMPLEMENTATION_HASH
    _check(hashes == resolved.source_hashes and _fingerprint(hashes) == resolved.contract_hash,
           "Resolved configuration was mutated; resolve it again")
    _validate_snapshot(resolved, current)
    name = resolved.task["task_id"]
    params = resolved.episode["parameters"]
    evidence = current
    reference = "Current"
    if name in {"RecallObjectLocation", "DetectSceneChange"}:
        ref = params["reference"]
        reference = f"{ref['phase']}.{ref['boundary']}"
        _check(history is not None and reference in history, f"Required memory snapshot unavailable: {reference}")
        evidence = history[reference]
        _validate_snapshot(resolved, evidence)
        _check(evidence.instance_id == current.instance_id, "Memory snapshot belongs to a different run")
        observe_steps = resolved.protocol["phases"][0]["duration_steps"]
        delay_steps = resolved.protocol["phases"][1]["duration_steps"]
        _check(evidence.step == observe_steps, "Memory reference must be captured at Observe.End (reset step=0)")
        _check(current.step >= observe_steps + delay_steps, "Recall requested before delay completed")
    positions = evidence.positions_m
    labels = {o["id"]: o.get("label", o["id"]) for o in resolved.layout["objects"]}
    if is_registered_task(name):
        from .registered_tasks import solve_registered
        registered=solve_registered(resolved,current,history=history)
        instruction=registered['instruction'];gt=registered['ground_truth'];calculation=registered['calculation']
        if params['temporal_plan']['required']:
            reference='observe.End'
            _check(history is not None and reference in history, f"Required registered reference unavailable: {reference}")
            evidence = history[reference]
            _validate_snapshot(resolved, evidence)
    elif name == "SelectByOrder":
        ordered = sorted(params["candidates"], key=lambda obj: positions[obj][0], reverse=params["from"] == "Right")
        gaps = [abs(positions[b][0] - positions[a][0]) for a, b in zip(ordered, ordered[1:])]
        _check(all(gap > params["tie_tolerance_m"] for gap in gaps), "Ambiguous order within tie tolerance")
        target = ordered[params["index"] - 1]
        instruction = (f"Select object number {params['index']} from the {params['from'].lower()} "
                       f"among {', '.join(labels[o] for o in params['candidates'])}, in frame {params['frame']}.")
        gt = {"type": "Object", "object_id": target}
        calculation = {"ordered_candidates": ordered, "axis": "x", "minimum_gap_m": min(gaps) if gaps else None}
    elif name == "InferMidpoint":
        a, b = params["endpoints"]
        _check(math.dist(positions[a], positions[b]) > 0, "Midpoint endpoints coincide")
        point = [(positions[a][i] + positions[b][i]) / 2 for i in range(3)]
        instruction = f"Report the 3D midpoint of {labels[a]} and {labels[b]} in frame {params['frame']}, in metres."
        gt = {"type": "Point", "position_m": point, "frame": params["frame"]}
        calculation = {"endpoints": {a: list(positions[a]), b: list(positions[b])}}
    elif name in {"ProjectOntoLine", "ReflectAcrossLine"}:
        p = positions[params["point"]]
        a, b = (positions[o] for o in params["line"])
        direction = [b[i] - a[i] for i in range(3)]
        length2 = sum(v * v for v in direction)
        _check(length2 > params["degeneracy_tolerance_m"] ** 2, "Degenerate line within tolerance")
        fraction = sum((p[i] - a[i]) * direction[i] for i in range(3)) / length2
        projected = [a[i] + fraction * direction[i] for i in range(3)]
        point = projected if name == "ProjectOntoLine" else [2 * projected[i] - p[i] for i in range(3)]
        operation = "orthogonal projection" if name == "ProjectOntoLine" else "axial reflection (a 180-degree rotation about the line)"
        instruction = (f"Report the {operation} of the centre of {labels[params['point']]} "
                       f"using the infinite 3D line through the centres of {labels[params['line'][0]]} "
                       f"and {labels[params['line'][1]]}, in frame {params['frame']}, in metres.")
        gt = {"type": "Point", "position_m": point, "frame": params["frame"]}
        calculation = {"line_parameter": fraction, "projection_m": projected, "geometry": params["geometry"]}
    elif name == "DetectSceneChange":
        displacements = {o: math.dist(positions[o], current.positions_m[o]) for o in params["objects"]}
        threshold, margin = params["change_threshold_m"], params["ambiguity_tolerance_m"]
        _check(all(abs(d - threshold) > margin for d in displacements.values()), "Ambiguous change at threshold")
        changed = [o for o, d in displacements.items() if d > threshold]
        _check(len(changed) == 1, f"Expected exactly one changed object; found {len(changed)}")
        instruction = (f"Select the one object among {', '.join(labels[o] for o in params['objects'])} "
                       f"whose centre moved more than {threshold} metres since the end of phase "
                       f"{params['reference']['phase']}, in frame {params['frame']}.")
        gt = {"type": "Object", "object_id": changed[0]}
        calculation = {"displacements_m": displacements, "change_threshold_m": threshold}
    else:
        target = params["query_object"]
        instruction = (f"Report the 3D location of {labels[target]} at the end of phase {params['reference']['phase']} "
                       f"in frame {params['frame']}, in metres.")
        gt = {"type": "Point", "position_m": list(positions[target]), "frame": params["frame"]}
        calculation = {"query_object": target}
    if any(o.get("physical_asset") for o in resolved.layout["objects"]):
        instruction = instruction.replace("centres", "reference points").replace("centre", "reference point")
        instruction += " Each object's reference point is its configured rigid-body origin."
        calculation["object_reference_points"] = {
            o["id"]: o.get("physical_asset", {}).get("reference_point", "GeometricCenter")
            for o in resolved.layout["objects"]}
    output = deepcopy({
        "episode_id": resolved.episode["episode_id"], "task_id": name,
        "domain": resolved.task["domain"], "instruction": instruction, "ground_truth": gt,
        "provenance": {"contract_hash": resolved.contract_hash, "source_hashes": hashes,
                       "instance_id": current.instance_id, "reference": reference,
                       "evidence_step": evidence.step, "query_step": current.step,
                       "evidence_hash": _fingerprint(evidence.to_dict()),
                       "current_state_hash": _fingerprint(current.to_dict()),
                       "parameters": params, "calculation": calculation},
    })
    _fingerprint(output)  # No nonfinite GT or diagnostic may escape validation.
    return output
