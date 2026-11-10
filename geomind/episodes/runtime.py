"""CPU semantic backend: geometry, temporal observations, events and answer grading.

This is a symbolic world, not a physics simulator or a robot motion backend.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import random
import uuid

from .resolver import EpisodeResolver, ResolvedEpisode, Snapshot, TASKS, _check, _fields, _id, _position, solve


def evaluate_answer(resolved: ResolvedEpisode, result: dict, answer: dict) -> dict:
    _check(result["provenance"]["contract_hash"] == resolved.contract_hash, "GT configuration mismatch")
    gt = result["ground_truth"]
    if gt["type"] == "Object":
        _fields(answer, {"object_id"})
        _check(isinstance(answer["object_id"], str), "object_id must be a string")
        success = answer["object_id"] == gt["object_id"]
        return {"success": success, "score": float(success)}
    _fields(answer, {"position_m", "frame"})
    _check(answer["frame"] == gt["frame"], "Answer coordinate frame mismatch")
    point = _position(answer["position_m"])
    if gt["type"] == "Relation":
        from .registered_tasks import validate_relation_ground_truth
        validate_relation_ground_truth(resolved, gt)
        bounds = resolved.scene["workspace"]["bounds_m"]
        _check(all(bounds[axis][0] <= point[i] <= bounds[axis][1]
                   for i, axis in enumerate(("x", "y", "z"))), "Relation action command outside declared workspace")
        return {"success": True, "score": None, "command_valid": True,
                "semantic_deferred_to_relation": True, "task_success": None,
                "validation_scope": "finite in-workspace action command only; task success requires measured physical relation"}
    _check(gt["type"] == "Point", "Unsupported GT response type")
    _check("evaluation" in resolved.episode, "Point response requires explicit evaluation.position_tolerance_m")
    tolerance = resolved.episode["evaluation"]["position_tolerance_m"]
    distance = math.dist(point, gt["position_m"])
    return {"success": distance <= tolerance, "distance_m": distance, "tolerance_m": tolerance,
            "score": float(distance <= tolerance)}


class SemanticEpisode:
    """Executable v3 episode with isolated agent observations and evaluator state."""

    def __init__(self, resolved: ResolvedEpisode):
        self.resolved = resolved
        if resolved.episode["response"] == "ReportPoint":
            _check("evaluation" in resolved.episode, "Point response requires explicit evaluation tolerance")
        self.closed = True

    def reset(self):
        rng = random.Random(self.resolved.episode["seed"])
        jitter = self.resolved.layout["randomization"]["position_jitter_m"]
        self.positions = {
            obj["id"]: [v + rng.uniform(-jitter, jitter) for v in obj["position_m"]]
            for obj in self.resolved.layout["objects"]
        }
        bounds = self.resolved.scene["workspace"]["bounds_m"]
        _check(all(bounds[axis][0] <= p[i] <= bounds[axis][1] for p in self.positions.values()
                   for i, axis in enumerate(("x", "y", "z"))), "Randomized position outside workspace")
        self.instance_id = "run_" + uuid.uuid4().hex
        self.step_index = 0
        self.history: dict[str, Snapshot] = {}
        self.result = None
        self.closed = False
        if self.resolved.protocol is None:
            self.result = solve(self.resolved, self.snapshot())
        return self.observation(), {"backend": "semantic", "contract_hash": self.resolved.contract_hash}

    def snapshot(self) -> Snapshot:
        return Snapshot.from_dict({"instance_id": self.instance_id,
            "contract_hash": self.resolved.contract_hash, "frame": self.resolved.layout["frame"],
            "step": self.step_index, "positions_m": deepcopy(self.positions)})

    def phase(self) -> dict:
        if self.resolved.protocol is None:
            return {"id": "query", "scene_visible": True, "allow_response": True}
        observe, delay, respond = self.resolved.protocol["phases"]
        if self.step_index < observe["duration_steps"]:
            return observe
        if self.step_index < observe["duration_steps"] + delay["duration_steps"]:
            return delay
        return respond

    def observation(self) -> dict:
        phase = self.phase()
        objects = []
        if phase["scene_visible"]:
            objects = [{"object_id": obj["id"], "label": obj.get("label", obj["id"]),
                        "position_m": list(self.positions[obj["id"]])} for obj in self.resolved.layout["objects"]]
        # No reference snapshot, event log, GT, provenance or future query leaks.
        if self.result is not None:
            instruction = self.result["instruction"]
        elif phase["type"] == "Observe":
            labels = {o["id"]: o.get("label", o["id"]) for o in self.resolved.layout["objects"]}
            instruction = "Remember the locations of " + ", ".join(
                labels[o] for o in self.resolved.episode["parameters"]["objects"]) + "."
        else:
            instruction = "Wait for the query."
        return {"instruction": instruction, "phase": phase["id"], "step": self.step_index,
                "scene_visible": phase["scene_visible"], "objects": objects,
                "frame": self.resolved.layout["frame"],
                "allow_response": phase.get("allow_response", False)}

    def step(self, answer: dict | None = None):
        _check(not self.closed, "Call reset before stepping; episode may already be finished")
        if self.phase().get("allow_response", False):
            _check(answer is not None, "A response is required in the response phase")
            evaluation = evaluate_answer(self.resolved, self.result, answer)
            self.closed = True
            if evaluation.get("semantic_deferred_to_relation"):
                # CPU validates a command but cannot execute release/contact or
                # observe the resulting relation. Do not report task success.
                evaluation = dict(evaluation, success=None, task_success=None,
                                  task_completed=False, physical_execution=False)
                return self.observation(), 0.0, True, False, evaluation
            return self.observation(), evaluation["score"], True, False, evaluation
        _check(answer is None, "Responses are not allowed before retrieval")
        self.step_index += 1
        observe, delay, _ = self.resolved.protocol["phases"]
        end = observe["duration_steps"]
        if self.step_index == end:
            self.history[f"{observe['id']}.End"] = self.snapshot()
        for event in delay.get("events", []):
            if self.step_index == end + event["after_steps"]:
                self.positions[event["object_id"]] = list(event["position_m"])
        if self.phase().get("allow_response", False):
            self.result = solve(self.resolved, self.snapshot(), history=self.history)
        return self.observation(), 0.0, False, False, {}

    def export_evidence(self) -> dict:
        """Privileged backend export for offline audit, not an agent observation."""
        return {"backend": "semantic", "current": self.snapshot().to_dict(),
                "history": {key: state.to_dict() for key, state in self.history.items()}}

    def close(self):
        self.closed = True


def create_episode_env(episode_id: str, *, root: str | Path | None = None, backend: str = "semantic",
                       device: str = "cuda:0", headless: bool = True) -> SemanticEpisode:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "configs"
    _id(episode_id)
    if (episode_id in TASKS and not (root / "episodes" / f"{episode_id}.json").exists()
            and (root / "episodes" / f"{episode_id}_v0.json").is_file()):
        episode_id += "_v0"
    resolved = EpisodeResolver(root).resolve(episode_id)
    if backend == "semantic":
        return SemanticEpisode(resolved)
    if backend == "isaaclab":
        from .isaaclab_physics import IsaacLabEpisode
        return IsaacLabEpisode(resolved, device=device, headless=headless)
    raise ValueError(f"Unsupported native episode backend: {backend}")
