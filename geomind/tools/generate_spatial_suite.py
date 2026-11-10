"""Generate and independently verify the original three native tasks on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

from geomind.core.registry import registry
from geomind.episodes import EpisodeResolver, SemanticEpisode, evaluate_answer, solve
from geomind.episodes.resolver import _check
from geomind.episodes.physical_contract import physics_profile, execution_profile
from .migrate_episode_configs import write


TASK_IDS = ("DetectSceneChange", "ProjectOntoLine", "ReflectAcrossLine")


def generate(root: Path, seeds: list[int], *, scene_id: int | None = None, object_set: str = "blocks") -> dict:
    _check(bool(seeds) and len(set(seeds)) == len(seeds)
           and all(type(s) is int and s >= 0 for s in seeds), "Seeds must be distinct nonnegative integers")
    registration = {}
    for name in TASK_IDS:
        spec = registry.get(name)
        _check(registry.capabilities(name)['semantic'] and registry.capabilities(name)['isaaclab_episode'],
               f"Expected registered native semantic and physical capability: {name}")
        registration[name] = {"registration_id": spec.task_id, "physical_status": spec.implementation_status,
                              "capabilities": registry.capabilities(name)}
        write(root / "task_definitions" / f"{name}.json", {
            "task_id": name, "domain": "SpatialMemory" if name == "DetectSceneChange" else "SpatialReasoning",
            "family": spec.task_family,
        })
    write(root / "scenes/semantic_table.json", {
        "scene_id": "semantic_table", "usd_path": "assets/stage/65/65_new.usda", "robot": "franka",
        "workspace": {"frame": "table", "axes": {"right": "+x", "forward": "+y", "up": "+z"},
                      "bounds_m": {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0, 0.5]}},
        "cameras": ["table_cam"],
        "physics": physics_profile(),
    })
    selected_scene = "semantic_table"
    if scene_id is not None:
        from geomind.episodes.registered_scene import scene_description
        scene = scene_description(scene_id)
        selected_scene = scene["scene_id"]
        write(root / "scenes" / f"{selected_scene}.json", scene)
    records = []
    sampling = []
    for seed in seeds:
        rng = random.Random(seed)
        yaw, pitch = rng.uniform(-math.pi, math.pi), rng.uniform(-0.4, 0.4)
        direction = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
        perpendicular = [-math.sin(yaw), math.cos(yaw), 0.0]
        centre = [rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.2]
        t = rng.uniform(-0.16, 0.16)  # Includes projections outside the segment.
        projection = [centre[i] + t * direction[i] for i in range(3)]
        point = [projection[i] + 0.07 * perpendicular[i] for i in range(3)]
        reflected = [projection[i] - 0.07 * perpendicular[i] for i in range(3)]
        line_a = [centre[i] - 0.1 * direction[i] for i in range(3)]
        line_b = [centre[i] + 0.1 * direction[i] for i in range(3)]
        layout_id = f"geometry_s{seed}"
        write(root / "layouts" / f"{layout_id}.json", {
            "layout_id": layout_id, "frame": "table",
            "objects": [
                {"id": identity, "label": label, "asset": asset, "position_m": position}
                for identity, label, asset, position in (
                    ("line_a", "the red endpoint", "red_block", line_a),
                    ("line_b", "the blue endpoint", "blue_block", line_b),
                    ("point", "the green point", "green_block", point))],
            "randomization": {"position_jitter_m": 0.0},
        })
        for name, expected in (("ProjectOntoLine", projection), ("ReflectAcrossLine", reflected)):
            identity = f"{name}_v0" if seed == seeds[0] else f"{name}_s{seed}_v0"
            write(root / "episodes" / f"{identity}.json", {
                "schema_version": 3, "release_version": "2.0", "episode_id": identity, "task_id": name,
                "scene_id": selected_scene, "layout_id": layout_id,
                "parameters": {"point": "point", "line": ["line_a", "line_b"], "frame": "table",
                               "geometry": "InfiniteLine3D", "degeneracy_tolerance_m": 0.0001},
                "response": "ReportPoint", "evaluation": {"position_tolerance_m": 0.000001}, "seed": seed,
                "physical_execution": execution_profile(),
            })
            records.append({"episode_id": identity, "task_id": name,
                            "expected_answer": {"position_m": expected, "frame": "table"}})
        identity = "DetectSceneChange_v0" if seed == seeds[0] else f"DetectSceneChange_s{seed}_v0"
        layout_id, protocol_id = f"change_s{seed}", f"change_protocol_s{seed}"
        objects = [f"block_{i}" for i in range(3)]
        moved = rng.choice(objects)
        positions = {o: [-0.15 + i * 0.15, rng.uniform(-0.2, 0.2), physics_profile()["object_size_m"][2] / 2]
                     for i, o in enumerate(objects)}
        destination = list(positions[moved])
        rejected = []
        for attempt in range(1, 101):
            destination[1] = rng.uniform(-0.2, 0.2)
            if abs(destination[1] - positions[moved][1]) >= 0.12:
                break
            rejected.append({"attempt": attempt, "destination_y": destination[1], "reason": "displacement below .12m"})
        sampling.append({"seed": seed, "attempts": attempt, "rejected": rejected,
                         "accepted_destination": list(destination), "object_id": moved})
        write(root / "reports/sampling.json", sampling)
        _check(abs(destination[1] - positions[moved][1]) >= 0.12, "Could not sample an unambiguous memory intervention")
        write(root / "layouts" / f"{layout_id}.json", {
            "layout_id": layout_id, "frame": "table", "objects": [
                {"id": o, "label": f"the {color} block", "asset": f"{color}_block", "position_m": positions[o]}
                for o, color in zip(objects, ("red", "blue", "green"))],
            "randomization": {"position_jitter_m": 0.0},
        })
        write(root / "protocols" / f"{protocol_id}.json", {
            "protocol_id": protocol_id, "query_at": "recall", "allow_history_replay": False,
            "phases": [
                {"id": "observe", "type": "Observe", "duration_steps": 2, "scene_visible": True},
                {"id": "delay", "type": "Delay", "duration_steps": 3, "scene_visible": False,
                 "events": [{"type": "MoveObject", "object_id": moved, "after_steps": 1, "position_m": destination}]},
                {"id": "recall", "type": "Respond", "scene_visible": True, "allow_response": True}],
        })
        write(root / "episodes" / f"{identity}.json", {
            "schema_version": 3, "release_version": "2.0", "episode_id": identity, "task_id": "DetectSceneChange",
            "scene_id": selected_scene, "layout_id": layout_id, "protocol_id": protocol_id,
            "parameters": {"objects": objects, "frame": "table", "reference": {"phase": "observe", "boundary": "End"},
                           "change_threshold_m": 0.03, "ambiguity_tolerance_m": 0.001},
            "response": "SelectObject", "seed": seed,
            "physical_execution": execution_profile(),
        })
        records.append({"episode_id": identity, "task_id": "DetectSceneChange",
                        "expected_answer": {"object_id": moved}})
    if object_set == "catalog":
        from geomind.episodes.object_assets import catalog_descriptor
        assets = [("blue_can", "the coffee can", 0.45), ("cube_can", "the meat tin", 0.6),
                  ("yellow_bottle", "the yellow mustard bottle", 0.4)]
        for seed in seeds:
            for kind in ("geometry", "change"):
                path = root / "layouts" / f"{kind}_s{seed}.json"
                layout = json.loads(path.read_text())
                for obj, (asset, label, scale) in zip(layout["objects"], assets):
                    obj.update(asset=asset, label=label, physical_asset=catalog_descriptor(asset, scale))
                    if kind == "change":
                        obj["position_m"][2] = -obj["physical_asset"]["bottom_offset_m"]
                write(path, layout)
                if kind == "change":
                    protocol_path = root / "protocols" / f"change_protocol_s{seed}.json"
                    protocol = json.loads(protocol_path.read_text())
                    event = protocol["phases"][1]["events"][0]
                    event["position_m"][2] = next(o["position_m"][2] for o in layout["objects"] if o["id"] == event["object_id"])
                    write(protocol_path, protocol)
    manifest = {"suite_id": "generated_spatial_suite", "backend": "semantic",
                "generator_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "seeds": seeds, "sampling": sampling, "registration_before_generation": registration, "cases": records}
    write(root / "suites/generated_spatial_suite.json", manifest)
    for domain in ("SpatialReasoning", "SpatialMemory"):
        selected = [case["episode_id"] for case in records
                    if (case["task_id"] == "DetectSceneChange") == (domain == "SpatialMemory")]
        suffix = "spatial_memory" if domain == "SpatialMemory" else "spatial_reasoning"
        write(root / "suites" / f"generated_{suffix}.json", {
            "suite_id": f"generated_{suffix}", "domain": domain, "backend": "semantic",
            "episodes": selected,
        })
    return manifest


def verify(root: Path, manifest: dict) -> dict:
    resolver, results = EpisodeResolver(root), []
    for case in manifest["cases"]:
        resolved = resolver.resolve(case["episode_id"])
        env = SemanticEpisode(resolved)
        obs, _ = env.reset()
        if resolved.protocol is not None:
            _check(not obs["allow_response"] and "moved" not in obs["instruction"], "Query leaked during encoding")
            while not obs["allow_response"]:
                obs, _, done, _, info = env.step()
                _check(not done and "ground_truth" not in info, "Evaluation leaked during observation")
                if not obs["scene_visible"]:
                    _check(obs["objects"] == [], "Hidden scene leaked positions")
        expected = case["expected_answer"]
        grading = evaluate_answer(resolved, env.result, expected)
        _check(grading["success"], f"Independent construction oracle disagrees: {case['episode_id']}")
        if "position_m" in expected:
            error = math.dist(expected["position_m"], env.result["ground_truth"]["position_m"])
            _check(error < 1e-12, "Geometry GT disagrees with construction oracle")
            wrong = {"position_m": [v + 0.01 for v in expected["position_m"]], "frame": "table"}
        else:
            error = 0.0
            wrong = {"object_id": next(o["id"] for o in resolved.layout["objects"] if o["id"] != expected["object_id"])}
        _check(not evaluate_answer(resolved, env.result, wrong)["success"], "Incorrect answer accepted")
        replay = solve(resolved, env.snapshot(), history=env.history)
        _check(replay == env.result, "Offline evidence replay disagrees with runtime")
        _, reward, done, _, evaluation = env.step(expected)
        _check(done and reward == 1.0 and evaluation["success"], "End-to-end response grading failed")
        results.append({"episode_id": case["episode_id"], "task_id": case["task_id"],
                        "domain": resolved.task["domain"],
                        "configuration_valid": True, "oracle_error_m": error, "incorrect_answer_rejected": True,
                        "evidence_replay_equal": True, "end_to_end_success": True,
                        "instruction": env.result["instruction"], "ground_truth": env.result["ground_truth"],
                        "provenance": env.result["provenance"]})
        write(root / "evidence" / f"{case['episode_id']}.json", env.export_evidence())
    report = {"backend": "semantic", "physical_execution_verified": False,
              "python_version": sys.version.split()[0], "generator_hash": manifest["generator_hash"],
              "verified_cases": len(results), "passed": True,
              "by_domain": {domain: sum(r["domain"] == domain for r in results)
                            for domain in ("SpatialReasoning", "SpatialMemory")},
              "results": results}
    write(root / "reports/generated_spatial_suite.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--scene-id", type=int, help="Reuse a registered GeoMind scene, e.g. 65")
    parser.add_argument("--object-set", choices=["blocks", "catalog"], default="blocks")
    args = parser.parse_args()
    manifest = generate(args.root, args.seeds, scene_id=args.scene_id, object_set=args.object_set)
    if args.verify:
        report = verify(args.root, manifest)
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    else:
        print(json.dumps({"generated_cases": len(manifest["cases"]), "verified": False}, indent=2))


if __name__ == "__main__":
    main()
