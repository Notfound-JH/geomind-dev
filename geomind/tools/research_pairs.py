"""Bounded CPU counterfactual checks for displacement-based DetectSceneChange.

No model, physics, network or credential access. The independent oracle uses
configuration states/events, never the production solve() function.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys

from geomind.episodes import EpisodeResolver, SemanticEpisode
from geomind.episodes.resolver import ContractError, load_json


DEFINITION = "DetectSceneChange/displacement-counterfactual-v1"
OBS_KEYS = {"instruction", "phase", "step", "scene_visible", "objects", "frame", "allow_response"}
OBJECT_KEYS = {"object_id", "label", "position_m"}
REPO = Path(__file__).resolve().parents[2]


class ValidationError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


def require(condition, code: str, detail: str):
    if not condition:
        raise ValidationError(code, detail)


def encoded(value) -> bytes:
    """Preserve list AND mapping insertion order for public-payload equality."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def public_observation(obs: dict) -> dict:
    """Reject unexpected metadata; return detached JSON, not an environment."""
    require(set(obs) == OBS_KEYS, "payload_fields", "Unexpected public observation fields")
    require(isinstance(obs["objects"], list), "payload_fields", "objects must be a list")
    for obj in obs["objects"]:
        require(set(obj) == OBJECT_KEYS, "payload_fields", "Unexpected object fields")
    require(obs["scene_visible"] or not obs["objects"], "hidden_objects", "Hidden scene exposes objects")
    return json.loads(encoded(obs))


def condition_payload(stream: list[dict], condition: str) -> dict:
    """Transport-neutral request DATA only. No service or model is implemented."""
    require(condition in {"full-history", "final-only"}, "condition", "Unknown input condition")
    clean = [public_observation(obs) for obs in stream]
    require(bool(clean) and clean[-1]["allow_response"], "query_missing", "Stream must end at query")
    return {"observations": clean if condition == "full-history" else clean[-1:]}


def independent_case(resolved) -> dict:
    """Reconstruct reference/final positions from files; no production GT read."""
    ep, layout, scene, protocol = resolved.episode, resolved.layout, resolved.scene, resolved.protocol
    require(ep["task_id"] == "DetectSceneChange", "task", "Only displacement detection is supported")
    require(layout["randomization"]["position_jitter_m"] == 0, "jitter", "This paired protocol requires zero jitter")
    ids = [obj["id"] for obj in layout["objects"]]
    require(set(ids) == set(ep["parameters"]["objects"]), "candidate_set", "All objects must be candidates")
    require(len(ids) == 3, "object_count", "This experiment uses three objects")
    phases = protocol["phases"]
    lengths = [phases[i]["duration_steps"] for i in (0, 1)]
    require(sum(lengths) <= 100, "step_budget", "CPU protocol exceeds 100 control steps")
    initial = {o["id"]: list(o["position_m"]) for o in layout["objects"]}
    final = deepcopy(initial)
    bounds = scene["workspace"]["bounds_m"]

    def check_positions(positions):
        for pos in positions.values():
            require(len(pos) == 3 and all(math.isfinite(v) for v in pos), "coordinates", "Nonfinite/invalid position")
            require(all(bounds[axis][0] <= pos[i] <= bounds[axis][1]
                        for i, axis in enumerate(("x", "y", "z"))), "workspace", "Position outside workspace")

    check_positions(initial)
    for event in sorted(phases[1].get("events", []), key=lambda e: e["after_steps"]):
        require(event["type"] == "MoveObject", "event", "Addition/removal is outside this definition")
        require(event["object_id"] in final and 1 <= event["after_steps"] <= lengths[1], "event", "Invalid event")
        final[event["object_id"]] = list(event["position_m"])
        check_positions(final)
    threshold = ep["parameters"]["change_threshold_m"]
    margin = ep["parameters"]["ambiguity_tolerance_m"]
    require(threshold > margin > 0, "threshold", "Require threshold > margin > 0")
    # Explicit distance calculation independent of resolver.solve's branch.
    distances = {o: math.sqrt(sum((final[o][i] - initial[o][i]) ** 2 for i in range(3))) for o in ids}
    require(all(abs(d - threshold) > margin for d in distances.values()),
            "threshold_boundary", "A displacement is within the ambiguity band")
    changed = [o for o in ids if distances[o] > threshold]
    require(len(changed) == 1, "changed_count", f"Expected one changed object, got {len(changed)}")
    require(len(phases[1].get("events", [])) == 1, "event_count", "Paired experiment requires one event")
    return {"initial": initial, "final": final, "distances_m": distances, "answer": changed[0],
            "amplitude_m": distances[changed[0]], "durations": lengths,
            "event_steps": [e["after_steps"] for e in phases[1]["events"]]}


def open_query(resolved):
    env = SemanticEpisode(resolved)
    try:
        obs, _ = env.reset()
        stream = [public_observation(obs)]
        for _ in range(100):
            if obs["allow_response"]:
                return env, stream
            obs, _, terminated, truncated, info = env.step()
            require(not terminated and not truncated and not info, "early_result", "Evaluation exposed before query")
            stream.append(public_observation(obs))
        raise ValidationError("step_budget", "Query not reached within CPU budget")
    except Exception:
        env.close()
        raise


def validate_case(root: Path, episode_id: str) -> dict:
    resolved = EpisodeResolver(root).resolve(episode_id)  # production contract, no solve
    expected = independent_case(resolved)
    env, stream = open_query(resolved)
    try:
        require(env.result["ground_truth"]["object_id"] == expected["answer"],
                "production_gt_mismatch", "Independent answer differs from production GT")
        require(env.positions == expected["final"], "runtime_state", "Runtime final state differs from reconstruction")
        ref = env.history["observe.End"]
        require({k: list(v) for k, v in ref.positions_m.items()} == expected["initial"],
                "runtime_history", "Runtime reference differs from initial state")
        _, reward, done, truncated, grading = env.step({"object_id": expected["answer"]})
        require(done and not truncated and reward == 1.0 and grading["success"],
                "scoring", "Correct independent answer was rejected")
    finally:
        env.close()
    rejected = []
    for wrong in expected["final"]:
        if wrong == expected["answer"]:
            continue
        negative_env, _ = open_query(resolved)
        try:
            _, reward, done, truncated, result = negative_env.step({"object_id": wrong})
            require(done and not truncated and reward == 0.0 and not result["success"],
                    "scoring", "Wrong answer was accepted")
            rejected.append(wrong)
        finally:
            negative_env.close()
    return {"episode_id": episode_id, **expected, "stream": stream,
            "contract_hash": resolved.contract_hash, "source_hashes": resolved.source_hashes,
            "correct_answer_accepted": True, "wrong_answers_rejected": rejected,
            "parameters": resolved.episode["parameters"], "scene": resolved.scene,
            "seed": resolved.episode["seed"], "object_order": [o["id"] for o in resolved.layout["objects"]]}


def validate_pair(root: Path, pair: dict) -> dict:
    a, b = (validate_case(root, pair[side]) for side in ("a", "b"))
    require(a["answer"] != b["answer"], "same_answer", "Counterfactual pair has identical labels")
    require(a["initial"] != b["initial"], "same_history", "Reference histories must differ")
    require(encoded(a["stream"][-1]) == encoded(b["stream"][-1]),
            "different_final_observation", "Final public observations differ (including order/text/step)")
    for key in ("parameters", "scene", "durations", "event_steps", "seed", "object_order"):
        require(a[key] == b[key], "unmatched_controls", f"Pair control differs: {key}")
    require(math.isclose(a["amplitude_m"], b["amplitude_m"], abs_tol=1e-12, rel_tol=0),
            "unmatched_amplitude", "Movement amplitudes differ")
    require(len(a["stream"]) == len(b["stream"]), "unmatched_controls", "Stream lengths differ")
    for stream in (a["stream"], b["stream"]):
        condition_payload(stream, "full-history")
        condition_payload(stream, "final-only")
    # A fixed stateless final-only rule receives identical data on both sides.
    prediction = a["stream"][-1]["objects"][0]["object_id"]
    hits = int(prediction == a["answer"]) + int(prediction == b["answer"])
    require(hits <= 1, "final_only_invariant", "Same-input deterministic rule exceeded pair bound")
    return {"pair_id": pair["pair_id"], "passed": True, "a": a, "b": b,
            "final_observation_sha256": digest(encoded(a["stream"][-1])),
            "final_only_rule": {"name": "first-visible-object-stateless", "prediction": prediction,
                                "correct_count": hits, "maximum_correct_per_pair": 1,
                                "is_model_experiment": False}}


def generate(root: Path, pairs: int, seed: int) -> dict:
    require(type(pairs) is int and pairs > 0 and pairs % 6 == 0, "pairs", "pairs must be a positive multiple of six")
    require(type(seed) is int and seed >= 0, "seed", "seed must be a nonnegative integer")
    require(not root.exists(), "output_exists", "Config root already exists; refusing overwrite")
    root.mkdir(parents=True)
    task = load_json(REPO / "geomind/configs/task_definitions/DetectSceneChange.json")
    write(root / "task_definitions/DetectSceneChange.json", task)
    scene = {"scene_id": "research_table", "usd_path": "assets/stage/65/65_new.usda", "robot": "franka",
             "workspace": {"frame": "table", "axes": {"right": "+x", "forward": "+y", "up": "+z"},
                           "bounds_m": {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0, 0.5]}},
             "cameras": ["table_cam"]}
    write(root / "scenes/research_table.json", scene)
    rng = random.Random(seed)
    ids = ["object_0", "object_1", "object_2"]
    assignments = list(itertools.permutations(ids, 2)) * (pairs // 6)
    rng.shuffle(assignments)
    records = []
    for index, (u, v) in enumerate(assignments):
        order = rng.sample(ids, len(ids))
        final = {o: [round(-0.2 + i * 0.2, 6), round(rng.uniform(-0.1, 0.1), 6), 0.04]
                 for i, o in enumerate(rng.sample(ids, len(ids)))}
        angle = rng.uniform(-math.pi, math.pi)
        amplitude = rng.choice([0.12, 0.15])
        delta = [amplitude * math.cos(angle), amplitude * math.sin(angle), 0.0]
        record = {"pair_id": f"pair_{index:03d}"}
        for side, moved in (("a", u), ("b", v)):
            identity = f"cf_{index:03d}_{side}"
            record[side] = identity
            initial = deepcopy(final)
            initial[moved] = [final[moved][j] - delta[j] for j in range(3)]
            write(root / "layouts" / f"{identity}.json", {
                "layout_id": identity, "frame": "table",
                "objects": [{"id": o, "label": o.replace("_", " "), "asset": "red_block", "position_m": initial[o]}
                            for o in order], "randomization": {"position_jitter_m": 0.0}})
            write(root / "protocols" / f"{identity}.json", {
                "protocol_id": identity, "query_at": "recall", "allow_history_replay": False,
                "phases": [{"id": "observe", "type": "Observe", "duration_steps": 2, "scene_visible": True},
                           {"id": "delay", "type": "Delay", "duration_steps": 3, "scene_visible": False,
                            "events": [{"type": "MoveObject", "object_id": moved, "after_steps": 1,
                                        "position_m": final[moved]}]},
                           {"id": "recall", "type": "Respond", "scene_visible": True, "allow_response": True}]})
            write(root / "episodes" / f"{identity}.json", {
                "schema_version": 3, "release_version": "2.0", "episode_id": identity, "task_id": "DetectSceneChange",
                "scene_id": "research_table", "layout_id": identity, "protocol_id": identity,
                "parameters": {"objects": order, "frame": "table", "reference": {"phase": "observe", "boundary": "End"},
                               "change_threshold_m": 0.03, "ambiguity_tolerance_m": 0.001},
                "response": "SelectObject", "seed": seed + index})
        records.append(record)
    presentation = [p[side] for p in records for side in ("a", "b")]
    rng.shuffle(presentation)
    manifest = {"definition": DEFINITION, "seed": seed, "pairs": records, "presentation_order": presentation,
                "evaluator_only": True, "scene_loaded": False,
                "assignment_method": "All six ordered distinct target pairs repeated equally, then shuffled with random.Random(seed); object order and episode presentation independently shuffled by the same recorded PRNG stream."}
    write(root.parent / "pair_index.json", manifest)
    return manifest


def failure_record(exc: Exception) -> dict:
    implementation_codes = {"production_gt_mismatch", "runtime_state", "runtime_history", "scoring", "early_result"}
    if isinstance(exc, ValidationError):
        return {"category": "implementation_failure" if exc.code in implementation_codes else "invalid_instance",
                "code": exc.code, "reason": str(exc)}
    if isinstance(exc, ContractError):
        return {"category": "invalid_instance", "code": "contract", "reason": str(exc)}
    return {"category": "system_failure", "code": type(exc).__name__, "reason": str(exc)}


def validate(root: Path, index: dict, report_dir: Path) -> dict:
    results, failures, counts_a, counts_b = [], [], Counter(), Counter()
    for pair in index["pairs"]:
        try:
            result = validate_pair(root, pair)
            counts_a[result["a"]["answer"]] += 1
            counts_b[result["b"]["answer"]] += 1
            for side in ("a", "b"):
                case = result[side]
                for condition in ("full-history", "final-only"):
                    write(report_dir / "payload_candidates" / condition / f"{case['episode_id']}.json",
                          condition_payload(case["stream"], condition))
            write(report_dir / "pairs" / f"{pair['pair_id']}.json", result)
            results.append(result)
        except Exception as exc:
            failures.append({"pair_id": pair.get("pair_id"), **failure_record(exc)})
    episodes = [p[side] for p in index["pairs"] for side in ("a", "b")]
    order_valid = len(set(episodes)) == len(episodes) and Counter(index["presentation_order"]) == Counter(episodes)
    balanced = (len(counts_a) == len(counts_b) == 3 and len(set(counts_a.values())) == len(set(counts_b.values())) == 1)
    report = {"definition": DEFINITION, "backend": "semantic_cpu", "model_evaluated": False,
              "model_results": None, "physics_executed": False,
              "pairs_total": len(index["pairs"]), "pairs_passed": len(results), "failures": failures,
              "presentation_is_permutation": order_valid, "balanced_targets": balanced,
              "targets_a": dict(counts_a), "targets_b": dict(counts_b),
              "correct_answer_checks": 2 * len(results), "wrong_answer_checks": 4 * len(results),
              "fixed_final_only_rule_correct": sum(r["final_only_rule"]["correct_count"] for r in results),
              "fixed_final_only_rule_is_model_experiment": False,
              "passed": not failures and bool(results) and order_valid and balanced}
    write(report_dir / "validation.json", report)
    return report


def failure_examples(root: Path, pair: dict, output: Path) -> dict:
    """Materialize reproducible, intentionally invalid pairs; never alter originals."""
    expected_codes = {"same_answer": "same_answer", "different_final": "different_final_observation",
                      "multiple_changes": "changed_count", "threshold_boundary": "threshold_boundary",
                      "outside_workspace": "contract", "protocol_conflict": "contract"}
    outcomes = []
    for name, expected_code in expected_codes.items():
        dest = output / name / "configs"
        dest.mkdir(parents=True, exist_ok=False)
        for folder in ("task_definitions", "scenes"):
            shutil.copytree(root / folder, dest / folder)
        for folder in ("episodes", "layouts", "protocols"):
            for side in ("a", "b"):
                source = root / folder / f"{pair[side]}.json"
                target = dest / folder / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        identity = pair["b"]
        lp, pp = dest / "layouts" / f"{identity}.json", dest / "protocols" / f"{identity}.json"
        layout, protocol = load_json(lp), load_json(pp)
        events = protocol["phases"][1]["events"]
        moved = events[0]["object_id"]
        other = next(o for o in layout["objects"] if o["id"] != moved)
        if name == "same_answer":
            a_layout = load_json(dest / "layouts" / f"{pair['a']}.json")
            a_protocol = load_json(dest / "protocols" / f"{pair['a']}.json")
            layout["objects"] = a_layout["objects"]
            protocol["phases"] = a_protocol["phases"]
        elif name == "different_final":
            other["position_m"][2] += 0.01  # Still one moved object; final public state changes.
        elif name == "multiple_changes":
            position = list(other["position_m"]); position[1] += 0.1
            events.append({"type": "MoveObject", "object_id": other["id"], "after_steps": 1, "position_m": position})
        elif name == "threshold_boundary":
            obj = next(o for o in layout["objects"] if o["id"] == moved)
            obj["position_m"] = list(events[0]["position_m"])
            obj["position_m"][0] += 0.0305
        elif name == "outside_workspace":
            other["position_m"][0] = 5.0
        elif name == "protocol_conflict":
            events.append(deepcopy(events[0]))
        write(lp, layout); write(pp, protocol)
        write(output / name / "pair_index.json", {"pairs": [pair], "evaluator_only": True})
        try:
            validate_pair(dest, pair)
            actual = {"category": "accepted", "code": "accepted", "reason": "Invalid pair was accepted"}
        except Exception as exc:
            actual = failure_record(exc)
        record = {"case": name, "expected_code": expected_code, **actual,
                  "rejected_as_expected": actual["code"] == expected_code}
        write(output / name / "rejection.json", record)
        outcomes.append(record)
    summary = {"passed": all(o["rejected_as_expected"] for o in outcomes), "cases": outcomes}
    write(output / "summary.json", summary)
    return summary


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def snapshot(run: Path, command: list[str], *, index_filename: str = "pair_index.json", extra_paths=()):
    """Copy only relevant source/config/skill files, including untracked code."""
    paths = set((REPO / "geomind/episodes").glob("*.py"))
    # Import-time dependencies actually loaded from the two project packages.
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file:
            path = Path(file).resolve()
            if path.is_relative_to(REPO) and path.suffix == ".py" and path.relative_to(REPO).parts[0] in {"geomind", "geomind_lab"}:
                paths.add(path)
    paths.update([Path(__file__).resolve(), REPO / "tests/test_research_pairs.py",
                  REPO / "tests/test_spatial_episode_pipeline.py", REPO / "tests/test_episode_contract_v3.py",
                  REPO / "pyproject.toml", REPO / "geomind/configs/task_definitions/DetectSceneChange.json"])
    paths.update(REPO / path for path in extra_paths)
    skill = REPO / ".agents/skills/geomind-research"
    paths.update(p for p in skill.rglob("*") if p.is_file() and p.suffix in {".md", ".yaml"})
    files = []
    for path in sorted(paths):
        relative = path.relative_to(REPO).as_posix()
        data = path.read_bytes()
        dest = run / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        files.append({"path": relative, "sha256": digest(data),
                      "git_status": git("status", "--porcelain=v1", "--", relative) or "clean"})
    config_files = {p.relative_to(run).as_posix(): digest(p.read_bytes()) for p in sorted((run / "configs").rglob("*.json"))}
    config_files[index_filename] = digest((run / index_filename).read_bytes())
    import gymnasium
    import numpy
    metadata = {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
                "dirty": bool(git("status", "--porcelain=v1")), "source_files": files,
                "source_snapshot_id": digest(encoded({f["path"]: f["sha256"] for f in files})),
                "config_files": config_files, "configuration_id": digest(encoded(config_files)),
                "command_argv": command, "working_directory": "repository root",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "numpy": numpy.__version__, "gymnasium": gymnasium.__version__,
                "collection_scope": "Relevant loaded project Python sources, episodes implementation-hash inputs, helper/test, packaging metadata, one public task definition and skill; no environment dump, credentials, assets, weights or unrelated datasets."}
    write(run / "provenance.json", metadata)
    return metadata


def check_provenance(run: Path) -> dict:
    metadata = load_json(run / "provenance.json")
    mismatches = []
    if digest(encoded({f["path"]: f["sha256"] for f in metadata["source_files"]})) != metadata["source_snapshot_id"]:
        mismatches.append("source_manifest")
    if digest(encoded(metadata["config_files"])) != metadata["configuration_id"]:
        mismatches.append("config_manifest")
    for item in metadata["source_files"]:
        relative = Path(item["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "snapshot_path", "Invalid source path")
        for label, base in (("current_source", REPO), ("saved_source", run / "source_snapshot")):
            path = base / relative
            if not path.is_file() or digest(path.read_bytes()) != item["sha256"]:
                mismatches.append(f"{label}:{relative}")
    for relative, expected in metadata["config_files"].items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "snapshot_path", "Invalid config path")
        if not (run / path).is_file() or digest((run / path).read_bytes()) != expected:
            mismatches.append(f"config:{relative}")
    return {"passed": not mismatches, "mismatches": mismatches,
            "source_snapshot_id": metadata["source_snapshot_id"], "configuration_id": metadata["configuration_id"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("generate")
    create.add_argument("--output", type=Path, required=True, help="New independent experiment directory")
    create.add_argument("--pairs", type=int, default=6)
    create.add_argument("--seed", type=int, default=20261107)
    create.add_argument("--failure-examples", action="store_true")
    verify = commands.add_parser("validate")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--report-dir", type=Path, required=True, help="New directory, never overwrite a prior report")
    single = commands.add_parser("check-pair", help="Replay a saved valid or intentionally invalid pair")
    single.add_argument("--case", type=Path, required=True, help="Directory containing configs/ and pair_index.json")
    single.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    command = ["python", "-m", "geomind.tools.research_pairs", *(sys.argv[1:] if argv is None else argv)]
    report_dir = None
    owns_report = False
    try:
        if args.command == "check-pair":
            report_dir = args.report_dir.resolve()
            require(not report_dir.exists(), "output_exists", "Report directory already exists")
            report_dir.mkdir(parents=True)
            owns_report = True
            pairs = load_json(args.case / "pair_index.json")["pairs"]
            require(len(pairs) == 1, "pair_count", "check-pair requires exactly one pair")
            result = validate_pair(args.case / "configs", pairs[0])
            write(report_dir / "validation.json", result)
            print(json.dumps({"passed": True, "pair_id": result["pair_id"]}))
            return 0
        if args.command == "generate":
            run = args.output.resolve()
            require(not run.exists(), "output_exists", "Output already exists; choose a new directory")
            require(not run.is_relative_to(REPO / "geomind/configs"), "protected_output", "Do not write formal configs")
            index = generate(run / "configs", args.pairs, args.seed)
            provenance = snapshot(run, command)
            report_dir = run / "reports"
        else:
            run = args.run.resolve()
            report_dir = args.report_dir.resolve()
            require(not report_dir.exists(), "output_exists", "Report directory already exists")
            integrity = check_provenance(run)
            require(integrity["passed"], "provenance_mismatch", json.dumps(integrity))
            provenance = load_json(run / "provenance.json")
            index = load_json(run / "pair_index.json")
        report_dir.mkdir(parents=True, exist_ok=False)
        owns_report = True
        report = validate(run / "configs", index, report_dir)
        if args.command == "generate" and args.failure_examples:
            negatives = failure_examples(run / "configs", index["pairs"][0], report_dir / "failure_cases")
            report["failure_examples_passed"] = negatives["passed"]
            report["passed"] = report["passed"] and negatives["passed"]
        report.update({"git_commit": provenance["git_commit"], "source_snapshot_id": provenance["source_snapshot_id"],
                       "configuration_id": provenance["configuration_id"], "command_argv": command})
        write(report_dir / "validation.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    except Exception as exc:
        failure = {"passed": False, **failure_record(exc), "command_argv": command}
        # Never touch an existing output on refusal. For new runs retain partial evidence.
        if report_dir is not None and (owns_report or not report_dir.exists()):
            write(report_dir / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
