"""Small CPU static-query and nonzero-z projection diagnostics. No model/physics."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

from geomind.episodes import ContractError, EpisodeResolver, SemanticEpisode
from geomind.episodes.resolver import load_json
from geomind.tools.research_pairs import REPO, public_observation, require, snapshot, write


def generate(root: Path, seed: int) -> dict:
    require(not root.exists(), "output_exists", "Never overwrite configs")
    require(seed >= 0, "seed", "Seed must be nonnegative")
    baseline = EpisodeResolver(REPO / "geomind/configs").resolve("ProjectOntoLine_v0")
    for folder, identity, value in (
        ("episodes", baseline.episode["episode_id"], baseline.episode),
        ("task_definitions", "ProjectOntoLine", baseline.task),
        ("scenes", baseline.scene["scene_id"], baseline.scene),
        ("layouts", baseline.layout["layout_id"], baseline.layout),
    ):
        write(root / folder / f"{identity}.json", value)
    positions = {o["id"]: np.asarray(o["position_m"]) for o in baseline.layout["objects"]}
    a, b, p = (positions[k] for k in ("line_a", "line_b", "point"))
    # Independent numeric least-squares baseline; not production solve.
    t = np.linalg.lstsq((b-a).reshape(3, 1), p-a, rcond=None)[0][0]
    cases = [{"episode_id": "ProjectOntoLine_v0", "kind": "current_config",
              "expected": (a+t*(b-a)).tolist()}]
    scene = load_json(REPO / "geomind/configs/scenes/semantic_table.json")
    scene.pop("physics", None)
    write(root / "scenes/semantic_table.json", scene)
    rng = random.Random(seed)
    for index, (pitch, along) in enumerate(zip((0.2, -0.35, 0.6), (-0.14, 0.0, 0.14))):
        yaw = rng.uniform(-math.pi, math.pi)
        d = np.array([math.cos(pitch)*math.cos(yaw), math.cos(pitch)*math.sin(yaw), math.sin(pitch)])
        n_a = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        n_b = np.cross(d, n_a)
        centre = np.array([0.0, 0.0, 0.25])
        q = centre + along*d
        for side, n in (("a", n_a), ("b", n_b)):
            identity = f"projection_{index}_{side}"
            points = {"line_a": centre-0.1*d, "line_b": centre+0.1*d, "point": q+0.07*n}
            ep = deepcopy(baseline.episode)
            ep.update(episode_id=identity, layout_id=identity, scene_id="semantic_table", seed=seed+index)
            ep.pop("physical_execution", None)
            layout = {"layout_id": identity, "frame": "table", "randomization": {"position_jitter_m": 0.0},
                      "objects": [{"id": k, "label": k, "asset": "red_block", "position_m": v.tolist()}
                                  for k,v in points.items()]}
            write(root / "episodes" / f"{identity}.json", ep)
            write(root / "layouts" / f"{identity}.json", layout)
            cases.append({"episode_id": identity, "pair": index, "side": side, "kind": "diagnostic",
                          "expected": q.tolist(), "normal": n.tolist(), "offset_m": 0.07})
    manifest = {"seed": seed, "definition": "ProjectOntoLine/structured-nonzero-z-v1", "cases": cases,
                "model_evaluated": False, "physics_executed": False}
    write(root.parent / "case_index.json", manifest)
    return manifest


def verify_case(root: Path, case: dict, output: Path) -> dict:
    resolved = EpisodeResolver(root).resolve(case["episode_id"])
    expected = np.array(case["expected"])
    env = SemanticEpisode(resolved)
    try:
        obs, _ = env.reset()
        obs = public_observation(obs)
        require(obs["step"] == 0 and obs["phase"] == "query" and obs["allow_response"], "static_query", "Not a reset-time query")
        require(not env.history and resolved.protocol is None, "static_history", "Static query unexpectedly needs history")
        pts = {o["object_id"]: np.array(o["position_m"]) for o in obs["objects"]}
        a,b,p = (pts[k] for k in ("line_a","line_b","point"))
        d = b-a
        require(np.linalg.norm(np.cross(expected-a,d)) < 1e-12 and abs(np.dot(p-expected,d)) < 1e-12,
                "independent_geometry", "Expected answer not on line or residual not perpendicular")
        require(math.dist(expected, env.result["ground_truth"]["position_m"]) < 1e-12,
                "production_gt", "Independent construction disagrees with production")
        gt = deepcopy(env.result["ground_truth"])
    finally:
        env.close()
    tau = resolved.episode["evaluation"]["position_tolerance_m"]
    answer = {"position_m": expected.tolist(), "frame": "table"}
    def shifted(distance):
        value = deepcopy(answer); value["position_m"][0] += distance
        return value
    # Exercise actual JSON loading and production answer validation, not a service parser.
    trials = [("correct", json.dumps(answer), True),
              ("inside_tolerance", json.dumps(shifted(0.5*tau)), True),
              ("outside_tolerance", json.dumps(shifted(2*tau)), False),
              ("wrong_point", json.dumps(shifted(0.01)), False),
              ("wrong_frame", json.dumps({**answer, "frame":"world"}), "contract"),
              ("missing_frame", json.dumps({"position_m": expected.tolist()}), "contract"),
              ("extra_field", json.dumps({**answer, "target":True}), "contract"),
              ("bad_dimension", '{"position_m":[0,0],"frame":"table"}', "contract"),
              ("nonfinite", '{"position_m":[NaN,0,0],"frame":"table"}', "contract"),
              ("duplicate_field", '{"frame":"table","frame":"world","position_m":[0,0,0]}', "contract"),
              ("invalid_json", '{', "contract")]
    checks=[]
    for name, raw, wanted in trials:
        path = output / "answers" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(raw+"\n")
        trial = SemanticEpisode(resolved)
        trial.reset()
        try:
            parsed = load_json(path)
            _, reward, done, truncated, result = trial.step(parsed)
            require(wanted != "contract" and result["success"] == wanted and done and not truncated,
                    "grading", f"Unexpected scoring: {name}")
            checks.append({"name":name,"passed":True,"evaluation":result,"reward":reward})
        except ContractError as exc:
            require(wanted == "contract", "grading", f"Unexpected rejection: {name}: {exc}")
            checks.append({"name":name,"passed":True,"contract_rejection":str(exc)})
        finally:
            trial.close()
    shortcut = a + ((p[2]-a[2])/d[2])*d
    result = {"episode_id":case["episode_id"], "observation":obs,"ground_truth":gt,
              "expected":expected.tolist(), "tolerance_m":tau,"checks":checks,
              "z_interpolation_error_m":float(np.linalg.norm(shortcut-expected)),
              "perpendicular_offset_m":float(np.linalg.norm(p-expected)),
              "contract_hash":resolved.contract_hash,"source_hashes":resolved.source_hashes}
    write(output / "result.json", result)
    return result


def verify(root: Path, manifest: dict, output: Path) -> dict:
    records = [verify_case(root,c,output/c["episode_id"]) for c in manifest["cases"]]
    for i in range(3):
        a,b = records[1+2*i:3+2*i]
        require(a["expected"] == b["expected"], "paired_target", "Pair targets differ")
        require(abs(a["perpendicular_offset_m"]-0.07)<1e-12 and abs(b["perpendicular_offset_m"]-0.07)<1e-12,
                "paired_offset", "Offset amplitudes differ")
        require(a["z_interpolation_error_m"]<1e-12 and b["z_interpolation_error_m"]>0.01,
                "shortcut_diagnostic", "Diagnostic did not break z interpolation")
    report = {"passed":True,"cases":len(records),"diagnostic_pairs":3,
              "answer_checks":sum(len(r["checks"]) for r in records),"model_evaluated":False,"physics_executed":False,
              "results":[{k:r[k] for k in ("episode_id","z_interpolation_error_m","contract_hash")} for r in records]}
    write(output / "validation.json", report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20261107)
    args=parser.parse_args()
    run=args.output.resolve()
    require(not run.exists(),"output_exists","Choose a new output directory")
    require(not run.is_relative_to(REPO/"geomind/configs"),"protected_output","Cannot write formal configs")
    run.mkdir(parents=True)
    try:
        manifest=generate(run/"configs",args.seed)
        provenance=snapshot(run,["python","-m","geomind.tools.research_projection",*sys.argv[1:]],
                            index_filename="case_index.json",extra_paths=["tests/test_research_projection.py"])
        report=verify(run/"configs",manifest,run/"reports")
        report.update({k:provenance[k] for k in ("git_commit","source_snapshot_id","configuration_id")})
        write(run/"reports/validation.json",report)
        print(json.dumps(report,indent=2))
    except Exception as exc:
        write(run/"failure.json",{"passed":False,"exception":type(exc).__name__,"reason":str(exc)})
        raise


if __name__ == "__main__":
    main()
