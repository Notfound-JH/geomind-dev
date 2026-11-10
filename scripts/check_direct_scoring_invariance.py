#!/usr/bin/env python3
"""Verification #5: scoring is blind to everything the direct ablation changes.

Takes real base traces from an existing eval run, re-scores them, then re-scores a mutated
copy in which the instruction text, the recorded colours and an extra ``direct_marker``
object have been injected exactly the way the direct variant would. The evaluator verdict
and the run_vla_eval level classification must be bit-identical, which is what licenses
comparing base and direct success rates.

    python scripts/check_direct_scoring_invariance.py --run-root run_eval_openpi_05 \
        --json docs/direct_ablation/scoring_invariance.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DIRECT_INSTRUCTION = "Move the end effector near the green cube."
FAKE_MARKER = {"name": "direct_marker", "position": [9.99, 9.99, 9.99]}


def mutate_like_direct(trace: list[dict], result: dict) -> tuple[list[dict], dict]:
    """Apply every change the direct variant makes that could conceivably leak into scoring."""
    mutated = copy.deepcopy(trace)
    for step in mutated:
        agent_obs = step.get("agent_obs")
        if isinstance(agent_obs, dict):
            agent_obs["instruction"] = DIRECT_INSTRUCTION
        meta = step.get("meta")
        if isinstance(meta, dict):
            meta["instruction"] = DIRECT_INSTRUCTION
            meta["direct_ablation"] = {"ablation_id": "direct", "mechanism": "recolor"}
            objects = meta.get("object_positions")
            if isinstance(objects, list):
                objects.append(FAKE_MARKER)
            elif isinstance(objects, dict):
                objects["direct_marker"] = FAKE_MARKER["position"]
            meta["runtime_object_count"] = int(meta.get("runtime_object_count", 0) or 0) + 1
    mutated_result = copy.deepcopy(result)
    mutated_result["ablation_id"] = "direct"
    mutated_result["effective_instruction"] = DIRECT_INSTRUCTION
    return mutated, mutated_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="run_eval_openpi_05")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    from geomind.core.resolver import resolve_task_request
    from geomind.tasks.evaluators import resolve_evaluator

    import run_vla_eval  # noqa: PLC0415 - reuse the exact production classifier

    tasks_by_id = {str(task.get("task_id", "")): task for task in run_vla_eval.load_vla_tasks(REPO_ROOT)}

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    trace_paths = sorted(run_root.glob("runs/*/*/seed_*/episode_0000/trace.json"))
    rng = random.Random(int(args.seed))
    if args.samples > 0 and len(trace_paths) > args.samples:
        trace_paths = sorted(rng.sample(trace_paths, args.samples))

    rows: list[dict] = []
    failures: list[str] = []
    evaluators: dict[str, object] = {}

    for trace_path in trace_paths:
        result_path = trace_path.parent / "result.json"
        if not result_path.is_file():
            continue
        task_id = trace_path.parts[-5]
        level = trace_path.parts[-4]
        task = tasks_by_id.get(task_id)
        if task is None:
            failures.append(f"{task_id}: not in the VLA task list")
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        evaluator_key = str(result.get("evaluator_key", "")) or str(resolve_task_request(task_id).spec.evaluator_key)
        if evaluator_key not in evaluators:
            evaluators[evaluator_key] = resolve_evaluator(evaluator_key)
        evaluator = evaluators[evaluator_key]

        base_eval = evaluator.evaluate(trace)
        mutated_trace, mutated_result = mutate_like_direct(trace, result)
        direct_eval = evaluator.evaluate(mutated_trace)

        base_class = run_vla_eval.classify_success(task=task, level=level, result=result)
        direct_class = run_vla_eval.classify_success(task=task, level=level, result=mutated_result)

        problems = []
        if json.dumps(base_eval, sort_keys=True, default=str) != json.dumps(direct_eval, sort_keys=True, default=str):
            problems.append("evaluator output differs")
        if json.dumps(base_class, sort_keys=True, default=str) != json.dumps(direct_class, sort_keys=True, default=str):
            problems.append("level classification differs")
        if problems:
            failures.append(f"{task_id}|{level}|{trace_path.parts[-3]}: {'; '.join(problems)}")
        rows.append(
            {
                "task_id": task_id,
                "level": level,
                "seed_dir": trace_path.parts[-3],
                "evaluator_key": evaluator_key,
                "base_success": bool(base_eval.get("success", False)),
                "direct_success": bool(direct_eval.get("success", False)),
                "base_level_success": bool(base_class.get("success", False)),
                "direct_level_success": bool(direct_class.get("success", False)),
                "problems": problems,
            }
        )

    by_evaluator: dict[str, int] = {}
    for row in rows:
        by_evaluator[row["evaluator_key"]] = by_evaluator.get(row["evaluator_key"], 0) + 1
    print(f"sampled_traces={len(rows)} failures={len(failures)}")
    print("evaluator coverage:", json.dumps(by_evaluator, ensure_ascii=False))
    for failure in failures:
        print(f"  FAIL {failure}")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "coverage": {
                        "run_root": str(run_root),
                        "sampled_traces": len(rows),
                        "evaluator_keys": by_evaluator,
                        "checked": [
                            "evaluator verdict is unchanged when instruction/colour/marker fields are injected",
                            "run_vla_eval.classify_success verdict is unchanged under the same injection",
                        ],
                        "not_checked": [
                            "evaluator keys that do not appear in the sampled run",
                            "whether the direct rollouts themselves succeed (that is the experiment, not the check)",
                        ],
                    },
                    "samples": rows,
                    "failures": failures,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"json={out_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
