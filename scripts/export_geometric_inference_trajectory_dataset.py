#!/usr/bin/env python3
"""Export trajectories for the five canonical geometric-inference tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_RUNNER = REPO_ROOT / "scripts" / "collect_anchor_reference_trajectory_dataset.py"
LIGHTING_POOL_SIZE = 15
LIGHTING_RECENT_WINDOW = 5


@dataclass(frozen=True)
class TaskSpec:
    name: str
    layout_key: str
    scene_id: int
    agent: str


GEOMETRIC_TASKS = (
    TaskSpec("perfect_midpoint", "geometric_inference_perfect_midpoint_line_2", 65, "explicit_midpoint_oracle"),
    TaskSpec("point_symmetry", "geometric_inference_point_symmetry_line_2", 65, "oracle"),
    TaskSpec("isometric_extension", "geometric_inference_isometric_extension_line_2", 65, "explicit_extension_oracle"),
    TaskSpec("square_completion", "geometric_inference_square_completion_corner_3", 65, "oracle"),
    TaskSpec("trend_continuation", "geometric_inference_implicit_trend_line_3", 65, "implicit_trend_oracle"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--layouts", type=int, default=10)
    parser.add_argument("--rollouts-per-layout", type=int, default=10)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--meta-seed", type=int, default=20260831)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--record-cameras", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _derived_seed(namespace: str, base_seed: int, task: TaskSpec) -> int:
    source = f"{namespace}|{base_seed}|{task.name}|{task.layout_key}|{task.scene_id}"
    digest = hashlib.blake2b(source.encode("utf-8"), digest_size=8).digest()
    return 1 + int.from_bytes(digest, "big") % (2**31 - 2)


def _lighting_pool_indices(base_seed: int, task: TaskSpec, count: int) -> list[int]:
    rng = random.Random(_derived_seed("lighting", base_seed, task))
    indices: list[int] = []
    for _ in range(int(count)):
        recent = set(indices[-LIGHTING_RECENT_WINDOW:])
        indices.append(rng.choice([index for index in range(LIGHTING_POOL_SIZE) if index not in recent]))
    return indices


def build_command(task: TaskSpec, args: argparse.Namespace, output_dir: Path) -> list[str]:
    lighting = _lighting_pool_indices(args.meta_seed, task, args.layouts)
    command = [
        sys.executable,
        str(DATASET_RUNNER),
        "--protocol", "geometric_inference",
        "--task-id", task.name,
        "--layout-key", task.layout_key,
        "--scene-id", str(task.scene_id),
        "--task-index", "0",
        "--object-mode", "geometric",
        "--layouts", str(args.layouts),
        "--rollouts-per-layout", str(args.rollouts_per_layout),
        "--sample-hz", str(args.sample_hz),
        "--meta-seed", str(_derived_seed("layout", args.meta_seed, task)),
        "--lighting-pool-indices", ",".join(str(index) for index in lighting),
        "--out-dir", str(output_dir),
        "--vary-trajectory-seed",
        "--trajectory-augmentation-enable",
        "--verify-phase2",
        "--verify-phase3",
        "--verify-phase6",
    ]
    if args.headless:
        command.append("--headless")
    if args.record_cameras:
        command.append("--record-cameras")
    if args.keep_going:
        command.append("--keep-going")
    return command


def main() -> int:
    args = parse_args()
    if args.layouts < 1 or args.rollouts_per_layout < 1:
        raise SystemExit("--layouts and --rollouts-per-layout must be positive.")
    if args.sample_hz <= 0.0:
        raise SystemExit("--sample-hz must be positive.")

    root = args.out_dir or (
        REPO_ROOT / "outputs" / f"geometric_inference_trajectory_dataset_{datetime.now():%Y%m%d_%H%M%S}"
    )
    root = root.expanduser().resolve()
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    print(f"dataset_root={root}")
    print(
        f"task_count={len(GEOMETRIC_TASKS)} layouts_per_task={args.layouts} "
        f"rollouts_per_layout={args.rollouts_per_layout} expected_episodes="
        f"{len(GEOMETRIC_TASKS) * args.layouts * args.rollouts_per_layout}"
    )
    for index, task in enumerate(GEOMETRIC_TASKS, start=1):
        command = build_command(task, args, root / task.name)
        print(f"[{index}/{len(GEOMETRIC_TASKS)}] {task.name} agent={task.agent}")
        print("  " + " ".join(command))
        if args.dry_run:
            manifest.append({"task": task.name, "status": "dry_run", "command": command})
            continue
        env = dict(__import__("os").environ)
        env["AGENT"] = task.agent
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        status = "success" if completed.returncode == 0 else "failed"
        manifest.append({"task": task.name, "status": status, "returncode": completed.returncode, "command": command})
        if completed.returncode != 0 and not args.keep_going:
            break

    if not args.dry_run:
        manifest_path = root / "collection_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"collection_manifest={manifest_path}")
    return 1 if any(item.get("status") == "failed" for item in manifest) else 0


if __name__ == "__main__":
    raise SystemExit(main())